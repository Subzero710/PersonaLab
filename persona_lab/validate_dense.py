"""Numerically validate native surface observations and restricted FP64 fitting."""

import argparse
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from .benchmark.model import load_benchmark_model
from .benchmark.storage import publish_directory, write_json
from .dense_dataset import ExtractionAttempt, load_dense_dataset, write_dense_dataset
from .dense_observability import ProjectionProtocol, projection_jacobian
from .dense_reference import ASSET, PROVIDER, PROVIDER_VERSION, oracle_definition, oracle_observations, oracle_state
from .fitting import FitConfig, LevenbergMarquardtConfig, Stage, fit_identity
from .fitting.configuration import DEFAULT_EXPRESSION
from .fitting.monitoring import peak_rss_bytes


def _gradient(function, values, probe):
    with torch.enable_grad():
        values = values.detach().requires_grad_()
        output = function(values)
        derivative, = torch.autograd.grad(output, values, grad_outputs=probe)
    return derivative.detach()


def validate_geometry(model, definition, seed, views):
    """Compare native operations to full GNM and independent NumPy references."""
    state = oracle_state(model, seed, views)
    observations = oracle_observations(model, definition, seed, state)
    bound = definition.bind(model)
    geometry = bound.sparse_geometry(model)
    head = model.head
    head_joint = tuple(head.joint_names).index("head")
    max_vertex_error, max_pixel_error, max_gradient_error, max_fd_error = 0., 0., 0., 0.
    expected = []
    jacobian_error = None
    for view, camera in enumerate(state.cameras):
        protocol = ProjectionProtocol(state, view, head_joint)
        values = protocol.parameters()

        def full(identity, expression, rotations, translation):
            vertices = head(identity=identity, expression=expression, rotations=rotations,
                            translation=translation)
            return bound.evaluate(vertices)

        sparse = protocol.project(geometry, values)
        vertices = head(identity=state.identity, expression=state.expressions[view],
                        rotations=state.rotations[view], translation=state.translations[view])
        native = np.einsum("lvd,lv->ld", vertices.detach().cpu().numpy()[bound.indices], bound.weights)
        points = geometry(state.identity, state.expressions[view], state.rotations[view],
                          state.translations[view])
        max_vertex_error = max(max_vertex_error, float(np.max(np.abs(points.detach().cpu().numpy() - native))))
        # The synthetic camera rotations are known zero. Project independently in
        # float64 after the sign change between GNM and OpenCV coordinates.
        xyz = native * (1., -1., -1.) + camera.translation.cpu().numpy()
        predicted = xyz[:, :2] / xyz[:, 2:] * camera.focal_length.cpu().numpy() + camera.principal_point.cpu().numpy()
        max_pixel_error = max(max_pixel_error, float(np.max(np.abs(sparse.detach().cpu().numpy() - predicted))))
        expected.append(sparse.detach())
        if view == 0:
            _, sparse_jacobian = projection_jacobian(protocol, geometry, 16)
            _, full_jacobian = projection_jacobian(protocol, full, 16)
            jacobian_error = float((sparse_jacobian - full_jacobian).abs().max())

        probe = sparse.new_tensor(np.random.default_rng(seed + view).normal(size=sparse.shape))
        sparse_gradient = _gradient(lambda x: protocol.project(geometry, x), values, probe)
        reference_gradient = _gradient(lambda x: protocol.project(full, x), values, probe)
        max_gradient_error = max(max_gradient_error, float((sparse_gradient - reference_gradient).abs().max()))
        indices = (0, 170, 173, head.identity_dim, head.identity_dim + 350,
                   head.identity_dim + 382, values.numel() - 1)
        for index in indices:
            offset = torch.zeros_like(values)
            step = 1e-5
            offset[index] = step
            positive = protocol.project(full, values + offset)
            negative = protocol.project(full, values - offset)
            derivative = ((positive - negative) * probe).sum() / (2 * step)
            error = float((derivative - sparse_gradient[index]).abs())
            max_fd_error = max(max_fd_error, error)
    per_view = torch.stack(expected)
    full_targets = torch.stack([obs.landmarks for obs in observations])
    torch.testing.assert_close(per_view, full_targets, atol=2e-10, rtol=0)
    batch = geometry(state.identity, state.expressions, state.rotations, state.translations)
    sequential = torch.stack([geometry(state.identity, expr, rot, trans)
        for expr, rot, trans in zip(state.expressions, state.rotations, state.translations, strict=True)])
    batch_error = float((batch - sequential).abs().max())
    return observations, {
        "seed": seed, "views": views, "point_count": len(definition.entries),
        "valid_by_view": [int(obs.validity.sum()) for obs in observations],
        "max_sparse_vs_numpy_mesh_m": max_vertex_error, "max_sparse_vs_numpy_camera_px": max_pixel_error,
        "max_sparse_vs_full_elementwise_jacobian_first_view_px": jacobian_error,
        "max_sparse_vs_full_gradient_in_probe_px": max_gradient_error,
        "max_central_difference_gradient_in_probe_px": max_fd_error,
        "max_batch_vs_single_m": batch_error, "gradient_columns": head.identity_dim + head.expression_dim + 7,
        "tolerances": {"mesh_m": 2e-12, "projection_px": 2e-10, "shared_gradient_px": 2e-9,
                       "finite_difference_px": 1e-5, "batch_m": 2e-12},
    }


def restricted_fit(model, definition, *, iterations, batch_size, seed):
    """Verify native QR consumes dense rows in a clean 170/24 representable test."""
    state = oracle_state(model, seed, 3)
    identity = state.identity.clone()
    identity[170:] = 0
    expression = torch.zeros_like(state.expressions)
    head = model.head
    selected = [list(head.expression_names).index(name) for name in DEFAULT_EXPRESSION]
    expression[:, selected] = state.expressions[:, selected]
    # Pose and identity initialization are blind; the synthetic camera is calibrated.
    state = replace(state, identity=identity, expressions=expression)
    observations = oracle_observations(model, definition, seed, state)
    config = FitConfig(
        batch_size=batch_size,
        lm=LevenbergMarquardtConfig(fp64_square_root=True, jacobian_rows=16),
        stages=(Stage("joint", iterations, ("identity", "expression", "pose"), 0.02),),
    )
    result = fit_identity(model, observations, config, correspondences=definition)
    record = {
        "status": result.status, "reason": result.reason, "views": len(observations),
        "point_count_per_view": len(definition.entries), "batch_size": batch_size,
        "trainable_identity": len(result.selection.identity), "trainable_expression": len(result.selection.expression),
        "initial_rms_px": None if result.initial_reprojection is None else result.initial_reprojection.overall.rms_px,
        "final_rms_px": None if result.final_reprojection is None else result.final_reprojection.overall.rms_px,
        "runtime": result.runtime,
    }
    return record


def run_validation(model, output, *, seeds, views, points, fit_iterations, fit_batch_size):
    if model.reference.dtype != torch.float64:
        raise ValueError("Synthetic validation requires the pinned FP64 GNM Head")
    if not seeds or len(set(seeds)) != len(seeds) or fit_iterations <= 0 or fit_batch_size <= 0:
        raise ValueError("Use unique seeds and positive fit iterations and batch size")
    started = perf_counter()
    definition = oracle_definition(model, points)
    checks = []
    with publish_directory(output, completion="validation.json") as stage:
        for seed in seeds:
            observations, check = validate_geometry(model, definition, seed, views)
            with TemporaryDirectory(dir=stage) as work:
                packed = Path(work) / "dataset"
                write_dense_dataset(packed, definition.schema, PROVIDER, PROVIDER_VERSION, ASSET,
                                    observations[0].extraction, (
                    ExtractionAttempt(obs.observation_id, obs.source.reference, obs) for obs in observations
                ))
                loaded = load_dense_dataset(
                    packed, schema=definition.schema, provider=PROVIDER, provider_version=PROVIDER_VERSION,
                    asset=ASSET, device=model.reference.device, dtype=torch.float64,
                )
                check["serialization_exact"] = all(
                    torch.equal(actual.landmarks, expected.landmarks)
                    and actual.observation_id == expected.observation_id
                    and torch.equal(actual.validity, expected.validity)
                    and torch.equal(actual.known_camera.translation, expected.known_camera.translation)
                    and actual.visibility is None and actual.confidence is None
                    for actual, expected in zip(loaded.select(), observations, strict=True)
                )
            limits = check["tolerances"]
            check["passed"] = check["serialization_exact"] and all(
                check[key] <= limits[limit] for key, limit in (
                    ("max_sparse_vs_numpy_mesh_m", "mesh_m"), ("max_sparse_vs_numpy_camera_px", "projection_px"),
                    ("max_sparse_vs_full_gradient_in_probe_px", "shared_gradient_px"),
                    ("max_sparse_vs_full_elementwise_jacobian_first_view_px", "shared_gradient_px"),
                    ("max_central_difference_gradient_in_probe_px", "finite_difference_px"),
                    ("max_batch_vs_single_m", "batch_m"),
                )
            )
            checks.append(check)
        fit = restricted_fit(model, definition, iterations=fit_iterations, batch_size=fit_batch_size, seed=seeds[0])
        fit["passed"] = fit["final_rms_px"] is not None and fit["final_rms_px"] < fit["initial_rms_px"]
        write_json(stage / "oracle_correspondences.json", definition.to_record())
        report = {
            "model": model.metadata, "device": str(model.reference.device), "dtype": "float64",
            "full_native_geometry_checks": checks, "restricted_fp64_qr": fit,
            "elapsed_seconds": perf_counter() - started, "peak_process_rss_bytes": peak_rss_bytes(),
        }
        write_json(stage / "validation.json", report)
    return all(check["passed"] for check in checks) and fit["passed"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--fit-iterations", type=int, default=8)
    parser.add_argument("--fit-batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        model = load_benchmark_model(args.model_file, dtype="float64", device=args.device)
        passed = run_validation(
            model, args.output, seeds=tuple(args.seed or (42, 43, 44)), views=args.views, points=args.points,
            fit_iterations=args.fit_iterations, fit_batch_size=args.fit_batch_size,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    if not passed:
        parser.exit(1, "Numerical validation failed; see validation.json for measurements.\n")


if __name__ == "__main__":
    main()
