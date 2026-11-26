"""Post-fit oracle diagnostics for raster and high/low FP32 synthetic archives.

This module must never feed ground truth to the fitting initializer, solver,
acceptance rule or stopping criteria. All diagnostics run on the requested
model device; old projection is measured, never used as a solver fallback.
"""

from dataclasses import replace

import torch

from ..fitting.batched_jacobian import batch_jacobian, fp32_geometry_batch, residual_fp32_batch
from ..fitting.levenberg_marquardt import StageCoordinates, bound_mask, square_root_blocks
from ..fitting.observations import prepare_observations
from ..fitting.parameters import FittingParameters, TensorState
from ..fitting.square_root import SquareRootBlocks, square_root_step
from ..fitting.losses import evaluate_objective
from .dataset import camera_from_record
from .model import project_dfl


def _stats(error, mask):
    coordinates = error[mask]
    if not coordinates.numel() or not torch.isfinite(coordinates).all():
        raise ValueError("Invalid oracle residual coordinates")
    distances = coordinates.square().sum(dim=-1).sqrt()
    return {"coordinate_rms_px": float(coordinates.square().mean().sqrt()),
            "landmark_rms_px": float(distances.square().mean().sqrt()),
            "maximum_absolute_coordinate_px": float(coordinates.abs().max()),
            "eligible_landmarks": int(mask.sum())}


def _probe(data, parameters, layout, point, *, absolute=False):
    config = parameters.config
    if absolute:
        residuals, jacobians = [], []
        for start in range(0, len(data), config.batch_size):
            stop = min(len(data), start + config.batch_size)
            r, j = batch_jacobian(data, parameters, layout, point, start, stop, legacy_absolute=True)
            residuals.append(r); jacobians.append(j)
        system = SquareRootBlocks.create(torch.cat(jacobians), torch.cat(residuals), point.identity.numel(), config.batch_size)
    else:
        system = square_root_blocks(data, parameters, layout, point)
    scales = system.scales(config.lm.diagonal_floor)
    fixed = bound_mask(point, system.gradient, layout.expression_slice(), config.expression_limit)
    step, _ = square_root_step(system, scales, config.lm.minimum_damping, fixed)
    gradient = torch.cat((2 * scales.identity * system.gradient.identity,
        (2 * scales.local * system.gradient.local).masked_fill(fixed, 0).flatten()))
    return {
        "scaled_projected_gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "weighted_objective": float(system.residual.square().sum()),
        "minimum_damping": config.lm.minimum_damping,
        "predicted_reduction": system.predicted_reduction(step),
        "identity_step_rmse": float(step.identity.square().mean().sqrt()),
        "identity_step_max_abs": float(step.identity.abs().max()),
        "identity_step": step.identity.tolist(),
        "local_step_rmse": float(step.local.square().mean().sqrt()),
        "local_coordinate_names": list(layout.names),
        "interpretation": "One linearized correction at the reference state, not a bound or recovered fit",
    }


def _camera_roundtrip(model, initial, cameras):
    def stack(items, name):
        return torch.stack([getattr(c, name).to(device=model.reference.device, dtype=torch.float32) for c in items])
    t0, t = stack(initial, "translation"), stack(cameras, "translation")
    f0, f = stack(initial, "focal_length"), stack(cameras, "focal_length")
    z0 = t0[:, 2:]
    xy = t0[:, :2] + ((t[:, :2] - t0[:, :2]) / z0) * z0
    z = z0 * (t[:, 2:] / z0).log().exp()
    restored = f0 * (f / f0).log().exp()
    error = torch.cat((xy, z), -1) - t
    return {"translation_rms_m": float(error.square().mean().sqrt()),
            "translation_max_abs_m": float(error.abs().max()),
            "focal_max_abs_px": float((restored - f).abs().max()),
            "focal_max_relative": float(((restored - f) / f).abs().max()),
            "coordinate_origin": "original cold-start camera; FP32 log/exp round trip"}


@torch.no_grad()
def reference_precision(model, dataset, subject, observations, fit):
    """Replay reference parameters and estimate the local least-squares bias."""
    truth = dataset.ground_truth(subject)
    config = fit.config
    data = prepare_observations(observations, model.reference, config)
    arrays = truth.arrays
    clean = model.tensor(arrays["clean"])
    if tuple(item["id"] for item in truth.observations) != data.ids:
        raise ValueError("Oracle observation order differs from the fitted order")
    if not torch.equal(clean[data.eligible], data.pixels[data.eligible]):
        raise ValueError("Residual precision oracle requires clean eligible observations")
    cameras = tuple(camera_from_record(item["camera"], dtype=model.reference.dtype,
                                       device=model.reference.device) for item in truth.observations)
    state = TensorState(*(model.tensor(arrays[name]) for name in ("identity", "expression", "rotations", "translation")),
                        cameras)
    parameters = FittingParameters(model, data, state, fit.selection, config)
    layout = StageCoordinates.from_parameters(parameters, config.stages[-1].groups)
    point = layout.pack(parameters.values(), model.reference, len(data))
    # Single-view native model replays the generation path. Compare with
    # single-view sparse, then batched sparse to isolate execution order.
    native, sparse, batched, precise = [], [], [], []
    native_xyz, sparse_xyz, batch_xyz = [], [], []
    for index, camera in enumerate(cameras):
        args = (arrays["identity"], arrays["expression"][index], arrays["rotations"][index], arrays["translation"][index])
        points = model.landmarks(*args)
        native_xyz.append(points)
        native.append(project_dfl(points, camera, dataset.correspondences))
        points = parameters.landmark_geometry(state.identity, state.expression[index:index+1],
                                              state.rotations[index:index+1], state.translation[index:index+1])[0]
        sparse_xyz.append(points)
        sparse.append(project_dfl(points, camera, dataset.correspondences))
    for start in range(0, len(data), config.batch_size):
        stop = min(start + config.batch_size, len(data))
        values = parameters.values().batch(start, stop)
        rotated, camera, xyz = fp32_geometry_batch(parameters, start, stop, values)
        batch_xyz.append(xyz)
        camera_points = rotated + camera.translation[:, None, :]
        pixels = camera_points[..., :2] / camera_points[..., 2:] * camera.focal_length[:, None, :] + camera.principal_point[:, None, :]
        indices = data.gnm_indices[start:stop, :, None].expand(-1, -1, 2)
        batched.append(pixels.gather(1, indices))
        _, residual = residual_fp32_batch(data, parameters, start, stop, values)
        precise.append(residual)
    native, sparse = torch.stack(native), torch.stack(sparse)
    batched, precise = torch.cat(batched), torch.cat(precise)
    mask = data.eligible
    native_xyz, sparse_xyz, batch_xyz = torch.stack(native_xyz), torch.stack(sparse_xyz), torch.cat(batch_xyz)
    def xyz_error(a, b):
        difference = a - b
        if not torch.isfinite(difference).all():
            raise ValueError("Nonfinite oracle geometry error")
        return {"landmark_rms_m": float(difference.square().sum(-1).mean().sqrt()),
                "max_absolute_coordinate_m": float(difference.abs().max())}
    # ULP envelope describes final raster rounding only, not all upstream
    # arithmetic or a statistical lower bound on recoverable coefficients.
    up = torch.nextafter(clean, torch.full_like(clean, float("inf"))) - clean
    down = clean - torch.nextafter(clean, torch.full_like(clean, -float("inf")))
    half_ulp = torch.maximum(up, down) * 0.5
    active_probe = _probe(data, parameters, layout, point)
    legacy_config = replace(config, lm=replace(config.lm, fp32_forward="projection"))
    legacy_data = prepare_observations(tuple(replace(obs, pixel_low=None) for obs in observations),
                                      model.reference, legacy_config)
    legacy = FittingParameters(model, legacy_data, state, fit.selection, legacy_config)
    legacy_layout = StageCoordinates.from_parameters(legacy, legacy_config.stages[-1].groups)
    legacy_point = legacy_layout.pack(legacy.values(), model.reference, len(data))
    legacy_absolute = _probe(legacy_data, legacy, legacy_layout, legacy_point, absolute=True)
    legacy_compensated = (_probe(legacy_data, legacy, legacy_layout, legacy_point)
                          if config.lm.fp32_forward != "projection" else active_probe)
    final_objective = None
    if fit.final is not None:
        from ..fitting.observability import _state
        final_state = _state(model, data, config, fit.selection, None, fit, "final")
        final_parameters = FittingParameters(model, data, final_state, fit.selection, config)
        final_objective = evaluate_objective(model, data, final_parameters, least_squares=True)[0].total
    target_low = torch.zeros_like(clean) if data.pixel_low is None else data.pixel_low
    if "pixel_low" in arrays and not torch.equal(model.tensor(arrays["pixel_low"]), target_low):
        raise ValueError("Oracle low pixels differ from the fitted observations")
    return {
        "status": "evaluated", "used_for_fitting": False,
        "device": str(model.reference.device), "dtype": str(model.reference.dtype),
        "sparse_single_view_geometry_minus_native": xyz_error(sparse_xyz, native_xyz),
        "sparse_batched_geometry_minus_sparse_single": xyz_error(batch_xyz, sparse_xyz),
        "native_single_view_minus_archive": _stats(native - clean, mask),
        "sparse_single_view_minus_native": _stats(sparse - native, mask),
        "sparse_batched_minus_sparse_single": _stats(batched - sparse, mask),
        "sparse_batched_absolute_minus_archive": _stats(batched - clean, mask),
        "compensated_minus_archive": _stats(precise, mask),
        "compensated_minus_absolute_residual": _stats(precise - (batched - clean), mask),
        "stored_pixel_half_ulp_envelope": _stats(half_ulp, mask),
        "truth_linearization": active_probe,
        "forward_mode": config.lm.fp32_forward,
        "native_and_sparse_projection_comparisons_target": "high raster component only",
        "compensated_residual_target": "high plus low components" if data.pixel_low is not None else "high raster component only",
        "target_representation": "raster-fp32" if data.pixel_low is None else "compensated-fp32-high-low-v1",
        "legacy_absolute_truth_linearization": legacy_absolute,
        "legacy_projection_compensated_truth_linearization": legacy_compensated,
        "legacy_probe_target": "high raster component only; original logarithmic camera coordinates",
        "legacy_camera_roundtrip": _camera_roundtrip(model, fit.initial.cameras, cameras),
        "archived_low_component": _stats(target_low, mask),
        "final_objective_recomputed_from_physical_snapshot": final_objective,
    }
