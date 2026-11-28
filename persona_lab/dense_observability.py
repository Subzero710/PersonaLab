"""Full-native observation Jacobians and rank diagnostics without optimization."""

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path
from time import perf_counter

from gnm.shape import gnm_utils
import numpy as np
import torch

from .benchmark.model import load_benchmark_model
from .benchmark.storage import digest_arrays, digest_json, publish_directory, write_json
from .camera import project
from .coordinates import gnm_to_frontal_axes
from .correspondences import DFL_TO_GNM
from .dense_reference import OracleState, oracle_definition, oracle_state
from .fitting.monitoring import peak_rss_bytes
from .fitting.observability import _compress, _remove_nuisance, _spectrum
from .fitting.parameters import FittingParameters
from .gnm_geometry import SparseHeadLandmarks
from .mediapipe_face import SCHEMAS
from .surface_correspondences import load_correspondences


@dataclass(frozen=True, slots=True)
class ProjectionProtocol:
    state: OracleState
    view: int
    head_index: int

    def parameters(self):
        state, view = self.state, self.view
        return torch.cat((state.identity, state.expressions[view], state.rotations[view, self.head_index],
                          state.identity.new_zeros(4)))

    def project(self, geometry, values):
        state, view = self.state, self.view
        identity, expression, pose, camera = values.split((state.identity.numel(),
                                                            state.expressions.shape[1], 3, 4))
        rotations = torch.cat((state.rotations[view, :self.head_index], pose[None],
                               state.rotations[view, self.head_index + 1:]))
        known = state.cameras[view]
        perspective = FittingParameters._decode_camera(
            known, camera[:3], camera[3:], values.new_tensor(known.image_size.height), str(view),
        )
        points = geometry(identity, expression, rotations, state.translations[view])
        return project(gnm_to_frontal_axes(points), perspective)


@torch.enable_grad()
def projection_jacobian(protocol, geometry, column_batch):
    """Compute exact AD columns in bounded blocks through native GNM projection."""
    if type(column_batch) is not int or column_batch < 1:
        raise ValueError("Jacobian column batch size must be positive")
    values = protocol.parameters().requires_grad_()
    pixels = protocol.project(geometry, values)
    # J^T probe is linear in the observation-space probe. Its derivative
    # with respect to that probe yields Jacobian columns without differencing.
    probe = torch.zeros_like(pixels, requires_grad=True)
    pulled_back, = torch.autograd.grad(pixels, values, grad_outputs=probe, create_graph=True)
    jacobian = pixels.new_empty((*pixels.shape, values.numel()))
    for start in range(0, values.numel(), column_batch):
        end = min(start + column_batch, values.numel())
        directions = values.new_zeros((end - start, values.numel()))
        directions[:, start:end].diagonal().fill_(1)
        columns, = torch.autograd.grad(pulled_back, probe, grad_outputs=directions,
                                       is_grads_batched=True, retain_graph=end < values.numel())
        jacobian[..., start:end] = columns.movedim(0, -1)
    if not torch.isfinite(jacobian).all():
        raise ValueError("Native projection produced a nonfinite Jacobian")
    return pixels.detach(), jacobian


def _metric(matrix, rows, threshold, svd, groups=()):
    _, singular, right = svd(matrix, full_matrices=False)
    record = _spectrum(matrix, singular, threshold)
    record["parameter_count"] = record.pop("identity_parameter_count")
    record["matrix_dimensions"] = [rows, matrix.shape[1]]
    record["compressed_dimensions"] = list(matrix.shape)
    record["rank_threshold"] = threshold
    record["full_space_condition_number"] = record.pop("condition_number")
    record["full_space_condition_status"] = "finite" if record["nullity"] == 0 else "rank_deficient"
    record["group_sensitivity_l2"] = {
        name: float(torch.linalg.vector_norm(matrix[:, indices])) for name, indices in groups
    }
    return record, right


def _save_subspace(arrays, name, metric, right):
    rank, nullity = metric["rank"], metric["nullity"]
    if rank < nullity:
        arrays[f"{name}__observable_basis"] = right[:rank].cpu().numpy()
    else:
        nullspace = right[rank:]
        if right.shape[0] < right.shape[1]:
            nullspace = torch.linalg.qr(right[:rank].T, mode="complete").Q[:, rank:].T
        arrays[f"{name}__null_basis"] = nullspace.cpu().numpy()


def _compensation_probes(right, rank):
    row_space = right[:rank]
    projector = torch.eye(right.shape[1], device=right.device, dtype=right.dtype) - row_space.T @ row_space
    scores = projector.diagonal()
    slots = torch.topk(scores, min(3, right.shape[1] - rank)).indices
    probes = projector[:, slots]
    return slots, probes / torch.linalg.vector_norm(probes, dim=0)


def _stack_views(identity, local):
    """Keep global identity columns and separate nuisance columns for each view."""
    return torch.cat((torch.cat(identity), torch.block_diag(*local)), dim=1)


def _compensation(direction, identity, locals_, threshold, svd, expression_dim):
    result = []
    for block, local in zip(identity, locals_, strict=True):
        u, singular, vh = svd(local, full_matrices=False)
        keep = singular > threshold
        rhs = block @ direction
        delta = -(vh[keep].T @ ((u[:, keep].T @ rhs) / singular[keep, None]))
        result.append({
            "residual_l2": torch.linalg.vector_norm(rhs + local @ delta, dim=0).tolist(),
            "expression_l2": torch.linalg.vector_norm(delta[:expression_dim], dim=0).tolist(),
            "head_rotation_xyz": delta[expression_dim:expression_dim + 3].T.tolist(),
            "camera_offsets_dx_dy_log_depth_log_focal": delta[expression_dim + 3:].T.tolist(),
        })
    return result


def analyze_jacobians(model, jacobians, point_regions, camera_mode, threshold):
    """Report complete spectra and final null bases using a common absolute cutoff."""
    if camera_mode not in ("known", "estimated") or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Select a camera mode and a finite positive rank threshold")
    if not jacobians or model.reference.dtype != torch.float64:
        raise ValueError("Full-native analysis requires at least one view and an FP64 model")
    head = model.head
    identity_dim, expression_dim = head.identity_dim, head.expression_dim
    columns = identity_dim + expression_dim + (3 if camera_mode == "known" else 7)
    rows_per_view = [2 * len(jacobian) for jacobian in jacobians]
    rows = sum(rows_per_view)
    identity_groups = tuple(gnm_utils.identity_regions_indices(head).items())
    expression_groups = tuple(gnm_utils.expression_regions_indices(head).items())
    svd = partial(torch.linalg.svd, driver="gesvd" if model.reference.is_cuda else None)
    blocks, sensitivities = [], []
    for jacobian in jacobians:
        if jacobian.shape != (len(point_regions), 2, identity_dim + expression_dim + 7):
            raise ValueError("Every view must have the same declared native observation and Jacobian dimensions")
        matrix = jacobian[..., :columns].reshape(-1, columns)
        matrix = matrix / (720. * math.sqrt(rows / 2))
        blocks.append(_compress(matrix.new_empty((0, columns)), matrix))
        sensitivities.append({
            region: {
                name: float(torch.linalg.vector_norm(matrix.reshape(len(point_regions), 2, columns)[
                    [index for index, value in enumerate(point_regions) if value == region], :, indices
                ]))
                for name, indices in (
                    ("identity", slice(0, identity_dim)),
                    ("expression", slice(identity_dim, identity_dim + expression_dim)),
                    ("pose_camera", slice(identity_dim + expression_dim, columns)),
                )
            }
            for region in dict.fromkeys(point_regions)
        })
    identity, expression, pose_camera = map(list, zip(*(
        block.split((identity_dim, expression_dim, columns - identity_dim - expression_dim), dim=1)
        for block in blocks
    ), strict=True))
    local = [torch.cat((expr, camera), dim=1) for expr, camera in zip(expression, pose_camera, strict=True)]

    def remove(source, nuisance):
        return _remove_nuisance(source, nuisance, 0.0, svd, atol=threshold)

    before = torch.cat(identity)
    after_expression = torch.cat([remove(i, e) for i, e in zip(identity, expression, strict=True)])
    after_pose_camera = torch.cat([
        remove(i, n) for i, n in zip(identity, pose_camera, strict=True)
    ])
    after_all = torch.cat([remove(i, n) for i, n in zip(identity, local, strict=True)])
    records, nulls = {}, {}
    for name, matrix in (("before_nuisance", before), ("after_expression", after_expression),
                         ("after_pose_camera", after_pose_camera), ("after_all_nuisance", after_all)):
        metric, right = _metric(matrix, rows, threshold, svd, identity_groups)
        records[name] = metric | {"native_groups": {
            group: _metric(matrix[:, indices], rows, threshold, svd)[0] for group, indices in identity_groups
        }}
        if name == "after_all_nuisance":
            _save_subspace(nulls, "identity_after_all_nuisance", metric, right)
            slots, directions = _compensation_probes(right, metric["rank"])

    nulls["compensation_probes_in_identity_order"] = directions.T.cpu().numpy()
    compensation = {
        "projected_native_coordinate_indices": slots.tolist(),
        "by_view": _compensation(directions, identity, local, threshold, svd, expression_dim)
        if directions.numel() else [],
    }
    expression_records = []
    after_camera_blocks = []
    for index, (i, e, pc) in enumerate(zip(identity, expression, pose_camera, strict=True)):
        after_camera = remove(torch.cat((i, e), dim=1), pc)
        after_camera_blocks.append(after_camera.split((identity_dim, expression_dim), dim=1))
        fixed, _ = _metric(e, rows_per_view[index], threshold, svd, expression_groups)
        after_pc, basis = _metric(after_camera[:, identity_dim:], rows_per_view[index],
                                  threshold, svd, expression_groups)
        _save_subspace(nulls, f"expression_view_{index}_after_pose_camera", after_pc, basis)
        expression_records.append({"before_nuisance": fixed, "after_pose_camera": after_pc})

    joint = _stack_views(identity, expression)
    joint_after = _stack_views(*map(list, zip(*after_camera_blocks, strict=True)))
    joint_fixed, _ = _metric(joint, rows, threshold, svd)
    joint_eliminated, basis = _metric(joint_after, rows, threshold, svd)
    _save_subspace(nulls, "joint_identity_expression_after_pose_camera", joint_eliminated, basis)
    return {
        "identity": records, "expression_by_view": expression_records,
        "joint_identity_expression": {"before_nuisance": joint_fixed, "after_pose_camera": joint_eliminated},
        "identity_nullspace_compensation": compensation,
        "observation_region_sensitivity_l2_by_view": sensitivities,
    }, nulls


@torch.no_grad()
def run_study(model, output, *, seeds, views, reference_count, column_batch, threshold, mapped=None):
    """Evaluate paired ground-truth GNM states with an explicitly declared point budget."""
    if model.reference.dtype != torch.float64 or reference_count < len(DFL_TO_GNM.entries):
        raise ValueError("The FP64 reference requires at least the sparse landmark point budget")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Select unique deterministic seeds")
    start = perf_counter()
    definition = oracle_definition(model, reference_count)
    native_reference = definition.bind(model)
    sparse_count = len(DFL_TO_GNM.entries)
    representations = [
        ("dfl68", SparseHeadLandmarks(model.head), tuple(entry.region.value for entry in DFL_TO_GNM.entries)),
        ("gnm_oracle", native_reference.sparse_geometry(model),
         tuple(point.region for point in definition.schema.points)),
    ]
    blockers = {}
    if mapped is None:
        blockers["mediapipe"] = "No independently reviewed MediaPipe-to-GNM correspondence definition supplied"
    else:
        bound = mapped.bind(model)
        if not bound.slots or any(mapped.entries[index].minimum_visibility is not None for index in bound.slots):
            raise ValueError("Mapped synthetic diagnostics need native locations without unmeasured visibility restrictions")
        representations.append(("mediapipe_mapped_oracle", bound.sparse_geometry(model),
                                tuple(mapped.schema.points[index].region for index in bound.slots)))
    studies = []
    head_index = tuple(model.head.joint_names).index("head")
    with publish_directory(output, completion="report.json") as stage:
        for seed in seeds:
            state = oracle_state(model, seed, views)
            state_digest = digest_arrays({
                "identity": state.identity.cpu().numpy(), "expression": state.expressions.cpu().numpy(),
                "joint_rotations": state.rotations.cpu().numpy(), "translations": state.translations.cpu().numpy(),
                "camera_translation": torch.stack([camera.translation for camera in state.cameras]).cpu().numpy(),
                "camera_focal": torch.stack([camera.focal_length for camera in state.cameras]).cpu().numpy(),
            })
            for name, geometry, regions in representations:
                jacobians = []
                for view in range(views):
                    protocol = ProjectionProtocol(state, view, head_index)
                    pixels, jacobian = projection_jacobian(protocol, geometry, column_batch)
                    size = state.cameras[view].image_size
                    if not ((pixels >= 0) & (pixels < pixels.new_tensor((size.width, size.height)))).all():
                        raise ValueError(f"Oracle points left the raster in seed {seed}, {name}, view {view}")
                    jacobians.append(jacobian)
                budgets = [("matched", sparse_count)] if len(regions) == sparse_count else [
                    ("natural", len(regions))
                ]
                if len(regions) > sparse_count:
                    budgets.append(("matched", sparse_count))
                elif len(regions) < sparse_count:
                    blockers["mediapipe_matched_budget"] = "Fewer reviewed locations than the 68-point budget"
                for budget, count in budgets:
                    for camera in ("known", "estimated"):
                        report, arrays = analyze_jacobians(
                            model, [jac[:count] for jac in jacobians], regions[:count], camera, threshold,
                        )
                        archive = f"nullspace_{seed}_{name}_{budget}_{camera}.npz"
                        np.savez_compressed(stage / archive, **arrays)
                        studies.append({
                            "seed": seed, "state_sha256": state_digest, "representation": name,
                            "view_count": views, "points_per_view": count, "budget": budget,
                            "camera_mode": camera, "observation_regions": Counter(regions[:count]),
                            "nullspace_archive": archive, **report,
                        })
                        print(f"seed {seed} {name} {budget} {camera}: "
                              f"identity rank {report['identity']['after_all_nuisance']['rank']}/253", flush=True)
        write_json(stage / "oracle_correspondences.json", definition.to_record())
        mapped_record = None
        if mapped is not None:
            write_json(stage / "mapped_correspondences.json", mapped.to_record())
            mapped_record = {
                "schema": mapped.schema.identifier, "schema_version": mapped.schema.version,
                "sha256": digest_json(mapped.to_record()),
                "ordered_mapped_point_ids": [mapped.entries[index].point_id for index in bound.slots],
                "path": "mapped_correspondences.json",
            }
        write_json(stage / "report.json", {
            "format": "personalab-full-native-observation-study-v1",
            "model": model.metadata, "seeds": list(seeds), "views": views,
            "reference_points": reference_count, "matched_points": sparse_count,
            "torch": str(torch.__version__), "device": str(model.reference.device), "dtype": "float64",
            "jacobian_column_batch": column_batch, "rank_threshold": threshold,
            "normalization": "pixels divided by short edge and sqrt(total point rows per representation)",
            "known_camera_gauge": "fixed perspective intrinsics and extrinsics; local head rotation only",
            "estimated_camera_gauge": "same state; local head rotation, relative camera XY, log depth and log focal",
            "fixed_in_both": "principal point, camera rotation, GNM root translation and non-head joint rotations",
            "visibility": "oracle locations include hidden surfaces; visibility is unavailable",
            "nullspace_convention": (
                "Rows are native directions. __null_basis spans the nullspace. For __observable_basis Q, "
                "x - Q.T @ (Q @ x) projects onto the nullspace. Joint columns are identity followed by "
                "each local expression block in view order."
            ),
            "elapsed_seconds": perf_counter() - start, "peak_process_rss_bytes": peak_rss_bytes(),
            "mapped_correspondences": mapped_record,
            "blockers": blockers, "studies": studies,
        })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", action="append", type=int, help="repeat for each seed; default: 42, 43, 44")
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--reference-points", type=int, default=2048)
    parser.add_argument("--jacobian-columns", type=int, default=24)
    parser.add_argument("--rank-threshold", type=float, default=1e-8)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--mapped-schema", choices=SCHEMAS)
    args = parser.parse_args(argv)
    try:
        if (args.mapping is None) != (args.mapped_schema is None):
            raise ValueError("--mapping and --mapped-schema must be supplied together")
        model = load_benchmark_model(args.model_file, dtype="float64", device=args.device)
        mapping = None if args.mapping is None else load_correspondences(args.mapping, SCHEMAS[args.mapped_schema])
        run_study(model, args.output, seeds=(42, 43, 44) if args.seed is None else tuple(args.seed),
                  views=args.views, reference_count=args.reference_points, column_batch=args.jacobian_columns,
                  threshold=args.rank_threshold, mapped=mapping)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
