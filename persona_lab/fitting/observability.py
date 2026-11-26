"""Local Jacobian diagnostics for the active native parameter selection."""

import math
from functools import partial

import torch

from ..benchmark.configuration import finite_number
from .configuration import GROUPS
from .initialization import initialize
from .losses import lm_dtype
from .levenberg_marquardt import StageCoordinates, observation_jacobian
from .batched_jacobian import batch_jacobian
from .observations import prepare_observations
from .parameters import (
    PARAMETER_GROUPS, FitInitialization, FittingParameters, TensorState, copy_camera, resolve_selection, validate_model,
)
from .results import ARRAY_FIELDS, FitResult


def _remove_nuisance(identity, nuisance, rtol, svd=torch.linalg.svd):
    if not nuisance.shape[1]:
        return identity
    u, s, _ = svd(nuisance, full_matrices=False)
    basis = u[:, s > s[0] * rtol]
    return identity - basis @ (basis.mT @ identity)


def _spectrum(matrix, values, threshold):
    width = matrix.shape[1]
    rank = int((values > threshold).sum())
    effective = float(values[0] / values[rank - 1]) if rank else None
    if effective is not None and not math.isfinite(effective):
        effective = None
    return {
        "rank": rank,
        "identity_parameter_count": width,
        "nullity": width - rank,
        "condition_number": effective if rank == width else None,
        "effective_condition_number": effective,
        "implicit_zero_singular_values": width - values.numel(),
        "singular_values": values.tolist() + [0.0] * (width - values.numel()),
        "column_l2": torch.linalg.vector_norm(matrix, dim=0).tolist(),
    }


def _compress(previous, rows, qr=torch.linalg.qr):
    return qr(torch.cat((previous, rows), dim=0), mode="r").R


def _describe(matrix, rows, threshold, rtol, svdvals=torch.linalg.svdvals):
    values = svdvals(matrix)
    record = _spectrum(matrix, values, threshold)
    record["parameter_count"] = record.pop("identity_parameter_count")
    return record | {
        "matrix_dimensions": [rows, matrix.shape[1]], "compressed_dimensions": list(matrix.shape),
        "absolute_threshold": threshold, "nuisance_relative_threshold": rtol,
        "minimum_singular_value": record["singular_values"][-1],
        "maximum_singular_value": record["singular_values"][0],
    }


def _project_with_depth(columns, nuisance, depth_column, rtol, svd=torch.linalg.svd):
    u, singular, vh = svd(nuisance, full_matrices=False)
    keep = singular > singular[0] * rtol
    basis = u[:, keep]
    coefficients = basis.mT @ columns
    depth = -(vh[keep, depth_column] / singular[keep]) @ coefficients
    return columns - basis @ coefficients, depth


def _direction(identity, focal, depth, names, singular):
    count = min(5, len(names))
    indices = torch.topk(identity.abs(), count).indices.tolist()
    return {
        "singular_value": singular, "identity_l2": float(torch.linalg.vector_norm(identity)),
        "largest_identity_loadings": {names[index]: float(identity[index]) for index in indices},
        "log_focal_loadings": focal.tolist(), "compensating_log_depth_by_observation": depth.tolist(),
    }


def _focal_observability(data, parameters, layout, base, position, width, rtol, atol,
                         stage_groups=(), operations=None):
    jacobian_at = observation_jacobian
    qr, svd, svdvals, pinv = torch.linalg.qr, torch.linalg.svd, torch.linalg.svdvals, torch.linalg.pinv
    if operations is not None:
        jacobian_at = partial(operations.call, "diagnostic_jacobian", jacobian_at)
        qr = partial(operations.call, "qr", qr)
        svd = partial(operations.call, "svd", svd)
        svdvals = partial(operations.call, "svdvals", svdvals)
        pinv = partial(operations.call, "pinv", pinv)
    names = parameters.selection.identity
    empty = position.identity.new_empty((0, width), dtype=lm_dtype(parameters.config))
    identity_blocks = dict.fromkeys((
        "fixed_nuisance", "after_expression", "after_expression_and_pose", "after_all_nuisance",
    ), empty)
    joint = empty.new_empty((0, width + 1))
    expression = layout.expression_slice()
    locals_ = dict(layout.columns())
    pose = locals_.get("pose")
    depth_column = locals_["camera_native" if parameters.config.lm.fp32_forward == "compensated" else "camera_delta"].start + 2
    depth_rows, distinct_focals, raw_focal_norms = [], [], []
    active_blocks = []
    rows = 0
    def jacobians():
        if parameters.config.lm.arithmetic_dtype != "float32":
            for index in range(len(data)):
                yield jacobian_at(data, parameters, layout, base, position, index)[1]
            return
        batch = parameters.config.batch_size
        for start in range(0, len(data), batch):
            stop = min(len(data), start + batch)
            args = (data, parameters, layout, position, start, stop)
            _, jacobian = (batch_jacobian(*args) if operations is None else
                           operations.call("diagnostic_jacobian_batch", batch_jacobian, *args))
            for row, mask in zip(jacobian, data.eligible[start:stop], strict=True):
                yield row[mask.repeat_interleave(2)]

    for index, jacobian in enumerate(jacobians()):
        identity = jacobian[:, :width]
        rows += len(jacobian)
        if layout.shared_focal:
            focal, local = jacobian[:, width:width + 1], jacobian[:, width + 1:]
            other = local
        else:
            local = jacobian[:, width:]
            # Local focal is the final column, after camera translation.
            focal, other = local[:, -1:], local[:, :-1]
        if stage_groups:
            active = [local[:, columns] for name, columns in locals_.items()
                      if PARAMETER_GROUPS[name] in stage_groups]
            if layout.shared_focal and "camera" in stage_groups:
                active.insert(0, focal)
            active_blocks.append(qr(torch.cat(active, dim=1), mode="r").R)
        raw_focal_norms.append(float(torch.linalg.vector_norm(focal)))
        expression_columns = local[:, expression] if expression is not None else local[:, :0]
        pose_columns = local[:, pose] if pose is not None else local[:, :0]
        blocks = {
            "fixed_nuisance": identity,
            "after_expression": _remove_nuisance(identity, expression_columns, rtol, svd),
            "after_expression_and_pose": _remove_nuisance(
                identity, torch.cat((expression_columns, pose_columns), dim=1), rtol, svd,
            ),
        }
        adjusted, depth = _project_with_depth(torch.cat((identity, focal), dim=1), other, depth_column, rtol, svd)
        depth_rows.append(depth)
        if layout.shared_focal:
            joint = _compress(joint, adjusted, qr)
        else:
            blocks["after_all_nuisance"] = _remove_nuisance(identity, local, rtol, svd)
            # Put the focal column first to leave exactly one focal-bearing row
            # per view. Compress the remaining identity-only rows as they arrive.
            reduced = qr(torch.cat((adjusted[:, -1:], adjusted[:, :-1]), dim=1), mode="r").R
            distinct_focals.append(reduced[0])
            joint = _compress(joint, reduced[1:, 1:], qr) if index else reduced[1:, 1:]
        for name, block in blocks.items():
            identity_blocks[name] = _compress(identity_blocks[name], block, qr)
        if operations is not None and (index + 1) % 32 == 0:
            operations.drain()

    depths = torch.stack(depth_rows)
    if layout.shared_focal:
        identity_blocks["after_all_nuisance"] = _remove_nuisance(joint[:, :width], joint[:, width:], rtol, svd)
    else:
        focal_rows = torch.stack(distinct_focals)
        focal_count = len(data)
        joint = torch.cat((
            torch.cat((joint, joint.new_zeros((len(joint), focal_count))), dim=1),
            torch.cat((focal_rows[:, 1:], focal_rows[:, 0].diag()), dim=1),
        ))

    threshold = max(atol, rtol * float(svdvals(identity_blocks["fixed_nuisance"])[0]))
    conditional = _remove_nuisance(joint[:, width:], joint[:, :width], rtol, svd)
    joint_record = _describe(joint, rows, threshold, rtol, svdvals)
    conditional_record = _describe(conditional, rows, threshold, rtol, svdvals)
    _, _, vh = svd(joint, full_matrices=True)

    def describe_direction(direction, singular):
        identity, focal = direction[:width], direction[width:]
        depth = depths[:, :width] @ identity + depths[:, width] * focal
        return _direction(identity, focal, depth, names, singular)

    joint_record["weakest_directions"] = [
        describe_direction(vh[index], joint_record["singular_values"][index])
        for index in range(len(vh) - 1, max(-1, len(vh) - 4), -1)
    ]
    _, _, focal_vh = svd(conditional, full_matrices=True)
    focal_direction = focal_vh[-1]
    identity_direction = -pinv(joint[:, :width], rtol=rtol) @ (joint[:, width:] @ focal_direction)
    conditional_record["weakest_direction"] = describe_direction(
        torch.cat((identity_direction, focal_direction)), conditional_record["minimum_singular_value"],
    )
    focal_names = ["shared_log_focal"] if layout.shared_focal else list(data.ids)
    local_priors = {}
    for name, _ in layout.columns():
        if name in ("local_log_focal", "local_focal"):
            continue
        weight = getattr(parameters.config, f"{PARAMETER_GROUPS[name]}_weight")
        local_priors[name] = weight / getattr(parameters, name).numel()
    result = {
        "absolute_threshold": threshold,
        **{name: _describe(matrix, rows, threshold, rtol, svdvals) for name, matrix in identity_blocks.items()},
        "joint_identity_and_focal": joint_record, "conditional_focal": conditional_record,
        "focal_coordinate_names": focal_names,
        **({"camera_coordinate_units": {"camera_native": "m", "local_focal": "px"}}
           if parameters.config.lm.fp32_forward == "compensated" else {}),
        "fixed_nuisance_focal_column_l2": [math.hypot(*raw_focal_norms)] if layout.shared_focal else raw_focal_norms,
        "local_prior_jtj_diagonal": local_priors, "focal_prior_jtj_diagonal": 0.0,
    }
    if active_blocks:
        shared = int(layout.shared_focal and "camera" in stage_groups)
        local = torch.block_diag(*(block[:, shared:] for block in active_blocks))
        active = torch.cat((torch.cat([block[:, :shared] for block in active_blocks]), local), dim=1)
        result["active_calibration"] = _describe(active, rows, threshold, rtol, svdvals) | {
            "parameter_groups": list(stage_groups),
            "local_coordinates": [name for name in locals_ if PARAMETER_GROUPS[name] in stage_groups],
            "shared_focal": bool(shared),
        }
    if operations is not None:
        result["compressed_tensors"] = {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
                   "storage_bytes": tensor.numel() * tensor.element_size()}
            for name, tensor in (identity_blocks | {
                "joint_identity_and_focal": joint, "conditional_focal": conditional,
            }).items()
        }
    return result


def _state(model, data, config, selection, initial, fit, at):
    if fit is None:
        if at != "initial":
            raise ValueError("Final-state observability requires a fitted result")
        state, _ = initialize(model, data, config, FitInitialization() if initial is None else initial)
        return state
    if (
        initial is not None or not isinstance(fit, FitResult) or fit.config != config
        or fit.model != model.metadata or fit.dtype != model.numpy_dtype.name
        or fit.observation_ids != data.ids or fit.selection != selection
    ):
        raise ValueError("Observability requires the same model, configuration and ordered observations as the fit")
    if at in ("initial", "final"):
        snapshot = getattr(fit, at)
    else:
        snapshot = next((stage.after.state for stage in fit.stage_diagnostics if stage.name == at), None)
    if snapshot is None:
        raise ValueError(f"The fit has no {at} snapshot")
    # Evaluate the nuisance tangent at the actual camera, using local offsets from
    # that camera. Other native coordinates are copied directly from the snapshot.
    return TensorState(
        *(model.tensor(getattr(snapshot, name)) for name in ARRAY_FIELDS),
        tuple(copy_camera(camera, device=model.reference.device) for camera in snapshot.cameras),
    )


@torch.no_grad()
def identity_observability(model, observations, config, *, initial=None, fit=None,
                           at="initial", rtol=1e-6, atol=0.0, operations=None):
    """Measure unregularized sensitivity at initialization, final fit or focal calibration.

    Use the same landmark weights as fitting. One view is differentiated at a
    time and each stacked identity Jacobian is compressed by QR as views arrive.
    Ranks share max(atol, rtol * largest fixed-nuisance singular value).
    """
    rtol = finite_number(rtol, "Jacobian relative tolerance")
    atol = finite_number(atol, "Jacobian absolute tolerance")
    if not 0 < rtol < 1 or atol < 0:
        raise ValueError("Jacobian tolerances require 0 < rtol < 1 and atol >= 0")
    stage_groups = ()
    if at not in ("initial", "final"):
        stage_groups = next((
            stage.active_groups for stage in fit.stages
            if stage.name == at and stage.status != "skipped"
        ), ()) if isinstance(fit, FitResult) else ()
        if config.focal_mode == "fixed" or stage_groups != ("pose", "camera"):
            raise ValueError("Observability state must be initial, final or a traced focal calibration stage")
    if torch.is_inference_mode_enabled():
        raise ValueError("Jacobian analysis requires PyTorch autograd outside inference mode")
    validate_model(model)
    data = prepare_observations(observations, model.reference, config)
    selection = resolve_selection(model, config)
    width = len(selection.identity)
    if not width:
        raise ValueError("Jacobian analysis requires selected identity coefficients")
    state = _state(model, data, config, selection, initial, fit, at)
    parameters = FittingParameters(model, data, state, selection, config)
    layout = StageCoordinates.from_parameters(parameters, GROUPS)
    base = parameters.values()
    position = layout.pack(base, model.reference, len(data))
    if config.focal_mode != "fixed":
        result = {
            "state": at, "observation_ids": list(data.ids), "identity_names": list(selection.identity),
            "identity_parameter_count": width, "relative_tolerance": rtol, "absolute_tolerance": atol,
            "identity_prior_jtj_diagonal": config.identity_weight / width, "focal_mode": config.focal_mode,
        } | _focal_observability(data, parameters, layout, base, position, width, rtol, atol, stage_groups, operations)
        if stage_groups:
            result["joint_parameter_groups"] = list(GROUPS)
        return result
    expression_slice = layout.expression_slice()

    names = ("fixed_nuisance", "after_expression", "after_all_nuisance")
    compressed = [model.reference.new_empty((0, width), dtype=lm_dtype(config)) for _ in names]
    for index in range(len(data)):
        _, jacobian = observation_jacobian(data, parameters, layout, base, position, index)
        identity, nuisance = jacobian[:, :width], jacobian[:, width:]
        blocks = (
            identity,
            _remove_nuisance(identity, nuisance[:, expression_slice], rtol)
            if expression_slice is not None else identity,
            _remove_nuisance(identity, nuisance, rtol),
        )
        for group, block in enumerate(blocks):
            matrix = torch.cat((compressed[group], block), dim=0)
            compressed[group] = torch.linalg.qr(matrix, mode="r").R

    singular_values = [torch.linalg.svdvals(matrix) for matrix in compressed]
    threshold = max(atol, rtol * float(singular_values[0][0]))
    if not math.isfinite(threshold):
        raise FloatingPointError("Nonfinite observability threshold")
    return {
        "state": at, "observation_ids": list(data.ids), "identity_names": list(selection.identity),
        "identity_parameter_count": width, "relative_tolerance": rtol,
        "absolute_tolerance": atol, "absolute_threshold": threshold,
        "identity_prior_jtj_diagonal": config.identity_weight / width,
        **{
            name: _spectrum(matrix, values, threshold)
            for name, matrix, values in zip(names, compressed, singular_values, strict=True)
        },
    }
