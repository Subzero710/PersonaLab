"""Levenberg–Marquardt fitting in the existing native parameterization."""

from dataclasses import dataclass, replace
import math

import torch

from .losses import lm_dtype, project_landmarks, regularization_residual, weighted_landmark_residuals
from .monitoring import Linearization, OptimizationUpdate, Trial, finite_norm
from .parameters import FOCAL_PARAMETERS, InvalidGeometry, PARAMETER_GROUPS, SHARED_PARAMETERS
from .schur import BlockVector, NormalBlocks, column_scales, damped_schur_step
from .square_root import SquareRootBlocks, square_root_step
from .batched_jacobian import batch_jacobian, residual_fp32_batch


def _lm_norm(*values):
    return finite_norm(*values, dtype=values[0].dtype)


@dataclass(frozen=True, slots=True)
class StageCoordinates:
    identity: bool
    names: tuple[str, ...]
    widths: tuple[int, ...]
    shared_focal: bool = False

    @classmethod
    def from_parameters(cls, parameters, groups):
        names = tuple(
            name for name, group in PARAMETER_GROUPS.items()
            if name not in SHARED_PARAMETERS and group in groups and getattr(parameters, name) is not None
        )
        return cls(
            "identity" in groups and parameters.identity is not None,
            names, tuple(getattr(parameters, name).shape[1] for name in names),
            "camera" in groups and parameters.shared_log_focal is not None,
        )

    def pack(self, values, reference, count):
        identity = values.identity.detach() if self.identity else reference.new_empty(0)
        if self.shared_focal:
            identity = torch.cat((identity, values.shared_log_focal.detach()))
        local = (
            torch.cat([getattr(values, name).detach() for name in self.names], dim=1)
            if self.names else reference.new_empty((count, 0))
        )
        return BlockVector(identity, local)

    def unpack(self, values, point):
        changes = {name: point.local[:, columns] for name, columns in self.columns()}
        changes.update((name, point.identity[columns]) for name, columns in self.shared_columns())
        return replace(values, **changes)

    def shared_columns(self):
        if self.identity:
            yield "identity", slice(None, -1 if self.shared_focal else None)
        if self.shared_focal:
            yield "shared_log_focal", slice(-1, None)

    def columns(self):
        start = 0
        for name, width in zip(self.names, self.widths, strict=True):
            stop = start + width
            yield name, slice(start, stop)
            start = stop

    def expression_slice(self):
        return next((columns for name, columns in self.columns() if name == "expression"), None)

    def focal_coordinates(self, vector):
        if self.shared_focal:
            return vector.identity[-1:]
        return next((vector.local[:, columns] for name, columns in self.columns() if name in ("local_log_focal", "local_focal")), None)

    def commit(self, parameters, point):
        for name, columns in self.shared_columns():
            getattr(parameters, name).copy_(point.identity[columns])
        for name, columns in self.columns():
            getattr(parameters, name).copy_(point.local[:, columns])


def observation_residual(data, parameters, layout, base, position, index, point):
    p = position.identity.numel()
    batch = layout.unpack(base.batch(index, index + 1), BlockVector(point[:p], point[None, p:]))
    if lm_dtype(parameters.config) == torch.float32:
        normalized, _ = residual_fp32_batch(data, parameters, index, index + 1, batch)
        from .losses import _weighted
        return _weighted(normalized, data, index, index + 1, dtype=torch.float32)
    state = parameters.state(index, index + 1, batch)
    landmarks = parameters.landmark_geometry(state.identity, state.expression, state.rotations, state.translation)
    projected = project_landmarks(landmarks, state.cameras, (data.ids[index],))
    return weighted_landmark_residuals(projected, data, index, index + 1, dtype=lm_dtype(parameters.config))


@torch.enable_grad()
def observation_jacobian(data, parameters, layout, base, position, index):
    point = torch.cat((position.identity, position.local[index])).detach().requires_grad_()
    residual = observation_residual(data, parameters, layout, base, position, index, point)
    if not torch.isfinite(residual).all():
        raise FloatingPointError("Nonfinite landmark residual at the accepted state")
    count = residual.numel()
    jacobian = residual.new_empty((count, point.numel()))
    rows = parameters.config.lm.jacobian_rows
    for start in range(0, count, rows):
        stop = min(start + rows, count)
        if rows == 1:
            gradient, = torch.autograd.grad(residual[start], point, retain_graph=stop < count)
            jacobian[start] = gradient
            continue
        # Differentiate a bounded group of output rows against just this view's
        # local coordinates and the shared coordinates.
        seeds = residual.new_zeros((stop - start, count))
        seeds[:, start:stop].diagonal().fill_(1)
        gradient, = torch.autograd.grad(
            residual, point, grad_outputs=seeds, is_grads_batched=True,
            retain_graph=stop < count,
        )
        jacobian[start:stop] = gradient
    if not torch.isfinite(jacobian).all():
        raise FloatingPointError("Nonfinite landmark Jacobian at the accepted state")
    return residual.detach(), jacobian


@torch.no_grad()
def normal_blocks(model, data, parameters, layout, base, position, *, operations=None):
    """Accumulate normal blocks in the configured LM precision without keeping view Jacobians."""
    n = len(data)
    p, q = position.identity.numel(), position.local.shape[1]
    ref = model.reference
    identity = ref.new_zeros((p, p), dtype=lm_dtype(parameters.config))
    coupling = ref.new_empty((n, p, q), dtype=lm_dtype(parameters.config))
    local = ref.new_empty((n, q, q), dtype=lm_dtype(parameters.config))
    gradient_a = ref.new_zeros(p, dtype=lm_dtype(parameters.config))
    gradient_b = ref.new_empty((n, q), dtype=lm_dtype(parameters.config))

    for index in range(n):
        if operations is None:
            residual, jacobian = observation_jacobian(data, parameters, layout, base, position, index)
        else:
            with operations.measure("observation_jacobian"):
                residual, jacobian = observation_jacobian(data, parameters, layout, base, position, index)
        hessian = jacobian.mT @ jacobian
        gradient = jacobian.mT @ residual
        identity.add_(hessian[:p, :p])
        gradient_a.add_(gradient[:p])
        coupling[index] = hessian[:p, p:]
        local[index] = hessian[p:, p:]
        gradient_b[index] = gradient[p:]

    # A selected prior is linear in its raw optimizer coordinates. Work on
    # its diagonal directly, with the same full-group denominators as Adam.
    for name, columns in layout.shared_columns():
        prior, slope = regularization_residual(parameters, name, position.identity[columns])
        identity.diagonal()[columns].add_(slope ** 2)
        gradient_a[columns].add_(prior * slope)
    for name, columns in layout.columns():
        prior, slope = regularization_residual(parameters, name, position.local[:, columns])
        local.diagonal(dim1=-2, dim2=-1)[:, columns].add_(slope ** 2)
        gradient_b[:, columns].add_(prior * slope)
    if any(not torch.isfinite(block).all() for block in (identity, coupling, local, gradient_a, gradient_b)):
        raise FloatingPointError("Nonfinite Gauss–Newton normal blocks")
    return NormalBlocks(identity, coupling, local, BlockVector(gradient_a, gradient_b))


@torch.no_grad()
def square_root_blocks(data, parameters, layout, position, *, operations=None):
    residuals, jacobians = [], []
    batch = parameters.config.batch_size
    for start in range(0, len(data), batch):
        stop = min(len(data), start + batch)
        if operations is None:
            residual, jacobian = batch_jacobian(data, parameters, layout, position, start, stop)
        else:
            with operations.measure("observation_jacobian_batch"):
                residual, jacobian = batch_jacobian(data, parameters, layout, position, start, stop)
        if operations is not None:
            operations.counts["observation_jacobian"] += stop - start
        residuals.append(residual)
        jacobians.append(jacobian)
    return SquareRootBlocks.create(torch.cat(jacobians), torch.cat(residuals), position.identity.numel(), batch)


def bound_mask(position, gradient, expression_slice, limit):
    mask = torch.zeros_like(position.local, dtype=torch.bool)
    if expression_slice is not None:
        expression = position.local[:, expression_slice]
        g = gradient.local[:, expression_slice]
        mask[:, expression_slice] = ((expression <= -limit) & (g > 0)) | ((expression >= limit) & (g < 0))
    return mask


def gradient_norms(layout, gradient, fixed_local):
    norms = {
        ("focal" if name in FOCAL_PARAMETERS else name): _lm_norm(2 * gradient.identity[columns])
        for name, columns in layout.shared_columns()
    }
    local = 2 * gradient.local.masked_fill(fixed_local, 0)
    for name, columns in layout.columns():
        group = "focal" if name in FOCAL_PARAMETERS else PARAMETER_GROUPS[name]
        norms[group] = math.hypot(norms.get(group, 0.0), _lm_norm(local[:, columns]))
    if not all(math.isfinite(value) for value in norms.values()):
        raise FloatingPointError("Nonfinite projected gradient norm")
    return norms


def _candidate(position, step, expression_slice, limit):
    identity = (position.identity.to(step.identity.dtype) + step.identity).to(position.identity.dtype)
    local = (position.local.to(step.local.dtype) + step.local).to(position.local.dtype)
    if not torch.isfinite(identity).all() or not torch.isfinite(local).all():
        raise FloatingPointError("Nonfinite trial parameter coordinates")
    if expression_slice is not None:
        local[:, expression_slice].clamp_(-limit, limit)
    return BlockVector(identity, local)


def _effective_step(position, point, dtype=torch.float64):
    return BlockVector(
        point.identity.to(dtype) - position.identity.to(dtype),
        point.local.to(dtype) - position.local.to(dtype),
    )


def _step_norms(effective, scales):
    return (
        _lm_norm(effective.identity, effective.local),
        _lm_norm(effective.identity / scales.identity, effective.local / scales.local),
    )


def _focal_norms(layout, vector, scales, *, gradient=False):
    values = layout.focal_coordinates(vector)
    if values is None:
        return None
    scale = layout.focal_coordinates(scales)
    if gradient:
        values = 2 * values
    scaled = values * scale if gradient else values / scale
    return {"native": _lm_norm(values), "scaled": _lm_norm(scaled)}


def _trial_focal_displacement(layout, position, step, scales):
    current = layout.focal_coordinates(position)
    if current is None:
        return None
    focal_step = layout.focal_coordinates(step)
    proposed = (current.to(focal_step.dtype) + focal_step).to(current.dtype)
    update = proposed.to(focal_step.dtype) - current.to(focal_step.dtype)
    return {"native": _lm_norm(update), "scaled": _lm_norm(update / layout.focal_coordinates(scales))}


def _trial_step(model, data, parameters, base, position, layout, system, scales, step,
                expression_slice, current_loss, measurements):
    norm = scaled = predicted = actual = ratio = None
    try:
        point = _candidate(position, step, expression_slice, parameters.config.expression_limit)
        effective = _effective_step(position, point, system.gradient.identity.dtype)
        norm, scaled = _step_norms(effective, scales)
        reduction = system.predicted_reduction(effective)
        if not math.isfinite(reduction):
            raise FloatingPointError("Nonfinite predicted reduction")
        predicted = reduction
        if predicted <= 0:
            return None, "nonpositive_predicted_reduction", (norm, scaled, predicted, actual, ratio)
        values, stats = measurements.evaluate(
            model, data, parameters, candidate=layout.unpack(base, point), least_squares=True,
        )
        actual = current_loss - values.total
        gain = actual / predicted
        if not math.isfinite(gain):
            raise FloatingPointError("Nonfinite acceptance ratio")
        ratio = gain
        if actual <= 0 or ratio < parameters.config.lm.acceptance_threshold:
            return None, "insufficient_reduction", (norm, scaled, predicted, actual, ratio)
    except InvalidGeometry as error:
        return None, f"invalid_geometry: {error}", (norm, scaled, predicted, actual, ratio)
    except FloatingPointError as error:
        return None, f"nonfinite_trial: {error}", (norm, scaled, predicted, actual, ratio)
    return (point, values, stats), "accepted", (norm, scaled, predicted, actual, ratio)


def _precision_limited(values, scaled_norm, reason):
    # Explicit near-roundoff stop, NOT strict convergence. No GT is consulted.
    eps = torch.finfo(torch.float32).eps
    return (reason in ("insufficient_reduction", "nonpositive_predicted_reduction")
            and math.sqrt(values.landmarks) <= 4 * eps
            and scaled_norm <= 8 * eps)


@torch.no_grad()
def fit_stage(model, data, parameters, stage, groups, values, stats, measurements, record):
    config = parameters.config
    options = config.lm
    layout = StageCoordinates.from_parameters(parameters, groups)
    base = parameters.values()
    position = layout.pack(base, model.reference, len(data))
    expression_slice = layout.expression_slice()
    damping = options.initial_damping
    measurements.stage_damping = damping
    operations = measurements.operations
    profile_options = {"operations": operations} if operations is not None else {}
    best, stale = values.total, 0
    square_root = options.arithmetic_dtype == "float32"
    solve_step = square_root_step if square_root else damped_schur_step
    scale_columns = (lambda system, floor: system.scales(floor)) if square_root else column_scales
    for index in range(stage.iterations):
        measurements.jacobian_evaluations += 1
        with measurements.time("jacobian"):
            if square_root:
                system = square_root_blocks(data, parameters, layout, position, **profile_options)
            else:
                system = normal_blocks(model, data, parameters, layout, base, position, **profile_options)
        if operations is None:
            scales = scale_columns(system, options.diagonal_floor)
        else:
            operations.drain()
            if measurements.normal_system is None and square_root:
                measurements.normal_system = {
                    "representation": "square_root_qr", "normal_equations_bytes": 0,
                    "shared_dimensions": [position.identity.numel()] * 2,
                    "local_dimensions": [position.local.shape[1]] * 2,
                    "local_blocks": len(data), "dtype": str(system.shared.dtype),
                    "local_coordinate_widths": dict(zip(layout.names, layout.widths, strict=True)),
                    "device": str(system.shared.device),
                    "active_trainable_parameters": position.identity.numel() + position.local.numel(),
                    "valid_residual_coordinates": 2 * int(data.counts.sum()),
                    "jacobian_storage_bytes": sum(t.numel() * t.element_size() for t in (system.shared, system.local, system.residual)),
                    "jacobian_observation_batch_size": config.batch_size,
                }
            elif measurements.normal_system is None:
                storage = {
                    name: block.numel() * block.element_size()
                    for name, block in (
                        ("shared", system.identity), ("local", system.local), ("coupling", system.coupling),
                        ("shared_gradient", system.gradient.identity), ("local_gradient", system.gradient.local),
                    )
                }
                measurements.normal_system = {
                    "shared_dimensions": list(system.identity.shape),
                    "local_dimensions": list(system.local.shape[1:]),
                    "local_coordinate_widths": dict(zip(layout.names, layout.widths, strict=True)),
                    "local_blocks": len(system.local),
                    "active_trainable_parameters": position.identity.numel() + position.local.numel(),
                    "valid_residual_coordinates": 2 * int(data.counts.sum()),
                    "normal_equations_bytes": sum(storage.values()), "storage_bytes": storage,
                    "dtype": str(system.gradient.identity.dtype), "device": str(system.identity.device),
                }
            with operations.measure("column_equilibration"):
                scales = scale_columns(system, options.diagonal_floor)
        fixed_local = bound_mask(position, system.gradient, expression_slice, config.expression_limit)
        gradient = system.gradient
        scaled_norm = _lm_norm(
            2 * (scales.identity * gradient.identity),
            2 * (scales.local * gradient.local).masked_fill(fixed_local, 0),
        )
        norms = gradient_norms(layout, gradient, fixed_local)
        projected_norm = math.hypot(*norms.values())
        gradient_ready = scaled_norm <= options.scaled_gradient_tolerance and (
            config.gradient_tolerance is None or projected_norm <= config.gradient_tolerance
        )
        probe_norms = (None, None)
        focal_probe = None
        step_tolerances = (options.native_step_tolerance, options.scaled_step_tolerance)
        check_step = any(tolerance is not None for tolerance in step_tolerances)
        if gradient_ready and check_step:
            # Strong damping can make a large unresolved displacement appear small.
            # Probe at the configured floor without evaluating or accepting a trial.
            with measurements.time("linear_solve"):
                probe, _ = solve_step(system, scales, options.minimum_damping, fixed_local)
            point = _candidate(position, probe, expression_slice, config.expression_limit)
            effective = _effective_step(position, point, system.gradient.identity.dtype)
            probe_norms = _step_norms(effective, scales)
            focal_probe = _focal_norms(layout, effective, scales)
        measurements.linearizations.append(Linearization(
            stage.name, index, values.total, damping, projected_norm, scaled_norm,
            int(fixed_local.sum()), *probe_norms,
            _focal_norms(layout, gradient, scales, gradient=True), focal_probe,
        ))
        if index == 0 or len(measurements.updates) % config.log_every == 0:
            record(stage.name, index, values, stats, norms)
        if gradient_ready and all(
            tolerance is None or norm <= tolerance
            for norm, tolerance in zip(probe_norms, step_tolerances, strict=True)
        ):
            reason = "scaled projected gradient and displacement tolerances reached" if check_step else (
                "scaled projected gradient tolerance reached"
            )
            return values, stats, "converged", reason

        trial = 0
        while trial < options.max_trials:
            trial += 1
            try:
                with measurements.time("linear_solve"):
                    step, condition = solve_step(system, scales, damping, fixed_local)
            except (FloatingPointError, RuntimeError) as error:
                measurements.trials.append(Trial(
                    stage.name, index, trial, damping, False, str(error), scaled_norm,
                    None, None, None, None, {},
                ))
                raise
            candidate, reason, reductions = _trial_step(
                model, data, parameters, base, position, layout, system, scales, step,
                expression_slice, values.total, measurements,
            )
            native_step, scaled_step, predicted, actual, ratio = reductions
            focal = _trial_focal_displacement(layout, position, step, scales) if native_step is not None else None
            measurements.trials.append(Trial(
                stage.name, index, trial, damping, candidate is not None, reason, scaled_norm,
                native_step, predicted, actual, ratio, condition, scaled_step, focal,
            ))
            if candidate is not None:
                position, values, stats = candidate
                layout.commit(parameters, position)
                measurements.last_objective = values
                measurements.updates.append(OptimizationUpdate(
                    stage.name, index + 1, native_step, scaled_step, parameters.focal_factors(),
                ))
                if ratio > 0.75:
                    damping = max(options.minimum_damping, damping / 3)
                elif ratio < 0.25:
                    damping = min(options.maximum_damping, damping * 4)
                measurements.stage_damping = damping
                measurements.trace_accepted_state(parameters, len(data), values)
                break
            if damping >= options.maximum_damping:
                if square_root and _precision_limited(values, scaled_norm, reason):
                    return values, stats, "stopped", "FP32 small-residual stagnation; recovery targets evaluated separately"
                return values, stats, "no_acceptable_step", f"damping limit reached ({reason})"
            damping = min(options.maximum_damping, damping * 4)
            measurements.stage_damping = damping
        else:
            if square_root and _precision_limited(values, scaled_norm, reason):
                return values, stats, "stopped", "FP32 small-residual stagnation; recovery targets evaluated separately"
            return values, stats, "no_acceptable_step", f"trial-step budget exhausted ({reason})"

        if config.improvement_tolerance is not None:
            if best - values.total > config.improvement_tolerance:
                best, stale = values.total, 0
            else:
                stale += 1
            if stale >= config.patience:
                return values, stats, "stopped", "objective improvement below tolerance"
    return values, stats, "budget_exhausted", "iteration budget reached"
