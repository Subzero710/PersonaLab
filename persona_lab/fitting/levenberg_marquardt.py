"""Levenberg–Marquardt fitting in the existing native parameterization."""

from dataclasses import dataclass, replace
import math

import torch

from .losses import project_batch, regularization_residual, weighted_landmark_residuals
from .monitoring import OptimizationUpdate, Trial, finite_norm
from .parameters import InvalidGeometry, PARAMETER_GROUPS
from .schur import BlockVector, NormalBlocks, column_scales, damped_schur_step


@dataclass(frozen=True, slots=True)
class StageCoordinates:
    identity: bool
    names: tuple[str, ...]
    widths: tuple[int, ...]

    @classmethod
    def from_parameters(cls, parameters, groups):
        names = tuple(
            name for name, group in PARAMETER_GROUPS.items()
            if name != "identity" and group in groups and getattr(parameters, name) is not None
        )
        return cls(
            "identity" in groups, names, tuple(getattr(parameters, name).shape[1] for name in names),
        )

    def pack(self, values, reference, count):
        identity = values.identity.detach() if self.identity else reference.new_empty(0)
        local = (
            torch.cat([getattr(values, name).detach() for name in self.names], dim=1)
            if self.names else reference.new_empty((count, 0))
        )
        return BlockVector(identity, local)

    def unpack(self, values, point):
        changes = {name: point.local[:, columns] for name, columns in self.columns()}
        if self.identity:
            changes["identity"] = point.identity
        return replace(values, **changes)

    def columns(self):
        start = 0
        for name, width in zip(self.names, self.widths, strict=True):
            stop = start + width
            yield name, slice(start, stop)
            start = stop

    def expression_slice(self):
        return next((columns for name, columns in self.columns() if name == "expression"), None)

    def commit(self, parameters, point):
        if self.identity:
            parameters.identity.copy_(point.identity)
        for name, columns in self.columns():
            getattr(parameters, name).copy_(point.local[:, columns])


def observation_residual(model, data, parameters, layout, base, position, index, point):
    p = position.identity.numel()
    batch = layout.unpack(base.batch(index, index + 1), BlockVector(point[:p], point[None, p:]))
    state = parameters.state(index, index + 1, batch)
    projected = project_batch(model, state, (data.observations[index].observation_id,))
    return weighted_landmark_residuals(projected, data, index, index + 1)


@torch.enable_grad()
def observation_jacobian(model, data, parameters, layout, base, position, index):
    point = torch.cat((position.identity, position.local[index])).detach().requires_grad_()
    residual = observation_residual(model, data, parameters, layout, base, position, index, point)
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
        # local coordinates and the shared identity.
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
def normal_blocks(model, data, parameters, layout, base, position):
    """Accumulate double-precision normal blocks without keeping view Jacobians."""
    n = len(data)
    p, q = position.identity.numel(), position.local.shape[1]
    ref = model.reference
    identity = ref.new_zeros((p, p), dtype=torch.float64)
    coupling = ref.new_empty((n, p, q), dtype=torch.float64)
    local = ref.new_empty((n, q, q), dtype=torch.float64)
    gradient_a = ref.new_zeros(p, dtype=torch.float64)
    gradient_b = ref.new_empty((n, q), dtype=torch.float64)

    for index in range(n):
        residual, jacobian = observation_jacobian(model, data, parameters, layout, base, position, index)
        hessian = jacobian.mT @ jacobian
        gradient = jacobian.mT @ residual
        identity.add_(hessian[:p, :p])
        gradient_a.add_(gradient[:p])
        coupling[index] = hessian[:p, p:]
        local[index] = hessian[p:, p:]
        gradient_b[index] = gradient[p:]

    # A selected prior is linear in its raw optimizer coordinates. Work on
    # its diagonal directly, with the same full-group denominators as Adam.
    if p:
        prior, slope = regularization_residual(parameters, "identity", position.identity)
        identity.diagonal().add_(slope ** 2)
        gradient_a.add_(prior * slope)
    for name, columns in layout.columns():
        prior, slope = regularization_residual(parameters, name, position.local[:, columns])
        local.diagonal(dim1=-2, dim2=-1)[:, columns].add_(slope ** 2)
        gradient_b[:, columns].add_(prior * slope)
    if any(not torch.isfinite(block).all() for block in (identity, coupling, local, gradient_a, gradient_b)):
        raise FloatingPointError("Nonfinite Gauss–Newton normal blocks")
    return NormalBlocks(identity, coupling, local, BlockVector(gradient_a, gradient_b))


def bound_mask(position, gradient, expression_slice, limit):
    mask = torch.zeros_like(position.local, dtype=torch.bool)
    if expression_slice is not None:
        expression = position.local[:, expression_slice]
        g = gradient.local[:, expression_slice]
        mask[:, expression_slice] = ((expression <= -limit) & (g > 0)) | ((expression >= limit) & (g < 0))
    return mask


def gradient_norms(layout, gradient, fixed_local):
    norms = {"identity": finite_norm(2 * gradient.identity)} if layout.identity else {}
    local = 2 * gradient.local.masked_fill(fixed_local, 0)
    for name, columns in layout.columns():
        group = PARAMETER_GROUPS[name]
        norms[group] = math.hypot(norms.get(group, 0.0), finite_norm(local[:, columns]))
    if not all(math.isfinite(value) for value in norms.values()):
        raise FloatingPointError("Nonfinite projected gradient norm")
    return norms


def _candidate(position, step, expression_slice, limit):
    identity = (position.identity.double() + step.identity).to(position.identity.dtype)
    local = (position.local.double() + step.local).to(position.local.dtype)
    if not torch.isfinite(identity).all() or not torch.isfinite(local).all():
        raise FloatingPointError("Nonfinite trial parameter coordinates")
    if expression_slice is not None:
        local[:, expression_slice].clamp_(-limit, limit)
    return BlockVector(identity, local)


def _trial_step(model, data, parameters, base, position, layout, system, step,
                expression_slice, current_loss, measurements):
    norm, predicted, actual, ratio = None, None, None, None
    try:
        point = _candidate(position, step, expression_slice, parameters.config.expression_limit)
        effective = BlockVector(
            point.identity.double() - position.identity.double(),
            point.local.double() - position.local.double(),
        )
        norm = finite_norm(effective.identity, effective.local)
        reduction = system.predicted_reduction(effective)
        if not math.isfinite(reduction):
            raise FloatingPointError("Nonfinite predicted reduction")
        predicted = reduction
        if predicted <= 0:
            return None, "nonpositive_predicted_reduction", (norm, predicted, actual, ratio)
        values, stats = measurements.evaluate(
            model, data, parameters, candidate=layout.unpack(base, point), least_squares=True,
        )
        actual = current_loss - values.total
        gain = actual / predicted
        if not math.isfinite(gain):
            raise FloatingPointError("Nonfinite acceptance ratio")
        ratio = gain
        if actual <= 0 or ratio < parameters.config.lm.acceptance_threshold:
            return None, "insufficient_reduction", (norm, predicted, actual, ratio)
    except InvalidGeometry as error:
        return None, f"invalid_geometry: {error}", (norm, predicted, actual, ratio)
    except FloatingPointError as error:
        return None, f"nonfinite_trial: {error}", (norm, predicted, actual, ratio)
    return (point, values, stats), "accepted", (norm, predicted, actual, ratio)


@torch.no_grad()
def fit_stage(model, data, parameters, stage, groups, values, stats, measurements, record):
    config = parameters.config
    options = config.lm
    layout = StageCoordinates.from_parameters(parameters, groups)
    base = parameters.values()
    position = layout.pack(base, model.reference, len(data))
    expression_slice = layout.expression_slice()
    damping = options.initial_damping
    best, stale = values.total, 0
    for index in range(stage.iterations):
        measurements.jacobian_evaluations += 1
        with measurements.time("jacobian"):
            system = normal_blocks(model, data, parameters, layout, base, position)
        scales = column_scales(system, options.diagonal_floor)
        fixed_local = bound_mask(position, system.gradient, expression_slice, config.expression_limit)
        gradient = system.gradient
        scaled_norm = finite_norm(
            2 * (scales.identity * gradient.identity),
            2 * (scales.local * gradient.local).masked_fill(fixed_local, 0),
        )
        norms = gradient_norms(layout, gradient, fixed_local)
        if index == 0 or len(measurements.updates) % config.log_every == 0:
            record(stage.name, index, values, stats, norms)
        if scaled_norm <= options.scaled_gradient_tolerance or (
            config.gradient_tolerance is not None and math.hypot(*norms.values()) <= config.gradient_tolerance
        ):
            return values, stats, "converged", "projected gradient tolerance reached"

        trial = 0
        while trial < options.max_trials:
            trial += 1
            try:
                with measurements.time("linear_solve"):
                    step, condition = damped_schur_step(system, scales, damping, fixed_local)
            except (FloatingPointError, RuntimeError) as error:
                measurements.trials.append(Trial(
                    stage.name, index, trial, damping, False, str(error), scaled_norm,
                    None, None, None, None, {},
                ))
                raise
            candidate, reason, reductions = _trial_step(
                model, data, parameters, base, position, layout, system, step,
                expression_slice, values.total, measurements,
            )
            measurements.trials.append(Trial(
                stage.name, index, trial, damping, candidate is not None, reason, scaled_norm,
                *reductions, condition,
            ))
            if candidate is not None:
                position, values, stats = candidate
                layout.commit(parameters, position)
                measurements.last_objective = values
                measurements.updates.append(OptimizationUpdate(stage.name, index + 1, reductions[0]))
                ratio = reductions[-1]
                if ratio > 0.75:
                    damping = max(options.minimum_damping, damping / 3)
                elif ratio < 0.25:
                    damping = min(options.maximum_damping, damping * 4)
                break
            if damping >= options.maximum_damping:
                return values, stats, "no_acceptable_step", f"damping limit reached ({reason})"
            damping = min(options.maximum_damping, damping * 4)
        else:
            return values, stats, "no_acceptable_step", f"trial-step budget exhausted ({reason})"

        if config.improvement_tolerance is not None:
            if best - values.total > config.improvement_tolerance:
                best, stale = values.total, 0
            else:
                stale += 1
            if stale >= config.patience:
                return values, stats, "stopped", "objective improvement below tolerance"
    return values, stats, "budget_exhausted", "iteration budget reached"
