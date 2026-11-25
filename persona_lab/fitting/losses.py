"""The globally normalized objective and separate unweighted pixel statistics."""

from dataclasses import dataclass
from collections import defaultdict
import math

import torch

from ..camera import project
from ..coordinates import gnm_to_frontal_axes
from ..gnm_geometry import extract_landmarks
from .parameters import InvalidGeometry, PARAMETER_GROUPS, stack_cameras


@dataclass(frozen=True, slots=True)
class LossValues:
    total: float
    landmarks: float
    identity: float
    expression: float
    pose: float
    camera: float


@dataclass(frozen=True, slots=True)
class PixelStatistics:
    eligible_count: int
    excluded_count: int
    mean_px: float
    rms_px: float
    max_px: float


@dataclass(frozen=True, slots=True)
class Reprojection:
    overall: PixelStatistics
    observations: tuple[PixelStatistics, ...]


def project_batch(model, state, observation_ids, *, check_geometry=False):
    vertices, landmarks = extract_landmarks(
        model.head, identity=state.identity, expression=state.expression,
        rotations=state.rotations, translation=state.translation,
    )
    if check_geometry and not torch.isfinite(vertices).all():
        raise InvalidGeometry("Nonfinite GNM vertex coordinates")
    if check_geometry and not torch.isfinite(landmarks).all():
        raise InvalidGeometry("Nonfinite GNM landmark coordinates")
    axes = gnm_to_frontal_axes(landmarks)
    groups = defaultdict(list)
    for index, camera in enumerate(state.cameras):
        groups[type(camera), camera.image_size].append(index)
    rows = [None] * len(axes)
    for indices in groups.values():
        cameras = stack_cameras([state.cameras[index] for index in indices])
        try:
            projected = project(axes[indices], cameras)
        except ValueError as error:
            identifiers = ", ".join(observation_ids[index] for index in indices)
            raise InvalidGeometry(f"Invalid projection in batch [{identifiers}]: {error}") from error
        for index, row in zip(indices, projected.unbind(0), strict=True):
            rows[index] = row
    return torch.stack(rows)


def _landmark_residuals(projected, data, start, stop):
    indices = data.gnm_indices[start:stop, :, None].expand(-1, -1, 2)
    mapped = projected.gather(1, indices)
    eligible = data.eligible[start:stop, :, None]
    # Zero both operands before subtraction to isolate masked observations.
    residual = mapped.masked_fill(~eligible, 0) - data.pixels[start:stop].masked_fill(~eligible, 0)
    return residual / data.short_edges[start:stop, None, None], residual


def landmark_contribution(projected, data, start, stop):
    """Return the batch's share of the total loss and squared pixel residuals."""
    normalized, residual = _landmark_residuals(projected, data, start, stop)
    loss = (normalized.square().sum(-1) * data.weights[start:stop]).sum() / data.weight_sum
    return loss, residual.detach().square().sum(-1)


def weighted_landmark_residuals(projected, data, start, stop):
    """Square-root weighting with the full dataset's eligible weight sum."""
    normalized, _ = _landmark_residuals(projected, data, start, stop)
    return _weighted(normalized, data, start, stop)


def _weighted(normalized, data, start, stop):
    mask = data.eligible[start:stop]
    scale = (data.weights[start:stop][mask].double() / data.weight_sum.double()).sqrt()
    return (normalized[mask].double() * scale[:, None]).flatten()


def regularization_residual(parameters, name, value, start=0, stop=None):
    """Return the residual and diagonal slope for one selected coordinate group."""
    group = PARAMETER_GROUPS[name]
    strength = getattr(parameters.config, f"{group}_weight")
    slope = math.sqrt(strength / getattr(parameters, name).numel())
    delta = parameters.penalty_delta(name, value, start, stop)
    return delta.double() * slope, slope


def _statistics(squared_pixels, counts):
    distances = squared_pixels.sqrt()
    total = counts.sum()
    eligible = int(total)
    overall = PixelStatistics(
        eligible, squared_pixels.numel() - eligible,
        float(distances.sum() / total),
        float((squared_pixels.sum() / total).sqrt()),
        float(distances.max()),
    )
    values = torch.stack((
        counts, distances.sum(1) / counts, (squared_pixels.sum(1) / counts).sqrt(),
        distances.amax(1),
    ), dim=1).cpu().tolist()
    observations = tuple(
        PixelStatistics(int(count), squared_pixels.shape[1] - int(count), mean, rms, maximum)
        for count, mean, rms, maximum in values
    )
    return Reprojection(overall, observations)


def evaluate_objective(model, data, parameters, active=(), *, candidate=None, least_squares=False):
    """Backpropagate each batch, then apply the global regularizers once."""
    batch_size = parameters.config.batch_size
    dtype = torch.float64 if least_squares else model.reference.dtype
    landmarks = model.reference.new_zeros((), dtype=dtype)
    squared_pixels = []
    with torch.set_grad_enabled(bool(active)):
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            values = None if candidate is None else candidate.batch(start, stop)
            state = parameters.state(start, stop, values)
            projected = project_batch(
                model, state, tuple(obs.observation_id for obs in data.observations[start:stop]),
                check_geometry=least_squares,
            )
            if least_squares:
                normalized, pixel_residual = _landmark_residuals(projected, data, start, stop)
                loss = _weighted(normalized, data, start, stop).square().sum()
                distances = pixel_residual.detach().double().square().sum(-1)
            else:
                loss, distances = landmark_contribution(projected, data, start, stop)
            if not torch.isfinite(loss) or not torch.isfinite(distances).all():
                raise FloatingPointError("Nonfinite landmark objective or pixel distances")
            if active:
                if not loss.requires_grad:
                    raise RuntimeError("Landmark objective has no differentiable fitting parameters")
                loss.backward()
            landmarks = landmarks + loss.detach()
            squared_pixels.append(distances)
            del state, projected, loss

        missing = [name for name, parameter in active if parameter.grad is None]
        if missing:
            raise RuntimeError(f"Missing landmark gradients for {', '.join(missing)}")
        if least_squares:
            values = parameters.values() if candidate is None else candidate
            penalties = dict.fromkeys(("identity", "expression", "pose", "camera"), landmarks.new_zeros(()))
            for name, group in PARAMETER_GROUPS.items():
                if (value := getattr(values, name)) is not None:
                    residual, _ = regularization_residual(parameters, name, value)
                    penalties[group] = penalties[group] + residual.square().sum()
            penalties = tuple(penalties.values())
        else:
            penalties = parameters.penalties(candidate)
        regularization = sum(penalties)
        total = landmarks + regularization
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite fitting objective")
        if active and regularization.requires_grad:
            regularization.backward()
    losses = LossValues(*torch.stack((total, landmarks, *penalties)).detach().cpu().tolist())
    stats = _statistics(torch.cat(squared_pixels), data.counts)
    if not all(math.isfinite(value) for value in (
        stats.overall.mean_px, stats.overall.rms_px, stats.overall.max_px,
    )):
        raise FloatingPointError("Nonfinite pixel statistics")
    return losses, stats
