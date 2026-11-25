"""The globally normalized objective and separate unweighted pixel statistics."""

from dataclasses import dataclass
from collections import defaultdict
import math

import torch

from ..camera import project
from ..coordinates import gnm_to_frontal_axes
from ..gnm_geometry import extract_landmarks
from .parameters import stack_cameras


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


def project_batch(model, state, observation_ids):
    _, landmarks = extract_landmarks(
        model.head, identity=state.identity, expression=state.expression,
        rotations=state.rotations, translation=state.translation,
    )
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
            raise ValueError(f"Invalid projection in batch [{identifiers}]: {error}") from error
        for index, row in zip(indices, projected.unbind(0), strict=True):
            rows[index] = row
    return torch.stack(rows)


def landmark_contribution(projected, data, start, stop):
    """Return the batch's share of the total loss and squared pixel residuals."""
    indices = data.gnm_indices[start:stop, :, None].expand(-1, -1, 2)
    mapped = projected.gather(1, indices)
    eligible = data.eligible[start:stop, :, None]
    # Zero both operands before subtraction to isolate masked observations.
    residual = mapped.masked_fill(~eligible, 0) - data.pixels[start:stop].masked_fill(~eligible, 0)
    normalized = residual / data.short_edges[start:stop, None, None]
    loss = (normalized.square().sum(-1) * data.weights[start:stop]).sum() / data.weight_sum
    return loss, residual.detach().square().sum(-1)


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


def evaluate_objective(model, data, parameters, active=()):
    """Backpropagate each batch, then apply the global regularizers once."""
    batch_size = parameters.config.batch_size
    landmarks = model.reference.new_zeros(())
    squared_pixels = []
    with torch.set_grad_enabled(bool(active)):
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            state = parameters.state(start, stop)
            projected = project_batch(
                model, state, tuple(obs.observation_id for obs in data.observations[start:stop]),
            )
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
        penalties = parameters.penalties()
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
