"""Prepare fitting-only observations without touching benchmark truth."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..coordinates import _validate_tensor
from ..correspondences import DFL_LANDMARK_COUNT
from ..dense_observations import DenseObservation, subset_mask
from ..observations import LandmarkObservation
from ..surface_correspondences import BoundSurface, SurfaceCorrespondences
from .configuration import FitConfig


@dataclass(frozen=True, slots=True)
class FittingObservations:
    """Stable observation ordering and the tensors shared by all iterations."""

    observations: tuple[LandmarkObservation | DenseObservation, ...]
    pixels: Tensor
    gnm_indices: Tensor
    weights: Tensor
    eligible: Tensor
    short_edges: Tensor
    weight_sum: Tensor
    counts: Tensor
    pixel_low: Tensor | None = None
    surface: BoundSurface | None = None

    @property
    def ids(self):
        return tuple(obs.observation_id for obs in self.observations)

    def __len__(self):
        return len(self.observations)


def prepare_observations(observations, reference: Tensor, config: FitConfig, *,
                         model=None, correspondences=None) -> FittingObservations:
    try:
        observations = tuple(observations)
    except TypeError as error:
        raise ValueError("Fitting observations must be an iterable of observations") from error
    if not observations:
        raise ValueError("Fitting requires at least one observation")
    dense = isinstance(observations[0], DenseObservation)
    kind = DenseObservation if dense else LandmarkObservation
    if any(not isinstance(obs, kind) for obs in observations):
        raise ValueError("Fitting requires one observation schema per run")
    surface = None
    if dense:
        first = observations[0]
        for obs in observations:
            if (obs.schema, obs.provider, obs.provider_version, obs.asset) != (
                first.schema, first.provider, first.provider_version, first.asset
            ):
                raise ValueError("Dense fitting requires identical schema, provider and asset across views")
        if not isinstance(correspondences, SurfaceCorrespondences) or model is None:
            raise ValueError("Dense fitting requires an explicit reviewed GNM correspondence definition")
        if correspondences.schema != first.schema:
            raise ValueError("Dense fitting correspondence definition has a different schema")
        surface = correspondences.bind(model)
        if not surface.slots:
            raise ValueError("Dense fitting has zero verified GNM correspondences")
        count = len(first.schema.points)
        dense_indices = [0] * count
        for row, slot in enumerate(surface.slots):
            dense_indices[slot] = row
    elif correspondences is not None:
        raise ValueError("Legacy observations retain their explicit DeepFaceLab mapping")
    else:
        count = DFL_LANDMARK_COUNT
    identifiers = tuple(obs.observation_id for obs in observations)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Fitting observation IDs must be unique")

    pixels, indices, weights, masks, edges, lows = [], [], [], [], [], []
    has_low = tuple(obs.pixel_low is not None for obs in observations)
    if any(has_low) and (not all(has_low) or config.lm.fp32_forward != "compensated"):
        raise ValueError("High/low targets require the explicit compensated FP32 forward for every view")
    for obs in observations:
        if dense:
            obs.validate()
            target = obs.landmarks
            mask = subset_mask(surface.eligible(obs), config.landmark_subset)
            observation_weights = obs.weights if obs.weights is not None else target.new_ones(count)
            mask = mask & (observation_weights > 0)
            indices.append(dense_indices)
        else:
            target = obs.aligned_pixels()
            mask = obs.fitting_mask(subset=config.landmark_subset)
            observation_weights = obs.weights
            indices.append([
                entry.gnm_index if entry.supported else 0 for entry in obs.correspondences.entries
            ])
        _validate_tensor(target, f"{obs.observation_id} landmarks", (count, 2), like=reference)
        # An observation holds tensors, so validate mutable weights at entry too.
        _validate_tensor(observation_weights, f"{obs.observation_id} weights", (count,),
                         like=reference)
        if (observation_weights < 0).any():
            raise ValueError(f"{obs.observation_id} contains negative landmark weights")
        if not mask.any():
            raise ValueError(f"{obs.observation_id} has no eligible fitting landmarks")
        masks.append(mask)
        pixels.append(target.detach())
        if obs.pixel_low is not None:
            _validate_tensor(obs.pixel_low, f"{obs.observation_id} pixel low", (DFL_LANDMARK_COUNT, 2), like=reference)
            lows.append(obs.pixel_low.detach())
        weights.append(observation_weights.detach().masked_fill(~mask, 0))
        edges.append(min(obs.image_size.width, obs.image_size.height))

    weight_tensor = torch.stack(weights)
    eligible = torch.stack(masks)
    weight_sum = weight_tensor.sum()
    short_edges = reference.new_tensor(edges)
    if not torch.isfinite(weight_sum) or not torch.isfinite(short_edges).all():
        raise ValueError("Fitting weights or image dimensions exceed the numerical range")
    return FittingObservations(
        observations, torch.stack(pixels),
        torch.tensor(indices, dtype=torch.long, device=reference.device),
        weight_tensor, eligible, short_edges, weight_sum, eligible.sum(dim=1),
        torch.stack(lows) if lows else None, surface,
    )
