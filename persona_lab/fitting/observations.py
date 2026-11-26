"""Prepare fitting-only observations without touching benchmark truth."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..coordinates import _validate_tensor
from ..correspondences import DFL_LANDMARK_COUNT
from ..observations import LandmarkObservation
from .configuration import FitConfig


@dataclass(frozen=True, slots=True)
class FittingObservations:
    """Stable observation ordering and the tensors shared by all iterations."""

    observations: tuple[LandmarkObservation, ...]
    pixels: Tensor
    gnm_indices: Tensor
    weights: Tensor
    eligible: Tensor
    short_edges: Tensor
    weight_sum: Tensor
    counts: Tensor
    pixel_low: Tensor | None = None

    @property
    def ids(self):
        return tuple(obs.observation_id for obs in self.observations)

    def __len__(self):
        return len(self.observations)


def prepare_observations(observations, reference: Tensor, config: FitConfig) -> FittingObservations:
    try:
        observations = tuple(observations)
    except TypeError as error:
        raise ValueError("Fitting observations must be an iterable of LandmarkObservation objects") from error
    if not observations or any(not isinstance(obs, LandmarkObservation) for obs in observations):
        raise ValueError("Fitting requires at least one LandmarkObservation")
    identifiers = tuple(obs.observation_id for obs in observations)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Fitting observation IDs must be unique")

    pixels, indices, weights, masks, edges, lows = [], [], [], [], [], []
    has_low = tuple(obs.pixel_low is not None for obs in observations)
    if any(has_low) and (not all(has_low) or config.lm.fp32_forward != "compensated"):
        raise ValueError("High/low targets require the explicit compensated FP32 forward for every view")
    for obs in observations:
        target = obs.aligned_pixels()
        _validate_tensor(target, f"{obs.observation_id} landmarks", (DFL_LANDMARK_COUNT, 2),
                         like=reference)
        # An observation holds tensors, so validate mutable weights at entry too.
        _validate_tensor(obs.weights, f"{obs.observation_id} weights", (DFL_LANDMARK_COUNT,),
                         like=reference)
        if (obs.weights < 0).any():
            raise ValueError(f"{obs.observation_id} contains negative landmark weights")
        mask = obs.fitting_mask(subset=config.landmark_subset)
        if not mask.any():
            raise ValueError(f"{obs.observation_id} has no eligible fitting landmarks")
        masks.append(mask)
        pixels.append(target.detach())
        if obs.pixel_low is not None:
            _validate_tensor(obs.pixel_low, f"{obs.observation_id} pixel low", (DFL_LANDMARK_COUNT, 2), like=reference)
            lows.append(obs.pixel_low.detach())
        weights.append(obs.weights.detach().masked_fill(~mask, 0))
        indices.append([
            entry.gnm_index if entry.supported else 0 for entry in obs.correspondences.entries
        ])
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
        torch.stack(lows) if lows else None,
    )
