"""Controlled landmark corruptions in DeepFaceLab's fixed ordering."""

import numpy as np

from ..correspondences import DFL_LANDMARK_COUNT
from .configuration import Noise, rng


def perturb(clean, eligible, mapping, noise: Noise, *, seed, subject, observation):
    """Occlude, sample missing slots, add Gaussian noise, then displace outliers."""
    count = DFL_LANDMARK_COUNT
    if (
        clean.shape != (count, 2) or clean.dtype not in (np.float32, np.float64)
        or not np.isfinite(clean).all()
        or eligible.shape != (count,) or eligible.dtype != np.bool_
    ):
        raise ValueError("Perturbations require finite (68, 2) pixels and a boolean (68,) mask")
    regions = [entry.region.value for entry in mapping.entries]
    occluded = eligible & np.isin(regions, noise.occlude)
    candidates = np.flatnonzero(eligible & ~occluded)
    if noise.missing + noise.outliers > len(candidates):
        raise ValueError("Too few eligible points for the requested missing and outlier counts")
    missing = np.zeros(count, dtype=bool)
    if noise.missing:
        missing[rng(seed, "missing", subject, observation).choice(
            candidates, size=noise.missing, replace=False,
        )] = True
    validity = eligible & ~missing & ~occluded
    observed = clean.copy()
    if noise.gaussian_px:
        offsets = rng(seed, "gaussian", subject, observation).normal(
            0, noise.gaussian_px, size=clean.shape,
        )
        observed[validity] += offsets[validity]
    outliers = np.zeros(count, dtype=bool)
    if noise.outliers:
        indices = rng(seed, "outlier_indices", subject, observation).choice(
            np.flatnonzero(validity), size=noise.outliers, replace=False,
        )
        outliers[indices] = True
        offsets = rng(seed, "outlier_offsets", subject, observation)
        angles = offsets.uniform(0, 2 * np.pi, size=noise.outliers)
        radii = offsets.uniform(*noise.outlier_radius_px, size=noise.outliers)
        observed[indices] += radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    offset_px = observed - clean
    if not np.isfinite(offset_px).all() or not np.isfinite(observed).all():
        raise ValueError("Configured noise produced nonfinite pixel values in the selected dtype")
    return {
        "observed": observed, "offset_px": offset_px, "validity": validity,
        "missing": missing, "occluded": occluded, "outliers": outliers,
    }
