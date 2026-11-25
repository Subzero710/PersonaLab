"""Controlled synthetic corruptions and scenario configuration checks."""

import numpy as np
import pytest

from persona_lab.benchmark.configuration import Noise, resolve_config, rng
from persona_lab.benchmark.noise import perturb
from persona_lab.correspondences import DFL_TO_GNM


def _sample(noise, index=0, *, eligible=None):
    clean = np.arange(136, dtype=np.float64).reshape(68, 2)
    valid = np.ones(68, dtype=bool) if eligible is None else eligible
    result = perturb(clean, valid, DFL_TO_GNM, noise, seed=17, subject=1, observation=index)
    np.testing.assert_array_equal(clean, np.arange(136).reshape(68, 2))
    return clean, result


def test_clean_copy_and_gaussian_noise_statistics():
    clean, result = _sample(Noise())
    np.testing.assert_array_equal(clean, result["observed"])
    assert not np.shares_memory(clean, result["observed"])
    assert result["validity"].all()
    offsets = np.concatenate([_sample(Noise(gaussian_px=2.0), i)[1]["offset_px"] for i in range(250)])
    np.testing.assert_allclose(offsets.mean(axis=0), [0, 0], atol=0.06, rtol=0)
    np.testing.assert_allclose(offsets.std(axis=0), [2, 2], atol=0.06, rtol=0)
    assert abs(np.corrcoef(offsets.T)[0, 1]) < 0.04


def test_regional_occlusion_missing_and_outlier_masks_with_known_radius():
    noise = Noise(missing=5, outliers=7, outlier_radius_px=(10, 10), occlude=("eyebrow",))
    clean, result = _sample(noise)
    assert np.flatnonzero(result["occluded"]).tolist() == list(range(17, 27))
    assert result["missing"].sum() == 5
    assert result["outliers"].sum() == 7
    assert not (result["occluded"] & result["missing"]).any()
    assert result["validity"][result["outliers"]].all()
    assert result["validity"][:17].any()
    np.testing.assert_array_equal(clean[~result["validity"]], result["observed"][~result["validity"]])
    np.testing.assert_allclose(
        np.linalg.norm(result["offset_px"][result["outliers"]], axis=1), 10, atol=1e-12, rtol=0,
    )
    _, repeated = _sample(noise)
    for key, array in result.items():
        np.testing.assert_array_equal(array, repeated[key])
    _, combined = _sample(Noise(
        missing=5, outliers=7, outlier_radius_px=(10, 10), occlude=("eyebrow",), gaussian_px=2,
    ))
    for key in ("validity", "missing", "occluded", "outliers"):
        np.testing.assert_array_equal(combined[key], result[key])
    assert np.any(combined["offset_px"] != result["offset_px"])


@pytest.mark.parametrize("options", [
    {"gaussian_px": -1}, {"gaussian_px": np.inf}, {"missing": True},
    {"outliers": 69}, {"outlier_radius_px": (0, 1)},
    {"outlier_radius_px": (11, 10)}, {"occlude": ("invalid",)},
    {"occlude": ("eye", "eye")},
])
def test_invalid_noise_configuration(options):
    with pytest.raises(ValueError):
        Noise(**options)


def test_insufficient_eligible_slots_is_an_explicit_error():
    eligible = np.zeros(68, dtype=bool)
    eligible[[7, 8]] = True
    with pytest.raises(ValueError, match="Too few"):
        _sample(Noise(missing=2, outliers=1), eligible=eligible)


@pytest.mark.parametrize("scenario,options", [
    ("A", {"camera": "both"}), ("A", {"observations_per_subject": 2}),
    ("C", {"observations_per_subject": 8}), ("C", {"expressions": ["neutral"]}),
    ("E", {"noise": {"structured": {}}}), ("F", {"subjects": 1}),
    ("F", {"view_degrees": {"frontal": 0}}), ("F", {"image_size": [0, 20]}),
    ("F", {"seed": -1}), ("F", {"identity_half_range": 0}),
])
def test_invalid_parameter_or_coverage_options(scenario, options):
    with pytest.raises(ValueError):
        resolve_config(scenario, **options)


def test_seeded_independence_and_different_seed():
    values = rng(42, "identity", 0).standard_normal(10)
    rng(42, "gaussian", 0).standard_normal(200)
    np.testing.assert_array_equal(values, rng(42, "identity", 0).standard_normal(10))
    assert not np.array_equal(values, rng(43, "identity", 0).standard_normal(10))
    assert not np.array_equal(values, rng(42, "identity", 1).standard_normal(10))
