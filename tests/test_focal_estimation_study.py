"""V5.2 experimental design, dataset provenance and truthful failure reporting."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from persona_lab.benchmark import fitting_study, focal_estimation
from persona_lab.benchmark.configuration import resolve_config
from persona_lab.benchmark.dataset import load_dataset
from persona_lab.benchmark.generate import generate_dataset
from persona_lab.benchmark.model import _cached_asset
from persona_lab.benchmark.storage import read_json
from persona_lab.fitting import FitConfig


STUDY = focal_estimation.STUDY
CONDITIONS = {
    "A1": ("fixed", 1.4), "A2": ("fixed", 1.5),
    "A3": ("shared", 1.5), "A4": ("per-observation", 1.5),
    "A5": ("shared", 1.0), "A6": ("shared", 2.0),
    "A7": ("per-observation", 1.0), "A8": ("per-observation", 2.0),
    "B1": ("fixed", 1.5), "B2": ("shared", 1.5),
    "B3": ("per-observation", 1.5), "B4": ("fixed", None),
}
SCHEDULE = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 1.1, 1.5, 1.9)


@pytest.fixture(scope="module")
def official_asset():
    return os.environ.get("PERSONALAB_GNM_MODEL_FILE") or str(_cached_asset())


@pytest.fixture(scope="module")
def study_jobs(tmp_path_factory, official_asset):
    output = tmp_path_factory.mktemp("v52")
    return list(fitting_study._corrected_jobs(SimpleNamespace(
        identity_studies=[STUDY], seeds=(42, 43, 44), focal_study_budget=100, lm_iterations=(0, 0, 30),
        model_file=official_asset, output=output, dtype="float64", device="cpu", spectra=True,
    )))


def test_exact_focal_study_matrix_and_cold_starts(study_jobs, monkeypatch):
    assert STUDY not in fitting_study.IDENTITY_STUDIES
    assert len(study_jobs) == 36
    by_case = {(job["seed"], job[STUDY]["condition"]): job for job in study_jobs}
    for seed in (42, 43, 44):
        for condition, (mode, factor) in CONDITIONS.items():
            job = by_case[seed, condition]
            expected = "known" if condition == "B4" else "estimated"
            config = FitConfig.from_dict(job["config"])
            assert (config.camera_mode, config.focal_mode) == (expected, mode)
            assert job[STUDY]["configured_initial_focal_factor"] == factor
            assert config.initial_focal_factor == (factor if mode != "fixed" else None)
            assert tuple(stage.iterations for stage in config.stages) == (0, 0, 100)
            assert config.lm == fitting_study.STRICT_LM
            assert config.identity_weight == config.expression_weight == config.pose_weight == config.camera_weight == 0
            assert job["observations"] == 9 and not job["known_pose_diagnostic"]
            assert job["spectra"] and job["final_spectra"]
        for experiment in "AB":
            matched = [job for job in study_jobs if job["seed"] == seed and job[STUDY]["experiment"] == experiment]
            assert len({job["dataset"] for job in matched}) == 1
            assert all(job["expected_dataset"] == matched[0]["expected_dataset"] for job in matched)

    class FitCalled(Exception):
        pass

    def verify_fit(model, observations, config, *, initial):
        assert initial is None
        assert all(row.camera_initialization is None for row in observations)
        assert all((row.known_camera is not None) == (config.camera_mode == "known") for row in observations)
        raise FitCalled

    monkeypatch.setattr(fitting_study, "fit_identity", verify_fit)
    for condition in ("A1", "A3", "B3", "B4"):
        with pytest.raises(FitCalled):
            fitting_study._run_job(by_case[42, condition])


def test_heterogeneous_dataset_schedule_is_deterministic_and_identified(study_jobs, tmp_path, official_asset):
    uniform_job = next(job for job in study_jobs if job[STUDY]["condition"] == "A1")
    hetero_job = next(job for job in study_jobs if job[STUDY]["condition"] == "B1")
    uniform, different = (load_dataset(job["dataset"]) for job in (uniform_job, hetero_job))
    subject, = different.subject_ids
    assert different.config.perspective_focal_factors == SCHEDULE
    manifest = read_json(Path(hetero_job["dataset"]) / "manifest.json")
    assert manifest["generator"]["focal_per_short_edge_schedule"] == list(SCHEDULE)
    repeat = generate_dataset(tmp_path / "repeat", different.config, model_file=official_asset)
    assert repeat.dataset_id == different.dataset_id != uniform.dataset_id
    a, b = (dataset.ground_truth(subject) for dataset in (uniform, different))
    np.testing.assert_array_equal(a.arrays["identity"], b.arrays["identity"])
    np.testing.assert_array_equal(a.arrays["rotations"], b.arrays["rotations"])
    for item, factor in zip(b.observations, SCHEDULE, strict=True):
        assert item["camera"]["focal_length"] == [factor * 512] * 2
    for camera, schedule in (("both", SCHEDULE), ("perspective", SCHEDULE[:-1]),
                             ("perspective", (*SCHEDULE[:-1], 0))):
        with pytest.raises(ValueError):
            resolve_config("C", camera=camera, perspective_focal_factors=schedule)


def test_worker_failure_preserves_unavailable_metrics(tmp_path, official_asset, study_jobs, monkeypatch):
    job = study_jobs[0]
    monkeypatch.setattr(fitting_study, "_corrected_jobs", lambda args: iter([dict(job)]))
    monkeypatch.setattr(fitting_study.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=17, stderr="ValueError: initialization failed\n",
    ))
    output = tmp_path / "failed"
    assert fitting_study.main([
        "--protocol", "corrected-identity", "--identity-studies", STUDY, "--model-file", official_asset,
        "--focal-study-budget", "100", "--spectra", "--output", str(output),
    ]) == 1
    summary = read_json(output / "summary.json")
    row, = summary["cases"]
    assert row["status"] == "worker_failed" and "initialization failed" in row["reason"]
    assert row["final_focal_factors"] is None and row["reprojection_metrics"] is None
    assert row["accepted_updates"] is None and row["observability"]["final"] is None
    assert row["report_case"] == "report.json#/cases/0"


@pytest.mark.parametrize("extras", (
    [], ["--focal-study-budget", "0"], ["--focal-study-budget", "101"],
    ["--focal-study-budget", "100", "--lm-iterations", "0", "0", "100"],
    ["--focal-study-budget", "100", "--camera-study-budgets", "60", "100"],
    ["--focal-study-budget", "100", "--dtype", "float32"],
    ["--focal-study-budget", "100", "--device", "cuda"],
    ["--focal-study-budget", "100", "--seeds", "42", "43"],
    ["--focal-study-budget", "100", "--identity-studies", STUDY, "exact"],
))
def test_focal_cli_rejects_incompatible_study_options(tmp_path, official_asset, extras):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        fitting_study.main([
            "--protocol", "corrected-identity", "--identity-studies", STUDY, "--model-file", official_asset,
            "--output", str(output), *extras,
        ])
    assert error.value.code == 2 and not output.exists()
