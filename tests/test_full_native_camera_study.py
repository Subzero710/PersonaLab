"""Full-native identity and camera diagnostics for the corrected study runner."""

from dataclasses import asdict, replace
from math import sqrt
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from persona_lab.benchmark import fitting_study
from persona_lab.benchmark.configuration import resolve_config
from persona_lab.benchmark.dataset import BenchmarkDataset, camera_to_record, load_dataset
from persona_lab.benchmark.evaluate import _camera_errors
from persona_lab.benchmark.fitting_study import _score_observations
from persona_lab.benchmark.full_native_camera import (
    HISTORICAL_SOURCE_ID, NUISANCE_GROUPS, STUDY, _camera_details, add_observation_details,
    case_matrix, classify_compatibility, coefficient_floor, fitted_camera_prediction, summarize,
)
from persona_lab.benchmark.generate import generate_dataset
from persona_lab.benchmark.metrics import coefficient_errors, reprojection_errors
from persona_lab.benchmark.model import _cached_asset, load_benchmark_model, project_dfl
from persona_lab.benchmark.storage import read_json
from persona_lab.camera import PerspectiveCamera
from persona_lab.coordinates import ImageSize
from persona_lab.fitting import FitConfig, Stage, fit_identity
from persona_lab.fitting.configuration import DEFAULT_EXPRESSION


@pytest.fixture(scope="module")
def official_asset():
    return os.environ.get("PERSONALAB_GNM_MODEL_FILE") or str(_cached_asset())


@pytest.fixture(scope="module")
def v5_cli(official_asset, tmp_path_factory):
    """Exercise CLI dispatch and dataset generation with an explicit worker exit."""
    output = tmp_path_factory.mktemp("v5") / "output"
    jobs = {}

    def worker(argv, *, check):
        assert not check
        job = read_json(Path(argv[-1]))
        assert job["case"] not in jobs
        jobs[job["case"]] = job
        return SimpleNamespace(returncode=17)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fitting_study.subprocess, "run", worker)
        code = fitting_study.main([
            "--protocol", "corrected-identity", "--identity-studies", STUDY,
            "--model-file", official_asset, "--device", "cpu", "--dtype", "float64",
            "--seeds", "42", "43", "44", "--lm-iterations", "0", "0", "30", "--spectra",
            "--output", str(output),
        ])
    assert code == 1
    return jobs, read_json(output / "report.json"), read_json(output / "summary.json")


def test_case_matrix():
    cases = list(case_matrix((42, 43, 44)))
    assert len(cases) == 25
    for seed in (42, 43, 44):
        for camera in ("perspective", "weak-perspective"):
            assert {
                (item["condition"], item["identity_ground_truth"], item["camera_mode"])
                for item in cases if item["seed"] == seed and item["camera_model"] == camera
            } == {
                ("A", "full-native", "known"), ("B", "full-native", "estimated"),
                ("C", "head-only", "estimated"), ("D", "head-only", "known"),
            }
    assert cases[-1] == {
        "condition": "historical", "seed": 42, "camera_model": "both",
        "identity_ground_truth": "full-native", "camera_mode": "known",
    }


def test_cli_jobs_reuse_matched_datasets_and_preserve_nuisance(v5_cli):
    jobs, report, _ = v5_cli
    assert report["requested_identity_studies"] == [STUDY]
    assert len(jobs) == 25
    assert [job[STUDY] for job in jobs.values()] == list(case_matrix((42, 43, 44)))
    for job in jobs.values():
        assert job["identity_studies"] == [STUDY] and job["observations"] == 9
        assert job["spectra"] and job["final_spectra"]
        assert not job["known_pose_diagnostic"] and job["held_out_dataset"] is None
        config = FitConfig.from_dict(job["config"])
        assert config.solver == "levenberg-marquardt" and config.lm == fitting_study.STRICT_LM
        assert config.train_identity == ("head",) and config.train_expression == DEFAULT_EXPRESSION
        assert config.identity_weight == config.expression_weight == 0
        assert tuple(stage.iterations for stage in config.stages) == (0, 0, 30)
        assert config.camera_mode == job[STUDY]["camera_mode"]
        assert config.camera_model == (job[STUDY]["camera_model"] if config.camera_mode == "estimated" else None)

    for seed in (42, 43, 44):
        for camera in ("perspective", "weak-perspective"):
            a, b, c, d = (jobs[f"{STUDY}-{condition}-{seed}-{camera}"] for condition in "ABCD")
            assert a["dataset"] == b["dataset"] and c["dataset"] == d["dataset"]
            full, head = (load_dataset(job["dataset"]) for job in (a, d))
            assert full.source_dataset_id != head.source_dataset_id
            assert full.config.camera == head.config.camera == camera
            assert full.config.identity_selection == "full" and head.config.identity_selection == "head"
            subject, = full.subject_ids
            assert head.subject_ids == full.subject_ids
            assert full.observation_ids(subject) == head.observation_ids(subject)
            full_truth, head_truth = full.ground_truth(subject), head.ground_truth(subject)
            assert full_truth.observations == head_truth.observations
            for field in ("expression", "rotations", "translation"):
                np.testing.assert_array_equal(full_truth.arrays[field], head_truth.arrays[field])
            names = full.model_metadata["identity_names"]
            columns = np.array([name.startswith("head_") for name in names])
            assert columns.sum() == 170
            np.testing.assert_array_equal(full_truth.arrays["identity"][columns], head_truth.arrays["identity"][columns])
            np.testing.assert_array_equal(head_truth.arrays["identity"][~columns], 0)
            assert np.count_nonzero(full_truth.arrays["identity"][~columns]) == 83

    historical = load_dataset(jobs[f"{STUDY}-historical"]["dataset"])
    assert historical.source_dataset_id == HISTORICAL_SOURCE_ID
    assert "identity_selection" not in historical.config.metadata()
    assert historical.config.camera == "both"
    assert len({job["dataset"] for job in jobs.values()}) == 13


def test_cli_worker_failures_are_retained_in_both_reports(v5_cli):
    _, report, summary = v5_cli
    assert len(report["cases"]) == len(summary["cases"]) == 25
    for index, (detail, row) in enumerate(zip(report["cases"], summary["cases"], strict=True)):
        assert detail["return_code"] == 17
        assert row["report_case"] == f"report.json#/cases/{index}"
        assert row["status"] == detail["status"] == "worker_failed"
        assert row["execution_status"] == "failed" and row["converged"] is None
        assert row["reason"] == detail["reason"] == "Fitting worker exited with status 17"
        assert row["representability"] == "unavailable"
        assert all(item["status"] == "unavailable" for item in row["observability_status"].values())
        assert all(value is None for value in row["metrics"].values())
    assert summary["historical_comparison"]["matches_source_dataset_id"] is None


def test_existing_protocol_defaults_are_preserved(monkeypatch, tmp_path, official_asset):
    captured = {}

    def capture(args):
        captured[args.protocol] = args
        return ()

    monkeypatch.setattr(fitting_study, "_corrected_jobs", capture)
    monkeypatch.setattr(fitting_study, "_jobs", capture)
    for protocol in ("corrected-identity", "historical"):
        assert fitting_study.main([
            "--protocol", protocol, "--model-file", official_asset, "--output", str(tmp_path / protocol),
        ]) == 0
    corrected, historical = captured["corrected-identity"], captured["historical"]
    assert corrected.identity_studies == ("exact", "convergence", "count", "held-out", "robustness")
    assert corrected.lm_iterations == (0, 0, 30)
    assert historical.identity_studies is None
    assert historical.experiments == tuple("ABCDEFGH") and historical.lm_iterations == (10, 12, 30)


def test_camera_study_rejects_an_excessive_joint_budget(tmp_path, official_asset, capsys):
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        fitting_study.main([
            "--protocol", "corrected-identity", "--identity-studies", STUDY, "--model-file", official_asset,
            "--lm-iterations", "0", "0", "31", "--output", str(output),
        ])
    assert error.value.code == 2 and not output.exists()
    assert "at most 30 joint LM iterations" in capsys.readouterr().err


def test_v5_worker_scores_blind_fit_and_handles_nonconvergence(v5_cli, monkeypatch):
    jobs, _, _ = v5_cli
    job = jobs[f"{STUDY}-B-42-perspective"]
    config = FitConfig.from_dict(job["config"])
    # One update exercises the reported budget exhaustion and actual final snapshot.
    config = replace(config, stages=(*config.stages[:2], replace(config.stages[2], iterations=1)))
    job = job | {"config": asdict(config)}
    read_truth = BenchmarkDataset.ground_truth
    fits = []

    def truth_after_fitting(dataset, subject):
        assert fits, "ground_truth must be reserved for evaluation"
        return read_truth(dataset, subject)

    def blind_fit(model, observations, fit_config, *, initial):
        assert initial is None and fit_config.camera_model == "perspective"
        assert all(item.known_camera is None and item.camera_initialization is None for item in observations)
        result = fit_identity(model, observations, fit_config)
        fits.append(result)
        return result

    monkeypatch.setattr(BenchmarkDataset, "ground_truth", truth_after_fitting)
    monkeypatch.setattr(fitting_study, "fit_identity", blind_fit)
    result = fitting_study._run_job(job)
    assert result["status"] == "budget_exhausted", result["reason"]
    assert not result["converged"] and result["execution_status"] == "completed"
    fit, = fits
    np.testing.assert_array_equal(fit.initial.identity, 0)
    np.testing.assert_array_equal(fit.initial.expression, 0)
    np.testing.assert_array_equal(fit.initial.translation, 0)
    np.testing.assert_array_equal(fit.final.identity[170:], fit.initial.identity[170:])
    assert all(camera.focal_length.tolist() == [768.0, 768.0] for camera in fit.initial.cameras)
    assert not result["compatibility"]["fully_representable"]
    assert result["representability_status"] == "expected_identity_incompatibility"
    assert result["identity_error_floor"]["incompatible_fixed_count"] == 83
    assert result["identity_error_floor"]["decomposition"]["valid"]
    assert result["final"]["fixed_identity_preserved"]
    gradient = result["optimization"]["final_gradient"]
    assert gradient["status"] == "evaluated"
    assert gradient["camera_offset_origin"] == "final_camera"
    assert np.isfinite(gradient["scaled_projected_gradient_norm"])
    assert set(gradient["projected_gradient_norms"]) == {"identity", "expression", "pose", "camera"}
    assert result["final"]["neutral_geometry"]["vertex_count"] == 17821
    assert result["camera_comparison"]["non_comparable_translation_observations"] == 9
    assert result["final"]["training_reprojection_and_nuisance_recovery"]["expression"]["trainable"]
    for at in ("initial", "final"):
        spectrum = result[f"{at}_identity_observability"]
        assert spectrum["state"] == at and spectrum["identity_prior_jtj_diagonal"] == 0
        assert spectrum["absolute_threshold"] > 0
        assert spectrum["observation_ids"] == list(result["observation_ids"])
        for name in NUISANCE_GROUPS:
            group = spectrum[name]
            assert group["rank"] + group["nullity"] == 170
            assert len(group["column_l2"]) == len(group["singular_values"]) == 170
            assert (group["condition_number"] is None) == bool(group["nullity"])

    def failed_fit(*args, **kwargs):
        return replace(fit, final=None, status="numerical_failure", reason="injected numerical failure")

    monkeypatch.setattr(fitting_study, "fit_identity", failed_fit)
    failed = fitting_study._run_job(job)
    assert failed["execution_status"] == "failed" and failed["reason"] == "injected numerical failure"
    assert failed["final"] is None and failed["final_identity_observability"] is None
    assert failed["optimization"]["final_gradient"] == {
        "status": "unavailable", "reason": "injected numerical failure",
    }
    assert failed["observability_status"]["final"]["status"] == "unavailable"
    assert failed["observability_status"]["initial"]["status"] == "evaluated"
    assert failed["identity_error_floor"]["full_native_rmse_floor"] > 0
    assert failed["identity_error_floor"]["full_native_rmse"] is None


def test_coefficient_rmse_floor_and_decomposition():
    target = np.r_[np.full(170, 0.3), np.full(83, 0.25)]
    initial = np.zeros(253)
    final = np.r_[target[:170] + 1e-10, initial[170:]]

    def metrics(values):
        return {
            "global": coefficient_errors(values, target),
            "trainable": coefficient_errors(values[:170], target[:170]),
            "fixed": coefficient_errors(values[170:], target[170:]),
        }

    names = [f"eyes_{index:03}" for index in range(3)] + [f"teeth_{index:03}" for index in range(80)]
    result = coefficient_floor(metrics(initial), metrics(final), names)
    assert result["incompatible_fixed_coefficients"] == names
    assert result["incompatible_fixed_count"] == result["fixed_count"] == 83
    assert result["fixed_rmse"] == 0.25
    assert result["full_native_rmse_floor"] == pytest.approx(0.25 * sqrt(83 / 253))
    assert result["excess_squared_coefficient_error"] == pytest.approx((170 / 253) * 1e-20, abs=1e-25)
    assert result["decomposition"]["valid"]
    assert result["full_native_rmse"] == metrics(final)["global"]["rmse"]
    unavailable = coefficient_floor(metrics(initial), None, names)
    assert unavailable["full_native_rmse_floor"] == result["full_native_rmse_floor"]
    assert unavailable["trainable_rmse"] is None and unavailable["decomposition"] is None


def test_historical_generation_uses_the_original_source_identifier(tmp_path):
    config = resolve_config("C", seed=42, observations_per_subject=9, camera="both")
    dataset = generate_dataset(
        tmp_path / "historical", config, model_file=os.environ.get("PERSONALAB_GNM_MODEL_FILE"),
    )
    subject, = dataset.subject_ids
    truth = dataset.ground_truth(subject)
    assert dataset.source_dataset_id == HISTORICAL_SOURCE_ID
    assert dataset.observation_ids(subject) == tuple(item["id"] for item in truth.observations)
    assert [item["camera"]["type"] for item in truth.observations] == (
        ["perspective", "weak-perspective"] * 4 + ["perspective"]
    )
    assert np.count_nonzero(truth.arrays["identity"]) == 253


def test_expected_identity_and_gauge_mismatches_do_not_mask_unexpected_nuisance():
    case = list(case_matrix((42,)))[1]
    observation = {
        "id": "view", "differing_fixed_joints": [], "differing_fixed_root": False,
        "differing_fixed_camera_fields": ["focal_length"],
        "expression": {"differing_coefficients": []}, "expression_outside_bounds": [],
    }
    compatibility = {
        "fully_representable": False, "identity": {"differing_coefficients": ["eyes_000"]},
        "observations": [observation],
    }
    classified = classify_compatibility(compatibility, case, ("eyes_000",))
    assert classified["status"] == "expected_identity_incompatibility"
    assert classified["fixed_focal_gauge_observations"] == ["view"]
    assert classified["expected_incompatible_identity_coefficients"] == ["eyes_000"]
    assert compatibility["fully_representable"] is False

    for updates in (
        {"expression": {"differing_coefficients": ["fixed_expression"]}},
        {"differing_fixed_joints": ["neck"]},
        {"differing_fixed_root": True},
        {"differing_fixed_camera_fields": ["focal_length", "principal_point"]},
    ):
        changed = compatibility | {"observations": [observation | updates]}
        classified = classify_compatibility(changed, case, ("eyes_000",))
        assert classified["status"] == "unexpected_incompatibility"
        assert classified["unexpected_incompatible_observations"] == ["view"]
    head_only = case | {"identity_ground_truth": "head-only"}
    assert classify_compatibility(compatibility, head_only, ("eyes_000",))["status"] == "unexpected_incompatibility"


def test_incompatible_perspective_gauge_reports_null_translation():
    def tensor(values):
        return torch.tensor(values, dtype=torch.float64)

    truth = PerspectiveCamera(
        ImageSize(512, 512), tensor([0, 0, 0]), tensor([0, 0.265, 0.7]),
        tensor([716.8, 716.8]), tensor([255.5, 255.5]),
    )
    fit = replace(truth, focal_length=tensor([768.0, 768.0]), translation=tensor([0, 0.265, 0.75]))
    score = _camera_errors(fit, truth, shared_root=True)
    gauge = {
        "camera_rotation": "fixed", "perspective": {
            "focal_length": "fixed", "principal_point": "fixed",
            "translation": "estimated_metric_xy_and_positive_log_z",
        },
    }
    assert _camera_details(score, camera_to_record(fit), camera_to_record(fit),
                           camera_to_record(truth), gauge, "estimated")
    assert score["trainable_fields"] == ["translation"]
    assert set(score["fixed_fields"]) == {"focal_length", "principal_point", "rotation", "image_size"}
    assert score["translation_l2_m"] is None
    assert score["translation_comparability"]["mismatched_fields"] == ["focal_length"]
    assert "focal_length" in score["translation_comparability"]["reason"]
    assert score["focal_length_abs_px"] == pytest.approx([51.2, 51.2])
    assert score["principal_point_comparability"]["status"] == "comparable"


@pytest.mark.parametrize("camera_model", ("perspective", "weak-perspective"))
def test_camera_details_use_blind_initialization_and_fitted_projection(camera_model, tmp_path):
    asset = os.environ.get("PERSONALAB_GNM_MODEL_FILE")
    parent = generate_dataset(tmp_path / "blind", resolve_config("C", camera=camera_model), model_file=asset)
    subject, = parent.subject_ids
    dataset = parent.select_observations(subject, parent.observation_ids(subject)[:1])
    observations = dataset.fitting_observations(subject, include_known_cameras=False)
    assert observations[0].known_camera is None and observations[0].camera_initialization is None
    model = load_benchmark_model(asset)
    config = FitConfig(
        solver="levenberg-marquardt", camera_mode="estimated", camera_model=camera_model,
        stages=(Stage("camera", 1, ("camera",), 0.004),),
    )
    fit = fit_identity(model, observations, config)
    assert fit.final is not None, fit.reason
    np.testing.assert_array_equal(fit.initial.identity, 0)
    np.testing.assert_array_equal(fit.initial.expression, 0)
    np.testing.assert_array_equal(fit.initial.translation, 0)

    truth = dataset.ground_truth(subject)
    prediction = fitted_camera_prediction(fit)
    training = _score_observations(
        model, dataset, truth, prediction, "complete", trainable_expression=fit.selection.expression,
    )
    overview = add_observation_details(training, truth, fit)
    scored, = training["observations"]
    camera = scored["camera_recovery"]
    assert camera["fitted_parameters"] == camera_to_record(fit.final.cameras[0])
    if camera_model == "perspective":
        assert camera["initial_parameters"]["focal_length"] == [768.0, 768.0]
        assert camera["translation_l2_m"] is None and overview["non_comparable_translation_observations"] == 1
    else:
        assert camera["trainable_fields"] == ["translation", "scale"]
        assert camera["principal_point_abs_px"] is None
        assert camera["scale_abs_px_per_m"] >= 0
        assert "absolute metric scale ambiguous" in overview["metric_scale_interpretation"]
    assert scored["pose_recovery"]["root_translation_comparability"]["status"] == "fixed_shared"
    assert scored["expression_recovery"]["trainable"]["coefficient_count"] == 24
    points = model.landmarks(fit.final.identity, fit.final.expression[0],
                             fit.final.rotations[0], fit.final.translation[0])
    projected = project_dfl(points, fit.final.cameras[0], dataset.correspondences).cpu().numpy()
    expected = reprojection_errors(projected, truth.arrays["clean"][0], truth.arrays["validity"][0])
    assert scored["reprojection"]["clean"] == expected


@pytest.mark.parametrize("historical_id", (HISTORICAL_SOURCE_ID, "different"))
def test_summary_preserves_failures_and_matches_detailed_cases(historical_id):
    cases = [{"case": "older-protocol-case"}]
    for item in case_matrix((42, 43, 44)):
        condition = item["condition"]
        failed = condition == "A"
        fit = {
            "case": f"{condition}-{item['seed']}-{item['camera_model']}", STUDY: item,
            "status": "numerical_failure" if failed else "budget_exhausted",
            "reason": "explicit test reason", "execution_status": "failed" if failed else "completed",
            "converged": False, "representability_status": "expected_identity_incompatibility",
            "source_dataset_id": historical_id if condition == "historical" else "matched",
            "evaluation_dataset_id": item["identity_ground_truth"],
            "initial_identity_observability": {
                name: {"rank": 168, "nullity": 2, "condition_number": None,
                       "effective_condition_number": 10.0, "singular_values": list(range(170))}
                for name in NUISANCE_GROUPS
            },
            "final": None if failed else {
                "identity": {kind: {"rmse": 0.1} for kind in ("global", "trainable", "fixed")},
                "neutral_geometry": {"rms_m": 0.0002},
            },
        }
        cases.append(fit)
    summary = summarize(cases)
    assert len(summary["cases"]) == 25
    for row, case in zip(summary["cases"], cases[1:], strict=True):
        index = int(row["report_case"].rsplit("/", 1)[-1])
        assert cases[index] is case
        assert (row["case"], row["status"], row["reason"]) == (case["case"], case["status"], case["reason"])
        assert row["observability"]["initial"]["after_all_nuisance"]["condition_number"] is None
        assert "singular_values" not in row["observability"]["initial"]["after_all_nuisance"]
        assert row["metrics"]["trainable_identity_rmse"] == (None if case["final"] is None else 0.1)
    assert len(summary["comparisons"]) == 4
    assert all(len(comparison["pairs"]) == 6 for comparison in summary["comparisons"])
    assert all(pair["same_evaluation_dataset"] for pair in summary["comparisons"][0]["pairs"])
    assert summary["comparisons"][0]["aggregate"]["metric_deltas"]["trainable_identity_rmse"]["n"] == 0
    historical = summary["historical_comparison"]
    assert historical["matches_source_dataset_id"] == (historical_id == HISTORICAL_SOURCE_ID)
    assert (historical["v5_minus_v2"]["trainable_identity_rmse"] is None) == (historical_id != HISTORICAL_SOURCE_ID)
