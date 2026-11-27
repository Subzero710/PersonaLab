"""Case construction, cold starts and reporting for the V5.1 diagnostic."""

from copy import deepcopy
from dataclasses import asdict, replace
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from persona_lab.benchmark import camera_calibration, fitting_study
from persona_lab.benchmark.dataset import BenchmarkDataset, load_dataset
from persona_lab.benchmark.model import _cached_asset
from persona_lab.benchmark.storage import read_json
from persona_lab.fitting import FitConfig, fit_identity
from persona_lab.fitting.configuration import DEFAULT_EXPRESSION


STUDY = camera_calibration.STUDY
EXPECTED = {
    "A1": ("weak-perspective", "not-applicable", None, 60),
    "A2": ("weak-perspective", "not-applicable", None, 100),
    "B1": ("perspective", "calibrated-intrinsics", 1.4, 60),
    "B2": ("perspective", "calibrated-intrinsics", 1.4, 100),
    "C1": ("perspective", "incorrect-fixed-intrinsics", 1.5, 60),
    "C2": ("perspective", "incorrect-fixed-intrinsics", 1.5, 100),
}


@pytest.fixture(scope="module")
def official_asset():
    return os.environ.get("PERSONALAB_GNM_MODEL_FILE") or str(_cached_asset())


@pytest.fixture(scope="module")
def calibration_cli(official_asset, tmp_path_factory):
    output = tmp_path_factory.mktemp("v5_1") / "outputs"
    jobs = {}
    run = fitting_study.subprocess.run

    def failed_worker(argv, **options):
        if len(argv) < 2 or argv[-2] != "--worker":
            return run(argv, **options)
        assert options == {"check": False, "text": True, "stderr": fitting_study.subprocess.PIPE}
        job = read_json(Path(argv[-1]))
        assert job["case"] not in jobs
        jobs[job["case"]] = job
        return SimpleNamespace(returncode=17, stderr="Traceback\nValueError: explicit worker failure\n")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fitting_study.subprocess, "run", failed_worker)
        result = fitting_study.main([
            "--protocol", "corrected-identity", "--identity-studies", STUDY, "--model-file", official_asset,
            "--device", "cpu", "--dtype", "float64", "--seeds", "42", "43", "44",
            "--camera-study-budgets", "60", "100", "--spectra", "--output", str(output),
        ])
    assert result == 1
    return jobs, read_json(output / "report.json"), read_json(output / "summary.json")


def test_cli_builds_exact_matrix_and_solver_configuration(calibration_cli):
    jobs, report, _ = calibration_cli
    assert report["requested_identity_studies"] == [STUDY]
    assert report["camera_study_budgets"] == [60, 100] and report["lm_iterations"] is None
    assert len(jobs) == 18
    fit_default = FitConfig()
    for seed in (42, 43, 44):
        for condition, (camera, calibration, factor, budget) in EXPECTED.items():
            job = jobs[f"{STUDY}-{condition}-{seed}"]
            assert job[STUDY] == {
                "condition": condition, "seed": seed, "camera_model": camera, "camera_mode": "estimated",
                "calibration_condition": calibration, "focal_factor": factor, "joint_lm_budget": budget,
                "identity_ground_truth": "head-only",
            }
            assert job["observations"] == 9 and job["device"] == "cpu" and job["dtype"] == "float64"
            assert job["spectra"] and job["final_spectra"]
            assert not job["known_pose_diagnostic"] and job["held_out_dataset"] is None
            config = FitConfig.from_dict(job["config"])
            assert (config.camera_mode, config.camera_model, config.solver) == (
                "estimated", camera, "levenberg-marquardt",
            )
            assert config.perspective_focal_per_short_edge == (1.5 if factor is None else factor)
            assert tuple(stage.iterations for stage in config.stages) == (0, 0, budget)
            assert config.stages[-1].groups == fit_default.stages[-1].groups
            assert config.train_identity == ("head",) and config.train_expression == DEFAULT_EXPRESSION
            assert config.pose_initializer == fit_default.pose_initializer and not config.train_root
            assert config.expression_limit == fit_default.expression_limit
            assert config.identity_weight == config.expression_weight == config.pose_weight == config.camera_weight == 0
            assert config.lm == fitting_study.STRICT_LM
            assert config.lm.scaled_gradient_tolerance == config.lm.minimum_damping == 1e-12
            assert config.lm.native_step_tolerance == 1e-6 and config.lm.scaled_step_tolerance == 1e-10


def test_same_generated_datasets_and_order_as_v5_condition_c(calibration_cli, official_asset, tmp_path, monkeypatch):
    jobs, _, _ = calibration_cli
    original_generate = fitting_study.generate_dataset

    def generate_head_only_sources(path, config, *, model_file):
        if config.identity_selection == "head":
            return original_generate(path, config, model_file=model_file)
        return None  # These V5 sources are outside condition C and will not be read by the test.

    monkeypatch.setattr(fitting_study, "generate_dataset", generate_head_only_sources)
    v5 = fitting_study.CAMERA_STUDY
    v5_jobs = {
        (job["seed"], job[v5]["camera_model"]): job for job in fitting_study._corrected_jobs(SimpleNamespace(
            identity_studies=(v5,), seeds=(42, 43, 44), lm_iterations=(0, 0, 30),
            output=tmp_path, model_file=official_asset, dtype="float64", device="cpu", spectra=True,
        )) if job[v5]["condition"] == "C"
    }
    assert len({job["dataset"] for job in jobs.values()}) == 6
    for (seed, camera), v5_job in v5_jobs.items():
        cases = [job for job in jobs.values() if job["seed"] == seed and job[STUDY]["camera_model"] == camera]
        assert len(cases) == (4 if camera == "perspective" else 2)
        assert len({job["dataset"] for job in cases}) == 1
        baseline = load_dataset(v5_job["dataset"])
        selected = load_dataset(cases[0]["dataset"])
        subject, = baseline.subject_ids
        assert selected.config == baseline.config
        assert selected.source_dataset_id == baseline.source_dataset_id == baseline.dataset_id
        assert selected.observation_ids(subject) == baseline.observation_ids(subject)
        assert len(selected.observation_ids(subject)) == 9
        for job in cases:
            assert job["expected_dataset"] == {
                "source_dataset_id": baseline.dataset_id, "evaluation_dataset_id": baseline.dataset_id,
                "observation_ids": list(baseline.observation_ids(subject)),
            }


@pytest.mark.parametrize(("options", "message"), (
    (("--identity-studies", STUDY), "requires --camera-study-budgets 60 100"),
    (("--identity-studies", STUDY, "--camera-study-budgets", "60", "120"), "requires --camera-study-budgets 60 100"),
    (("--identity-studies", STUDY, "--camera-study-budgets", "60", "100", "--lm-iterations", "0", "0", "60"),
     "joint budgets from --camera-study-budgets exclusively"),
    (("--identity-studies", STUDY, "--camera-study-budgets", "60", "100", "--device", "cuda"), "requires CPU"),
    (("--identity-studies", STUDY, "--camera-study-budgets", "60", "100", "--seeds", "42", "43"), "seeds 42 43 44"),
    (("--identity-studies", STUDY, "--camera-study-budgets", "60", "100", "--dtype", "float32"), "requires float64"),
    (("--identity-studies", STUDY, "full-native-camera", "--camera-study-budgets", "60", "100"), "only identity study"),
    (("--identity-studies", "full-native-camera", "--lm-iterations", "0", "0", "60"), "at most 30"),
    (("--camera-study-budgets", "60", "100"), "requires the corrected-identity camera-calibration study"),
    (("--protocol", "historical", "--camera-study-budgets", "60", "100"), "requires the corrected-identity"),
))
def test_rejects_unsupported_configurations(options, message, tmp_path, capsys):
    asset = tmp_path / "parser-only.npz"
    asset.touch()
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as failure:
        fitting_study.main([
            "--protocol", "corrected-identity", "--model-file", str(asset), "--output", str(output), *options,
        ])
    assert failure.value.code == 2 and message in capsys.readouterr().err
    assert not output.exists()


def test_calibration_is_excluded_from_default_studies(monkeypatch, tmp_path):
    asset = tmp_path / "parser-only.npz"
    asset.touch()

    def existing_selection(args):
        assert args.identity_studies == ("exact", "convergence", "count", "held-out", "robustness")
        assert args.lm_iterations == (0, 0, 30)
        return iter(())

    monkeypatch.setattr(fitting_study, "_corrected_jobs", existing_selection)
    assert fitting_study.main([
        "--protocol", "corrected-identity", "--model-file", str(asset), "--output", str(tmp_path / "default"),
    ]) == 0


def _short_job(job, updates):
    config = FitConfig.from_dict(job["config"])
    config = replace(config, stages=(*config.stages[:-1], replace(config.stages[-1], iterations=updates)))
    return job | {"config": asdict(config), STUDY: job[STUDY] | {"joint_lm_budget": updates}}


@pytest.mark.parametrize("family", ("A", "B", "C"))
def test_official_cold_starts_are_blind_and_keep_the_budget_prefix(calibration_cli, monkeypatch, family):
    jobs, _, _ = calibration_cli
    read_truth = BenchmarkDataset.ground_truth
    fits = []
    fitting_done = False

    def truth_after_fitting(dataset, subject):
        assert fitting_done
        return read_truth(dataset, subject)

    def blind_fit(model, observations, config, *, initial):
        nonlocal fitting_done
        assert not fitting_done and initial is None
        assert len(observations) == 9 and config.camera_mode == "estimated"
        assert all(item.known_camera is None and item.camera_initialization is None for item in observations)
        fit = fit_identity(model, observations, config, initial=initial)
        np.testing.assert_array_equal(fit.initial.identity, 0)
        np.testing.assert_array_equal(fit.initial.expression, 0)
        np.testing.assert_array_equal(fit.initial.translation, 0)
        assert len(fit.selection.identity) == 170 and len(fit.selection.fixed_identity) == 83
        assert len(fit.selection.expression) == 24
        fits.append(fit)
        fitting_done = True
        return fit

    monkeypatch.setattr(fitting_study, "fit_identity", blind_fit)
    monkeypatch.setattr(BenchmarkDataset, "ground_truth", truth_after_fitting)
    records = []
    for updates in (1, 2):
        fitting_done = False
        record = fitting_study._run_job(_short_job(jobs[f"{STUDY}-{family}{updates}-42"], updates))
        records.append(record)
        assert record["status"] == "budget_exhausted", record["reason"]
        assert record["execution_status"] == "completed" and record["converged"] is False
        assert record["optimization"]["accepted_updates"] == updates
        assert record["dataset_verified"] and record["focal_calibration_check"]["matches_expected"]
        assert record["initial"]["identity"]["fixed"]["coefficient_count"] == 83
        assert record["final"]["neutral_geometry"]["vertex_count"] == 17821
        assert record["final"]["fixed_identity_preserved"]
        for at in ("initial", "final"):
            spectrum = record[f"{at}_identity_observability"]
            assert spectrum["identity_prior_jtj_diagonal"] == 0
            for name in camera_calibration.NUISANCE_GROUPS:
                group = spectrum[name]
                assert len(group["singular_values"]) == len(group["column_l2"]) == 170
                assert group["minimum_singular_value"] == min(group["singular_values"])
                assert group["maximum_singular_value"] == max(group["singular_values"])
                if group["nullity"]:
                    assert group["condition_number"] is None
        assert len(record["camera_parameter_changes"]) == 9
        if family == "C":
            assert record["camera_comparison"]["non_comparable_translation_observations"] == 9
        else:
            assert record["camera_comparison"]["comparable_translation_observations"] == 9

    first, second = records
    assert fits[0].initial is not fits[1].initial
    assert first["parameter_snapshots"]["initial"] == second["parameter_snapshots"]["initial"]
    assert first["parameter_snapshots"]["final"] != second["parameter_snapshots"]["initial"]
    summary = camera_calibration.summarize(records)
    pair, = [pair for comparison in summary["comparisons"] for pair in comparison["pairs"]]
    assert pair["status"] == "matched" and all(pair["dataset_checks"].values())
    assert pair["optimization_prefix"]["status"] == "matched"
    for index, (record, row) in enumerate(zip(records, summary["cases"], strict=True)):
        assert row["report_case"] == f"report.json#/cases/{index}"
        assert row["identity_metrics"] == record["final"]["identity"]
        assert row["reprojection_metrics"]["clean"] == row["reprojection_metrics"]["observed"]
        assert len(row["camera_recovery"]) == 9 and row["runtime"]["case_peak_process_rss_bytes"] > 0
        assert row["reference_targets"] == record["reference_targets"]

    mismatched = second | {"observation_ids": second["observation_ids"][::-1]}
    summary = camera_calibration.summarize([first, mismatched])
    pair, = [pair for comparison in summary["comparisons"] for pair in comparison["pairs"]]
    assert pair["status"] == "dataset_mismatch" and pair["dataset_checks"]["observation_ids"] is False
    assert pair["optimization_prefix"] is None and all(value is None for value in pair["metric_deltas"].values())


def test_worker_failures_have_explicit_unavailable_data(calibration_cli):
    jobs, report, summary = calibration_cli
    assert len(report["cases"]) == len(summary["cases"]) == 18
    for index, (detail, row) in enumerate(zip(report["cases"], summary["cases"], strict=True)):
        assert row["report_case"] == f"report.json#/cases/{index}"
        assert row["case"] == detail["case"] and row["status"] == detail["status"] == "worker_failed"
        assert row["reason"].endswith("ValueError: explicit worker failure") and detail["return_code"] == 17
        assert row["execution_status"] == "failed" and row["converged"] is None
        assert row["source_dataset_id"] == jobs[row["case"]]["expected_dataset"]["source_dataset_id"]
        assert not row["dataset_verified"] and row["accepted_updates"] is None
        assert row["identity_metrics"] is row["geometry_metrics"] is row["camera_recovery"] is None
        assert row["observability_status"]["final"]["status"] == "unavailable"
    assert len(summary["comparisons"]) == 5
    for comparison in summary["comparisons"]:
        assert len(comparison["pairs"]) == 3
        assert all(pair["status"] == "unavailable" for pair in comparison["pairs"])


def test_solver_failure_keeps_actual_reason_and_available_diagnostics(calibration_cli, monkeypatch):
    jobs, _, _ = calibration_cli

    def failure(model, observations, config, *, initial):
        fit = fit_identity(model, observations, config, initial=initial)
        return replace(fit, final=None, final_reprojection=None, status="no_acceptable_step", reason="test stagnation")

    monkeypatch.setattr(fitting_study, "fit_identity", failure)
    record = fitting_study._run_job(_short_job(jobs[f"{STUDY}-C1-42"], 1))
    assert record["status"] == "no_acceptable_step" and record["reason"] == "test stagnation"
    assert record["execution_status"] == "failed" and record["converged"] is False
    assert record["final"] is record["camera_parameter_changes"] is record["parameter_snapshots"]["final"] is None
    assert record["optimization"]["accepted_updates"] == 1 and record["optimization"]["trials"]
    assert record["initial_identity_observability"] is not None
    assert record["observability_status"]["final"] == {"status": "unavailable", "reason": "test stagnation"}
    assert all(item["met"] is None for item in record["reference_targets"].values())


def test_source_mismatch_stops_before_fitting(calibration_cli, monkeypatch):
    jobs, _, _ = calibration_cli
    job = deepcopy(jobs[f"{STUDY}-B1-42"])
    job["expected_dataset"]["source_dataset_id"] = "invalid"

    def unexpected_fit(*args, **kwargs):
        pytest.fail("A mismatched dataset reached the fitter")

    monkeypatch.setattr(fitting_study, "fit_identity", unexpected_fit)
    with pytest.raises(ValueError, match="dataset or observation order differs"):
        fitting_study._run_job(job)


def test_checkpoint_states_and_deterministic_prefix_comparison():
    trace = {
        "status": "budget_exhausted", "reason": "joint: iteration budget reached", "final": {"scored": True},
        "optimization": {
            "updates": [{"stage": "joint", "stage_iteration": i + 1} for i in range(60)],
            "trials": [{"stage": "joint", "stage_iteration": i, "damping": 1e-7, "accepted": True} for i in range(60)],
            "linearizations": [{"stage": "joint", "stage_iteration": i} for i in range(60)],
            "initial_objective": {"total": 7.0},
        },
        "history": [{"iteration": 60, "gradient_norms": {}, "losses": {"total": 1e-16}, "elapsed_seconds": 3.0}],
        "parameter_snapshots": {"initial": {"identity": [0.0]}}, "initialization": [],
    }
    checkpoints = {item["accepted_update"]: item for item in camera_calibration._checkpoints(trace)}
    assert list(checkpoints) == [10, 20, 30, 60, 100]
    assert checkpoints[10]["status"] == "reached" and checkpoints[10]["recovery_metrics"]["status"] == "unavailable"
    assert checkpoints[60]["objective_after_update"] == {"total": 1e-16}
    assert checkpoints[60]["recovery_metrics"]["status"] == "available"
    assert checkpoints[100]["status"] == "unavailable"
    test_row = {"iteration": 0, "gradient_norms": {"identity": 0.4}, "elapsed_seconds": 1.0}
    trace["history"].insert(0, test_row)
    other = deepcopy(trace)
    other["history"][0]["elapsed_seconds"] = 900.0
    other["history"][0]["gradient_norms"]["identity"] += 1e-12
    other["history"].pop()  # An extended run has no terminal row at update 60.
    assert camera_calibration.compare_prefixes(trace, other)["status"] == "matched"
    other["optimization"]["trials"][3]["damping"] *= 2
    result = camera_calibration.compare_prefixes(trace, other)
    assert result["status"] == "discrepant" and result["compared"]["trials"]["common"] == 60
    assert result["differences"] == [{"field": "trials/3/damping", "before": 1e-7, "after": 2e-7}]
