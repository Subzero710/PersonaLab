"""V5.3 protocol and stage-boundary checks on the pinned GNM v3.0 Head."""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import os
from types import SimpleNamespace

import numpy as np
import pytest

from persona_lab.benchmark import fitting_study, focal_convergence, focal_estimation
from persona_lab.benchmark.camera_calibration import _snapshot
from persona_lab.benchmark.dataset import BenchmarkDataset, load_dataset
from persona_lab.benchmark.model import _cached_asset, load_benchmark_model
from persona_lab.benchmark.storage import read_json
from persona_lab.fitting import FitConfig, FitInitialization, fit_identity
from persona_lab.fitting import levenberg_marquardt as lm
from persona_lab.fitting import optimizer
from persona_lab.fitting.initialization import initialize
from persona_lab.fitting.observations import prepare_observations
from persona_lab.fitting.observability import identity_observability
from persona_lab.fitting.results import FitState


STUDY = focal_convergence.STUDY
STRATEGIES = {"J100": (0, 0, 100), "J200": (0, 0, 200), "P100": (20, 0, 80)}
CONDITIONS = {"A3", "A4", "A5", "A6", "A7", "A8", "B3"}


@pytest.fixture(scope="module")
def asset():
    return os.environ.get("PERSONALAB_GNM_MODEL_FILE") or str(_cached_asset())


@pytest.fixture(scope="module")
def study_jobs(tmp_path_factory, asset):
    def jobs(study, **options):
        return list(fitting_study._corrected_jobs(SimpleNamespace(
            identity_studies=[study], output=tmp_path_factory.mktemp(study), model_file=asset,
            seeds=(42, 43, 44), dtype="float64", device="cpu", spectra=True,
            lm_iterations=(0, 0, 30), **options,
        )))

    current = jobs(STUDY, focal_strategies=tuple(STRATEGIES))
    previous = jobs(focal_estimation.STUDY, focal_study_budget=100)
    return (
        {(job["seed"], job[STUDY]["condition"], job[STUDY]["strategy"]): job for job in current},
        {(job["seed"], job[focal_estimation.STUDY]["condition"]): job for job in previous},
    )


def _short_job(job, budget):
    result = deepcopy(job)
    config = FitConfig.from_dict(result["config"])
    result["config"] = replace(config, stages=tuple(
        replace(stage, iterations=count) for stage, count in zip(config.stages, budget, strict=True)
    )).to_dict()
    result[STUDY]["requested_stage_budgets"] = dict(zip(focal_convergence.STAGE_NAMES, budget, strict=True))
    return result


def test_matrix_reuses_v52_datasets_configurations_and_order(study_jobs):
    jobs, previous = study_jobs
    assert STUDY not in fitting_study.IDENTITY_STUDIES
    assert set(jobs) == {(seed, condition, strategy) for seed in (42, 43, 44)
                         for condition in CONDITIONS for strategy in STRATEGIES}
    assert Counter(job[STUDY]["experiment"] for job in jobs.values()) == {"A": 54, "B": 9}
    assert len({job["dataset"] for job in jobs.values()}) == 6
    for (seed, condition, strategy), job in jobs.items():
        case = job[STUDY]
        old = previous[seed, condition]
        config = FitConfig.from_dict(job["config"])
        assert tuple(stage.iterations for stage in config.stages) == STRATEGIES[strategy]
        assert case["requested_stage_budgets"] == {
            stage.name: stage.iterations for stage in config.stages
        }
        assert config.stages[0].groups == ("pose", "camera")
        assert config.stages[-1].groups == ("identity", "expression", "pose", "camera")
        assert config.lm == fitting_study.STRICT_LM
        assert config.identity_weight == config.expression_weight == config.pose_weight == config.camera_weight == 0
        assert config.focal_mode == case["focal_mode"]
        assert config.initial_focal_factor == case["configured_initial_focal_factor"]
        assert job["expected_dataset"] == old["expected_dataset"]
        assert focal_convergence.dataset_comparison(job["expected_dataset"] | {STUDY: case})["status"] == "matched"
        assert job["dataset"] == jobs[seed, condition, "J100"]["dataset"]
        if strategy == "J100":
            assert job["config"] == old["config"]
        assert job["observations"] == 9 and not job["known_pose_diagnostic"]
    assert FitConfig().focal_mode == "fixed"
    assert tuple(stage.iterations for stage in FitConfig().stages) == (30, 70, 100)


def test_worker_starts_each_strategy_from_the_same_blind_initializer(study_jobs, monkeypatch):
    jobs, _ = study_jobs
    states = []

    class Initialized(Exception):
        pass

    def unexpected_truth(*args):
        pytest.fail("Ground truth was accessed before the fit completed")

    def verify_fit(model, observations, config, *, initial, trace_stages):
        assert initial is None and trace_stages
        assert all(observation.camera_initialization is None and observation.known_camera is None
                   for observation in observations)
        data = prepare_observations(observations, model.reference, config)
        state, choices = initialize(model, data, config, FitInitialization())
        snapshot = _snapshot(FitState.from_tensors(state))
        assert not np.any(snapshot["identity"]) and not np.any(snapshot["expression"])
        assert all(camera["focal_length"] == [2.0 * min(camera["image_size"])] * 2
                   for camera in snapshot["cameras"])
        states.append((snapshot, choices))
        raise Initialized

    monkeypatch.setattr(BenchmarkDataset, "ground_truth", unexpected_truth)
    monkeypatch.setattr(fitting_study, "fit_identity", verify_fit)
    for strategy in STRATEGIES:
        with pytest.raises(Initialized):
            fitting_study._run_job(jobs[42, "A6", strategy])
    assert len(states) == 3 and states[0] == states[1] == states[2]


@pytest.mark.parametrize("condition,coordinate,width", (
    ("A6", "shared_log_focal", 55), ("A8", "local_log_focal", 63),
))
def test_lm_calibration_carries_the_live_parameters_into_joint_and_scopes_spectra(
    study_jobs, monkeypatch, condition, coordinate, width,
):
    jobs, _ = study_jobs
    job = _short_job(jobs[42, condition, "P100"], (3, 0, 2))
    original = lm.normal_blocks
    first_at = {}

    def inspect(model, data, parameters, layout, base, position):
        at = "joint" if layout.identity else "pose_camera"
        if at not in first_at:
            values = layout.unpack(base, position)
            first_at[at] = (
                _snapshot(FitState.from_tensors(parameters.state(0, len(data), values))),
                getattr(values, coordinate).detach().tolist(),
                getattr(values, "camera_delta").detach().tolist(), layout.expression_slice(),
            )
        return original(model, data, parameters, layout, base, position)

    monkeypatch.setattr(lm, "normal_blocks", inspect)
    record = fitting_study._run_job(job)
    calibration, skipped, joint = record["stage_diagnostics"]
    assert (calibration["status"], skipped["status"], joint["status"]) == (
        "budget_exhausted", "skipped", "budget_exhausted",
    )
    assert (calibration["steps"], skipped["steps"], joint["steps"]) == (3, 0, 2)
    before, after = calibration["before"], calibration["after"]
    assert first_at["pose_camera"][0] == before["physical_parameters"]
    assert first_at["pose_camera"][3] is None
    entry, focal, delta, expression_slice = first_at["joint"]
    assert entry == after["physical_parameters"] == joint["before"]["physical_parameters"]
    assert focal == after["optimizer_coordinates"][coordinate]
    assert np.any(focal) and focal != before["optimizer_coordinates"][coordinate]
    assert delta == after["optimizer_coordinates"]["camera_delta"] and np.any(delta)
    assert expression_slice is not None
    assert any(old["translation"] != new["translation"]
               and old["translation"][2] != new["translation"][2]
               and old["focal_length"] != new["focal_length"]
               for old, new in zip(before["physical_parameters"]["cameras"], entry["cameras"], strict=True))
    assert after["complete_objective"] == joint["before"]["complete_objective"]
    assert record["stage_transition"]["status"] == "continuous"
    assert record["stage_transition"]["objective_delta"] == 0
    assert joint["initial_damping"] == fitting_study.STRICT_LM.initial_damping != calibration["final_damping"]
    diagnostics = record["calibration_diagnostics"]
    assert diagnostics["identity_frozen"] and diagnostics["expression_frozen"]
    assert not np.any(entry["identity"]) and not np.any(entry["expression"])
    assert np.any(record["parameter_snapshots"]["final"]["identity"])
    assert np.any(record["parameter_snapshots"]["final"]["expression"])
    assert all(item["status"] == "evaluated" for item in diagnostics["states"].values())
    spectrum = record["post_calibration_identity_observability"]
    assert spectrum["state"] == "pose_camera"
    assert spectrum["joint_parameter_groups"] == ["identity", "expression", "pose", "camera"]
    assert spectrum["active_calibration"]["parameter_groups"] == ["pose", "camera"]
    assert spectrum["active_calibration"]["parameter_count"] == width
    assert len(spectrum["active_calibration"]["singular_values"]) == width
    assert spectrum["after_all_nuisance"]["parameter_count"] == 170
    assert spectrum["identity_prior_jtj_diagonal"] == spectrum["focal_prior_jtj_diagonal"] == 0
    assert not any(spectrum["local_prior_jtj_diagonal"].values())
    for counter in ("jacobian_evaluations", "rejected_trials"):
        assert sum(stage[counter] for stage in (calibration, skipped, joint)) == record["optimization"][counter]
    assert 1 + sum(stage["objective_evaluations"] for stage in (calibration, skipped, joint)) == (
        record["optimization"]["objective_evaluations"]
    )
    assert record["optimization"]["boundary_objective_evaluations"] == 4
    assert len(calibration["accepted_states"]) == 3 and len(joint["accepted_states"]) == 2


@pytest.mark.parametrize("terminal", ("converged", "no_acceptable_step", "numerical_failure"))
def test_calibration_termination_and_unused_budget(study_jobs, asset, monkeypatch, terminal):
    jobs, _ = study_jobs
    job = _short_job(jobs[42, "A3", "P100"], (3, 0, 2))
    dataset = load_dataset(job["dataset"])
    subject, = dataset.subject_ids
    observations = dataset.fitting_observations(subject, dtype="float64", device="cpu", include_known_cameras=False)
    config = FitConfig.from_dict(job["config"])
    model = load_benchmark_model(asset)
    original = optimizer.fit_lm_stage
    actual_calls = []

    def finish_calibration(model, data, parameters, stage, names, values, stats, run, record):
        actual_calls.append((stage.name, stage.iterations))
        if stage.name == "joint":
            return original(model, data, parameters, stage, names, values, stats, run, record)
        if terminal == "numerical_failure":
            raise FloatingPointError("injected calibration failure")
        return values, stats, terminal, "forced calibration termination for accounting"

    monkeypatch.setattr(optimizer, "fit_lm_stage", finish_calibration)
    fit = fit_identity(model, observations, config, trace_stages=True)
    assert fit.stages[0].steps == 0 and fit.stages[0].status == terminal
    if terminal == "converged":
        assert actual_calls == [("pose_camera", 3), ("joint", 2)]
        assert fit.stages[-1].steps == fit.optimization.accepted_updates == 2
    else:
        assert actual_calls == [("pose_camera", 3)]
        assert fit.status == terminal and fit.final is None
        assert len(fit.stage_diagnostics) == 1
        spectrum = identity_observability(model, observations, config, fit=fit, at="pose_camera")
        assert spectrum["active_calibration"]["parameter_groups"] == ["pose", "camera"]


def test_matched_prefix_and_archived_comparison_report_discrepancies(study_jobs):
    jobs, _ = study_jobs
    first = fitting_study._run_job(_short_job(jobs[42, "A3", "J100"], (0, 0, 2)))
    second = fitting_study._run_job(_short_job(jobs[42, "A3", "J200"], (0, 0, 3)))
    archived = deepcopy(first)
    archived[focal_estimation.STUDY] = archived.pop(STUDY)
    history = {"cases": [archived]}
    summary = focal_convergence.summarize([first, second], ["J100", "J200"], historical=history)
    pair, = summary["comparisons"]["budget_isolation"]
    assert pair["inputs_matched"] and pair["same_physical_initialization"]
    assert pair["common_prefix"]["status"] == "matched"
    assert pair["extension"]["additional_accepted_updates"] == 1
    assert pair["extension"]["continued_objective_decrease"]
    historical, = summary["historical_comparison"]["comparisons"]
    assert historical["status"] == "matched"
    for index, (row, detail) in enumerate(zip(summary["cases"], (first, second), strict=True)):
        assert row["report_case"] == f"report.json#/cases/{index}"
        assert row["status"] == detail["status"]
        assert row["accepted_updates"] == sum(stage["steps"] for stage in row["stages"])
        assert row["recovery"] == detail["recovery"]

    second["stage_diagnostics"][-1]["accepted_states"][0]["camera_translations"][0][2] += 1
    archived["observation_ids"] = archived["observation_ids"][::-1]
    summary = focal_convergence.summarize([first, second], ["J100", "J200"], historical=history)
    assert summary["comparisons"]["budget_isolation"][0]["common_prefix"]["status"] == "discrepant"
    assert summary["historical_comparison"]["comparisons"][0]["status"] == "input_mismatch"
    assert focal_convergence.dataset_comparison(first | {"observation_ids": archived["observation_ids"]})[
        "status"
    ] == "mismatch"


def _cli(asset, output):
    return ["--protocol", "corrected-identity", "--identity-studies", STUDY, "--model-file",
            asset, "--output", str(output)]


@pytest.mark.parametrize("extras", (
    ["--focal-strategies", "J100", "J100"],
    ["--focal-strategies", "invalid"],
    ["--focal-study-budget", "100"],
    ["--camera-study-budgets", "60", "100"],
    ["--lm-iterations", "0", "0", "100"],
    ["--dtype", "float32"],
    ["--device", "cuda"],
    ["--seeds", "42", "43"],
    ["--identity-studies", STUDY, "exact"],
    ["--protocol", "historical"],
))
def test_cli_rejects_contradictions(tmp_path, asset, extras):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        fitting_study.main([*_cli(asset, output), *extras])
    assert error.value.code == 2 and not output.exists()


def test_focused_cli_writes_failed_case_once_and_preserves_unavailable_metrics(
    study_jobs, asset, tmp_path, monkeypatch,
):
    jobs, _ = study_jobs
    job = jobs[42, "A6", "P100"]
    calls = []

    def select(args):
        assert args.focal_strategies == ["P100"]
        yield deepcopy(job)

    def failed_worker(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=17, stderr="ValueError: failed worker\n")

    monkeypatch.setattr(fitting_study, "_corrected_jobs", select)
    monkeypatch.setattr(fitting_study.subprocess, "run", failed_worker)
    monkeypatch.setattr(fitting_study, "__spec__", SimpleNamespace(name=fitting_study.__name__))
    output = tmp_path / "study"
    argv = [*_cli(asset, output), "--focal-strategies", "P100", "--spectra"]
    assert fitting_study.main(argv) == 1 and len(calls) == 1
    report, summary = (read_json(output / filename) for filename in ("report.json", "summary.json"))
    detail, = report["cases"]
    row, = summary["cases"]
    assert row["case"] == detail["case"] and row["status"] == detail["status"] == "worker_failed"
    assert row["report_case"] == "report.json#/cases/0"
    assert row["recovery"] is row["accepted_updates"] is row["final_focal_factors"] is None
    assert row["camera_recovery"] is row["geometry_metrics"] is row["observability"]["final"] is None
    assert report["strategy_schedules"] == summary["strategy_schedules"]
    assert summary["aggregates"][0]["failed_count"] == 1
    assert summary["aggregates"][0]["unavailable_recovery_count"] == 1
    with pytest.raises(FileExistsError):
        fitting_study.main(argv)
    assert len(calls) == 1
