"""V6 orchestration and small fitting checks against the pinned GNM asset."""

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from persona_lab.benchmark import fitting_study, focal_scalability, scalability_data, scalability_monitoring
from persona_lab.benchmark.scalability_cases import STUDY, case_matrix
from persona_lab.benchmark.storage import read_json
from persona_lab.fitting import FitConfig, fit_identity
from persona_lab.fitting import losses
from persona_lab.fitting.monitoring import OperationTimings
from persona_lab.fitting.observability import identity_observability


def test_j200_settings_schedule_and_explicit_selection():
    config = fitting_study._scalability_config(4)
    assert config.lm == replace(fitting_study.STRICT_LM, arithmetic_dtype="float32", jacobian_rows=32)
    assert config.to_dict()["lm"]["arithmetic_dtype"] == "float32"
    assert FitConfig.from_dict(config.to_dict()) == config
    assert "arithmetic_dtype" not in FitConfig().to_dict()["lm"]
    assert config.solver == "levenberg-marquardt" and config.camera_model == "perspective"
    assert config.camera_mode == "estimated" and config.focal_mode == "per-observation"
    assert config.initial_focal_factor == 1.5
    assert (config.identity_weight, config.expression_weight, config.pose_weight, config.camera_weight) == (0,) * 4
    assert tuple(stage.iterations for stage in config.stages) == (0, 0, 200)
    assert config.train_identity == ("head",) and len(config.train_expression) == 24
    assert config.landmark_subset is None
    assert STUDY not in fitting_study.IDENTITY_STUDIES
    assert FitConfig().focal_mode == "fixed"
    assert tuple(stage.iterations for stage in FitConfig().stages) == (30, 70, 100)
    factors = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 1.1, 1.5, 1.9)
    assert scalability_data.SCHEDULE == (factors * 67)[:600]
    generation = scalability_data.generation_config(42)
    assert generation.device == "cpu" and generation.dtype == "float32"
    assert generation.identity_selection == "head" and generation.observations_per_subject == 600
    assert generation.perspective_focal_factors == scalability_data.SCHEDULE


def _cli(tmp_path, device):
    return [
        "--protocol", "corrected-identity", "--identity-studies", STUDY,
        "--model-file", str(tmp_path / "gnm_head.npz"), "--device", device, "--dtype", "float32",
        "--scalability-counts", "50", "100", "300", "600", "--scalability-batch-size", "4",
        "--spectra", "--output", str(tmp_path / "out"),
    ]


def test_cli_rejects_missing_cuda_input_and_contradictory_flags(tmp_path):
    for args in (
        _cli(tmp_path, "cuda"),
        _cli(tmp_path, "cpu") + ["--lm-iterations", "0", "0", "10"],
        _cli(tmp_path, "cpu") + ["--scalability-counts", "100", "100"],
        _cli(tmp_path, "cpu") + ["--dtype", "float64"],
    ):
        with pytest.raises(SystemExit) as rejected:
            fitting_study.main(args)
        assert rejected.value.code == 2
    assert not (tmp_path / "out").exists()


def test_unavailable_cuda_records_all_cases_without_starting_a_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden(*args, **kwargs):
        pytest.fail("An unavailable CUDA request started a generator or worker")

    monkeypatch.setattr(focal_scalability, "generate_parent", forbidden)
    monkeypatch.setattr(focal_scalability, "_run_subprocess", forbidden)
    assert fitting_study.main(_cli(tmp_path, "cuda") + [
        "--scalability-datasets-in", str(tmp_path / "reference"),
    ]) == 1
    results = read_json(tmp_path / "out" / "summary.json")["cases"]
    assert len(results) == 16
    assert all(row["failure"]["kind"] == "cuda_unavailable" for row in results)
    assert all(row["termination_status"] is None and row["actual_execution_device"] is None for row in results)


def test_worker_error_retains_published_fit(tmp_path, monkeypatch):
    config = fitting_study._scalability_config(4)
    case = case_matrix(devices=("cpu",))[0]
    job = {"case": case, "config": config.to_dict(), "spectra": True}

    def fail(command, **kwargs):
        from persona_lab.benchmark.storage import write_json

        path = Path(read_json(Path(command[-1]))["result"])
        partial = focal_scalability._record(job) | {"execution_status": "completed", "termination_status": "converged"}
        write_json(path.with_suffix(".fit.json"), partial)
        return SimpleNamespace(returncode=-9, stderr="")

    monkeypatch.setattr(focal_scalability.subprocess, "run", fail)
    row = focal_scalability._run_subprocess(job, tmp_path)
    assert row["termination_status"] == "converged" and row["execution_status"] == "completed"
    assert row["failure"]["kind"] == "worker_process_failure"
    assert row["failure"]["return_code"] == -9
    assert row["observability_status"]["final"]["status"] == "unavailable"


def test_phase_rss_sampling_differs_from_lifetime_high_water_mark(monkeypatch):
    current = [10]
    monkeypatch.setattr(scalability_monitoring, "_rss", lambda: current[0])
    monkeypatch.setattr(scalability_monitoring, "peak_rss_bytes", lambda: 90)
    monitor = scalability_monitoring.CaseMeasurements()
    monitor.device = torch.device("cpu")
    monitor.start("initialization")
    current[0] = 50
    monitor.end()
    current[0] = 30
    monitor.start("optimization")
    monitor.end()
    _, memory = monitor.finish()
    assert memory["baseline_rss_bytes"] == 10 and memory["peak_increase_above_baseline_bytes"] == 80
    assert memory["peak_process_rss_bytes"] == 90
    assert memory["phases"]["initialization"]["sampled_peak_rss_bytes"] == 50
    assert memory["phases"]["optimization"]["sampled_peak_rss_bytes"] == 30
    assert memory["cuda_peak_allocated_bytes"] is None and memory["cuda_peak_reserved_bytes"] is None


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    model = Path(os.environ.get("PERSONALAB_GNM_MODEL_FILE", "/cache/gnm/v3_0/gnm_head.npz"))
    if not model.is_file():
        pytest.skip("Pinned official GNM v3.0 Head asset is unavailable")
    dataset = tmp_path_factory.mktemp("v6-reference") / "seed-42"
    return model, dataset, scalability_data.generate_parent(dataset, 42, model)


def test_reference_archive_reuses_nested_observations_and_rejects_byte_changes(reference, tmp_path):
    from shutil import copytree

    _, archive, record = reference
    parent = scalability_data.load_parent(archive, 42, record)
    previous, identity = (), None
    for count in (50, 100, 300, 600):
        dataset, subject, provenance = scalability_data.verified_subset(parent, count)
        assert provenance["observation_ids"][:len(previous)] == previous
        assert len(provenance["observation_ids"]) == count
        assert sum(provenance["distribution"]["viewpoint_counts"].values()) == count
        assert sum(provenance["distribution"]["expression_preset_counts"].values()) == count
        assert sum(provenance["distribution"]["focal_factor_counts"].values()) == count
        if identity is not None:
            assert provenance["identity_sha256"] == identity
        previous, identity = provenance["observation_ids"], provenance["identity_sha256"]
        observations = dataset.fitting_observations(subject, dtype="float32", include_known_cameras=False)
        scalability_data.verify_fitter_input(observations, provenance, device=torch.device("cpu"))
        with pytest.raises(ValueError, match="differ from"):
            scalability_data.verify_fitter_input(observations[::-1], provenance, device=torch.device("cpu"))
    damaged = copytree(archive, tmp_path / "damaged")
    with (damaged / "subject_0000.npz").open("ab") as destination:
        destination.write(b"modified")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        scalability_data.load_parent(damaged, 42, record)


def _short_configuration(batch_size):
    config = fitting_study._scalability_config(batch_size)
    return replace(config, stages=(*config.stages[:-1], replace(config.stages[-1], iterations=2)))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requested_device_cold_start_and_isolated_diagnostic_failure(reference, tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model_file, archive, provenance = reference
    case = case_matrix(devices=(device,))[0]
    result_path = tmp_path / "case.result.json"
    job = {"case": case, "config": _short_configuration(4).to_dict(), "model_file": str(model_file),
           "dataset": str(archive), "reference": provenance, "spectra": True, "result": str(result_path)}
    calls = []
    from persona_lab.benchmark.dataset import BenchmarkDataset
    original_truth = BenchmarkDataset.ground_truth
    in_fit = False
    def guarded_truth(dataset, subject):
        assert not in_fit, "Ground truth entered the fitting function"
        return original_truth(dataset, subject)
    monkeypatch.setattr(BenchmarkDataset, "ground_truth", guarded_truth)
    load = focal_scalability.load_benchmark_model

    def inspect_load(*args, **kwargs):
        calls.append(kwargs["device"])
        return load(*args, **kwargs)

    def blind_fit(model, observations, config, *, initial, trace_stages, profile, progress):
        assert initial is None
        assert len(observations) == 50 and config.initial_focal_factor == 1.5
        assert all(item.known_camera is None and item.camera_initialization is None for item in observations)
        assert all(item.landmarks.device.type == device and item.landmarks.dtype == torch.float32
                   for item in observations)
        nonlocal in_fit
        in_fit = True
        try:
            fit = fit_identity(model, observations, config, initial=initial, trace_stages=trace_stages, profile=profile, progress=progress)
        finally:
            in_fit = False
        assert not np.any(fit.initial.identity) and not np.any(fit.initial.expression)
        assert all(camera.focal_length.tolist() == [1.5 * min(camera.image_size.width, camera.image_size.height)] * 2
                   for camera in fit.initial.cameras)
        return fit

    def unavailable(*args, **kwargs):
        raise RuntimeError("explicit diagnostic failure for test")

    monkeypatch.setattr(focal_scalability, "load_benchmark_model", inspect_load)
    monkeypatch.setattr(focal_scalability, "identity_observability", unavailable)
    assert focal_scalability.run_worker(job, blind_fit, fitting_study._score_scalability_fit) == 0
    row = read_json(result_path)
    assert len(calls) == 1 and calls[0].type == device
    assert row["actual_execution_device"].split(":")[0] == device and row["actual_dtype"] == "float32"
    assert row["dataset_verified"]
    oracle = row["reference_precision"]
    assert oracle["status"] == "evaluated" and not oracle["used_for_fitting"]
    assert oracle["compensated_minus_archive"]["eligible_landmarks"] == 3400
    assert len(oracle["truth_linearization"]["identity_step"]) == 170
    assert row["environment"]["determinism"]["float32_matmul_precision"] == "highest"
    assert not row["environment"]["determinism"]["cuda_matmul_allow_tf32"]
    assert row["execution_status"] == "completed" and row["final"] is not None
    assert {at: value["status"] for at, value in row["observability_status"].items()} == dict.fromkeys(
        ("initial", "final"), "failed",
    )
    assert len(row["selection"]["identity"]) == 170 and len(row["selection"]["fixed_identity"]) == 83
    assert len(row["selection"]["expression"]) == 24
    blocks = row["optimizer_runtime"]["normal_system"]
    assert blocks["shared_dimensions"] == [170, 170] and blocks["local_dimensions"] == [31, 31]
    assert blocks["dtype"] == "torch.float32"
    assert blocks["representation"] == "square_root_qr" and blocks["normal_equations_bytes"] == 0
    assert blocks["local_blocks"] == 50 and blocks["local_coordinate_widths"]["local_log_focal"] == 1
    memory = read_json(result_path.with_suffix(".telemetry.json"))["memory"]
    assert (memory["cuda_peak_allocated_bytes"] is not None) == (device == "cuda")


def test_batch_size_controls_lm_projection_and_preserves_numerics(reference, monkeypatch):
    model_file, archive, record = reference
    parent = scalability_data.load_parent(archive, 42, record)
    dataset, subject, _ = scalability_data.verified_subset(parent, 9)
    observations = dataset.fitting_observations(subject, dtype="float32", include_known_cameras=False)
    model = focal_scalability.load_benchmark_model(model_file, dtype="float32")
    from persona_lab.fitting import batched_jacobian
    project_batch = batched_jacobian.residual_fp32_batch
    executed = []

    def tracked(data, parameters, start, stop, values):
        executed.append(stop - start)
        return project_batch(data, parameters, start, stop, values)

    monkeypatch.setattr(batched_jacobian, "residual_fp32_batch", tracked)
    first = fit_identity(model, observations, _short_configuration(1))
    assert set(executed) == {1}
    executed.clear()
    second = fit_identity(model, observations, _short_configuration(4))
    assert set(executed) == {1, 4}
    np.testing.assert_array_equal(first.initial.identity, second.initial.identity)
    # Short-run FP32 batching checks, not the scientific exact-recovery targets.
    np.testing.assert_allclose(first.initial.rotations, second.initial.rotations, atol=1e-6)
    np.testing.assert_allclose(first.final.identity, second.final.identity, atol=2e-4, rtol=2e-3)
    np.testing.assert_allclose(first.final.expression, second.final.expression, atol=2e-4, rtol=2e-3)
    assert first.optimization.final_objective.total == pytest.approx(
        second.optimization.final_objective.total, abs=1e-8, rel=1e-3,
    )
    operations = OperationTimings(model.reference.device)
    original = identity_observability(model, observations, second.config, fit=second, at="initial")
    profiled = identity_observability(model, observations, second.config, fit=second, at="initial", operations=operations)
    tensors = profiled.pop("compressed_tensors")
    assert tensors["joint_identity_and_focal"]["shape"][-1] == 179
    assert tensors["joint_identity_and_focal"]["dtype"] == "torch.float32"
    assert operations.record()["calls"]["qr"] > 0
    assert profiled == original


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_square_root_batched_jacobian_against_single_view(reference, device):
    """The batched derivative must agree with independent single-view derivatives on real GNM."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from persona_lab.fitting.batched_jacobian import batch_jacobian
    from persona_lab.fitting.configuration import GROUPS
    from persona_lab.fitting.initialization import initialize
    from persona_lab.fitting.levenberg_marquardt import StageCoordinates, observation_jacobian
    from persona_lab.fitting.observations import prepare_observations
    from persona_lab.fitting.parameters import FitInitialization, FittingParameters, resolve_selection
    from persona_lab.fitting.losses import evaluate_objective

    model_file, archive, provenance = reference
    parent = scalability_data.load_parent(archive, 42, provenance)
    dataset, subject, _ = scalability_data.verified_subset(parent, 9)
    model = focal_scalability.load_benchmark_model(model_file, dtype=torch.float32, device=torch.device(device))
    observations = dataset.fitting_observations(subject, dtype="float32", device=device, include_known_cameras=False)
    config = _short_configuration(4)
    data = prepare_observations(observations, model.reference, config)
    selection = resolve_selection(model, config)
    state, _ = initialize(model, data, config, FitInitialization())
    parameters = FittingParameters(model, data, state, selection, config)
    layout = StageCoordinates.from_parameters(parameters, GROUPS)
    base = parameters.values()
    point = layout.pack(base, model.reference, len(data))
    squared = model.reference.new_zeros(())
    for start in range(0, len(data), 4):
        stop = min(len(data), start + 4)
        residuals, jacobians = batch_jacobian(data, parameters, layout, point, start, stop)
        squared += residuals.square().sum()
        for local, index in enumerate(range(start, stop)):
            old_r, old_j = observation_jacobian(data, parameters, layout, base, point, index)
            mask = data.eligible[index].repeat_interleave(2)
            torch.testing.assert_close(residuals[local, mask], old_r, atol=2e-8, rtol=5e-4)
            torch.testing.assert_close(jacobians[local, mask], old_j, atol=1e-7, rtol=5e-4)
            assert not torch.count_nonzero(jacobians[local, ~mask])
            assert jacobians.dtype == torch.float32 and jacobians.device.type == device
    objective, _ = evaluate_objective(model, data, parameters, least_squares=True)
    assert objective.landmarks == pytest.approx(float(squared), rel=2e-6, abs=1e-12)
