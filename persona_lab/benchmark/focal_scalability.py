"""Orchestrate the V6 reference datasets, isolated fits and diagnostics."""

from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from ..fitting import FitConfig
from ..fitting.parameters import resolve_selection
from ..fitting.monitoring import OperationTimings, backend_failure, optimization_record
from ..fitting.observability import identity_observability
from ..fitting.results import FAILED_STATUSES
from .camera_calibration import _snapshot
from .configuration import torch_options
from .focal_convergence import _stage_rows
from .full_native_camera import final_gradient
from .model import MODEL_SHA256, load_benchmark_model
from .residual_precision import reference_precision
from .scalability_cases import BASE_COMMIT, SCHEMA_VERSION, STUDY, case_matrix
from .scalability_data import (
    REFERENCE_SCHEMA, SCHEDULE, generate_parent, load_parent, verified_subset, verify_fitter_input,
)
from .scalability_monitoring import CaseMeasurements, environment
from .storage import json_bytes, read_json, write_json


def _failure(kind, phase, error):
    return {"kind": kind, "phase": phase, "reason": str(error)}


def _record(job):
    config = job["config"]
    reference = job.get("reference") or {}
    return job["case"] | {
        "config": config,
        "requested_stage_budgets": {stage["name"]: stage["iterations"] for stage in config["stages"]},
        "source_dataset_id": reference.get("dataset_id"),
        "reference_files_sha256": reference.get("files_sha256"),
        "dataset_verified": False, "dataset_id": None, "actual_observations": None,
        "actual_execution_device": None, "execution_status": "not_run", "termination_status": None,
        "worker_started": False,
        "converged": None, "reason": None, "failure": None,
        "spectra_requested": job["spectra"],
        "observability_status": {at: {"status": "unavailable" if job["spectra"] else "not_requested",
                                     "reason": None} for at in ("initial", "final")},
    }


def _mark_failure(record, failure):
    record.update(failure=failure, reason=failure["reason"])
    for item in record["observability_status"].values():
        if item["status"] == "unavailable":
            item["reason"] = failure["reason"]
    return record


def _publish(path, value):
    pending = path.with_suffix(".tmp")
    write_json(pending, value)
    pending.replace(path)


def _observability(record, model, observations, fit, measurements):
    if not record["spectra_requested"]:
        return
    for at, status in record["observability_status"].items():
        if getattr(fit, at) is None:
            status["reason"] = fit.reason
            continue
        measurements.start(f"{at}_observability")
        operations = OperationTimings(model.reference.device)
        try:
            spectrum = identity_observability(model, observations, fit.config, fit=fit, at=at, operations=operations)
            measurements.start(f"{at}_diagnostic_record_encoding")
            status["scientific_record_compact_json_bytes"] = len(json_bytes(spectrum))
            status.update(status="evaluated", reason=None)
            record[f"{at}_identity_observability"] = spectrum
        except (ArithmeticError, RuntimeError, TypeError, ValueError, MemoryError) as error:
            status.update(status="failed", reason=str(error), failure=backend_failure(error) or "diagnostic_failure")
        try:
            status["operation_timings"] = operations.record()
        except RuntimeError as error:
            status["operation_timings"] = {"error": str(error)}
        measurements.end()


def _fit_fields(fit):
    trace = optimization_record(fit.optimization)
    factorization_failed = "Cholesky factorization failed"
    trace["cholesky_factorization_failures"] = max(
        sum(trial["reason"].startswith(factorization_failed) for trial in trace["trials"]),
        int(factorization_failed in fit.reason),
    )
    return {
        "execution_status": "failed" if fit.status in FAILED_STATUSES else "completed",
        "termination_status": fit.status, "converged": fit.status == "converged", "reason": fit.reason,
        "selection": asdict(fit.selection), "gauge": fit.gauge,
        "parameter_snapshots": {"initial": _snapshot(fit.initial),
                                "final": _snapshot(fit.final) if fit.final is not None else None},
        "initialization": [asdict(choice) for choice in fit.initialization],
        "history": [asdict(item) for item in fit.history], "stages": [asdict(item) for item in fit.stages],
        "stage_diagnostics": _stage_rows(fit),
        "initial_reprojection": asdict(fit.initial_reprojection) if fit.initial_reprojection else None,
        "final_reprojection": asdict(fit.final_reprojection) if fit.final_reprojection else None,
        "optimization": trace, "optimizer_runtime": fit.runtime,
    }


def _profile_kernels(model, observations, fit, destination):
    """Profile a separate initial linearization and QR solve, not timed fitting."""
    from ..fitting.configuration import GROUPS
    from ..fitting.levenberg_marquardt import StageCoordinates, square_root_blocks, bound_mask
    from ..fitting.observability import _state
    from ..fitting.observations import prepare_observations
    from ..fitting.parameters import FittingParameters
    from ..fitting.square_root import square_root_step
    device = model.reference.device
    if device.type != "cuda":
        raise ValueError("CUDA kernel profiling requires actual CUDA execution")
    data = prepare_observations(observations, model.reference, fit.config)
    state = _state(model, data, fit.config, fit.selection, None, fit, "initial")
    parameters = FittingParameters(model, data, state, fit.selection, fit.config)
    layout = StageCoordinates.from_parameters(parameters, GROUPS)
    point = layout.pack(parameters.values(), model.reference, len(data))
    torch.cuda.synchronize(device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        with torch.profiler.record_function("personalab_batched_jacobian"):
            system = square_root_blocks(data, parameters, layout, point)
        with torch.profiler.record_function("personalab_square_root_qr"):
            scales = system.scales(fit.config.lm.diagonal_floor)
            fixed = bound_mask(point, system.gradient, layout.expression_slice(), fit.config.expression_limit)
            square_root_step(system, scales, fit.config.lm.initial_damping, fixed)
        torch.cuda.synchronize(device)
    profiler.export_chrome_trace(str(destination))
    events = sum(event.device_type == torch.autograd.DeviceType.CUDA for event in profiler.events())
    if events == 0:
        raise RuntimeError("Profiler captured no CUDA device events")
    return {"status": "evaluated", "cuda_device_events": events, "trace": destination.name,
            "scope": "separate initial linearization and square-root solve", "device": str(device)}


def run_worker(job, fit_function, score_fit):
    """A worker receives only immutable reference paths and an explicit cold-start configuration."""
    result_path = Path(job["result"])
    fit_path = result_path.with_suffix(".fit.json")
    telemetry_path = result_path.with_suffix(".telemetry.json")
    record = _record(job)
    record["worker_started"] = True
    measurements = CaseMeasurements()
    phase, error_kind = "device", "cuda_unavailable"
    try:
        dtype, device = torch_options(job["case"]["dtype"], job["case"]["requested_device"])
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(device)
        # Isolated worker: explicitly use full FP32, not TF32 matmul.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        measurements.device = device
        measurements.start("environment")
        record["environment"] = environment(device)
        config = FitConfig.from_dict(job["config"])
        if dtype != torch.float32 or config.lm.arithmetic_dtype != "float32":
            raise ValueError("V6 requires FP32 model and LM arithmetic")

        phase, error_kind = "dataset_loading", "invalid_dataset"
        measurements.start(phase)
        parent = load_parent(Path(job["dataset"]), record["seed"], job["reference"])
        dataset, subject, provenance = verified_subset(parent, record["requested_observations"])
        record.update(provenance, generation=parent.config.metadata())
        del parent

        phase, error_kind = "model_loading", "model_asset_failure"
        measurements.start(phase)
        model = load_benchmark_model(Path(job["model_file"]), dtype=dtype, device=device, profile=measurements)
        record.update(model=model.metadata, model_loads=1, actual_dtype=model.numpy_dtype.name)
        if model.reference.device != device or model.reference.dtype != dtype:
            raise RuntimeError("Loaded GNM model differs from the requested device or dtype")
        if model.metadata != dataset.model_metadata or not np.array_equal(model.triangles, dataset.triangles):
            raise ValueError("Fitting model and reference dataset provenance differ")
        selection = resolve_selection(model, config)
        if (len(selection.identity), len(selection.fixed_identity), len(selection.expression)) != (170, 83, 24):
            raise ValueError("V6 requires the native 170/83 identity and 24-expression fitting selection")

        phase, error_kind = "device_transfer", "invalid_dataset"
        measurements.start(phase)
        observations = dataset.fitting_observations(
            subject, dtype=dtype, device=device, include_known_cameras=False,
        )
        measurements.start("input_verification")
        verify_fitter_input(observations, provenance, device=device)
        record["dataset_verified"] = True

        phase, error_kind = "fitting", "fitting_initialization_failure"
        record.update(actual_execution_device=str(model.reference.device), execution_status="failed")
        def progress(log):
            print(f"{record['case']} {log.stage} update={log.iteration} "
                  f"objective={log.losses.total:.7g} rms_px={log.reprojection.rms_px:.7g} "
                  f"elapsed={log.elapsed_seconds:.2f}s device={model.reference.device}", flush=True)

        print(f"{record['case']} fitting on {model.reference.device} FP32 square-root QR compensated residual", flush=True)
        fit = fit_function(model, observations, config, initial=None, trace_stages=True, profile=measurements,
                           progress=progress)
        measurements.start("fit_record_preparation")
        record.update(_fit_fields(fit))
        if reason := fit.runtime.get("infrastructure_failure"):
            _mark_failure(record, _failure(reason, phase, fit.reason))

        # Publish the fit before optional diagnostics. The parent can retain it
        # if the worker is killed during evaluation or observability.
        phase, error_kind = "fit_record_serialization", "serialization_failure"
        measurements.start(phase)
        _publish(fit_path, record)
        phase, error_kind = "evaluation", "evaluation_failure"
        measurements.start(phase)
        try:
            score_fit(record, model, dataset, subject, fit)
        except (ArithmeticError, RuntimeError, ValueError, MemoryError) as error:
            _mark_failure(record, _failure(backend_failure(error) or error_kind, phase, error))
        # The fit snapshot has already been published. Oracle truth is now
        # permitted for this separate diagnostic only.
        measurements.start("reference_precision")
        try:
            record["reference_precision"] = reference_precision(model, dataset, subject, observations, fit)
        except (ArithmeticError, RuntimeError, TypeError, ValueError, MemoryError) as error:
            record["reference_precision"] = {"status": "failed", "reason": str(error), "used_for_fitting": False}
        measurements.end()
        measurements.start("final_gradient")
        record["optimization"]["final_gradient"] = final_gradient(model, observations, fit, config.stages[-1].groups)
        measurements.end()
        phase, error_kind = "observability", "diagnostic_failure"
        _observability(record, model, observations, fit, measurements)
        if job.get("profile_kernels", False):
            measurements.start("kernel_profiling")
            try:
                record["kernel_profile"] = _profile_kernels(model, observations, fit, result_path.with_suffix(".cuda-trace.json"))
            except (ArithmeticError, OSError, RuntimeError, ValueError, MemoryError) as error:
                record["kernel_profile"] = {"status": "failed", "reason": str(error)}
            finally:
                measurements.end()
    except (ArithmeticError, OSError, RuntimeError, KeyError, TypeError, ValueError, MemoryError) as error:
        _mark_failure(record, _failure(backend_failure(error) or error_kind, phase, error))

    try:
        measurements.start("result_serialization")
        _publish(result_path, record)
    except (OSError, TypeError, ValueError, MemoryError) as error:
        _mark_failure(record, _failure("serialization_failure", "result_serialization", error))
    finally:
        timing, memory = measurements.finish()
    gpu = record.get("environment", {}).get("gpu")
    memory["available_device_memory_bytes"] = gpu["available_memory_bytes"] if gpu is not None else None
    write_json(telemetry_path, {
        "timing": timing, "memory": memory, "failure": record["failure"],
        "case_result_bytes": result_path.stat().st_size if result_path.is_file() else None,
    })
    return 0 if result_path.is_file() else 1


def _run_subprocess(job, cases_path):
    case_id = job["case"]["case"]
    path = cases_path / f"{case_id}.json"
    result_path = cases_path / f"{case_id}.result.json"
    job["result"] = str(result_path)
    try:
        write_json(path, job)
    except (OSError, TypeError, ValueError, MemoryError) as error:
        return _mark_failure(_record(job), _failure("serialization_failure", "worker_job_serialization", error))
    started = perf_counter()
    failure = None
    worker_started = False
    try:
        process = subprocess.run(
            [sys.executable, "-m", "persona_lab.benchmark.fitting_study", "--worker", str(path)],
            stderr=subprocess.PIPE, text=True, check=False,
        )
        worker_started = True
        if process.returncode != 0:
            reason = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else (
                f"Fitting worker exited with status {process.returncode}"
            )
            failure = _failure("worker_process_failure", "worker", reason) | {"return_code": process.returncode}
    except OSError as error:
        failure = _failure("worker_process_failure", "worker", error)
    wall_seconds = perf_counter() - started
    saved_path = result_path if result_path.is_file() else result_path.with_suffix(".fit.json")
    record = _record(job)
    try:
        if saved_path.is_file():
            record = read_json(saved_path)
        elif failure is None:
            failure = _failure("worker_process_failure", "worker", "Worker did not publish a result")
        path = result_path.with_suffix(".telemetry.json")
        if path.is_file():
            telemetry = read_json(path)
            failure = telemetry.pop("failure") or failure
            record.update(telemetry)
    except (OSError, ValueError) as error:
        failure = _failure("serialization_failure", "worker_result_loading", error)
    record["worker_started"] = worker_started
    if worker_started and not saved_path.is_file() and failure is not None:
        record["execution_status"] = "failed"
    record.setdefault("timing", {})["worker_wall_seconds"] = wall_seconds
    return _mark_failure(record, failure) if failure is not None else record


def _references(args, output, seeds, *, model_available, device_error):
    root = args.scalability_datasets_in or args.scalability_datasets_out or output / "datasets"
    root = root.resolve()
    index_path = root / "reference.json"
    if args.scalability_datasets_in is not None:
        try:
            reference = read_json(index_path)
            if (reference.get("schema") != REFERENCE_SCHEMA or reference.get("focal_schedule") != list(SCHEDULE)
                    or reference.get("model_asset_sha256") != MODEL_SHA256
                    or not isinstance(reference.get("parents"), dict)):
                raise ValueError("V6 reference index has incompatible schema, focal schedule or model asset")
        except (OSError, ValueError) as error:
            return root, {}, {seed: _failure("invalid_dataset", "dataset_loading", error) for seed in seeds}
        parents = reference["parents"]
        return root, parents, {
            seed: _failure("invalid_dataset", "dataset_loading", f"Reference dataset for seed {seed} is missing")
            for seed in seeds if not isinstance(parents.get(str(seed)), dict)
        }
    if not model_available or device_error:
        return root, {}, {}
    root.mkdir(parents=True, exist_ok=False)
    parents, failures = {}, {}
    for seed in seeds:
        started = perf_counter()
        print(f"Generating CPU float32 reference dataset for seed {seed}", flush=True)
        try:
            parents[str(seed)] = generate_parent(root / f"seed-{seed}", seed, args.model_file)
            parents[str(seed)]["generation_seconds"] = perf_counter() - started
        except (ArithmeticError, OSError, RuntimeError, ValueError, MemoryError) as error:
            failures[seed] = _failure(backend_failure(error) or "dataset_generation_failure", "dataset_generation", error)
    _publish(index_path, {
        "schema": REFERENCE_SCHEMA, "focal_schedule": SCHEDULE,
        "model_asset_sha256": MODEL_SHA256, "parents": parents,
    })
    return root, parents, failures


def _case_records(args, config_for, cases, cases_path, root, parents, model_available, device_error, reference_errors):
    reusable = {row["reuse_case"] for row in cases if "reuse_case" in row}
    completed = {}
    for index, case in enumerate(cases):
        reference = parents.get(str(case["seed"]))
        job = {
            STUDY: True, "case": case, "config": config_for(case["batch_size"]).to_dict(),
            "model_file": args.model_file, "dataset": str(root / f"seed-{case['seed']}"),
            "reference": reference, "spectra": args.spectra, "profile_kernels": args.scalability_profile_kernels,
        }
        if reuse := case.get("reuse_case"):
            previous = completed[reuse]
            if json_bytes(previous["config"]) != json_bytes(job["config"]) or (
                previous["source_dataset_id"] != (reference or {}).get("dataset_id")
                or previous["reference_files_sha256"] != (reference or {}).get("files_sha256")
            ):
                raise ValueError(f"Batching case {case['case']} does not match its primary case")
            yield case | {"report_case": previous["report_case"]}
            continue
        failure = device_error or reference_errors.get(case["seed"])
        if not model_available and failure is None:
            failure = _failure("model_asset_unavailable", "model_loading",
                               f"Pinned GNM v3.0 Head asset is unavailable: {args.model_file}")
        print(f"Running {case['case']}" if failure is None else f"Blocked {case['case']}: {failure['reason']}", flush=True)
        record = _run_subprocess(job, cases_path) if failure is None else _mark_failure(_record(job), failure)
        record.setdefault("timing", {})["dataset_generation"] = (reference or {}).get("generation_seconds") if (
            args.scalability_datasets_in is None
        ) else None
        record["report_case"] = f"report.json#/cases/{index}"
        if case["case"] in reusable:
            completed[case["case"]] = {key: record[key] for key in (
                "config", "source_dataset_id", "reference_files_sha256", "report_case",
            )}
        yield record


def run(args, config_for, write_reports):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases_path = output / "cases"
    cases_path.mkdir()
    cases = case_matrix(args.seeds, args.scalability_counts, (args.device,), args.scalability_batch_size,
                        primary_only=args.scalability_primary_only)
    model_available = args.model_file is not None and Path(args.model_file).is_file()
    device_error = None
    try:
        torch_options(args.dtype, args.device)
    except (RuntimeError, ValueError) as error:
        device_error = _failure("cuda_unavailable", "device", error)
    seeds = tuple(dict.fromkeys(case["seed"] for case in cases))
    root, parents, reference_errors = _references(args, output, seeds, model_available=model_available,
                                                   device_error=device_error)
    report = {
        "schema_version": SCHEMA_VERSION, "protocol": "corrected-identity",
        "requested_identity_studies": [STUDY], "reference_base_commit": BASE_COMMIT,
        "requested_device": args.device, "dtype": args.dtype, "model_asset_sha256": MODEL_SHA256,
        "seeds": args.seeds, "observation_counts": args.scalability_counts,
        "primary_batch_size": args.scalability_batch_size, "spectra_requested": args.spectra,
        "primary_only": args.scalability_primary_only, "kernel_profiling_requested": args.scalability_profile_kernels,
        "preflight": {"model_file_exists": model_available, "device_error": device_error,
                      "reference_errors": {str(seed): error for seed, error in reference_errors.items()}},
        "reference_directory": str(root), "reference_parents": parents, "focal_schedule": SCHEDULE,
        "batch_semantics": {
            "configurable": ["pose grid initialization", "sparse objective", "observation Jacobians", "local QR elimination"],
            "observation_jacobians": "independent batched view coordinates; bounded autograd residual-row batches",
            "schur": "square-root local QR elimination and shared residual QR; no normal equations",
            "residual": "compensated homogeneous perspective FP32; immutable V6 raster observations",
        },
        "device_transfers": [
            "Dataset archives are loaded and validated on CPU. The verified model starts on CPU and moves to the requested device once.",
            "Fitting observations are copied to the requested device and explicitly read back once to verify the input fingerprint.",
            "The fitter copies parameter snapshots, camera records and reprojection statistics to CPU for reporting.",
            "Evaluation scores NumPy geometry and parameter errors on CPU using device-computed model outputs.",
            "Compressed observability QR and SVD execute on the model device. Scientific records are converted to CPU values.",
        ],
    }
    records = _case_records(args, config_for, cases, cases_path, root, parents, model_available,
                            device_error, reference_errors)
    summary = write_reports(output, report, records)
    failed_case = any(row["termination_status"] != "converged" or row["failure"]
                      or not (row["recovery"] or {}).get("exact_compatible_recovery")
                      or row["spectra_requested"] and not row["observability_complete"]
                      or row["final_gradient_status"] == "failed"
                      or (row.get("reference_precision") or {}).get("status") != "evaluated"
                      or args.scalability_profile_kernels and (row.get("kernel_profile") or {}).get("status") != "evaluated"
                      for row in summary["cases"])
    return int(failed_case or any(pair["status"] != "consistent" for pair in summary["batching_consistency"]))
