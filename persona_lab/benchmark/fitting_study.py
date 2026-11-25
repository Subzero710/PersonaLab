"""Run the Issue #5 synthetic protocol and consolidate model-independent metrics."""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

from ..fitting import FitConfig, FitInitialization, ObservationInitialization, Stage, fit_identity
from ..fitting.configuration import SOLVERS
from ..fitting.observability import identity_observability
from .configuration import resolve_config
from .dataset import load_dataset
from .evaluate import _pool_reprojection, _score_observation
from .generate import generate_dataset
from .metrics import coefficient_errors, vertex_errors
from .model import load_benchmark_model
from .predictions import SubjectPrediction
from .storage import json_bytes, read_json, write_json


COUNTS = (1, 5, 10, 25, 50, 100)
SCALING_COUNTS = (50, 100, 300, 600)
REFERENCE_ID = "d9ed77a6e5f39022b4dedd371749642ed626cb1b68ee008b21f5763920d7832f"
CONTROLLED = {"head_000": 0.42, "head_001": -0.25, "head_002": 0.16}
REGION_THRESHOLD = 1e-4


def _fit_config(solver, budget, overrides, *, original_adam_rates=False):
    default = FitConfig()
    stages = list(default.stages)
    if solver == "adam" and not original_adam_rates:
        joint = stages[-1]
        stages[-1] = replace(
            joint, learning_rate=joint.learning_rate / 10,
            group_rates=tuple((name, rate / 10) for name, rate in joint.group_rates),
        )
    stages = tuple(replace(stage, iterations=count) for stage, count in zip(stages, budget, strict=True))
    return replace(default, solver=solver, stages=stages, **overrides)


def _scoring_geometry(model, state, truth, selection, names):
    target = truth.arrays["identity"]
    truth_vertices = truth.arrays["neutral_vertices"]
    vertices = model.neutral(state.identity)
    region_weights = model.head.vertex_groups.detach().cpu().numpy()
    regions = {}
    for name, weights in zip(model.head.vertex_group_names, region_weights, strict=True):
        mask = weights > REGION_THRESHOLD
        if mask.any():
            regions[name] = vertex_errors(vertices[mask], truth_vertices[mask])
    indices = {name: index for index, name in enumerate(names)}
    score = {"global": coefficient_errors(state.identity, target)}
    for name, group in (("trainable", selection.identity), ("fixed", selection.fixed_identity)):
        columns = [indices[coefficient] for coefficient in group]
        score[name] = coefficient_errors(state.identity[columns], target[columns]) if columns else None
    return {
        "identity": score, "neutral_geometry": vertex_errors(vertices, truth_vertices),
        "vertex_groups": regions, "vertex_group_weight_threshold": REGION_THRESHOLD,
    }


def _score_observations(model, dataset, truth, prediction, mode):
    observations = [
        _score_observation(
            model, dataset, truth, index, prediction.identity, prediction, mode,
            compare_camera_gauges=mode == "complete",
        )
        for index in range(len(truth.observations))
    ]
    return {"aggregate": _pool_reprojection(observations), "observations": observations}


def _held_out(model, job, config, fit, training_truth):
    if fit.final is None:
        return {"status": "training_fit_failed"}
    parent = load_dataset(job["held_out_dataset"])
    subject, = parent.subject_ids
    ids = parent.observation_ids(subject)[100:]
    held = parent.select_observations(subject, ids)
    truth = held.ground_truth(subject)
    if set(ids) & set(fit.observation_ids) or not np.array_equal(
        truth.arrays["identity"], training_truth.arrays["identity"],
    ) or held.model_metadata != model.metadata:
        raise ValueError("Held-out data must use a separate observation range and the same synthetic identity")
    observations = held.fitting_observations(
        subject, dtype=job["dtype"], device=model.reference.device, include_known_cameras=True,
    )
    initial = FitInitialization(identity=model.tensor(fit.final.identity))
    nuisance_config = replace(config, train_identity=(), stages=(
        Stage("held_out_nuisance", config.stages[-1].iterations, ("expression", "pose"), 0.0022,
              (("pose", 0.0005),)),
    ))
    nuisance_fit = fit_identity(model, observations, nuisance_config, initial=initial)
    conditional = _score_observations(model, held, truth, SubjectPrediction(fit.final.identity), "conditional")
    estimated = None
    if nuisance_fit.final is not None:
        if not np.array_equal(nuisance_fit.final.identity, fit.final.identity):
            raise ValueError("Held-out nuisance fit changed the frozen identity")
        estimated = _score_observations(model, held, truth, nuisance_fit.to_prediction(), "known-camera")
    return {
        "status": nuisance_fit.status, "reason": nuisance_fit.reason,
        "source_dataset_id": held.source_dataset_id, "evaluation_dataset_id": held.dataset_id,
        "observation_ids": ids, "camera_parameters": "known for both held-out diagnostics",
        "conditional_ground_truth_nuisance": conditional,
        "estimated_expression_and_pose": estimated,
        "runtime": nuisance_fit.runtime, "config": asdict(nuisance_config),
    }


def _run_job(job):
    config = FitConfig.from_dict(job["config"])
    parent = load_dataset(job["dataset"])
    subject, = parent.subject_ids
    ids = parent.observation_ids(subject)[:job["observations"]]
    if len(ids) != job["observations"]:
        raise ValueError("The source dataset does not contain the requested observation count")
    dataset = parent.select_observations(subject, ids)
    started = perf_counter()
    model = load_benchmark_model(job["model_file"], dtype=job["dtype"], device=job["device"])
    load_seconds = perf_counter() - started
    if model.metadata != dataset.model_metadata or not np.array_equal(model.triangles, dataset.triangles):
        raise ValueError("Study dataset and fitting model provenance differ")
    observations = dataset.fitting_observations(
        subject, dtype=job["dtype"], device=model.reference.device,
        include_known_cameras=config.camera_mode == "known",
    )
    initial = None
    if job["known_pose_diagnostic"]:
        diagnostic = dataset.ground_truth(subject)
        initial = FitInitialization(observations={
            key: ObservationInitialization(rotations=model.tensor(rotation))
            for key, rotation in zip(ids, diagnostic.arrays["rotations"], strict=True)
        })
    fit = fit_identity(model, observations, config, initial=initial)
    truth = dataset.ground_truth(subject)
    score_initial = _scoring_geometry(model, fit.initial, truth, fit.selection, model.metadata["identity_names"])
    record = {
        "case": job["case"], "experiment": job["experiment"], "generation": parent.config.metadata(),
        "seed": job["seed"], "solver": config.solver, "config": asdict(config),
        "observations": len(ids), "observation_ids": ids, "status": fit.status, "reason": fit.reason,
        "source_dataset_id": parent.source_dataset_id, "evaluation_dataset_id": dataset.dataset_id,
        "model_asset_sha256": model.metadata["asset_sha256"], "geometry_aligned": False,
        "budget": job["budget"], "known_pose_diagnostic": job["known_pose_diagnostic"],
        "selection": asdict(fit.selection), "gauge": fit.gauge,
        "initial": score_initial | {
            "reprojection": asdict(fit.initial_reprojection) if fit.initial_reprojection else None,
        },
        "final": None, "runtime": fit.runtime | {"model_loading_seconds": load_seconds},
        "optimization": asdict(fit.optimization), "stages": [asdict(stage) for stage in fit.stages],
    }
    if fit.final is not None:
        estimated = config.camera_mode == "estimated"
        score_final = _scoring_geometry(model, fit.final, truth, fit.selection, model.metadata["identity_names"])
        fixed_names = set(fit.selection.fixed_identity)
        fixed = [name in fixed_names for name in model.metadata["identity_names"]]
        score_final["fixed_identity_preserved"] = bool(
            np.array_equal(fit.initial.identity[fixed], fit.final.identity[fixed]),
        )
        training = _score_observations(
            model, dataset, truth, fit.to_prediction(), "complete" if estimated else "known-camera",
        )
        score_final["training_reprojection_and_nuisance_recovery"] = training
        if estimated:
            comparable = sum(
                item["camera_recovery"]["translation_comparability"]["status"] == "comparable"
                for item in training["observations"]
            )
            record["camera_comparison"] = {
                "comparable_translation_observations": comparable,
                "non_comparable_translation_observations": len(training["observations"]) - comparable,
            }
        record["final"] = score_final
    if job["spectra"] and fit.selection.identity:
        spectrum = identity_observability(model, observations, config, initial=initial)
        for name in ("fixed_nuisance", "after_expression", "after_all_nuisance"):
            group = spectrum[name]
            values = group["singular_values"]
            group["condition_number"] = values[0] / values[-1] if values[-1] > 0 else None
        record["initial_identity_observability"] = spectrum
    if job["held_out_dataset"] is not None:
        record["held_out"] = _held_out(model, job, config, fit, truth)
    return record


def _jobs(args):
    parents = {}

    def parent(scenario, seed, count, *, camera="both", controlled=False):
        key = (scenario, seed, count, camera, controlled)
        if key not in parents:
            suffix = "-controlled" if controlled else ""
            path = args.output / "datasets" / f"{scenario}-{seed}-{count}-{camera}{suffix}"
            options = resolve_config(
                scenario, seed=seed, observations_per_subject=count, camera=camera,
                dtype=args.dtype, device=args.device,
            )
            generate_dataset(path, options, model_file=args.model_file,
                             identity_coefficients=CONTROLLED if controlled else None)
            parents[key] = path
        return parents[key]

    def paired(experiment, name, source, seed, count, *, overrides=None, known_pose=False,
               held_out=None, profiling=False, original_adam_rates=False):
        for solver in SOLVERS:
            budget = args.adam_iterations if solver == "adam" else args.lm_iterations
            if profiling:
                budget = (1, 1, 1)
            elif original_adam_rates and solver == "adam":
                budget = (30, 70, 100)
            config = _fit_config(solver, budget, overrides or {}, original_adam_rates=original_adam_rates)
            yield {
                "experiment": experiment, "case": f"{name}-{solver}", "dataset": str(source),
                "seed": seed, "observations": count, "config": asdict(config),
                "known_pose_diagnostic": known_pose, "device": args.device, "dtype": args.dtype,
                "model_file": args.model_file, "held_out_dataset": str(held_out) if held_out else None,
                "budget": "profiling_only" if profiling else "fit",
                "spectra": args.spectra and experiment in ("A", "B", "C"),
            }

    selected = set(args.experiments)
    neutral = {"train_expression": ()}
    if "A" in selected:
        controlled = parent("B", 42, 9, controlled=True)
        overrides = neutral | {"train_identity": tuple(CONTROLLED), "identity_weight": 0.0}
        yield from paired("A", "A-known-pose", controlled, 42, 9, known_pose=True,
                          overrides=overrides | {"train_head": False})
        yield from paired("A", "A-unknown-pose", controlled, 42, 9, overrides=overrides)
    if "B" in selected:
        source = parent("B", 42, 9)
        yield from paired("B", "B-neutral", source, 42, 9, overrides=neutral)
        yield from paired("B", "B2-stage-transition", source, 42, 9, original_adam_rates=True,
                          overrides=neutral | {"identity_weight": 0.0})
    if selected & {"C", "D"}:
        source = parent("C", 42, 9)
        if "C" in selected:
            yield from paired("C", "C-variable", source, 42, 9)
            yield from paired("C", "C-single-stage-transition", source, 42, 1, original_adam_rates=True)
        if "D" in selected:
            for label, overrides in (
                ("standard", {}), ("zero-identity", {"identity_weight": 0.0}),
                ("zero-expression", {"expression_weight": 0.0}),
                ("reduced-expression", {"expression_weight": 2e-6}),
                ("zero-identity-and-expression", {"identity_weight": 0.0, "expression_weight": 0.0}),
            ):
                yield from paired("D", f"D-{label}", source, 42, 9, overrides=overrides)
    if selected & {"E", "F"}:
        for scenario in ("B", "C"):
            for seed in args.seeds:
                source = parent(scenario, seed, 100)
                # The generator uses independent seeded streams per view.
                # The first 100 rows are reserved for fitting, including the
                # original seed-42 source manifest. Only the tail is scored.
                held = parent(scenario, seed, 109) if "F" in selected else None
                for count in COUNTS:
                    yield from paired(
                        "E", f"E-{scenario}-{seed}-{count:03}", source, seed, count, held_out=held,
                        overrides={"log_every": 100} | (neutral if scenario == "B" else {}),
                    )
    if "G" in selected:
        for camera in ("weak-perspective", "perspective"):
            yield from paired("G", f"G-{camera}", parent("C", 42, 9, camera=camera), 42, 9,
                              overrides={"camera_mode": "estimated", "camera_model": camera})
    if "H" in selected:
        source = parent("C", 42, 600)
        for count in SCALING_COUNTS:
            yield from paired("H", f"H-{count:03}", source, 42, count, profiling=not args.scaling_full)


def _summarize_counts(records):
    cohorts = defaultdict(list)
    curves = defaultdict(list)
    for item in records:
        if item["experiment"] != "E" or "generation" not in item:
            continue
        scenario, solver, count = item["generation"]["scenario"], item["solver"], item["observations"]
        cohorts[(scenario, solver, count)].append(item)
        curves[(scenario, solver, item["seed"])].append(item)
    aggregates = []
    for (scenario, solver, count), fits in sorted(cohorts.items()):
        completed = [item for item in fits if item["final"] is not None]
        distributions = {
            "identity_rmse": [item["final"]["identity"]["global"]["rmse"] for item in completed],
            "neutral_geometry_rms_m": [item["final"]["neutral_geometry"]["rms_m"] for item in completed],
            "training_rms_px": [
                item["final"]["training_reprojection_and_nuisance_recovery"]["aggregate"]["observed"]["rms_px"]
                for item in completed
            ],
            "optimization_seconds": [item["runtime"]["optimization_seconds"] for item in fits],
        }
        aggregates.append({
            "scenario": scenario, "solver": solver, "observations": count, "runs": len(fits),
            "statuses": {status: sum(item["status"] == status for item in fits)
                         for status in sorted({item["status"] for item in fits})},
            "metrics": {name: {"n": len(values), "mean": float(np.mean(values)), "sd": float(np.std(values))}
                        if values else None for name, values in distributions.items()},
        })
    changes = []
    for (scenario, solver, seed), fits in sorted(curves.items()):
        ordered = sorted(fits, key=lambda item: item["observations"])
        for before, after in zip(ordered, ordered[1:]):
            a, b = before["final"], after["final"]
            changes.append({
                "scenario": scenario, "solver": solver, "seed": seed,
                "from": before["observations"], "to": after["observations"],
                "geometry_rms_change_m": b["neutral_geometry"]["rms_m"] - a["neutral_geometry"]["rms_m"]
                if a is not None and b is not None else None,
            })
    return {"aggregates": aggregates, "adjacent_count_changes": changes}


def _reference_results(path):
    if path is None:
        return {}
    with path.open(encoding="utf-8") as source:
        rows = json.load(source)
    json_bytes(rows)
    if not isinstance(rows, list) or tuple(item["observations"] for item in rows) != COUNTS or any(
        item["source_dataset_id"] != REFERENCE_ID for item in rows
    ):
        raise ValueError("The historical report must contain the original Scenario C seed-42 ordered count study")
    return {item["observations"]: item for item in rows}


def _compare_reference(record, historical):
    record["matches_reference_source_id"] = record["source_dataset_id"] == REFERENCE_ID
    if historical is None:
        return
    matches = record["matches_reference_source_id"] and (
        tuple(record["observation_ids"]) == tuple(historical["observation_ids"])
    )
    comparison = {
        "same_source_and_order": matches, "historical_status": historical["status"],
        "historical_identity_rmse": historical["identity"]["rmse"],
        "historical_neutral_geometry_rms_m": historical["neutral_geometry"]["rms_m"],
        "historical_reprojection_rms_px": historical["final_reprojection"]["rms_px"],
        "historical_optimization_seconds": historical["runtime"]["optimization_seconds"],
        "historical_peak_rss_bytes": historical["runtime"]["peak_process_rss_bytes"],
    }
    if matches and record["final"] is not None:
        final = record["final"]
        comparison["delta_from_historical"] = {
            "identity_rmse": final["identity"]["global"]["rmse"] - comparison["historical_identity_rmse"],
            "neutral_geometry_rms_m": final["neutral_geometry"]["rms_m"] - comparison["historical_neutral_geometry_rms_m"],
            "reprojection_rms_px": (
                final["training_reprojection_and_nuisance_recovery"]["aggregate"]["observed"]["rms_px"]
                - comparison["historical_reprojection_rms_px"]
            ),
        }
    record["historical_comparison"] = comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="create a new study directory")
    parser.add_argument("--model-file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--experiments", choices=tuple("ABCDEFGH"), nargs="+", default=tuple("ABCDEFGH"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--adam-iterations", type=int, nargs=3, default=(30, 70, 900))
    parser.add_argument("--lm-iterations", type=int, nargs=3, default=(10, 12, 30))
    parser.add_argument("--scaling-full", action="store_true")
    parser.add_argument("--spectra", action="store_true", help="initial local SVDs for experiments A–C")
    parser.add_argument("--historical-report", type=Path, help="the original count-study-v2.json")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        job = read_json(args.worker)
        write_json(Path(job["result"]), _run_job(job))
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.model_file is None or not Path(args.model_file).is_file():
        parser.error("--model-file must identify the existing pinned GNM asset")
    if any(min(counts) < 0 or not counts[-1] for counts in (args.adam_iterations, args.lm_iterations)):
        parser.error("stage budgets must be nonnegative with a positive joint budget")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**64 for seed in args.seeds):
        parser.error("study seeds must be unique unsigned 64-bit integers")
    if set(args.experiments) & {"E", "F"} and (len(args.seeds) < 2 or 42 not in args.seeds):
        parser.error("count studies require seed 42 and at least one additional identity seed")
    args.output = args.output.resolve()
    reference = _reference_results(args.historical_report)
    args.output.mkdir(parents=True, exist_ok=False)
    jobs_path = args.output / "cases"
    jobs_path.mkdir()
    report = {
        "schema_version": 1, "reference_seed_42_scenario_c_source_id": REFERENCE_ID,
        "requested_experiments": args.experiments, "device": args.device, "dtype": args.dtype,
        "seeds": args.seeds, "adam_iterations": args.adam_iterations, "lm_iterations": args.lm_iterations,
        "cases": [],
    }
    report_path = args.output / "report.json"
    failed = False
    for job in _jobs(args):
        path = jobs_path / (job["case"] + ".json")
        job["result"] = str(jobs_path / (job["case"] + ".result.json"))
        write_json(path, job)
        print(f"Running {job['case']}", flush=True)
        process = subprocess.run([sys.executable, "-m", __spec__.name, "--worker", str(path)], check=False)
        if process.returncode:
            failed = True
            result = {"case": job["case"], "experiment": job["experiment"], "status": "worker_failed",
                      "return_code": process.returncode}
        else:
            result = read_json(Path(job["result"]))
            if result["experiment"] == "E" and result["generation"]["scenario"] == "C" and result["seed"] == 42:
                _compare_reference(result, reference.get(result["observations"]))
        report["cases"].append(result)
        report["count_study"] = _summarize_counts(report["cases"])
        temporary = report_path.with_suffix(".tmp")
        write_json(temporary, report)
        temporary.replace(report_path)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
