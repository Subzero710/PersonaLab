"""Run the Issue #5 synthetic protocol and consolidate model-independent metrics."""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from gnm.shape import gnm_utils
import numpy as np

from ..camera import PerspectiveCamera
from ..fitting import FitConfig, FitInitialization, ObservationInitialization, Stage, fit_identity
from ..fitting.configuration import SOLVERS, LevenbergMarquardtConfig
from ..fitting.monitoring import optimization_record, peak_rss_bytes
from ..fitting.observability import identity_observability
from ..fitting.results import FAILED_STATUSES
from . import camera_calibration, focal_estimation
from .configuration import resolve_config
from .dataset import camera_from_record, camera_to_record, load_dataset
from .evaluate import _pool_expression, _pool_reprojection, _score_observation
from .full_native_camera import (
    STUDY as CAMERA_STUDY,
    add_observation_details, camera_protocol, case_matrix, classify_compatibility, coefficient_floor,
    final_gradient, fitted_camera_prediction, historical_comparison, summarize as summarize_camera_study,
)
from .generate import generate_dataset
from .metrics import (
    METRIC_UNITS, fixed_coefficient_error, grouped_coefficient_errors, neutral_geometry_errors,
)
from .model import load_benchmark_model
from .predictions import SubjectPrediction
from .storage import json_bytes, read_json, write_json


COUNTS = (1, 5, 10, 25, 50, 100)
IDENTITY_STUDIES = ("exact", "convergence", "count", "held-out", "robustness")
CALIBRATION_STUDY = camera_calibration.STUDY
FOCAL_STUDY = focal_estimation.STUDY
STRICT_LM = LevenbergMarquardtConfig(
    scaled_gradient_tolerance=1e-12, minimum_damping=1e-12,
    native_step_tolerance=1e-6, scaled_step_tolerance=1e-10,
)
IDENTITY_TARGETS = {
    "trainable_identity_rmse": 1e-4, "neutral_mesh_rms_m": 1e-6,
    "facial_region_rms_m": 1e-6, "reprojection_rms_px": 1e-4,
}
SCALING_COUNTS = (50, 100, 300, 600)
REFERENCE_ID = "d9ed77a6e5f39022b4dedd371749642ed626cb1b68ee008b21f5763920d7832f"
CONTROLLED = {"head_000": 0.42, "head_001": -0.25, "head_002": 0.16}


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


def _scoring_geometry(model, state, truth, trainable):
    return {
        "identity": grouped_coefficient_errors(
            state.identity, truth.arrays["identity"], model.metadata["identity_names"],
            trainable=trainable, regions=gnm_utils.identity_regions_indices(model.head),
        ),
        **neutral_geometry_errors(model, state.identity, truth.arrays["neutral_vertices"]),
    }


def _active_groups(config):
    return {group for stage in config.stages if stage.iterations for group in stage.groups}


def _compatibility(model, truth, fit, groups):
    initial, arrays = fit.initial, truth.arrays
    identity = fixed_coefficient_error(
        initial.identity, arrays["identity"], model.metadata["identity_names"],
        fit.selection.identity if "identity" in groups else (),
    )
    train_expression = fit.selection.expression if "expression" in groups else ()
    expression_names = model.metadata["expression_names"]
    train_columns = np.isin(expression_names, train_expression)
    fixed_joints = np.isin(
        model.metadata["joint_names"], fit.selection.fixed_joints if "pose" in groups
        else model.metadata["joint_names"],
    )
    root_free = fit.selection.root_translation and "pose" in groups
    camera_free = fit.config.camera_mode == "estimated" and "camera" in groups
    observations = []
    for i, camera in enumerate(initial.cameras):
        item = truth.observations[i]
        target = camera_to_record(camera_from_record(
            item["camera"], dtype=camera.rotation.dtype, device=camera.rotation.device,
        ))
        actual = camera_to_record(camera)
        if not camera_free or target["type"] != actual["type"]:
            fields = actual.keys() | target.keys()
        else:
            fields = ("type", "image_size", "rotation")
            if isinstance(camera, PerspectiveCamera):
                fields += ("focal_length", "principal_point") if fit.config.focal_mode == "fixed" else ("principal_point",)
        mismatched_camera = sorted(key for key in fields if actual.get(key) != target.get(key))
        expression = fixed_coefficient_error(
            initial.expression[i], arrays["expression"][i], expression_names, train_expression,
        )
        outside = np.abs(arrays["expression"][i]) > fit.config.expression_limit
        bounded = [name for name, mask in zip(expression_names, outside & train_columns, strict=True) if mask]
        joints = [
            name for name, fixed, a, b in zip(
                model.metadata["joint_names"], fixed_joints, initial.rotations[i], arrays["rotations"][i], strict=True,
            )
            if fixed and not np.array_equal(a, b)
        ]
        root = not root_free and not np.array_equal(initial.translation[i], arrays["translation"][i])
        observations.append({
            "id": item["id"], "expression": expression, "expression_outside_bounds": bounded,
            "differing_fixed_joints": joints, "differing_fixed_root": root,
            "differing_fixed_camera_fields": mismatched_camera,
            "representable": not (expression["differing_coefficients"] or bounded or joints or root or mismatched_camera),
        })
    return {
        "fully_representable": not identity["differing_coefficients"] and all(
            item["representable"] for item in observations
        ),
        "identity": identity, "observations": observations,
        "full_native_expression_rmse_floor": float(np.sqrt(np.mean([
            item["expression"]["full_native_rmse_floor"] ** 2 for item in observations
        ]))),
    }


def _score_observations(model, dataset, truth, prediction, mode, *, trainable_expression=None):
    observations = [
        _score_observation(
            model, dataset, truth, index, prediction.identity, prediction, mode,
            compare_camera_gauges=mode == "complete", trainable_expression=trainable_expression,
        )
        for index in range(len(truth.observations))
    ]
    return {
        "aggregate": _pool_reprojection(observations), "expression": _pool_expression(observations),
        "observations": observations,
    }


def _held_out(model, job, config, fit, training_truth):
    if fit.final is None:
        return {"status": "training_fit_failed"}
    parent = load_dataset(job["held_out_dataset"])
    subject, = parent.subject_ids
    start = job.get("held_out_start", 100)
    ids = parent.observation_ids(subject)[start:]
    if not ids:
        raise ValueError("Held-out dataset contains no reserved observations")
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
    conditional["diagnostic"] = "oracle_ground_truth_nuisance"
    estimated = None
    if nuisance_fit.final is not None:
        if not np.array_equal(nuisance_fit.final.identity, fit.final.identity):
            raise ValueError("Held-out nuisance fit changed the frozen identity")
        estimated = _score_observations(
            model, held, truth, nuisance_fit.to_prediction(), "known-camera",
            trainable_expression=nuisance_fit.selection.expression,
        )
        estimated["diagnostic"] = "estimated_expression_and_pose"
    return {
        "status": nuisance_fit.status, "reason": nuisance_fit.reason,
        "source_dataset_id": held.source_dataset_id, "evaluation_dataset_id": held.dataset_id,
        "observation_ids": ids, "camera_parameters": "known for both held-out diagnostics",
        "frozen_identity_exact": True if nuisance_fit.final is not None else None,
        "conditional_ground_truth_nuisance": conditional,
        "estimated_expression_and_pose": estimated,
        "runtime": nuisance_fit.runtime, "config": nuisance_config.to_dict(),
    }


def _run_job(job):
    case_started = perf_counter()
    config = FitConfig.from_dict(job["config"])
    calibration = CALIBRATION_STUDY in job
    camera_study = CALIBRATION_STUDY if calibration else CAMERA_STUDY
    camera_case = job.get(camera_study)
    focal_case = job.get(FOCAL_STUDY)
    if calibration and (config.camera_mode != "estimated" or job["known_pose_diagnostic"]):
        raise ValueError("Camera calibration requires blind initialization and estimated cameras")
    parent = load_dataset(job["dataset"])
    subject, = parent.subject_ids
    ids = parent.observation_ids(subject)[:job["observations"]]
    if len(ids) != job["observations"]:
        raise ValueError("The source dataset does not contain the requested observation count")
    dataset = parent.select_observations(subject, ids)
    provenance = {
        "source_dataset_id": parent.source_dataset_id, "evaluation_dataset_id": dataset.dataset_id,
        "observation_ids": ids,
    }
    if (calibration or focal_case) and json_bytes(provenance) != json_bytes(job["expected_dataset"]):
        label = FOCAL_STUDY if focal_case else "Camera calibration"
        raise ValueError(f"{label} dataset or observation order differs from the generated source: {provenance}")
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
        if camera_case or focal_case:
            raise ValueError("The full-native camera study requires the normal pose initializer")
        diagnostic = dataset.ground_truth(subject)
        initial = FitInitialization(observations={
            key: ObservationInitialization(rotations=model.tensor(rotation))
            for key, rotation in zip(ids, diagnostic.arrays["rotations"], strict=True)
        })
    fit = fit_identity(model, observations, config, initial=initial)
    truth = dataset.ground_truth(subject)
    groups = _active_groups(config)
    train_identity = fit.selection.identity if "identity" in groups else ()
    train_expression = fit.selection.expression if "expression" in groups else ()
    score_initial = _scoring_geometry(model, fit.initial, truth, train_identity)
    record = {
        "case": job["case"], "experiment": job["experiment"], "generation": parent.config.metadata(),
        "seed": job["seed"], "solver": config.solver, "config": config.to_dict(),
        "observations": len(ids), "status": fit.status, "reason": fit.reason, **provenance,
        "model_asset_sha256": model.metadata["asset_sha256"], "geometry_aligned": False,
        "budget": job["budget"], "known_pose_diagnostic": job["known_pose_diagnostic"],
        "selection": asdict(fit.selection), "gauge": fit.gauge,
        "compatibility": _compatibility(model, truth, fit, groups), "metric_units": METRIC_UNITS,
        "initial": score_initial | {
            "reprojection": asdict(fit.initial_reprojection) if fit.initial_reprojection else None,
        },
        "final": None, "runtime": fit.runtime | {"model_loading_seconds": load_seconds},
        "optimization": optimization_record(fit.optimization), "stages": [asdict(stage) for stage in fit.stages],
    }
    if fit.final is not None:
        estimated = config.camera_mode == "estimated"
        score_final = _scoring_geometry(model, fit.final, truth, train_identity)
        fixed_names = set(model.metadata["identity_names"]) - set(train_identity)
        fixed = [name in fixed_names for name in model.metadata["identity_names"]]
        score_final["fixed_identity_preserved"] = bool(
            np.array_equal(fit.initial.identity[fixed], fit.final.identity[fixed]),
        )
        prediction = fitted_camera_prediction(fit) if camera_case or focal_case else fit.to_prediction()
        training = _score_observations(
            model, dataset, truth, prediction, "complete" if estimated or camera_case or focal_case else "known-camera",
            trainable_expression=train_expression,
        )
        score_final["training_reprojection_and_nuisance_recovery"] = training
        if camera_case:
            record["camera_comparison"] = add_observation_details(training, truth, fit)
        elif estimated and not focal_case:
            comparable = sum(
                item["camera_recovery"]["translation_comparability"]["status"] == "comparable"
                for item in training["observations"]
            )
            record["camera_comparison"] = {
                "comparable_translation_observations": comparable,
                "non_comparable_translation_observations": len(training["observations"]) - comparable,
            }
        record["final"] = score_final
    diagnostic_started = perf_counter()
    spectra_requested = (job["spectra"] and bool(fit.selection.identity), job.get("final_spectra", False))
    spectra_status = {}
    for at, requested in zip(("initial", "final"), spectra_requested, strict=True):
        if not requested:
            continue
        key = f"{at}_identity_observability"
        record[key] = None
        if getattr(fit, at) is None:
            spectra_status[at] = {"status": "unavailable", "reason": fit.reason}
            continue
        try:
            record[key] = identity_observability(model, observations, config, fit=fit, at=at)
        except (ArithmeticError, RuntimeError, ValueError) as error:
            if not (camera_case or focal_case):
                raise
            spectra_status[at] = {"status": "failed", "reason": str(error)}
        else:
            spectra_status[at] = {"status": "evaluated", "reason": None}
    if job["spectra"] or job.get("final_spectra"):
        record["runtime"]["observability_seconds"] = perf_counter() - diagnostic_started
    if camera_case:
        record["observability_status"] = spectra_status
        record["optimization"]["final_gradient"] = final_gradient(model, observations, fit, config.stages[-1].groups)
        _camera_study_result(record, camera_case, model, fit, study=camera_study)
    if calibration:
        camera_calibration.add_diagnostics(record, fit, _judge_targets(record["final"]))
        mismatched = record["representability_details"]["fixed_focal_gauge_observations"]
        expected = list(ids) if camera_case["calibration_condition"] == "incorrect-fixed-intrinsics" else []
        record.update(dataset_verified=True, focal_calibration_check={
            "matches_expected": mismatched == expected,
            "expected_mismatch_observation_ids": expected, "actual_mismatch_observation_ids": mismatched,
        })
    if focal_case:
        record["observability_status"] = spectra_status
        record["optimization"]["final_gradient"] = final_gradient(model, observations, fit, config.stages[-1].groups)
        record["history"] = [asdict(item) for item in fit.history]
        focal_estimation.add_diagnostics(record, focal_case, truth, fit, _judge_targets(record["final"]))
        focal_estimation.complete_record(record)
    if job["held_out_dataset"] is not None:
        record["held_out"] = _held_out(model, job, config, fit, truth)
    if "identity_studies" in job:
        record.update(
            identity_studies=job["identity_studies"], convergence_settings=job["convergence_settings"],
            mono_image_underdetermined=len(ids) == 1,
        )
        if set(job["identity_studies"]) & {"exact", "convergence"}:
            record["validation_targets"] = _judge_targets(record["final"])
        record["runtime"].update(
            case_elapsed_seconds=perf_counter() - case_started,
            case_peak_process_rss_bytes=peak_rss_bytes(),
        )
    return record


def _judge_targets(final):
    actual = {} if final is None else {
        "trainable_identity_rmse": final["identity"]["trainable"]["rmse"],
        "neutral_mesh_rms_m": final["neutral_geometry"]["rms_m"],
        "facial_region_rms_m": final["neutral_geometry_regions"]["facial"]["rms_m"],
        "reprojection_rms_px": final["training_reprojection_and_nuisance_recovery"]["aggregate"]["clean"]["rms_px"],
    }
    scores = {
        name: {"maximum": maximum, "actual": actual.get(name),
               "met": actual[name] <= maximum if name in actual and actual[name] is not None else None}
        for name, maximum in IDENTITY_TARGETS.items()
    }
    scores["fixed_identity_preserved"] = {
        "required": True, "met": final["fixed_identity_preserved"] if final is not None else None,
    }
    return scores


def _camera_study_result(record, case, model, fit, *, study=CAMERA_STUDY):
    compatibility = record["compatibility"]
    native_names = model.metadata["identity_names"]
    regions = gnm_utils.identity_regions_indices(model.head)
    eye_and_teeth_names = [native_names[i] for group in ("eyes", "teeth") for i in regions[group]]
    classification = classify_compatibility(compatibility, case, eye_and_teeth_names)
    final = record["final"]
    floor = coefficient_floor(
        record["initial"]["identity"], final["identity"] if final is not None else None,
        compatibility["identity"]["differing_coefficients"],
    )
    targets = _judge_targets(final)
    head_target = targets["trainable_identity_rmse"]
    all_targets_met = all(item["met"] for item in targets.values()) if final is not None else None
    spectrum = record.get("final_identity_observability")
    record.update({
        study: case,
        "execution_status": "failed" if fit.status in FAILED_STATUSES else "completed",
        "converged": fit.status == "converged",
        "representability_status": classification["status"],
        "representability_details": classification,
        "identity_error_floor": floor,
        "scientific_validation": {
            "trainable_identity_matches_known_camera_tolerance": head_target["met"],
            "known_camera_reference_trainable_rmse_maximum": head_target["maximum"],
            "matches_known_camera_recovery_targets": all_targets_met,
            "full_native_identity_representable": not floor["incompatible_fixed_count"],
            "full_native_recovery_at_known_camera_tolerances": (
                False if floor["incompatible_fixed_count"] else all_targets_met
            ),
            "final_effective_identity_jacobian_rank_deficient": (
                spectrum["after_all_nuisance"]["nullity"] > 0 if spectrum is not None else None
            ),
        },
    })
    if "camera_comparison" not in record:
        record["camera_comparison"] = camera_protocol(fit)
    if case["condition"] == "D":
        record["validation_targets"] = targets
    if case["condition"] == "historical":
        record["historical_comparison"] = historical_comparison(record)


def _corrected_jobs(args):
    selected = set(args.identity_studies)
    sources = {}
    dataset_provenance = {}

    def source(scenario, seed, count, noise=0.0, *, camera="both", identity_selection="head", focal_schedule=None):
        key = (scenario, seed, count, noise, camera, identity_selection, focal_schedule)
        if key not in sources:
            suffix = f"head-gaussian-{noise!r}" if camera == "both" and identity_selection == "head" else (
                f"{camera}-{identity_selection or 'legacy'}"
            )
            if focal_schedule is not None:
                suffix += "-heterogeneous-focal"
            path = args.output / "datasets" / f"{scenario}-{seed}-{count}-{suffix}"
            options = {"gaussian_noise_px": noise} if noise else {}
            if identity_selection is not None:
                options.update(identity_selection=identity_selection, dtype=args.dtype, device=args.device)
            if focal_schedule is not None:
                options["perspective_focal_factors"] = focal_schedule
            config = resolve_config(
                scenario, seed=seed, observations_per_subject=count, camera=camera, **options,
            )
            dataset = generate_dataset(path, config, model_file=args.model_file)
            sources[key] = str(path)
            if selected & {CALIBRATION_STUDY, FOCAL_STUDY}:
                subject, = dataset.subject_ids
                dataset_provenance[str(path)] = {
                    "source_dataset_id": dataset.source_dataset_id, "evaluation_dataset_id": dataset.dataset_id,
                    "observation_ids": dataset.observation_ids(subject),
                }
        return sources[key]

    def job(name, studies, dataset, seed, count, *, strict=True, held_out=False,
            identity_weight=0.0, expression_weight=0.0, camera_mode="known", camera_model=None,
            joint_budget=None, focal_factor=None, focal_mode="fixed"):
        budget = args.lm_iterations if joint_budget is None else (0, 0, joint_budget)
        overrides = {
            "camera_mode": camera_mode, "camera_model": camera_model, "train_identity": ("head",),
            "identity_weight": identity_weight, "expression_weight": expression_weight,
            "lm": STRICT_LM if strict else LevenbergMarquardtConfig(),
        }
        if joint_budget is not None:
            overrides.update(pose_weight=0.0, camera_weight=0.0)
        if focal_mode != "fixed":
            overrides.update(focal_mode=focal_mode, initial_focal_factor=focal_factor)
        elif focal_factor is not None:
            overrides["perspective_focal_per_short_edge"] = focal_factor
        config = _fit_config("levenberg-marquardt", budget, overrides)
        both_spectra = args.spectra or "convergence" in studies
        return {
            "case": name, "experiment": "corrected-identity", "identity_studies": studies,
            "convergence_settings": "strict" if strict else "original", "dataset": dataset,
            "seed": seed, "observations": count, "config": config.to_dict(),
            "known_pose_diagnostic": False, "device": args.device, "dtype": args.dtype,
            "model_file": args.model_file, "budget": "fit",
            "held_out_dataset": dataset if held_out else None, "held_out_start": count,
            "spectra": both_spectra or count == 1, "final_spectra": both_spectra,
        }

    for seed in args.seeds:
        exact = [name for name in ("exact", "convergence") if name in selected]
        if exact:
            for scenario in ("B", "C"):
                path = source(scenario, seed, 9)
                yield job(f"exact-{scenario}-{seed}-strict", exact, path, seed, 9)
                if "convergence" in selected:
                    yield job(f"convergence-{scenario}-{seed}-original", ["convergence"], path, seed, 9, strict=False)
        if "count" in selected:
            path = source("C", seed, max(COUNTS))
            for count in COUNTS:
                yield job(f"count-C-{seed}-{count:03}", ["count"], path, seed, count)
        if "held-out" in selected:
            path = source("C", seed, 18)
            yield job(f"held-out-C-{seed}", ["held-out"], path, seed, 9, held_out=True)
        if "robustness" in selected:
            for noise_index, noise in enumerate(args.noise_levels):
                path = source("C", seed, 9, noise)
                for id_index, identity_weight in enumerate(args.identity_weights):
                    for expr_index, expression_weight in enumerate(args.expression_weights):
                        yield job(
                            f"robustness-C-{seed}-{noise_index}-{id_index}-{expr_index}", ["robustness"],
                            path, seed, 9, strict=False, identity_weight=identity_weight,
                            expression_weight=expression_weight,
                        )
    if CAMERA_STUDY in selected:
        for case in case_matrix(args.seeds):
            condition, seed = case["condition"], case["seed"]
            camera, mode = case["camera_model"], case["camera_mode"]
            if condition == "historical":
                name = f"{CAMERA_STUDY}-historical"
                selection = None
            else:
                name = f"{CAMERA_STUDY}-{condition}-{seed}-{camera}"
                selection = "head" if case["identity_ground_truth"] == "head-only" else "full"
            path = source("C", seed, 9, camera=camera, identity_selection=selection)
            yield job(
                name, [CAMERA_STUDY], path, seed, 9,
                camera_mode=mode, camera_model=camera if mode == "estimated" else None,
            ) | {CAMERA_STUDY: case}
    if CALIBRATION_STUDY in selected:
        for case in camera_calibration.case_matrix(args.seeds, args.camera_study_budgets):
            condition, seed, camera = case["condition"], case["seed"], case["camera_model"]
            path = source("C", seed, 9, camera=camera, identity_selection="head")
            yield job(
                f"{CALIBRATION_STUDY}-{condition}-{seed}", [CALIBRATION_STUDY], path, seed, 9,
                camera_mode=case["camera_mode"], camera_model=camera,
                joint_budget=case["joint_lm_budget"], focal_factor=case["focal_factor"],
            ) | {CALIBRATION_STUDY: case, "expected_dataset": dataset_provenance[path]}
    if FOCAL_STUDY in selected:
        for case in focal_estimation.case_matrix(args.seeds, args.focal_study_budget):
            condition, seed, mode = case["condition"], case["seed"], case["camera_mode"]
            schedule = focal_estimation.HETEROGENEOUS_FACTORS if case["experiment"] == "B" else None
            path = source("C", seed, 9, camera="perspective", focal_schedule=schedule)
            yield job(
                f"{FOCAL_STUDY}-{condition}-{seed}", [FOCAL_STUDY], path, seed, 9,
                camera_mode=mode, camera_model="perspective" if mode == "estimated" else None,
                joint_budget=case["joint_lm_budget"], focal_factor=case["configured_initial_focal_factor"],
                focal_mode=case["focal_mode"],
            ) | {FOCAL_STUDY: case, "expected_dataset": dataset_provenance[path]}


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
                "seed": seed, "observations": count, "config": config.to_dict(),
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


def _summarize_counts(records, *, corrected=False):
    cohorts = defaultdict(list)
    curves = defaultdict(list)
    for item in records:
        included = "count" in item.get("identity_studies", ()) if corrected else item["experiment"] == "E"
        if not included or "generation" not in item:
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
        if corrected:
            distributions["trainable_identity_rmse"] = [
                item["final"]["identity"]["trainable"]["rmse"] for item in completed
            ]
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
    parser.add_argument("--protocol", choices=("historical", "corrected-identity"), default="historical")
    parser.add_argument("--experiments", choices=tuple("ABCDEFGH"), nargs="+")
    parser.add_argument("--identity-studies", choices=IDENTITY_STUDIES + (CAMERA_STUDY, CALIBRATION_STUDY, FOCAL_STUDY),
                        nargs="+")
    parser.add_argument("--noise-levels", type=float, nargs="+", metavar="PX")
    parser.add_argument("--identity-weights", type=float, nargs="+")
    parser.add_argument("--expression-weights", type=float, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--adam-iterations", type=int, nargs=3)
    parser.add_argument("--lm-iterations", type=int, nargs=3, metavar=("POSE", "IDENTITY", "JOINT"))
    parser.add_argument("--camera-study-budgets", type=int, nargs=2, metavar=("SHORT", "LONG"),
                        help="explicit joint LM budgets for camera-calibration (60 100)")
    parser.add_argument("--focal-study-budget", type=int, metavar="JOINT",
                        help="explicit accepted joint LM update budget for focal-estimation (1–100)")
    parser.add_argument("--v51-report", type=Path, help="V5.1 camera-calibration report for matched focal controls")
    parser.add_argument("--scaling-full", action="store_true")
    parser.add_argument("--spectra", action="store_true",
                        help="initial spectra for A–C or both states in corrected studies")
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
    corrected = args.protocol == "corrected-identity"
    calibration = CALIBRATION_STUDY in (args.identity_studies or ())
    focal = FOCAL_STUDY in (args.identity_studies or ())
    if (args.focal_study_budget is not None or args.v51_report is not None) and not (corrected and focal):
        parser.error("--focal-study-budget and --v51-report require the corrected-identity focal-estimation study")
    if corrected and focal:
        if args.identity_studies != [FOCAL_STUDY]:
            parser.error("select focal-estimation as the only identity study")
        if args.focal_study_budget is None or not 1 <= args.focal_study_budget <= 100:
            parser.error("focal-estimation requires --focal-study-budget between 1 and 100")
        if args.lm_iterations is not None:
            parser.error("focal-estimation takes its joint budget from --focal-study-budget exclusively")
        if args.device != "cpu" or set(args.seeds) != set(focal_estimation.SEEDS):
            parser.error("focal-estimation requires CPU and seeds 42 43 44")
    if args.camera_study_budgets is not None and not (corrected and calibration):
        parser.error("--camera-study-budgets requires the corrected-identity camera-calibration study")
    if corrected and calibration:
        if args.identity_studies != [CALIBRATION_STUDY]:
            parser.error("select camera-calibration as the only identity study")
        if tuple(args.camera_study_budgets or ()) != camera_calibration.BUDGETS:
            parser.error("camera-calibration requires --camera-study-budgets 60 100")
        if args.lm_iterations is not None:
            parser.error("camera-calibration takes its joint budgets from --camera-study-budgets exclusively")
        if args.device != "cpu" or set(args.seeds) != set(camera_calibration.SEEDS):
            parser.error("camera-calibration requires CPU and seeds 42 43 44")
    if corrected and args.adam_iterations is not None:
        parser.error("--adam-iterations requires the historical protocol")
    args.adam_iterations = args.adam_iterations or (30, 70, 900)
    if args.lm_iterations is None:
        args.lm_iterations = (0, 0, 30) if corrected else (10, 12, 30)
    if corrected and any(args.lm_iterations[:2]):
        parser.error("corrected identity recovery uses a joint stage, requiring --lm-iterations 0 0 JOINT")
    if any(min(counts) < 0 or not counts[-1] for counts in (args.adam_iterations, args.lm_iterations)):
        parser.error("stage budgets must be nonnegative with a positive joint budget")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**64 for seed in args.seeds):
        parser.error("study seeds must be unique unsigned 64-bit integers")
    if corrected:
        if args.experiments is not None or args.historical_report is not None or args.scaling_full:
            parser.error("historical experiment options require the historical protocol")
        if args.dtype != "float64":
            parser.error("corrected identity recovery requires float64")
        args.identity_studies = args.identity_studies or IDENTITY_STUDIES
        if len(set(args.identity_studies)) != len(args.identity_studies):
            parser.error("select each identity study once")
        if CAMERA_STUDY in args.identity_studies and args.lm_iterations[-1] > 30:
            parser.error(f"{CAMERA_STUDY} allows at most 30 joint LM iterations")
        if len(args.seeds) < 2 and set(args.identity_studies) & {"count", "robustness"}:
            parser.error("corrected count and robustness studies require multiple seeds")
        if "robustness" in args.identity_studies:
            args.noise_levels = args.noise_levels or (0.5, 2.0)
            args.identity_weights = args.identity_weights or (0.0, 1e-6)
            args.expression_weights = args.expression_weights or (0.0, 2e-6, 2e-5)
            for name, values in (
                ("noise levels", args.noise_levels), ("identity weights", args.identity_weights),
                ("expression weights", args.expression_weights),
            ):
                if len(set(values)) != len(values) or len(values) < 2 or any(
                    not np.isfinite(value) or value < 0 for value in values
                ):
                    parser.error(f"robustness {name} require multiple distinct nonnegative finite values")
            if 0 in args.noise_levels or 0 not in args.identity_weights or 0 not in args.expression_weights:
                parser.error("robustness requires positive noise levels and zero baselines for both weights")
    elif args.identity_studies is not None:
        parser.error("--identity-studies requires --protocol corrected-identity")
    if not (corrected and "robustness" in args.identity_studies) and any(
        values is not None for values in (args.noise_levels, args.identity_weights, args.expression_weights)
    ):
        parser.error("noise and weight sweep options require the corrected robustness study")
    args.experiments = args.experiments or tuple("ABCDEFGH")
    if not corrected and set(args.experiments) & {"E", "F"} and (len(args.seeds) < 2 or 42 not in args.seeds):
        parser.error("count studies require seed 42 and at least one additional identity seed")
    args.output = args.output.resolve()
    reference = _reference_results(args.historical_report)
    v51 = None
    if args.v51_report is not None:
        v51 = read_json(args.v51_report)
        if not isinstance(v51, dict) or v51.get("requested_identity_studies") != [CALIBRATION_STUDY] or not isinstance(
            v51.get("cases"), list
        ):
            parser.error("--v51-report requires a detailed V5.1 camera-calibration report")
    args.output.mkdir(parents=True, exist_ok=False)
    jobs_path = args.output / "cases"
    jobs_path.mkdir()
    report = {
        "schema_version": 2, "reference_seed_42_scenario_c_source_id": REFERENCE_ID,
        "protocol": args.protocol, "requested_experiments": () if corrected else args.experiments,
        "device": args.device, "dtype": args.dtype, "metric_units": METRIC_UNITS,
        "seeds": args.seeds, "adam_iterations": args.adam_iterations, "lm_iterations": args.lm_iterations,
        "cases": [],
    }
    if corrected:
        report["requested_identity_studies"] = args.identity_studies
        if "robustness" in args.identity_studies:
            report["robustness_grid"] = {
                "gaussian_noise_px": args.noise_levels, "identity_weights": args.identity_weights,
                "expression_weights": args.expression_weights,
            }
    if calibration:
        report.update(adam_iterations=None, lm_iterations=None, camera_study_budgets=args.camera_study_budgets)
    if focal:
        report.update(adam_iterations=None, lm_iterations=None, focal_study_budget=args.focal_study_budget)
    report_path = args.output / "report.json"
    summary_path = args.output / "summary.json"
    failed = False
    for job in (_corrected_jobs(args) if corrected else _jobs(args)):
        path = jobs_path / (job["case"] + ".json")
        job["result"] = str(jobs_path / (job["case"] + ".result.json"))
        write_json(path, job)
        print(f"Running {job['case']}", flush=True)
        worker_options = {"stderr": subprocess.PIPE, "text": True} if calibration or focal else {}
        process = subprocess.run([sys.executable, "-m", __spec__.name, "--worker", str(path)],
                                 check=False, **worker_options)
        if process.returncode:
            failed = True
            result = {"case": job["case"], "experiment": job["experiment"], "status": "worker_failed",
                      "return_code": process.returncode}
            if FOCAL_STUDY in job:
                reason = f"Fitting worker exited with status {process.returncode}"
                if process.stderr.strip():
                    reason = f"{reason}: {process.stderr.strip().splitlines()[-1]}"
                result.update(job["expected_dataset"], dataset_verified=False, config=job["config"],
                              reason=reason, execution_status="failed", converged=None, final=None,
                              reference_targets=_judge_targets(None), worker_stderr=process.stderr,
                              observability_status={at: {"status": "unavailable", "reason": reason}
                                                    for at in ("initial", "final")} if args.spectra else {})
                result[FOCAL_STUDY] = job[FOCAL_STUDY]
            if CAMERA_STUDY in job or CALIBRATION_STUDY in job:
                study = CALIBRATION_STUDY if calibration else CAMERA_STUDY
                reason = f"Fitting worker exited with status {process.returncode}"
                if calibration and process.stderr.strip():
                    reason = f"{reason}: {process.stderr.strip().splitlines()[-1]}"
                result.update({
                    study: job[study], "reason": reason,
                    "execution_status": "failed", "converged": None, "final": None,
                    "representability_status": "unavailable",
                    "observability_status": {
                        at: {"status": "unavailable", "reason": reason} for at in ("initial", "final")
                    } if job["spectra"] else {},
                })
                if calibration:
                    result.update(job["expected_dataset"], dataset_verified=False, config=job["config"],
                                  worker_stderr=process.stderr, reference_targets=_judge_targets(None))
        else:
            result = read_json(Path(job["result"]))
            if result["experiment"] == "E" and result["generation"]["scenario"] == "C" and result["seed"] == 42:
                _compare_reference(result, reference.get(result["observations"]))
            if CAMERA_STUDY in result or CALIBRATION_STUDY in result:
                if (
                    result["execution_status"] == "failed"
                    or result["representability_status"] == "unexpected_incompatibility"
                    or result["optimization"]["final_gradient"]["status"] == "failed"
                    or any(item["status"] == "failed" for item in result["observability_status"].values())
                    or (result.get("historical_comparison") or {}).get("matches_source_dataset_id") is False
                    or (calibration and not result["focal_calibration_check"]["matches_expected"])
                    or (result["final"] is not None and (
                        not result["final"]["fixed_identity_preserved"]
                        or not result["identity_error_floor"]["decomposition"]["valid"]
                    ))
                ):
                    failed = True
            elif FOCAL_STUDY in result:
                failed |= (
                    result["execution_status"] == "failed"
                    or not result["compatibility"]["matches_experimental_design"]
                    or result["optimization"]["final_gradient"]["status"] == "failed"
                    or any(item["status"] == "failed" for item in result["observability_status"].values())
                )
            elif corrected and (
                result["status"] in FAILED_STATUSES or not result["compatibility"]["fully_representable"]
                or (result.get("held_out") or {}).get("status") in FAILED_STATUSES
            ):
                failed = True
        report["cases"].append(result)
        report["count_study"] = _summarize_counts(report["cases"], corrected=corrected)
        if calibration:
            summary = camera_calibration.summarize(report["cases"])
            report.update({key: summary[key] for key in (
                "comparisons", "optimization_prefix_tolerances", "historical_comparison",
            )})
            failed |= any(
                pair["status"] != "matched" or (pair["optimization_prefix"] or {}).get("status") == "discrepant"
                for comparison in summary["comparisons"] for pair in comparison["pairs"]
            )
        if focal:
            summary = focal_estimation.summarize(report["cases"], v51)
            report.update({key: summary[key] for key in ("comparisons", "aggregates", "historical_comparison")})
        temporary = report_path.with_suffix(".tmp")
        write_json(temporary, report)
        temporary.replace(report_path)
        if calibration or focal or CAMERA_STUDY in result:
            temporary = summary_path.with_suffix(".tmp")
            write_json(temporary, summary if calibration or focal else summarize_camera_study(report["cases"]))
            temporary.replace(summary_path)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
