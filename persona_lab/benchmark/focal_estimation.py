"""Matched diagnostics for the V5.2 focal-estimation study."""

from dataclasses import asdict
from itertools import combinations
from math import isclose

import numpy as np

from ..fitting.results import FAILED_STATUSES
from .camera_calibration import DATASET_FIELDS, _camera_changes, _checkpoints, _identity_norms, _snapshot
from .full_native_camera import _metrics, coefficient_floor


STUDY = "focal-estimation"
SEEDS = (42, 43, 44)
HETEROGENEOUS_FACTORS = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 1.1, 1.5, 1.9)
FOCAL_TARGET = 1e-5
CONDITIONS = (
    ("A1", "fixed", 1.4), ("A2", "fixed", 1.5),
    ("A3", "shared", 1.5), ("A4", "per-observation", 1.5),
    ("A5", "shared", 1.0), ("A6", "shared", 2.0),
    ("A7", "per-observation", 1.0), ("A8", "per-observation", 2.0),
    ("B1", "fixed", 1.5), ("B2", "shared", 1.5),
    ("B3", "per-observation", 1.5), ("B4", "fixed", None),
)
COMPARISONS = {
    "calibration_recovery": ("A1", "A2", "A3", "A4"),
    "shared_initialization_sensitivity": ("A3", "A5", "A6"),
    "per_observation_initialization_sensitivity": ("A4", "A7", "A8"),
    "heterogeneous_cameras": ("B1", "B2", "B3", "B4"),
}
SPECTRAL_FIELDS = (
    "matrix_dimensions", "parameter_count", "rank", "nullity", "absolute_threshold",
    "minimum_singular_value", "maximum_singular_value", "condition_number", "effective_condition_number",
)


def case_matrix(seeds, budget):
    for seed in seeds:
        for condition, mode, factor in CONDITIONS:
            yield {
                "experiment": condition[0], "condition": condition, "seed": seed,
                "focal_mode": mode, "configured_initial_focal_factor": factor,
                "camera_mode": "known" if condition == "B4" else "estimated",
                "identity_ground_truth": "head-only", "joint_lm_budget": budget,
            }


def _factor(camera):
    return camera["focal_length"][0] / min(camera["image_size"])


def _camera_row(item, before, after, fit, index, reference_root):
    reference = item["camera"]
    ref_focal = reference["focal_length"][0]
    fitted_focal = after["focal_length"][0] if after is not None else None
    error = abs(fitted_focal - ref_focal) if fitted_focal is not None else None
    mismatches = None
    shared_root = None
    if after is not None:
        mismatches = [name for name in ("image_size", "rotation", "principal_point")
                      if after[name] != reference[name]]
        same_focal = (error == 0 if fit.config.focal_mode == "fixed" else error / ref_focal <= FOCAL_TARGET)
        if not same_focal:
            mismatches.append("focal_length")
        shared_root = np.array_equal(fit.final.translation[index], reference_root)
        if not shared_root:
            mismatches.append("root_translation")
    comparable = mismatches == []
    return {
        "id": item["id"], "focal_mode": fit.config.focal_mode,
        "reference_focal_length_px": ref_focal, "initial_focal_length_px": before["focal_length"][0],
        "fitted_focal_length_px": fitted_focal, "absolute_focal_error_px": error,
        "relative_focal_error": error / ref_focal if error is not None else None,
        "reference_focal_factor": _factor(reference), "initial_focal_factor": _factor(before),
        "fitted_focal_factor": _factor(after) if after is not None else None,
        "reference_translation_m": reference["translation"], "initial_translation_m": before["translation"],
        "fitted_translation_m": after["translation"] if after is not None else None,
        "reference_camera_depth_m": reference["translation"][2], "initial_camera_depth_m": before["translation"][2],
        "fitted_camera_depth_m": after["translation"][2] if after is not None else None,
        "translation_comparability": {
            "status": "unavailable" if mismatches is None else ("comparable" if comparable else "not_comparable"),
            "mismatched_fields": mismatches, "shared_root": shared_root,
            "trainable_focal_relative_tolerance": FOCAL_TARGET if fit.config.focal_mode != "fixed" else None,
        },
        "translation_l2_m": float(np.linalg.norm(np.subtract(after["translation"], reference["translation"])))
        if comparable else None,
        "camera_depth_abs_error_m": abs(after["translation"][2] - reference["translation"][2])
        if comparable else None,
    }


def add_diagnostics(record, case, truth, fit, targets):
    initial = _snapshot(fit.initial)
    final = _snapshot(fit.final) if fit.final is not None else None
    cameras = [
        _camera_row(item, initial["cameras"][i], final["cameras"][i] if final is not None else None,
                    fit, i, truth.arrays["translation"][i])
        for i, item in enumerate(truth.observations)
    ]
    reference_factors = [camera["reference_focal_factor"] for camera in cameras]
    initial_factors = [camera["initial_focal_factor"] for camera in cameras]
    final_factors = [camera["fitted_focal_factor"] for camera in cameras] if final is not None else None
    shared = fit.config.focal_mode == "shared"
    heterogeneous = not all(isclose(factor, reference_factors[0], rel_tol=1e-14)
                            for factor in reference_factors)
    compatibility = record["compatibility"]
    incompatible_shared = shared and heterogeneous
    compatible = compatibility["fully_representable"] and not incompatible_shared
    expected_incompatible = case["condition"] in ("A2", "B1", "B2")
    compatibility.update(
        fully_representable=compatible, shared_focal_heterogeneous_reference=incompatible_shared,
        status="compatible" if compatible else ("designed_incompatibility" if expected_incompatible
                                                else "unexpected_incompatibility"),
        matches_experimental_design=(compatible != expected_incompatible),
    )
    relative = [camera["relative_focal_error"] for camera in cameras] if final is not None else None
    focal = {
        "maximum_relative_error": max(relative) if relative is not None else None,
        "rms_relative_error": float(np.sqrt(np.mean(np.square(relative)))) if relative is not None else None,
        "experimental_relative_target": FOCAL_TARGET,
        "target_met": max(relative) <= FOCAL_TARGET if relative is not None else None,
        "reference_factor_stddev": float(np.std(reference_factors)),
        "fitted_factor_stddev": float(np.std(final_factors)) if final_factors is not None else None,
        "shared_initial_factor": initial_factors[0] if shared else None,
        "shared_fitted_factor": final_factors[0] if shared and final_factors is not None else None,
    }
    if final is not None:
        for observation, camera in zip(record["final"]["training_reprojection_and_nuisance_recovery"]["observations"],
                                       cameras, strict=True):
            score = observation["camera_recovery"]
            score.update(
                fixed_parameters=("rotation", "principal_point", "focal_length")
                if fit.config.focal_mode == "fixed" else ("rotation", "principal_point"),
                translation_comparability=camera["translation_comparability"],
                translation_l2_m=camera["translation_l2_m"],
                status="evaluated" if camera["translation_l2_m"] is not None else "partially_evaluated",
            )
    record.update({
        STUDY: case, "dataset_verified": True, "camera_recovery": cameras, "focal_recovery": focal,
        "reference_focal_factors": reference_factors, "initial_focal_factors": initial_factors,
        "final_focal_factors": final_factors, "identity_coefficient_norms": _identity_norms(fit),
        "parameter_snapshots": {"initial": initial, "final": final},
        "initialization": [asdict(choice) for choice in fit.initialization],
        "execution_status": "failed" if fit.status in FAILED_STATUSES else "completed",
        "converged": fit.status == "converged", "reference_targets": targets,
        "recovery": {
            "identity": targets["trainable_identity_rmse"]["met"],
            "geometry": all(targets[name]["met"] for name in ("neutral_mesh_rms_m", "facial_region_rms_m"))
            if final is not None else None,
            "focal": focal["target_met"],
            "exact_compatible_recovery": bool(compatible and focal["target_met"]
                                               and all(item["met"] for item in targets.values()))
            if final is not None else None,
        },
    })


def complete_record(record):
    """Add derived report fields using the preserved metrics and optimizer trace."""
    if not record.get("dataset_verified"):
        return
    initial, final = record["initial"], record["final"]
    snapshots = record["parameter_snapshots"]
    record["identity_error_floor"] = coefficient_floor(
        initial["identity"], final["identity"] if final is not None else None,
        record["compatibility"]["identity"]["differing_coefficients"],
    )
    record["camera_parameter_changes"] = _camera_changes(
        record["observation_ids"], snapshots["initial"], snapshots["final"],
    ) if final is not None else None
    record["optimization"]["checkpoints"] = _checkpoints(record)
    for at in ("initial", "final"):
        spectrum = record.get(f"{at}_identity_observability")
        if spectrum is None or record[STUDY]["focal_mode"] != "fixed":
            continue
        score = record[at]
        if at == "initial":
            statistics = score["reprojection"]
            rows = 2 * statistics["overall"]["eligible_count"] if statistics is not None else None
        else:
            statistics = score["training_reprojection_and_nuisance_recovery"]["aggregate"]["observed"]
            rows = 2 * statistics["valid_count"]
        for block in spectrum.values():
            if isinstance(block, dict) and "singular_values" in block:
                width = block["identity_parameter_count"]
                block.update(
                    parameter_count=width, matrix_dimensions=[rows, width],
                    absolute_threshold=spectrum["absolute_threshold"],
                    minimum_singular_value=block["singular_values"][-1],
                    maximum_singular_value=block["singular_values"][0],
                )


def _observability(spectrum):
    if spectrum is None:
        return None
    return {name: {key: block.get(key, spectrum.get(key)) for key in SPECTRAL_FIELDS}
            for name, block in spectrum.items() if isinstance(block, dict) and "rank" in block}


def _case_row(index, record):
    final = record.get("final")
    optimization = record.get("optimization") or {}
    return {
        "case": record["case"], "report_case": f"report.json#/cases/{index}", **record[STUDY],
        **{key: record.get(key) for key in DATASET_FIELDS},
        **{key: record.get(key) for key in (
            "dataset_verified", "status", "reason", "execution_status", "converged",
            "reference_focal_factors", "initial_focal_factors", "final_focal_factors", "focal_recovery",
            "recovery", "reference_targets", "observability_status", "runtime", "identity_coefficient_norms",
        )},
        "model_compatibility": (record.get("compatibility") or {}).get("status"),
        "accepted_updates": optimization.get("accepted_updates"), "rejected_trials": optimization.get("rejected_trials"),
        "metrics": _metrics(record),
        "identity_metrics": final["identity"] if final is not None else None,
        "geometry_metrics": {"full_mesh": final["neutral_geometry"], "regions": final["neutral_geometry_regions"]}
        if final is not None else None,
        "reprojection_metrics": final["training_reprojection_and_nuisance_recovery"]["aggregate"]
        if final is not None else None,
        "observability": {at: _observability(record.get(f"{at}_identity_observability"))
                          for at in ("initial", "final")},
    }


def _configuration_except_camera(config):
    varied = {"camera_mode", "camera_model", "focal_mode", "initial_focal_factor", "perspective_focal_per_short_edge"}
    return {key: value for key, value in config.items() if key not in varied}


def _paired_measurements(row):
    runtime = row["runtime"] or {}
    spectrum = row["observability"]["final"] or {}
    identity = spectrum.get("after_all_nuisance", {})
    joint = spectrum.get("joint_identity_and_focal", {})
    return row["metrics"] | {
        key: (row["focal_recovery"] or {}).get(key)
        for key in ("maximum_relative_error", "rms_relative_error")
    } | {
        "accepted_updates": row["accepted_updates"], "rejected_trials": row["rejected_trials"],
        "fit_seconds": runtime.get("total_seconds"), "case_seconds": runtime.get("case_elapsed_seconds"),
        "peak_process_rss_bytes": runtime.get("case_peak_process_rss_bytes"),
        "effective_identity_condition_number": identity.get("effective_condition_number"),
        "joint_effective_condition_number": joint.get("effective_condition_number"),
    }


def _pair(before, after, first, second):
    dataset = {field: first.get(field) == second.get(field) for field in DATASET_FIELDS}
    same_config = _configuration_except_camera(first["config"]) == _configuration_except_camera(second["config"])
    matched = bool(first.get("dataset_verified") and second.get("dataset_verified")
                   and all(dataset.values()) and same_config)
    final_a = (first.get("parameter_snapshots") or {}).get("final")
    final_b = (second.get("parameter_snapshots") or {}).get("final")
    valid = matched and final_a is not None and final_b is not None
    a, b = _paired_measurements(before), _paired_measurements(after)
    operational = {
        key: b[key] - a[key] for key in (
            "accepted_updates", "rejected_trials", "fit_seconds", "case_seconds", "peak_process_rss_bytes",
        ) if matched and a[key] is not None and b[key] is not None
    }
    deltas = None
    if valid:
        deltas = {key: b[key] - a[key] for key in a if key not in operational
                  and a[key] is not None and b[key] is not None}
        focal_a, focal_b = np.asarray(before["final_focal_factors"]), np.asarray(after["final_focal_factors"])
        depth_a, depth_b = (
            np.asarray([camera["translation"][2] for camera in final["cameras"]])
            for final in (final_a, final_b)
        )
        identity_delta = np.subtract(final_b["identity"], final_a["identity"])
        deltas.update(
            between_fit_identity_rmse=float(np.sqrt(np.mean(np.square(identity_delta)))),
            between_fit_max_relative_focal_difference=float(np.max(np.abs(focal_b - focal_a) / focal_a)),
            between_fit_max_relative_depth_coordinate_difference=float(np.max(np.abs(depth_b - depth_a) / depth_a)),
        )
    return {
        "seed": before["seed"], "before": before["report_case"], "after": after["report_case"],
        "before_status": before["status"], "after_status": after["status"], "datasets_match": dataset,
        "same_controlled_configuration": same_config, "inputs_matched": matched, "valid_metrics": valid,
        "operational_deltas_after_minus_before": operational, "metric_deltas_after_minus_before": deltas,
    }


def _historical(rows, records, historical):
    if historical is None:
        return {"status": "unavailable", "reason": "No V5.1 camera-calibration report was supplied."}
    baseline = {(item["seed"], item["camera-calibration"]["condition"]): item
                for item in historical["cases"] if "camera-calibration" in item}
    comparisons = []
    for row in rows:
        if row["condition"] not in ("A1", "A2"):
            continue
        condition = "B2" if row["condition"] == "A1" else "C2"
        old = baseline.get((row["seed"], condition))
        new = records[row["case"]]
        matches = old is not None and old.get("dataset_verified") and new.get("dataset_verified") and all(
            old.get(field) == new.get(field) for field in (*DATASET_FIELDS, "config", "model_asset_sha256", "generation")
        )
        comparisons.append({
            "seed": row["seed"], "v5_1_condition": condition, "v5_2_case": row["report_case"],
            "status": "matched" if matches else "unverified",
            "v5_1_status": old["status"] if matches else None, "v5_2_status": row["status"],
            "v5_1_metrics": _metrics(old) if matches else None, "v5_2_metrics": row["metrics"] if matches else None,
        })
    return {"status": "evaluated", "comparisons": comparisons}


def _metric_ranges(rows):
    measurements = [_paired_measurements(row) for row in rows]
    if not measurements:
        return {}
    ranges = {}
    for key in measurements[0]:
        values = [item[key] for item in measurements if item[key] is not None]
        if values:
            ranges[key] = {"count": len(values), "minimum": min(values), "maximum": max(values)}
    return ranges


def summarize(records, historical=None):
    selected = {item["case"]: item for item in records if STUDY in item}
    rows = [_case_row(index, item) for index, item in enumerate(records) if STUDY in item]
    by_condition = {(row["seed"], row["condition"]): row for row in rows}
    comparisons = []
    for comparison, conditions in COMPARISONS.items():
        for before, after in combinations(conditions, 2):
            pairs = [
                _pair(a, b, selected[a["case"]], selected[b["case"]])
                for seed in SEEDS
                if (a := by_condition.get((seed, before))) is not None
                and (b := by_condition.get((seed, after))) is not None
            ]
            comparisons.append({
                "comparison": comparison, "before_condition": before, "after_condition": after,
                "valid_count": sum(pair["valid_metrics"] for pair in pairs),
                "failed_count": sum(not pair["valid_metrics"] for pair in pairs), "pairs": pairs,
            })
    aggregates = []
    for condition, _, _ in CONDITIONS:
        cases = [row for row in rows if row["condition"] == condition]
        valid = [row for row in cases if row["execution_status"] == "completed"]
        recovered = [row for row in valid if row["recovery"]["exact_compatible_recovery"] is True]
        missed = [row for row in valid if row["recovery"]["exact_compatible_recovery"] is False]
        aggregates.append({
            "condition": condition, "cases": len(cases), "valid_count": len(valid),
            "failed_count": len(cases) - len(valid), "converged_count": sum(row["converged"] is True for row in cases),
            "budget_exhausted_count": sum(row["status"] == "budget_exhausted" for row in cases),
            "exact_compatible_recovery_count": len(recovered), "missed_recovery_count": len(missed),
            "exact_compatible_recovery_metric_ranges": _metric_ranges(recovered),
            "missed_recovery_metric_ranges": _metric_ranges(missed),
        })
    return {
        "schema_version": 1, "study": STUDY, "detail_report": "report.json", "cases": rows,
        "comparisons": comparisons, "aggregates": aggregates,
        "historical_comparison": _historical(rows, selected, historical),
    }
