"""Matched diagnostics for the V5.1 camera calibration study."""

from dataclasses import asdict
from math import isclose

import numpy as np

from ..fitting.results import ARRAY_FIELDS
from .dataset import camera_to_record
from .full_native_camera import NUISANCE_GROUPS, STUDY as V5_STUDY, _metrics


STUDY = "camera-calibration"
SEEDS = (42, 43, 44)
BUDGETS = (60, 100)
CHECKPOINTS = (10, 20, 30, 60, 100)
PREFIX_TOLERANCES = {"rel_tol": 1e-10, "abs_tol": 1e-15}
DATASET_FIELDS = ("source_dataset_id", "evaluation_dataset_id", "observation_ids")
CONDITIONS = (
    ("A", "weak-perspective", "not-applicable", None),
    ("B", "perspective", "calibrated-intrinsics", 1.4),
    ("C", "perspective", "incorrect-fixed-intrinsics", 1.5),
)
COMPARISONS = (
    ("A1", "A2", "weak_perspective_budget", "budget"),
    ("B1", "C1", "perspective_calibration_at_60", "focal_factor"),
    ("B2", "C2", "perspective_calibration_at_100", "focal_factor"),
    ("B1", "B2", "calibrated_perspective_budget", "budget"),
    ("C1", "C2", "incorrect_intrinsics_perspective_budget", "budget"),
)


def case_matrix(seeds, budgets):
    for seed in seeds:
        for label, camera, calibration, focal_factor in CONDITIONS:
            for index, budget in enumerate(budgets, start=1):
                yield {
                    "condition": f"{label}{index}", "seed": seed, "camera_model": camera,
                    "calibration_condition": calibration, "focal_factor": focal_factor,
                    "camera_mode": "estimated", "identity_ground_truth": "head-only",
                    "joint_lm_budget": budget,
                }


def _snapshot(state):
    return {name: getattr(state, name).tolist() for name in ARRAY_FIELDS} | {
        "cameras": [camera_to_record(camera) for camera in state.cameras],
    }


def _identity_norms(fit):
    active = np.isin(fit.model["identity_names"], fit.selection.identity)

    def norms(identity):
        return {
            "full_native": float(np.linalg.norm(identity)),
            "trainable": float(np.linalg.norm(identity[active])),
            "fixed": float(np.linalg.norm(identity[~active])),
        }

    return {
        "initial_l2": norms(fit.initial.identity),
        "final_l2": norms(fit.final.identity) if fit.final is not None else None,
        "change_l2": norms(fit.final.identity - fit.initial.identity) if fit.final is not None else None,
    }


def _camera_changes(ids, initial, final):
    changes = []
    for key, before, after in zip(ids, initial["cameras"], final["cameras"], strict=True):
        delta = np.subtract(after["translation"], before["translation"])
        perspective = before["type"] == "perspective"
        changes.append({
            "id": key, "translation_delta": delta.tolist(), "translation_l2": float(np.linalg.norm(delta)),
            "translation_unit": "m" if perspective else "px",
            "scale_delta_px_per_m": None if perspective else after["scale"] - before["scale"],
        })
    return changes


def _checkpoints(record):
    optimization = record["optimization"]
    updates = optimization["updates"]
    trials = {item["stage_iteration"] + 1: item for item in optimization["trials"] if item["accepted"]}
    linearizations = {item["stage_iteration"]: item for item in optimization["linearizations"]}
    losses = {item["iteration"]: item["losses"] for item in record["history"]}
    checkpoints = []
    for index in CHECKPOINTS:
        if index > len(updates):
            checkpoints.append({"accepted_update": index, "status": "unavailable", "reason": record["reason"]})
            continue
        has_snapshot = index == len(updates) and record["final"] is not None
        checkpoints.append({
            "accepted_update": index, "status": "reached", "update": updates[index - 1],
            "accepted_trial": trials[index], "objective_after_update": losses.get(index),
            "linearization_before_update": linearizations.get(index - 1),
            "linearization_after_update": linearizations.get(index),
            "recovery_metrics": {
                "status": "available", "case_pointer": "#/final",
            } if has_snapshot else {
                "status": "unavailable", "reason": "No parameter snapshot is stored at this update.",
            },
        })
    return checkpoints


def add_diagnostics(record, fit, targets):
    """Keep the entire existing trace and label the available parameter snapshots."""
    initial = _snapshot(fit.initial)
    final = _snapshot(fit.final) if fit.final is not None else None
    record.update(
        history=[asdict(item) for item in fit.history],
        initialization=[asdict(item) for item in fit.initialization],
        parameter_snapshots={"initial": initial, "final": final},
        identity_coefficient_norms=_identity_norms(fit),
        camera_parameter_changes=_camera_changes(fit.observation_ids, initial, final) if final is not None else None,
        reference_targets=targets,
    )
    record["optimization"]["checkpoints"] = _checkpoints(record)
    for at in ("initial", "final"):
        spectrum = record.get(f"{at}_identity_observability")
        if spectrum is not None:
            for name in NUISANCE_GROUPS:
                group = spectrum[name]
                group["minimum_singular_value"] = group["singular_values"][-1]
                group["maximum_singular_value"] = group["singular_values"][0]


def _differences(before, after, path, differences):
    if isinstance(before, dict) and isinstance(after, dict):
        if before.keys() != after.keys():
            differences.append({"field": path, "before_keys": sorted(before), "after_keys": sorted(after)})
        for key in sorted(before.keys() & after.keys()):
            _differences(before[key], after[key], f"{path}/{key}", differences)
        return
    if isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
        if len(before) != len(after):
            differences.append({"field": path, "before_length": len(before), "after_length": len(after)})
        for index, (left, right) in enumerate(zip(before, after)):
            _differences(left, right, f"{path}/{index}", differences)
        return
    matches = type(before) is type(after) and (
        isclose(before, after, **PREFIX_TOLERANCES) if isinstance(before, float) else before == after
    )
    if not matches:
        differences.append({"field": path, "before": before, "after": after})


def _iteration_log(record):
    # The stage-end row has empty gradient norms and is specific to the run's budget.
    return [{key: value for key, value in item.items() if key != "elapsed_seconds"}
            for item in record["history"] if item["gradient_norms"]]


def compare_prefixes(before, after):
    if before.get("optimization") is None or after.get("optimization") is None:
        return {"status": "unavailable", "reason": "A worker did not return an optimization trace."}
    first, second = before["optimization"], after["optimization"]
    differences, compared = [], {}
    for field in ("updates", "trials", "linearizations", "history"):
        left, right = (
            (_iteration_log(before), _iteration_log(after)) if field == "history" else (first[field], second[field])
        )
        count = min(len(left), len(right))
        compared[field] = {"common": count, "before": len(left), "after": len(right)}
        _differences(left[:count], right[:count], field, differences)
    for field, left, right in (
        ("initial_objective", first["initial_objective"], second["initial_objective"]),
        ("initial_parameters", before["parameter_snapshots"]["initial"], after["parameter_snapshots"]["initial"]),
        ("initialization", before["initialization"], after["initialization"]),
    ):
        _differences(left, right, field, differences)
    if before["status"] != "budget_exhausted":
        for field in ("status", "reason"):
            _differences(before[field], after[field], field, differences)
        for field, lengths in compared.items():
            _differences(lengths["before"], lengths["after"], f"{field}/length_after_early_termination", differences)
    return {"status": "discrepant" if differences else "matched", "compared": compared, "differences": differences}


def _observability(record, state):
    spectrum = record.get(f"{state}_identity_observability")
    if spectrum is None:
        return None
    return {name: {key: spectrum[name][key] for key in (
        "rank", "nullity", "minimum_singular_value", "maximum_singular_value",
        "condition_number", "effective_condition_number",
    )} | {"absolute_threshold": spectrum["absolute_threshold"]} for name in NUISANCE_GROUPS}


def _case_row(index, record):
    final = record["final"] or {}
    training = final.get("training_reprojection_and_nuisance_recovery")
    camera_parameters = {"initial_parameters", "fitted_parameters", "reference_parameters"}
    return {
        "case": record["case"], "report_case": f"report.json#/cases/{index}", **record[STUDY],
        **{key: record.get(key) for key in DATASET_FIELDS},
        "dataset_verified": record["dataset_verified"],
        "status": record["status"], "reason": record["reason"],
        "execution_status": record["execution_status"], "converged": record["converged"],
        "accepted_updates": (record.get("optimization") or {}).get("accepted_updates"),
        "representability_status": record["representability_status"],
        "metrics": _metrics(record), "identity_metrics": final.get("identity"),
        "geometry_metrics": {"full_mesh": final["neutral_geometry"],
                             "regions": final["neutral_geometry_regions"]} if final else None,
        "reprojection_metrics": training["aggregate"] if training is not None else None,
        "camera_comparison": record.get("camera_comparison"),
        "camera_recovery": [
            {"id": item["id"]} | {key: value for key, value in item["camera_recovery"].items()
                                  if key not in camera_parameters}
            for item in training["observations"]
        ] if training is not None else None,
        "observability": {at: _observability(record, at) for at in ("initial", "final")},
        "observability_status": record["observability_status"],
        "reference_targets": record["reference_targets"], "runtime": record.get("runtime"),
    }


def _controlled_config(config, varied):
    if varied == "focal_factor":
        return {key: value for key, value in config.items() if key != "perspective_focal_per_short_edge"}
    *stages, joint = config["stages"]
    return config | {"stages": [*stages, {key: value for key, value in joint.items() if key != "iterations"}]}


def _pair(before, after, before_record, after_record, varied):
    dataset_checks = dict.fromkeys(DATASET_FIELDS)
    if before["dataset_verified"] and after["dataset_verified"]:
        dataset_checks = {field: before[field] == after[field] for field in DATASET_FIELDS}
    same_config = _controlled_config(before_record["config"], varied) == _controlled_config(after_record["config"], varied)
    if None in dataset_checks.values():
        status, reason = "unavailable", "A worker did not verify its fitting dataset."
    elif not all(dataset_checks.values()):
        status, reason = "dataset_mismatch", "Source or evaluation dataset IDs or ordered observation IDs differ."
    elif not same_config:
        status, reason = "configuration_mismatch", "Settings differ outside the controlled variable."
    else:
        status, reason = "matched", None
    metrics = before["metrics"]
    return {
        "seed": before["seed"], "before": before["report_case"], "after": after["report_case"],
        "status": status, "reason": reason, "dataset_checks": dataset_checks,
        "same_controlled_configuration": same_config,
        "outcomes": {side: {key: row[key] for key in ("status", "reason", "converged", "accepted_updates")}
                     for side, row in (("before", before), ("after", after))},
        "metric_deltas": {
            key: value - metrics[key] if status == "matched" and value is not None and metrics[key] is not None else None
            for key, value in after["metrics"].items()
        },
        "optimization_prefix": compare_prefixes(before_record, after_record)
        if varied == "budget" and status == "matched" else None,
    }


def summarize(records):
    """Link all fitting cases and compute only comparisons with verified inputs."""
    indexed = {record["case"]: (index, record) for index, record in enumerate(records) if STUDY in record}
    rows = [_case_row(index, record) for index, record in indexed.values()]
    by_condition = {(row["seed"], row["condition"]): row for row in rows}
    comparisons = []
    for first, second, name, varied in COMPARISONS:
        pairs = []
        for seed in SEEDS:
            before, after = by_condition.get((seed, first)), by_condition.get((seed, second))
            if before is not None and after is not None:
                pairs.append(_pair(before, after, indexed[before["case"]][1], indexed[after["case"]][1], varied))
        comparisons.append({"comparison": name, "before_condition": first, "after_condition": second, "pairs": pairs})
    return {
        "schema_version": 1, "study": STUDY, "detail_report": "report.json", "cases": rows,
        "comparisons": comparisons, "optimization_prefix_tolerances": PREFIX_TOLERANCES,
        "historical_comparison": {
            "status": "requires_v5_report", "study": V5_STUDY, "condition": "C", "joint_lm_budget": 30,
            "reason": "Verify the same dataset IDs, observation order and fitting configuration except joint budget. "
                      "The calibrated perspective cases use a different focal factor from V5 condition C.",
        },
    }
