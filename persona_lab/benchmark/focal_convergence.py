"""Explicit V5.3 focal convergence schedules and matched scientific comparisons."""

from collections import defaultdict
from dataclasses import asdict
from itertools import combinations

import numpy as np

from . import camera_calibration, focal_estimation
from .full_native_camera import _metrics


STUDY = "focal-convergence"
STAGE_NAMES = ("pose_camera", "identity", "joint")
STRATEGIES = {
    "J100": (0, 0, 100),
    "J200": (0, 0, 200),
    "P100": (20, 0, 80),
}
V52_DATASET_COMMIT = "9875b252d9e1995cb2821cae6579d4e6dc9d19e5"
# Generated independently with V5.2's corrected-identity focal-estimation jobs
# at the commit above, on CPU float64 with the pinned GNM asset.
V52_DATASET_IDS = {
    ("A", 42): "64298dbe524a4811ba95cc380282e82bcea7f8b1908126acee5245528d48476f",
    ("A", 43): "046ff7ac40242d94bdf466e81aab91cb32f15c5b4db6d5c92e4d026a12f45097",
    ("A", 44): "4a56fef574080f4f32df41ddc4a076d8c35cac473f05dace30fa3a4b3682e28f",
    ("B", 42): "9bb9d10400fd8606ead4015af9640e63e7e0b1b275591c06bd01861098caa34f",
    ("B", 43): "c44bbdff7615ebe1b5e263988e1ede0791e908251cdc27d1f640e962f953f926",
    ("B", 44): "90725e8e684ab32996f552ca17b71405a0cc69f2bdfcf0ff0f4689ecd0ff5367",
}
V52_OBSERVATION_IDS = tuple(f"subject_0000/observation_{index:04}" for index in range(9))
DATASET_FIELDS = camera_calibration.DATASET_FIELDS
PREFIX_TOLERANCES = camera_calibration.PREFIX_TOLERANCES


def case_matrix(seeds, strategies):
    for base in focal_estimation.case_matrix(seeds, STRATEGIES["J100"][-1]):
        if base["focal_mode"] == "fixed" or base["condition"] == "B2":
            continue
        del base["joint_lm_budget"]
        for strategy in strategies:
            yield base | {
                "strategy": strategy, "camera_model": "perspective",
                "requested_stage_budgets": dict(zip(STAGE_NAMES, STRATEGIES[strategy], strict=True)),
            }


def dataset_comparison(record):
    case = record[STUDY]
    expected_id = V52_DATASET_IDS[case["experiment"], case["seed"]]
    expected = dict(zip(DATASET_FIELDS, (expected_id, expected_id, V52_OBSERVATION_IDS), strict=True))
    checks = {
        field: tuple(record[field]) == value if field == "observation_ids" else record[field] == value
        for field, value in expected.items()
    }
    return {
        "status": "matched" if all(checks.values()) else "mismatch",
        "reference_commit": V52_DATASET_COMMIT, "expected": expected, "matches": checks,
    }


def _boundary(boundary):
    return {
        "physical_parameters": camera_calibration._snapshot(boundary.state) if boundary.state is not None else None,
        "optimizer_coordinates": boundary.coordinates,
        "complete_objective": asdict(boundary.objective) if boundary.objective is not None else None,
        "error": boundary.error,
    }


def _stage_rows(fit):
    summaries = {stage.name: asdict(stage) for stage in fit.stages}
    diagnostics = {stage.name: stage for stage in fit.stage_diagnostics}
    rows = []
    for stage in fit.config.stages:
        row = summaries.get(stage.name)
        if row is None:
            row = dict(name=stage.name, active_groups=None, steps=None, status="not_run", reason=fit.reason,
                       initial_loss=None, final_loss=None)
        trace = diagnostics.get(stage.name)
        row.update(
            requested_updates=stage.iterations,
            objective_evaluations=trace.objective_evaluations if trace is not None else (0 if row["status"] == "skipped" else None),
            jacobian_evaluations=trace.jacobian_evaluations if trace is not None else (0 if row["status"] == "skipped" else None),
            rejected_trials=trace.rejected_trials if trace is not None else (0 if row["status"] == "skipped" else None),
            before=_boundary(trace.before) if trace is not None else None,
            after=_boundary(trace.after) if trace is not None else None,
            boundary_objective_evaluations=sum(point.state is not None for point in (trace.before, trace.after))
            if trace is not None else 0,
            initial_damping=trace.initial_damping if trace is not None else None,
            final_damping=trace.final_damping if trace is not None else None,
            accepted_states=[asdict(item) | {"camera_depths_m": [xyz[2] for xyz in item.camera_translations]}
                             for item in trace.accepted_states] if trace is not None else [],
        )
        rows.append(row)
    return rows


def _transition(stages):
    calibration, _, joint = stages
    before, after = calibration["after"], joint["before"]
    if before is None or after is None:
        return {"status": "not_reached", "reason": calibration["reason"]}
    a, b = before["physical_parameters"], after["physical_parameters"]
    first, second = before["complete_objective"], after["complete_objective"]
    if a is None or b is None or first is None or second is None:
        return {"status": "unavailable", "reason": before["error"] or after["error"]}
    physical = {key: value == b[key] for key, value in a.items()}
    coordinates = {key: value == after["optimizer_coordinates"][key]
                   for key, value in before["optimizer_coordinates"].items()}
    return {
        "status": "continuous" if all(physical.values()) and all(coordinates.values()) and first == second
        else "discontinuous",
        "physical_parameter_matches": physical, "optimizer_coordinate_matches": coordinates,
        "complete_objective_matches": first == second, "calibration_end_objective": first,
        "joint_start_objective": second, "objective_delta": second["total"] - first["total"],
        "damping": {
            "behavior": "initialized_from_config_for_each_lm_stage",
            "calibration_final": calibration["final_damping"], "joint_initial": joint["initial_damping"],
        },
    }


def add_diagnostics(record, fit):
    record["v52_dataset_comparison"] = dataset_comparison(record)
    record["recovery"]["reprojection"] = record["reference_targets"]["reprojection_rms_px"]["met"]
    stages = _stage_rows(fit)
    record["stage_diagnostics"] = stages
    record["stage_transition"] = _transition(stages) if fit.config.stages[0].iterations else None
    record["optimization"]["initialization_objective_evaluations"] = fit.optimization.objective_evaluations - sum(
        stage["objective_evaluations"] or 0 for stage in stages
    )
    record["optimization"]["boundary_objective_evaluations"] = sum(
        stage["boundary_objective_evaluations"] for stage in stages
    )


def camera_diagnostics(cameras):
    if not cameras or cameras[0]["fitted_camera_depth_m"] is None:
        return None
    depths = [row["fitted_camera_depth_m"] - row["reference_camera_depth_m"] for row in cameras]
    translations = [row["translation_l2_m"] for row in cameras if row["translation_l2_m"] is not None]
    focal = [row["relative_focal_error"] for row in cameras]
    return {
        "maximum_relative_focal_error": max(focal),
        "camera_depth_coordinate_rms_difference_m": float(np.sqrt(np.mean(np.square(depths)))),
        "comparable_translation_observations": len(translations),
        "comparable_translation_rms_error_m": float(np.sqrt(np.mean(np.square(translations)))) if translations else None,
    }


def _compact_calibration(record):
    states = (record.get("calibration_diagnostics") or {}).get("states")
    if states is None:
        return None
    return {
        name: ({"status": state["status"], "reason": state.get("reason")} if state["status"] != "evaluated" else {
            "status": "evaluated",
            "clean_reprojection_rms_px": state["reprojection"]["clean"]["rms_px"],
            "head_pose_rms_deg": state["head_pose_rms_deg"],
            "trainable_identity_rmse": state["identity"]["trainable"]["rmse"],
            "full_native_expression_rmse": state["expression"]["global"]["rmse"],
            **state["camera_recovery"],
        })
        for name, state in states.items()
    }


def _case_row(index, record):
    row = focal_estimation._case_row(index, record, study=STUDY)
    stages = record.get("stage_diagnostics") or []
    training = (record.get("final") or {}).get("training_reprojection_and_nuisance_recovery")
    return row | {
        "stages": [{key: item[key] for key in (
            "name", "active_groups", "requested_updates", "steps", "rejected_trials", "status", "reason",
            "initial_loss", "final_loss", "objective_evaluations", "jacobian_evaluations",
        )} for item in stages] if stages else None,
        "v52_dataset_comparison": (record.get("v52_dataset_comparison") or {}).get("status"),
        "stage_transition_status": (record.get("stage_transition") or {}).get("status"),
        "calibration_diagnostics": _compact_calibration(record),
        "camera_recovery": camera_diagnostics(record.get("camera_recovery")),
        "expression_recovery": training["expression"] if training is not None else None,
        "pose_recovery": [{"id": item["id"], **item["pose_recovery"]}
                          for item in training["observations"]] if training is not None else None,
        "observability": row["observability"] | {
            "post_calibration": focal_estimation._observability(record.get("post_calibration_identity_observability")),
        },
        "peak_memory_bytes": (record.get("runtime") or {}).get("case_peak_process_rss_bytes"),
    }


def _schedule_independent_config(record):
    config = record["config"]
    return config | {"stages": [
        {key: value for key, value in stage.items() if key != "iterations"} for stage in config["stages"]
    ]}


def _measurements(row):
    return focal_estimation._paired_measurements(row) | (row["camera_recovery"] or {})


def _recovered(row):
    return (row["recovery"] or {}).get("exact_compatible_recovery")


def _pair(before, after, records, *, varied):
    a, b = records[before["case"]], records[after["case"]]
    same_dataset = {key: a.get(key) == b.get(key) for key in DATASET_FIELDS}
    config_a, config_b = a["config"], b["config"]
    if varied == "strategy":
        config_a, config_b = _schedule_independent_config(a), _schedule_independent_config(b)
    elif varied == "initial_focal_factor":
        config_a, config_b = ({key: value for key, value in config.items() if key != varied}
                              for config in (config_a, config_b))
    same_config = config_a == config_b
    same_model = a.get("model_asset_sha256") == b.get("model_asset_sha256") and a.get("model_asset_sha256") is not None
    dataset_verified = before["dataset_verified"] and after["dataset_verified"]
    if varied == "reference_focal_schedule":
        satisfied = all(row["v52_dataset_comparison"] == "matched" for row in (before, after))
        satisfied &= same_dataset["observation_ids"]
    else:
        satisfied = all(same_dataset.values()) and a.get("generation") == b.get("generation")
    matched = bool(dataset_verified and same_config and same_model and satisfied)
    snapshots = [(item.get("parameter_snapshots") or {}).get("initial") for item in (a, b)]
    same_initial = snapshots[0] == snapshots[1] if all(state is not None for state in snapshots) else None
    if varied == "strategy":
        matched &= same_initial is True
    first, second = _measurements(before), _measurements(after)
    recovered_a, recovered_b = _recovered(before), _recovered(after)
    return {
        "before": before["report_case"], "after": after["report_case"],
        "seed": before["seed"], "focal_mode": before["focal_mode"],
        "before_status": before["status"], "after_status": after["status"],
        "same_dataset": same_dataset, "same_model": same_model,
        "same_controlled_configuration": same_config, "same_physical_initialization": same_initial,
        "inputs_matched": matched,
        "before_exact_recovery": recovered_a, "after_exact_recovery": recovered_b,
        "recovery_regression": recovered_a is True and recovered_b is not True if matched else None,
        "additional_recovery": recovered_a is not True and recovered_b is True if matched else None,
        "after_minus_before": {key: second[key] - value for key, value in first.items()
                               if value is not None and second.get(key) is not None} if matched else None,
    }


def _budget_diagnostics(before, after):
    result = camera_calibration.compare_prefixes(before, after)
    if result["status"] == "unavailable":
        return {"common_prefix": result, "extension": None}
    first, second = before["stage_diagnostics"][-1], after["stage_diagnostics"][-1]
    left, right = first["accepted_states"], second["accepted_states"]
    camera_calibration._differences(left, right[:len(left)], "accepted_parameter_states", result["differences"])
    for field, counts in result["compared"].items():
        if counts["after"] < counts["before"]:
            result["differences"].append({"field": field, "before_length": counts["before"], "after_length": counts["after"]})
    result["status"] = "discrepant" if result["differences"] else "matched"
    tail = right[len(left):]
    start_objective = first["final_loss"]
    return {
        "common_prefix": result,
        "extension": {
            "additional_accepted_updates": len(tail),
            "objective_at_j100_endpoint": start_objective,
            "objective_at_j200_endpoint": second["final_loss"],
            "continued_objective_decrease": any(item["objective"] < start_objective for item in tail)
            if start_objective is not None else None,
            "j200_numerical_termination": after["status"],
            "j200_termination_reason": after["reason"],
            "j200_exact_recovery": after["recovery"]["exact_compatible_recovery"],
        },
    }


def _historical(rows, records, historical):
    prior = {} if historical is None else {
        (case[focal_estimation.STUDY]["seed"], case[focal_estimation.STUDY]["condition"]): case
        for case in historical["cases"]
    }
    comparisons = []
    for row in rows:
        if row["strategy"] != "J100":
            continue
        new = records[row["case"]]
        old = prior.get((row["seed"], row["condition"]))
        comparison = {
            "case": row["report_case"], "seed": row["seed"], "v52_condition": row["condition"],
            "verified_base_dataset": row["v52_dataset_comparison"],
        }
        if old is None:
            comparison.update(status="unavailable", reason="Archived V5.2 fit record unavailable")
        else:
            fields = (*DATASET_FIELDS, "config", "generation", "model_asset_sha256")
            checks = {key: old.get(key) == new.get(key) for key in fields}
            trajectories = None
            differences = []
            if all(checks.values()) and old.get("dataset_verified") and new.get("dataset_verified"):
                for field in ("parameter_snapshots", "status", "reason"):
                    camera_calibration._differences(old.get(field), new.get(field), field, differences)
                camera_calibration._differences(_metrics(old), row["metrics"], "recovery_metrics", differences)
                if old.get("optimization") is not None and new.get("optimization") is not None:
                    trajectories = camera_calibration.compare_prefixes(old, new)
                    for field, counts in trajectories["compared"].items():
                        camera_calibration._differences(
                            counts["before"], counts["after"], f"trajectory_length/{field}", differences,
                        )
                status = "unavailable" if trajectories is None else (
                    "matched" if not differences and trajectories["status"] == "matched" else "discrepant"
                )
            else:
                status = "input_mismatch"
            comparison.update(status=status, input_checks=checks, snapshot_and_outcome_differences=differences,
                              trajectory=trajectories,
                              numerical_environments={label: {key: (item.get("runtime") or {}).get(key)
                                                             for key in ("python", "numpy", "torch")}
                                                      for label, item in (("v5.2", old), ("v5.3", new))})
        comparisons.append(comparison)
    return {
        "archived_report_supplied": historical is not None,
        "verified_dataset_reference_commit": V52_DATASET_COMMIT,
        "comparisons": comparisons,
    }


def summarize(records, strategies, historical=None):
    rows = [_case_row(index, record) for index, record in enumerate(records)]
    by_case = {record["case"]: record for record in records}
    by_control = defaultdict(dict)
    by_factor = defaultdict(dict)
    by_experiment = defaultdict(dict)
    for row in rows:
        experiment, seed, mode, factor, strategy = (
            row[key] for key in ("experiment", "seed", "focal_mode", "configured_initial_focal_factor", "strategy")
        )
        by_control[experiment, seed, mode, factor][strategy] = row
        if experiment == "A":
            by_factor[seed, mode, strategy][factor] = row
        if mode == "per-observation" and factor == 1.5:
            by_experiment[seed, strategy][experiment] = row

    budget, progressive = [], []
    for cases in by_control.values():
        if "J100" in cases and "J200" in cases:
            first, second = cases["J100"], cases["J200"]
            pair = _pair(first, second, by_case, varied="strategy")
            budget.append(pair | (_budget_diagnostics(by_case[first["case"]], by_case[second["case"]])
                                  if pair["inputs_matched"] else {"common_prefix": None, "extension": None}))
        if "J100" in cases and "P100" in cases:
            progressive.append(_pair(cases["J100"], cases["P100"], by_case, varied="strategy"))

    aggregates = []
    for strategy in strategies:
        cases = [row for row in rows if row["strategy"] == strategy]
        pairs = {"J100": (), "J200": budget, "P100": progressive}[strategy]
        aggregates.append({
            "strategy": strategy, "case_count": len(cases),
            "exact_compatible_recovery_count": sum(_recovered(row) is True for row in cases),
            "missed_recovery_count": sum(_recovered(row) is False for row in cases),
            "unavailable_recovery_count": sum(_recovered(row) is None for row in cases),
            "regression_count_against_j100": sum(pair["recovery_regression"] is True for pair in pairs)
            if "J100" in strategies else None,
            "converged_count": sum(row["converged"] is True for row in cases),
            "completed_count": sum(row["execution_status"] == "completed" for row in cases),
            "failed_count": sum(row["execution_status"] == "failed" for row in cases),
            "budget_exhausted_count": sum(row["status"] == "budget_exhausted" for row in cases),
            "status_counts": {status: sum(row["status"] == status for row in cases)
                              for status in sorted({row["status"] for row in cases})},
            "metric_ranges": focal_estimation._metric_ranges(cases),
        })
    return {
        "schema_version": 1, "study": STUDY, "detail_report": "report.json",
        "strategy_schedules": {key: dict(zip(STAGE_NAMES, budget, strict=True)) for key, budget in STRATEGIES.items()},
        "selected_strategies": strategies, "cases": rows, "aggregates": aggregates,
        "comparisons": {
            "budget_isolation": budget, "progressive_calibration": progressive,
            "initialization_sensitivity": [
                {"strategy": strategy, "focal_mode": mode, "initial_factors": [first, second]}
                | _pair(cases[first], cases[second], by_case, varied="initial_focal_factor")
                for (_, mode, strategy), cases in by_factor.items() for first, second in combinations(sorted(cases), 2)
            ],
            "uniform_vs_heterogeneous": [
                {"strategy": strategy} | _pair(cases["A"], cases["B"], by_case, varied="reference_focal_schedule")
                for (_, strategy), cases in by_experiment.items() if cases.keys() == {"A", "B"}
            ],
        },
        "optimization_prefix_tolerances": PREFIX_TOLERANCES,
        "historical_comparison": _historical(rows, by_case, historical),
    }
