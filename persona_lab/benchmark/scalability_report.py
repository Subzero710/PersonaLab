"""V6 summary tables and comparisons of independently executed reports."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np

from .scalability_cases import BATCHING_COUNT, BATCHING_SEED, BASE_COMMIT, PRIMARY_BATCH_SIZE, SCHEMA_VERSION, STUDY
from .storage import is_sha256, json_bytes, read_json, write_json


TOLERANCES = {
    "coefficients": (1e-7, 1e-6), "rotation_rad": (1e-8, 1e-6),
    "geometry_m": (1e-8, 1e-6), "focal_px": (1e-5, 1e-8),
    "reprojection_px": (1e-6, 1e-6), "objective": (1e-14, 1e-6), "spectra": (1e-10, 1e-6),
}
SPECTRAL_BLOCKS = (
    "fixed_nuisance", "after_expression", "after_expression_and_pose", "after_all_nuisance",
    "joint_identity_and_focal", "conditional_focal",
)
DATASET_FIELDS = (
    "source_dataset_id", "dataset_id", "content_sha256", "identity_sha256", "fitter_input_sha256",
    "reference_files_sha256", "observation_ids", "actual_observations",
)


def _paired_rows(records):
    identifiers, reusable = set(), {}
    for index, row in enumerate(records):
        raw = row
        if row["case"] in identifiers:
            raise ValueError(f"Duplicate case identifier {row['case']}")
        identifiers.add(row["case"])
        if "reuse_case" in row:
            primary = reusable.get(row["reuse_case"])
            if primary is None or primary["experiment"] != "primary" or row["report_case"] != primary["report_case"]:
                raise ValueError(f"Invalid primary case reference for {row['case']}")
            for key in ("seed", "requested_observations", "requested_device", "batch_size", "dtype"):
                if row[key] != primary[key]:
                    raise ValueError(f"Incompatible reused case field {key}")
            row = primary | row
        elif row["report_case"] != f"report.json#/cases/{index}":
            raise ValueError(f"Invalid detailed report pointer for {row['case']}")
        elif row["experiment"] == "primary" and (
            row["seed"], row["requested_observations"], row["batch_size"]
        ) == (BATCHING_SEED, BATCHING_COUNT, row["batch_size"] if row["dtype"] == "float64" else PRIMARY_BATCH_SIZE):
            reusable[row["case"]] = row
        yield raw, row


def _ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator is not None and denominator > 0 else None


def _measurements(row):
    runtime = row.get("optimizer_runtime", {})
    seconds = row.get("timing", {})
    memory = row.get("memory", {})
    operations = runtime.get("operations", {})
    operation_seconds = operations.get("seconds", {})
    optimization = row.get("optimization", {})
    elapsed = seconds.get("optimization")
    jacobian = operation_seconds.get("observation_jacobian_batch")
    traced = runtime.get("stage_trace_seconds")
    instrumentation = operations.get("host_instrumentation_seconds")
    count = row.get("actual_observations") or 0
    evaluations = optimization.get("objective_evaluations") or 0
    return {
        "optimization_seconds": elapsed,
        "stage_trace_seconds": traced,
        "operation_instrumentation_host_seconds": instrumentation,
        "optimization_seconds_per_linearization": _ratio(elapsed, optimization.get("jacobian_evaluations")),
        "time_per_observation_seconds": _ratio(elapsed, row.get("actual_observations")),
        "time_per_accepted_update_seconds": _ratio(elapsed, optimization.get("accepted_updates")),
        "jacobian_seconds": jacobian,
        "jacobian_seconds_per_observation_per_linearization": _ratio(
            jacobian, operations.get("calls", {}).get("observation_jacobian"),
        ),
        "linearization_including_normal_blocks_seconds": runtime.get("jacobian_seconds"),
        "objective_seconds": runtime.get("objective_seconds"),
        "objective_seconds_per_observation_per_evaluation": _ratio(runtime.get("objective_seconds"), count * evaluations),
        "schur_solve_including_damping_seconds": runtime.get("linear_solve_seconds"),
        "column_equilibration_seconds": operation_seconds.get("column_equilibration"),
        "initial_observability_seconds": seconds.get("initial_observability"),
        "final_observability_seconds": seconds.get("final_observability"),
        "evaluation_seconds": seconds.get("evaluation"),
        "reference_precision_seconds": seconds.get("reference_precision"),
        "fit_snapshot_serialization_seconds": seconds.get("fit_record_serialization"),
        "serialization_seconds": seconds.get("result_serialization"),
        "peak_ram_bytes": memory.get("peak_process_rss_bytes"),
        "peak_vram_allocated_bytes": memory.get("cuda_peak_allocated_bytes"),
        "peak_vram_reserved_bytes": memory.get("cuda_peak_reserved_bytes"),
        "case_result_bytes": row.get("case_result_bytes"),
    }


def _spectral_summary(spectrum):
    if spectrum is None:
        return None
    return {name: {key: value for key, value in spectrum[name].items()
                   if key in ("matrix_dimensions", "compressed_dimensions", "rank", "nullity",
                              "absolute_threshold", "condition_number", "effective_condition_number")}
            for name in SPECTRAL_BLOCKS}


def _compact(row):
    optimization = row.get("optimization") or {}
    final = row.get("final")
    statuses = row["observability_status"]
    return {name: row.get(name) for name in (
        "case", "experiment", "seed", "requested_observations", "actual_observations", "dataset_id",
        "requested_device", "actual_execution_device", "dtype", "batch_size", "focal_mode",
        "requested_stage_budgets", "reuse_case", "termination_status", "execution_status", "converged",
        "recovery", "reference_targets", "focal_recovery", "reason", "failure", "report_case", "spectra_requested",
        "worker_started", "actual_dtype", "pixel_representation", "precision_source_dataset_id",
    )} | {
        "accepted_updates": optimization.get("accepted_updates"), "rejected_trials": optimization.get("rejected_trials"),
        "last_accepted_evaluation": row.get("last_accepted_evaluation"),
        "kernel_profile": row.get("kernel_profile"),
        "reference_precision": row.get("reference_precision"),
        "objective_evaluations": optimization.get("objective_evaluations"),
        "jacobian_evaluations": optimization.get("jacobian_evaluations"),
        "cholesky_factorization_failures": optimization.get("cholesky_factorization_failures"),
        "final_gradient_status": (optimization.get("final_gradient") or {}).get("status"),
        "camera_metrics": row.get("camera_metrics"),
        "identity_metrics": final["identity"] if final is not None else None,
        "geometry_metrics": final["neutral_geometry"] if final is not None else None,
        "reprojection_metrics": final["training_reprojection_and_nuisance_recovery"]["aggregate"]
        if final is not None else None,
        "expression_metrics": final["training_reprojection_and_nuisance_recovery"]["expression"]
        if final is not None else None,
        "observability_status": statuses,
        "observability_complete": all(item["status"] == "evaluated" for item in statuses.values())
        if row["spectra_requested"] else None,
        "observability": {at: _spectral_summary(row.get(f"{at}_identity_observability"))
                          for at in ("initial", "final")},
        "timing": row.get("timing"), "memory": row.get("memory"), "measurements": _measurements(row),
    }


def _statistics(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {"count": len(values), "mean": mean(values), "median": median(values),
            "minimum": min(values), "maximum": max(values)}


def _aggregates(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["experiment"] == "primary":
            groups[row["requested_device"], row["requested_observations"]].append(row)
    return [{
        "requested_device": device, "requested_observations": count, "case_count": len(group),
        "completed_count": sum(row["execution_status"] == "completed" for row in group),
        "failed_count": sum(row["execution_status"] == "failed" for row in group),
        "unexecuted_count": sum(row["execution_status"] == "not_run" for row in group),
        "infrastructure_failure_count": sum(row["failure"] is not None for row in group),
        "observability_failure_count": sum(
            any(item["status"] == "failed" for item in row["observability_status"].values()) for row in group
        ),
        "converged_count": sum(row["converged"] is True for row in group),
        "budget_exhausted_count": sum(row["termination_status"] == "budget_exhausted" for row in group),
        "exact_recovery_count": sum(bool((row["recovery"] or {}).get("exact_compatible_recovery")) for row in group),
        "accepted_updates": _statistics(row["accepted_updates"] for row in group),
        "measurements": {name: _statistics(row["measurements"][name] for row in group)
                         for name in group[0]["measurements"]},
    } for (device, count), group in sorted(groups.items())]


def _numeric(before, after, tolerance):
    atol, rtol = TOLERANCES[tolerance]
    result = {"absolute_tolerance": atol, "relative_tolerance": rtol}
    if before is None or after is None:
        return result | {"matches": False, "reason": "Missing values"}
    left, right = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    if left.shape != right.shape:
        return result | {"matches": False, "reason": "Shapes differ"}
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return result | {"matches": False, "reason": "Nonfinite values"}
    delta = np.abs(left - right)
    scale = np.maximum(np.abs(left), np.abs(right))
    relative = np.divide(delta, scale, out=np.zeros_like(delta), where=scale != 0)
    return result | {
        "matches": bool(np.all(delta <= atol + rtol * scale)),
        "maximum_absolute_difference": float(delta.max(initial=0.0)),
        "maximum_relative_difference": float(relative.max(initial=0.0)),
        "compared_coordinates": left.size,
    }


def _parameters(before, after):
    result = {key: _numeric(before.get(key), after.get(key), tolerance) for key, tolerance in (
        ("identity", "coefficients"), ("expression", "coefficients"), ("rotations", "rotation_rad"),
        ("translation", "geometry_m"),
    )}
    cameras = before["cameras"], after["cameras"]
    labels = [[(item["type"], item["image_size"]) for item in rows] for rows in cameras]
    result["camera_model_and_image_size"] = {"matches": labels[0] == labels[1]}
    for key, tolerance in (
        ("translation", "geometry_m"), ("rotation", "rotation_rad"),
        ("focal_length", "focal_px"), ("principal_point", "reprojection_px"),
    ):
        result[f"camera_{key}"] = _numeric(*([item[key] for item in rows] for rows in cameras), tolerance)
    return result


def _reprojection(left, right):
    training = [row["training_reprojection_and_nuisance_recovery"] for row in (left, right)]
    result = {}
    for kind in ("observed", "clean"):
        records = [[row["aggregate"][kind], *(item["reprojection"][kind] for item in row["observations"])]
                   for row in training]
        counts = [[(row["valid_count"], row["excluded_count"]) for row in rows] for rows in records]
        result[f"{kind}_counts"] = {"matches": counts[0] == counts[1]}
        result[kind] = _numeric(*([
            [row["mean_px"], row["median_px"], row["rms_px"]] for row in rows
        ] for rows in records), "reprojection_px")
    return result


def _objective(before, after, name):
    values = [row.get("optimization", {}).get(name) for row in (before, after)]
    if all(v is not None for v in values) and values[0].keys() == values[1].keys():
        return _numeric(*([row[key] for key in sorted(row)] for row in values), "objective")
    return {"matches": False, "reason": "Objective values unavailable or incompatible"}


def _compare_spectra(left, right, at):
    statuses = [row["observability_status"][at]["status"] for row in (left, right)]
    result = {"statuses": statuses}
    if statuses != ["evaluated", "evaluated"]:
        return result | {"status": "not_requested" if statuses == ["not_requested"] * 2 else "unavailable"}
    spectra = [row[f"{at}_identity_observability"] for row in (left, right)]
    blocks = {}
    for name in SPECTRAL_BLOCKS:
        first, second = (spectrum[name] for spectrum in spectra)
        blocks[name] = {
            "ranks": [first["rank"], second["rank"]], "nullities": [first["nullity"], second["nullity"]],
            "rank_and_shape_match": all(first[field] == second[field]
                                        for field in ("rank", "nullity", "matrix_dimensions", "compressed_dimensions")),
            "singular_values": _numeric(first["singular_values"], second["singular_values"], "spectra"),
            "threshold": _numeric(first["absolute_threshold"], second["absolute_threshold"], "spectra"),
        }
    consistent = all(block["rank_and_shape_match"] and block["singular_values"]["matches"]
                     and block["threshold"]["matches"] for block in blocks.values())
    return result | {"status": "consistent" if consistent else "disagreement", "blocks": blocks}


def _valid_fingerprints(row):
    files = row.get("reference_files_sha256")
    return (
        row.get("dataset_verified") is True
        and all(is_sha256(row.get(name)) for name in DATASET_FIELDS[:5])
        and isinstance(files, dict) and set(files) == {"manifest.json", "subject_0000.npz", "topology.npz"}
        and all(is_sha256(value) for value in files.values())
        and row.get("actual_observations") == row["requested_observations"] == len(row.get("observation_ids") or ())
    )


def compare_cases(before, after, *, different_batch=False):
    """Validate comparable inputs before measuring any numerical difference."""
    runtime = [_measurements(row) for row in (before, after)]
    result = {
        "case_ids": [before["case"], after["case"]],
        "report_cases": [before["report_case"], after["report_case"]],
        "requested_devices": [before["requested_device"], after["requested_device"]],
        "batch_sizes": [before["batch_size"], after["batch_size"]],
        "recovery": [row.get("recovery") for row in (before, after)],
        "termination_statuses": [row["termination_status"] for row in (before, after)],
        "failures": [row.get("failure") for row in (before, after)],
        "accepted_updates": [row.get("optimization", {}).get("accepted_updates") for row in (before, after)],
        "rejected_trials": [row.get("optimization", {}).get("rejected_trials") for row in (before, after)],
        "measurements": runtime,
        "camera_metrics": [row.get("camera_metrics") for row in (before, after)],
    }
    first, second = (row["config"].copy() for row in (before, after))
    if different_batch:
        del first["batch_size"], second["batch_size"]
    checks = {
        key: before.get(key) == after.get(key) for key in ("seed", "requested_observations", "dtype", "spectra_requested")
    } | {"configuration": json_bytes(first) == json_bytes(second)}
    if not all(checks.values()):
        return result | {"status": "incompatible_configuration", "compatibility": checks}
    if not all(_valid_fingerprints(row) for row in (before, after)):
        return result | {"status": "unavailable", "reason": "Both cases require a verified reference dataset",
                         "compatibility": checks}
    checks.update({key: before[key] == after[key] for key in DATASET_FIELDS})
    if not all(checks.values()):
        return result | {"status": "incompatible_dataset", "compatibility": checks}
    checks.update({key: before[key] == after[key] for key in ("model", "selection", "gauge")
                   if before.get(key) is not None and after.get(key) is not None})
    if not all(checks.values()):
        return result | {"status": "incompatible_configuration", "compatibility": checks}
    result["compatibility"] = checks
    if all(row["execution_status"] == "completed" for row in (before, after)):
        result["optimization_speedup_first_over_second"] = _ratio(
            runtime[0]["optimization_seconds"], runtime[1]["optimization_seconds"],
        )
        result["worker_speedup_first_over_second"] = _ratio(
            *[row.get("timing", {}).get("worker_wall_seconds") for row in (before, after)],
        )
    result["observability"] = {at: _compare_spectra(before, after, at) for at in ("initial", "final")}
    snapshots = [row.get("parameter_snapshots", {}) for row in (before, after)]
    comparisons = {}
    for at in ("initial", "final"):
        states = [item.get(at) for item in snapshots]
        if all(state is not None for state in states):
            comparisons[at] = _parameters(*states)
            comparisons[at]["objective"] = _objective(before, after, f"{at}_objective")
    result["numerical_comparisons"] = comparisons
    finals = [row.get("final") for row in (before, after)]
    if not all(final is not None for final in finals) or set(comparisons) != {"initial", "final"}:
        return result | {"status": "unavailable", "reason": "Final evaluated fits are required on both sides"}
    comparisons["final"]["neutral_mesh"] = _numeric(*[final["neutral_vertices_m"] for final in finals], "geometry_m")
    comparisons["final"].update(_reprojection(*finals))
    comparisons["final"]["termination_status"] = {
        "matches": before["termination_status"] == after["termination_status"],
    }
    matches = (
        all(item["matches"] for table in comparisons.values() for item in table.values())
        and all(item["status"] != "disagreement" for item in result["observability"].values())
    )
    result["both_meet_exact_recovery_targets"] = all(
        (row.get("recovery") or {}).get("exact_compatible_recovery") is True for row in (before, after)
    )
    return result | {"status": "consistent" if matches else "disagreement"}


def _scaling(rows):
    primary = defaultdict(dict)
    for row in rows:
        if row["experiment"] == "primary":
            primary[row["requested_device"], row["seed"]][row["requested_observations"]] = row
    pairs = []
    for (device, seed), group in primary.items():
        counts = sorted(group)
        for lower, upper in zip(counts, counts[1:]):
            first, second = group[lower], group[upper]
            identity_and_order = (
                _valid_fingerprints(first) and _valid_fingerprints(second)
                and first["source_dataset_id"] == second["source_dataset_id"]
                and first["identity_sha256"] == second["identity_sha256"]
                and first["observation_ids"] == second["observation_ids"][:lower]
                and first["config"] == second["config"]
            )
            a, b = (row["measurements"] for row in (first, second))
            pairs.append({
                "device": device, "seed": seed, "observations": [lower, upper],
                "dataset_prefix_and_config_verified": identity_and_order,
                "accepted_updates": [row.get("accepted_updates") for row in (first, second)],
                "observations_ratio": upper / lower,
                "measured_growth": {key: _ratio(b[key], a[key]) if identity_and_order else None for key in (
                    "optimization_seconds", "optimization_seconds_per_linearization",
                    "jacobian_seconds_per_observation_per_linearization",
                    "peak_ram_bytes", "peak_vram_allocated_bytes", "peak_vram_reserved_bytes",
                )},
            })
    return pairs


def _batching(rows):
    cases = defaultdict(dict)
    for row in rows:
        if row["experiment"] == "batching":
            cases[row["requested_device"]][row["batch_size"]] = row
    return [compare_cases(group[4], row, different_batch=True)
            for group in cases.values() for size, row in sorted(group.items()) if size != 4]


def write_reports(output, report, records):
    """Stream scientific records to disk, keeping only summary and batching data."""
    rows, scaling, batching = [], [], []
    serialization_seconds = 0.0
    detailed = output / "report.json"
    temporary = detailed.with_suffix(".tmp")
    with temporary.open("wb") as destination:
        header = json_bytes(report)
        destination.write(header[:-1] + b',\n"cases":[\n')
        for raw, row in _paired_rows(records):
            started = perf_counter()
            if rows:
                destination.write(b",\n")
            destination.write(json_bytes(raw))
            serialization_seconds += perf_counter() - started
            summary = _compact(row)
            rows.append(summary)
            if row["experiment"] == "primary":
                scaling.append(summary | {name: row.get(name) for name in (
                    *DATASET_FIELDS, "config", "dataset_verified",
                )})
            else:
                batching.append(row)
        destination.write(b"\n]}\n")
    temporary.replace(detailed)
    summary = {
        "schema_version": SCHEMA_VERSION, "study": STUDY, "requested_device": report["requested_device"],
        "report_case_serialization_seconds": serialization_seconds, "report_size_bytes": detailed.stat().st_size,
        "cases": rows, "primary_by_observation_count": _aggregates(rows),
        "batching_consistency": _batching(batching), "matched_observation_count_changes": _scaling(scaling),
    }
    write_json(output / "summary.json", summary)
    return summary


def _validate_report(report, device_type):
    requested = report.get("requested_device", "")
    if report.get("kernel_profiling_requested"):
        raise ValueError("Kernel-profile runs are excluded from timing comparisons")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("requested_identity_studies") != [STUDY]
        or report.get("protocol") != "corrected-identity"
        or report.get("reference_base_commit") != BASE_COMMIT
        or report.get("dtype") not in ("float32", "float64")
        or report.get("dtype") == "float64" and report.get("fp64_square_root") is not True
        or requested.split(":", 1)[0] != device_type
        or not isinstance(report.get("cases"), list) or not report["cases"]
    ):
        raise ValueError(f"Expected a V6 {device_type.upper()} detail report at {BASE_COMMIT}")
    rows = {}
    for _, row in _paired_rows(report["cases"]):
        requested_device = row["requested_device"]
        actual = row.get("actual_execution_device")
        if row["dtype"] != report["dtype"] or requested_device != requested or (
            actual is not None and not (actual == requested or requested == "cuda" and actual.startswith("cuda:"))
        ):
            raise ValueError(f"Case {row['case']} device disagrees with its report")
        if row["execution_status"] == "completed" and actual is None:
            raise ValueError(f"Case {row['case']} lacks its execution device")
        if row["execution_status"] == "completed" and (
            row.get("actual_dtype") != report["dtype"]
            or row.get("model", {}).get("asset_sha256") != report["model_asset_sha256"]
        ):
            raise ValueError(f"Case {row['case']} model or dtype disagrees with its report")
        key = (row["experiment"], row["seed"], row["requested_observations"], row["batch_size"])
        if key in rows:
            raise ValueError(f"Duplicate experiment configuration in {device_type} report: {key}")
        rows[key] = row
    return rows


def compare_reports(cpu, cuda):
    cpu_cases, cuda_cases = _validate_report(cpu, "cpu"), _validate_report(cuda, "cuda")
    unequal = [key for key in ("model_asset_sha256", "focal_schedule", "spectra_requested", "dtype", "fp64_square_root") if cpu.get(key) != cuda.get(key)]
    if unequal or not is_sha256(cpu.get("model_asset_sha256")) or len(cpu.get("focal_schedule", ())) != 600:
        raise ValueError(f"Incompatible V6 report metadata: {', '.join(unequal) or 'missing model or focal schedule'}")
    keys = cpu_cases.keys() & cuda_cases.keys()
    return {
        "schema_version": SCHEMA_VERSION, "study": STUDY, "reference_base_commit": BASE_COMMIT,
        "model_asset_sha256": cpu["model_asset_sha256"],
        "tolerances": {key: {"absolute": a, "relative": r} for key, (a, r) in TOLERANCES.items()},
        "case_order": ["cpu", "cuda"],
        "spectra_requested": cpu["spectra_requested"],
        "cpu_only_configurations": sorted(cpu_cases.keys() - keys),
        "cuda_only_configurations": sorted(cuda_cases.keys() - keys),
        "pairs": [
            {"experiment": kind, "seed": seed, "observation_count": count, "batch_size": batch}
            | compare_cases(cpu_cases[kind, seed, count, batch], cuda_cases[kind, seed, count, batch])
            for kind, seed, count, batch in sorted(keys)
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--cuda-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="create a new comparison directory")
    args = parser.parse_args(argv)
    try:
        comparison = compare_reports(read_json(args.cpu_report), read_json(args.cuda_report))
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "comparison.json", comparison)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    def failed(pair):
        return (pair["status"] != "consistent" or not pair["both_meet_exact_recovery_targets"]
                or comparison["spectra_requested"] and any(
                    value["status"] != "consistent" for value in pair["observability"].values()
                ))

    return int(bool(comparison["cpu_only_configurations"] or comparison["cuda_only_configurations"])
               or any(failed(pair) for pair in comparison["pairs"]))


if __name__ == "__main__":
    raise SystemExit(main())
