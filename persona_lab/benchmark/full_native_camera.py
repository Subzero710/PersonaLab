"""Cases and report reductions for the full-native camera identity study."""

from dataclasses import replace
from math import hypot, isclose, sqrt
from statistics import mean, pstdev
import sys

from ..fitting.levenberg_marquardt import StageCoordinates, bound_mask, gradient_norms, normal_blocks, square_root_blocks
from ..fitting.monitoring import finite_norm
from ..fitting.observability import _state
from ..fitting.observations import prepare_observations
from ..fitting.parameters import FittingParameters
from ..fitting.schur import column_scales
from .dataset import camera_to_record


STUDY = "full-native-camera"
HISTORICAL_SOURCE_ID = "45274d8fb702063f484efcdae27125301913de08cf6e093e8eec3679bacf21c3"
HISTORICAL_METRICS = {
    "trainable_identity_rmse": 0.0108688552,
    "full_native_identity_rmse": 0.1616430511,
    "neutral_mesh_rms_m": 0.140473709 / 1000,
    "clean_reprojection_rms_px": 0.0000204532,
}
CONDITIONS = (
    ("A", "full-native", "known"),
    ("B", "full-native", "estimated"),
    ("C", "head-only", "estimated"),
    ("D", "head-only", "known"),
)
COMPARISONS = (
    ("A", "B", "camera_estimation_full_native"),
    ("D", "C", "camera_estimation_head_only"),
    ("D", "A", "fixed_identity_known_camera"),
    ("C", "B", "fixed_identity_estimated_camera"),
)
NUISANCE_GROUPS = ("fixed_nuisance", "after_expression", "after_all_nuisance")


def case_matrix(seeds):
    """Describe the matched Scenario C cases before any dataset is generated."""
    for seed in seeds:
        for camera in ("perspective", "weak-perspective"):
            for condition, identity, mode in CONDITIONS:
                yield {
                    "condition": condition, "seed": seed, "camera_model": camera,
                    "identity_ground_truth": identity, "camera_mode": mode,
                }
    yield {
        "condition": "historical", "seed": 42, "camera_model": "both",
        "identity_ground_truth": "full-native", "camera_mode": "known",
    }


def coefficient_floor(initial, final, incompatible_names):
    """Decompose total coefficient MSE using the fixed initialization as the floor."""
    count = initial["global"]["coefficient_count"]
    fixed_count = initial["fixed"]["coefficient_count"]
    fixed_initial_mse = (fixed_count / count) * initial["fixed"]["rmse"] ** 2
    result = {
        "incompatible_fixed_coefficients": list(incompatible_names),
        "incompatible_fixed_count": len(incompatible_names),
        "fixed_count": fixed_count,
        "trainable_count": initial["trainable"]["coefficient_count"],
        "fixed_initial_rmse": initial["fixed"]["rmse"],
        "full_native_rmse_floor": sqrt(fixed_initial_mse),
        "fixed_rmse": None,
        "full_native_rmse": None,
        "trainable_rmse": None,
        "excess_squared_coefficient_error": None,
        "decomposition": None,
    }
    if final is not None:
        trainable, fixed, full = (final[key] for key in ("trainable", "fixed", "global"))
        head_mse = (trainable["coefficient_count"] / count) * trainable["rmse"] ** 2
        fixed_mse = (fixed_count / count) * fixed["rmse"] ** 2
        global_mse = full["rmse"] ** 2
        result.update(
            fixed_rmse=fixed["rmse"], full_native_rmse=full["rmse"],
            trainable_rmse=trainable["rmse"],
            # Add the fixed excess before the head error to avoid cancellation at the floor.
            excess_squared_coefficient_error=(fixed_mse - fixed_initial_mse) + head_mse,
            decomposition={
                "trainable_mse_contribution": head_mse, "fixed_mse_contribution": fixed_mse,
                "global_mse": global_mse, "residual": global_mse - (head_mse + fixed_mse),
                "valid": (
                    trainable["coefficient_count"] + fixed_count == full["coefficient_count"] == count
                    and fixed["coefficient_count"] == fixed_count
                    and isclose(global_mse, head_mse + fixed_mse, rel_tol=32 * sys.float_info.epsilon)
                ),
            },
        )
    return result


def final_gradient(model, observations, fit, groups):
    """Measure the projected objective gradient at the actual final snapshot."""
    if fit.final is None:
        return {"status": "unavailable", "reason": fit.reason}
    config = fit.config
    try:
        data = prepare_observations(observations, model.reference, config)
        state = _state(model, data, config, fit.selection, None, fit, "final")
        parameters = FittingParameters(model, data, state, fit.selection, config)
        layout = StageCoordinates.from_parameters(parameters, groups)
        base = parameters.values()
        point = layout.pack(base, model.reference, len(data))
        if config.lm.uses_square_root:
            system = square_root_blocks(data, parameters, layout, point)
            scales = system.scales(config.lm.diagonal_floor)
        else:
            system = normal_blocks(model, data, parameters, layout, base, point)
            scales = column_scales(system, config.lm.diagonal_floor)
        gradient = system.gradient
        active_bounds = bound_mask(point, gradient, layout.expression_slice(), config.expression_limit)
        norms = gradient_norms(layout, gradient, active_bounds)
        scaled = finite_norm(
            2 * scales.identity * gradient.identity,
            (2 * scales.local * gradient.local).masked_fill(active_bounds, 0),
            dtype=gradient.identity.dtype,
        )
    except (ArithmeticError, RuntimeError, ValueError) as error:
        return {"status": "failed", "reason": str(error)}
    return {
        "status": "evaluated", "projected_gradient_norm": hypot(*norms.values()),
        "projected_gradient_norms": norms, "scaled_projected_gradient_norm": scaled,
        "active_expression_bounds": int(active_bounds.sum()),
        "camera_offset_origin": "final_camera" if config.camera_mode == "estimated" else None,
    }


def classify_compatibility(compatibility, case, eye_and_teeth_names):
    """Recognize only the identity and camera differences declared by the experiment."""
    identity = compatibility["identity"]["differing_coefficients"]
    allowances = set(eye_and_teeth_names) if case["identity_ground_truth"] == "full-native" else set()
    expected_identity = [name for name in identity if name in allowances]
    unexpected_identity = [name for name in identity if name not in allowances]
    fixed_focal_gauge = case["camera_mode"] == "estimated" and case["camera_model"] == "perspective"
    focal_mismatches, unexpected_observations = [], []
    for observation in compatibility["observations"]:
        camera_fields = set(observation["differing_fixed_camera_fields"])
        if fixed_focal_gauge and "focal_length" in camera_fields:
            focal_mismatches.append(observation["id"])
            camera_fields.remove("focal_length")
        if (camera_fields or observation["differing_fixed_joints"] or observation["differing_fixed_root"]
                or observation["expression"]["differing_coefficients"] or observation["expression_outside_bounds"]):
            unexpected_observations.append(observation["id"])
    unexpected = bool(unexpected_identity or unexpected_observations or (
        not compatibility["fully_representable"] and not (expected_identity or focal_mismatches)
    ))
    if unexpected:
        status = "unexpected_incompatibility"
    elif expected_identity:
        status = "expected_identity_incompatibility"
    elif focal_mismatches:
        status = "fixed_focal_gauge_difference"
    else:
        status = "fully_representable"
    return {
        "status": status,
        "expected_incompatible_identity_coefficients": expected_identity,
        "unexpected_incompatible_identity_coefficients": unexpected_identity,
        "unexpected_incompatible_observations": unexpected_observations,
        "fixed_focal_gauge_observations": focal_mismatches,
    }


def fitted_camera_prediction(fit):
    """Use the actual camera snapshots, including for fixed-camera cases."""
    prediction = fit.to_prediction()
    observations = {
        key: replace(prediction.observations[key], camera=camera)
        for key, camera in zip(fit.observation_ids, fit.final.cameras, strict=True)
    }
    return replace(prediction, observations=observations)


def _comparison_status(status, reason):
    return {"status": status, "reason": reason}


def _camera_fields(gauge, model):
    fields = gauge[model.replace("-", "_")] | {"rotation": gauge["camera_rotation"], "image_size": "fixed"}
    fixed = [name for name, status in fields.items() if status == "fixed"]
    trainable = [name for name, status in fields.items() if status != "fixed"]
    return fixed, trainable


def camera_protocol(fit):
    """Record the actual parameterization even when the optimizer has no final state."""
    models = list(dict.fromkeys(camera_to_record(camera)["type"] for camera in fit.initial.cameras))
    fields = {model: _camera_fields(fit.gauge, model) for model in models}
    if fit.config.camera_mode == "known":
        interpretation = "The known cameras use the calibrated synthetic metric scale."
    elif "weak-perspective" in models:
        interpretation = "Weak perspective leaves absolute metric scale ambiguous. Geometry uses unaligned native vertices."
    else:
        interpretation = (
            "Estimated perspective uses the prescribed fixed focal length. Geometry uses unaligned native vertices."
        )
    return {
        "models": models, "fixed_camera_fields": {model: fixed for model, (fixed, _) in fields.items()},
        "trainable_camera_fields": {model: trainable for model, (_, trainable) in fields.items()},
        "comparable_translation_observations": None,
        "non_comparable_translation_observations": None,
        "metric_scale_interpretation": interpretation,
    }


def _camera_details(camera_score, initial, fitted, truth, gauge, mode):
    model = fitted["type"]
    fixed, trainable = _camera_fields(gauge, model)
    camera_score.update(
        mode=mode, initial_parameters=initial, fitted_parameters=fitted, reference_parameters=truth,
        fixed_fields=fixed, trainable_fields=trainable,
    )
    translation = camera_score["translation_comparability"]
    mismatches = translation["mismatched_fields"]
    translation["reason"] = (
        f"The fit and reference differ in fixed gauge fields ({', '.join(mismatches)})."
        if mismatches else "The fit and reference share the fixed translation gauge."
    )
    rotation_comparable = "rotation" not in mismatches
    camera_score["rotation_comparability"] = (
        _comparison_status("comparable", "The camera rotation is fixed in the same axes.")
        if rotation_comparable else
        _comparison_status("not_comparable", "The fixed camera rotations define different pose frames.")
    )
    if not rotation_comparable:
        camera_score["rotation_deg"] = None
    if model == "perspective":
        camera_score["principal_point_comparability"] = _comparison_status(
            "comparable", "The principal points use the same raster coordinates.",
        )
        camera_score["focal_length_comparability"] = _comparison_status(
            "comparable", "The focal lengths use the same raster units. The fitted focal length is fixed.",
        )
    else:
        camera_score["principal_point_abs_px"] = None
        camera_score["principal_point_comparability"] = _comparison_status(
            "not_applicable", "Weak perspective represents image offsets as translation.",
        )
        camera_score["scale_comparability"] = _comparison_status(
            "comparable" if mode == "known" else "conditional",
            "The scale comparison uses the native model's metric units.",
        )
    return rotation_comparable


def add_observation_details(training, truth, fit):
    """Describe what can be compared in the already-scored fitted gauge."""
    mode = fit.config.camera_mode
    gauge = fit.gauge
    for score, initial, fitted, reference in zip(
        training["observations"], fit.initial.cameras, fit.final.cameras, truth.observations, strict=True,
    ):
        camera_score = score["camera_recovery"]
        rotation_comparable = _camera_details(
            camera_score, camera_to_record(initial), camera_to_record(fitted), reference["camera"], gauge, mode,
        )
        pose = score["pose_recovery"]
        pose["joint_rotation_comparability"] = camera_score["rotation_comparability"]
        if not rotation_comparable:
            pose["joint_rotation_deg"] = dict.fromkeys(pose["joint_rotation_deg"])
        if not rotation_comparable:
            root = _comparison_status("not_comparable", "The camera rotation gauges differ.")
        elif mode == "estimated":
            shared = "root_translation" not in camera_score["translation_comparability"]["mismatched_fields"]
            root = (
                _comparison_status("fixed_shared", "The root is fixed at the shared origin.")
                if shared else
                _comparison_status("not_comparable", "The fitted camera absorbs translation in a different root gauge.")
            )
        else:
            root = _comparison_status("comparable", "The known camera provides the same root reference frame.")
        pose["root_translation_comparability"] = root
        if root["status"] == "not_comparable":
            pose["root_translation_l2_m"] = None
    scores = [item["camera_recovery"] for item in training["observations"]]
    comparable = sum(item["translation_comparability"]["status"] == "comparable" for item in scores)
    return camera_protocol(fit) | {
        "comparable_translation_observations": comparable,
        "non_comparable_translation_observations": len(scores) - comparable,
    }


def _metrics(record):
    final = record.get("final") or {}
    identity = final.get("identity", {})
    reprojection = final.get("training_reprojection_and_nuisance_recovery", {}).get("aggregate", {})
    regions = final.get("neutral_geometry_regions", {})
    return {
        "trainable_identity_rmse": identity.get("trainable", {}).get("rmse"),
        "full_native_identity_rmse": identity.get("global", {}).get("rmse"),
        "fixed_identity_rmse": identity.get("fixed", {}).get("rmse"),
        "full_native_rmse_floor": record.get("identity_error_floor", {}).get("full_native_rmse_floor"),
        "neutral_mesh_rms_m": final.get("neutral_geometry", {}).get("rms_m"),
        "facial_region_rms_m": regions.get("facial", {}).get("rms_m"),
        "exterior_skin_rms_m": regions.get("exterior_skin", {}).get("rms_m"),
        "eye_geometry_rms_m": regions.get("eyes", {}).get("rms_m"),
        "teeth_geometry_rms_m": regions.get("teeth", {}).get("rms_m"),
        "clean_reprojection_rms_px": reprojection.get("clean", {}).get("rms_px"),
        "observed_reprojection_rms_px": reprojection.get("observed", {}).get("rms_px"),
    }


def historical_comparison(record):
    actual = record.get("source_dataset_id")
    measured = _metrics(record)
    same_dataset = None if actual is None else actual == HISTORICAL_SOURCE_ID
    return {
        "expected_source_dataset_id": HISTORICAL_SOURCE_ID,
        "actual_source_dataset_id": actual,
        "matches_source_dataset_id": same_dataset,
        "reference_metrics": HISTORICAL_METRICS,
        "measured_metrics": {key: measured[key] for key in HISTORICAL_METRICS},
        "v5_minus_v2": {
            key: measured[key] - baseline if same_dataset and measured[key] is not None else None
            for key, baseline in HISTORICAL_METRICS.items()
        },
        "comparison_context": "The optimization schedule and LM convergence configuration both changed.",
    }


def _spectra_row(spectrum):
    if spectrum is None:
        return None
    return {
        name: {
            key: spectrum[name][key] for key in (
                "rank", "nullity", "condition_number", "effective_condition_number",
            )
        }
        for name in NUISANCE_GROUPS
    }


def _case_row(index, record):
    return {
        "case": record["case"], "report_case": f"report.json#/cases/{index}",
        **record[STUDY],
        "source_dataset_id": record.get("source_dataset_id"),
        "evaluation_dataset_id": record.get("evaluation_dataset_id"),
        "status": record["status"], "reason": record["reason"],
        "execution_status": record["execution_status"], "converged": record["converged"],
        "representability": record["representability_status"],
        "fully_representable": record.get("compatibility", {}).get("fully_representable"),
        "scientific_validation": record.get("scientific_validation"),
        "camera_comparison": record.get("camera_comparison"),
        "metrics": _metrics(record),
        "observability": {
            state: _spectra_row(record.get(f"{state}_identity_observability"))
            for state in ("initial", "final")
        },
        "observability_status": record.get("observability_status"),
        "optimization_seconds": record.get("runtime", {}).get("optimization_seconds"),
    }


def _delta(before, after):
    return after - before if before is not None and after is not None else None


def _pair(before, after):
    before_source, after_source = before["source_dataset_id"], after["source_dataset_id"]
    return {
        "seed": before["seed"], "camera_model": before["camera_model"],
        "before": before["report_case"], "after": after["report_case"],
        "same_source_dataset": before_source == after_source if before_source and after_source else None,
        "same_evaluation_dataset": (
            before["evaluation_dataset_id"] == after["evaluation_dataset_id"]
            if before["evaluation_dataset_id"] and after["evaluation_dataset_id"] else None
        ),
        "metric_deltas": {
            name: _delta(before["metrics"][name], value) for name, value in after["metrics"].items()
        },
        "rank_deltas": {
            state: {
                name: after["observability"][state][name]["rank"] - baseline[name]["rank"]
                for name in NUISANCE_GROUPS
            } if baseline is not None and after["observability"][state] is not None else None
            for state, baseline in before["observability"].items()
        },
    }


def _aggregate(pairs):
    metrics = {}
    for name in pairs[0]["metric_deltas"]:
        values = [pair["metric_deltas"][name] for pair in pairs if pair["metric_deltas"][name] is not None]
        metrics[name] = {"n": len(values), "mean": mean(values) if values else None,
                         "sd": pstdev(values) if values else None}
    return {"pairs": len(pairs), "metric_deltas": metrics}


def summarize(records):
    """Keep compact rows and matched comparisons pointing at detailed cases."""
    rows = [_case_row(index, record) for index, record in enumerate(records) if STUDY in record]
    primary = {(row["seed"], row["camera_model"], row["condition"]): row for row in rows
               if row["condition"] != "historical"}
    comparisons = []
    for before, after, name in COMPARISONS:
        pairs = [
            _pair(row, primary[seed, camera, after])
            for (seed, camera, condition), row in primary.items()
            if condition == before and (seed, camera, after) in primary
        ]
        cameras = dict.fromkeys(pair["camera_model"] for pair in pairs)
        comparisons.append({
            "comparison": name, "before_condition": before, "after_condition": after,
            "pairs": pairs, "aggregate": _aggregate(pairs) if pairs else None,
            "by_camera_model": {
                camera: _aggregate([pair for pair in pairs if pair["camera_model"] == camera])
                for camera in cameras
            },
        })
    historical = next(
        ({"report_case": f"report.json#/cases/{index}"} | historical_comparison(record)
         for index, record in enumerate(records)
         if record.get(STUDY, {}).get("condition") == "historical"),
        None,
    )
    return {"schema_version": 1, "study": STUDY, "detail_report": "report.json",
            "cases": rows, "comparisons": comparisons, "historical_comparison": historical}
