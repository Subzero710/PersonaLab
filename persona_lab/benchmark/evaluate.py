"""Evaluate independently supplied identity and optional observation predictions."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..camera import PerspectiveCamera
from .configuration import torch_options
from .dataset import camera_from_record, camera_to_record, load_dataset
from .metrics import (
    coefficient_errors, reprojection_errors, rotation_errors_deg, summarize_pixels,
    translation_error, vertex_errors,
)
from .model import load_benchmark_model, project_dfl
from .predictions import read_predictions
from .storage import publish_directory, write_json


MODES = ("identity-only", "conditional", "known-camera", "complete", "projected")
_REFERENCES = {"clean": "clean", "observed": "observed"}
_NOT_PROVIDED = {"status": "not_provided"}


def _validate_mode(prediction, mode, subject):
    if mode in ("identity-only", "conditional"):
        if prediction.observations:
            raise ValueError(f"{mode} requires identity-only predictions for {subject}")
        return
    if not prediction.observations:
        raise ValueError(f"{mode} requires every observation prediction for {subject}")
    for key, item in prediction.observations.items():
        if mode == "projected":
            if item.projected is None:
                raise ValueError(f"Missing externally projected landmarks for {key}")
        else:
            if item.expression is None or item.rotations is None or item.translation is None:
                raise ValueError(f"Missing expression or pose prediction for {key}")
            if (item.camera is not None) != (mode == "complete"):
                raise ValueError(f"Explicit camera predictions are required only in complete mode: {key}")
            if item.projected is not None:
                raise ValueError(f"Externally projected landmarks require projected mode: {key}")


def _camera_errors(estimate, truth, *, shared_root=None):
    a, b = camera_to_record(estimate), camera_to_record(truth)
    if a["type"] != b["type"] or a["image_size"] != b["image_size"]:
        raise ValueError("Camera predictions must share model and raster size")
    error = {
        "status": "evaluated", "model": a["type"],
        "rotation_deg": float(rotation_errors_deg(a["rotation"], b["rotation"])),
    }
    perspective = isinstance(truth, PerspectiveCamera)
    mismatches = []
    if shared_root is not None:
        fixed = ("rotation", "focal_length", "principal_point") if perspective else ("rotation",)
        mismatches = [name for name in fixed if a[name] != b[name]]
        if not shared_root:
            mismatches.append("root_translation")
        error["fixed_parameters"] = fixed
        error["translation_comparability"] = {
            "status": "not_comparable" if mismatches else "comparable",
            "mismatched_fields": mismatches,
        }
        if mismatches:
            error["status"] = "partially_evaluated"
    translation = None if mismatches else translation_error(a["translation"], b["translation"])
    if perspective:
        error.update(
            translation_l2_m=translation,
            focal_length_abs_px=np.abs(np.array(a["focal_length"]) - b["focal_length"]).tolist(),
            principal_point_abs_px=np.abs(np.array(a["principal_point"]) - b["principal_point"]).tolist(),
        )
    else:
        error.update(
            image_translation_l2_px=translation,
            scale_abs_px_per_m=abs(a["scale"] - b["scale"]),
        )
    return error


def _pose_errors(item, truth, index, joint_names):
    if item.rotations is None:
        return dict(_NOT_PROVIDED)
    angles = rotation_errors_deg(item.rotations, truth["rotations"][index])
    return {
        "status": "evaluated",
        "joint_rotation_deg": dict(zip(joint_names, angles.tolist(), strict=True)),
        "root_translation_l2_m": translation_error(item.translation, truth["translation"][index]),
    }


def _score_observation(model, dataset, truth, index, identity, prediction, mode, *, compare_camera_gauges=False):
    item = truth.observations[index]
    key = item["id"]
    arrays = truth.arrays
    dtype, device = model.reference.dtype, model.reference.device
    fixed_camera = camera_from_record(item["camera"], dtype=dtype, device=device)
    estimate = prediction.observations.get(key)
    camera_recovery = dict(_NOT_PROVIDED)
    if mode == "conditional":
        expression, rotations, translation = (
            arrays[name][index] for name in ("expression", "rotations", "translation")
        )
        camera = fixed_camera
        camera_recovery = {"status": "fixed_known"}
    else:
        expression, rotations, translation = estimate.expression, estimate.rotations, estimate.translation
        camera = fixed_camera
        if estimate.camera is not None:
            camera = camera_from_record(camera_to_record(estimate.camera), dtype=dtype, device=device)
            shared_root = (
                np.array_equal(translation, arrays["translation"][index]) if compare_camera_gauges else None
            )
            camera_recovery = _camera_errors(camera, fixed_camera, shared_root=shared_root)
        elif mode == "known-camera":
            camera_recovery = {"status": "fixed_known"}
    if mode == "projected":
        projected = estimate.projected
    else:
        points = model.landmarks(identity, expression, rotations, translation)
        projected = project_dfl(points, camera, dataset.correspondences).cpu().numpy()
    expression_recovery = dict(_NOT_PROVIDED)
    pose_recovery = dict(_NOT_PROVIDED)
    if estimate is not None:
        pose_recovery = _pose_errors(estimate, arrays, index, model.metadata["joint_names"])
        if estimate.expression is not None:
            expression_recovery = {"status": "evaluated"} | coefficient_errors(
                estimate.expression, arrays["expression"][index],
            )
    valid = arrays["validity"][index]
    return {
        "id": key, "view": item["view"], "expression_preset": item["expression"],
        "noise_profile": item["noise_profile"],
        "expression_recovery": expression_recovery, "pose_recovery": pose_recovery,
        "camera_recovery": camera_recovery,
        "missing_count": int(arrays["missing"][index].sum()),
        "occluded_count": int(arrays["occluded"][index].sum()),
        "outlier_count": int(arrays["outliers"][index].sum()),
        "reprojection": {
            kind: reprojection_errors(projected, arrays[field][index], valid)
            for kind, field in _REFERENCES.items()
        },
    }


def _pool_reprojection(observations):
    result = {}
    for kind in _REFERENCES:
        metrics = [item["reprojection"][kind] for item in observations]
        values = [value for metric in metrics for value in metric["per_landmark_px"] if value is not None]
        total = sum(metric["valid_count"] + metric["excluded_count"] for metric in metrics)
        result[kind] = summarize_pixels(np.asarray(values, dtype=np.float64), total)
    return result


def _pool_reconstruction(records):
    identities = [record["identity"] for record in records]
    geometry = [record["neutral_geometry"] for record in records]
    # Both vector lengths and mesh vertex counts are constant across a dataset.
    return {
        "identity": {
            "mae": float(np.mean([item["mae"] for item in identities])),
            "rmse": float(np.sqrt(np.mean([item["rmse"] ** 2 for item in identities]))),
            "max_absolute": max(item["max_absolute"] for item in identities),
            "coefficient_count": sum(item["coefficient_count"] for item in identities),
        },
        "neutral_geometry": {
            "mean_m": float(np.mean([item["mean_m"] for item in geometry])),
            "rms_m": float(np.sqrt(np.mean([item["rms_m"] ** 2 for item in geometry]))),
            "max_m": max(item["max_m"] for item in geometry),
            "vertex_count": sum(item["vertex_count"] for item in geometry),
        },
    }


def evaluate_predictions(dataset, predictions, *, mode, model_file=None, device="cpu", subject_ids=None):
    """Return a strict machine-readable report for an external prediction directory."""
    if mode not in MODES:
        raise ValueError(f"Evaluation mode must be one of {', '.join(MODES)}")
    all_subjects = dataset.subject_ids
    selected = all_subjects if subject_ids is None else tuple(subject_ids)
    if not selected or len(set(selected)) != len(selected) or any(key not in all_subjects for key in selected):
        raise ValueError("Select known subjects by unique identifier")
    incoming = read_predictions(predictions, dataset)
    if missing := [key for key in selected if key not in incoming]:
        raise ValueError(f"Missing required prediction subjects: {', '.join(missing)}")
    for key in selected:
        _validate_mode(incoming[key], mode, key)
    dtype, device = torch_options(dataset.config.dtype, device)
    model = load_benchmark_model(model_file, dtype=dtype, device=device)
    if model.metadata != dataset.model_metadata or not np.array_equal(model.triangles, dataset.triangles):
        raise ValueError("Active model, parameter ordering, topology or landmarks differ from the benchmark")
    records = []
    pooled = []
    for key in selected:
        truth = dataset.ground_truth(key)
        prediction = incoming[key]
        observations = []
        if mode != "identity-only":
            observations = [
                _score_observation(model, dataset, truth, index, prediction.identity, prediction, mode)
                for index in range(len(truth.observations))
            ]
            pooled.extend(observations)
        record = {
            "id": key, "observation_count": len(truth.observations),
            "identity": coefficient_errors(prediction.identity, truth.arrays["identity"]),
            "neutral_geometry": vertex_errors(model.neutral(prediction.identity), truth.arrays["neutral_vertices"]),
            "reprojection": _pool_reprojection(observations) if observations else None,
            "observations": observations,
        }
        if mode == "identity-only":
            record["parameter_recovery"] = {
                name: dict(_NOT_PROVIDED) for name in ("expression", "pose", "camera")
            }
        records.append(record)
    profiles = dict.fromkeys(item["noise_profile"] for item in pooled)
    return {
        "schema_version": 1, "dataset_id": dataset.dataset_id, "scenario": dataset.config.scenario,
        "source_dataset_id": dataset.source_dataset_id, "evaluation_dataset_id": dataset.dataset_id,
        "model_asset_sha256": model.metadata["asset_sha256"],
        "mode": mode, "dtype": dataset.config.dtype, "evaluation_device": str(model.reference.device),
        "omitted_subject_ids": [key for key in all_subjects if key not in selected],
        "subjects": records, "subject_count": len(records), "scored_observation_count": len(pooled),
        "aggregate": _pool_reconstruction(records) | {
            "reprojection": _pool_reprojection(pooled) if pooled else None,
            "reprojection_by_noise_profile": {
                name: _pool_reprojection([item for item in pooled if item["noise_profile"] == name])
                for name in profiles
            },
            **{name: sum(item[name] for item in pooled) for name in (
                "missing_count", "occluded_count", "outlier_count",
            )},
        },
    }


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--subject", action="append", help="evaluate only selected subject IDs; repeatable")
    parser.add_argument("--observation-id", action="append",
                        help="evaluate an ordered subset of one subject; repeatable")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Output already exists: {args.output}")
        dataset = load_dataset(args.ground_truth)
        if args.observation_id:
            if args.subject is None or len(args.subject) != 1:
                raise ValueError("--observation-id requires exactly one --subject")
            dataset = dataset.select_observations(args.subject[0], args.observation_id)
        report = evaluate_predictions(
            dataset, args.predictions, mode=args.mode,
            model_file=args.model_file, device=args.device, subject_ids=args.subject,
        )
        with publish_directory(args.output, completion="report.json") as stage:
            write_json(stage / "report.json", report)
    except (ArithmeticError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    aggregate = report["aggregate"]
    identity = aggregate["identity"]["rmse"]
    neutral = aggregate["neutral_geometry"]["rms_m"]
    print(
        f"Scenario {report['scenario']} ({args.mode}), {report['subject_count']} subjects: "
        f"identity RMSE {identity:.6g}, neutral vertex RMS {neutral:.6g} m"
    )
    if reprojection := aggregate["reprojection"]:
        for reference, metrics in reprojection.items():
            error = metrics["rms_px"]
            print(f"{reference} reprojection RMS: "
                  + (f"{error:.6g} px ({metrics['valid_count']} points)" if error is not None else "unavailable"))
    print(f"Report written to {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
