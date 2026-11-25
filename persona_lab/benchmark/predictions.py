"""Stable-ID interchange for parameters from an external fitting implementation."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..camera import Camera
from ..coordinates import ImageSize
from ..correspondences import DFL_LANDMARK_COUNT
from .configuration import torch_options
from .dataset import camera_from_record, camera_to_record
from .storage import publish_directory, read_arrays, read_json, require_keys, validate_arrays, write_json


PREDICTION_VERSION = 1


@dataclass(frozen=True, slots=True)
class ObservationPrediction:
    expression: np.ndarray | None = None
    rotations: np.ndarray | None = None
    translation: np.ndarray | None = None
    camera: Camera | None = None
    projected: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class SubjectPrediction:
    identity: np.ndarray
    observations: dict[str, ObservationPrediction] = field(default_factory=dict)


def _prediction_specs(dataset, observation_count):
    model = dataset.model_metadata
    dtype = np.dtype(dataset.config.dtype)
    n = observation_count
    return {
        "identity": (dtype, (model["identity_count"],)),
        "expression": (dtype, (n, model["expression_count"])),
        "rotations": (dtype, (n, model["joint_count"], 3)),
        "translation": (dtype, (n, 3)),
        "projected": (dtype, (n, DFL_LANDMARK_COUNT, 2)),
    }


def _check_observations(dataset, subject, ids):
    if any(not isinstance(key, str) for key in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Invalid or duplicate prediction observation IDs for {subject}")
    if ids and set(ids) != set(dataset.observation_ids(subject)):
        raise ValueError(f"Missing, unknown or misassociated prediction observation IDs for {subject}")


def read_predictions(path, dataset):
    """Validate an external prediction directory against the associated benchmark."""
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    require_keys(manifest, {"schema_version", "dataset_id", "model", "subjects"}, "Prediction manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != PREDICTION_VERSION:
        raise ValueError("Unsupported prediction schema version")
    if manifest["dataset_id"] != dataset.dataset_id:
        raise ValueError("Predictions are associated with a different benchmark dataset")
    if manifest["model"] != dataset.model_metadata:
        raise ValueError("Prediction model, topology or parameter ordering differs from the benchmark")
    records = manifest["subjects"]
    if not isinstance(records, list) or not records:
        raise ValueError("Predictions require at least one subject")
    dtype, device = torch_options(dataset.config.dtype, "cpu")
    size = ImageSize(*dataset.config.image_size)
    predictions = {}
    for record in records:
        require_keys(record, {"id", "arrays", "observations"}, "Prediction subject")
        subject, names, observations = record["id"], record["arrays"], record["observations"]
        if not isinstance(subject, str) or subject not in dataset.subject_ids or subject in predictions:
            raise ValueError(f"Unknown or duplicate prediction subject {subject!r}")
        if not isinstance(observations, list):
            raise ValueError("Prediction observations must be a list")
        for item in observations:
            if not isinstance(item, dict) or set(item) not in ({"id"}, {"id", "camera"}):
                raise ValueError("Prediction observations require an ID and an optional camera")
        ids = [item["id"] for item in observations]
        _check_observations(dataset, subject, ids)
        specs = _prediction_specs(dataset, len(ids))
        if (
            not isinstance(names, list) or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names) or "identity" not in names
            or not set(names) <= specs.keys() or not ids and names != ["identity"]
        ):
            raise ValueError(f"Invalid prediction array names for {subject}")
        if ("rotations" in names) != ("translation" in names):
            raise ValueError("A pose prediction requires rotations and root translation")
        arrays = read_arrays(path / f"{subject}.npz", {name: specs[name] for name in names})
        decoded = {}
        camera_count = sum("camera" in item for item in observations)
        if camera_count not in (0, len(ids)):
            raise ValueError("Supply cameras for all or none of a subject's observations")
        known_kinds = {
            item["id"]: item["camera"]["type"] for item in dataset.ground_truth(subject).observations
        } if camera_count else {}
        for index, item in enumerate(observations):
            key = item["id"]
            camera = None
            if "camera" in item:
                camera = camera_from_record(item["camera"], dtype=dtype, device=device, image_size=size)
                if item["camera"]["type"] != known_kinds[key]:
                    raise ValueError(f"Prediction camera type is incompatible for {key}")
            decoded[key] = ObservationPrediction(
                **{name: arrays[name][index] for name in names if name != "identity"}, camera=camera,
            )
        if ids and len(names) == 1 and not camera_count:
            raise ValueError(f"Observation IDs for {subject} require prediction data")
        predictions[subject] = SubjectPrediction(arrays["identity"], decoded)
    return predictions


def write_predictions(path, dataset, predictions):
    """Publish predictions using the same schema checked for external producers."""
    if not predictions:
        raise ValueError("Predictions require at least one subject")
    if any(subject not in dataset.subject_ids for subject in predictions):
        raise ValueError("Unknown prediction subject identifier")
    manifest = {
        "schema_version": PREDICTION_VERSION,
        "dataset_id": dataset.dataset_id,
        "model": dataset.model_metadata,
        "subjects": [],
    }
    with publish_directory(path, completion="manifest.json") as stage:
        for subject, prediction in predictions.items():
            ids = list(prediction.observations)
            _check_observations(dataset, subject, ids)
            entries = list(prediction.observations.values())
            specs = _prediction_specs(dataset, len(ids))
            arrays = {"identity": prediction.identity}
            for name in specs.keys() - {"identity"}:
                values = [getattr(item, name) for item in entries]
                populated = [value for value in values if value is not None]
                if populated:
                    if len(populated) != len(ids):
                        raise ValueError(f"Supply {name} for all or none of {subject}'s observations")
                    shape = specs[name][1][1:]
                    validate_arrays(
                        dict(zip(ids, values, strict=True)),
                        dict.fromkeys(ids, (specs[name][0], shape)), name,
                    )
                    arrays[name] = np.stack(values)
            validate_arrays(arrays, {name: specs[name] for name in arrays}, subject)
            records = []
            for key, item in prediction.observations.items():
                record = {"id": key}
                if item.camera is not None:
                    record["camera"] = camera_to_record(item.camera)
                records.append(record)
            manifest["subjects"].append({"id": subject, "observations": records, "arrays": sorted(arrays)})
            np.savez_compressed(stage / f"{subject}.npz", **arrays)
        write_json(stage / "manifest.json", manifest)
        read_predictions(stage, dataset)


def oracle_predictions(dataset, *, mode="complete", subject_ids=None):
    """Diagnostic only. Construct explicit predictions directly from ground truth."""
    if mode not in ("identity-only", "conditional", "known-camera", "complete", "projected"):
        raise ValueError("Unknown oracle prediction mode")
    dtype, device = torch_options(dataset.config.dtype, "cpu")
    result = {}
    for subject in dataset.subject_ids if subject_ids is None else subject_ids:
        if subject in result:
            raise ValueError("Duplicate oracle subject")
        truth = dataset.ground_truth(subject)
        arrays = truth.arrays
        observations = {}
        for index, item in enumerate(truth.observations):
            if mode in ("identity-only", "conditional"):
                break
            if mode == "projected":
                prediction = ObservationPrediction(projected=arrays["clean"][index])
            else:
                camera = camera_from_record(item["camera"], dtype=dtype, device=device) if mode == "complete" else None
                prediction = ObservationPrediction(
                    expression=arrays["expression"][index], rotations=arrays["rotations"][index],
                    translation=arrays["translation"][index], camera=camera,
                )
            observations[item["id"]] = prediction
        result[subject] = SubjectPrediction(arrays["identity"], observations)
    return result
