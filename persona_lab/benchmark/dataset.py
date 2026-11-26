"""Versioned synthetic datasets and the optimizer's fitting-only accessor."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from ..camera import PerspectiveCamera, WeakPerspectiveCamera, _camera_batch
from ..coordinates import ImageSize
from ..correspondences import (
    CorrespondenceMap, DFL_LANDMARK_COUNT, LandmarkCorrespondence, Region,
)
from ..observations import LandmarkObservation
from .configuration import resolve_config, torch_options
from .model import MODEL_SHA256, MODEL_URL, in_frame
from .storage import (
    digest_arrays, digest_json, is_sha256, json_bytes, publish_directory, read_arrays, read_json,
    require_keys, validate_arrays, write_json,
)


SCHEMA_VERSION = 1
MODEL_FIELDS = {
    "asset", "asset_sha256", "version", "variant", "identity_count", "identity_names",
    "expression_count", "expression_names", "joint_count", "joint_names", "vertex_count",
    "triangle_count", "topology_sha256", "landmark_configuration", "landmarks_sha256",
}
COORDINATES = {
    "geometry": "GNM native +X subject-left, +Y up, +Z anterior, meters",
    "camera": "OpenCV +X right, +Y down, +Z forward, world-to-camera",
    "landmarks": LandmarkObservation.landmark_ordering,
    "pixels": "aligned_pixels; pixel centers at integers; half-open raster edges",
    "rotations": "right-handed axis-angle vectors in radians",
    "visibility": "physical visibility not evaluated",
}
MASKS = ("supported", "projection_valid", "in_frame", "validity", "missing", "occluded", "outliers")


def subject_id(index):
    return f"subject_{index:04d}"


def observation_id(subject, index):
    return f"{subject}/observation_{index:04d}"


def camera_to_record(camera):
    if _camera_batch(camera):
        raise ValueError("Expected an unbatched camera")
    size = camera.image_size
    record = {
        "image_size": [size.width, size.height],
        "rotation": camera.rotation.detach().cpu().tolist(),
        "translation": camera.translation.detach().cpu().tolist(),
    }
    if isinstance(camera, PerspectiveCamera):
        return record | {
            "type": "perspective",
            "focal_length": camera.focal_length.detach().cpu().tolist(),
            "principal_point": camera.principal_point.detach().cpu().tolist(),
        }
    return record | {"type": "weak-perspective", "scale": camera.scale.detach().cpu().item()}


def camera_from_record(record, *, dtype, device, image_size=None):
    common = {"type", "image_size", "rotation", "translation"}
    kind = record.get("type") if isinstance(record, dict) else None
    if kind == "perspective":
        cls = PerspectiveCamera
        specs = {"rotation": (3,), "translation": (3,), "focal_length": (2,), "principal_point": (2,)}
    elif kind == "weak-perspective":
        cls = WeakPerspectiveCamera
        specs = {"rotation": (3,), "translation": (2,), "scale": ()}
    else:
        raise ValueError("Unknown benchmark camera model")
    require_keys(record, common | specs.keys(), "Camera")
    size = record["image_size"]
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("Camera image_size requires [width, height]")
    size = ImageSize(*size)
    if image_size is not None and size != image_size:
        raise ValueError("Camera raster does not match its benchmark observation")
    parameters = {}
    for name, shape in specs.items():
        values = record[name]
        sequence = values if shape and isinstance(values, list) else [values]
        if any(type(value) not in (int, float) for value in sequence):
            raise ValueError(f"Camera {name} requires finite numeric parameters")
        tensor = torch.tensor(values, dtype=dtype, device=device)
        if tensor.shape != shape:
            raise ValueError(f"Camera {name} must have shape {shape}")
        parameters[name] = tensor
    camera = cls(image_size=size, **parameters)
    _camera_batch(camera)
    return camera


def _load_mapping(records):
    if not isinstance(records, list):
        raise ValueError("Dataset correspondences must be a list")
    fields = {
        "dfl_index", "gnm_index", "description", "region", "view_dependent",
        "enabled_by_default", "limitation",
    }
    entries = []
    for item in records:
        require_keys(item, fields, "Landmark correspondence")
        if item["limitation"] is not None and not isinstance(item["limitation"], str):
            raise ValueError("Correspondence limitation must be a string or null")
        entries.append(LandmarkCorrespondence(**(item | {"region": Region(item["region"])})))
    return CorrespondenceMap(tuple(entries))


def _validate_model(metadata):
    require_keys(metadata, MODEL_FIELDS, "Model metadata")
    if (
        metadata["version"] != "3.0" or metadata["variant"] != "head"
        or metadata["asset"] != MODEL_URL or metadata["asset_sha256"] != MODEL_SHA256
        or metadata["landmark_configuration"] != "HEAD_SPARSE_68"
    ):
        raise ValueError("Dataset requires the pinned GNM v3.0 Head and its sparse landmarks")
    for kind in ("identity", "expression", "joint"):
        names, count = metadata[f"{kind}_names"], metadata[f"{kind}_count"]
        if (
            not isinstance(names, list) or type(count) is not int or count < 1
            or len(names) != count or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != count
        ):
            raise ValueError(f"Invalid ordered model {kind} parameters")
    for key in ("vertex_count", "triangle_count"):
        if type(metadata[key]) is not int or metadata[key] < 1:
            raise ValueError(f"Invalid model {key}")
    for key in ("topology_sha256", "landmarks_sha256"):
        if not is_sha256(metadata[key]):
            raise ValueError(f"Invalid model {key}")


def _subject_specs(config, model):
    dtype = np.dtype(config.dtype)
    n = config.observations_per_subject
    slots = (n, DFL_LANDMARK_COUNT)
    return {
        "identity": (dtype, (model["identity_count"],)),
        "neutral_vertices": (dtype, (model["vertex_count"], 3)),
        "expression": (dtype, (n, model["expression_count"])),
        "rotations": (dtype, (n, model["joint_count"], 3)),
        "translation": (dtype, (n, 3)),
        "clean": (dtype, (*slots, 2)),
        "observed": (dtype, (*slots, 2)),
        "offset_px": (dtype, (*slots, 2)),
        "weights": (dtype, slots),
        **{name: (np.dtype(bool), slots) for name in MASKS},
    }


def _validate_subject(record, arrays, config, mapping):
    items = record["observations"]
    size = ImageSize(*config.image_size)
    if not isinstance(items, list) or len(items) != config.observations_per_subject:
        raise ValueError("Observation metadata does not match the resolved count")
    supported = np.array([entry.supported for entry in mapping.entries])
    enabled = np.array([entry.enabled_by_default for entry in mapping.entries])
    regions = np.array([entry.region.value for entry in mapping.entries])
    dtype, device = torch_options(config.dtype, "cpu")
    for index, item in enumerate(items):
        require_keys(item, {"id", "view", "expression", "noise_profile", "camera"}, "Observation")
        view, expression, kind, profile = config.schedule(index)
        if (
            item["id"] != observation_id(record["id"], index)
            or item["view"] != view or item["expression"] != expression
            or item["noise_profile"] != profile
        ):
            raise ValueError("Invalid, duplicate or out-of-schedule observation identifier or preset")
        camera = camera_from_record(item["camera"], dtype=dtype, device=device, image_size=size)
        if isinstance(camera, PerspectiveCamera) != (kind == "perspective"):
            raise ValueError("Observation camera model does not match the configured schedule")
        if config.perspective_focal_factors is not None:
            expected = config.perspective_focal_factors[index] * min(size.width, size.height)
            if not torch.equal(camera.focal_length, camera.focal_length.new_full((2,), expected)):
                raise ValueError(f"Observation focal length does not match the schedule for {item['id']}")
        slot = {field: array[index] for field, array in arrays.items()
                if field not in ("identity", "neutral_vertices")}
        calculated_frame = in_frame(
            torch.from_numpy(slot["clean"]), size, torch.from_numpy(supported),
        ).numpy()
        clean_valid = supported & enabled & calculated_frame
        noise = config.noise[profile]
        occluded = clean_valid & np.isin(regions, noise.occlude)
        missing = slot["missing"]
        valid = clean_valid & ~occluded & ~missing
        outliers = slot["outliers"]
        if (
            not np.array_equal(slot["supported"], supported)
            or not np.array_equal(slot["projection_valid"], supported)
            or not np.array_equal(slot["in_frame"], calculated_frame)
            or not np.array_equal(slot["occluded"], occluded)
            or not np.array_equal(slot["validity"], valid)
            or (missing & (~clean_valid | occluded)).any() or missing.sum() != noise.missing
            or (outliers & ~valid).any() or outliers.sum() != noise.outliers
            or not (slot["weights"] == 1).all()
        ):
            raise ValueError(f"Inconsistent landmark masks or weights for {item['id']}")
        eligible_offsets = valid if noise.gaussian_px else outliers
        if (
            (slot["clean"][~supported] != 0).any()
            or not np.array_equal(slot["offset_px"], slot["observed"] - slot["clean"])
            or (slot["offset_px"][~eligible_offsets] != 0).any()
        ):
            raise ValueError(f"Inconsistent clean placeholders or perturbations for {item['id']}")


@dataclass(frozen=True, slots=True)
class SubjectGroundTruth:
    subject_id: str
    observations: tuple[dict, ...]
    arrays: dict[str, np.ndarray]


class BenchmarkDataset:
    """A validated dataset. Fitting only needs fitting_observations()."""

    def __init__(self, manifest, config, triangles, mapping, subjects, *, source=None):
        self._manifest = manifest
        self._subjects = subjects
        self._source = source
        self.config = config
        self.triangles = triangles
        self.correspondences = mapping

    @property
    def dataset_id(self):
        return self._manifest["dataset_id"]

    @property
    def source_dataset_id(self):
        return self.dataset_id if self._source is None else self._source.dataset_id

    @property
    def subject_ids(self):
        return tuple(self._subjects)

    @property
    def model_metadata(self):
        return deepcopy(self._manifest["model"])

    def observation_ids(self, subject):
        return tuple(item["id"] for item in self._subject(subject).observations)

    def _subject(self, subject):
        if subject not in self._subjects:
            raise ValueError(f"Unknown benchmark subject {subject!r}")
        return self._subjects[subject]

    def ground_truth(self, subject):
        """Explicit evaluation access to clean coordinates, parameters and corruption."""
        return deepcopy(self._subject(subject))

    def _observation_selection(self, subject, observation_ids):
        all_ids = self.observation_ids(subject)
        selected = all_ids if observation_ids is None else tuple(observation_ids)
        known = set(all_ids)
        if not selected:
            raise ValueError("Select at least one observation ID")
        if any(not isinstance(key, str) or key not in known for key in selected) or (
            len(set(selected)) != len(selected)
        ):
            raise ValueError("Observation selection contains duplicate or unknown identifiers")
        return all_ids, selected

    def select_observations(self, subject, observation_ids=None):
        """Keep the current dataset for its full order, otherwise select from its source."""
        all_ids, selected = self._observation_selection(subject, observation_ids)
        if selected == all_ids:
            return self
        if self._source is not None:
            return self._source.select_observations(subject, selected)
        truth = self._subject(subject)
        positions = {key: index for index, key in enumerate(all_ids)}
        indices = [positions[key] for key in selected]
        arrays = {
            name: values if name in ("identity", "neutral_vertices") else values[indices]
            for name, values in truth.arrays.items()
        }
        records = tuple(truth.observations[index] for index in indices)
        manifest = self._manifest | {"dataset_id": digest_json({
            "source_dataset": self.dataset_id, "subject": subject, "observation_ids": selected,
        })}
        return BenchmarkDataset(
            manifest, self.config, self.triangles, self.correspondences,
            {subject: SubjectGroundTruth(subject, records, arrays)}, source=self,
        )

    def fitting_observations(
        self, subject, *, observation_ids=None, dtype="float64", device="cpu",
        include_known_cameras=True,
    ):
        """Return observed DFL landmarks, masks, weights, and explicitly requested cameras."""
        if type(include_known_cameras) is not bool:
            raise ValueError("include_known_cameras must be a boolean")
        truth = self._subject(subject)
        dtype, device = torch_options(dtype, device)
        size = ImageSize(*self.config.image_size)
        indices = {item["id"]: index for index, item in enumerate(truth.observations)}
        _, selected = self._observation_selection(subject, observation_ids)
        result = []
        for key in selected:
            index = indices[key]
            camera = None
            if include_known_cameras:
                camera = camera_from_record(
                    truth.observations[index]["camera"], dtype=dtype, device=device, image_size=size,
                )
            result.append(LandmarkObservation(
                observation_id=key, image_size=size, correspondences=self.correspondences,
                landmarks=torch.tensor(truth.arrays["observed"][index], dtype=dtype, device=device),
                validity=torch.tensor(truth.arrays["validity"][index], device=device),
                weights=torch.tensor(truth.arrays["weights"][index], dtype=dtype, device=device),
                known_camera=camera,
            ))
        return tuple(result)


def _manifest_id(manifest):
    # Runtime provenance is descriptive. Configuration and numeric contents bind predictions.
    return digest_json({key: value for key, value in manifest.items() if key not in ("dataset_id", "runtime")})


def _read_dataset(path, manifest):
    require_keys(manifest, {
        "schema_version", "dataset_id", "config", "model", "coordinates", "correspondences",
        "runtime", "generator", "subjects",
    }, "Dataset manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark dataset schema version")
    if manifest["dataset_id"] != _manifest_id(manifest):
        raise ValueError("Benchmark manifest does not match its dataset identifier")
    if manifest["coordinates"] != COORDINATES or not isinstance(manifest["runtime"], dict) or (
        not isinstance(manifest["generator"], dict)
    ):
        raise ValueError("Invalid benchmark conventions or provenance")
    if not isinstance(manifest["config"], dict):
        raise ValueError("Benchmark generation configuration must be an object")
    config = resolve_config(**manifest["config"])
    if digest_json(config.metadata()) != digest_json(manifest["config"]):
        raise ValueError("Dataset must preserve the fully resolved generation configuration")
    if config.perspective_focal_factors is not None and (
        manifest["generator"].get("focal_per_short_edge_schedule") != list(config.perspective_focal_factors)
    ):
        raise ValueError("Dataset focal schedule differs from its generation provenance")
    model = manifest["model"]
    _validate_model(model)
    fixed_identity = []
    if config.identity_selection is not None:
        names = model["identity_names"]
        # Native component metadata uses "<region>_<index>" names.
        selected = [
            name for name in names
            if config.identity_selection == "full" or name.rpartition("_")[0] == "head"
        ]
        if not selected or manifest["generator"].get("identity_coefficient_names") != selected:
            raise ValueError("Dataset identity selection differs from its native coefficient provenance")
        fixed_identity = [i for i, name in enumerate(names) if name not in selected]
    mapping = _load_mapping(manifest["correspondences"])
    topology = read_arrays(
        path / "topology.npz", {"triangles": (np.dtype("int32"), (model["triangle_count"], 3))},
    )
    triangles = topology["triangles"]
    if (
        digest_arrays(topology) != model["topology_sha256"] or triangles.min() < 0
        or triangles.max() >= model["vertex_count"] or (triangles == np.roll(triangles, 1, axis=1)).any()
    ):
        raise ValueError("Invalid or inconsistent benchmark topology")
    records = manifest["subjects"]
    if not isinstance(records, list) or len(records) != config.subjects:
        raise ValueError("Subject metadata does not match the resolved count")
    subjects = {}
    identities = set()
    for index, record in enumerate(records):
        require_keys(record, {"id", "observations", "arrays_sha256"}, "Subject record")
        if record["id"] != subject_id(index):
            raise ValueError("Invalid, duplicate or out-of-order subject identifier")
        arrays = read_arrays(path / f"{record['id']}.npz", _subject_specs(config, model))
        if np.any(arrays["identity"][fixed_identity] != 0):
            raise ValueError("Unselected synthetic identity coefficients must be exactly zero")
        if digest_arrays(arrays) != record["arrays_sha256"]:
            raise ValueError(f"Inconsistent arrays for {record['id']}")
        fingerprint = digest_arrays({"identity": arrays["identity"]})
        if fingerprint in identities:
            raise ValueError("Benchmark subject identity vectors must be distinct")
        identities.add(fingerprint)
        _validate_subject(record, arrays, config, mapping)
        subjects[record["id"]] = SubjectGroundTruth(record["id"], tuple(record["observations"]), arrays)
    return BenchmarkDataset(manifest, config, triangles, mapping, subjects)


def load_dataset(path):
    """Open a complete benchmark and validate JSON, numeric arrays and relationships."""
    path = Path(path)
    return _read_dataset(path, read_json(path / "manifest.json"))


def write_dataset(destination, *, config, model, subjects, mapping, runtime, generator):
    """Write verified numeric data, then publish the validated completion manifest."""
    records = []
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config": config.metadata(),
        "model": model.metadata,
        "coordinates": COORDINATES,
        "correspondences": [asdict(entry) for entry in mapping.entries],
        "runtime": runtime,
        "generator": generator,
        "subjects": records,
    }
    with publish_directory(destination, completion="manifest.json") as stage:
        np.savez_compressed(stage / "topology.npz", triangles=model.triangles)
        for index, subject in enumerate(subjects):
            if subject.subject_id != subject_id(index):
                raise ValueError("Subjects must be supplied in stable identifier order")
            validate_arrays(subject.arrays, _subject_specs(config, model.metadata), subject.subject_id)
            records.append({
                "id": subject.subject_id,
                "observations": list(subject.observations),
                "arrays_sha256": digest_arrays(subject.arrays),
            })
            np.savez_compressed(stage / f"{subject.subject_id}.npz", **subject.arrays)
        manifest["dataset_id"] = _manifest_id(manifest)
        # Normalize JSON values before using the same checks as an independent reader.
        manifest = json.loads(json_bytes(manifest))
        result = _read_dataset(stage, manifest)
        write_json(stage / "manifest.json", manifest)
    return result
