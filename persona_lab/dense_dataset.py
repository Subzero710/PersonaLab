"""Validated, versioned dense observations with a record for every attempt."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .benchmark.dataset import camera_from_record, camera_to_record
from .benchmark.storage import (
    digest_arrays, is_sha256, publish_directory, read_arrays, read_json, require_keys, write_json,
)
from .coordinates import ImageSize
from .correspondences import DFL_LANDMARK_COUNT
from .dense_observations import DenseObservation, ImageProvenance, ModelAsset, ObservationSchema
from .observations import AlignmentMetadata


_FORMAT = "personalab-dense-observations-v1"
_OPTIONAL = ("visibility", "confidence", "presence", "weights")
_ALIGNMENT = {"source_rect": (4,), "source_landmarks": (DFL_LANDMARK_COUNT, 2), "image_to_face_mat": (2, 3)}


@dataclass(frozen=True, slots=True)
class ExtractionAttempt:
    observation_id: str
    image_ref: str | None
    observation: DenseObservation | None
    error: str | None = None
    image_size: ImageSize | None = None
    image_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.observation_id, str) or not self.observation_id.strip():
            raise ValueError("An attempt requires an observation ID")
        if self.image_ref is not None and (
            not isinstance(self.image_ref, str) or not self.image_ref.strip()
        ):
            raise ValueError("Image reference must be nonempty when available")
        if self.image_size is not None and not isinstance(self.image_size, ImageSize):
            raise ValueError("Invalid attempt image dimensions")
        if self.image_sha256 is not None and not is_sha256(self.image_sha256):
            raise ValueError("Invalid attempt image digest")
        if self.observation is None:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("A failed attempt requires a reason")
        elif (
            not isinstance(self.observation, DenseObservation) or self.error is not None
            or self.observation.observation_id != self.observation_id
            or self.image_size not in (None, self.observation.image_size)
            or self.image_sha256 != self.observation.source.sha256
            or self.image_ref != self.observation.source.reference
        ):
            raise ValueError("Successful attempt provenance must match its observation")


@dataclass(frozen=True, slots=True)
class DenseDataset:
    schema: ObservationSchema
    provider: str
    provider_version: str
    asset: ModelAsset
    configuration: dict
    attempts: tuple[ExtractionAttempt, ...]

    @property
    def observations(self):
        return tuple(attempt.observation for attempt in self.attempts if attempt.observation is not None)

    def select(self, ids=None):
        by_id = {attempt.observation_id: attempt for attempt in self.attempts}
        ids = tuple(by_id) if ids is None else tuple(ids)
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
            raise ValueError("Select nonempty, unique observation IDs from the dense dataset")
        selected = tuple(by_id[identifier] for identifier in ids)
        if any(attempt.observation is None for attempt in selected):
            failures = ", ".join(attempt.observation_id for attempt in selected if attempt.observation is None)
            raise ValueError(f"Dense selection contains failed observations: {failures}")
        return tuple(attempt.observation for attempt in selected)


def _size_record(value):
    return [value.width, value.height] if value is not None else None


def _image_size(value):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Dimensions must be [width, height]")
    return ImageSize(*value)


def _check_header(schema, provider, provider_version, asset, configuration):
    if not isinstance(schema, ObservationSchema) or not isinstance(asset, ModelAsset) or any(
        not isinstance(item, str) or not item.strip() for item in (provider, provider_version)
    ) or not isinstance(configuration, dict) or not configuration:
        raise ValueError("Dense dataset requires schema, provider, model asset and extraction configuration")


def _write_rows(stage, schema, provider, provider_version, asset, configuration, attempts):
    _check_header(schema, provider, provider_version, asset, configuration)
    rows, identifiers = [], set()
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, ExtractionAttempt) or attempt.observation_id in identifiers:
            raise ValueError("Every extraction attempt needs a unique ID")
        identifiers.add(attempt.observation_id)
        obs = attempt.observation
        row = {
            "observation_id": attempt.observation_id, "image_ref": attempt.image_ref,
            "image_sha256": attempt.image_sha256, "error": attempt.error,
            "image_size": _size_record(obs.image_size if obs is not None else attempt.image_size),
        }
        if obs is not None:
            obs.validate()
            if (obs.schema, obs.provider, obs.provider_version, obs.asset, obs.extraction) != (
                schema, provider, provider_version, asset, configuration
            ):
                raise ValueError("Dense observation does not match its dataset provenance")
            names = ["landmarks", "validity"]
            names.extend(name for name in _OPTIONAL if getattr(obs, name) is not None)
            arrays = {name: getattr(obs, name).detach().cpu().numpy() for name in names}
            alignment = None
            if obs.alignment is not None:
                alignment = {"face_type": obs.alignment.face_type,
                             "source_filename": obs.alignment.source_filename}
                for name in _ALIGNMENT:
                    if (value := getattr(obs.alignment, name)) is not None:
                        arrays[name] = value.detach().cpu().numpy()
            np.savez_compressed(stage / f"observation_{index:05d}.npz", **arrays)
            row |= {
                "dtype": obs.landmarks.detach().cpu().numpy().dtype.name,
                "arrays": sorted(arrays), "sha256": digest_arrays(arrays),
                "alignment": alignment, "source_size": _size_record(obs.source.source_image_size),
                "source_kind": obs.source.kind,
                "known_camera": camera_to_record(obs.known_camera) if obs.known_camera is not None else None,
                "camera_initialization": (
                    camera_to_record(obs.camera_initialization) if obs.camera_initialization is not None else None
                ),
            }
        rows.append(row)
    if not rows:
        raise ValueError("The input contains no selected image or synthetic observation")
    write_json(stage / "manifest.json", {
        "format": _FORMAT, "schema": asdict(schema), "provider": provider,
        "provider_version": provider_version, "asset": asdict(asset),
        "configuration": configuration, "attempts": rows,
    })


def write_dense_dataset(output, schema, provider, provider_version, asset, configuration, attempts):
    """Write a reusable dataset from an ordered stream of extraction attempts."""
    with publish_directory(output, completion="manifest.json") as stage:
        _write_rows(stage, schema, provider, provider_version, asset, configuration, attempts)


def load_dense_dataset(directory, *, schema, provider, provider_version, asset, device="cpu", dtype=None):
    """Load an explicitly selected schema and provider without changing pixel precision."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    require_keys(manifest, {"format", "schema", "provider", "provider_version",
                            "asset", "configuration", "attempts"}, "Dense dataset manifest")
    configuration = manifest["configuration"]
    _check_header(schema, provider, provider_version, asset, configuration)
    schema_record = asdict(schema)
    schema_record["points"] = list(schema_record["points"])
    if (manifest["format"], manifest["schema"], manifest["provider"], manifest["provider_version"],
        manifest["asset"]) != (_FORMAT, schema_record, provider, provider_version, asdict(asset)):
        raise ValueError("Dense dataset schema, provider version or model asset is incompatible")
    if type(manifest["schema"]["version"]) is not int:
        raise ValueError("Dense dataset schema version must be an integer")
    if not isinstance(manifest["attempts"], list) or not manifest["attempts"]:
        raise ValueError("Dense manifest requires ordered extraction attempts")
    rows, identifiers = [], set()
    count = len(schema.points)
    base_keys = {"observation_id", "image_ref", "image_sha256", "error", "image_size"}
    success_keys = {"dtype", "arrays", "sha256", "alignment", "source_size", "known_camera",
                    "camera_initialization", "source_kind"}
    for index, row in enumerate(manifest["attempts"]):
        failed = isinstance(row, dict) and row.get("error") is not None
        require_keys(row, base_keys if failed else base_keys | success_keys, "Dense attempt")
        size = _image_size(row["image_size"])
        obs = None
        if not failed:
            names, precision = row["arrays"], row["dtype"]
            if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
                raise ValueError("Invalid dense observation array names")
            if precision not in ("float32", "float64") or len(set(names)) != len(names):
                raise ValueError("Invalid dense array precision or duplicated arrays")
            torch_dtype = {"float32": torch.float32, "float64": torch.float64}[precision]
            if dtype is not None and dtype != torch_dtype:
                raise ValueError(f"Expected {dtype}, dense observations store {precision}")
            float_dtype = np.dtype(precision)
            candidates = {
                "landmarks": (float_dtype, (count, 2)), "validity": (np.dtype("bool"), (count,)),
                **{name: (float_dtype, (count,)) for name in _OPTIONAL},
                **{name: (float_dtype, shape) for name, shape in _ALIGNMENT.items()},
            }
            if not {"landmarks", "validity"} <= set(names) <= candidates.keys():
                raise ValueError("Dense observation requires exactly its declared numeric arrays")
            alignment = row["alignment"]
            if alignment is not None:
                require_keys(alignment, {"face_type", "source_filename"}, "Dense alignment")
            elif set(names) & _ALIGNMENT.keys():
                raise ValueError("Dense alignment arrays require alignment metadata")
            arrays = read_arrays(directory / f"observation_{index:05d}.npz",
                                 {name: candidates[name] for name in names})
            if digest_arrays(arrays) != row["sha256"]:
                raise ValueError(f"Dense array digest mismatch at attempt {index}")
            tensors = {name: torch.from_numpy(value).to(device=device) for name, value in arrays.items()}
            if alignment is not None:
                alignment = AlignmentMetadata(**alignment, **{
                    name: tensors.pop(name, None) for name in _ALIGNMENT
                })

            cameras = {
                name: camera_from_record(row[name], dtype=torch_dtype, device=device, image_size=size)
                for name in ("known_camera", "camera_initialization") if row[name] is not None
            }
            obs = DenseObservation(
                observation_id=row["observation_id"], schema=schema, provider=provider,
                provider_version=provider_version, asset=asset, extraction=configuration,
                source=ImageProvenance(row["image_ref"], row["image_sha256"], _image_size(row["source_size"]),
                                       kind=row["source_kind"]),
                image_size=size, alignment=alignment, **cameras, **tensors,
            )
        attempt = ExtractionAttempt(row["observation_id"], row["image_ref"], obs,
                                    row["error"], size, row["image_sha256"])
        if attempt.observation_id in identifiers:
            raise ValueError("Duplicate dense observation ID")
        identifiers.add(attempt.observation_id)
        rows.append(attempt)
    return DenseDataset(schema, provider, provider_version, asset, configuration, tuple(rows))
