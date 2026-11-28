"""Versioned 2D measurements and provenance for variable-size observation schemas."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .benchmark.storage import is_sha256, json_bytes
from .camera import Camera, _camera_batch
from .coordinates import ImageSize, _validate_tensor, aligned_to_source
from .correspondences import DFL_LANDMARK_COUNT
from .observations import AlignmentMetadata


@dataclass(frozen=True, slots=True)
class ObservationPoint:
    """Stable provider slot, anatomical annotation and subject-relative side."""

    identifier: str
    region: str
    side: str = "unspecified"

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.identifier, self.region)):
            raise ValueError("A point needs a nonempty identifier and anatomical region")
        if self.side not in ("left", "right", "midline", "unspecified"):
            raise ValueError("Unknown subject-relative side")


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    identifier: str
    version: int
    coordinate_convention: str
    points: tuple[ObservationPoint, ...]

    def __post_init__(self):
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise ValueError("Schema identifier must be nonempty")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("Schema version must be a positive integer")
        if self.coordinate_convention not in ("source_pixels", "aligned_pixels"):
            raise ValueError("Unknown schema coordinate convention")
        if not isinstance(self.points, tuple) or not self.points or any(
            not isinstance(point, ObservationPoint) for point in self.points
        ):
            raise ValueError("A schema needs a nonempty ordered tuple of observation points")
        if len(set(self.point_ids)) != len(self.points):
            raise ValueError("Schema point identifiers must be unique")

    @property
    def point_ids(self):
        return tuple(point.identifier for point in self.points)


@dataclass(frozen=True, slots=True)
class ModelAsset:
    version: str
    uri: str
    sha256: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.version, self.uri)) or not is_sha256(self.sha256):
            raise ValueError("Model asset requires a version, URI and SHA-256")


@dataclass(frozen=True, slots=True)
class ImageProvenance:
    reference: str
    sha256: str | None
    source_image_size: ImageSize | None
    kind: str = "image"

    def __post_init__(self):
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("A source reference is required")
        if self.kind == "image":
            if not is_sha256(self.sha256):
                raise ValueError("A source image requires its SHA-256")
        elif self.kind != "synthetic" or self.sha256 is not None:
            raise ValueError("Synthetic provenance has no image digest")
        if self.source_image_size is not None and not isinstance(self.source_image_size, ImageSize):
            raise ValueError("Source image size must be an ImageSize")


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class DenseObservation:
    """One raster of ordered measurements with explicit validity and optional scores.

    A score tensor uses -1 for an unavailable slot. An absent tensor means the
    provider supplied no score of that type. Presence is distinct from both
    visibility and localization confidence.
    """

    observation_id: str
    schema: ObservationSchema
    provider: str
    provider_version: str
    asset: ModelAsset
    source: ImageProvenance
    extraction: dict
    image_size: ImageSize
    landmarks: Tensor
    validity: Tensor
    visibility: Tensor | None = None
    confidence: Tensor | None = None
    presence: Tensor | None = None
    weights: Tensor | None = None
    alignment: AlignmentMetadata | None = None
    known_camera: Camera | None = None
    camera_initialization: Camera | None = None
    # Dense serialization preserves its declared precision directly.
    pixel_low: None = field(default=None, init=False)

    def __post_init__(self):
        self.validate()

    def validate(self):
        for value in (self.observation_id, self.provider, self.provider_version):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Observation ID, provider and provider version must be nonempty")
        if not isinstance(self.schema, ObservationSchema) or not isinstance(self.image_size, ImageSize):
            raise ValueError("Dense observation requires a schema and raster dimensions")
        if not isinstance(self.asset, ModelAsset) or not isinstance(self.source, ImageProvenance):
            raise ValueError("Dense observation requires model and source provenance")
        if not isinstance(self.extraction, dict) or not self.extraction:
            raise ValueError("Dense observation requires explicit extraction configuration")
        json_bytes(self.extraction)
        count = len(self.schema.points)
        shape = (count, 2)
        _validate_tensor(self.landmarks, "Dense observed pixels", shape)
        if self.landmarks.shape != shape:
            raise ValueError(f"Dense observed pixels must have shape {shape}")
        if self.coordinate_convention == "source_pixels":
            if self.source.source_image_size != self.image_size or self.alignment is not None:
                raise ValueError("Source pixels require matching source dimensions and no crop alignment")
        if not isinstance(self.validity, Tensor) or self.validity.shape != (count,) or (
            self.validity.dtype != torch.bool or self.validity.device != self.landmarks.device
            or self.validity.layout != torch.strided
        ):
            raise ValueError("Observation validity must be a boolean tensor with one slot per point")
        for name in ("visibility", "confidence", "presence"):
            values = getattr(self, name)
            if values is not None:
                self._validate_values(values, name)
                available = (values >= 0) & (values <= 1)
                if not available.any() or not (available | (values == -1)).all():
                    raise ValueError(f"{name} requires scores in [0, 1] and -1 for unavailable slots")
        if self.weights is not None:
            self._validate_values(self.weights, "weights")
            if (self.weights < 0).any():
                raise ValueError("Dense fitting weights must be nonnegative")
        if self.alignment is not None:
            if not isinstance(self.alignment, AlignmentMetadata):
                raise ValueError("Alignment must be an AlignmentMetadata record")
            for name, shape in (("source_rect", (4,)), ("source_landmarks", (DFL_LANDMARK_COUNT, 2)),
                                ("image_to_face_mat", (2, 3))):
                value = getattr(self.alignment, name)
                if value is not None:
                    _validate_tensor(value, name, shape, like=self.landmarks)
                    if value.shape != shape:
                        raise ValueError(f"{name} must have shape {shape}")
            for value in (self.alignment.face_type, self.alignment.source_filename):
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError("Alignment text fields must be nonempty strings when supplied")
        for camera in (self.known_camera, self.camera_initialization):
            if camera is not None and (
                _camera_batch(camera, self.landmarks) or camera.image_size != self.image_size
            ):
                raise ValueError("A camera must be unbatched and match the observation raster")

    def _validate_values(self, values, name):
        shape = (len(self.schema.points),)
        _validate_tensor(values, name, shape, like=self.landmarks)
        if values.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    @property
    def coordinate_convention(self):
        return self.schema.coordinate_convention

    def source_pixels(self):
        if self.coordinate_convention == "source_pixels":
            return self.landmarks
        matrix = None if self.alignment is None else self.alignment.image_to_face_mat
        return aligned_to_source(self.landmarks, matrix)


def subset_mask(mask: Tensor, subset: Sequence[int] | None) -> Tensor:
    """Apply an explicitly selected unique index subset in schema order."""
    if subset is None:
        return mask
    indices = tuple(subset)
    if any(type(index) is not int or not 0 <= index < len(mask) for index in indices):
        raise ValueError(f"Subset indices must be integers in [0, {len(mask) - 1}]")
    if len(set(indices)) != len(indices):
        raise ValueError("Subset indices must be unique")
    selected = torch.zeros_like(mask)
    selected[torch.tensor(indices, dtype=torch.long, device=mask.device)] = True
    return selected & mask
