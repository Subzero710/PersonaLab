"""Real and synthetic landmark observations in original DeepFaceLab order."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal, NamedTuple

import numpy as np
import torch
from torch import Tensor

from .camera import Camera, _camera_batch
from .coordinates import ImageSize, _FLOAT_DTYPES, _validate_tensor
from .coordinates import aligned_to_source, normalized_to_pixels
from .correspondences import CorrespondenceMap, DFL_LANDMARK_COUNT, DFL_TO_GNM
from .dfl import DFLFaceSample
from .gnm_geometry import GNM_LANDMARK_COUNT


@dataclass(frozen=True, slots=True, eq=False)
class AlignmentMetadata:
    """DFL source name, face type and optional source pixel/affine metadata."""

    face_type: str | None = None
    source_filename: str | None = None
    source_rect: Tensor | None = None
    source_landmarks: Tensor | None = None
    image_to_face_mat: Tensor | None = None


class LandmarkPairs(NamedTuple):
    """Selected observed/projected pixel pairs, weights and original indices."""

    observed: Tensor
    projected: Tensor
    weights: Tensor
    dfl_indices: Tensor
    gnm_indices: Tensor


def _check_mask(mask: Tensor, name: str, device: torch.device) -> None:
    if (
        not isinstance(mask, Tensor)
        or mask.shape != (DFL_LANDMARK_COUNT,)
        or mask.dtype != torch.bool
        or mask.device != device
        or mask.layout != torch.strided
    ):
        raise ValueError(f"{name} must be a (68,) boolean tensor on the landmark device")


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class LandmarkObservation:
    """One aligned crop or synthetic raster, with 68 ordered 2D landmarks.

    Omitted validity and weights mean all slots initially valid and unit
    weight. Correspondence, subset and pose masks further restrict
    fitting. Known cameras and user-provided starting values remain separate.
    """

    observation_id: str
    image_size: ImageSize
    landmarks: Tensor
    coordinate_convention: Literal["aligned_pixels", "normalized_aligned"] = "aligned_pixels"
    correspondences: CorrespondenceMap = DFL_TO_GNM
    validity: Tensor | None = None
    weights: Tensor | None = None
    alignment: AlignmentMetadata | None = None
    known_camera: Camera | None = None
    camera_initialization: Camera | None = None

    landmark_ordering: ClassVar[str] = "deepfacelab_68"

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id.strip():
            raise ValueError("Observation identifier must be nonempty")
        if not isinstance(self.image_size, ImageSize):
            raise ValueError("Observation image_size must be an ImageSize")
        if self.coordinate_convention not in ("aligned_pixels", "normalized_aligned"):
            raise ValueError("Unknown observation coordinate convention")
        if not isinstance(self.correspondences, CorrespondenceMap):
            raise ValueError("Observation requires a CorrespondenceMap")
        _validate_tensor(self.landmarks, "Observed landmarks", (DFL_LANDMARK_COUNT, 2))
        if self.landmarks.ndim != 2:
            raise ValueError("An observation stores exactly one set of (68, 2) landmarks")

        device = self.landmarks.device
        if self.validity is None:
            validity = torch.ones(DFL_LANDMARK_COUNT, dtype=torch.bool, device=device)
            object.__setattr__(self, "validity", validity)
        _check_mask(self.validity, "Observation validity", device)
        if self.weights is None:
            object.__setattr__(self, "weights", self.landmarks.new_ones(DFL_LANDMARK_COUNT))
        _validate_tensor(
            self.weights, "Landmark weights", (DFL_LANDMARK_COUNT,), like=self.landmarks
        )
        if self.weights.ndim != 1 or (self.weights < 0).any():
            raise ValueError("Landmark weights must be nonnegative and shaped (68,)")

        if self.alignment is not None:
            if not isinstance(self.alignment, AlignmentMetadata):
                raise ValueError("Observation alignment must be AlignmentMetadata")
            for name, shape in (
                ("source_rect", (4,)),
                ("source_landmarks", (DFL_LANDMARK_COUNT, 2)),
                ("image_to_face_mat", (2, 3)),
            ):
                value = getattr(self.alignment, name)
                if value is not None:
                    _validate_tensor(value, name, shape, like=self.landmarks)
                    if value.shape != shape:
                        raise ValueError(f"{name} must have unbatched shape {shape}")

        for camera in (self.known_camera, self.camera_initialization):
            if camera is not None:
                if _camera_batch(camera, self.landmarks) or camera.image_size != self.image_size:
                    raise ValueError(
                        "An observation camera must be unbatched and match the raster size"
                    )

    @classmethod
    def from_dfl(
        cls,
        sample: DFLFaceSample,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        validity: Tensor | None = None,
        weights: Tensor | None = None,
        correspondences: CorrespondenceMap = DFL_TO_GNM,
        known_camera: Camera | None = None,
        camera_initialization: Camera | None = None,
    ) -> "LandmarkObservation":
        """Copy Issue #2's numeric metadata to the requested device and dtype.

        The original JPEG is never reopened. XSeg and other non-alignment
        metadata remain accessible on the source sample.
        """
        if not isinstance(sample, DFLFaceSample):
            raise ValueError("Expected a validated DeepFaceLab sample")
        if dtype not in _FLOAT_DTYPES:
            raise ValueError("Observation dtype must be float32 or float64")

        numpy_dtype = np.float64 if dtype == torch.float64 else np.float32

        def copy(values: np.ndarray | None) -> Tensor | None:
            if values is None:
                return None
            # Copy read-only, potentially non-native-endian ingestion arrays.
            return torch.from_numpy(np.array(values, dtype=numpy_dtype, copy=True)).to(device)

        return cls(
            observation_id=sample.sample_id,
            image_size=ImageSize(*sample.image_size),
            landmarks=copy(sample.landmarks),
            validity=validity,
            weights=weights,
            correspondences=correspondences,
            alignment=AlignmentMetadata(
                face_type=sample.face_type,
                source_filename=sample.source_filename,
                source_rect=copy(sample.source_rect),
                source_landmarks=copy(sample.source_landmarks),
                image_to_face_mat=copy(sample.image_to_face_mat),
            ),
            known_camera=known_camera,
            camera_initialization=camera_initialization,
        )

    @property
    def supported_mask(self) -> Tensor:
        return self.correspondences.supported_mask(device=self.landmarks.device)

    def aligned_pixels(self) -> Tensor:
        if self.coordinate_convention == "normalized_aligned":
            return normalized_to_pixels(self.landmarks, self.image_size)
        return self.landmarks

    def source_pixels(self) -> Tensor:
        """Reconstruct source pixels from aligned observations using the affine."""
        matrix = None if self.alignment is None else self.alignment.image_to_face_mat
        return aligned_to_source(self.aligned_pixels(), matrix)

    def fitting_mask(
        self,
        *,
        subset: Sequence[int] | None = None,
        pose_validity: Tensor | None = None,
    ) -> Tensor:
        """Return the 68-slot mask that an objective must apply.

        All 68 default correspondences are eligible, including the jawline.
        False validity, zero weights and unsupported or disabled entries exclude.
        Subset indices refer to DeepFaceLab order.
        """
        device = self.landmarks.device
        eligible = torch.tensor([
            entry.supported and entry.enabled_by_default
            for entry in self.correspondences.entries
        ], device=device)
        mask = self.validity & eligible & (self.weights > 0)
        if pose_validity is not None:
            _check_mask(pose_validity, "Pose validity", device)
            mask = mask & pose_validity
        if subset is not None:
            indices = tuple(subset)
            if any(
                type(index) is not int or not 0 <= index < DFL_LANDMARK_COUNT
                for index in indices
            ):
                raise ValueError("Subset indices must be integers in [0, 67]")
            if len(set(indices)) != len(indices):
                raise ValueError("Subset indices must be unique")
            selected = torch.zeros_like(mask)
            selected[torch.tensor(indices, device=device, dtype=torch.long)] = True
            mask = mask & selected
        return mask

    def landmark_pairs(
        self,
        projected_gnm: Tensor,
        *,
        subset: Sequence[int] | None = None,
        pose_validity: Tensor | None = None,
    ) -> LandmarkPairs:
        """Gather active pixel pairs from one projection in official GNM order."""
        _validate_tensor(
            projected_gnm, "Projected GNM landmarks", (GNM_LANDMARK_COUNT, 2), like=self.landmarks
        )
        if projected_gnm.ndim != 2:
            raise ValueError("Expected one set of (68, 2) projected landmarks")
        mask = self.fitting_mask(
            subset=subset, pose_validity=pose_validity
        )
        targets = torch.tensor([
            -1 if entry.gnm_index is None else entry.gnm_index
            for entry in self.correspondences.entries
        ], device=projected_gnm.device, dtype=torch.long)
        gnm_indices = targets[mask]
        return LandmarkPairs(
            self.aligned_pixels()[mask],
            projected_gnm.index_select(0, gnm_indices),
            self.weights[mask],
            torch.nonzero(mask, as_tuple=True)[0],
            gnm_indices,
        )
