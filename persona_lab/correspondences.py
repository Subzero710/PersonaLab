"""Anatomical correspondence from DeepFaceLab indices to GNM v3.0 Head."""

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor


DFL_LANDMARK_COUNT = 68


class Region(StrEnum):
    CONTOUR = "facial_contour"
    EYEBROW = "eyebrow"
    NOSE_BRIDGE = "nose_bridge"
    NOSE_BASE = "nose_base"
    EYE = "eye"
    OUTER_MOUTH = "outer_mouth"
    INNER_MOUTH = "inner_mouth"


@dataclass(frozen=True, slots=True)
class LandmarkCorrespondence:
    dfl_index: int
    gnm_index: int | None
    description: str
    region: Region
    # Descriptive risk annotation only; never an automatic fitting exclusion.
    view_dependent: bool = False
    enabled_by_default: bool = True
    limitation: str | None = None

    @property
    def supported(self) -> bool:
        return self.gnm_index is not None


@dataclass(frozen=True, slots=True)
class CorrespondenceMap:
    """One explicit entry per DeepFaceLab index in original observation order."""

    entries: tuple[LandmarkCorrespondence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or len(self.entries) != DFL_LANDMARK_COUNT:
            raise ValueError("A correspondence map requires 68 ordered entries")
        used = set()
        for index, entry in enumerate(self.entries):
            if (
                not isinstance(entry, LandmarkCorrespondence)
                or type(entry.dfl_index) is not int
                or index != entry.dfl_index
            ):
                raise ValueError("Correspondence entries must preserve DeepFaceLab ordering")
            if (
                not isinstance(entry.region, Region)
                or not isinstance(entry.description, str)
                or not entry.description.strip()
                or type(entry.view_dependent) is not bool
                or type(entry.enabled_by_default) is not bool
            ):
                raise ValueError(f"Invalid anatomical metadata at DeepFaceLab index {index}")
            if entry.supported:
                target = entry.gnm_index
                if type(target) is not int or not 0 <= target < 68 or target in used:
                    raise ValueError(f"Invalid or duplicate GNM index at DeepFaceLab index {index}")
                used.add(target)
            if not entry.supported:
                if entry.enabled_by_default or not (
                    isinstance(entry.limitation, str) and entry.limitation.strip()
                ):
                    raise ValueError(
                        "Unsupported entries must be disabled and have an exclusion reason"
                    )

    def supported_mask(self, *, device: torch.device | str) -> Tensor:
        return torch.tensor([entry.supported for entry in self.entries], device=device)


_CONTOUR_LIMIT = (
    "Potential viewpoint-dependent correspondence error; not established for DFL observations. "
    "This annotation does not exclude the landmark."
)
# Compare DFL LandmarksProcessor's indexed template and side definitions with
# the official GNM rest mesh, its vertex groups, and its mirror-index table.
# Left and right below always refer to the subject's anatomy.
_REGIONS = (
    (Region.CONTOUR, (
        "Upper right facial outline", "Right facial outline below the temple",
        "Right upper cheek outline", "Right mid-cheek outline", "Right lower cheek outline",
        "Outer right jawline", "Right mandibular outline", "Right side of the chin",
        "Center of the chin", "Left side of the chin", "Left mandibular outline",
        "Outer left jawline", "Left lower cheek outline", "Left mid-cheek outline",
        "Left upper cheek outline", "Left facial outline below the temple",
        "Upper left facial outline",
    )),
    (Region.EYEBROW, (
        "Right eyebrow lateral end", "Right eyebrow lateral slope", "Right eyebrow arch",
        "Right eyebrow medial slope", "Right eyebrow medial end",
        "Left eyebrow medial end", "Left eyebrow medial slope", "Left eyebrow arch",
        "Left eyebrow lateral slope", "Left eyebrow lateral end",
    )),
    (Region.NOSE_BRIDGE, (
        "Nasal root", "Upper nasal bridge", "Lower nasal bridge", "Nose tip",
    )),
    (Region.NOSE_BASE, (
        "Right alar edge", "Right nostril base", "Midline nasal base",
        "Left nostril base", "Left alar edge",
    )),
    (Region.EYE, (
        "Right lateral eye corner", "Right upper eyelid lateral point",
        "Right upper eyelid medial point", "Right medial eye corner",
        "Right lower eyelid medial point", "Right lower eyelid lateral point",
        "Left medial eye corner", "Left upper eyelid medial point",
        "Left upper eyelid lateral point", "Left lateral eye corner",
        "Left lower eyelid lateral point", "Left lower eyelid medial point",
    )),
    (Region.OUTER_MOUTH, (
        "Right outer mouth corner", "Right upper outer lip lateral point",
        "Right upper outer lip at Cupid's bow", "Center of the upper outer lip",
        "Left upper outer lip at Cupid's bow", "Left upper outer lip lateral point",
        "Left outer mouth corner", "Left lower outer lip lateral point",
        "Left lower outer lip medial point", "Center of the lower outer lip",
        "Right lower outer lip medial point", "Right lower outer lip lateral point",
    )),
    (Region.INNER_MOUTH, (
        "Right inner mouth corner", "Right upper inner lip", "Center of the upper inner lip",
        "Left upper inner lip", "Left inner mouth corner", "Left lower inner lip",
        "Center of the lower inner lip", "Right lower inner lip",
    )),
)

DFL_TO_GNM = CorrespondenceMap(tuple(
    LandmarkCorrespondence(
        dfl_index=index,
        gnm_index=index,
        description=description,
        region=region,
        view_dependent=region == Region.CONTOUR,
        enabled_by_default=True,
        limitation=_CONTOUR_LIMIT if region == Region.CONTOUR else None,
    )
    for index, (region, description) in enumerate(
        (region, description) for region, descriptions in _REGIONS for description in descriptions
    )
))
