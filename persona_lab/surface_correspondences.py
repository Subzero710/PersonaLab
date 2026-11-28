"""Schema-ordered surface correspondences on the pinned GNM Head topology."""

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np
import torch

from .benchmark.model import MODEL_SHA256
from .benchmark.storage import digest_arrays, is_sha256, read_json, require_keys
from .dense_observations import ObservationSchema
from .gnm_geometry import SparseHeadLandmarks, load_sparse_landmarks


# These are compatibility checks on the target anatomy. They do not establish
# the real-image landmark association, which requires independent review.
_REQUIRED_GROUPS = {
    "iris_boundary": (("irises",),),
    "iris_center": (("pupils",),),
    "eyebrow": (("left_brow_region",), ("right_brow_region",), ("middle_brow_region",)),
    "upper_eyelid": (("left_orbital_region", "skin_exterior"), ("right_orbital_region", "skin_exterior")),
    "lower_eyelid": (("left_orbital_region", "skin_exterior"), ("right_orbital_region", "skin_exterior")),
    "eyelid_corner": (("left_orbital_region", "skin_exterior"), ("right_orbital_region", "skin_exterior")),
    "inner_lip": (("upper_lip_region",), ("lower_lip_region",)),
    "outer_lip": (("upper_lip_region",), ("lower_lip_region",)),
}


@dataclass(frozen=True, slots=True)
class NativeLocation:
    """An official landmark index, vertex index, or oriented native triangle row."""

    kind: str
    index: int
    weights: tuple[float, float, float] | None = None

    def __post_init__(self):
        if self.kind not in ("landmark", "vertex", "triangle") or (
            type(self.index) is not int or self.index < 0
        ):
            raise ValueError("Expected a nonnegative landmark, vertex or triangle index")
        if self.kind == "triangle":
            w = self.weights
            if not isinstance(w, tuple) or len(w) != 3 or any(
                type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                for value in w
            ) or not math.isclose(sum(w), 1, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Triangle needs three finite weights in [0, 1] with sum one")
        elif self.weights is not None:
            raise ValueError("Only an explicit triangle takes custom weights")


@dataclass(frozen=True, slots=True)
class SurfaceCorrespondence:
    point_id: str
    location: NativeLocation | None
    native_groups: tuple[str, ...]
    evidence: str
    limitations: str
    minimum_visibility: float | None = None

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.point_id, self.evidence, self.limitations)):
            raise ValueError("A correspondence needs a point ID, evidence and limitations")
        if self.location is not None and not isinstance(self.location, NativeLocation):
            raise ValueError("Expected a NativeLocation or an explicitly unsupported slot")
        if not isinstance(self.native_groups, tuple) or any(
            not isinstance(group, str) or not group.strip() for group in self.native_groups
        ) or len(set(self.native_groups)) != len(self.native_groups):
            raise ValueError("Native group names must be unique nonempty strings")
        if bool(self.native_groups) != (self.location is not None):
            raise ValueError("Each mapped point requires native group evidence")
        if self.minimum_visibility is not None and (
            self.location is None or type(self.minimum_visibility) not in (float, int)
            or not 0 <= self.minimum_visibility <= 1
        ):
            raise ValueError("A visibility restriction requires a mapped point and a score in [0, 1]")


@dataclass(frozen=True, slots=True)
class SurfaceCorrespondences:
    schema: ObservationSchema
    topology_sha256: str
    entries: tuple[SurfaceCorrespondence, ...]

    def __post_init__(self):
        if not isinstance(self.schema, ObservationSchema) or not is_sha256(self.topology_sha256):
            raise ValueError("Correspondences require an explicit schema and topology SHA-256")
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, SurfaceCorrespondence) for entry in self.entries
        ) or tuple(entry.point_id for entry in self.entries) != self.schema.point_ids:
            raise ValueError("Correspondences must cover every schema slot in schema order")

    def to_record(self):
        return {
            "format": "gnm-surface-correspondences-v1",
            "schema": self.schema.identifier,
            "schema_version": self.schema.version,
            "model_sha256": MODEL_SHA256,
            "topology_sha256": self.topology_sha256,
            "entries": [asdict(entry) for entry in self.entries],
        }

    def bind(self, model):
        """Resolve locations against the official asset, preserving provider slot order."""
        head, triangles = model.head, model.triangles
        if model.metadata["asset_sha256"] != MODEL_SHA256 or (
            digest_arrays({"triangles": triangles}) != self.topology_sha256
            or model.metadata["topology_sha256"] != self.topology_sha256
            or not np.array_equal(triangles, head.triangles.detach().cpu().numpy())
        ):
            raise ValueError("Correspondence topology does not match the pinned GNM Head")
        official = load_sparse_landmarks(head)
        triangle_set = {tuple(sorted(row)) for row in triangles.tolist()}
        groups = dict(zip(head.vertex_group_names, head.vertex_groups.detach().cpu().numpy(), strict=True))
        mirrors = head.mirror_indices.detach().cpu().numpy()
        indices, weights, mapped_slots = [], [], []
        used = set()
        for slot, (point, entry) in enumerate(zip(self.schema.points, self.entries, strict=True)):
            if entry.location is None:
                continue
            location = entry.location
            index = location.index
            if location.kind == "landmark" and index < len(official.indices):
                native, w = official.indices[index], official.weights[index]
                if tuple(sorted(native.tolist())) not in triangle_set:
                    raise ValueError(f"{point.identifier} does not reference a native triangle")
            elif location.kind == "triangle" and index < len(triangles):
                native, w = triangles[index], np.asarray(location.weights)
            elif location.kind == "vertex" and index < head.num_vertices:
                native, w = np.full(3, index), np.array([1., 0., 0.])
            else:
                raise ValueError(f"{point.identifier} references a location outside the native mesh")

            active = native[w > 0]
            if self.schema.identifier in ("mediapipe-face-478", "mediapipe-face-478-dfl"):
                if point.region == "facial_oval":
                    raise ValueError(f"{point.identifier} is a silhouette slot with no reviewed viewpoint model")
                alternatives = _REQUIRED_GROUPS.get(point.region, (("skin_exterior",),))
                if not any(all((groups[group][active] > 0).all() for group in option) for option in alternatives):
                    raise ValueError(f"{point.identifier} does not lie in the corresponding native anatomy")
            if any(group not in groups or not (groups[group][active] > 0).all()
                   for group in entry.native_groups):
                raise ValueError(f"{point.identifier} disagrees with its declared native groups")
            if point.side in ("left", "right"):
                opposite = "right" if point.side == "left" else "left"
                if not (groups[point.side][active] > 0).all() or (
                    (groups[opposite][active] > 0).all()
                    or not (groups[opposite][mirrors[active]] > 0).all()
                ):
                    raise ValueError(f"{point.identifier} disagrees with the native side or mirror table")
            location_key = tuple(sorted(zip(active.tolist(), np.asarray(w)[w > 0].tolist())))
            if location_key in used:
                raise ValueError(f"{point.identifier} duplicates another native surface location")
            used.add(location_key)
            indices.append(native)
            weights.append(w)
            mapped_slots.append(slot)
        return BoundSurface(self, tuple(mapped_slots), np.asarray(indices, dtype=np.int64).reshape(-1, 3),
                            np.asarray(weights, dtype=np.float64).reshape(-1, 3))


@dataclass(frozen=True, slots=True, eq=False)
class BoundSurface:
    definition: SurfaceCorrespondences
    slots: tuple[int, ...]
    indices: np.ndarray
    weights: np.ndarray

    def sparse_geometry(self, model):
        if not self.slots:
            raise ValueError("No verified correspondences available for native geometric evaluation")
        return SparseHeadLandmarks(model.head, indices=self.indices, weights=self.weights)

    def evaluate(self, vertices):
        """Evaluate supported points on a full mesh in schema order, with arbitrary batches."""
        indices = torch.as_tensor(self.indices, device=vertices.device)
        weights = torch.as_tensor(self.weights, device=vertices.device, dtype=vertices.dtype)
        return (vertices[..., indices, :] * weights[..., None]).sum(dim=-2)

    def eligible(self, observation):
        mask = torch.zeros_like(observation.validity)
        if self.slots:
            mask[list(self.slots)] = True
        for slot in self.slots:
            entry = self.definition.entries[slot]
            minimum = entry.minimum_visibility
            if minimum is not None and observation.validity[slot]:
                if observation.visibility is None or observation.visibility[slot] < 0:
                    raise ValueError(f"{entry.point_id} requires an actual visibility score")
                mask[slot] = observation.visibility[slot] >= minimum
        return mask & observation.validity


def load_correspondences(path, schema):
    """Load an explicitly supplied mapping, with geometry validated by bind()."""
    record = read_json(Path(path))
    require_keys(record, {"format", "schema", "schema_version", "model_sha256",
                          "topology_sha256", "entries"}, "Correspondence definition")
    if (record["format"], record["schema"], record["schema_version"], record["model_sha256"]) != (
        "gnm-surface-correspondences-v1", schema.identifier, schema.version, MODEL_SHA256
    ) or type(record["schema_version"]) is not int or not isinstance(record["entries"], list):
        raise ValueError("Correspondence definition does not match the requested schema and model")
    entries = []
    for item in record["entries"]:
        require_keys(item, {"point_id", "location", "native_groups", "evidence", "limitations",
                            "minimum_visibility"}, "Correspondence entry")
        if not isinstance(item["native_groups"], list):
            raise ValueError("Native groups must be a JSON array")
        location = item["location"]
        if location is not None:
            require_keys(location, {"kind", "index", "weights"}, "Native location")
            weights = location["weights"]
            if weights is not None and not isinstance(weights, list):
                raise ValueError("Triangle weights must be a JSON array")
            location = NativeLocation(location["kind"], location["index"],
                                      None if weights is None else tuple(weights))
        entries.append(SurfaceCorrespondence(item["point_id"], location, tuple(item["native_groups"]),
                                             item["evidence"], item["limitations"],
                                             item["minimum_visibility"]))
    return SurfaceCorrespondences(schema, record["topology_sha256"], tuple(entries))
