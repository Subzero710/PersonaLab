"""Oracle native surface observations for controlled numerical validation."""

from collections import deque
from itertools import zip_longest
from dataclasses import dataclass

import numpy as np
import torch

from .benchmark.model import MODEL_SHA256, MODEL_URL
from .camera import PerspectiveCamera, project
from .coordinates import ImageSize, gnm_to_frontal_axes
from .dense_observations import DenseObservation, ImageProvenance, ModelAsset, ObservationPoint, ObservationSchema
from .surface_correspondences import NativeLocation, SurfaceCorrespondence, SurfaceCorrespondences


PROVIDER = "gnm-native-surface-oracle"
PROVIDER_VERSION = "1"
ASSET = ModelAsset("3.0-head", MODEL_URL, MODEL_SHA256)
_REFERENCE_GROUPS = (
    ("pupil_surface", ("pupils",)),
    ("iris_surface", ("irises",)),
    ("sclera_surface", ("scleras",)),
    ("upper_teeth", ("teeth", "upper_teeth_and_gums")),
    ("lower_teeth", ("teeth", "lower_teeth_and_gums")),
    ("gingiva", ("gums",)),
    ("tongue", ("tongue",)),
    ("mouth_interior", ("mouth_sock",)),
    ("ears", ("ears",)),
    ("upper_lip", ("upper_lip",)),
    ("lower_lip", ("lower_lip",)),
    ("forehead", ("forehead_region",)),
    ("nose", ("nose_region",)),
    ("chin", ("chin_region",)),
    ("left_cheek", ("left_cheek_region",)),
    ("right_cheek", ("right_cheek_region",)),
    ("left_orbital_skin", ("left_orbital_region", "skin_exterior")),
    ("right_orbital_skin", ("right_orbital_region", "skin_exterior")),
    ("other_face_skin", ("hockey_mask", "skin_exterior")),
)


def oracle_definition(model, count):
    """Sample fixed native vertices and triangle interiors across published groups."""
    if type(count) is not int or count < 1:
        raise ValueError("Reference point count must be positive")
    rng = np.random.default_rng(0)
    head, triangles = model.head, model.triangles
    groups = dict(zip(head.vertex_group_names, head.vertex_groups.detach().cpu().numpy(), strict=True))
    used_vertices = np.zeros(head.num_vertices, dtype=np.bool_)
    used_triangles = np.zeros(len(triangles), dtype=np.bool_)
    strata = []
    for region, names in _REFERENCE_GROUPS:
        mask = np.logical_and.reduce([groups[name] > 0 for name in names])
        vertex_mask = mask & ~used_vertices
        triangle_mask = mask[triangles].all(axis=1) & ~used_triangles
        used_vertices |= vertex_mask
        used_triangles |= triangle_mask
        vertices = [NativeLocation("vertex", index)
                    for index in rng.permutation(np.flatnonzero(vertex_mask)).tolist()]
        faces = [NativeLocation("triangle", index, (0.2, 0.3, 0.5))
                 for index in rng.permutation(np.flatnonzero(triangle_mask)).tolist()]
        if not vertices or not faces:
            raise ValueError(f"Pinned topology lacks a vertex or triangle stratum for {region}")
        interleaved = deque(location for pair in zip_longest(vertices, faces)
                            for location in pair if location is not None)
        strata.append((region, names, interleaved))

    points, entries = [], []
    while any(pool for _, _, pool in strata):
        for region, names, pool in strata:
            if not pool:
                continue
            location = pool.popleft()
            identifier = f"oracle:{location.kind}:{location.index}"
            points.append(ObservationPoint(identifier, region))
            entries.append(SurfaceCorrespondence(
                identifier, location, names, f"{MODEL_URL} vertex_groups and triangles",
                "Oracle projection of a known surface location; image visibility is unavailable",
            ))
            if len(points) == count:
                schema = ObservationSchema(f"gnm-surface-oracle-{count}", 1, "source_pixels", tuple(points))
                return SurfaceCorrespondences(schema, model.metadata["topology_sha256"], tuple(entries))
    raise ValueError(f"The declared reference contains only {len(points)} distinct native locations")


@dataclass(frozen=True, slots=True)
class OracleState:
    identity: torch.Tensor
    expressions: torch.Tensor
    rotations: torch.Tensor
    translations: torch.Tensor
    cameras: tuple[PerspectiveCamera, ...]


def oracle_state(model, seed, view_count):
    """Sample all native identity, expression and joint coordinates from one seed."""
    if type(seed) is not int or not 0 <= seed < 2**64 or type(view_count) is not int or view_count < 1:
        raise ValueError("Oracle seeds must fit uint64 and view count must be positive")
    rng = np.random.default_rng(seed)
    head = model.head
    identity = model.tensor(rng.normal(0, 0.15, size=head.identity_dim))
    expression = model.tensor(rng.normal(0, 0.15, size=(view_count, head.expression_dim)))
    rotation = model.tensor(rng.normal(0, 0.025, size=(view_count, head.num_joints, 3)))
    yaw = np.deg2rad(np.resize([0., -38., 66., 28., -63.], view_count))
    rotation[:, tuple(head.joint_names).index("head"), 1] += model.tensor(yaw)
    # The root translation gauge is fixed when estimating camera offsets.
    translation = model.reference.new_zeros((view_count, 3))
    size = ImageSize(960, 720)
    cameras = []
    for _ in range(view_count):
        focal = 800. + rng.uniform(-40, 40)
        cameras.append(PerspectiveCamera(
            size, model.tensor([0., 0., 0.]),
            model.tensor([rng.uniform(-0.035, 0.035), 0.235, rng.uniform(1.15, 1.35)]),
            model.tensor([focal, focal]), model.tensor([479.5, 359.5]),
        ))
    return OracleState(identity, expression, rotation, translation, tuple(cameras))


@torch.no_grad()
def oracle_observations(model, definition, seed, state):
    """Generate independent targets by evaluating Google's full mesh once per view."""
    bound = definition.bind(model)
    if (definition.schema.coordinate_convention != "source_pixels"
        or len(bound.slots) != len(definition.schema.points)):
        raise ValueError("Oracle export needs a fully specified source-pixel surface definition")
    output = []
    configuration = {"seed": seed, "views": len(state.cameras), "projection": "perspective",
                     "device": str(model.reference.device), "visibility": "unspecified oracle surface"}
    for index, camera in enumerate(state.cameras):
        vertices = model.head(
            identity=state.identity, expression=state.expressions[index], rotations=state.rotations[index],
            translation=state.translations[index],
        )
        pixels = project(gnm_to_frontal_axes(bound.evaluate(vertices)), camera)
        valid = ((pixels >= 0) & (pixels < pixels.new_tensor((camera.image_size.width,
                                                             camera.image_size.height)))).all(dim=-1)
        identifier = f"oracle:{seed}:{index:04d}"
        output.append(DenseObservation(
            observation_id=identifier, schema=definition.schema, provider=PROVIDER,
            provider_version=PROVIDER_VERSION, asset=ASSET,
            source=ImageProvenance(identifier, None, camera.image_size, kind="synthetic"),
            extraction=configuration, image_size=camera.image_size, landmarks=pixels,
            validity=valid, known_camera=camera,
        ))
    return tuple(output)
