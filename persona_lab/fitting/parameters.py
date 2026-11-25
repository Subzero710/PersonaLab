"""Native GNM selections and gauge-constrained fitting parameters."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from gnm.shape import gnm_utils
import torch
from torch import Tensor, nn

from ..benchmark.dataset import _validate_model
from ..benchmark.model import BenchmarkModel
from ..camera import Camera, PerspectiveCamera, WeakPerspectiveCamera, _camera_batch
from ..coordinates import _validate_tensor
from .configuration import FitConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationInitialization:
    """Explicit initial values in the complete native spaces."""

    expression: Tensor | None = None
    rotations: Tensor | None = None
    translation: Tensor | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class FitInitialization:
    identity: Tensor | None = None
    observations: Mapping[str, ObservationInitialization] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParameterSelection:
    identity: tuple[str, ...]
    expression: tuple[str, ...]
    joints: tuple[str, ...]
    root_translation: bool
    fixed_identity: tuple[str, ...]
    fixed_expression: tuple[str, ...]
    fixed_joints: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class TensorState:
    identity: Tensor
    expression: Tensor
    rotations: Tensor
    translation: Tensor
    cameras: tuple[Camera, ...]


def validate_model(model):
    if not isinstance(model, BenchmarkModel):
        raise ValueError("Use load_benchmark_model to load the verified GNM v3.0 Head")
    _validate_model(model.metadata)
    head = model.head
    dimensions = (
        head.identity_dim, head.expression_dim, head.num_joints, head.num_vertices,
        len(model.triangles),
    )
    if (
        dimensions != (253, 383, 4, 17821, 35324)
        or tuple(head.joint_names) != ("neck", "head", "left_eye", "right_eye")
        or head.version.value != model.metadata["version"]
        or head.variant.value != model.metadata["variant"]
    ):
        raise ValueError("Incompatible GNM v3.0 Head parameter spaces or topology")
    for kind in ("identity", "expression", "joint"):
        if list(getattr(head, f"{kind}_names")) != model.metadata[f"{kind}_names"]:
            raise ValueError(f"GNM {kind} names differ from verified model metadata")
    for kind, field_name in (
        ("identity", "identity_dim"), ("expression", "expression_dim"), ("joint", "num_joints"),
        ("vertex", "num_vertices"),
    ):
        if model.metadata[f"{kind}_count"] != getattr(head, field_name):
            raise ValueError(f"GNM {kind} count differs from verified model metadata")
    if model.metadata["triangle_count"] != dimensions[-1]:
        raise ValueError("GNM triangle count differs from verified model metadata")
    # GNM is an unhashable dataclass and registers its model tensors directly.
    # Recursive Module traversal would try to hash the GNM instance.
    if tuple(head.children()):
        raise ValueError("Expected a flat GNM Head module without submodules")
    if any(tensor.requires_grad for tensor in (
        *head.parameters(recurse=False), *head.buffers(recurse=False),
    )):
        raise ValueError("GNM model tensors must be fixed")
    if model.reference.dtype not in (torch.float32, torch.float64) or (
        model.reference.device.type not in ("cpu", "cuda")
    ):
        raise ValueError("Fitting requires a float32 or float64 CPU or CUDA model")


def _select(names, regions, train, fixed):
    def expand(items):
        selected = set()
        for name in items:
            if name in names:
                selected.add(names.index(name))
            elif name in regions:
                selected.update(int(index) for index in regions[name])
            else:
                raise ValueError(f"Unknown native coefficient or region {name!r}")
        return selected

    indices = sorted(expand(train) - expand(fixed))
    return tuple(names[index] for index in indices)


def resolve_selection(model, config: FitConfig) -> ParameterSelection:
    head = model.head
    identity_names, expression_names = list(head.identity_names), list(head.expression_names)
    identity = _select(identity_names, gnm_utils.identity_regions_indices(head),
                       config.train_identity, config.fixed_identity)
    expression = _select(expression_names, gnm_utils.expression_regions_indices(head),
                         config.train_expression, config.fixed_expression)
    if len(expression) == head.expression_dim and config.expression_weight == 0:
        raise ValueError("Training the complete expression basis requires positive regularization")
    joints = ("head",) if config.train_head else ()
    return ParameterSelection(
        identity, expression, joints, config.train_root,
        tuple(name for name in identity_names if name not in identity),
        tuple(name for name in expression_names if name not in expression),
        tuple(name for name in head.joint_names if name not in joints),
    )


def native_tensor(value, shape, reference, name):
    if value is None:
        return reference.new_zeros(shape)
    _validate_tensor(value, name, shape, like=reference)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return value.detach().clone()


def copy_camera(camera: Camera, *, device=None) -> Camera:
    names = ("rotation", "translation")
    if isinstance(camera, PerspectiveCamera):
        names += ("focal_length", "principal_point")
    else:
        names += ("scale",)
    values = {}
    for name in names:
        tensor = getattr(camera, name).detach()
        values[name] = tensor.to(device=device or tensor.device, copy=True)
    return replace(camera, **values)


def observation_camera(obs, config):
    """Resolve a fixed camera or explicit seed; otherwise return the requested model."""
    if config.camera_mode == "known":
        if obs.known_camera is None or obs.camera_initialization is not None:
            raise ValueError(f"{obs.observation_id} requires exactly one fixed known camera")
        camera = obs.known_camera
    else:
        if obs.known_camera is not None:
            raise ValueError(f"{obs.observation_id} has a known camera in estimated-camera mode")
        camera = obs.camera_initialization
        if camera is None:
            if config.camera_model is None:
                raise ValueError(f"{obs.observation_id} requires an explicit estimated camera model")
            return config.camera_model, None

    if _camera_batch(camera, obs.landmarks) or camera.image_size != obs.image_size:
        raise ValueError(f"{obs.observation_id} requires an unbatched camera with its raster size")
    kind = "perspective" if isinstance(camera, PerspectiveCamera) else "weak-perspective"
    if config.camera_model is not None and config.camera_model != kind:
        raise ValueError(f"{obs.observation_id} initial camera contradicts its configured model")
    if config.train_root and kind != "perspective":
        raise ValueError(f"{obs.observation_id} requires a fixed perspective camera to train root translation")
    if config.camera_mode == "estimated" and kind == "perspective" and camera.translation[2] <= 0:
        raise ValueError(f"{obs.observation_id} requires positive initial perspective translation Z")
    return kind, copy_camera(camera)


class FittingParameters:
    """Only selected entries are optimizer leaves. Fixed entries live in the initial state."""

    def __init__(self, model, observations, initial, selection, config):
        self.initial = initial
        self.selection = selection
        self.config = config
        self.observation_ids = observations.ids
        self.short_edges = observations.short_edges
        reference = model.reference
        head = model.head
        self.identity_indices = torch.tensor(
            [list(head.identity_names).index(name) for name in selection.identity],
            dtype=torch.long, device=reference.device,
        )
        self.expression_indices = torch.tensor(
            [list(head.expression_names).index(name) for name in selection.expression],
            dtype=torch.long, device=reference.device,
        )
        self.head_index = tuple(head.joint_names).index("head")
        self.head_indices = torch.tensor([self.head_index], dtype=torch.long, device=reference.device)
        self.identity = self._parameter(initial.identity.index_select(0, self.identity_indices))
        self.expression = self._parameter(initial.expression.index_select(1, self.expression_indices))
        self.pose = nn.Parameter(initial.rotations[:, self.head_index].clone()) if selection.joints else None
        self.root = nn.Parameter(initial.translation.clone()) if selection.root_translation else None
        self.camera_delta = (
            nn.Parameter(reference.new_zeros((len(observations), 3)))
            if config.camera_mode == "estimated" else None
        )

    @staticmethod
    def _parameter(values):
        return nn.Parameter(values.clone()) if values.numel() else None

    def groups(self):
        groups = {
            "identity": (self.identity,), "expression": (self.expression,),
            "pose": (self.pose, self.root), "camera": (self.camera_delta,),
        }
        return {name: tuple(p for p in values if p is not None) for name, values in groups.items()}

    def _camera(self, index):
        camera = self.initial.cameras[index]
        if self.camera_delta is None:
            return camera
        offsets = self.camera_delta[index]
        if isinstance(camera, WeakPerspectiveCamera):
            return replace(
                camera, translation=camera.translation + offsets[:2] * self.short_edges[index],
                scale=camera.scale * offsets[2].exp(),
            )
        initial_xy, initial_z = camera.translation[:2], camera.translation[2]
        depth = initial_z * offsets[2].exp()
        if not torch.isfinite(depth) or depth <= 0:
            identifier = self.observation_ids[index]
            raise ValueError(f"{identifier} has invalid estimated perspective translation Z")
        translation = torch.cat((initial_xy + offsets[:2] * initial_z,
                                 depth.unsqueeze(0)))
        return replace(camera, translation=translation)

    def state(self, start, stop):
        identity = self.initial.identity
        if self.identity is not None:
            identity = identity.index_copy(0, self.identity_indices, self.identity)
        expression = self.initial.expression[start:stop]
        if self.expression is not None:
            expression = expression.index_copy(1, self.expression_indices, self.expression[start:stop])
        rotations = self.initial.rotations[start:stop]
        if self.pose is not None:
            rotations = rotations.index_copy(1, self.head_indices, self.pose[start:stop, None, :])
        translation = self.initial.translation[start:stop] if self.root is None else self.root[start:stop]
        cameras = tuple(self._camera(i) for i in range(start, stop))
        return TensorState(identity, expression, rotations, translation, cameras)

    def penalties(self):
        zero = self.initial.identity.new_zeros(())
        identity = zero if self.identity is None else self.identity.square().mean()
        expression = zero if self.expression is None else self.expression.square().mean()
        pose = zero
        if self.pose is not None:
            pose = (self.pose - self.initial.rotations[:, self.head_index]).square().mean()
        if self.root is not None:
            pose = pose + (self.root - self.initial.translation).square().mean()
        camera = zero if self.camera_delta is None else self.camera_delta.square().mean()
        config = self.config
        return (
            identity * config.identity_weight, expression * config.expression_weight,
            pose * config.pose_weight, camera * config.camera_weight,
        )


def stack_cameras(cameras):
    """Stack same-type cameras with a common image size for the projection API."""
    camera = cameras[0]

    def stack(name):
        return torch.stack([getattr(item, name) for item in cameras])

    rotation, translation = stack("rotation"), stack("translation")
    if isinstance(camera, PerspectiveCamera):
        return PerspectiveCamera(camera.image_size, rotation, translation,
                                 stack("focal_length"), stack("principal_point"))
    return WeakPerspectiveCamera(camera.image_size, rotation, translation, stack("scale"))
