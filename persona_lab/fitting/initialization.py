"""Deterministic pose and uncalibrated camera initialization."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import product
import math

from scipy.spatial.transform import Rotation
import torch

from ..camera import PerspectiveCamera, WeakPerspectiveCamera, project, world_to_camera
from ..coordinates import gnm_to_frontal_axes
from ..gnm_geometry import extract_landmarks
from .parameters import (
    FitInitialization, ObservationInitialization, TensorState, native_tensor, observation_camera,
)


@dataclass(frozen=True, slots=True)
class InitializationChoice:
    strategy: str
    candidate: int
    candidates: int
    admissible: int


def _candidates(model, identity, expression, rotations, translation, grid, batch_size):
    if grid:
        angles = list(product((0, -40, 40, -75, 75), (0, -25, 25), (0, -20, 20)))
        offsets = Rotation.from_euler("yxz", angles, degrees=True).as_rotvec()
        rotations = rotations.expand(len(angles), -1, -1).clone()
        rotations[:, tuple(model.head.joint_names).index("head")] = model.tensor(offsets)
    else:
        rotations = rotations.unsqueeze(0)
    landmarks = []
    for batch in rotations.split(batch_size):
        _, points = extract_landmarks(
            model.head, identity=identity, expression=expression.expand(len(batch), -1),
            rotations=batch, translation=translation.expand(len(batch), -1),
        )
        landmarks.append(gnm_to_frontal_axes(points))
    return rotations, torch.cat(landmarks)


def _initial_focal(reference, image_size, config):
    edge = min(image_size.width, image_size.height)
    if config.focal_mode == "fixed":
        return reference.new_full((2,), edge * config.perspective_focal_per_short_edge)
    pixels = edge * config.initial_focal_factor
    if not math.isfinite(pixels) or pixels > torch.finfo(reference.dtype).max:
        raise ValueError("Configured initial focal length is outside the finite range of the fitting dtype")
    focal = reference.new_full((2,), pixels)
    if focal[0] <= 0:
        raise ValueError("Configured initial focal length must be positive in the fitting dtype")
    return focal


def _uncalibrated_cameras(axes, targets, weights, gnm_indices, image_size, kind, config):
    """Fit a camera for each viable pose using weighted landmark centroids."""
    zero = axes.new_zeros(3)
    unit = WeakPerspectiveCamera(image_size, zero, zero[:2], axes.new_ones(()))
    pixels = project(axes, unit)[:, gnm_indices]
    observed_center = (weights[:, None] * targets).sum(0)
    projected_center = (weights[:, None] * pixels).sum(1)
    centered = pixels - projected_center[:, None]
    spread = (weights * centered.square().sum(-1)).sum(-1)
    covariance = (weights * (centered * (targets - observed_center)).sum(-1)).sum(-1)
    if not (torch.isfinite(spread) & torch.isfinite(covariance)).all():
        raise ValueError("Nonfinite landmark moments during camera initialization")
    indices = torch.nonzero((spread > 0) & (covariance > 0), as_tuple=True)[0]
    scale = covariance[indices] / spread[indices]
    finite_scale = torch.isfinite(scale) & (scale > 0)
    indices, scale = indices[finite_scale], scale[finite_scale]
    if kind == "weak-perspective":
        return indices, replace(
            unit, scale=scale,
            translation=observed_center - scale[:, None] * projected_center[indices],
        )

    focal = _initial_focal(axes, image_size, config)
    center = axes.new_tensor(((image_size.width - 1) / 2, (image_size.height - 1) / 2))
    centroid_z = (weights * axes[indices][:, gnm_indices, 2]).sum(-1)
    depth = focal[0] / scale - centroid_z
    positive = torch.isfinite(depth) & (depth > 0) & (
        (axes[indices, :, 2] + depth[:, None]) > 0
    ).all(-1)
    indices, depth = indices[positive], depth[positive, None]
    origin = torch.cat((axes.new_zeros((len(depth), 2)), depth), dim=1)
    camera = PerspectiveCamera(image_size, zero, origin, focal, center)
    if not len(indices):
        return indices, camera
    base = project(axes[indices], camera)[:, gnm_indices]
    shifted = replace(camera, translation=torch.cat((axes.new_ones((len(depth), 2)), depth), dim=1))
    slope = project(axes[indices], shifted)[:, gnm_indices] - base
    translation = (weights[:, None] * slope * (targets - base)).sum(1)
    translation = translation / (weights[:, None] * slope.square()).sum(1)
    return indices, replace(camera, translation=torch.cat((translation, depth), dim=1))


def _choose_pose(data, index, candidates, camera, kind, config, strategy):
    obs = data.observations[index]
    rotations, landmarks = candidates
    mask = data.eligible[index]
    targets, indices, weights = (
        data.pixels[index, mask], data.gnm_indices[index, mask], data.weights[index, mask],
    )
    weights = weights / weights.sum()
    if camera is None:
        candidates, batch_camera = _uncalibrated_cameras(
            landmarks, targets, weights, indices, obs.image_size, kind, config,
        )
    else:
        admissible = torch.ones(len(landmarks), dtype=torch.bool, device=landmarks.device)
        if isinstance(camera, PerspectiveCamera):
            depths = world_to_camera(landmarks, camera.rotation, camera.translation)[..., 2]
            admissible = (depths > 0).all(-1)
        candidates = torch.nonzero(admissible, as_tuple=True)[0]
        batch_camera = camera
    if not len(candidates):
        raise ValueError(f"{obs.observation_id} has no admissible pose and camera initialization")

    projected = project(landmarks[candidates], batch_camera)
    residual = (projected[:, indices] - targets) / data.short_edges[index]
    losses = (weights * residual.square().sum(-1)).sum(-1)
    if not torch.isfinite(losses).all():
        raise ValueError(f"{obs.observation_id} has a nonfinite initialization objective")
    best = int(losses.argmin())
    candidate = int(candidates[best])
    if camera is None:
        camera = replace(batch_camera, translation=batch_camera.translation[best])
        if isinstance(camera, WeakPerspectiveCamera):
            camera = replace(camera, scale=camera.scale[best])
    return rotations[candidate], camera, InitializationChoice(
        strategy, candidate, len(landmarks), len(candidates),
    )


@torch.no_grad()
def initialize(model, data, config, initial):
    if not isinstance(initial, FitInitialization):
        raise ValueError("Expected a FitInitialization with native tensors")
    if not isinstance(initial.observations, Mapping) or set(initial.observations) - set(data.ids) or any(
        not isinstance(item, ObservationInitialization) for item in initial.observations.values()
    ):
        raise ValueError("Observation initialization must reference known fitting IDs")
    ref, head = model.reference, model.head
    identity = native_tensor(initial.identity, (head.identity_dim,), ref, "Initial identity")
    base = ObservationInitialization()
    expressions, rotations, translations, cameras, choices = [], [], [], [], []
    default_candidates = None
    for index, obs in enumerate(data.observations):
        kind, camera = observation_camera(obs, config)
        if camera is not None and config.focal_mode != "fixed":
            camera = replace(camera, focal_length=_initial_focal(ref, obs.image_size, config))
        item = initial.observations.get(obs.observation_id, base)
        label = f"{obs.observation_id} initial"
        expression = native_tensor(item.expression, (head.expression_dim,), ref, f"{label} expression")
        pose = native_tensor(item.rotations, (head.num_joints, 3), ref, f"{label} rotations")
        translation = native_tensor(item.translation, (3,), ref, f"{label} translation")
        if (expression.abs() > config.expression_limit).any():
            raise ValueError(f"{label} expression exceeds the configured operating range")

        grid = config.train_head and config.pose_initializer == "grid" and item.rotations is None
        strategy = "explicit" if item.rotations is not None else ("grid" if grid else "neutral")
        if item is base and default_candidates is not None:
            candidates = default_candidates
        else:
            candidates = _candidates(model, identity, expression, pose, translation, grid, config.batch_size)
            if item is base:
                default_candidates = candidates
        try:
            pose, camera, choice = _choose_pose(data, index, candidates, camera, kind, config, strategy)
        except ValueError as error:
            raise ValueError(f"Initialization for {obs.observation_id}: {error}") from error
        expressions.append(expression)
        rotations.append(pose)
        translations.append(translation)
        cameras.append(camera)
        choices.append(choice)
    state = TensorState(identity, torch.stack(expressions), torch.stack(rotations),
                        torch.stack(translations), tuple(cameras))
    return state, tuple(choices)
