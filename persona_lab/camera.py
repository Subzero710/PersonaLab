"""Differentiable projection into a raster with OpenCV camera axes."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .coordinates import ImageSize, _broadcast, _validate_tensor


@dataclass(frozen=True, slots=True, eq=False)
class PerspectiveCamera:
    """Pinhole camera with zero skew and no distortion.

    All parameters are tensors with broadcastable leading batch dimensions.
    Focal length (fx, fy) and principal point (cx, cy) are in raster pixels.
    Rotation is a world-to-camera axis-angle vector in radians. Translation
    is added in camera coordinates and has the same units as the 3D points.
    Raster size is shared by the camera batch.
    """

    image_size: ImageSize
    rotation: Tensor       # (..., 3)
    translation: Tensor    # (..., 3)
    focal_length: Tensor   # (..., 2)
    principal_point: Tensor  # (..., 2)


@dataclass(frozen=True, slots=True, eq=False)
class WeakPerspectiveCamera:
    """Rotated orthographic projection with one common scale in pixels per unit.

    Rotation uses the same axis-angle convention as the perspective camera.
    Translation (tx, ty) is in raster pixels. There is no 3D translation or
    separate principal point. Raster size is shared by the camera batch.
    """

    image_size: ImageSize
    rotation: Tensor       # (..., 3)
    translation: Tensor    # (..., 2)
    scale: Tensor          # (...) or a scalar tensor


Camera = PerspectiveCamera | WeakPerspectiveCamera


def _camera_batch(camera: Camera, like: Tensor | None = None) -> tuple[int, ...]:
    if not isinstance(camera, (PerspectiveCamera, WeakPerspectiveCamera)):
        raise ValueError("Expected a perspective or weak-perspective camera")
    if not isinstance(camera.image_size, ImageSize):
        raise ValueError("Camera image_size must be an ImageSize")
    _validate_tensor(camera.rotation, "Camera rotation", (3,), like=like)
    reference = camera.rotation
    if isinstance(camera, PerspectiveCamera):
        _validate_tensor(camera.translation, "Camera translation", (3,), like=reference)
        _validate_tensor(camera.focal_length, "Focal length", (2,), like=reference)
        _validate_tensor(camera.principal_point, "Principal point", (2,), like=reference)
        if (camera.focal_length <= 0).any():
            raise ValueError("Focal lengths must be positive")
        lens_batches = (camera.focal_length.shape[:-1], camera.principal_point.shape[:-1])
    else:
        _validate_tensor(camera.translation, "Image translation", (2,), like=reference)
        _validate_tensor(camera.scale, "Weak-perspective scale", (), like=reference)
        if (camera.scale <= 0).any():
            raise ValueError("Weak-perspective scale must be positive")
        lens_batches = (camera.scale.shape,)
    return _broadcast(camera.rotation.shape[:-1], camera.translation.shape[:-1], *lens_batches)


def _rotation_matrix(rotation: Tensor) -> Tensor:
    x, y, z = rotation.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    return torch.linalg.matrix_exp(skew.unflatten(-1, (3, 3)))


def world_to_camera(points: Tensor, rotation: Tensor, translation: Tensor) -> Tensor:
    """Apply X_camera = R @ X_world + t to (..., N, 3) row-stored points."""
    _validate_tensor(points, "3D points", (None, 3))
    _validate_tensor(rotation, "Camera rotation", (3,), like=points)
    _validate_tensor(translation, "Camera translation", (3,), like=points)
    _broadcast(points.shape[:-2], rotation.shape[:-1], translation.shape[:-1])
    result = points @ _rotation_matrix(rotation).mT + translation[..., None, :]
    _validate_tensor(result, "Camera-space points", (None, 3))
    return result


def camera_position(rotation: Tensor, translation: Tensor) -> Tensor:
    """Return the camera origin in world coordinates, -R.T @ t."""
    _validate_tensor(rotation, "Camera rotation", (3,))
    _validate_tensor(translation, "Camera translation", (3,), like=rotation)
    _broadcast(rotation.shape[:-1], translation.shape[:-1])
    position = -(_rotation_matrix(rotation).mT @ translation[..., None]).squeeze(-1)
    _validate_tensor(position, "World-space camera position", (3,))
    return position


def project(points: Tensor, camera: Camera) -> Tensor:
    """Project (..., N, 3) to (..., N, 2) in integer-centered raster pixels.

    Leading dimensions of points and all camera tensors broadcast together.
    Points are expressed in the world axes used by the supplied camera.
    Perspective projection raises ValueError if any depth is nonpositive
    or nonfinite. Weak perspective uses the first two rotated coordinates.
    Off-frame pixel coordinates remain valid.
    """
    _validate_tensor(points, "3D points", (None, 3))
    _broadcast(points.shape[:-2], _camera_batch(camera, points))
    rotated = points @ _rotation_matrix(camera.rotation).mT

    if isinstance(camera, PerspectiveCamera):
        camera_points = rotated + camera.translation[..., None, :]
        _validate_tensor(camera_points, "Camera-space points", (None, 3))
        depth = camera_points[..., 2:]
        if (depth <= 0).any():
            raise ValueError("Perspective projection requires positive camera depth")
        normalized = camera_points[..., :2] / depth
        pixels = (
            normalized * camera.focal_length[..., None, :]
            + camera.principal_point[..., None, :]
        )
    else:
        _validate_tensor(rotated, "Rotated points", (None, 3))
        pixels = rotated[..., :2] * camera.scale[..., None, None] + camera.translation[..., None, :]

    _validate_tensor(pixels, "Projected pixel coordinates", (None, 2))
    return pixels
