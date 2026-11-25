"""Tensor coordinate conversions for GNM and DeepFaceLab aligned images."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


_FLOAT_DTYPES = (torch.float32, torch.float64)


def _validate_tensor(
    value: Tensor,
    name: str,
    suffix: tuple[int | None, ...],
    *,
    like: Tensor | None = None,
) -> None:
    if not isinstance(value, Tensor) or value.dtype not in _FLOAT_DTYPES:
        raise ValueError(f"{name} must be a float32 or float64 tensor")
    if value.device.type not in {"cpu", "cuda"} or value.layout != torch.strided:
        raise ValueError(f"{name} must be a dense CPU or CUDA tensor")
    if like is not None and (value.dtype, value.device) != (like.dtype, like.device):
        raise ValueError(f"{name} must share the reference tensor's dtype and device")
    if value.ndim < len(suffix) or 0 in value.shape or any(
        size is not None and value.shape[-offset] != size
        for offset, size in enumerate(reversed(suffix), start=1)
    ):
        raise ValueError(f"{name} must have a nonempty shape with trailing dimensions {suffix}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _broadcast(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    try:
        return tuple(torch.broadcast_shapes(*shapes))
    except RuntimeError as error:
        raise ValueError(f"Incompatible batch dimensions: {shapes}") from error


@dataclass(frozen=True, slots=True)
class ImageSize:
    """Raster dimensions in width, height order."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in (self.width, self.height)):
            raise ValueError("Image width and height must be positive integers")


def _image_scale(points: Tensor, image_size: ImageSize) -> Tensor:
    if not isinstance(image_size, ImageSize):
        raise ValueError("image_size must be an ImageSize")
    scale = points.new_tensor((image_size.width, image_size.height))
    if not torch.isfinite(scale).all():
        raise ValueError("Image dimensions exceed the range of the point dtype")
    return scale


def pixels_to_normalized(points: Tensor, image_size: ImageSize) -> Tensor:
    """Convert (..., N, 2) pixel centers to edge-normalized (x, y) coordinates.

    The raster edges are 0 and 1. The upper-left pixel center is
    (0.5 / width, 0.5 / height) after conversion. Out-of-frame points are valid.
    """
    _validate_tensor(points, "Pixel coordinates", (None, 2))
    return (points + 0.5) / _image_scale(points, image_size)


def normalized_to_pixels(points: Tensor, image_size: ImageSize) -> Tensor:
    """Invert edge normalization, keeping the upper-left pixel center at (0, 0)."""
    _validate_tensor(points, "Normalized coordinates", (None, 2))
    pixels = points * _image_scale(points, image_size) - 0.5
    _validate_tensor(pixels, "Converted pixel coordinates", (None, 2))
    return pixels


def flip_camera_axes(points: Tensor) -> Tensor:
    """Convert (..., N, 3) coordinates between OpenGL and OpenCV camera axes."""
    _validate_tensor(points, "3D points", (None, 3))
    return points * points.new_tensor((1, -1, -1))


def gnm_to_frontal_axes(points: Tensor) -> Tensor:
    """Re-express posed GNM points in frontal OpenCV-aligned axes at the GNM origin."""
    return flip_camera_axes(points)


def convert_camera_matrix(
    matrix: Tensor,
    *,
    direction: Literal["world_to_camera", "camera_to_world"],
) -> Tensor:
    """Change only the camera convention of a homogeneous 4×4 transformation.

    This conversion is its own inverse. Left-multiply by diag(1,-1,-1,1)
    for world-to-camera and right-multiply for camera-to-world.
    """
    _validate_tensor(matrix, "Camera transformation", (4, 4))
    signs = matrix.new_tensor((1, -1, -1, 1))
    if direction == "world_to_camera":
        return signs[:, None] * matrix
    if direction == "camera_to_world":
        return matrix * signs
    raise ValueError("direction must be 'world_to_camera' or 'camera_to_world'")


def _alignment_parts(points: Tensor, matrix: Tensor | None) -> tuple[Tensor, Tensor]:
    _validate_tensor(points, "Image points", (None, 2))
    if matrix is None:
        raise ValueError("DeepFaceLab image_to_face_mat is unavailable")
    _validate_tensor(matrix, "DeepFaceLab image_to_face_mat", (2, 3), like=points)
    _broadcast(points.shape[:-2], matrix.shape[:-2])
    linear, offset = matrix[..., :2], matrix[..., 2]
    if (torch.linalg.slogdet(linear).sign == 0).any():
        raise ValueError("DeepFaceLab image_to_face_mat is singular")
    return linear, offset[..., None, :]


def source_to_aligned(points: Tensor, image_to_face_mat: Tensor | None) -> Tensor:
    """Transform source pixel centers via the stored source-to-aligned affine."""
    linear, offset = _alignment_parts(points, image_to_face_mat)
    result = points @ linear.mT + offset
    _validate_tensor(result, "Converted aligned coordinates", (None, 2))
    return result


def aligned_to_source(points: Tensor, image_to_face_mat: Tensor | None) -> Tensor:
    """Invert the stored affine to map aligned pixel centers into source pixels."""
    linear, offset = _alignment_parts(points, image_to_face_mat)
    result = torch.linalg.solve(linear, (points - offset).mT).mT
    _validate_tensor(result, "Converted source coordinates", (None, 2))
    return result
