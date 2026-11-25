"""Independent numerical checks of the PersonaLab camera and image conventions."""

from dataclasses import replace
from math import pi

from gnm.shape.visualization import camera_conversions
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from persona_lab.camera import PerspectiveCamera, WeakPerspectiveCamera, camera_position, project, world_to_camera
from persona_lab.coordinates import (
    ImageSize,
    aligned_to_source,
    convert_camera_matrix,
    flip_camera_axes,
    gnm_to_frontal_axes,
    normalized_to_pixels,
    pixels_to_normalized,
    source_to_aligned,
)


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def perspective():
    return PerspectiveCamera(
        image_size=ImageSize(100, 80),
        rotation=tensor((0, 0, pi / 2)),
        translation=tensor((1, -1, 2)),
        focal_length=tensor((120, 80)),
        principal_point=tensor((49.5, 39.5)),
    )


def weak():
    return WeakPerspectiveCamera(
        image_size=ImageSize(100, 80),
        rotation=tensor((0, 0, pi / 2)),
        translation=tensor((10, 20)),
        scale=tensor(2),
    )


def test_hand_calculated_perspective_and_camera_position():
    camera = perspective()
    points = tensor(((1, 2, 4), (-2, 1, 3)))
    camera_points = world_to_camera(points, camera.rotation, camera.translation)
    torch.testing.assert_close(camera_points, tensor(((-1, 0, 6), (0, -3, 5))), atol=1e-12, rtol=0)
    torch.testing.assert_close(camera_position(camera.rotation, camera.translation), tensor((1, 1, -2)), atol=1e-12, rtol=0)
    torch.testing.assert_close(project(points, camera), tensor(((29.5, 39.5), (49.5, -8.5))), atol=1e-12, rtol=0)


def test_hand_calculated_weak_perspective():
    points = tensor(((1, 2, 4), (-2, 1, -3)))
    torch.testing.assert_close(project(points, weak()), tensor(((6, 22), (8, 16))), atol=1e-12, rtol=0)


@pytest.mark.parametrize("model", [perspective, weak])
def test_batched_projects_against_scipy_and_numpy(model):
    points = tensor([
        [[(0.15, -0.1, 2.2), (0.45, 0.35, 2.9)]],
        [[(-0.1, 0.6, 3.1), (0.2, -0.25, 1.9)]],
    ])
    rotation = tensor([[0.27, 0.105, -0.095], [-0.19, 0.18, 0.31], [0.02, -0.34, 0.14]])
    camera = replace(model(), rotation=rotation)
    expected = np.empty((2, 3, 2, 2))
    for i, points_for_view in enumerate(points[:, 0].numpy()):
        for j, scipy_rotation in enumerate(Rotation.from_rotvec(rotation.numpy())):
            rotated = scipy_rotation.apply(points_for_view)
            if isinstance(camera, PerspectiveCamera):
                moved = rotated + camera.translation.numpy()
                expected[i, j] = moved[:, :2] / moved[:, 2:] * camera.focal_length.numpy() + camera.principal_point.numpy()
            else:
                expected[i, j] = rotated[:, :2] * float(camera.scale) + camera.translation.numpy()
    actual = project(points, camera)
    error = np.abs(actual.numpy() - expected)
    print(f"{model.__name__} pixel error max={error.max():.3g}, mean={error.mean():.3g}")
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-11, rtol=0)
    for i in range(2):
        for j in range(3):
            torch.testing.assert_close(actual[i, j], project(points[i, 0], replace(camera, rotation=rotation[j])))


def test_batched_intrinsics_translation_and_weak_scale():
    points = tensor([[1, 2, 4]])
    camera = PerspectiveCamera(
        ImageSize(200, 100), tensor([0, 0, 0]), tensor([[0, 0, 1], [1, 1, 0]]),
        tensor([[100, 50], [80, 40]]), tensor([10, 12]),
    )
    np.testing.assert_allclose(project(points, camera).numpy(), [[[30, 32]], [[50, 42]]], atol=1e-12, rtol=0)
    camera = WeakPerspectiveCamera(ImageSize(200, 100), tensor([0, 0, 0]), tensor([3, 4]), tensor([2, 5]))
    np.testing.assert_allclose(project(points, camera).numpy(), [[[5, 8]], [[8, 14]]], atol=1e-12, rtol=0)


@pytest.mark.parametrize("model", [perspective, weak])
def test_resolution_rescales_only_with_matching_camera_parameters(model):
    camera = model()
    points = tensor([[1, 2, 4]])
    initial = project(points, camera)
    larger = replace(camera, image_size=ImageSize(200, 160))
    torch.testing.assert_close(project(points, larger), initial)
    if isinstance(camera, PerspectiveCamera):
        larger = replace(larger, focal_length=camera.focal_length * 2, principal_point=(camera.principal_point + 0.5) * 2 - 0.5)
    else:
        larger = replace(larger, scale=camera.scale * 2, translation=(camera.translation + 0.5) * 2 - 0.5)
    torch.testing.assert_close(project(points, larger), (initial + 0.5) * 2 - 0.5)
    torch.testing.assert_close(
        pixels_to_normalized(project(points, larger), larger.image_size),
        pixels_to_normalized(initial, camera.image_size),
    )


@pytest.mark.parametrize("width,height", [(0, 2), (-2, 2), (3, 0), (True, 2), (2, 3.5)])
def test_invalid_image_size(width, height):
    with pytest.raises(ValueError, match="positive integers"):
        ImageSize(width, height)


@pytest.mark.parametrize("depth", [0, -1])
def test_perspective_rejects_nonpositive_depth(depth):
    points = tensor([[1, 2, depth], [1, 2, 3]])
    with pytest.raises(ValueError, match="positive camera depth"):
        project(points, replace(perspective(), translation=tensor([0, 0, 0])))


@pytest.mark.parametrize(("changes", "message"), [
    ({"rotation": tensor([[0, 0], [0, 0]])}, "Camera rotation"),
    ({"translation": tensor([float("inf"), 0, 0])}, "finite"),
    ({"focal_length": tensor([-1, 2])}, "positive"),
    ({"principal_point": tensor([1, 2]).float()}, "dtype and device"),
    ({"image_size": (100, 80)}, "ImageSize"),
])
def test_invalid_camera_parameters(changes, message):
    with pytest.raises(ValueError, match=message):
        project(tensor([[1, 2, 3]]), replace(perspective(), **changes))


@pytest.mark.parametrize("scale", [0, -1, float("nan")])
def test_invalid_weak_perspective_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        project(tensor([[1, 2, 3]]), replace(weak(), scale=tensor(scale)))


@pytest.mark.parametrize("points", [
    torch.zeros(3), torch.zeros(2, 2), torch.zeros(2, 3, dtype=torch.int64),
    torch.zeros(2, 3, dtype=torch.float16), torch.zeros(0, 3),
    torch.zeros(2, 3, device="meta"), tensor([[0, float("nan"), 1]]),
])
def test_invalid_points(points):
    with pytest.raises(ValueError, match="3D points"):
        project(points, perspective())


def test_incompatible_batch_sizes():
    with pytest.raises(ValueError, match="batch dimensions"):
        project(tensor([[[1, 2, 3]]] * 2), replace(perspective(), focal_length=tensor([[10, 10]] * 3)))


def test_camera_coordinate_changes_and_round_trips():
    point = tensor([[1, 2, 3]])
    expected = tensor([[1, -2, -3]])
    assert torch.equal(gnm_to_frontal_axes(point), expected)
    assert torch.equal(flip_camera_axes(expected), point)
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = Rotation.from_rotvec([0.2, 0.35, -0.3]).as_matrix()
    extrinsics[:3, 3] = [0.15, -0.25, 1.2]
    values = tensor(np.stack([extrinsics, np.eye(4)]))
    converted = convert_camera_matrix(values, direction="world_to_camera")
    np.testing.assert_array_equal(converted.numpy(), camera_conversions.opengl_extrinsics_to_opencv(values.numpy()))
    inverse = torch.linalg.inv(values)
    converted_inverse = convert_camera_matrix(inverse, direction="camera_to_world")
    torch.testing.assert_close(converted_inverse, torch.linalg.inv(converted), atol=1e-12, rtol=0)
    for direction, first, second in (("world_to_camera", values, converted), ("camera_to_world", inverse, converted_inverse)):
        assert torch.equal(convert_camera_matrix(second, direction=direction), first)
    with pytest.raises(ValueError, match="direction"):
        convert_camera_matrix(values, direction="unspecified")


@pytest.mark.parametrize("size", [ImageSize(320, 240), ImageSize(1, 1), ImageSize(953, 501)])
def test_pixel_centers_and_outside_image(size):
    pixels = tensor([[-0.5, -0.5], [0, 0], [size.width - 0.5, size.height - 0.5], [-2, 34.5]])
    reference = tensor([[0, 0], [0.5 / size.width, 0.5 / size.height], [1, 1], [-1.5 / size.width, 35 / size.height]])
    normalized = pixels_to_normalized(pixels, size)
    torch.testing.assert_close(normalized, reference, atol=1e-15, rtol=0)
    torch.testing.assert_close(normalized_to_pixels(normalized, size), pixels, atol=1e-12, rtol=0)


@pytest.mark.parametrize(("matrix", "expected"), [
    ([[1, 0, 0], [0, 1, 0]], [[1, 2], [-1, 4]]),
    ([[1, 0, 3], [0, 1, -5]], [[4, -3], [2, -1]]),
    ([[2, 0, 0], [0, 3, 0]], [[2, 6], [-2, 12]]),
    ([[0, -1, 0], [1, 0, 0]], [[-2, 1], [-4, -1]]),
    ([[0, -2, 3], [4, 0, 5]], [[-1, 9], [-5, 1]]),
])
def test_alignment_reference_and_inverse(matrix, expected):
    points = tensor([[1, 2], [-1, 4]])
    matrix = tensor(matrix)
    transformed = source_to_aligned(points, matrix)
    torch.testing.assert_close(transformed, tensor(expected), atol=1e-12, rtol=0)
    torch.testing.assert_close(aligned_to_source(transformed, matrix), points, atol=1e-12, rtol=0)
    batched_matrix = torch.stack((matrix, tensor([[1, 0, 0], [0, 1, 0]])))
    torch.testing.assert_close(source_to_aligned(points, batched_matrix), torch.stack((transformed, points)))
    torch.testing.assert_close(aligned_to_source(source_to_aligned(points, batched_matrix), batched_matrix), points.expand(2, 2, 2))


@pytest.mark.parametrize("transform", [source_to_aligned, aligned_to_source])
@pytest.mark.parametrize(("matrix", "message"), [
    (None, "unavailable"), (tensor([[1, 2, 0], [2, 4, 0]]), "singular"),
    (tensor([[1, 0], [0, 1]]), "shape"), (tensor([[1, 0, 0], [0, 1, float("nan")]]), "finite"),
])
def test_bad_alignment_metadata(transform, matrix, message):
    with pytest.raises(ValueError, match=message):
        transform(tensor([[1, 2]]), matrix)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_projection_preserves_cpu_dtype_and_gradients_at_zero_rotation(dtype):
    points = torch.tensor([[1, 2, 3]], dtype=dtype)
    rotation = torch.zeros(3, dtype=dtype, requires_grad=True)
    camera = WeakPerspectiveCamera(ImageSize(19, 13), rotation, points.new_zeros(2), points.new_tensor(2))
    output = project(points, camera)
    assert output.dtype == dtype and output.device == points.device
    gradient = torch.autograd.grad(output.sum(), rotation)[0]
    torch.testing.assert_close(gradient, points.new_tensor([-6, 6, -2]), atol=1e-6, rtol=0)
