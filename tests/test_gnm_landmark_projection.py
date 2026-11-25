"""End-to-end CPU and CUDA regressions using the official GNM v3.0 Head."""

from dataclasses import replace
from math import radians
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageColor
import pytest
from scipy.spatial.transform import Rotation
import torch

from persona_lab.camera import PerspectiveCamera, WeakPerspectiveCamera, project
from persona_lab.coordinates import ImageSize, gnm_to_frontal_axes
from persona_lab.correspondences import CorrespondenceMap, DFL_TO_GNM
from persona_lab.gnm_geometry import BARYCENTRIC_ATOL, SPARSE_HEAD, extract_landmarks, load_sparse_landmarks
from persona_lab.gnm_mesh import load_head_model
from persona_lab.landmark_diagnostics import (
    CONTOUR_COLOR, GNM_COLOR, OBSERVED_COLOR, RESIDUAL_COLOR, UNSUPPORTED_COLOR,
    save_landmark_overlay, write_synthetic_diagnostics,
)
from persona_lab.observations import LandmarkObservation


SIZE = ImageSize(512, 384)


@pytest.fixture(scope="module")
def head_model():
    return load_head_model(os.environ.get("PERSONALAB_GNM_MODEL_FILE")).double()


def frontal_camera(reference):
    return PerspectiveCamera(
        image_size=SIZE,
        rotation=reference.new_zeros(3),
        translation=reference.new_tensor((0, 0.265, 0.65)),
        focal_length=reference.new_tensor((820, 760)),
        principal_point=reference.new_tensor((255.5, 191.5)),
    )


def reference_landmarks(vertices, config):
    return np.einsum("...ijk,ij->...ik", vertices[..., config.indices, :], config.weights)


def reference_pinhole(gnm_points, camera):
    frontal = np.array([1, -1, -1]) * gnm_points
    camera_points = Rotation.from_rotvec(camera.rotation.numpy()).apply(frontal) + camera.translation.numpy()
    fx, fy = camera.focal_length.numpy()
    cx, cy = camera.principal_point.numpy()
    x, y, z = np.moveaxis(camera_points, -1, 0)
    return np.stack((fx * x / z + cx, fy * y / z + cy), axis=-1)


def test_official_resource_and_batched_barycentric_reference(head_model):
    config = load_sparse_landmarks(head_model)
    assert config.indices.shape == config.weights.shape == (68, 3)
    assert config.indices.dtype == np.int32 and config.weights.dtype == np.float32
    assert config.indices.min() >= 0 and config.indices.max() < 17821 == head_model.num_vertices
    assert np.isfinite(config.weights).all() and ((config.weights >= 0) & (config.weights <= 1)).all()
    np.testing.assert_allclose(config.weights.sum(axis=1), 1, atol=BARYCENTRIC_ATOL, rtol=0)

    template = head_model.template_vertex_positions
    identity = template.new_zeros((2, head_model.identity_dim))
    expression = template.new_zeros((2, head_model.expression_dim))
    rotations = template.new_zeros((2, head_model.num_joints, 3))
    identity[1, head_model.identity_names.index("head_000")] = 0.7
    expression[1, head_model.expression_names.index("lower_face_region_000")] = 1.5
    rotations[1, head_model.joint_names.index("head"), 1] = 0.35
    vertices, landmarks = extract_landmarks(head_model, identity=identity, expression=expression, rotations=rotations)
    assert vertices.shape == (2, 17821, 3) and landmarks.shape == (2, 68, 3)
    assert vertices.dtype == landmarks.dtype == torch.float64 and landmarks.device.type == "cpu"
    np.testing.assert_allclose(landmarks.numpy(), reference_landmarks(vertices.numpy(), config), atol=1e-12, rtol=0)
    assert not torch.allclose(landmarks[0], landmarks[1])
    for batch in range(2):
        _, separate = head_model.vertices_and_landmarks(SPARSE_HEAD, identity[batch], expression[batch], rotations[batch])
        torch.testing.assert_close(landmarks[batch], separate, atol=1e-12, rtol=0)


def test_native_landmarks_on_cpu_float32_and_invalid_gnm_batches():
    model = load_head_model(os.environ.get("PERSONALAB_GNM_MODEL_FILE"))
    template = model.template_vertex_positions
    vertices, points = extract_landmarks(model)
    assert vertices.shape == (17821, 3) and points.shape == (68, 3)
    assert points.dtype == torch.float32 and points.device.type == "cpu"
    torch.testing.assert_close(extract_landmarks(model)[1], points, atol=0, rtol=0)
    with pytest.raises(ValueError, match="matching batches"):
        extract_landmarks(
            model,
            expression=template.new_zeros((2, model.expression_dim)),
            rotations=template.new_zeros((3, model.num_joints, 3)),
        )
    with pytest.raises(ValueError, match="Batched GNM identity"):
        extract_landmarks(
            model,
            identity=template.new_zeros((1, model.identity_dim)),
            expression=template.new_zeros((2, model.expression_dim)),
        )
    with pytest.raises(ValueError, match="dtype and device"):
        extract_landmarks(model, identity=template.new_zeros(model.identity_dim).double())


def test_mapping_against_real_template_sides_vertex_groups_and_anatomy(head_model):
    config = load_sparse_landmarks(head_model)
    _, native = extract_landmarks(head_model)
    landmarks = native.numpy()
    vertex_groups = head_model.vertex_groups.numpy()
    names = head_model.vertex_group_names
    # Test the referenced surface, including GNM's documented left/right groups.
    for indices, group in [
        ([8], "chin_region"), ([17, 18, 19, 20], "right_brow_region"),
        ([23, 24, 25, 26], "left_brow_region"),
        (range(27, 36), "nose_region"), (range(36, 42), "right_orbital_region"),
        (range(42, 48), "left_orbital_region"),
        ([49, 50, 51, 52, 53, 61, 62, 63], "upper_lip"),
        ([55, 56, 57, 58, 59, 60, 64, 65, 66, 67], "lower_lip"),
    ]:
        selected = [DFL_TO_GNM.entries[index].gnm_index for index in indices]
        weights = config.weights[selected]
        group_membership = vertex_groups[names.index(group)][config.indices[selected]]
        assert (np.sum(weights * group_membership, axis=1) > 0.5).all(), group
    mirror_indices = head_model.mirror_indices.numpy()
    for right, left in [(0, 16), (4, 12), (17, 26), (21, 22), (31, 35), (36, 45), (39, 42), (48, 54), (60, 64), (61, 63), (67, 65)]:
        r = DFL_TO_GNM.entries[right].gnm_index
        l = DFL_TO_GNM.entries[left].gnm_index
        np.testing.assert_array_equal(np.sort(mirror_indices[config.indices[r]]), np.sort(config.indices[l]))
        assert landmarks[r, 0] < 0 < landmarks[l, 0]
    assert landmarks[30, 2] > max(landmarks[27:30, 2])
    assert landmarks[30, 1] > landmarks[33, 1] > landmarks[51, 1] > landmarks[57, 1] > landmarks[8, 1]
    assert landmarks[50, 1] > landmarks[62, 1] > landmarks[66, 1] > landmarks[57, 1]


def test_neutral_head_projection_against_independent_reference(head_model):
    _, landmarks = extract_landmarks(head_model)
    camera = frontal_camera(landmarks)
    result = project(gnm_to_frontal_axes(landmarks), camera).numpy()
    expected = reference_pinhole(landmarks.numpy(), camera)
    error = np.abs(result - expected)
    print(f"neutral GNM pixel error max={error.max():.3g}, mean={error.mean():.3g}")
    np.testing.assert_allclose(result, expected, atol=1e-10, rtol=0)
    assert result[36, 0] < 255.5 < result[45, 0] and result[30, 1] < result[8, 1]


@pytest.mark.parametrize(("axis", "angle", "landmark", "pixel_axis", "direction"), [
    (1, -25, 30, 0, -1), (1, 25, 30, 0, 1),
    (0, -19, 30, 1, -1), (0, 19, 30, 1, 1),
    (2, 16, 45, 1, -1),
], ids=["yaw-image-left", "yaw-image-right", "pitch-up", "pitch-down", "roll"])
def test_head_rotations_follow_native_axes(head_model, axis, angle, landmark, pixel_axis, direction):
    vertices, neutral = extract_landmarks(head_model)
    config = load_sparse_landmarks(head_model)
    head = head_model.joint_names.index("head")
    rotations = vertices.new_zeros((head_model.num_joints, 3))
    rotations[head, axis] = radians(angle)
    _, moved = extract_landmarks(head_model, rotations=rotations)

    selected = [30, 36, 45, 48, 54]
    triangles = config.indices[selected]
    assert (head_model.skinning_weights[head, triangles] == 1).all()
    pivot = head_model.template_joint_positions[head].numpy()
    selected_vertices = vertices.numpy()[triangles]
    rot = Rotation.from_rotvec(rotations[head].numpy())
    expected_vertices = rot.apply(selected_vertices.reshape(-1, 3) - pivot).reshape(-1, 3, 3) + pivot
    expected = np.einsum("ijk,ij->ik", expected_vertices, config.weights[selected])
    np.testing.assert_allclose(moved[selected].numpy(), expected, atol=3e-7, rtol=0)

    camera = frontal_camera(vertices)
    before = project(gnm_to_frontal_axes(neutral), camera)
    after = project(gnm_to_frontal_axes(moved), camera)
    np.testing.assert_allclose(after[selected].numpy(), reference_pinhole(expected, camera), atol=7e-4, rtol=0)
    assert direction * (after[landmark, pixel_axis] - before[landmark, pixel_axis]) > 1


def test_head_identity_and_lower_face_expression_match_model_bases(head_model):
    _, neutral = extract_landmarks(head_model)
    config = load_sparse_landmarks(head_model)
    template = head_model.template_vertex_positions
    i = head_model.identity_names.index("head_000")
    e = head_model.expression_names.index("lower_face_region_000")
    identity = template.new_zeros(head_model.identity_dim)
    identity[i] = 0.8
    expression = template.new_zeros(head_model.expression_dim)
    expression[e] = 2.0
    _, changed_identity = extract_landmarks(head_model, identity=identity)
    _, changed_expression = extract_landmarks(head_model, expression=expression)
    identity_reference = reference_landmarks(head_model.vertex_identity_basis[i].numpy() * 0.8, config)
    expression_reference = reference_landmarks(head_model.expression_basis[e].numpy() * 2.0, config)
    np.testing.assert_allclose((changed_identity - neutral).numpy(), identity_reference, atol=3e-7, rtol=0)
    np.testing.assert_allclose((changed_expression - neutral).numpy(), expression_reference, atol=3e-7, rtol=0)
    assert np.linalg.norm(identity_reference).max() > 1e-4
    assert np.linalg.norm(expression_reference[48:68], axis=1).max() > 1e-4
    camera = frontal_camera(neutral)
    neutral_pixels = project(gnm_to_frontal_axes(neutral), camera)
    expression_pixels = project(gnm_to_frontal_axes(changed_expression), camera)
    expected = reference_pinhole(neutral.numpy() + expression_reference, camera)
    np.testing.assert_allclose(expression_pixels.numpy(), expected, atol=7e-4, rtol=0)
    assert torch.linalg.vector_norm((neutral_pixels - expression_pixels)[48:68], dim=-1).max() > 0.1
    torch.testing.assert_close(expression_pixels[36:48], neutral_pixels[36:48], atol=7e-4, rtol=0)


@pytest.mark.parametrize("perspective", [True, False], ids=["perspective", "weak-perspective"])
def test_end_to_end_gradients_and_shared_identity_finite_difference(head_model, perspective):
    template = head_model.template_vertex_positions
    lower_face = head_model.expression_names.index("lower_face_region_000")
    identity = template.new_zeros(head_model.identity_dim)
    identity[head_model.identity_names.index("head_000")] = 0.4
    expression = template.new_zeros((2, head_model.expression_dim))
    expression[:, lower_face] = template.new_tensor((0.7, -0.35))
    pose = template.new_zeros((2, head_model.num_joints, 3))
    pose[:, head_model.joint_names.index("head"), 1] = template.new_tensor((0.15, -0.19))
    rotation = template.new_tensor((0.09, -0.18, 0.07))
    if perspective:
        tail = (template.new_tensor((0.017, 0.267, 0.7)), template.new_tensor((825, 762)), template.new_tensor((252, 192)))
    else:
        tail = (template.new_tensor((250, 543)), template.new_tensor(1330.0))
    variables = (identity, expression, pose, rotation, *tail)
    for parameter in variables:
        parameter.requires_grad_()

    def predict(i, e, p, r, *camera_parameters):
        _, points = extract_landmarks(head_model, identity=i, expression=e, rotations=p)
        camera_class = PerspectiveCamera if perspective else WeakPerspectiveCamera
        camera = camera_class(SIZE, r, *camera_parameters)
        return project(gnm_to_frontal_axes(points), camera)

    target = predict(*variables).detach() + template.new_tensor((2.5, -1.75))
    assert all(parameter.requires_grad for parameter in variables), (
        "GNM must not mutate the caller's requires_grad flags"
    )
    observations = [
        LandmarkObservation(observation_id=f"view-{index}", image_size=SIZE, landmarks=values)
        for index, values in enumerate(target)
    ]

    def view_losses(pixels):
        losses = []
        for observation, projection in zip(observations, pixels, strict=True):
            pairs = observation.landmark_pairs(projection)
            residual = pairs.projected - pairs.observed
            losses.append((residual.square().sum(-1) * pairs.weights).mean())
        return losses

    losses = view_losses(predict(*variables))
    derivatives = torch.autograd.grad(sum(losses), variables, retain_graph=True)
    individual = [torch.autograd.grad(loss, identity, retain_graph=True)[0] for loss in losses]
    assert all(torch.linalg.vector_norm(gradient) > 1e-4 for gradient in individual)
    torch.testing.assert_close(derivatives[0], sum(individual), atol=1e-10, rtol=1e-10)
    for view in range(2):
        batched = predict(*variables)[view]
        individual_prediction = predict(identity, expression[view], pose[view], rotation, *tail)
        torch.testing.assert_close(batched, individual_prediction, atol=1e-10, rtol=0)

    abs_errors = []
    for index, (parameter, analytical) in enumerate(zip(variables, derivatives, strict=True)):
        assert analytical.shape == parameter.shape and torch.isfinite(analytical).all()
        coordinate = analytical.abs().argmax().item()
        automatic = analytical.reshape(-1)[coordinate]
        assert automatic.abs() > 1e-5
        numeric_predictions = []
        step = 1e-5
        for sign in (+1, -1):
            moved = parameter.detach().clone()
            moved.reshape(-1)[coordinate] += sign * step
            displaced = [value.detach() for value in variables]
            displaced[index] = moved
            numeric_predictions.append(predict(*displaced))
        plus, minus = numeric_predictions
        assert torch.max(torch.abs(plus - minus)) > 1e-8
        finite_difference = (sum(view_losses(plus)) - sum(view_losses(minus))) / (2 * step)
        abs_errors.append(abs(float(automatic - finite_difference)))
        torch.testing.assert_close(automatic, finite_difference, atol=2e-5, rtol=2e-4)
    print(f"finite-difference abs error max={max(abs_errors):.3g}, mean={np.mean(abs_errors):.3g}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Compatible CUDA device required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_native_cache_and_projection_after_model_moves(dtype):
    model = load_head_model(os.environ.get("PERSONALAB_GNM_MODEL_FILE")).to(dtype=dtype)
    _, cpu = extract_landmarks(model)
    cpu_again = model.vertices_and_landmarks(SPARSE_HEAD)[1]
    torch.testing.assert_close(cpu, cpu_again, rtol=0, atol=0)
    assert all(value.device.type == "cpu" for value in model._landmarks[SPARSE_HEAD])

    model.cuda()
    _, cuda_points = extract_landmarks(model)
    assert cuda_points.device.type == "cuda" and cuda_points.dtype == dtype
    assert all(value.device.type == "cuda" for value in model._landmarks[SPARSE_HEAD])
    template = model.template_vertex_positions
    identity = template.new_zeros(model.identity_dim).requires_grad_()
    _, differentiable = extract_landmarks(model, identity=identity)
    camera = frontal_camera(differentiable)
    projected = project(gnm_to_frontal_axes(differentiable), camera)
    assert projected.device.type == "cuda" and projected.dtype == dtype
    expected = project(gnm_to_frontal_axes(cpu), frontal_camera(cpu))
    torch.testing.assert_close(projected.detach().cpu(), expected, atol=4e-4, rtol=0)
    derivative = torch.autograd.grad(projected[30].sum(), identity)[0]
    assert torch.isfinite(derivative).all() and derivative.abs().max() > 1e-5
    with pytest.raises(ValueError, match="dtype and device"):
        project(gnm_to_frontal_axes(cpu), camera)
    with pytest.raises(ValueError, match="dtype and device"):
        extract_landmarks(model, identity=identity.cpu())

    del model._landmarks[SPARSE_HEAD]
    cold_cuda = model.vertices_and_landmarks(SPARSE_HEAD)[1]
    assert all(value.device.type == "cuda" for value in model._landmarks[SPARSE_HEAD])
    torch.testing.assert_close(cold_cuda, cuda_points, atol=0, rtol=0)

    model.cpu()
    _, returned = extract_landmarks(model)
    assert all(value.device.type == "cpu" for value in model._landmarks[SPARSE_HEAD])
    torch.testing.assert_close(returned, cpu, rtol=0, atol=0)


def test_synthetic_diagnostic_views_on_real_gnm_head(head_model, tmp_path):
    paths = write_synthetic_diagnostics(head_model, tmp_path)
    assert [path.name for path in paths] == [
        "frontal-neutral.png", "three-quarter-image-left.png", "three-quarter-image-right.png",
        "profile.png", "lower-face-expression.png",
    ]
    rasters = []
    for path in paths:
        with Image.open(path) as result:
            assert result.format == "PNG" and result.size == (1174, 710)
            rasters.append(np.array(result))
    for view in rasters[1:]:
        assert (rasters[0][56:616] != view[56:616]).any()


def test_geometry_and_camera_import_without_the_diagnostic_module():
    subprocess.run(
        [sys.executable, "-c",
         "import sys; import persona_lab.gnm_geometry, persona_lab.camera; "
         "assert 'persona_lab.landmark_diagnostics' not in sys.modules"],
        check=True,
    )


def test_overlay_includes_indices_correspondences_and_mask_colors(tmp_path):
    size = ImageSize(192, 152)
    projected = torch.tensor([(24 + i % 8 * 18, 24 + i // 8 * 12) for i in range(68)], dtype=torch.float64)
    observed = projected + projected.new_tensor([10, 7])
    entries = list(DFL_TO_GNM.entries)
    entries[19] = replace(entries[19], gnm_index=None, enabled_by_default=False, limitation="Synthetic unknown")
    observation = LandmarkObservation(
        observation_id="synthetic", image_size=size, landmarks=observed,
        correspondences=CorrespondenceMap(tuple(entries)),
    )
    # Excluded contour coloring is only requested by an explicit mask.
    assert observation.fitting_mask()[:17].all()
    pose = torch.ones(68, dtype=torch.bool)
    pose[0] = False
    path = save_landmark_overlay(
        tmp_path / "diagnostic.png", projected, observation, pose_validity=pose,
    )
    with Image.open(path) as result:
        pixels = np.array(result)
    # Check plot regions, excluding the legend and headings.
    left = pixels[56:56 + size.height, 18:18 + size.width]
    right = pixels[56:56 + size.height, 296:296 + size.width]
    assert np.any(np.all(left == ImageColor.getrgb(GNM_COLOR), axis=-1))
    for color in [OBSERVED_COLOR, RESIDUAL_COLOR, CONTOUR_COLOR, UNSUPPORTED_COLOR]:
        assert np.any(np.all(right == ImageColor.getrgb(color), axis=-1)), color
    # Check the actual observed centers, not just the presence of a color.
    for index, color in [(0, CONTOUR_COLOR), (19, UNSUPPORTED_COLOR)]:
        x, y = observed[index].to(torch.int64).tolist()
        np.testing.assert_array_equal(right[y, x], ImageColor.getrgb(color))
    # Index text extends beyond its landmark marker.
    x, y = projected[0].to(torch.int64).tolist()
    index_text = left[y - 12:y - 2, x + 3:x + 12]
    assert np.any(index_text != ImageColor.getrgb("#f4f8fc"))
