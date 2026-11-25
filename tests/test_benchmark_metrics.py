"""Numerical references for benchmark reductions and camera conventions."""

import numpy as np
import pytest
import torch

from persona_lab.benchmark.metrics import (
    coefficient_errors, reprojection_errors, rotation_errors_deg, vertex_errors,
)
from persona_lab.benchmark.model import in_frame, project_dfl
from persona_lab.camera import PerspectiveCamera, WeakPerspectiveCamera
from persona_lab.coordinates import ImageSize
from persona_lab.correspondences import DFL_TO_GNM


def test_coefficient_and_geometry_errors_have_independent_expected_values():
    identity = coefficient_errors(np.array([3., -4.]), np.zeros(2))
    assert identity == {
        "mae": 3.5, "rmse": pytest.approx(np.sqrt(12.5)), "max_absolute": 4.0,
        "coefficient_count": 2,
    }
    vertices = np.array([[3., 4., 0.], [0., 0., 12.]])
    geometry = vertex_errors(vertices, np.zeros((2, 3)))
    assert geometry == {
        "mean_m": 8.5, "rms_m": pytest.approx(np.sqrt(84.5)), "max_m": 12.0,
        "vertex_count": 2,
    }


@pytest.mark.parametrize("estimated,truth", [
    ([1., 2.], [1.]), ([], []), ([float("nan")], [0.]), ([1], [0]),
])
def test_reject_invalid_coefficient_inputs(estimated, truth):
    with pytest.raises(ValueError):
        coefficient_errors(np.array(estimated), np.array(truth))


def test_reject_coefficient_reordering():
    with pytest.raises(ValueError, match="ordering"):
        coefficient_errors(
            np.zeros(2), np.zeros(2), estimated_names=("b", "a"), ground_truth_names=("a", "b"),
        )


def test_2d_metrics_keep_slots_null_and_allow_empty_valid_selections():
    estimate = np.zeros((68, 2))
    estimate[0], estimate[45], estimate[67] = (3, 4), (12, 0), (99, 99)
    valid = np.zeros(68, dtype=bool)
    valid[[0, 45]] = True
    result = reprojection_errors(estimate, np.zeros_like(estimate), valid)
    assert result["rms_px"] == pytest.approx(np.sqrt(84.5))
    assert result["mean_px"] == result["median_px"] == 8.5
    assert result["per_landmark_px"][0] == 5
    assert result["per_landmark_px"][45] == 12
    assert result["per_landmark_px"][67] is None
    assert (result["valid_count"], result["excluded_count"]) == (2, 66)
    empty = reprojection_errors(estimate, estimate, np.zeros_like(valid))
    assert [empty[key] for key in ("mean_px", "median_px", "rms_px")] == [None] * 3
    assert empty["per_landmark_px"] == [None] * 68


def test_pose_metric_respects_equivalent_axis_angle_encodings():
    estimated = np.array([[0., 0., 2 * np.pi], [np.pi / 2, 0., 0.]])
    result = rotation_errors_deg(estimated, np.zeros((2, 3)))
    np.testing.assert_allclose(result, [0, 90], atol=1e-12, rtol=0)
    reversed_pi = rotation_errors_deg([0., 0., -np.pi], [0., 0., np.pi])
    assert reversed_pi == pytest.approx(0, abs=1e-12)


def test_project_dfl_known_points_and_raster_edges():
    native = torch.zeros((68, 3), dtype=torch.float64)
    native[0] = native.new_tensor([2, 3, 4])
    size = ImageSize(100, 60)
    rotation = native.new_zeros(3)
    perspective = PerspectiveCamera(
        size, rotation, native.new_tensor([0, 1, 6]),
        native.new_tensor([20, 10]), native.new_tensor([49.5, 29.5]),
    )
    weak = WeakPerspectiveCamera(size, rotation, native.new_tensor([49.5, 29.5]), native.new_tensor(5))
    perspective_pixels = project_dfl(native, perspective, DFL_TO_GNM)
    weak_pixels = project_dfl(native, weak, DFL_TO_GNM)
    # Native [2, 3, 4] becomes [2, -3, -4]. Perspective depth is 6 - 4 = 2.
    assert perspective_pixels[0].tolist() == [69.5, 19.5]
    assert weak_pixels[0].tolist() == [59.5, 14.5]
    edge_points = native.new_tensor([[-0.5, -0.5], [99.5, 0], [0, 59.5], [0, -0.5001], [2, 2]])
    supported = torch.tensor([True, True, True, True, False])
    assert in_frame(edge_points, size, supported).tolist() == [True, False, False, False, False]
