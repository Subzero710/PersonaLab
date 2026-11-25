"""DeepFaceLab correspondence, masks, and the Issue #2 adapter."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from persona_lab.camera import WeakPerspectiveCamera, project
from persona_lab.coordinates import ImageSize, pixels_to_normalized
from persona_lab.correspondences import CorrespondenceMap, DFL_TO_GNM, Region
from persona_lab.dfl import read_jpeg
from persona_lab.observations import LandmarkObservation


SIZE = ImageSize(384, 276)


def points():
    return torch.arange(136, dtype=torch.float64).reshape(68, 2)


def observation(**changes):
    return LandmarkObservation(**{
        "observation_id": "synthetic", "image_size": SIZE, "landmarks": points(), **changes,
    })


def test_regions_and_anatomical_index_anchors():
    entries = DFL_TO_GNM.entries
    assert tuple(entry.dfl_index for entry in entries) == tuple(range(68))
    assert sum(entry.supported for entry in entries) == 68
    assert sum(entry.enabled_by_default for entry in entries) == 68
    assert [entry.dfl_index for entry in entries if entry.view_dependent] == list(range(17))
    expected = {
        0: ("Upper right facial outline", Region.CONTOUR),
        8: ("Center of the chin", Region.CONTOUR),
        16: ("Upper left facial outline", Region.CONTOUR),
        17: ("Right eyebrow lateral end", Region.EYEBROW),
        22: ("Left eyebrow medial end", Region.EYEBROW),
        30: ("Nose tip", Region.NOSE_BRIDGE),
        33: ("Midline nasal base", Region.NOSE_BASE),
        36: ("Right lateral eye corner", Region.EYE),
        39: ("Right medial eye corner", Region.EYE),
        42: ("Left medial eye corner", Region.EYE),
        45: ("Left lateral eye corner", Region.EYE),
        48: ("Right outer mouth corner", Region.OUTER_MOUTH),
        54: ("Left outer mouth corner", Region.OUTER_MOUTH),
        60: ("Right inner mouth corner", Region.INNER_MOUTH),
        64: ("Left inner mouth corner", Region.INNER_MOUTH),
    }
    for index, (description, region) in expected.items():
        assert entries[index].gnm_index == index
        assert (entries[index].description, entries[index].region) == (description, region)


def test_remapped_and_unsupported_slots_keep_original_order_and_are_masked():
    entries = list(DFL_TO_GNM.entries)
    entries[18] = replace(entries[18], gnm_index=19)
    entries[19] = replace(entries[19], gnm_index=18)
    entries[20] = replace(entries[20], gnm_index=None, enabled_by_default=False, limitation="Insufficient evidence")
    weights = torch.ones(68, dtype=torch.float64)
    weights[19] = 2
    validity = torch.ones(68, dtype=torch.bool)
    validity[21] = False
    weights[22] = 0
    item = observation(correspondences=CorrespondenceMap(tuple(entries)), weights=weights, validity=validity)
    predicted = points().requires_grad_()
    pairs = item.landmark_pairs(predicted, subset=[22, 21, 20, 19, 18, 0])
    assert pairs.dfl_indices.tolist() == [0, 18, 19]
    assert pairs.gnm_indices.tolist() == [0, 19, 18]
    assert pairs.weights.tolist() == [1, 1, 2]
    assert not item.supported_mask[20]
    assert item.landmarks.shape == (68, 2) and torch.equal(item.landmarks[20], points()[20])
    torch.testing.assert_close(pairs.observed, points()[[0, 18, 19]])
    torch.testing.assert_close(pairs.projected, points()[[0, 19, 18]])
    (pairs.projected * pairs.weights[:, None]).sum().backward()
    assert predicted.grad[19].tolist() == [1, 1] and predicted.grad[18].tolist() == [2, 2]
    assert predicted.grad[0].tolist() == [1, 1]
    assert not predicted.grad[20].any()
    assert item.landmark_pairs(predicted, subset=[]).projected.shape == (0, 2)


def test_all_landmarks_are_active_and_differentiable_by_default():
    item = observation()
    predicted = points().requires_grad_()
    assert item.fitting_mask().all()
    pairs = item.landmark_pairs(predicted)
    assert pairs.dfl_indices.tolist() == list(range(68))
    assert pairs.gnm_indices.tolist() == list(range(68))
    assert pairs.weights.tolist() == [1] * 68
    pairs.projected.sum().backward()
    torch.testing.assert_close(predicted.grad, torch.ones_like(predicted))


def test_explicit_pose_mask_and_subset_apply_to_all_regions():
    item = observation()
    pose = torch.ones(68, dtype=torch.bool)
    pose[0] = pose[36] = False
    mask = item.fitting_mask(pose_validity=pose)
    assert mask.sum() == 66
    assert not mask[0] and mask[1:17].all() and not mask[36]
    assert item.fitting_mask().all()
    assert item.fitting_mask(subset=[0, 8, 16]).sum() == 3
    assert not item.fitting_mask(subset=list(range(17, 68)))[:17].any()
    with pytest.raises(ValueError, match="Pose validity"):
        item.fitting_mask(pose_validity=torch.ones(68))
    for subset in ([1, 1], [-1], [68], [True]):
        with pytest.raises(ValueError, match="Subset"):
            item.fitting_mask(subset=subset)


@pytest.mark.parametrize(("index", "changes"), [
    (21, {"gnm_index": 22}), (21, {"dfl_index": 17}), (1, {"view_dependent": "invalid"}),
    (21, {"gnm_index": 68}), (21, {"gnm_index": None}), (21, {"description": ""}),
])
def test_rejects_inconsistent_correspondence_maps(index, changes):
    entries = list(DFL_TO_GNM.entries)
    entries[index] = replace(entries[index], **changes)
    with pytest.raises(ValueError):
        CorrespondenceMap(tuple(entries))
    with pytest.raises(ValueError, match="68"):
        CorrespondenceMap(tuple(entries[:-1]))


def test_dfl_adapter_reuses_actual_issue_2_ingestion(dfl_file, dfl_points):
    source_to_crop = [[1.5, 0, -3], [0, 1.5, 4]]
    source_points = (dfl_points.astype(np.float64) - [-3, 4]) / 1.5
    sample = read_jpeg(dfl_file({
        "landmarks": dfl_points.tolist(), "source_landmarks": source_points,
        "face_type": "whole_face", "source_filename": "synthetic-source.jpg",
        "source_rect": [1, 2, 101, 112], "image_to_face_mat": source_to_crop,
    }, size=(384, 276)))
    camera = WeakPerspectiveCamera(SIZE, torch.zeros(3, dtype=torch.float64), points()[1], points().new_tensor(4.5))
    item = LandmarkObservation.from_dfl(sample, camera_initialization=camera)
    assert item.image_size == SIZE and item.observation_id == sample.sample_id
    assert item.landmark_ordering == "deepfacelab_68"
    np.testing.assert_array_equal(item.landmarks.numpy(), sample.landmarks)
    assert item.alignment.face_type == "whole_face"
    assert item.alignment.source_filename == "synthetic-source.jpg"
    assert item.alignment.source_rect.tolist() == [1, 2, 101, 112]
    np.testing.assert_allclose(item.source_pixels().numpy(), source_points, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(item.alignment.source_landmarks.numpy(), source_points)
    assert item.known_camera is None and item.camera_initialization is camera
    assert item.validity.all() and item.supported_mask.all() and item.fitting_mask().sum() == 68
    assert item.alignment.image_to_face_mat.tolist() == source_to_crop

    gnm_points = torch.cat([points(), points().new_ones((68, 1))], dim=1)
    pixels = project(gnm_points, camera)
    synthetic = observation(
        landmarks=pixels_to_normalized(pixels, SIZE),
        known_camera=camera, coordinate_convention="normalized_aligned",
    )
    torch.testing.assert_close(synthetic.landmark_pairs(pixels).observed, synthetic.landmark_pairs(pixels).projected)
    assert torch.equal(item.landmark_pairs(pixels).projected, synthetic.landmark_pairs(pixels).projected)
    assert synthetic.camera_initialization is None


def test_missing_affine_and_other_optional_fields(dfl_file):
    item = LandmarkObservation.from_dfl(read_jpeg(dfl_file()), dtype=torch.float32)
    assert item.alignment.image_to_face_mat is None and item.alignment.source_landmarks is None
    assert item.known_camera is item.camera_initialization is None
    assert item.landmarks.dtype == torch.float32
    with pytest.raises(ValueError, match="unavailable"):
        item.source_pixels()


@pytest.mark.parametrize("changes", [
    {"observation_id": ""}, {"image_size": (512, 512)}, {"landmarks": torch.zeros(67, 2)},
    {"landmarks": torch.zeros(1, 68, 2)}, {"landmarks": torch.full((68, 2), float("nan"))},
    {"validity": torch.ones(68)}, {"weights": -torch.ones(68, dtype=torch.float64)},
    {"coordinate_convention": "original_pixels"},
    {"known_camera": WeakPerspectiveCamera(ImageSize(1, 1), points().new_zeros(3), points().new_zeros(2), points().new_tensor(1))},
])
def test_invalid_observation(changes):
    with pytest.raises(ValueError):
        observation(**changes)
