"""Official-model coverage, physical probes and paired scenarios."""

from dataclasses import replace
from itertools import chain
import os

import numpy as np
import pytest
import torch

from persona_lab.benchmark.configuration import EXPRESSION_NAMES, VIEW_AXES, resolve_config
from persona_lab.benchmark.dataset import load_dataset
from persona_lab.benchmark.generate import generate_dataset
from persona_lab.benchmark.model import MODEL_SHA256, load_benchmark_model
from persona_lab.correspondences import CorrespondenceMap, DFL_TO_GNM
from persona_lab.gnm_geometry import load_sparse_landmarks


@pytest.fixture(scope="module")
def asset():
    return os.environ.get("PERSONALAB_GNM_MODEL_FILE")


@pytest.fixture(scope="module")
def datasets(tmp_path_factory, asset):
    root = tmp_path_factory.mktemp("benchmark_scenarios")
    datasets = {}
    for scenario in "ABCDEF":
        options = {"subjects": 2} if scenario == "F" else {}
        output = root / scenario
        written = generate_dataset(output, resolve_config(scenario, **options), model_file=asset)
        decoded = load_dataset(output)
        assert written.dataset_id == decoded.dataset_id
        datasets[scenario] = decoded
    return datasets


def test_model_metadata_neutral_mesh_and_independent_frontal_projection(datasets, asset):
    dataset = datasets["A"]
    metadata = dataset.model_metadata
    assert metadata["asset_sha256"] == MODEL_SHA256
    assert (metadata["version"], metadata["variant"]) == ("3.0", "head")
    assert (metadata["identity_count"], metadata["expression_count"]) == (253, 383)
    assert metadata["joint_names"] == ["neck", "head", "left_eye", "right_eye"]
    assert (metadata["vertex_count"], metadata["triangle_count"]) == (17821, 35324)
    assert dataset.triangles.shape == (35324, 3)
    subject = dataset.subject_ids[0]
    truth = dataset.ground_truth(subject)
    assert truth.arrays["identity"].shape == (253,)
    assert np.any(truth.arrays["identity"] != 0)
    assert not truth.arrays["expression"].any() and not truth.arrays["rotations"].any()
    assert not truth.arrays["translation"].any()
    model = load_benchmark_model(asset)
    assert metadata == model.metadata
    np.testing.assert_array_equal(dataset.triangles, model.triangles)
    vertices = model.head(identity=model.tensor(truth.arrays["identity"])).detach().numpy()
    np.testing.assert_allclose(truth.arrays["neutral_vertices"], vertices, atol=1e-12, rtol=0)
    sparse = load_sparse_landmarks(model.head)
    native = np.einsum("ijk,ij->ik", vertices[sparse.indices], sparse.weights)
    camera = truth.observations[0]["camera"]
    px, py, pz = (np.array(camera["translation"]) + native * [1, -1, -1]).T
    focal = np.array(camera["focal_length"])
    # Independent pinhole reference using the stored intrinsics and frontal axes.
    expected = np.column_stack((px / pz, py / pz)) * focal + camera["principal_point"]
    np.testing.assert_allclose(truth.arrays["clean"][0], expected, atol=1e-10, rtol=0)
    fitted, = dataset.fitting_observations(subject)
    assert fitted.fitting_mask().tolist() == [True] * 68
    assert fitted.weights.tolist() == [1] * 68 and fitted.known_camera is not None
    np.testing.assert_array_equal(fitted.landmarks.numpy(), truth.arrays["observed"][0])


def test_scenario_coverage_and_paired_clean_observations(datasets):
    for scenario, dataset in datasets.items():
        for subject in dataset.subject_ids:
            truth = dataset.ground_truth(subject)
            views = {item["view"] for item in truth.observations}
            expressions = {item["expression"] for item in truth.observations}
            assert views == ({"frontal"} if scenario == "A" else set(VIEW_AXES))
            assert expressions == ({"neutral"} if scenario in "AB" else set(EXPRESSION_NAMES))
            assert truth.arrays["identity"].shape == (253,)
            assert truth.arrays["expression"].shape == (len(truth.observations), 383)
            assert truth.arrays["rotations"].shape == (len(truth.observations), 4, 3)
            assert truth.arrays["validity"].any(axis=1).all()
            if scenario in "ABC":
                assert truth.arrays["validity"].all()
                np.testing.assert_array_equal(truth.arrays["observed"], truth.arrays["clean"])
            if scenario == "A":
                continue
            cameras = [item["camera"] for item in truth.observations]
            assert len({cam["scale"] for cam in cameras if cam["type"] == "weak-perspective"}) >= 2
            assert len({cam["translation"][2] for cam in cameras if cam["type"] == "perspective"}) >= 2
            for index, item in enumerate(truth.observations):
                axis, angle = VIEW_AXES[item["view"]]
                expected = np.zeros((4, 3))
                if axis is not None:
                    expected[1, axis] = np.deg2rad(angle)
                np.testing.assert_allclose(truth.arrays["rotations"][index], expected, atol=1e-15, rtol=0)
                assert item["camera"]["rotation"] == [0, 0, 0]
    c = datasets["C"].ground_truth(datasets["C"].subject_ids[0])
    for scenario in "DE":
        dataset = datasets[scenario]
        comparison = dataset.ground_truth(dataset.subject_ids[0])
        for field in ("identity", "neutral_vertices", "expression", "rotations", "translation", "clean"):
            np.testing.assert_array_equal(c.arrays[field], comparison.arrays[field])
        assert [item["camera"] for item in c.observations] == [
            item["camera"] for item in comparison.observations
        ]
        assert np.any(comparison.arrays["observed"] != c.arrays["observed"])
    e = datasets["E"].ground_truth(datasets["E"].subject_ids[0]).arrays
    assert (e["missing"].sum(axis=1) == 4).all() and (e["outliers"].sum(axis=1) == 3).all()
    f = datasets["F"]
    a, b = (f.ground_truth(subject) for subject in f.subject_ids)
    assert not np.array_equal(a.arrays["identity"], b.arrays["identity"])
    assert not np.array_equal(a.arrays["neutral_vertices"], b.arrays["neutral_vertices"])
    assert {item["noise_profile"] for item in chain(a.observations, b.observations)} == {
        "clean", "small", "moderate", "structured", "combined",
    }
    assert a.arrays["occluded"].any()


def test_remapped_and_unsupported_slots_round_trip(tmp_path, asset):
    entries = list(DFL_TO_GNM.entries)
    entries[0] = replace(entries[0], gnm_index=1)
    entries[1] = replace(entries[1], gnm_index=0)
    entries[2] = replace(
        entries[2], gnm_index=None, enabled_by_default=False, limitation="Test unsupported row",
    )
    mapping = CorrespondenceMap(tuple(entries))
    config = resolve_config("A", image_size=(640, 360), camera="weak-perspective", dtype="float32")
    dataset = generate_dataset(tmp_path / "remapped", config, model_file=asset, correspondences=mapping)
    item, = dataset.fitting_observations(dataset.subject_ids[0], dtype=torch.float32)
    assert dataset.correspondences == mapping
    assert item.landmarks.dtype == torch.float32
    assert (item.image_size.width, item.image_size.height) == (640, 360)
    assert item.fitting_mask().sum().item() == 67
    assert not item.fitting_mask()[2] and item.landmarks[2].tolist() == [0, 0]
    assert item.landmarks[1, 0] < item.landmarks[0, 0]


def test_named_expressions_move_model_anatomy_with_one_identity(datasets, asset):
    dataset = datasets["C"]
    truth = dataset.ground_truth(dataset.subject_ids[0])
    identity = truth.arrays["identity"]
    model = load_benchmark_model(asset)
    resource = load_sparse_landmarks(model.head)
    reference = truth.arrays["neutral_vertices"]

    def points(vertices):
        return np.einsum("ijk,ij->ik", vertices[resource.indices], resource.weights)

    neutral = points(reference)
    by_name = {}
    for index, item in enumerate(truth.observations):
        name = item["expression"]
        if name in by_name:
            continue
        expression = truth.arrays["expression"][index]
        vertices = model.head(identity=model.tensor(identity), expression=model.tensor(expression))
        by_name[name] = points(vertices.detach().numpy())
        if name != "neutral":
            assert np.max(np.abs(by_name[name] - neutral)) > 1e-5
    np.testing.assert_allclose(by_name["neutral"], neutral, atol=1e-12, rtol=0)
    for name in ("smile", "smile_raise_brows"):
        moved = by_name[name] - neutral
        assert (moved[[48, 54], 1] > moved[[51, 57], 1].mean()).all()
        assert moved[54, 0] > moved[48, 0]
    for name in ("raise_brows", "smile_raise_brows"):
        moved = by_name[name] - neutral
        assert moved[17:22, 1].mean() > moved[[36, 39], 1].mean()
        assert moved[22:27, 1].mean() > moved[[42, 45], 1].mean()
    assert (by_name["mouth_open"][62, 1] - by_name["mouth_open"][66, 1]) > neutral[62, 1] - neutral[66, 1]
    for upper, lower in (([37, 38], [40, 41]), ([43, 44], [46, 47])):
        closed, resting, opened = (
            landmarks[upper, 1].mean() - landmarks[lower, 1].mean()
            for landmarks in (by_name["eyes_close"], neutral, by_name["eyes_open"])
        )
        assert 0 < closed < resting < opened
    np.testing.assert_array_equal(identity, dataset.ground_truth(truth.subject_id).arrays["identity"])


def test_different_seeds_change_sampled_identity_and_clean_projection(datasets, tmp_path, asset):
    a = datasets["A"]
    b = generate_dataset(tmp_path / "new-seed", resolve_config("A", seed=43), model_file=asset)
    args = (a.ground_truth(a.subject_ids[0]), b.ground_truth(b.subject_ids[0]))
    for name in ("identity", "neutral_vertices", "clean"):
        assert not np.array_equal(args[0].arrays[name], args[1].arrays[name])
