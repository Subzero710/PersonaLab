"""Real-model benchmark IO, external predictions and separate-process validation."""

from dataclasses import replace
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from persona_lab.benchmark.configuration import resolve_config
from persona_lab.benchmark.dataset import _manifest_id, load_dataset
from persona_lab.benchmark.evaluate import evaluate_predictions
from persona_lab.benchmark.generate import generate_dataset
from persona_lab.benchmark.predictions import oracle_predictions, read_predictions, write_predictions
from persona_lab.benchmark.storage import read_json, write_json


@pytest.fixture(scope="module")
def asset():
    return os.environ.get("PERSONALAB_GNM_MODEL_FILE")


@pytest.fixture(scope="module")
def mixed(tmp_path_factory, asset):
    path = tmp_path_factory.mktemp("benchmark_io") / "mixed"
    dataset = generate_dataset(path, resolve_config("F", subjects=2), model_file=asset)
    return path, dataset


def test_fitting_loader_uses_ids_without_exposing_hidden_state(mixed):
    path, generated = mixed
    assert read_json(path / "manifest.json")["coordinates"]["visibility"] == "physical visibility not evaluated"
    dataset = load_dataset(path)
    subject = dataset.subject_ids[0]
    ids = dataset.observation_ids(subject)
    truth = dataset.ground_truth(subject)
    selected = dataset.fitting_observations(subject, observation_ids=(ids[4], ids[0]), dtype="float32")
    assert [item.observation_id for item in selected] == [ids[4], ids[0]]
    assert dataset.dataset_id == generated.dataset_id
    item = selected[0]
    for name in ("identity", "expression", "rotations", "outliers", "missing", "clean"):
        assert not hasattr(item, name)
    assert item.camera_initialization is None
    assert item.alignment is None
    np.testing.assert_array_equal(item.landmarks.numpy(), truth.arrays["observed"][4].astype(np.float32))
    np.testing.assert_array_equal(item.fitting_mask().numpy(), truth.arrays["validity"][4])
    np.testing.assert_array_equal(item.weights.numpy(), truth.arrays["weights"][4])
    assert dataset.fitting_observations(subject)[4].known_camera.image_size == item.image_size
    assert truth.arrays["outliers"][4].any()
    assert truth.arrays["validity"][4, truth.arrays["outliers"][4]].all()
    with pytest.raises(ValueError, match="duplicate or unknown"):
        dataset.fitting_observations(subject, observation_ids=(ids[0], ids[0]))
    with pytest.raises(ValueError, match="duplicate or unknown"):
        dataset.fitting_observations(subject, observation_ids=(dataset.observation_ids(dataset.subject_ids[1])[0],))


def _edit_manifest(path, edit, *, dataset=False):
    file = path / "manifest.json"
    manifest = read_json(file)
    edit(manifest)
    if dataset:
        manifest["dataset_id"] = _manifest_id(manifest)
    write_json(file, manifest)


@pytest.mark.parametrize("case,match", [
    ("schema", "schema version"), ("asset", "pinned GNM"), ("duplicate", "subject identifier"),
    ("camera", "positive"), ("topology", "topology"), ("dtype", "must be float64"),
    ("shape", "shape"), ("nan", "nonfinite"), ("object", "Object arrays"),
    ("missing", "Cannot load"), ("corrupt", "Cannot load"),
])
def test_dataset_rejects_malformed_serialization(mixed, tmp_path, case, match):
    original, dataset = mixed
    copy = tmp_path / "dataset"
    shutil.copytree(original, copy)
    subject = dataset.subject_ids[0]
    archive = copy / f"{subject}.npz"
    if case == "schema":
        _edit_manifest(copy, lambda manifest: manifest.update(schema_version=99), dataset=True)
    elif case == "asset":
        _edit_manifest(copy, lambda manifest: manifest["model"].update(asset_sha256="0" * 64), dataset=True)
    elif case == "duplicate":
        _edit_manifest(copy, lambda manifest: manifest["subjects"][1].update(id=subject), dataset=True)
    elif case == "camera":
        def invalid_focal(manifest):
            manifest["subjects"][0]["observations"][0]["camera"]["focal_length"][0] = 0
        _edit_manifest(copy, invalid_focal, dataset=True)
    elif case == "topology":
        np.savez_compressed(copy / "topology.npz", triangles=np.roll(dataset.triangles, 1, axis=0))
    elif case == "missing":
        archive.unlink()
    elif case == "corrupt":
        archive.write_bytes(b"PK\x03\x04truncated archive")
    else:
        arrays = dataset.ground_truth(subject).arrays
        identity = arrays["identity"]
        if case == "dtype":
            identity = identity.astype(np.float32)
        elif case == "shape":
            identity = identity[:-1]
        elif case == "object":
            identity = identity.astype(object)
        else:
            identity[0] = float("nan")
        np.savez_compressed(archive, **(arrays | {"identity": identity}))
    with pytest.raises(ValueError, match=match):
        load_dataset(copy)


def test_output_destinations_are_create_only(mixed, asset, tmp_path):
    path, dataset = mixed
    with pytest.raises(FileExistsError):
        generate_dataset(path, dataset.config, model_file=asset)
    predictions = tmp_path / "predictions"
    oracle = oracle_predictions(dataset, mode="identity-only")
    write_predictions(predictions, dataset, oracle)
    with pytest.raises(FileExistsError):
        write_predictions(predictions, dataset, oracle)
    assert read_predictions(predictions, dataset).keys() == oracle.keys()


def test_explicit_complete_oracle_and_perturbed_geometry_and_parameters(mixed, tmp_path, asset):
    _, dataset = mixed
    oracle = oracle_predictions(dataset)
    # Change both subject and row order. Association must use IDs in all modes.
    reverse = {
        key: replace(item, observations=dict(reversed(tuple(item.observations.items()))))
        for key, item in reversed(tuple(oracle.items()))
    }
    output = tmp_path / "oracle"
    write_predictions(output, dataset, reverse)
    report = evaluate_predictions(dataset, output, mode="complete", model_file=asset)
    assert report["subject_count"] == 2 and report["scored_observation_count"] == 18
    metrics = report["aggregate"]
    assert metrics["identity"]["rmse"] == 0
    assert metrics["neutral_geometry"]["rms_m"] < 1e-12
    assert metrics["reprojection"]["clean"]["rms_px"] < 1e-10
    assert metrics["reprojection"]["observed"]["rms_px"] > 0
    assert metrics["reprojection_by_noise_profile"]["clean"]["observed"]["rms_px"] < 1e-10
    assert metrics["reprojection_by_noise_profile"]["moderate"]["observed"]["rms_px"] > 0
    assert metrics["missing_count"] > 0 and metrics["occluded_count"] > 0 and metrics["outlier_count"] > 0
    clean_distances = [
        value for subject in report["subjects"] for observation in subject["observations"]
        for value in observation["reprojection"]["observed"]["per_landmark_px"] if value is not None
    ]
    assert metrics["reprojection"]["observed"]["rms_px"] == pytest.approx(
        (sum(value * value for value in clean_distances) / len(clean_distances)) ** 0.5,
    )
    first = report["subjects"][0]["observations"][0]
    assert first["expression_recovery"]["max_absolute"] == 0
    assert first["pose_recovery"]["joint_rotation_deg"]["head"] < 1e-12
    assert first["camera_recovery"]["translation_l2_m"] == 0
    second = report["subjects"][0]["observations"][1]
    assert second["camera_recovery"]["scale_abs_px_per_m"] == 0

    subject = dataset.subject_ids[0]
    prediction = oracle[subject]
    key = dataset.observation_ids(subject)[0]
    observation = prediction.observations[key]
    identity = prediction.identity.copy()
    identity[dataset.model_metadata["identity_names"].index("head_000")] += 0.2
    expression = observation.expression.copy()
    expression[dataset.model_metadata["expression_names"].index("lower_face_region_000")] -= 0.3
    rotations = observation.rotations.copy()
    rotations[1, 2] = np.deg2rad(2.5)
    translation = observation.translation.copy()
    translation[1] += 0.003
    camera = replace(
        observation.camera,
        translation=observation.camera.translation + observation.camera.translation.new_tensor([0.002, 0, 0]),
    )
    changed = replace(observation, expression=expression, rotations=rotations, translation=translation, camera=camera)
    changed_path = tmp_path / "perturbed"
    write_predictions(changed_path, dataset, {subject: replace(
        prediction, identity=identity, observations=prediction.observations | {key: changed},
    )})
    with pytest.raises(ValueError, match="Missing required prediction subjects"):
        evaluate_predictions(dataset, changed_path, mode="complete", model_file=asset)
    measured = evaluate_predictions(dataset, changed_path, mode="complete", subject_ids=[subject], model_file=asset)
    assert measured["omitted_subject_ids"] == [dataset.subject_ids[1]]
    metrics = measured["subjects"][0]
    assert metrics["identity"]["max_absolute"] == pytest.approx(0.2)
    assert metrics["neutral_geometry"]["rms_m"] > 0
    first = metrics["observations"][0]
    assert first["reprojection"]["clean"]["mean_px"] > 0
    assert first["expression_recovery"]["max_absolute"] == pytest.approx(0.3)
    assert first["pose_recovery"]["joint_rotation_deg"]["head"] == pytest.approx(2.5)
    assert first["pose_recovery"]["root_translation_l2_m"] == pytest.approx(0.003)
    assert first["camera_recovery"]["translation_l2_m"] == pytest.approx(0.002)


@pytest.mark.parametrize("mode", ["identity-only", "conditional", "known-camera", "projected"])
def test_explicit_evaluation_modes(mixed, tmp_path, asset, mode):
    _, dataset = mixed
    subject = dataset.subject_ids[0]
    prediction, = oracle_predictions(dataset, mode=mode, subject_ids=[subject]).values()
    if mode == "projected":
        key = dataset.observation_ids(subject)[0]
        item = prediction.observations[key]
        points = item.projected.copy()
        points[0] += [3, 4]
        prediction = replace(prediction, observations=prediction.observations | {key: replace(item, projected=points)})
    path = tmp_path / "predictions"
    write_predictions(path, dataset, {subject: prediction})
    report = evaluate_predictions(dataset, path, mode=mode, subject_ids=[subject], model_file=asset)
    metrics = report["subjects"][0]
    if mode == "identity-only":
        assert report["scored_observation_count"] == 0
        assert metrics["reprojection"] is None and metrics["observations"] == []
        assert all(group["status"] == "not_provided" for group in metrics["parameter_recovery"].values())
        with pytest.raises(ValueError, match="requires every observation"):
            evaluate_predictions(dataset, path, mode="complete", subject_ids=[subject], model_file=asset)
        return
    first = metrics["observations"][0]
    if mode == "projected":
        assert first["reprojection"]["clean"]["per_landmark_px"][0] == pytest.approx(5)
        assert first["expression_recovery"]["status"] == first["camera_recovery"]["status"] == "not_provided"
    else:
        assert metrics["reprojection"]["clean"]["rms_px"] < 1e-10
        assert first["camera_recovery"] == {"status": "fixed_known"}
        assert first["pose_recovery"]["status"] == ("evaluated" if mode == "known-camera" else "not_provided")


@pytest.mark.parametrize("case,match", [
    ("model", "parameter ordering"), ("dataset", "different benchmark dataset"),
    ("subject", "Unknown or duplicate"), ("observation", "duplicate prediction observation"),
    ("misassociation", "misassociated"), ("dimensions", "shape"),
    ("nonfinite", "nonfinite"), ("camera", "Unknown benchmark camera"),
    ("schema", "prediction schema version"),
])
def test_external_predictions_reject_invalid_association_and_parameters(mixed, tmp_path, case, match):
    _, dataset = mixed
    oracle = oracle_predictions(dataset)
    path = tmp_path / "predictions"
    write_predictions(path, dataset, oracle)
    if case == "model":
        _edit_manifest(path, lambda manifest: manifest["model"]["identity_names"].reverse())
    elif case == "dataset":
        _edit_manifest(path, lambda manifest: manifest.update(dataset_id="0" * 64))
    elif case == "subject":
        _edit_manifest(path, lambda manifest: manifest["subjects"][1].update(id=dataset.subject_ids[0]))
    elif case in ("observation", "misassociation", "camera"):
        def edit(manifest):
            a, b = manifest["subjects"]
            if case == "observation":
                a["observations"][0]["id"] = a["observations"][1]["id"]
            elif case == "misassociation":
                a["observations"][0]["id"] = b["observations"][0]["id"]
            else:
                a["observations"][0]["camera"]["type"] = "orthographic"
        _edit_manifest(path, edit)
    elif case == "schema":
        _edit_manifest(path, lambda manifest: manifest.update(schema_version=0))
    else:
        archive = path / f"{dataset.subject_ids[0]}.npz"
        with np.load(archive, allow_pickle=False) as file:
            arrays = dict(file)
        if case == "dimensions":
            arrays["identity"] = arrays["identity"][:-1]
        else:
            arrays["identity"][0] = float("inf")
        np.savez_compressed(archive, **arrays)
    with pytest.raises(ValueError, match=match):
        read_predictions(path, dataset)


def _cli(name, *args, asset):
    command = [sys.executable, "-m", f"persona_lab.benchmark.{name}", *map(str, args)]
    if asset is not None:
        command.extend(["--model-file", asset])
    return subprocess.run(command, text=True, capture_output=True, check=False)


def test_generation_and_evaluation_cli_in_separate_processes(tmp_path, asset):
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    for output in (output_a, output_b):
        result = _cli("generate", "--scenario", "D", "--seed", 27, "--output", output, asset=asset)
        assert result.returncode == 0, result.stderr
    first, second = (load_dataset(path) for path in (output_a, output_b))
    assert first.dataset_id == second.dataset_id
    subject = first.subject_ids[0]
    a, b = first.ground_truth(subject), second.ground_truth(subject)
    assert a.observations == b.observations
    for name in a.arrays:
        np.testing.assert_array_equal(a.arrays[name], b.arrays[name])
    prediction_path = tmp_path / "predictions"
    write_predictions(prediction_path, first, oracle_predictions(first, mode="known-camera"))
    report_path = tmp_path / "report"
    result = _cli(
        "evaluate", "--ground-truth", output_a, "--predictions", prediction_path,
        "--output", report_path, "--mode", "known-camera", asset=asset,
    )
    assert result.returncode == 0, result.stderr
    report = read_json(report_path / "report.json")
    assert report["aggregate"]["identity"]["max_absolute"] == 0
    assert report["aggregate"]["reprojection"]["clean"]["rms_px"] < 1e-10
    assert report["aggregate"]["reprojection"]["observed"]["rms_px"] > 0
    duplicate = _cli("generate", "--scenario", "D", "--output", output_a, asset=asset)
    assert duplicate.returncode != 0 and "already exists" in duplicate.stderr
    missing = _cli(
        "evaluate", "--ground-truth", output_a, "--predictions", tmp_path / "missing",
        "--output", tmp_path / "no-report", "--mode", "known-camera", asset=asset,
    )
    assert missing.returncode != 0 and not (tmp_path / "no-report").exists()


def test_explicit_model_path_must_match_the_pinned_checksum(tmp_path):
    wrong_asset = tmp_path / "wrong.npz"
    wrong_asset.write_bytes(b"not the official asset")
    with pytest.raises(ValueError, match="checksum mismatch"):
        generate_dataset(tmp_path / "output", resolve_config("A"), model_file=wrong_asset)
    assert not (tmp_path / "output").exists()
