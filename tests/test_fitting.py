"""Integration and numerical checks using the pinned GNM Head asset."""

from dataclasses import replace
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from persona_lab.benchmark.dataset import camera_to_record, load_dataset
from persona_lab.benchmark.evaluate import evaluate_predictions
from persona_lab.benchmark.configuration import resolve_config
from persona_lab.benchmark.generate import generate_dataset
from persona_lab.benchmark.metrics import coefficient_errors, rotation_errors_deg, vertex_errors
from persona_lab.benchmark.model import load_benchmark_model, project_dfl
from persona_lab.benchmark.predictions import write_predictions
from persona_lab.camera import PerspectiveCamera, WeakPerspectiveCamera
from persona_lab.coordinates import ImageSize, pixels_to_normalized
from persona_lab.correspondences import CorrespondenceMap, DFL_TO_GNM
from persona_lab.fit import main
from persona_lab.fitting import (
    FitConfig, FitInitialization, LevenbergMarquardtConfig, ObservationInitialization, Stage,
    fit_identity, load_result, save_result,
)
from persona_lab.fitting.initialization import initialize
from persona_lab.fitting.losses import evaluate_objective, project_batch
from persona_lab.fitting.observations import prepare_observations
from persona_lab.fitting.observability import identity_observability
from persona_lab.fitting.parameters import FittingParameters, resolve_selection, validate_model
from persona_lab.observations import LandmarkObservation


@pytest.fixture(scope="module")
def asset():
    return os.environ.get("PERSONALAB_GNM_MODEL_FILE")


@pytest.fixture(scope="module")
def model(asset):
    return load_benchmark_model(asset)


@pytest.fixture(scope="module")
def benchmark_path(tmp_path_factory, asset):
    path = tmp_path_factory.mktemp("fitting_cases") / "expression"
    generate_dataset(path, resolve_config("C", camera="weak-perspective"), model_file=asset)
    return path


def _synthesize(model, *, camera_kind="both", include_cameras=True, varied=True, count=3):
    head_index = model.metadata["joint_names"].index("head")
    coefficient = model.metadata["identity_names"].index("head_000")
    lower_face = model.metadata["expression_names"].index("lower_face_region_000")
    ref = model.reference
    identity = ref.new_zeros(model.head.identity_dim)
    identity[coefficient] = 0.42
    root = ref.new_zeros(3)
    observations, expressions, poses, cameras = [], [], [], []
    for index in range(count):
        size = ImageSize(512 + 32 * index, 512 - 16 * index)
        center = ref.new_tensor(((size.width - 1) / 2, (size.height - 1) / 2))
        kind = ("weak-perspective" if index % 2 else "perspective") if camera_kind == "both" else camera_kind
        if kind == "perspective":
            camera = PerspectiveCamera(size, root, ref.new_tensor((0, 0.265, 0.65)),
                                       ref.new_full((2,), 1.5 * min(size.width, size.height)), center)
        else:
            scale = ref.new_tensor(970 + index * 80)
            camera = WeakPerspectiveCamera(size, root, center + scale * ref.new_tensor((0, 0.265)), scale)
        expression = ref.new_zeros(model.head.expression_dim)
        pose = ref.new_zeros((model.head.num_joints, 3))
        if varied:
            expression[lower_face] = (0, -0.45, 0.3)[index % 3]
            pose[head_index] = ref.new_tensor(((0.08, 0.58, -0.08) if index % 2 else (0, -0.28, 0.05)))
        points = model.landmarks(identity, expression, pose, root)
        pixels = project_dfl(points, camera, DFL_TO_GNM)
        observations.append(LandmarkObservation(
            observation_id=f"image_{index}", image_size=size, landmarks=pixels,
            known_camera=camera if include_cameras else None,
        ))
        expressions.append(expression)
        poses.append(pose)
        cameras.append(camera)
    return tuple(observations), identity, torch.stack(expressions), torch.stack(poses), tuple(cameras)


def _config(**overrides):
    defaults = {
        "train_identity": ("head_000",), "train_expression": ("lower_face_region_000",),
        "pose_initializer": "neutral", "identity_weight": 0, "expression_weight": 0,
        "stages": (Stage("fit", 1, ("identity", "expression", "pose", "camera"), 0.005),),
    }
    return FitConfig(**(defaults | overrides))


def _parameters(model, observations, config, initial=None):
    data = prepare_observations(observations, model.reference, config)
    selection = resolve_selection(model, config)
    state, choices = initialize(model, data, config, FitInitialization() if initial is None else initial)
    return data, FittingParameters(model, data, state, selection, config), choices


def test_validate_model_checks_flat_unhashable_gnm_tensors(model):
    head = model.head
    validate_model(model)
    buffer = model.reference
    assert not buffer.requires_grad
    try:
        buffer.requires_grad_(True)
        with pytest.raises(ValueError, match="model tensors must be fixed"):
            validate_model(model)
    finally:
        buffer.requires_grad_(False)
    head.register_parameter("_fitting_test_parameter", torch.nn.Parameter(buffer.new_zeros(1)))
    try:
        with pytest.raises(ValueError, match="model tensors must be fixed"):
            validate_model(model)
    finally:
        delattr(head, "_fitting_test_parameter")
    head.add_module("_fitting_test_submodule", torch.nn.Identity())
    try:
        with pytest.raises(ValueError, match="without submodules"):
            validate_model(model)
    finally:
        delattr(head, "_fitting_test_submodule")
    validate_model(model)


def test_global_loss_and_batch_gradients_match_for_mixed_inputs(model):
    observations, *_ = _synthesize(model)
    first, second, third = observations
    mask = first.validity.clone()
    mask[::6] = False
    weights = second.weights.clone()
    weights[::4] = 2.5
    weights[2] = 0
    entries = list(third.correspondences.entries)
    entries[0], entries[1] = replace(entries[0], gnm_index=1), replace(entries[1], gnm_index=0)
    entries[2] = replace(entries[2], gnm_index=None, enabled_by_default=False, limitation="test")
    observations = (
        replace(first, validity=mask),
        replace(second, weights=weights,
                landmarks=pixels_to_normalized(second.landmarks, second.image_size),
                coordinate_convention="normalized_aligned"),
        replace(third, correspondences=CorrespondenceMap(tuple(entries))),
    )
    initial_identity = model.reference.new_zeros(253)
    initial_identity[0] = 0.12
    initial_expression = model.reference.new_zeros(383)
    component = model.metadata["expression_names"].index("lower_face_region_000")
    initial_expression[component] = -0.1
    initial = FitInitialization(identity=initial_identity, observations={
        second.observation_id: ObservationInitialization(expression=initial_expression),
    })
    checks = []
    for batch_size in (3, 2, 1):
        config = _config(batch_size=batch_size, identity_weight=0.03, expression_weight=0.12)
        data, params, _ = _parameters(model, observations, config, initial)
        assert params.identity.shape == (1,)
        active = tuple((key, p) for key, group in params.groups().items() for p in group)
        losses, stats = evaluate_objective(model, data, params, active)
        assert losses.identity == pytest.approx(0.03 * 0.12**2)
        assert losses.expression == pytest.approx(0.12 * 0.1**2 / 3)
        assert stats.overall.eligible_count == 3 * 68 - int((~mask).sum()) - 2
        if batch_size == len(observations):
            with torch.no_grad():
                projected = project_batch(model, params.state(0, len(observations)), data.ids).cpu().numpy()
            numerator, denominator = 0.0, 0.0
            for obs, projection in zip(observations, projected, strict=True):
                expected = obs.aligned_pixels().cpu().numpy()
                for entry in obs.correspondences.entries:
                    index = entry.dfl_index
                    weight = float(obs.weights[index])
                    if not (entry.supported and entry.enabled_by_default and obs.validity[index] and weight):
                        continue
                    residual = (projection[entry.gnm_index] - expected[index]) / min(
                        obs.image_size.width, obs.image_size.height,
                    )
                    numerator += weight * float(residual @ residual)
                    denominator += weight
            assert losses.landmarks == pytest.approx(numerator / denominator, rel=1e-12)
        grads = tuple(p.grad.detach().clone() for _, p in active)
        assert [g.shape for g in grads] == [(1,), (3, 1), (3, 3)]
        assert all(g.isfinite().all() and g.abs().max() > 0 for g in grads)

        baseline = params.identity.item()
        delta = 1e-5
        with torch.no_grad():
            params.identity.fill_(baseline + delta)
            upper, _ = evaluate_objective(model, data, params)
            params.identity.fill_(baseline - delta)
            lower, _ = evaluate_objective(model, data, params)
        assert grads[0].item() == pytest.approx((upper.total - lower.total) / (2 * delta), rel=2e-4)

        checks.append((losses.total, stats.overall.rms_px, grads))
    for loss, rms, grads in checks[1:]:
        assert loss == pytest.approx(checks[0][0], rel=1e-9)
        assert rms == pytest.approx(checks[0][1], rel=1e-9)
        for grad, expected in zip(grads, checks[0][2], strict=True):
            torch.testing.assert_close(grad, expected, atol=1e-10, rtol=1e-8)


def test_controlled_identity_improves_independent_neutral_geometry_and_round_trips(model, tmp_path):
    observations, target, _, _, cameras = _synthesize(model, varied=False)
    config = _config(
        train_head=False, train_expression=(),
        stages=(Stage("identity", 45, ("identity",), 0.035),),
    )
    result = fit_identity(model, observations, config)
    assert result.status == "budget_exhausted", result.reason
    assert result.initial_reprojection.overall.eligible_count == 3 * 68
    assert result.final_reprojection.overall.rms_px < result.initial_reprojection.overall.rms_px / 2
    assert not result.initial.identity.any()
    np.testing.assert_array_equal(result.final.identity[1:], 0)
    truth = target.cpu().numpy()
    assert coefficient_errors(result.final.identity, truth)["rmse"] < coefficient_errors(
        result.initial.identity, truth,
    )["rmse"] / 2
    neutral = model.neutral(truth)
    assert vertex_errors(model.neutral(result.final.identity), neutral)["rms_m"] < vertex_errors(
        model.neutral(result.initial.identity), neutral,
    )["rms_m"] / 2
    for actual, camera in zip(result.final.cameras, cameras, strict=True):
        assert camera_to_record(actual) == camera_to_record(camera)
    output = tmp_path / "fit"
    save_result(output, result)
    decoded = load_result(output, model=model)
    assert (decoded.config, decoded.selection, decoded.history, decoded.stages, decoded.observation_ids) == (
        result.config, result.selection, result.history, result.stages, result.observation_ids,
    )
    np.testing.assert_array_equal(decoded.final.identity, result.final.identity)
    with pytest.raises(FileExistsError, match="exists"):
        save_result(output, result)


@pytest.mark.parametrize("kind", ("weak-perspective", "perspective"))
def test_camera_gradients_and_independent_updates(model, kind):
    observations, _, _, _, _ = _synthesize(model, camera_kind=kind, include_cameras=False)
    config = _config(
        camera_mode="estimated", camera_model=kind, pose_initializer="grid",
        stages=(Stage("camera", 5, ("pose", "camera"), 0.003),),
    )
    data, parameters, choices = _parameters(model, observations, config)
    assert all(choice.strategy == "grid" and choice.candidates == 45 for choice in choices)
    with torch.no_grad():
        parameters.camera_delta[:, 0] += 0.02
        parameters.camera_delta[:, 2] += 0.01
    active = (("camera", parameters.camera_delta),)
    evaluate_objective(model, data, parameters, active)
    gradient = parameters.camera_delta.grad.detach().clone()
    assert gradient.shape == (3, 3) and torch.isfinite(gradient).all()
    assert gradient[:, 2].abs().max() > 0
    assert gradient[:, :2].abs().max() > 0
    result = fit_identity(model, observations, config)
    assert result.status == "budget_exhausted", result.reason
    assert result.final_reprojection.overall.rms_px < result.initial_reprojection.overall.rms_px
    np.testing.assert_array_equal(result.final.translation, 0)
    for initial, final in zip(result.initial.cameras, result.final.cameras, strict=True):
        torch.testing.assert_close(initial.rotation, final.rotation, rtol=0, atol=0)
        if kind == "perspective":
            torch.testing.assert_close(initial.focal_length, final.focal_length, rtol=0, atol=0)
            torch.testing.assert_close(initial.principal_point, final.principal_point, rtol=0, atol=0)
            assert final.translation[2] > 0
        else:
            assert final.scale > 0


def test_stage_freezing_explicit_fixed_values_and_batch_size(model):
    observations, _, _, _, _ = _synthesize(model)
    expression = model.reference.new_zeros(383)
    expression[model.metadata["expression_names"].index("pupils_000")] = 0.19
    identity = model.reference.new_zeros(253)
    identity[model.metadata["identity_names"].index("teeth_000")] = -0.31
    initial = FitInitialization(identity=identity, observations={
        observations[1].observation_id: ObservationInitialization(expression=expression),
    })
    config = _config(
        train_identity=("head_000", "teeth"), fixed_identity=("teeth",), train_head=False,
        stages=(Stage("identity", 1, ("identity",), 0.02),
                Stage("expression", 1, ("expression",), 0.02)),
    )
    identity_only = fit_identity(model, observations, replace(config, stages=config.stages[:1]), initial=initial)
    result = fit_identity(model, observations, config, initial=initial)
    chunked = fit_identity(model, observations, replace(config, batch_size=2), initial=initial)
    assert identity_only.status != "failed" and result.status != "failed" and chunked.status != "failed"
    assert result.selection.identity == ("head_000",)
    assert result.stages[0].active_groups == ("identity",)
    assert result.stages[1].active_groups == ("expression",)
    np.testing.assert_array_equal(identity_only.initial.expression, identity_only.final.expression)
    np.testing.assert_array_equal(identity_only.final.identity, result.final.identity)
    assert result.final.identity[model.metadata["identity_names"].index("teeth_000")] == -0.31
    assert result.final.expression[1, model.metadata["expression_names"].index("pupils_000")] == 0.19
    np.testing.assert_allclose(result.final.identity, chunked.final.identity, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(result.final.expression, chunked.final.expression, rtol=1e-8, atol=1e-12)
    np.testing.assert_array_equal(result.initial.rotations, result.final.rotations)

    data, parameters, _ = _parameters(model, observations, config, initial)
    before = parameters.state(0, len(observations))
    with torch.no_grad():
        parameters.expression[1, 0] += 0.25
    after = parameters.state(0, len(observations))
    assert not torch.equal(before.expression[1], after.expression[1])
    assert torch.equal(before.expression[[0, 2]], after.expression[[0, 2]])
    assert torch.equal(before.identity, after.identity)
    assert data.ids == tuple(obs.observation_id for obs in observations)


def test_invalid_inputs_and_known_camera_gauges(model):
    observations, _, _, _, cameras = _synthesize(model)
    first = observations[0]
    with pytest.raises(ValueError, match="at least one"):
        fit_identity(model, (), _config())
    with pytest.raises(ValueError, match="IDs must be unique"):
        fit_identity(model, (first, first), _config())
    with pytest.raises(ValueError, match=f"{first.observation_id} has no eligible"):
        fit_identity(model, (replace(first, weights=first.weights * 0),), _config())
    with pytest.raises(ValueError, match="has a known camera"):
        fit_identity(model, (first,), _config(camera_mode="estimated", camera_model="perspective"))
    with pytest.raises(ValueError, match="fixed perspective camera"):
        fit_identity(model, observations, _config(train_root=True))
    with pytest.raises(ValueError, match="requires positive"):
        fit_identity(
            model, (replace(first, known_camera=None, camera_initialization=replace(
                cameras[0], translation=cameras[0].translation * 0,
            )),),
            _config(camera_mode="estimated", camera_model="perspective"),
        )
    with pytest.raises(ValueError, match="no admissible"):
        camera = replace(cameras[0], translation=model.reference.new_tensor((0, 0, -1)))
        fit_identity(model, (replace(first, known_camera=camera),), _config())
    with pytest.raises(ValueError, match="Unknown native"):
        fit_identity(model, (first,), _config(train_expression=("jaw_joint",)))
    with pytest.raises(ValueError, match="Initial identity"):
        fit_identity(model, (first,), _config(), initial=FitInitialization(
            identity=model.reference.new_zeros(253).float(),
        ))
    with pytest.raises(ValueError, match="operating range"):
        fit_identity(model, (first,), _config(), initial=FitInitialization(observations={
            first.observation_id: ObservationInitialization(expression=model.reference.new_full((383,), 4)),
        }))


def test_convergence_and_failed_gradient_round_trip(model, monkeypatch, tmp_path):
    observations, *_ = _synthesize(model)
    already_converged = fit_identity(model, observations, _config(gradient_tolerance=1e12))
    assert already_converged.status == "converged"
    assert already_converged.runtime["optimizer_updates"] == 0

    def corrupt(gradient):
        return torch.full_like(gradient, float("nan"))

    old_parameter = torch.nn.Parameter
    with monkeypatch.context() as patch:
        def broken_parameter(*args, **kwargs):
            value = old_parameter(*args, **kwargs)
            value.register_hook(corrupt)
            return value

        from persona_lab.fitting import parameters as implementation
        patch.setattr(implementation.FittingParameters, "_parameter", staticmethod(broken_parameter))
        failed = fit_identity(model, observations, _config())
    assert failed.status == "failed" and "nonfinite gradient" in failed.reason.lower()
    assert failed.final is None and failed.final_reprojection is None
    output = tmp_path / "failed"
    save_result(output, failed)
    restored = load_result(output, model=model)
    assert (restored.status, restored.reason, restored.final) == ("failed", failed.reason, None)
    with pytest.raises(ValueError, match="failed"):
        restored.to_prediction()


def test_saved_result_rejects_incompatible_and_malformed_data(model, tmp_path):
    observations, *_ = _synthesize(model)
    fit = fit_identity(model, observations, _config())
    path = tmp_path / "fit"
    save_result(path, fit)
    manifest = path / "result.json"
    original = json.loads(manifest.read_text())
    for updates, expected in (
        ({"schema_version": 99}, "schema"),
        ({"observation_ids": [fit.observation_ids[0]] * 3}, "unique"),
        ({"dtype": "int64"}, "dtype"),
        ({"model": original["model"] | {"asset_sha256": "0" * 64}}, "Incompatible"),
        ({"initial_cameras": list(reversed(original["initial_cameras"]))}, "order"),
        ({"final_reprojection": None}, "final parameters"),
    ):
        manifest.write_text(json.dumps(original | updates))
        with pytest.raises(ValueError, match=expected):
            load_result(path, model=model)
    manifest.write_text(json.dumps(original))
    parameters = path / "parameters.npz"
    with np.load(parameters, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    np.savez_compressed(parameters, **(arrays | {"final_identity": np.full((253,), np.nan)}))
    with pytest.raises(ValueError, match="nonfinite"):
        load_result(path, model=model)
    np.savez_compressed(parameters, **(arrays | {"final_identity": np.zeros(252)}))
    with pytest.raises(ValueError, match="shape"):
        load_result(path, model=model)


def test_benchmark_inputs_are_blind_and_predictions_evaluate(model, asset, benchmark_path, tmp_path, monkeypatch):
    dataset = load_dataset(benchmark_path)
    subject, = dataset.subject_ids
    ids = dataset.observation_ids(subject)[::3]

    def forbidden(*args, **kwargs):
        raise AssertionError("Hidden synthetic ground truth must stay out of the fitter")

    with monkeypatch.context() as patch:
        patch.setattr(type(dataset), "ground_truth", forbidden)
        unknown = dataset.fitting_observations(subject, observation_ids=ids, include_known_cameras=False)
        assert all(obs.known_camera is None and obs.camera_initialization is None for obs in unknown)
        assert all(not hasattr(obs, name) for obs in unknown for name in ("view", "expression", "outliers", "clean"))
        fit = fit_identity(model, unknown, _config(
            camera_mode="estimated", camera_model="weak-perspective", pose_initializer="grid",
            stages=(Stage("pose_camera", 3, ("pose", "camera"), 0.001),),
        ))
    assert fit.status == "budget_exhausted", fit.reason
    selected = dataset.select_observations(subject, ids)
    assert selected.observation_ids(subject) == ids
    assert selected.dataset_id != dataset.dataset_id
    prediction_path = tmp_path / "prediction"
    write_predictions(prediction_path, selected, {subject: fit.to_prediction()})
    report = evaluate_predictions(selected, prediction_path, mode="complete", model_file=asset)
    assert report["scored_observation_count"] == len(ids)
    assert report["aggregate"]["identity"]["coefficient_count"] == 253
    assert report["aggregate"]["neutral_geometry"]["vertex_count"] == 17821
    assert report["aggregate"]["reprojection"]["observed"]["rms_px"] == pytest.approx(
        fit.final_reprojection.overall.rms_px,
    )


def test_unknown_pose_improves_rotation_error(model):
    observations, _, _, truth, _ = _synthesize(model)
    config = _config(stages=(Stage("pose", 30, ("pose",), 0.015),))
    result = fit_identity(model, observations, config)
    assert result.status == "budget_exhausted", result.reason
    index = model.metadata["joint_names"].index("head")
    angles = truth[:, index].cpu().numpy()
    assert rotation_errors_deg(result.final.rotations[:, index], angles).mean() < (
        rotation_errors_deg(result.initial.rotations[:, index], angles).mean()
    )


def test_local_observability_separates_identity_and_expression(model):
    observations, *_ = _synthesize(model, varied=False, count=1)
    config = _config(landmark_subset=(8, 30, 36, 45, 51, 57), train_head=False)
    report = identity_observability(model, observations, config)
    assert report["identity_names"] == ["head_000"]
    assert report["fixed_nuisance"]["rank"] == 1
    assert report["after_expression"]["column_l2"][0] <= report["fixed_nuisance"]["column_l2"][0]
    assert report["after_expression"] == report["after_all_nuisance"]


def test_known_camera_cli_reproducible_across_processes(model, asset, benchmark_path, tmp_path):
    dataset = load_dataset(benchmark_path)
    subject, = dataset.subject_ids
    results = []
    for index in range(2):
        output = tmp_path / f"fit-{index}"
        command = [
            sys.executable, "-m", "persona_lab.fit", "--benchmark", str(benchmark_path),
            "--subject", subject, "--camera-mode", "known", "--max-images", "3",
            "--iterations", "1", "1", "1", "--batch-size", "2", "--no-neutral-mesh",
            "--output", str(output), "--predictions", str(tmp_path / f"predictions-{index}"),
        ]
        if asset is not None:
            command.extend(("--model-file", asset))
        subprocess.run(command, check=True, capture_output=True, text=True)
        results.append(load_result(output, model=model))
    first, second = results
    assert first.observation_ids == second.observation_ids
    assert first.initialization == second.initialization
    np.testing.assert_array_equal(first.initial.rotations, second.initial.rotations)
    for field in ("identity", "expression", "rotations", "translation"):
        np.testing.assert_allclose(
            getattr(first.final, field), getattr(second.final, field), rtol=1e-10, atol=1e-12,
        )
    subset = dataset.select_observations(subject, first.observation_ids)
    report = evaluate_predictions(
        subset, tmp_path / "predictions-0", mode="known-camera", model_file=asset,
    )
    assert report["scored_observation_count"] == 3
    assert report["aggregate"]["reprojection"]["observed"]["rms_px"] == pytest.approx(
        first.final_reprojection.overall.rms_px,
    )


def test_real_input_cli_with_synthetic_dfl_jpeg(model, asset, dfl_jpeg, tmp_path):
    observations, *_ = _synthesize(model, camera_kind="weak-perspective", count=1)
    observation, = observations
    jpeg = tmp_path / "aligned.jpg"
    jpeg.write_bytes(dfl_jpeg({"landmarks": observation.landmarks.cpu().tolist()}, size=(512, 512)))
    output = tmp_path / "result"
    args = [
        "--input", str(jpeg), "--camera", "weak-perspective", "--output", str(output),
        "--max-images", "1", "--iterations", "1", "0", "0", "--overlays", "1",
    ]
    if asset is not None:
        args.extend(("--model-file", asset))
    assert main(args) == 0
    fitted = load_result(output, model=model)
    assert len(fitted.observation_ids) == 1
    assert (output / "neutral.obj").stat().st_size > 0
    assert (output / "overlay_0001_initial.png").is_file()
    assert (output / "overlay_0001_final.png").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_official_model_fitting(asset):
    model = load_benchmark_model(asset, dtype="float32", device="cuda")
    observations, *_ = _synthesize(model, varied=False, count=1)
    fit = fit_identity(model, observations, _config(
        train_head=False, stages=(Stage("id", 2, ("identity",), 0.01),),
    ))
    assert fit.device.startswith("cuda:") and fit.status == "budget_exhausted", fit.reason


def _lm_config(**overrides):
    return _config(**({"solver": "levenberg-marquardt"} | overrides))


@pytest.mark.parametrize("mode,kind,root", [
    ("known", "both", False), ("known", "perspective", True), ("estimated", "perspective", False),
])
def test_lm_residuals_match_all_objective_terms_and_view_derivatives(model, mode, kind, root):
    from persona_lab.fitting.levenberg_marquardt import (
        StageCoordinates, normal_blocks, observation_jacobian, observation_residual,
    )

    observations, *_ = _synthesize(model, camera_kind=kind, include_cameras=mode == "known", count=2)
    first, second = observations
    validity = first.validity.clone()
    validity[::3] = False
    observations = (replace(first, validity=validity), replace(second, weights=second.weights * 2.3))
    config = _lm_config(
        camera_mode=mode, camera_model=kind if mode == "estimated" else None, train_root=root,
        identity_weight=0.03, expression_weight=0.02, pose_weight=0.01, camera_weight=0.04,
    )
    data, params, _ = _parameters(model, observations, config)
    with torch.no_grad():
        for group in params.groups().values():
            for parameter in group:
                parameter.add_(0.03)
    original, _ = evaluate_objective(model, data, params)
    least_squares, _ = evaluate_objective(model, data, params, least_squares=True)
    assert least_squares.total == pytest.approx(original.total, rel=1e-12)
    for name in ("landmarks", "identity", "expression", "pose", "camera"):
        assert getattr(least_squares, name) == pytest.approx(getattr(original, name), abs=1e-14)

    layout = StageCoordinates.from_parameters(params, tuple(name for name, group in params.groups().items() if group))
    base = params.values()
    position = layout.pack(base, model.reference, len(data))
    blocks = normal_blocks(model, data, params, layout, base, position)
    data_curvature, data_gradient = 0.0, 0.0
    for index in range(len(data)):
        residual, jacobian = observation_jacobian(model, data, params, layout, base, position, index)
        point = torch.cat((position.identity, position.local[index]))
        epsilon = 1e-6
        with torch.no_grad():
            numeric = torch.stack([
                (observation_residual(model, data, params, layout, base, position, index, point + direction)
                 - observation_residual(model, data, params, layout, base, position, index, point - direction))
                / (2 * epsilon)
                for direction in epsilon * torch.eye(point.numel(), dtype=point.dtype)
            ], dim=1)
        torch.testing.assert_close(jacobian, numeric, atol=2e-8, rtol=3e-5)
        torch.testing.assert_close(blocks.coupling[index, 0], jacobian[:, 0] @ jacobian[:, 1:])
        assert float(jacobian[:, 0].square().sum()) > 0
        data_curvature += jacobian[:, 0].square().sum()
        data_gradient += jacobian[:, 0] @ residual
    torch.testing.assert_close(blocks.identity[0, 0], data_curvature + config.identity_weight)
    torch.testing.assert_close(
        blocks.gradient.identity[0], data_gradient + config.identity_weight * position.identity[0],
    )


def test_lm_recovers_controlled_neutral_identity_and_round_trips(model, tmp_path):
    observations, identity, *_ = _synthesize(model, varied=False, count=3)
    config = _lm_config(
        train_head=False, train_expression=(), stages=(Stage("identity", 8, ("identity",), 0.045),),
    )
    result = fit_identity(model, observations, config)
    assert result.final is not None, result.reason
    np.testing.assert_allclose(result.final.identity, identity.numpy(), atol=1e-5, rtol=0)
    assert vertex_errors(model.neutral(result.final.identity), model.neutral(identity))["rms_m"] < 1e-7
    assert result.final_reprojection.overall.rms_px < 1e-4
    assert result.optimization.initial_objective.total > result.optimization.final_objective.total
    assert len(result.optimization.updates) == sum(trial.accepted for trial in result.optimization.trials)
    for start, stop in zip(result.initial.cameras, result.final.cameras, strict=True):
        assert camera_to_record(start) == camera_to_record(stop)
    saved = tmp_path / "lm"
    save_result(saved, result)
    assert load_result(saved, model=model).optimization == result.optimization
    repeated = fit_identity(model, observations, config)
    np.testing.assert_array_equal(repeated.final.identity, result.final.identity)
    assert repeated.optimization == result.optimization


def test_lm_joint_expression_and_pose_recovery(model):
    observations, _, expressions, rotations, _ = _synthesize(model)
    config = _lm_config(stages=(
        Stage("pose", 6, ("pose",), 0.018),
        Stage("joint", 10, ("identity", "expression", "pose"), 0.022),
    ))
    result = fit_identity(model, observations, config)
    assert result.final is not None, result.reason
    assert result.final_reprojection.overall.rms_px < result.initial_reprojection.overall.rms_px * 0.1
    assert np.linalg.norm(result.final.expression - expressions.numpy()) < np.linalg.norm(expressions.numpy())
    head = model.metadata["joint_names"].index("head")
    initial_error = rotation_errors_deg(result.initial.rotations[:, head], rotations[:, head].numpy())
    final_error = rotation_errors_deg(result.final.rotations[:, head], rotations[:, head].numpy())
    assert np.mean(final_error) < np.mean(initial_error)
    assert np.unique(result.final.expression, axis=0).shape[0] > 1
    np.testing.assert_array_equal(result.final.translation, result.initial.translation)
    np.testing.assert_array_equal(result.final.rotations[:, [0, 2, 3]], result.initial.rotations[:, [0, 2, 3]])


@pytest.mark.parametrize("kind", ["weak-perspective", "perspective"])
def test_lm_estimated_cameras_preserve_gauge(model, kind):
    observations, *_ = _synthesize(model, camera_kind=kind, include_cameras=False, count=2)
    config = _lm_config(camera_mode="estimated", camera_model=kind, stages=(
        Stage("pose_camera", 6, ("pose", "camera"), 0.018),
        Stage("joint", 8, ("identity", "expression", "pose", "camera"), 0.022),
    ))
    result = fit_identity(model, observations, config)
    assert result.final is not None, result.reason
    assert result.final_reprojection.overall.rms_px < result.initial_reprojection.overall.rms_px
    final = result.final.cameras
    assert not torch.equal(final[0].translation, final[1].translation)
    for before, after in zip(result.initial.cameras, final, strict=True):
        assert torch.equal(before.rotation, after.rotation)
        if isinstance(after, PerspectiveCamera):
            assert after.translation[2] > 0
            assert torch.equal(before.focal_length, after.focal_length)
            assert torch.equal(before.principal_point, after.principal_point)
        else:
            assert after.scale > 0


@pytest.mark.parametrize("limit", ["transient", "trials", "damping"])
def test_lm_rejects_invalid_geometry_without_committing_trial(model, monkeypatch, tmp_path, limit):
    from persona_lab.fitting import monitoring
    from persona_lab.fitting.parameters import InvalidGeometry

    observations, *_ = _synthesize(model, varied=False, count=1)
    options = LevenbergMarquardtConfig(
        max_trials=2, maximum_damping=1e-3 if limit == "damping" else 1e8,
    )
    config = _lm_config(lm=options, train_expression=(), train_head=False,
                        stages=(Stage("identity", 1, ("identity",), 0.045),))
    calculate = monitoring.evaluate_objective
    seen = []

    def with_invalid_trial(model, data, parameters, active=(), **kwargs):
        if kwargs.get("candidate") is not None:
            seen.append(parameters.identity.detach().clone())
            assert not torch.equal(parameters.identity, kwargs["candidate"].identity)
            if limit != "transient" or len(seen) == 1:
                raise InvalidGeometry("Perspective projection requires positive camera depth")
        return calculate(model, data, parameters, active, **kwargs)

    monkeypatch.setattr(monitoring, "evaluate_objective", with_invalid_trial)
    result = fit_identity(model, observations, config)
    assert all(torch.equal(value, seen[0]) for value in seen)
    trials = result.optimization.trials
    assert "invalid_geometry" in trials[0].reason and not trials[0].accepted
    if limit == "transient":
        assert result.status == "budget_exhausted", result.reason
        assert trials[1].accepted and trials[1].damping > trials[0].damping
        assert result.optimization.accepted_updates == 1
    else:
        assert result.status == "no_acceptable_step" and result.final is None
        assert result.optimization.final_objective == result.optimization.initial_objective
        assert result.optimization.rejected_trials == (2 if limit == "trials" else 1)
        path = tmp_path / "rejected"
        save_result(path, result)
        with pytest.raises(ValueError, match="failed"):
            load_result(path, model=model).to_prediction()


@pytest.mark.parametrize("failure", ["residual", "Jacobian", "factorization"])
def test_lm_reports_nonfinite_linearization_or_factorization_failure(model, monkeypatch, failure):
    from persona_lab.fitting import levenberg_marquardt as lm

    observations, *_ = _synthesize(model, varied=False, count=1)
    config = _lm_config(train_expression=(), train_head=False)
    original = lm.weighted_landmark_residuals

    def corrupt(*args):
        residual = original(*args)
        if failure == "residual":
            return residual * float("nan")
        residual.register_hook(lambda gradient: gradient * float("nan"))
        return residual

    if failure == "factorization":
        cholesky = torch.linalg.cholesky_ex

        def failed_factor(*args, **kwargs):
            factor, info = cholesky(*args, **kwargs)
            return factor, torch.ones_like(info)

        monkeypatch.setattr(torch.linalg, "cholesky_ex", failed_factor)
    else:
        monkeypatch.setattr(lm, "weighted_landmark_residuals", corrupt)

    def forbidden(*args, **kwargs):
        raise AssertionError("An LM failure must never start Adam")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    fit = fit_identity(model, observations, config)
    assert fit.status == "numerical_failure" and fit.final is None
    assert failure in fit.reason, fit.reason


@pytest.mark.parametrize("count,start,end,identity_weight", [
    (3, 2.11171e-8, 3.85021e-6, 0.0), (1, 5.9825e-8, 7.17356e-6, 1e-6),
])
def test_adam_stage_transition_guard_rejects_recorded_loss_jump(
    model, monkeypatch, count, start, end, identity_weight,
):
    from persona_lab.fitting import monitoring, optimizer

    observations, *_ = _synthesize(model, varied=count == 1, count=count)
    stages = (
        Stage("identity", 1, ("identity",), 0.045),
        Stage("joint", 1, ("identity", "expression", "pose"), 0.022, (("pose", 0.005),)),
    )
    config = _config(identity_weight=identity_weight, expression_weight=2e-5, stages=stages)
    calculate = monitoring.evaluate_objective
    calls = 0

    def recorded_losses(*args, **kwargs):
        nonlocal calls
        losses, stats = calculate(*args, **kwargs)
        calls += 1
        # Reproduce the observed objective jump at the next Adam stage,
        # independent of model and backend roundoff in the preceding stages.
        return replace(losses, total=end if calls == 5 else start), stats

    def forbidden(*args, **kwargs):
        raise AssertionError("An Adam failure must never start LM")

    monkeypatch.setattr(monitoring, "evaluate_objective", recorded_losses)
    monkeypatch.setattr(optimizer, "fit_lm_stage", forbidden)
    failed = fit_identity(model, observations, config)
    assert failed.status == "failed" and failed.final is None
    assert failed.reason.startswith("joint: Objective diverged")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_lm_preserves_requested_precision_and_device(asset, dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    model = load_benchmark_model(asset, dtype=dtype, device=device)
    observations, *_ = _synthesize(model, varied=False, count=1)
    result = fit_identity(model, observations, _lm_config(
        train_head=False, train_expression=(), stages=(Stage("identity", 2, ("identity",), 0.045),),
    ))
    assert result.final is not None, result.reason
    assert result.dtype == dtype and result.device.split(":")[0] == device
    assert result.final.identity.dtype == np.dtype(dtype)
    assert result.final_reprojection.overall.rms_px < result.initial_reprojection.overall.rms_px


def test_explicit_lm_cli_prediction_uses_complete_dataset_id(model, asset, benchmark_path, tmp_path):
    dataset = load_dataset(benchmark_path)
    subject, = dataset.subject_ids
    ids = dataset.observation_ids(subject)
    output, predictions = tmp_path / "fit-lm", tmp_path / "predictions"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"train_head": False, "train_expression": [], "identity_weight": 0}))
    args = [
        "--benchmark", str(benchmark_path), "--subject", subject,
        "--config", str(config), "--solver", "levenberg-marquardt",
        "--iterations", "0", "1", "0", "--train-identity", "head_000",
        "--output", str(output), "--predictions", str(predictions), "--no-neutral-mesh",
    ]
    if asset is not None:
        args.extend(("--model-file", asset))
    for key in ids:
        args.extend(("--observation-id", key))
    assert main(args) == 0
    result = load_result(output, model=model)
    assert result.config.solver == "levenberg-marquardt"
    assert result.observation_ids == ids
    assert result.runtime["input"]["evaluation_dataset_id"] == dataset.dataset_id
    report = evaluate_predictions(
        dataset.select_observations(subject, ids), predictions, mode="known-camera", model_file=asset,
    )
    assert report["scored_observation_count"] == len(ids)


def test_original_adam_result_version_one_round_trip(model, tmp_path):
    observations, *_ = _synthesize(model, varied=False, count=1)
    result = fit_identity(model, observations, _config(train_head=False, train_expression=()))
    path = tmp_path / "original-adam"
    save_result(path, result)
    manifest_path = path / "result.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest.pop("optimization")
    manifest["config"].pop("solver")
    manifest["config"].pop("lm")
    manifest_path.write_text(json.dumps(manifest))
    restored = load_result(path, model=model)
    assert restored.config.solver == "adam" and restored.optimization is None
    np.testing.assert_array_equal(restored.final.identity, result.final.identity)


def test_study_pairs_preserve_observation_order_and_comparison_controls(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from persona_lab.benchmark import fitting_study

    generated = {}

    def generate(path, config, **kwargs):
        assert str(path) not in generated
        generated[str(path)] = config

    monkeypatch.setattr(fitting_study, "generate_dataset", generate)
    args = SimpleNamespace(
        output=tmp_path, device="cpu", dtype="float64", model_file=None,
        experiments=("E", "F", "H"), seeds=(42, 43), scaling_full=False,
        adam_iterations=(30, 70, 900), lm_iterations=(10, 12, 30), spectra=False,
    )
    jobs = list(fitting_study._jobs(args))
    adam_cases = [item for item in jobs if item["config"]["solver"] == "adam"]
    lm_cases = [item for item in jobs if item["config"]["solver"] == "levenberg-marquardt"]
    assert len(adam_cases) == len(lm_cases) == 2 * 2 * 6 + 4
    for adam, lm in zip(adam_cases, lm_cases, strict=True):
        for key in ("dataset", "observations", "seed", "held_out_dataset", "budget"):
            assert adam[key] == lm[key]
        for key in ("identity_weight", "expression_weight", "train_identity", "train_expression", "pose_initializer"):
            assert adam["config"][key] == lm["config"][key]
        if adam["experiment"] == "E":
            assert generated[adam["dataset"]].observations_per_subject == 100
            assert generated[adam["held_out_dataset"]].observations_per_subject == 109
        else:
            assert adam["budget"] == "profiling_only"
            assert [stage["iterations"] for stage in lm["config"]["stages"]] == [1, 1, 1]
    assert {item["observations"] for item in adam_cases if item["experiment"] == "H"} == {50, 100, 300, 600}
    assert fit.final_reprojection.overall.rms_px < fit.initial_reprojection.overall.rms_px
