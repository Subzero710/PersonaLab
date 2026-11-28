"""Native dense correspondence, precision and square-root integration regression tests."""

from dataclasses import replace
from hashlib import sha256

from PIL import Image
import pytest
import torch

from persona_lab.benchmark.model import load_benchmark_model
from persona_lab.benchmark.storage import read_json, write_json
from persona_lab.dense_fit_diagnostics import write_dense_projections
from persona_lab.dense_observations import ImageProvenance
from persona_lab.dense_observability import ProjectionProtocol, analyze_jacobians, projection_jacobian
from persona_lab.dense_reference import oracle_definition, oracle_observations, oracle_state
from persona_lab.fitting import FitConfig, LevenbergMarquardtConfig, Stage, fit_identity
from persona_lab.fitting.batched_jacobian import batch_jacobian
from persona_lab.fitting.configuration import GROUPS
from persona_lab.fitting.initialization import initialize
from persona_lab.fitting.levenberg_marquardt import StageCoordinates, observation_jacobian
from persona_lab.fitting.observations import prepare_observations
from persona_lab.fitting.parameters import FitInitialization, FittingParameters, TensorState, resolve_selection
from persona_lab.surface_correspondences import NativeLocation, load_correspondences
from persona_lab.validate_dense import validate_geometry


@pytest.fixture(scope="module")
def model():
    # As with the historical tests, this requires the pinned official GNM NPZ.
    return load_benchmark_model(dtype="float64", device="cpu")


def test_native_location_validation_and_order(tmp_path, model):
    definition = oracle_definition(model, 83)
    with pytest.raises(ValueError, match="weights"):
        NativeLocation("triangle", 0, (0.2, 0.1, 0.1))
    with pytest.raises(ValueError, match="schema order"):
        replace(definition, entries=tuple(reversed(definition.entries)))
    with pytest.raises(ValueError, match="topology"):
        replace(definition, topology_sha256="0" * 64).bind(model)
    first, *remaining = definition.entries
    missing = replace(first, location=NativeLocation("vertex", model.head.num_vertices))
    with pytest.raises(ValueError, match="outside"):
        replace(definition, entries=(missing, *remaining)).bind(model)
    mismatch = replace(first, native_groups=("teeth",))
    with pytest.raises(ValueError, match="native groups"):
        replace(definition, entries=(mismatch, *remaining)).bind(model)
    left = next(i for i, point in enumerate(definition.schema.points) if point.region == "left_cheek")
    points = list(definition.schema.points)
    points[left] = replace(points[left], side="left")
    replace(definition, schema=replace(definition.schema, points=tuple(points))).bind(model)
    points[left] = replace(points[left], side="right")
    with pytest.raises(ValueError, match="native side"):
        replace(definition, schema=replace(definition.schema, points=tuple(points))).bind(model)

    path = tmp_path / "map.json"
    write_json(path, definition.to_record())
    assert load_correspondences(path, definition.schema) == definition
    official = replace(first, location=NativeLocation("landmark", 30), native_groups=("skin",))
    bound = replace(definition, entries=(official, *remaining)).bind(model)
    state = oracle_state(model, 43, 1)
    vertices, landmarks = model.head.vertices_and_landmarks(
        landmarks_type="head_sparse_68", identity=state.identity, expression=state.expressions[0],
        rotations=state.rotations[0], translation=state.translations[0],
    )
    torch.testing.assert_close(bound.evaluate(vertices)[0], landmarks[30], rtol=0, atol=1e-12)


def test_visibility_does_not_supply_mapping_or_confidence(model):
    definition = oracle_definition(model, 45)
    state = oracle_state(model, 44, 1)
    obs, = oracle_observations(model, definition, 44, state)
    first, second, *rest = definition.entries
    restricted = replace(first, minimum_visibility=0.8)
    unmapped = replace(second, location=None, native_groups=())
    bound = replace(definition, entries=(restricted, unmapped, *rest)).bind(model)
    with pytest.raises(ValueError, match="actual visibility"):
        bound.eligible(obs)
    values = obs.landmarks.new_full((45,), -1)
    values[:2] = 0.5
    found = bound.eligible(replace(obs, visibility=values))
    assert not found[0] and not found[1] and found[2:].all()
    assert obs.validity[0] and obs.confidence is None


@pytest.mark.parametrize("count", [79, 131])
def test_dense_fit_residual_order_and_batch_parity(model, count):
    definition = oracle_definition(model, count)
    state = oracle_state(model, 42, 3)
    observations = oracle_observations(model, definition, 42, state)
    middle = observations[1]
    mask = middle.validity.clone()
    mask[1] = False
    observations = (observations[0], replace(middle, validity=mask), observations[2])
    config = FitConfig(
        train_identity=("head_000", "head_002"), train_expression=("left_eye_region_002",),
        lm=LevenbergMarquardtConfig(fp64_square_root=True, jacobian_rows=24),
        stages=(Stage("joint", 2, ("identity", "expression", "pose"), 0.02),),
    )
    with pytest.raises(ValueError, match="explicit reviewed"):
        fit_identity(model, observations, config)
    data = prepare_observations(observations, model.reference, config, model=model, correspondences=definition)
    assert data.counts.tolist() == [count, count - 1, count]
    geometry = data.surface.sparse_geometry(model)
    initial, _ = initialize(model, data, config, FitInitialization(), geometry=geometry)
    parameters = FittingParameters(model, data, initial, resolve_selection(model, config), config, geometry=geometry)
    layout = StageCoordinates.from_parameters(parameters, GROUPS)
    base = parameters.values()
    position = layout.pack(base, model.reference, len(data))
    residual, jacobian = batch_jacobian(data, parameters, layout, position, 0, len(data))
    assert residual.shape == (3, 2 * count)
    assert jacobian.shape == (3, 2 * count, 6)
    assert not jacobian[1, 2:4].count_nonzero()
    for index in range(len(data)):
        r, j = observation_jacobian(data, parameters, layout, base, position, index)
        active = data.eligible[index].repeat_interleave(2)
        torch.testing.assert_close(residual[index, active], r, rtol=1e-11, atol=1e-13)
        torch.testing.assert_close(jacobian[index, active], j, rtol=1e-10, atol=1e-12)
    result = fit_identity(model, observations, config, correspondences=definition)
    assert result.final_reprojection.overall.rms_px < result.initial_reprojection.overall.rms_px
    chosen = replace(config, landmark_subset=(68, 69, 70))
    data = prepare_observations(observations, model.reference, chosen, model=model, correspondences=definition)
    assert data.counts.tolist() == [3, 3, 3]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_mesh_gradient_and_projection_validation(model, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware and CUDA PyTorch required")
    if device == "cuda":
        model = load_benchmark_model(device=device)
    definition = oracle_definition(model, 96)
    observations, errors = validate_geometry(model, definition, 44, 2)
    assert all(int(obs.validity.sum()) == 96 for obs in observations)
    assert errors["gradient_columns"] == 643
    assert errors["max_sparse_vs_numpy_mesh_m"] < 2e-12
    assert errors["max_sparse_vs_full_gradient_in_probe_px"] < 2e-9
    assert errors["max_sparse_vs_full_elementwise_jacobian_first_view_px"] < 2e-9
    assert errors["max_central_difference_gradient_in_probe_px"] < 1e-5
    assert errors["max_batch_vs_single_m"] < 2e-12


def test_seed_ordering_and_full_native_numerical_rank(model):
    definition = oracle_definition(model, 37)
    assert definition == oracle_definition(model, 37)
    state = oracle_state(model, 42, 1)
    assert torch.equal(state.expressions, oracle_state(model, 42, 1).expressions)
    assert state.identity[170:].count_nonzero() == 83
    assert state.expressions.count_nonzero() == 383
    protocol = ProjectionProtocol(state, 0, 1)
    geometry = definition.bind(model).sparse_geometry(model)
    pixels, jacobian = projection_jacobian(protocol, geometry, 24)
    p, j = projection_jacobian(protocol, geometry, 61)
    torch.testing.assert_close(pixels, p, rtol=0, atol=0)
    torch.testing.assert_close(jacobian, j, rtol=1e-10, atol=1e-12)
    report, nulls = analyze_jacobians(
        model, [jacobian], tuple(point.region for point in definition.schema.points),
        "estimated", 1e-8,
    )
    result = report["identity"]["after_all_nuisance"]
    assert result["matrix_dimensions"] == [74, 253]
    assert len(result["singular_values"]) == 253
    assert result["nullity"] == 253 - result["rank"]
    assert result["full_space_condition_number"] is None
    assert result["implicit_zero_singular_values"] == 179
    assert report["expression_by_view"][0]["before_nuisance"]["parameter_count"] == 383


def test_dense_projection_overlay_and_source_validation(tmp_path, model):
    definition = oracle_definition(model, 35)
    state = oracle_state(model, 42, 1)
    obs, = oracle_observations(model, definition, 42, state)
    raster = tmp_path / "raster.png"
    Image.new("RGB", (obs.image_size.width, obs.image_size.height)).save(raster)
    obs = replace(obs, source=ImageProvenance(raster.name, sha256(raster.read_bytes()).hexdigest(),
                                             obs.image_size))
    snapshot = TensorState(state.identity, state.expressions, state.rotations, state.translations, state.cameras)
    bound = definition.bind(model)
    write_dense_projections(tmp_path, model, (obs,), bound, (("initial", snapshot),), (0,), raster)
    manifest = read_json(tmp_path / "dense_mapping_diagnostics.json")
    row, = manifest["observations"]
    predicted = row["initial"]["projection_by_point_id"]
    assert row["valid_mapped"] == 35 and set(predicted) == set(definition.schema.point_ids)
    assert (tmp_path / row["initial"]["overlay"]).is_file()
    Image.new("RGB", (obs.image_size.width, obs.image_size.height), color="white").save(raster)
    with pytest.raises(ValueError, match="raster changed"):
        write_dense_projections(tmp_path, model, (obs,), bound, (("initial", snapshot),), (0,), raster)
