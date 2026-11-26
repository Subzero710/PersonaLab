"""Structured fitting results and safe, versioned local interchange."""

from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from ..benchmark.dataset import camera_from_record, camera_to_record
from ..benchmark.predictions import ObservationPrediction, SubjectPrediction
from ..benchmark.storage import (
    json_bytes, publish_directory, read_arrays, read_json, require_keys, validate_arrays, write_json,
)
from ..camera import Camera, PerspectiveCamera
from ..correspondences import DFL_LANDMARK_COUNT
from .configuration import FitConfig
from .initialization import InitializationChoice
from .losses import LossValues, PixelStatistics, Reprojection
from .monitoring import Linearization, OptimizationDiagnostics, OptimizationUpdate, Trial, optimization_record
from .parameters import ParameterSelection, TensorState, copy_camera, resolve_selection, validate_model


SCHEMA_VERSION = 4
FAILED_STATUSES = ("failed", "numerical_failure", "no_acceptable_step")
STATUSES = ("budget_exhausted", "converged", "stopped", *FAILED_STATUSES)
ARRAY_FIELDS = ("identity", "expression", "rotations", "translation")


@dataclass(frozen=True, slots=True)
class Iteration:
    iteration: int
    stage: str
    stage_iteration: int
    losses: LossValues
    reprojection: PixelStatistics
    gradient_norms: dict[str, float]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class StageSummary:
    name: str
    active_groups: tuple[str, ...]
    steps: int
    status: str
    initial_loss: float
    final_loss: float
    reason: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class FitState:
    """One snapshot in native order with CPU camera tensors."""

    identity: np.ndarray
    expression: np.ndarray
    rotations: np.ndarray
    translation: np.ndarray
    cameras: tuple[Camera, ...]

    @classmethod
    def from_tensors(cls, state: TensorState):
        arrays = [
            getattr(state, name).detach().cpu().numpy().copy() for name in ARRAY_FIELDS
        ]
        return cls(*arrays, tuple(copy_camera(camera, device="cpu") for camera in state.cameras))


@dataclass(frozen=True, slots=True, eq=False)
class FitResult:
    model: dict
    dtype: str
    device: str
    config: FitConfig
    selection: ParameterSelection
    observation_ids: tuple[str, ...]
    initialization: tuple[InitializationChoice, ...]
    initial: FitState
    final: FitState | None
    initial_reprojection: Reprojection | None
    final_reprojection: Reprojection | None
    history: tuple[Iteration, ...]
    stages: tuple[StageSummary, ...]
    status: str
    reason: str
    runtime: dict
    optimization: OptimizationDiagnostics | None = None

    @property
    def gauge(self):
        estimated = self.config.camera_mode == "estimated"
        return {
            "camera_rotation": "fixed",
            "pose_joints": self.selection.joints,
            "root_translation": "estimated" if self.selection.root_translation else "fixed",
            "weak_perspective": {
                "translation": "estimated_pixels" if estimated else "fixed",
                "scale": "positive_log_ratio" if estimated else "fixed",
            },
            "perspective": {
                "translation": "estimated_metric_xy_and_positive_log_z" if estimated else "fixed",
                "focal_length": {
                    "fixed": "fixed", "shared": "estimated_shared_positive_log_ratio",
                    "per-observation": "estimated_per_observation_positive_log_ratio",
                }[self.config.focal_mode],
                "principal_point": "fixed",
            },
        }

    def to_prediction(self, *, initial=False, dtype=None) -> SubjectPrediction:
        """Convert actual parameters to Issue #4's known-camera or complete input."""
        if self.status in FAILED_STATUSES or self.final is None:
            raise ValueError("A failed run cannot produce a benchmark prediction")
        state = self.initial if initial else self.final
        dtype = np.dtype(self.dtype if dtype is None else dtype)
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("Prediction dtype must be float32 or float64")
        def cast(value):
            return value.astype(dtype, copy=True)

        camera_dtype = torch.float32 if dtype == "float32" else torch.float64
        observations = {}
        for index, key in enumerate(self.observation_ids):
            camera = None
            if self.config.camera_mode == "estimated":
                camera = camera_from_record(
                    camera_to_record(state.cameras[index]), dtype=camera_dtype, device="cpu",
                )
            observations[key] = ObservationPrediction(
                expression=cast(state.expression[index]), rotations=cast(state.rotations[index]),
                translation=cast(state.translation[index]), camera=camera,
            )
        return SubjectPrediction(cast(state.identity), observations)


def _specs(model, count, dtype):
    return {
        "identity": (dtype, (model["identity_count"],)),
        "expression": (dtype, (count, model["expression_count"])),
        "rotations": (dtype, (count, model["joint_count"], 3)),
        "translation": (dtype, (count, 3)),
    }


def _camera_records(state, ids):
    return [
        {"id": key, "camera": camera_to_record(camera)}
        for key, camera in zip(ids, state.cameras, strict=True)
    ]


def _metadata(result):
    return {
        "schema_version": 3 if result.config.focal_mode == "fixed" else SCHEMA_VERSION,
        "model": deepcopy(result.model),
        "dtype": result.dtype,
        "device": result.device,
        "config": result.config.to_dict(),
        "selection": asdict(result.selection),
        "gauge": result.gauge,
        "observation_ids": result.observation_ids,
        "initialization": [asdict(choice) for choice in result.initialization],
        "initial_cameras": _camera_records(result.initial, result.observation_ids),
        "final_cameras": None if result.final is None else _camera_records(result.final, result.observation_ids),
        "initial_reprojection": None if result.initial_reprojection is None else asdict(result.initial_reprojection),
        "final_reprojection": None if result.final_reprojection is None else asdict(result.final_reprojection),
        "history": [asdict(item) for item in result.history],
        "stages": [asdict(item) for item in result.stages],
        "status": result.status,
        "reason": result.reason,
        "runtime": result.runtime,
        "optimization": None if result.optimization is None else optimization_record(result.optimization),
    }


def _write_result(stage, result):
    manifest = _metadata(result)
    dtype = np.dtype(result.dtype)
    specs = _specs(result.model, len(result.observation_ids), dtype)
    arrays = {}
    for prefix in ("initial", "final"):
        state = getattr(result, prefix)
        if state is not None:
            current = {name: getattr(state, name) for name in ARRAY_FIELDS}
            validate_arrays(current, specs, f"{prefix} fitting parameters")
            arrays.update({f"{prefix}_{name}": array for name, array in current.items()})
    np.savez_compressed(stage / "parameters.npz", **arrays)
    write_json(stage / "result.json", manifest)


def save_result(path, result: FitResult):
    """Publish parameters and manifest without overwriting an existing result."""
    with publish_directory(path, completion="result.json") as stage:
        _write_result(stage, result)


def _fields(cls, value, *, legacy=False, optional=()):
    declared = fields(cls)
    if (legacy or optional) and isinstance(value, dict):
        defaults = {field.name: field.default for field in declared
                    if field.default is not MISSING and (legacy or field.name in optional)}
        value = defaults | value
    require_keys(value, {field.name for field in declared}, cls.__name__)
    return value


def _decode_reprojection(record, count):
    if record is None:
        return None
    fields_ = _fields(Reprojection, record)
    per_observation = tuple(PixelStatistics(**_fields(PixelStatistics, item))
                            for item in fields_["observations"])
    overall = PixelStatistics(**_fields(PixelStatistics, fields_["overall"]))
    if len(per_observation) != count or overall.eligible_count != sum(
        item.eligible_count for item in per_observation
    ):
        raise ValueError("Reprojection statistics do not match the fitting observations")
    for item, slots in (
        (overall, count * DFL_LANDMARK_COUNT),
        *((item, DFL_LANDMARK_COUNT) for item in per_observation),
    ):
        if (
            type(item.eligible_count) is not int or type(item.excluded_count) is not int
            or item.eligible_count <= 0 or item.excluded_count < 0
            or item.eligible_count + item.excluded_count != slots
            or any(type(value) not in (int, float) or value < 0
                   for value in (item.mean_px, item.rms_px, item.max_px))
        ):
            raise ValueError("Invalid reprojection statistics")
    return Reprojection(overall, per_observation)


def _decode_cameras(records, ids, dtype, config):
    if not isinstance(records, list) or len(records) != len(ids):
        raise ValueError("Saved cameras must match the observation IDs")
    cameras = []
    for identifier, item in zip(ids, records, strict=True):
        require_keys(item, {"id", "camera"}, "Fitting camera")
        if item["id"] != identifier:
            raise ValueError("Saved cameras do not preserve observation order")
        camera = camera_from_record(item["camera"], dtype=dtype, device="cpu")
        if config.camera_model is not None and config.camera_model != item["camera"]["type"]:
            raise ValueError("Saved camera disagrees with the fitting configuration")
        if isinstance(camera, PerspectiveCamera):
            if config.camera_mode == "estimated" and camera.translation[2] <= 0:
                raise ValueError("Estimated cameras require positive perspective translation Z")
        elif config.train_root:
            raise ValueError("Trainable root translation requires fixed perspective cameras")
        cameras.append(camera)
    return tuple(cameras)


def _validate_focal_cameras(cameras, config, dtype, *, initial):
    if any(not torch.equal(camera.focal_length[0], camera.focal_length[1]) for camera in cameras):
        raise ValueError("Estimated focal lengths must be isotropic")
    factors = torch.stack([
        camera.focal_length[0].double() / min(camera.image_size.width, camera.image_size.height)
        for camera in cameras
    ])
    if initial or config.focal_mode == "shared":
        expected = config.initial_focal_factor if initial else factors[0]
        if not torch.allclose(factors, torch.as_tensor(expected, dtype=factors.dtype).expand_as(factors),
                              rtol=8 * torch.finfo(dtype).eps, atol=0):
            raise ValueError("Saved normalized focal lengths disagree with the fitting configuration")


def _decode_optimization(record, config, count, *, legacy):
    if record is None:
        return None
    options = dict(_fields(OptimizationDiagnostics, record, legacy=legacy))
    for name in ("initial_objective", "final_objective"):
        value = options[name]
        if value is not None:
            options[name] = LossValues(**_fields(LossValues, value))
    updates = []
    for item in options["updates"]:
        entry = dict(_fields(OptimizationUpdate, item, legacy=legacy, optional=("focal_factors",)))
        factors = entry["focal_factors"]
        if config.focal_mode != "fixed":
            if not isinstance(factors, list) or len(factors) != count or any(
                type(factor) not in (int, float) or factor <= 0 for factor in factors
            ):
                raise ValueError("Focal trajectory must contain one positive factor per observation")
            entry["focal_factors"] = tuple(factors)
        elif factors is not None:
            raise ValueError("Fixed-focal optimization cannot contain a focal trajectory")
        updates.append(OptimizationUpdate(**entry))
    trials = tuple(Trial(**_fields(Trial, item, legacy=legacy, optional=("focal_step_norms",)))
                   for item in options["trials"])
    linearizations = tuple(
        Linearization(**_fields(Linearization, item, optional=(
            "focal_gradient_norms", "minimum_damping_probe_focal_norms",
        ))) for item in options["linearizations"]
    )
    names = {stage.name for stage in config.stages}
    for count in ("objective_evaluations", "jacobian_evaluations", "accepted_updates", "rejected_trials"):
        if type(options[count]) is not int or options[count] < 0:
            raise ValueError("Invalid optimization evaluation or step count")
    if (
        options["accepted_updates"] != len(updates)
        or options["rejected_trials"] != sum(not trial.accepted for trial in trials)
        or len(linearizations) > options["jacobian_evaluations"]
        or any(item.stage not in names for item in (*updates, *trials, *linearizations))
        or any(type(trial.accepted) is not bool for trial in trials)
        or config.solver == "adam" and (trials or options["jacobian_evaluations"])
        or config.solver == "levenberg-marquardt" and sum(trial.accepted for trial in trials) != len(updates)
    ):
        raise ValueError("Optimization diagnostics disagree with the fitting schedule or solver")
    return OptimizationDiagnostics(**(options | {
        "updates": tuple(updates), "trials": trials, "linearizations": linearizations,
    }))


def load_result(path, *, model) -> FitResult:
    """Read a saved result using a verified model with the same asset and native ordering."""
    validate_model(model)
    try:
        return _load_result(Path(path), model)
    except (TypeError, KeyError, OverflowError) as error:
        raise ValueError(f"Invalid fitting result: {error}") from error


def _load_result(path, model):
    manifest = read_json(path / "result.json")
    # Also rejects finite-looking literals that overflow the JSON decoder's float.
    json_bytes(manifest)
    version = manifest.get("schema_version")
    if type(version) is not int or version not in (1, 2, 3, SCHEMA_VERSION):
        raise ValueError("Unsupported fitting result schema")
    keys = {
        "schema_version", "model", "dtype", "device", "config", "selection", "gauge", "observation_ids",
        "initialization", "initial_cameras", "final_cameras", "initial_reprojection",
        "final_reprojection", "history", "stages", "status", "reason", "runtime",
    }
    if version >= 2:
        keys.add("optimization")
    require_keys(manifest, keys, "Fitting result")
    if manifest["model"] != model.metadata:
        raise ValueError("Incompatible fitting model asset, topology or parameter ordering")
    if manifest["dtype"] not in ("float32", "float64"):
        raise ValueError("Invalid fitting result dtype")
    if not isinstance(manifest["device"], str) or not (
        manifest["device"] == "cpu" or manifest["device"].startswith("cuda:")
        and manifest["device"][5:].isdigit()
    ):
        raise ValueError("Invalid recorded fitting device")
    ids = manifest["observation_ids"]
    if (
        not isinstance(ids, list) or not ids
        or any(not isinstance(identifier, str) or not identifier.strip() for identifier in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Fitting result requires unique observation IDs")

    options = manifest["config"]
    if version == 1 and isinstance(options, dict):
        options = {"solver": "adam"} | options
    config = FitConfig.from_dict(options)
    if version < 4 and config.focal_mode != "fixed":
        raise ValueError("Trainable focal length requires fitting result schema 4")
    if version == 1 and (config.solver != "adam" or "lm" in manifest["config"]):
        raise ValueError("Version 1 fitting results require the original Adam configuration")
    if version >= 2 and not {"solver", "lm"} <= manifest["config"].keys():
        raise ValueError("Fitting results since version 2 require explicit solver configuration")
    selection = resolve_selection(model, config)
    if json_bytes(asdict(selection)) != json_bytes(manifest["selection"]):
        raise ValueError("Saved native parameter selections disagree with the configuration")
    dtype = np.dtype(manifest["dtype"])
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    status = manifest["status"]
    reason = manifest["reason"]
    if status not in STATUSES or not isinstance(reason, str) or not reason.strip():
        raise ValueError("Fitting result requires a valid termination status and reason")
    failed = status in FAILED_STATUSES
    if failed != (manifest["final_cameras"] is None) or failed != (manifest["final_reprojection"] is None):
        raise ValueError("Only a completed fitting run may contain final parameters and statistics")
    if not failed and manifest["initial_reprojection"] is None:
        raise ValueError("Completed fitting requires initial statistics")
    specs = _specs(model.metadata, len(ids), dtype)
    prefixes = ("initial",) if failed else ("initial", "final")
    arrays = read_arrays(path / "parameters.npz", {
        f"{prefix}_{name}": spec for prefix in prefixes for name, spec in specs.items()
    })
    states = {}
    for prefix in prefixes:
        cameras = _decode_cameras(manifest[f"{prefix}_cameras"], ids, torch_dtype, config)
        if config.focal_mode != "fixed":
            _validate_focal_cameras(cameras, config, torch_dtype, initial=prefix == "initial")
        states[prefix] = FitState(*(arrays[f"{prefix}_{name}"] for name in ARRAY_FIELDS), cameras)
    if not failed:
        initial, final = states["initial"], states["final"]
        for start, stop in zip(initial.cameras, final.cameras, strict=True):
            if type(start) is not type(stop) or start.image_size != stop.image_size:
                raise ValueError("Fitting camera model or image size changed")
            if config.camera_mode == "known" and camera_to_record(start) != camera_to_record(stop):
                raise ValueError("Fixed known camera parameters changed")
            if not torch.equal(start.rotation, stop.rotation):
                raise ValueError("Fixed camera rotation changed")
            if isinstance(start, PerspectiveCamera) and (
                (config.focal_mode == "fixed" and not torch.equal(start.focal_length, stop.focal_length))
                or not torch.equal(start.principal_point, stop.principal_point)
            ):
                raise ValueError("Fixed camera intrinsics changed")
        for array, group, names in (
            ("identity", "identity", selection.fixed_identity),
            ("expression", "expression", selection.fixed_expression),
            ("rotations", "joint", selection.fixed_joints),
        ):
            axis = 0 if array == "identity" else 1
            indices = [model.metadata[f"{group}_names"].index(name) for name in names]
            before = getattr(initial, array).take(indices, axis=axis)
            after = getattr(final, array).take(indices, axis=axis)
            if not np.array_equal(before, after):
                raise ValueError(f"Fixed {array} parameters changed")
        if not selection.root_translation and not np.array_equal(initial.translation, final.translation):
            raise ValueError("Fixed root translation changed")

    initialization = tuple(InitializationChoice(**_fields(InitializationChoice, item))
                           for item in manifest["initialization"])
    if len(initialization) != len(ids):
        raise ValueError("Saved initialization choices do not match the observation IDs")
    history = []
    for item in manifest["history"]:
        entry = _fields(Iteration, item).copy()
        entry["losses"] = LossValues(**_fields(LossValues, entry["losses"]))
        entry["reprojection"] = PixelStatistics(**_fields(PixelStatistics, entry["reprojection"]))
        history.append(Iteration(**entry))
    stages = tuple(
        StageSummary(**(_fields(StageSummary, item, legacy=version < 3) | {
            "active_groups": tuple(item["active_groups"]),
        }))
        for item in manifest["stages"]
    )
    if not isinstance(manifest["runtime"], dict):
        raise ValueError("Fitting runtime metadata must be an object")
    result = FitResult(
        deepcopy(model.metadata), dtype.name, manifest["device"], config, selection, tuple(ids),
        initialization, states["initial"], states.get("final"),
        _decode_reprojection(manifest["initial_reprojection"], len(ids)),
        _decode_reprojection(manifest["final_reprojection"], len(ids)),
        tuple(history), stages, status, reason, manifest["runtime"],
        _decode_optimization(manifest.get("optimization"), config, len(ids), legacy=version < 3),
    )
    if json_bytes(result.gauge) != json_bytes(manifest["gauge"]):
        raise ValueError("Saved parameter gauge disagrees with the fitting configuration")
    return result
