"""Generate synthetic GNM benchmark scenarios using known camera parameters."""

import argparse
from collections import defaultdict
from collections.abc import Sequence
import math
from pathlib import Path
import platform

import numpy as np
import torch

from ..camera import PerspectiveCamera, WeakPerspectiveCamera
from ..coordinates import ImageSize
from ..correspondences import DFL_TO_GNM
from ..fitting.parameters import resolve_identity_coefficients
from .configuration import IDENTITY_SELECTIONS, STREAMS, VIEW_AXES, resolve_config, rng, torch_options
from .dataset import (
    SubjectGroundTruth, camera_to_record, observation_id, subject_id, write_dataset,
)
from .model import in_frame, load_benchmark_model, project_dfl
from .noise import perturb
from .storage import read_json


# These are signed directions measured on the pinned GNM basis using the
# official sparse landmarks. See _expression_signals for the anatomical probes.
_SMILE = {"lower_face_region_005": -1.5, "lower_face_region_004": -0.7}
_BROWS = {
    f"{side}_eye_region_{index:03d}": value
    for side in ("left", "right") for index, value in enumerate((1.0, -1.0))
}
EXPRESSIONS = {
    "neutral": {},
    "smile": _SMILE,
    "mouth_open": {"lower_face_region_000": -1.5},
    "raise_brows": _BROWS,
    "eyes_close": dict.fromkeys(_BROWS, -0.95),
    "eyes_open": dict.fromkeys(_BROWS, 0.75),
    "smile_raise_brows": _SMILE | _BROWS,
}
_CAMERA_Y_M = 0.265
_FOCAL_PER_SHORT_EDGE = 1.4
# Far and near intervals are disjoint and visited by each selected projection type.
_TIER_CENTERS = (0.25, 0.75)
_TIER_HALF_WIDTH = 0.125
_ANATOMY_ATOL_M = 1e-5


def _expression_signals(points):
    x, y, _ = points.T
    lip_center = (y[51] + y[57]) / 2
    return {
        "smile": np.array([y[48] - lip_center, y[54] - lip_center, x[54] - x[48]]),
        "mouth_open": np.array([y[62] - y[66]]),
        "raise_brows": np.array([
            y[17:22].mean() - y[[36, 39]].mean(),
            y[22:27].mean() - y[[42, 45]].mean(),
        ]),
        "eyes_open": np.array([
            y[[37, 38]].mean() - y[[40, 41]].mean(),
            y[[43, 44]].mean() - y[[46, 47]].mean(),
        ]),
    }


def _preset_vectors(model, config):
    vectors = {}
    names = model.head.expression_names
    for preset in config.expressions:
        values = np.zeros(model.head.expression_dim, dtype=model.numpy_dtype)
        for name, coefficient in EXPRESSIONS[preset].items():
            if name not in names:
                raise ValueError(f"GNM is missing expression control {name}")
            values[names.index(name)] = coefficient
        vectors[preset] = values
    return vectors


def _verify_expressions(model, config, identity, presets, rotations, translation, subject):
    if len(presets) == 1:
        return
    before = model.landmarks(identity, presets["neutral"], rotations, translation)
    baseline = _expression_signals(before.cpu().numpy())
    checks = {
        "smile": ("smile",), "mouth_open": ("mouth_open",),
        "raise_brows": ("raise_brows",), "eyes_close": ("eyes_open",),
        "eyes_open": ("eyes_open",), "smile_raise_brows": ("smile", "raise_brows"),
    }
    for preset, signals in checks.items():
        sign = -1 if preset == "eyes_close" else 1
        for strength in set(config.expression_strength):
            moved = model.landmarks(identity, presets[preset] * strength, rotations, translation)
            result = _expression_signals(moved.cpu().numpy())
            if any(
                np.any(sign * (result[signal] - baseline[signal]) <= _ANATOMY_ATOL_M)
                for signal in signals
            ) or (preset == "eyes_close" and np.any(result["eyes_open"] <= 0)):
                raise ValueError(f"{subject} does not satisfy the anatomical probe for {preset}")


def _camera(model, config, kind, subject, index):
    size = ImageSize(*config.image_size)
    short_edge = min(size.width, size.height)
    center = ((size.width - 1) / 2, (size.height - 1) / 2)
    occurrence = index // 2 if config.camera == "both" else index
    tier = _TIER_CENTERS[occurrence % len(_TIER_CENTERS)]
    jitter = rng(config.seed, "camera", subject, index).uniform(-_TIER_HALF_WIDTH, _TIER_HALF_WIDTH)
    fraction = tier + jitter
    rotation = model.reference.new_zeros(3)
    if kind == "perspective":
        low, high = config.distance_m
        depth = low + (high - low) * fraction
        schedule = config.perspective_focal_factors
        factor = _FOCAL_PER_SHORT_EDGE if schedule is None else schedule[index]
        return PerspectiveCamera(
            image_size=size, rotation=rotation,
            translation=model.tensor([0, _CAMERA_Y_M, depth]),
            focal_length=model.tensor([factor * short_edge] * 2),
            principal_point=model.tensor(center),
        )
    low, high = config.weak_scale_per_short_edge
    scale = short_edge * (low + (high - low) * fraction)
    return WeakPerspectiveCamera(
        image_size=size, rotation=rotation,
        translation=model.tensor([center[0], center[1] + scale * _CAMERA_Y_M]),
        scale=model.tensor(scale),
    )


def _sample_identity(model, config, subject, sampled):
    # Draw in native order even in head mode to preserve the seeded full-native stream.
    identity = rng(config.seed, "identity", subject).uniform(
        -config.identity_half_range, config.identity_half_range, size=model.head.identity_dim,
    ).astype(model.numpy_dtype)
    identity[~sampled] = 0
    return identity


def _generate_subject(model, config, mapping, presets, subject, sampled, controlled_identity):
    identifier = subject_id(subject)
    identity = controlled_identity
    if identity is None:
        identity = _sample_identity(model, config, subject, sampled)
    if not np.any(identity):
        raise ValueError(f"Identity sampling produced a neutral coefficient vector for {identifier}")
    neutral = model.neutral(identity)
    if np.array_equal(neutral, model.reference.detach().cpu().numpy()):
        raise ValueError(f"Sampled identity does not change the neutral geometry for {identifier}")
    rest = np.zeros((model.head.num_joints, 3), dtype=model.numpy_dtype)
    translation = np.zeros(3, dtype=model.numpy_dtype)
    _verify_expressions(model, config, identity, presets, rest, translation, identifier)
    head_joint = model.head.joint_names.index("head")
    supported = np.array([entry.supported for entry in mapping.entries])
    enabled = np.array([entry.enabled_by_default for entry in mapping.entries])
    weights = np.ones_like(supported, dtype=model.numpy_dtype)
    arrays = defaultdict(list)
    observations = []
    camera_values = defaultdict(set)
    supported_tensor = torch.from_numpy(supported)
    for index in range(config.observations_per_subject):
        view, preset, camera_kind, profile = config.schedule(index)
        identifier_2d = observation_id(identifier, index)
        camera = _camera(model, config, camera_kind, subject, index)
        strength = rng(config.seed, "expression", subject, index).uniform(*config.expression_strength)
        expression = presets[preset] * strength
        rotations = rest.copy()
        axis, _ = VIEW_AXES[view]
        if axis is not None:
            rotations[head_joint, axis] = math.radians(config.view_degrees[view])
        points = model.landmarks(identity, expression, rotations, translation)
        pixels = project_dfl(points, camera, mapping).cpu()
        frame_mask = in_frame(pixels, camera.image_size, supported_tensor).numpy()
        clean = pixels.numpy()
        noise = perturb(
            clean, supported & enabled & frame_mask, mapping, config.noise[profile],
            seed=config.seed, subject=subject, observation=index,
        )
        fields = {
            "expression": expression, "rotations": rotations, "translation": translation,
            "clean": clean, "weights": weights, "in_frame": frame_mask,
            "supported": supported, "projection_valid": supported,
        } | noise
        for name, value in fields.items():
            arrays[name].append(value)
        camera_record = camera_to_record(camera)
        camera_values[camera_kind].add(
            camera_record["translation"][2] if camera_kind == "perspective" else camera_record["scale"]
        )
        observations.append({
            "id": identifier_2d, "view": view, "expression": preset, "noise_profile": profile,
            "camera": camera_record,
        })
    if len(observations) > 1 and any(len(values) < 2 for values in camera_values.values()):
        raise ValueError(f"Camera distances or scales do not vary in the selected dtype for {identifier}")
    subject_arrays = {name: np.stack(values) for name, values in arrays.items()}
    subject_arrays.update(identity=identity, neutral_vertices=neutral)
    return SubjectGroundTruth(identifier, tuple(observations), subject_arrays)


def generate_dataset(destination, config, *, model_file=None, correspondences=DFL_TO_GNM,
                     identity_coefficients=None):
    """Generate subjects, optionally with a controlled sparse identity for one subject."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output already exists: {destination}")
    config = resolve_config(**config.metadata())
    dtype, device = torch_options(config.dtype, config.device)
    model = load_benchmark_model(model_file, dtype=dtype, device=device)
    names = model.metadata["identity_names"]
    selected = (
        resolve_identity_coefficients(model.head, ("head",))
        if config.identity_selection == "head" else tuple(names)
    )
    sampled = np.isin(names, selected)
    identity = None
    provenance = (
        {"identity_coefficient_names": selected} if config.identity_selection is not None else {}
    )
    if identity_coefficients is not None:
        from .configuration import finite_number

        if (
            config.subjects != 1 or not isinstance(identity_coefficients, dict)
            or not identity_coefficients or not identity_coefficients.keys() <= set(names)
        ):
            raise ValueError("A controlled synthetic identity requires one subject and named native coefficients")
        if not identity_coefficients.keys() <= set(selected):
            raise ValueError("Controlled coefficients must belong to the configured identity selection")
        coefficients = {name: finite_number(value, name) for name, value in identity_coefficients.items()}
        if max(map(abs, coefficients.values())) > config.identity_half_range:
            raise ValueError("Controlled coefficients must be within the configured identity range")
        identity = np.asarray([coefficients.get(name, 0.0) for name in names], dtype=model.numpy_dtype)
        provenance["controlled_identity_coefficients"] = coefficients
    presets = _preset_vectors(model, config)
    subjects = (
        _generate_subject(model, config, correspondences, presets, index, sampled, identity)
        for index in range(config.subjects)
    )
    return write_dataset(
        destination, config=config, model=model, mapping=correspondences, subjects=subjects,
        runtime={
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "torch": str(torch.__version__),
            "device": str(model.reference.device),
        },
        generator={
            "rng": "NumPy PCG64 with SeedSequence",
            "rng_streams": STREAMS,
            "expression_base_coefficients": {name: EXPRESSIONS[name] for name in presets},
            "expression_probe_margin_m": _ANATOMY_ATOL_M,
            "view_joint": "head",
            "view_axis": {name: VIEW_AXES[name][0] for name in config.view_degrees},
            "camera_center_y_m": _CAMERA_Y_M,
            "camera_tier_centers": _TIER_CENTERS,
            "camera_tier_half_width": _TIER_HALF_WIDTH,
        } | (
            {"focal_per_short_edge": _FOCAL_PER_SHORT_EDGE} if config.perspective_focal_factors is None
            else {"focal_per_short_edge_schedule": config.perspective_focal_factors}
        ) | provenance,
    )


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON scenario and generation overrides")
    parser.add_argument("--scenario", choices=tuple("ABCDEF"))
    parser.add_argument("--subjects", type=int)
    parser.add_argument("--observations-per-subject", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--camera", choices=("perspective", "weak-perspective", "both"))
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--identity-half-range", type=float)
    parser.add_argument("--identity-selection", choices=IDENTITY_SELECTIONS,
                        help="sample the full native basis or only the native head coefficients")
    parser.add_argument("--gaussian-noise-px", type=float, help="uniform Scenario C Gaussian deviation in pixels")
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    try:
        options = read_json(args.config) if args.config is not None else {}
        options.update({
            name: value for name, value in vars(args).items()
            if name not in ("config", "output", "model_file") and value is not None
        })
        config = resolve_config(**options)
        dataset = generate_dataset(args.output, config, model_file=args.model_file)
    except (OSError, RuntimeError, TypeError, ValueError, OverflowError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(
        f"Scenario {config.scenario}: {config.subjects} subjects, "
        f"{config.subjects * config.observations_per_subject} observations, "
        f"dataset {dataset.dataset_id} -> {args.output}"
    )


if __name__ == "__main__":
    main()
