"""Scenario definitions and reproducible, independently keyed random streams."""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from ..coordinates import ImageSize
from ..correspondences import DFL_LANDMARK_COUNT, Region


VIEW_AXES = {
    "frontal": (None, 0),
    "three_quarter_left": (1, -35),
    "three_quarter_right": (1, 35),
    "profile_left": (1, -70),
    "profile_right": (1, 70),
    "pitch_up": (0, -18),
    "pitch_down": (0, 18),
    "roll_positive": (2, 15),
    "roll_negative": (2, -15),
}
EXPRESSION_NAMES = (
    "neutral", "smile", "mouth_open", "raise_brows", "eyes_close",
    "eyes_open", "smile_raise_brows",
)
SCENARIO_NOISE = {
    "A": ("clean",),
    "B": ("clean",),
    "C": ("clean",),
    "D": ("small", "moderate"),
    "E": ("structured",),
    "F": ("clean", "small", "moderate", "structured", "combined"),
}
STREAMS = {
    "identity": 0, "expression": 1, "camera": 2, "gaussian": 3,
    "missing": 4, "outlier_indices": 5, "outlier_offsets": 6,
}
IDENTITY_SELECTIONS = ("full", "head")


def rng(seed, stream, subject, observation=0):
    """Use PCG64 and a SeedSequence keyed by indices, independent of call order."""
    key = (STREAMS[stream], subject, observation)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=key)))


def _parse_device(device):
    try:
        device = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError("Device must be cpu, cuda, or cuda:INDEX") from error
    if device.type not in ("cpu", "cuda") or (device.type == "cpu" and device.index is not None):
        raise ValueError("Device must be cpu, cuda, or cuda:INDEX")
    return device


def torch_options(dtype, device):
    if dtype not in ("float32", "float64", torch.float32, torch.float64):
        raise ValueError("Expected float32 or float64")
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(dtype, dtype)
    device = _parse_device(device)
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError(f"Requested device {device} is unavailable")
    return dtype, device


def finite_number(value, name):
    if type(value) in (int, float):
        try:
            if math.isfinite(value):
                return float(value)
        except OverflowError:
            pass
    raise ValueError(f"{name} must be a finite number")


def interval(value, name, *, allow_fixed=False):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} requires [lower, upper]")
    lower, upper = (finite_number(item, name) for item in value)
    if lower <= 0 or upper < lower or (lower == upper and not allow_fixed):
        raise ValueError(f"{name} requires positive ordered bounds")
    return lower, upper


@dataclass(frozen=True, slots=True)
class Noise:
    gaussian_px: float = 0
    missing: int = 0
    outliers: int = 0
    outlier_radius_px: tuple[float, float] = (8, 18)
    occlude: tuple[str, ...] = ()

    def __post_init__(self):
        if finite_number(self.gaussian_px, "Gaussian standard deviation") < 0:
            raise ValueError("Gaussian standard deviation must be nonnegative")
        if any(type(n) is not int or n < 0 for n in (self.missing, self.outliers)):
            raise ValueError("Missing and outlier counts must be nonnegative integers")
        if self.missing + self.outliers > DFL_LANDMARK_COUNT:
            raise ValueError("Missing and outlier counts exceed 68 slots")
        bounds = interval(self.outlier_radius_px, "Outlier radius", allow_fixed=True)
        object.__setattr__(self, "outlier_radius_px", bounds)
        if not isinstance(self.occlude, (list, tuple)):
            raise ValueError("Occlusion must be a list of anatomical regions")
        regions = tuple(Region(region).value for region in self.occlude)
        if len(set(regions)) != len(regions):
            raise ValueError("Occlusion regions must be unique")
        object.__setattr__(self, "occlude", regions)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    scenario: str
    subjects: int
    observations_per_subject: int
    seed: int
    camera: str
    image_size: tuple[int, int]
    identity_half_range: float
    expression_strength: tuple[float, float]
    expressions: tuple[str, ...]
    view_degrees: dict[str, float]
    distance_m: tuple[float, float]
    weak_scale_per_short_edge: tuple[float, float]
    noise: dict[str, Noise]
    dtype: str
    device: str
    identity_selection: str | None = None
    gaussian_noise_px: float | None = None
    perspective_focal_factors: tuple[float, ...] | None = None

    def metadata(self):
        metadata = asdict(self)
        # Omission denotes the original full-native protocol and retains its dataset ID.
        for name in ("identity_selection", "gaussian_noise_px", "perspective_focal_factors"):
            if metadata[name] is None:
                del metadata[name]
        return metadata

    def schedule(self, index):
        """Return view, expression, camera type and noise profile for one index."""
        views = tuple(name for name in VIEW_AXES if name in self.view_degrees)
        view = views[index % len(views)]
        expression = self.expressions[(index + index // len(views)) % len(self.expressions)]
        camera = self.camera
        if camera == "both":
            camera = ("perspective", "weak-perspective")[index % 2]
        profiles = SCENARIO_NOISE[self.scenario] if self.gaussian_noise_px is None else ("gaussian",)
        return view, expression, camera, profiles[index % len(profiles)]


def _resolve_views(value, scenario):
    names = ("frontal",) if scenario == "A" else tuple(VIEW_AXES)
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"Scenario {scenario} requires these views: {', '.join(names)}")
    angles = {}
    for name in names:
        degrees = finite_number(value[name], name)
        reference = VIEW_AXES[name][1]
        if abs(degrees) >= 90 or (
            degrees != 0 if reference == 0 else degrees * reference <= 0
        ):
            raise ValueError(f"Invalid signed angle for {name}")
        angles[name] = degrees
    if scenario != "A":
        for side in ("left", "right"):
            quarter = abs(angles[f"three_quarter_{side}"])
            profile = abs(angles[f"profile_{side}"])
            if not 20 <= quarter <= 55 or not 60 <= profile < 90:
                raise ValueError("Three-quarter yaw requires 20–55° and profile yaw 60–<90°")
        for name in ("pitch_up", "pitch_down", "roll_positive", "roll_negative"):
            if not 5 <= abs(angles[name]) <= 35:
                raise ValueError(f"{name} requires a magnitude from 5 through 35 degrees")
    return angles


def _resolve_noise(value, scenario, gaussian_noise_px=None):
    profiles = SCENARIO_NOISE[scenario] if gaussian_noise_px is None else ("gaussian",)
    if not isinstance(value, dict) or set(value) != set(profiles):
        raise ValueError(f"Scenario {scenario} requires these noise profiles: {', '.join(profiles)}")
    noise = {}
    for name, options in value.items():
        if isinstance(options, Noise):
            noise[name] = options
        elif isinstance(options, dict):
            try:
                noise[name] = Noise(**options)
            except TypeError as error:
                raise ValueError(f"Unknown fields in the {name} noise profile") from error
        else:
            raise ValueError(f"The {name} noise profile must be an object")
    if gaussian_noise_px is not None and noise["gaussian"] != Noise(gaussian_px=gaussian_noise_px):
        raise ValueError("The Gaussian profile must match the configured deviation and use only Gaussian noise")
    if "clean" in noise:
        clean = noise["clean"]
        if clean.gaussian_px or clean.missing or clean.outliers or clean.occlude:
            raise ValueError("The clean profile must leave every eligible point unchanged")
    if "small" in noise:
        if not 0 < noise["small"].gaussian_px < noise["moderate"].gaussian_px:
            raise ValueError("Gaussian profiles require 0 < small deviation < moderate deviation")
        for name in ("small", "moderate"):
            profile = noise[name]
            if profile.missing or profile.outliers or profile.occlude:
                raise ValueError(f"The {name} profile allows only Gaussian noise")
    if "structured" in noise:
        profile = noise["structured"]
        if profile.gaussian_px or not profile.missing or not profile.outliers:
            raise ValueError("Structured noise requires missing points and outliers and zero Gaussian noise")
    if "combined" in noise:
        profile = noise["combined"]
        if not (profile.gaussian_px and profile.missing and profile.outliers and profile.occlude):
            raise ValueError("Combined noise requires Gaussian noise, missing points, outliers and occlusion")
    return noise


def resolve_config(scenario="F", **overrides):
    """Resolve scenario defaults and reject options that would omit coverage."""
    if not isinstance(scenario, str) or scenario not in SCENARIO_NOISE:
        raise ValueError("Scenario must be A, B, C, D, E, or F")
    single_view = scenario == "A"
    profiles = {
        "clean": Noise(),
        "small": Noise(gaussian_px=0.5),
        "moderate": Noise(gaussian_px=2.0),
        "structured": Noise(missing=4, outliers=3),
        "combined": Noise(gaussian_px=2.0, missing=4, outliers=3, occlude=("eyebrow",)),
    }
    required_expressions = ("neutral",) if scenario in ("A", "B") else EXPRESSION_NAMES
    defaults = {
        "subjects": 5 if scenario == "F" else 1,
        "observations_per_subject": 1 if single_view else len(VIEW_AXES),
        "seed": 42,
        "camera": "perspective" if single_view else "both",
        "image_size": (512, 512),
        "identity_half_range": 0.5,
        "identity_selection": None,
        "gaussian_noise_px": None,
        "perspective_focal_factors": None,
        "expression_strength": (0.8, 1.0),
        "expressions": required_expressions,
        "view_degrees": {
            name: angle for name, (_, angle) in VIEW_AXES.items()
            if not single_view or name == "frontal"
        },
        "distance_m": (0.60, 0.80),
        "weak_scale_per_short_edge": (2.0, 2.4),
        "noise": {name: profiles[name] for name in SCENARIO_NOISE[scenario]},
        "dtype": "float64",
        "device": "cpu",
    }
    if unknown := sorted(overrides.keys() - defaults.keys()):
        raise ValueError(f"Unknown generation options: {', '.join(unknown)}")
    values = defaults | overrides
    if "identity_selection" in overrides and values["identity_selection"] not in IDENTITY_SELECTIONS:
        raise ValueError("Identity selection must be full or head")
    subjects, observations, seed = (values[name] for name in (
        "subjects", "observations_per_subject", "seed",
    ))
    if type(subjects) is not int or not 1 <= subjects < 2**32 or (
        subjects < 2 if scenario == "F" else subjects != 1
    ):
        raise ValueError("Scenario F requires multiple subjects. Scenarios A–E require one.")
    if type(observations) is not int or not 1 <= observations < 2**32 or (
        observations != 1 if single_view else observations < len(VIEW_AXES)
    ):
        raise ValueError("Scenario A requires one observation. Scenarios B–F require at least nine.")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Seed must be an integer in [0, 2**64)")
    camera = values["camera"]
    if camera not in ("perspective", "weak-perspective", "both") or (
        single_view and camera == "both"
    ):
        raise ValueError("Select perspective, weak-perspective, or both for a multi-view scenario")
    if "perspective_focal_factors" in overrides:
        schedule = values["perspective_focal_factors"]
        if camera != "perspective" or not isinstance(schedule, (list, tuple)) or len(schedule) != observations:
            raise ValueError("A focal schedule requires one factor per observation and perspective cameras")
        factors = tuple(finite_number(factor, "Perspective focal factor") for factor in schedule)
        if any(factor <= 0 for factor in factors):
            raise ValueError("Perspective focal factors must be positive")
        values["perspective_focal_factors"] = factors
    if not isinstance(values["image_size"], (tuple, list)) or len(values["image_size"]) != 2:
        raise ValueError("Image size requires [width, height]")
    ImageSize(*values["image_size"])
    values["image_size"] = tuple(values["image_size"])
    half_range = finite_number(values["identity_half_range"], "Identity half-range")
    if not 0 < half_range <= 3:
        raise ValueError("Identity half-range must be within (0, 3], the documented typical range")
    values["identity_half_range"] = half_range
    strength = interval(values["expression_strength"], "Expression strength", allow_fixed=True)
    if strength[1] > 1:
        raise ValueError("Expression strength cannot exceed the defined presets")
    values["expression_strength"] = strength
    expressions = values["expressions"]
    if not isinstance(expressions, (tuple, list)) or len(expressions) != len(required_expressions) or (
        any(name not in expressions for name in required_expressions)
    ):
        raise ValueError(f"Scenario {scenario} requires these expressions: {', '.join(required_expressions)}")
    values["expressions"] = tuple(expressions)
    values["view_degrees"] = _resolve_views(values["view_degrees"], scenario)
    for name in ("distance_m", "weak_scale_per_short_edge"):
        values[name] = interval(values[name], name)
    deviation = None
    if "gaussian_noise_px" in overrides:
        deviation = finite_number(values["gaussian_noise_px"], "Gaussian noise deviation (px)")
        if scenario != "C" or deviation <= 0:
            raise ValueError("Uniform Gaussian noise requires Scenario C and a positive deviation")
        values["gaussian_noise_px"] = deviation
        if "noise" not in overrides:
            values["noise"] = {"gaussian": Noise(gaussian_px=deviation)}
    values["noise"] = _resolve_noise(values["noise"], scenario, deviation)
    if not isinstance(values["dtype"], str) or values["dtype"] not in ("float32", "float64"):
        raise ValueError("Generation dtype must be float32 or float64")
    if not isinstance(values["device"], str):
        raise ValueError("Generation device must be a string")
    _parse_device(values["device"])
    return GenerationConfig(scenario=scenario, **values)
