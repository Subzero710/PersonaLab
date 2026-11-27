"""Configuration for one shared-identity fitting session."""

from dataclasses import asdict, dataclass, field
from collections.abc import Mapping

from ..benchmark.configuration import finite_number
from ..correspondences import DFL_LANDMARK_COUNT


GROUPS = ("identity", "expression", "pose", "camera")
CAMERA_MODELS = ("weak-perspective", "perspective")
FOCAL_MODES = ("fixed", "shared", "per-observation")
SOLVERS = ("adam", "levenberg-marquardt")
DEFAULT_SOLVER = "levenberg-marquardt"

# These low-order components cover the mouth and eyelid/brow motions
# independently probed by the existing benchmark, with room for nearby modes.
DEFAULT_EXPRESSION = (
    *(f"lower_face_region_{i:03d}" for i in range(12)),
    *(f"left_eye_region_{i:03d}" for i in range(6)),
    *(f"right_eye_region_{i:03d}" for i in range(6)),
)


def _sequence(value, name):
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} requires a list or tuple")
    return tuple(value)


def _names(value, name):
    items = _sequence(value, name)
    if any(not isinstance(item, str) or not item for item in items) or len(set(items)) != len(items):
        raise ValueError(f"{name} requires unique nonempty names")
    return items


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


@dataclass(frozen=True, slots=True)
class Stage:
    """One fitting stage. Learning rates apply to Adam."""

    name: str
    iterations: int
    groups: tuple[str, ...]
    learning_rate: float
    group_rates: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Stage name must be nonempty")
        _integer(self.iterations, "Stage iterations", 0)
        groups = _names(self.groups, "Stage groups")
        if not groups or not set(groups) <= set(GROUPS):
            raise ValueError(f"Stage groups must select from {', '.join(GROUPS)}")
        object.__setattr__(self, "groups", groups)
        if finite_number(self.learning_rate, "Learning rate") <= 0:
            raise ValueError("Learning rate must be positive")
        pairs = (
            self.group_rates.items() if isinstance(self.group_rates, Mapping)
            else _sequence(self.group_rates, "Group rates")
        )
        rates = []
        seen = set()
        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Group rates must associate a group with a learning rate")
            name, rate = pair
            rate = finite_number(rate, "Group learning rate")
            if name not in groups or name in seen or rate <= 0:
                raise ValueError("Group learning rates require unique stage groups and positive rates")
            rates.append((name, rate))
            seen.add(name)
        object.__setattr__(self, "group_rates", tuple(rates))


def default_stages():
    return (
        Stage("pose_camera", 30, ("pose", "camera"), 0.018),
        Stage("identity", 70, ("identity", "pose", "camera"), 0.045,
              (("pose", 0.008), ("camera", 0.006))),
        Stage("joint", 100, GROUPS, 0.022,
              (("pose", 0.005), ("camera", 0.004))),
    )


@dataclass(frozen=True, slots=True)
class LevenbergMarquardtConfig:
    """Damping, scaling, convergence and Jacobian controls shared by all stages."""

    initial_damping: float = 1e-3
    minimum_damping: float = 1e-8
    maximum_damping: float = 1e8
    max_trials: int = 12
    acceptance_threshold: float = 1e-4
    diagonal_floor: float = 1e-12
    scaled_gradient_tolerance: float = 1e-10
    jacobian_rows: int = 8
    native_step_tolerance: float | None = None
    scaled_step_tolerance: float | None = None
    arithmetic_dtype: str = "float64"
    fp32_forward: str = "projection"
    fp64_square_root: bool = False

    @property
    def uses_square_root(self):
        return self.arithmetic_dtype == "float32" or self.fp64_square_root

    def __post_init__(self):
        if type(self.fp64_square_root) is not bool or (self.fp64_square_root and (
                self.arithmetic_dtype != "float64" or self.fp32_forward != "projection")):
            raise ValueError("FP64 square-root requires explicit float64 arithmetic and native geometry")
        if self.fp32_forward not in ("projection", "compensated") or (
                self.fp32_forward == "compensated" and self.arithmetic_dtype != "float32"):
            raise ValueError("Compensated geometry requires an explicit FP32 LM configuration")
        if self.arithmetic_dtype not in ("float32", "float64"):
            raise ValueError("LM arithmetic_dtype must be float32 or float64")
        for name in ("max_trials", "jacobian_rows"):
            _integer(getattr(self, name), name, 1)
        for name in (
            "initial_damping", "minimum_damping", "maximum_damping",
            "acceptance_threshold", "diagonal_floor", "scaled_gradient_tolerance",
        ):
            object.__setattr__(self, name, finite_number(getattr(self, name), name))
        if not 0 < self.minimum_damping <= self.initial_damping <= self.maximum_damping:
            raise ValueError("LM damping must satisfy 0 < minimum <= initial <= maximum")
        if not 0 <= self.acceptance_threshold < 1:
            raise ValueError("LM acceptance_threshold must be in [0, 1)")
        if not 0 < self.diagonal_floor <= 1 or self.scaled_gradient_tolerance < 0:
            raise ValueError("LM diagonal_floor must be in (0, 1] and gradient tolerance nonnegative")
        for name in ("native_step_tolerance", "scaled_step_tolerance"):
            value = getattr(self, name)
            if value is not None:
                value = finite_number(value, name)
                if value < 0:
                    raise ValueError(f"{name} must be nonnegative")
                object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Explicit optimization controls. Empty selections fix the entire group."""

    camera_mode: str = "known"
    camera_model: str | None = None
    train_identity: tuple[str, ...] = ("head",)
    fixed_identity: tuple[str, ...] = ()
    train_expression: tuple[str, ...] = DEFAULT_EXPRESSION
    fixed_expression: tuple[str, ...] = ()
    train_head: bool = True
    train_root: bool = False
    landmark_subset: tuple[int, ...] | None = None
    expression_limit: float = 3.0
    identity_weight: float = 1e-6
    expression_weight: float = 2e-5
    pose_weight: float = 0.0
    camera_weight: float = 0.0
    perspective_focal_per_short_edge: float = 1.5
    pose_initializer: str = "grid"
    batch_size: int = 4
    seed: int = 0
    log_every: int = 10
    improvement_tolerance: float | None = None
    gradient_tolerance: float | None = None
    patience: int = 10
    divergence_factor: float = 100.0
    stages: tuple[Stage, ...] = field(default_factory=default_stages)
    solver: str = DEFAULT_SOLVER
    lm: LevenbergMarquardtConfig = field(default_factory=LevenbergMarquardtConfig)
    focal_mode: str = "fixed"
    initial_focal_factor: float | None = None

    def __post_init__(self):
        if self.solver not in SOLVERS or not isinstance(self.lm, LevenbergMarquardtConfig):
            raise ValueError("Select a supported solver and a LevenbergMarquardtConfig")
        if self.lm.fp64_square_root and (
                (self.camera_mode, self.camera_model, self.focal_mode, self.solver) !=
                ("estimated", "perspective", "per-observation", "levenberg-marquardt")
                or self.train_root or not self.train_head
                or any(getattr(self, f"{g}_weight") != 0 for g in GROUPS)):
            raise ValueError("FP64 square-root requires explicit joint zero-prior estimated perspective fitting")
        if self.lm.fp32_forward == "compensated" and (
                (self.camera_mode, self.camera_model, self.focal_mode, self.solver) !=
                ("estimated", "perspective", "per-observation", "levenberg-marquardt")
                or self.train_root or not self.train_head
                or any(getattr(self, f"{g}_weight") != 0 for g in GROUPS)):
            raise ValueError("Compensated FP32 requires joint zero-prior estimated perspective fitting")
        if self.camera_mode not in ("known", "estimated"):
            raise ValueError("Camera mode must be known or estimated")
        if self.camera_model is not None and self.camera_model not in CAMERA_MODELS:
            raise ValueError("Camera model must be weak-perspective or perspective")
        if self.camera_mode == "known" and self.camera_model is not None:
            raise ValueError("Known-camera mode uses the camera type of each observation")
        if self.focal_mode not in FOCAL_MODES:
            raise ValueError(f"Focal mode must be one of {', '.join(FOCAL_MODES)}")
        if self.focal_mode == "fixed":
            if self.initial_focal_factor is not None:
                raise ValueError("initial_focal_factor requires trainable focal length")
        else:
            if (self.camera_mode, self.camera_model, self.solver) != (
                "estimated", "perspective", "levenberg-marquardt",
            ):
                raise ValueError("Trainable focal length requires estimated perspective cameras and LM")
            factor = finite_number(self.initial_focal_factor, "Explicit initial focal factor")
            if factor <= 0:
                raise ValueError("Initial focal factor must be positive")
            object.__setattr__(self, "initial_focal_factor", factor)
        if self.pose_initializer not in ("grid", "neutral"):
            raise ValueError("Pose initializer must be grid or neutral")
        for name in ("train_head", "train_root"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.train_root and self.camera_mode != "known":
            raise ValueError("Trainable root translation requires fixed perspective cameras")

        for name in ("train_identity", "fixed_identity", "train_expression", "fixed_expression"):
            object.__setattr__(self, name, _names(getattr(self, name), name))
        if self.landmark_subset is not None:
            subset = _sequence(self.landmark_subset, "Landmark subset")
            if any(type(i) is not int or not 0 <= i < DFL_LANDMARK_COUNT for i in subset):
                raise ValueError("Landmark subset indices must be integers in [0, 67]")
            if not subset or len(set(subset)) != len(subset):
                raise ValueError("Landmark subset must contain at least one index without duplicates")
            object.__setattr__(self, "landmark_subset", subset)

        for name in ("batch_size", "log_every", "patience"):
            _integer(getattr(self, name), name, 1)
        _integer(self.seed, "Seed", 0)
        if self.seed >= 2**64:
            raise ValueError("Seed must be smaller than 2**64")
        for name in ("identity_weight", "expression_weight", "pose_weight", "camera_weight",
                     "improvement_tolerance", "gradient_tolerance"):
            value = getattr(self, name)
            if name.endswith("_tolerance") and value is None:
                continue
            if finite_number(value, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("expression_limit", "perspective_focal_per_short_edge"):
            if finite_number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if finite_number(self.divergence_factor, "Divergence factor") <= 1:
            raise ValueError("Divergence factor must be greater than one")

        stages = _sequence(self.stages, "Stages")
        if (
            not stages or any(not isinstance(stage, Stage) for stage in stages)
            or not any(stage.iterations for stage in stages)
            or len({stage.name for stage in stages}) != len(stages)
        ):
            raise ValueError("Stages require unique names and a positive iteration budget")
        if self.lm.fp64_square_root and any(stage.iterations and set(stage.groups) != set(GROUPS) for stage in stages):
            raise ValueError("FP64 square-root requires joint stages with all four parameter groups")
        object.__setattr__(self, "stages", stages)

    def to_dict(self):
        """Keep the established fixed-focal configuration representation."""
        options = asdict(self)
        if self.lm.fp32_forward == "projection":
            del options["lm"]["fp32_forward"]
        if not self.lm.fp64_square_root:
            del options["lm"]["fp64_square_root"]
        if self.lm.arithmetic_dtype == "float64" and not self.lm.fp64_square_root:
            del options["lm"]["arithmetic_dtype"]
        if self.focal_mode == "fixed":
            del options["focal_mode"], options["initial_focal_factor"]
        return options

    @classmethod
    def from_dict(cls, options):
        """Parse the same configuration recorded in the result manifest."""
        if not isinstance(options, dict):
            raise ValueError("Fitting configuration must be a JSON object")
        values = dict(options)
        try:
            unknown = values.keys() - cls.__dataclass_fields__.keys()
            if unknown:
                raise ValueError(f"Unknown fitting options: {', '.join(sorted(unknown))}")
            if "stages" in values:
                stages = _sequence(values["stages"], "Stages")
                if any(not isinstance(item, dict) for item in stages):
                    raise ValueError("Each stage configuration must be an object")
                values["stages"] = tuple(Stage(**item) for item in stages)
            if "lm" in values:
                if not isinstance(values["lm"], dict):
                    raise ValueError("LM configuration must be an object")
                values["lm"] = LevenbergMarquardtConfig(**values["lm"])
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid fitting configuration: {error}") from error
