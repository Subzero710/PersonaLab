"""Objective, step and timing diagnostics shared by the fitting solvers."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import sys
from time import perf_counter

import torch

from .losses import LossValues, evaluate_objective


@dataclass(frozen=True, slots=True)
class OptimizationUpdate:
    stage: str
    stage_iteration: int
    parameter_update_norm: float
    scaled_parameter_update_norm: float | None = None
    focal_factors: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class Trial:
    stage: str
    stage_iteration: int
    trial: int
    damping: float
    accepted: bool
    reason: str
    scaled_gradient_norm: float
    parameter_update_norm: float | None
    predicted_reduction: float | None
    actual_reduction: float | None
    acceptance_ratio: float | None
    conditioning: dict[str, float | None]
    scaled_parameter_update_norm: float | None = None
    focal_step_norms: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class Linearization:
    stage: str
    stage_iteration: int
    objective: float
    damping: float
    projected_gradient_norm: float
    scaled_gradient_norm: float
    active_expression_bounds: int
    minimum_damping_probe_native_norm: float | None
    minimum_damping_probe_scaled_norm: float | None
    focal_gradient_norms: dict[str, float] | None = None
    minimum_damping_probe_focal_norms: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class OptimizationDiagnostics:
    initial_objective: LossValues | None
    final_objective: LossValues | None
    objective_evaluations: int
    jacobian_evaluations: int
    accepted_updates: int
    rejected_trials: int
    updates: tuple[OptimizationUpdate, ...]
    trials: tuple[Trial, ...]
    linearizations: tuple[Linearization, ...] = ()


def optimization_record(diagnostics):
    """Serialize focal diagnostics when applicable, preserving fixed-focal records."""
    record = asdict(diagnostics)
    for section, names in (
        ("updates", ("focal_factors",)), ("trials", ("focal_step_norms",)),
        ("linearizations", ("focal_gradient_norms", "minimum_damping_probe_focal_norms")),
    ):
        for item in record[section]:
            for name in names:
                if item[name] is None:
                    del item[name]
    return record


def finite_norm(*values):
    norm = math.hypot(*(float(torch.linalg.vector_norm(value.double())) for value in values))
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite parameter or gradient norm")
    return norm


def peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


class FitMeasurements:
    def __init__(self, device):
        self.device = device
        self.initial_objective = None
        self.last_objective = None
        self.objective_evaluations = 0
        self.jacobian_evaluations = 0
        self.seconds = dict.fromkeys(("objective", "jacobian", "linear_solve"), 0.0)
        self.updates = []
        self.trials = []
        self.linearizations = []

    @contextmanager
    def time(self, phase):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.seconds[phase] += perf_counter() - started

    def evaluate(self, model, data, parameters, active=(), **options):
        self.objective_evaluations += 1
        with self.time("objective"):
            return evaluate_objective(model, data, parameters, active, **options)

    def diagnostics(self):
        return OptimizationDiagnostics(
            self.initial_objective, self.last_objective, self.objective_evaluations,
            self.jacobian_evaluations, len(self.updates),
            sum(not trial.accepted for trial in self.trials), tuple(self.updates), tuple(self.trials),
            tuple(self.linearizations),
        )
