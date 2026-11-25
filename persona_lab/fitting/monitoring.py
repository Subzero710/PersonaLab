"""Objective, step and timing diagnostics shared by the fitting solvers."""

from contextlib import contextmanager
from dataclasses import dataclass
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
        )
