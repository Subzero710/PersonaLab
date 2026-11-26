"""Objective, step and timing diagnostics shared by the fitting solvers."""

from contextlib import contextmanager
from collections import defaultdict
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
class AcceptedState:
    stage_iteration: int
    objective: float
    damping_after_update: float
    full_identity_l2: float
    trainable_identity_l2: float | None
    focal_factors: tuple[float, ...] | None
    camera_translations: list[list[float]]


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


def finite_norm(*values, dtype=torch.float64):
    norm = math.hypot(*(float(torch.linalg.vector_norm(value.to(dtype))) for value in values))
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


def backend_failure(error):
    """Recognize infrastructure errors without changing solver termination statuses."""
    message = str(error).lower()
    if isinstance(error, (MemoryError, torch.OutOfMemoryError)) or any(
        marker in message for marker in ("out of memory", "can't allocate memory", "cannot allocate memory")
    ):
        return "out_of_memory"
    if isinstance(error, NotImplementedError) or any(
        marker in message for marker in ("not implemented for", "not implemented on", "'cuda' backend")
    ):
        return "unsupported_operation"
    return None


class OperationTimings:
    """Opt-in operation timings. CUDA event pairs are drained after the enclosing phase."""

    def __init__(self, device):
        self.device = device
        self.seconds = defaultdict(float)
        self.counts = defaultdict(int)
        self.events = []
        self.host_instrumentation_seconds = 0.0

    @contextmanager
    def measure(self, name):
        self.counts[name] += 1
        if self.device.type == "cpu":
            started = perf_counter()
            try:
                yield
            finally:
                self.seconds[name] += perf_counter() - started
            return

        setup = perf_counter()
        stream = torch.cuda.current_stream(self.device)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        self.host_instrumentation_seconds += perf_counter() - setup
        try:
            yield
        finally:
            setup = perf_counter()
            stop.record(stream)
            self.events.append((name, start, stop))
            self.host_instrumentation_seconds += perf_counter() - setup

    def drain(self):
        if self.events:
            started = perf_counter()
            # Each pair was recorded on the fitting device's current stream.
            self.events[-1][2].synchronize()
            for name, start, stop in self.events:
                self.seconds[name] += start.elapsed_time(stop) / 1000
            self.events.clear()
            self.host_instrumentation_seconds += perf_counter() - started

    def call(self, name, function, *args, **kwargs):
        with self.measure(name):
            return function(*args, **kwargs)

    def record(self):
        self.drain()
        return {
            "seconds": dict(self.seconds), "calls": dict(self.counts),
            "method": "CPU elapsed wall time" if self.device.type == "cpu" else "CUDA stream event elapsed time",
            "host_instrumentation_seconds": self.host_instrumentation_seconds,
        }


class FitMeasurements:
    def __init__(self, device, *, trace_stages=False, profile_operations=False):
        self.device = device
        self.initial_objective = None
        self.last_objective = None
        self.objective_evaluations = 0
        self.jacobian_evaluations = 0
        self.seconds = dict.fromkeys(("objective", "jacobian", "linear_solve"), 0.0)
        self.updates = []
        self.trials = []
        self.linearizations = []
        self.accepted_states = [] if trace_stages else None
        self.stage_damping = None
        self.operations = OperationTimings(device) if profile_operations else None
        self.normal_system = None
        self.stage_trace_seconds = 0.0

    def trace_accepted_state(self, parameters, count, values):
        if self.accepted_states is None:
            return
        started = perf_counter() if self.operations is not None else None
        update = self.updates[-1]
        dtype = {"float32": torch.float32, "float64": torch.float64}[parameters.config.lm.arithmetic_dtype]
        if dtype == torch.float32:
            identity = parameters.initial.identity.index_copy(0, parameters.identity_indices, parameters.identity)
            initial = torch.stack([camera.translation for camera in parameters.initial.cameras])
            offsets = parameters.camera_delta
            translations = torch.cat((initial[:, :2] + offsets[:, :2] * initial[:, 2:],
                                      initial[:, 2:] * offsets[:, 2:].exp()), dim=1).tolist()
        else:
            state = parameters.state(0, count)
            identity = state.identity
            translations = [camera.translation.tolist() for camera in state.cameras]
        self.accepted_states.append(AcceptedState(
            update.stage_iteration, values.total, self.stage_damping, finite_norm(identity, dtype=dtype),
            finite_norm(parameters.identity, dtype=dtype) if parameters.identity is not None else None,
            update.focal_factors, translations,
        ))
        if started is not None:
            self.stage_trace_seconds += perf_counter() - started

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
