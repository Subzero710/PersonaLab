"""Staged fitting of one native GNM identity across multiple observations."""

from copy import deepcopy
import math
import platform
from time import perf_counter

import numpy as np
import torch

from .configuration import FitConfig
from .initialization import initialize
from .levenberg_marquardt import fit_stage as fit_lm_stage
from .losses import evaluate_objective
from .monitoring import FitMeasurements, OptimizationUpdate, backend_failure, finite_norm, peak_rss_bytes
from .observations import prepare_observations
from .parameters import PARAMETER_GROUPS, FitInitialization, FittingParameters, resolve_selection, validate_model
from .results import FAILED_STATUSES, FitResult, FitState, Iteration, StageBoundary, StageDiagnostics, StageSummary


def _gradient_norms(active):
    norms = {}
    for name, parameter in active:
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Missing or nonfinite gradient for {name}")
        norm = float(torch.linalg.vector_norm(gradient.detach().to(dtype=torch.float64)))
        if not math.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient norm for {name}")
        norms[name] = math.hypot(norms.get(name, 0.0), norm)
    return norms


def _validate_step(active, optimizer):
    for name, parameter in active:
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Nonfinite parameters after Adam step for {name}")
        if any(not torch.isfinite(state).all() for state in optimizer.state[parameter].values()):
            raise FloatingPointError(f"Nonfinite Adam state for {name}")


def _check_divergence(loss, start, factor):
    if loss > factor * max(start, 1e-12):
        raise FloatingPointError(f"Objective diverged from {start:.6g} to {loss:.6g}")


def _fit_adam_stage(model, data, parameters, stage, names, values, run, record):
    config = parameters.config
    groups = parameters.groups()
    active = tuple((name, p) for name in names for p in groups[name])
    rates = dict(stage.group_rates)
    optimizer = torch.optim.Adam([
        {"params": groups[name], "lr": rates.get(name, stage.learning_rate)}
        for name in names
    ])
    start_loss = values.total
    best, stale = start_loss, 0
    status, reason = "budget_exhausted", "iteration budget reached"
    for step in range(stage.iterations):
        optimizer.zero_grad(set_to_none=True)
        values, stats = run.evaluate(model, data, parameters, active)
        _check_divergence(values.total, start_loss, config.divergence_factor)
        run.last_objective = values
        norms = _gradient_norms(active)
        if step == 0 or len(run.updates) % config.log_every == 0:
            record(stage.name, step, values, stats, norms)

        if config.gradient_tolerance is not None and math.hypot(*norms.values()) <= config.gradient_tolerance:
            status, reason = "converged", "gradient tolerance reached"
            break

        if config.improvement_tolerance is not None and step:
            if best - values.total > config.improvement_tolerance:
                best, stale = values.total, 0
            else:
                stale += 1
            if stale >= config.patience:
                status, reason = "stopped", "objective improvement below tolerance"
                break

        before = tuple(parameter.detach().clone() for _, parameter in active)
        optimizer.step()
        _validate_step(active, optimizer)
        with torch.no_grad():
            if parameters.expression is not None and "expression" in names:
                parameters.expression.clamp_(-config.expression_limit, config.expression_limit)
            norm = finite_norm(*(
                parameter.double() - previous.double()
                for (_, parameter), previous in zip(active, before, strict=True)
            ))
        run.updates.append(OptimizationUpdate(stage.name, step + 1, norm))

    values, stats = run.evaluate(model, data, parameters)
    _check_divergence(values.total, start_loss, config.divergence_factor)
    run.last_objective = values
    return values, stats, status, reason


def _stage_boundary(model, data, parameters):
    """Evaluate the full LM objective in the live coordinates without affecting the solver trace."""
    state = None
    coordinates = {
        name: value.detach().tolist() if (value := getattr(parameters, name)) is not None else None
        for name in PARAMETER_GROUPS
    }
    try:
        with torch.no_grad():
            state = FitState.from_tensors(parameters.state(0, len(data)))
        values, _ = evaluate_objective(model, data, parameters, least_squares=True)
    except (ArithmeticError, RuntimeError, ValueError) as error:
        return StageBoundary(state, coordinates, None, str(error))
    return StageBoundary(state, coordinates, values, None)


def fit_identity(model, observations, config=None, *, initial=None, progress=None,
                 trace_stages=False, profile=None, correspondences=None) -> FitResult:
    """Fit a single subject using only observations, a verified model and explicit inputs.

    Configuration and initialization errors raise ValueError. Numerical failures
    during optimization return a failed result with no final parameters.
    Opt-in LM stage diagnostics are kept in memory for benchmark reporting.
    An optional benchmark profile is notified at the initialization and optimization boundaries.
    """
    config = FitConfig() if config is None else config
    initial = FitInitialization() if initial is None else initial
    if not isinstance(config, FitConfig):
        raise ValueError("Expected a FitConfig")
    if torch.is_inference_mode_enabled():
        raise ValueError("Identity fitting requires PyTorch autograd outside inference mode")
    validate_model(model)
    use_adam = config.solver == "adam"
    if config.lm.arithmetic_dtype == "float32" and (use_adam or model.reference.dtype != torch.float32):
        raise ValueError("FP32 LM arithmetic requires an FP32 model and Levenberg-Marquardt")
    if config.lm.fp64_square_root and (use_adam or model.reference.dtype != torch.float64):
        raise ValueError("FP64 square-root requires an FP64 model and Levenberg-Marquardt")
    if trace_stages and use_adam:
        raise ValueError("Stage tracing requires the Levenberg–Marquardt solver")
    device = model.reference.device
    if profile is not None:
        profile.start("initialization")
    started = perf_counter()
    data = prepare_observations(observations, model.reference, config, model=model,
                                correspondences=correspondences)
    selection = resolve_selection(model, config)
    geometry = None
    if data.surface is not None:
        if len(selection.identity) == model.head.identity_dim or len(selection.expression) == model.head.expression_dim:
            raise ValueError("Dense fitting requires a restricted native parameter selection")
        geometry = data.surface.sparse_geometry(model)
    initial_state, choices = initialize(model, data, config, initial, geometry=geometry)
    parameters = FittingParameters(model, data, initial_state, selection, config, geometry=geometry)
    groups = parameters.groups()
    if not any(stage.iterations and any(groups[name] for name in stage.groups)
               for stage in config.stages):
        raise ValueError("No stage has an available trainable parameter group")
    initial_snapshot = FitState.from_tensors(initial_state)
    initialization_seconds = perf_counter() - started
    if profile is not None:
        profile.start("optimization")

    history, stages = [], []
    run = FitMeasurements(device, trace_stages=trace_stages, profile_operations=profile is not None)
    stage_diagnostics = []
    initial_stats, final, final_stats = None, None, None
    status, reason = "budget_exhausted", "Iteration budget reached"
    stage_name = "initial evaluation"
    running_stage = None
    trace_start = None
    failure = None

    def traced_boundary():
        trace_started = perf_counter()
        boundary = _stage_boundary(model, data, parameters)
        run.stage_trace_seconds += perf_counter() - trace_started
        return boundary

    def record(stage, stage_iteration, losses, stats, norms):
        log = Iteration(len(run.updates), stage, stage_iteration, losses, stats.overall,
                        norms, perf_counter() - started)
        history.append(log)
        if progress is not None:
            progress(log)

    def finish_trace(stage):
        if trace_start is None:
            return
        boundary, objectives, jacobians, trials, updates = trace_start
        stage_diagnostics.append(StageDiagnostics(
            stage.name, boundary, traced_boundary(),
            run.objective_evaluations - objectives, run.jacobian_evaluations - jacobians,
            sum(not trial.accepted for trial in run.trials[trials:]),
            config.lm.initial_damping if run.stage_damping is not None else None, run.stage_damping,
            tuple(run.accepted_states[updates:]),
        ))

    try:
        values, initial_stats = run.evaluate(model, data, parameters, least_squares=not use_adam)
        run.initial_objective = run.last_objective = values
        current_stats = initial_stats
        for stage in config.stages:
            stage_name = stage.name
            for name, group in groups.items():
                for parameter in group:
                    parameter.requires_grad_(use_adam and name in stage.groups)
                    parameter.grad = None
            names = tuple(name for name in stage.groups if groups[name])
            start_loss = values.total
            if not stage.iterations or not names:
                skip_reason = "zero iteration budget" if not stage.iterations else "no active parameter groups"
                stages.append(StageSummary(stage.name, names, 0, "skipped", start_loss, start_loss, skip_reason))
                continue

            before = len(run.updates)
            running_stage = (stage, names, before, start_loss)
            if trace_stages:
                run.stage_damping = None
                trace_start = (
                    traced_boundary(), run.objective_evaluations,
                    run.jacobian_evaluations, len(run.trials), before,
                )
            if use_adam:
                values, current_stats, status, reason = _fit_adam_stage(
                    model, data, parameters, stage, names, values, run, record,
                )
            else:
                values, current_stats, status, reason = fit_lm_stage(
                    model, data, parameters, stage, names, values, current_stats, run, record,
                )
            steps = len(run.updates) - before
            finish_trace(stage)
            trace_start = None
            stages.append(StageSummary(stage.name, names, steps, status, start_loss, values.total, reason))
            reason = f"{stage.name}: {reason}"
            running_stage = None
            record(stage.name, steps, values, current_stats, {})
            if status in FAILED_STATUSES:
                break

        if status not in FAILED_STATUSES:
            with torch.no_grad():
                final = FitState.from_tensors(parameters.state(0, len(data)))
            final_stats = current_stats
    except (FloatingPointError, RuntimeError, ValueError) as error:
        if profile is not None:
            failure = backend_failure(error)
        status = "failed" if use_adam else "numerical_failure"
        reason = f"{stage_name}: {error}"
        if running_stage is not None:
            stage, names, before, start_loss = running_stage
            finish_trace(stage)
            stages.append(StageSummary(
                stage.name, names, len(run.updates) - before, status, start_loss, run.last_objective.total, str(error),
            ))

    instrumentation = {}
    if profile is not None:
        try:
            operations = run.operations.record()
        except RuntimeError as error:
            operations = {"error": str(error)}
        instrumentation = {
            "operations": operations, "normal_system": run.normal_system,
            "stage_trace_seconds": run.stage_trace_seconds, "infrastructure_failure": failure,
        }
    if config.lm.fp64_square_root:
        instrumentation["square_root_stages"] = run.square_root_stages
    elapsed = perf_counter() - started
    provenance = data.dense_provenance()
    dense_provenance = {} if provenance is None else {"dense_observations": provenance}
    return FitResult(
        model=deepcopy(model.metadata), dtype=model.numpy_dtype.name,
        device=str(device), config=config, selection=selection,
        observation_ids=data.ids, initialization=choices, initial=initial_snapshot,
        final=final, initial_reprojection=initial_stats, final_reprojection=final_stats,
        history=tuple(history), stages=tuple(stages), status=status, reason=reason,
        runtime={
            **dense_provenance,
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "initialization_seconds": initialization_seconds,
            "optimization_seconds": elapsed - initialization_seconds, "total_seconds": elapsed,
            "optimizer_updates": len(run.updates), "observations_per_iteration": len(data),
            "model_loading_seconds": None, "peak_process_rss_bytes": peak_rss_bytes(),
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "objective_seconds": run.seconds["objective"],
            "jacobian_seconds": run.seconds["jacobian"] if not use_adam else None,
            "linear_solve_seconds": run.seconds["linear_solve"] if not use_adam else None,
            **instrumentation,
        },
        optimization=run.diagnostics(), stage_diagnostics=tuple(stage_diagnostics),
    )
