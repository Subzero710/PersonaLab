"""Staged Adam fitting of one native GNM identity across multiple observations."""

from copy import deepcopy
import math
import platform
from time import perf_counter

import numpy as np
import torch

from .configuration import FitConfig
from .initialization import initialize
from .losses import evaluate_objective
from .observations import prepare_observations
from .parameters import FitInitialization, FittingParameters, resolve_selection, validate_model
from .results import FitResult, FitState, Iteration, StageSummary


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


def fit_identity(model, observations, config=None, *, initial=None, progress=None) -> FitResult:
    """Fit a single subject using only observations, a verified model and explicit inputs.

    Configuration and initialization errors raise ValueError. Numerical failures
    during optimization return a failed result with no final parameters.
    """
    config = FitConfig() if config is None else config
    initial = FitInitialization() if initial is None else initial
    if not isinstance(config, FitConfig):
        raise ValueError("Expected a FitConfig")
    if torch.is_inference_mode_enabled():
        raise ValueError("Identity fitting requires PyTorch autograd outside inference mode")
    validate_model(model)
    started = perf_counter()
    data = prepare_observations(observations, model.reference, config)
    selection = resolve_selection(model, config)
    initial_state, choices = initialize(model, data, config, initial)
    parameters = FittingParameters(model, data, initial_state, selection, config)
    groups = parameters.groups()
    if not any(stage.iterations and any(groups[name] for name in stage.groups)
               for stage in config.stages):
        raise ValueError("No stage has an available trainable parameter group")
    initial_snapshot = FitState.from_tensors(initial_state)
    initialization_seconds = perf_counter() - started

    history, stages = [], []
    updates = 0
    initial_stats, final, final_stats = None, None, None
    status, reason = "budget_exhausted", "Iteration budget reached"
    stage_name = "initial evaluation"

    def record(stage, stage_iteration, losses, stats, norms):
        log = Iteration(updates, stage, stage_iteration, losses, stats.overall,
                        norms, perf_counter() - started)
        history.append(log)
        if progress is not None:
            progress(log)

    try:
        values, initial_stats = evaluate_objective(model, data, parameters)
        current_stats = initial_stats
        for stage in config.stages:
            stage_name = stage.name
            for name, group in groups.items():
                for parameter in group:
                    parameter.requires_grad_(name in stage.groups)
                    parameter.grad = None
            names = tuple(name for name in stage.groups if groups[name])
            start_loss = values.total
            if not stage.iterations or not names:
                stages.append(StageSummary(stage.name, names, 0, "skipped", start_loss, start_loss))
                continue

            active = tuple((name, p) for name in names for p in groups[name])
            rates = dict(stage.group_rates)
            optimizer = torch.optim.Adam([
                {"params": groups[name], "lr": rates.get(name, stage.learning_rate)}
                for name in names
            ])
            before = updates
            best, stale = start_loss, 0
            status, reason = "budget_exhausted", f"{stage.name}: iteration budget reached"
            for step in range(stage.iterations):
                optimizer.zero_grad(set_to_none=True)
                values, stats = evaluate_objective(model, data, parameters, active)
                _check_divergence(values.total, start_loss, config.divergence_factor)
                norms = _gradient_norms(active)
                if step == 0 or updates % config.log_every == 0:
                    record(stage.name, step, values, stats, norms)

                if config.gradient_tolerance is not None and (
                    math.hypot(*norms.values()) <= config.gradient_tolerance
                ):
                    status, reason = "converged", f"{stage.name}: gradient tolerance reached"
                    break

                if config.improvement_tolerance is not None and step:
                    if best - values.total > config.improvement_tolerance:
                        best, stale = values.total, 0
                    else:
                        stale += 1
                    if stale >= config.patience:
                        status, reason = "stopped", f"{stage.name}: objective improvement below tolerance"
                        break

                optimizer.step()
                _validate_step(active, optimizer)
                if parameters.expression is not None and "expression" in names:
                    with torch.no_grad():
                        parameters.expression.clamp_(-config.expression_limit, config.expression_limit)
                updates += 1

            values, current_stats = evaluate_objective(model, data, parameters)
            _check_divergence(values.total, start_loss, config.divergence_factor)
            stages.append(StageSummary(stage.name, names, updates - before,
                                       status, start_loss, values.total))
            record(stage.name, updates - before, values, current_stats, {})

        with torch.no_grad():
            final = FitState.from_tensors(parameters.state(0, len(data)))
        final_stats = current_stats
    except (FloatingPointError, RuntimeError, ValueError) as error:
        status, reason = "failed", f"{stage_name}: {error}"

    elapsed = perf_counter() - started
    return FitResult(
        model=deepcopy(model.metadata), dtype=model.numpy_dtype.name,
        device=str(model.reference.device), config=config, selection=selection,
        observation_ids=data.ids, initialization=choices, initial=initial_snapshot,
        final=final, initial_reprojection=initial_stats, final_reprojection=final_stats,
        history=tuple(history), stages=tuple(stages), status=status, reason=reason,
        runtime={
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": str(torch.__version__), "initialization_seconds": initialization_seconds,
            "optimization_seconds": elapsed - initialization_seconds, "total_seconds": elapsed,
            "optimizer_updates": updates, "observations_per_iteration": len(data),
        },
    )
