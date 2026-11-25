"""Local Jacobian diagnostics for the active native parameter selection."""

import math

import torch

from ..benchmark.configuration import finite_number
from .initialization import initialize
from .losses import project_batch
from .observations import prepare_observations
from .parameters import FitInitialization, FittingParameters, resolve_selection, validate_model


def _remove_nuisance(identity, columns, rtol):
    if not columns:
        return identity
    nuisance = torch.cat(columns, dim=1)
    u, s, _ = torch.linalg.svd(nuisance, full_matrices=False)
    basis = u[:, s > s[0] * rtol]
    return identity - basis @ (basis.mT @ identity)


def _jacobian_blocks(model, data, parameters, index):
    groups = parameters.groups()
    leaves = [(name, parameter) for name, group in groups.items() for parameter in group]
    state = parameters.state(index, index + 1)
    projected, = project_batch(model, state, (data.ids[index],))
    eligible = data.eligible[index]
    weights = (data.weights[index, eligible] / data.weight_sum).sqrt()
    selected = projected[data.gnm_indices[index, eligible]]
    outputs = (selected * weights[:, None] / data.short_edges[index]).flatten()
    rows = [[] for _ in leaves]
    for row, value in enumerate(outputs):
        gradients = torch.autograd.grad(
            value, tuple(parameter for _, parameter in leaves),
            retain_graph=row + 1 < len(outputs),
        )
        for (name, _), gradient, block in zip(leaves, gradients, rows, strict=True):
            column = gradient if name == "identity" else gradient[index]
            block.append(column.detach().reshape(-1))
    return [
        (name, torch.stack(block).to(dtype=torch.float64))
        for (name, _), block in zip(leaves, rows, strict=True)
    ]


def identity_observability(model, observations, config, *, initial=None, rtol=1e-6):
    """Measure local identity sensitivity before and after removing nuisance directions.

    Computes the weighted 2D reprojection Jacobian at the initialized state.
    The graph is scoped to one observation. Reported ranks use a common
    threshold relative to the largest fixed-nuisance identity singular value.
    This is a local unregularized diagnostic.
    """
    rtol = finite_number(rtol, "Jacobian relative tolerance")
    if not 0 < rtol < 1:
        raise ValueError("Jacobian relative tolerance must be between zero and one")
    if torch.is_inference_mode_enabled():
        raise ValueError("Jacobian analysis requires PyTorch autograd outside inference mode")
    validate_model(model)
    data = prepare_observations(observations, model.reference, config)
    selection = resolve_selection(model, config)
    if not selection.identity:
        raise ValueError("Jacobian analysis requires selected identity coefficients")
    initial = FitInitialization() if initial is None else initial
    state, _ = initialize(model, data, config, initial)
    parameters = FittingParameters(model, data, state, selection, config)

    views = []
    with torch.enable_grad():
        for index in range(len(data)):
            blocks = _jacobian_blocks(model, data, parameters, index)
            identity = next(block for name, block in blocks if name == "identity")
            expression = [block for name, block in blocks if name == "expression"]
            all_nuisance = [block for name, block in blocks if name != "identity"]
            views.append((
                identity, _remove_nuisance(identity, expression, rtol),
                _remove_nuisance(identity, all_nuisance, rtol),
            ))

    diagnostics = {}
    threshold = None
    for name, blocks in zip(
        ("fixed_nuisance", "after_expression", "after_all_nuisance"), zip(*views, strict=True),
        strict=True,
    ):
        matrix = torch.cat(blocks, dim=0)
        values = torch.linalg.svdvals(matrix)
        if threshold is None:
            threshold = rtol * float(values[0])
        diagnostics[name] = {
            "rank": int((values > threshold).sum()),
            "singular_values": values.tolist(),
            "column_l2": torch.linalg.vector_norm(matrix, dim=0).tolist(),
        }
    if not math.isfinite(threshold):
        raise FloatingPointError("Nonfinite observability threshold")
    return {
        "observation_ids": list(data.ids), "identity_names": list(selection.identity),
        "relative_tolerance": rtol, "absolute_threshold": threshold,
        **diagnostics,
    }
