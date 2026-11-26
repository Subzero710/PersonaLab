"""Local Jacobian diagnostics for the active native parameter selection."""

import math

import torch

from ..benchmark.configuration import finite_number
from .configuration import GROUPS
from .initialization import initialize
from .levenberg_marquardt import StageCoordinates, observation_jacobian
from .observations import prepare_observations
from .parameters import (
    FitInitialization, FittingParameters, TensorState, copy_camera, resolve_selection, validate_model,
)
from .results import ARRAY_FIELDS, FitResult


def _remove_nuisance(identity, nuisance, rtol):
    if not nuisance.shape[1]:
        return identity
    u, s, _ = torch.linalg.svd(nuisance, full_matrices=False)
    basis = u[:, s > s[0] * rtol]
    return identity - basis @ (basis.mT @ identity)


def _spectrum(matrix, values, threshold):
    width = matrix.shape[1]
    rank = int((values > threshold).sum())
    effective = float(values[0] / values[rank - 1]) if rank else None
    if effective is not None and not math.isfinite(effective):
        effective = None
    return {
        "rank": rank,
        "identity_parameter_count": width,
        "nullity": width - rank,
        "condition_number": effective if rank == width else None,
        "effective_condition_number": effective,
        "implicit_zero_singular_values": width - values.numel(),
        "singular_values": values.tolist() + [0.0] * (width - values.numel()),
        "column_l2": torch.linalg.vector_norm(matrix, dim=0).tolist(),
    }


def _state(model, data, config, selection, initial, fit, at):
    if fit is None:
        if at != "initial":
            raise ValueError("Final-state observability requires a fitted result")
        state, _ = initialize(model, data, config, FitInitialization() if initial is None else initial)
        return state
    if (
        initial is not None or not isinstance(fit, FitResult) or fit.config != config
        or fit.model != model.metadata or fit.dtype != model.numpy_dtype.name
        or fit.observation_ids != data.ids or fit.selection != selection
    ):
        raise ValueError("Observability requires the same model, configuration and ordered observations as the fit")
    snapshot = getattr(fit, at)
    if snapshot is None:
        raise ValueError("The fit has no final snapshot")
    # Evaluate the nuisance tangent at the actual camera, using local offsets from
    # that camera. Other native coordinates are copied directly from the snapshot.
    return TensorState(
        *(model.tensor(getattr(snapshot, name)) for name in ARRAY_FIELDS),
        tuple(copy_camera(camera, device=model.reference.device) for camera in snapshot.cameras),
    )


@torch.no_grad()
def identity_observability(model, observations, config, *, initial=None, fit=None,
                           at="initial", rtol=1e-6, atol=0.0):
    """Measure unregularized identity sensitivity at initialization or a fit snapshot.

    Use the same landmark weights as fitting. One view is differentiated at a
    time and each stacked identity Jacobian is compressed by QR as views arrive.
    Ranks share max(atol, rtol * largest fixed-nuisance singular value).
    """
    rtol = finite_number(rtol, "Jacobian relative tolerance")
    atol = finite_number(atol, "Jacobian absolute tolerance")
    if not 0 < rtol < 1 or atol < 0:
        raise ValueError("Jacobian tolerances require 0 < rtol < 1 and atol >= 0")
    if at not in ("initial", "final"):
        raise ValueError("Observability state must be initial or final")
    if torch.is_inference_mode_enabled():
        raise ValueError("Jacobian analysis requires PyTorch autograd outside inference mode")
    validate_model(model)
    data = prepare_observations(observations, model.reference, config)
    selection = resolve_selection(model, config)
    width = len(selection.identity)
    if not width:
        raise ValueError("Jacobian analysis requires selected identity coefficients")
    state = _state(model, data, config, selection, initial, fit, at)
    parameters = FittingParameters(model, data, state, selection, config)
    layout = StageCoordinates.from_parameters(parameters, GROUPS)
    base = parameters.values()
    position = layout.pack(base, model.reference, len(data))
    expression_slice = layout.expression_slice()

    names = ("fixed_nuisance", "after_expression", "after_all_nuisance")
    compressed = [model.reference.new_empty((0, width), dtype=torch.float64) for _ in names]
    for index in range(len(data)):
        _, jacobian = observation_jacobian(data, parameters, layout, base, position, index)
        identity, nuisance = jacobian[:, :width], jacobian[:, width:]
        blocks = (
            identity,
            _remove_nuisance(identity, nuisance[:, expression_slice], rtol)
            if expression_slice is not None else identity,
            _remove_nuisance(identity, nuisance, rtol),
        )
        for group, block in enumerate(blocks):
            matrix = torch.cat((compressed[group], block), dim=0)
            compressed[group] = torch.linalg.qr(matrix, mode="r").R

    singular_values = [torch.linalg.svdvals(matrix) for matrix in compressed]
    threshold = max(atol, rtol * float(singular_values[0][0]))
    if not math.isfinite(threshold):
        raise FloatingPointError("Nonfinite observability threshold")
    return {
        "state": at, "observation_ids": list(data.ids), "identity_names": list(selection.identity),
        "identity_parameter_count": width, "relative_tolerance": rtol,
        "absolute_tolerance": atol, "absolute_threshold": threshold,
        "identity_prior_jtj_diagonal": config.identity_weight / width,
        **{
            name: _spectrum(matrix, values, threshold)
            for name, matrix, values in zip(names, compressed, singular_values, strict=True)
        },
    }
