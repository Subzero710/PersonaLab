"""Damped Gauss–Newton steps for shared and independent local blocks."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BlockVector:
    # The historical identity field stores the entire shared block.
    identity: Tensor
    local: Tensor


@dataclass(frozen=True, slots=True)
class NormalBlocks:
    identity: Tensor
    coupling: Tensor
    local: Tensor
    gradient: BlockVector

    def predicted_reduction(self, step):
        """Change of the undamped squared-residual model for the actual step."""
        a, b = step.identity, step.local
        linear = self.gradient.identity @ a + (self.gradient.local * b).sum()
        local_quadratic = (b * (self.local @ b.unsqueeze(-1)).squeeze(-1)).sum()
        cross = a @ (self.coupling @ b.unsqueeze(-1)).sum(0).squeeze(-1)
        return float(-2 * linear - a @ (self.identity @ a) - 2 * cross - local_quadratic)


def _scale(diagonal, floor):
    if not diagonal.numel():
        return diagonal
    if not torch.isfinite(diagonal).all() or (diagonal < 0).any():
        raise FloatingPointError("Invalid normal-matrix diagonal")
    minimum = (diagonal.amax(dim=-1, keepdim=True) * floor).clamp_min(
        torch.finfo(diagonal.dtype).eps ** 2
    )
    return torch.maximum(diagonal, minimum).rsqrt()


def column_scales(system, relative_floor):
    """Equilibrate columns using squared sensitivities, with a positive floor."""
    return BlockVector(
        _scale(system.identity.diagonal(), relative_floor),
        _scale(system.local.diagonal(dim1=-2, dim2=-1), relative_floor),
    )


def _factor(matrix, label):
    if not torch.isfinite(matrix).all():
        raise FloatingPointError(f"Nonfinite {label} normal matrix")
    # Symmetrize the accumulated products before testing positive definiteness.
    factor, info = torch.linalg.cholesky_ex((matrix + matrix.mT) * 0.5)
    if (info != 0).any() or not torch.isfinite(factor).all():
        raise FloatingPointError(f"Cholesky factorization failed for {label} normal matrix")
    return factor


def _pivot_ratio(factor):
    diagonal = factor.diagonal(dim1=-2, dim2=-1)
    pivots = diagonal.amax(dim=-1) / diagonal.amin(dim=-1)
    ratio = float(pivots.max())
    return ratio if math.isfinite(ratio) else None


def damped_schur_step(system, scales, damping, fixed_local):
    """Solve (S H S + damping I)y = -Sg and return the raw step S y."""
    scale_a = scales.identity
    scale_b = scales.local.masked_fill(fixed_local, 0)
    g_a = system.gradient.identity * scale_a
    g_b = system.gradient.local * scale_b
    p, q = g_a.numel(), g_b.shape[1]
    reduced = system.identity * scale_a[:, None] * scale_a[None, :]
    reduced.diagonal().add_(damping)
    condition = {}

    if q:
        local = system.local * scale_b.unsqueeze(-1) * scale_b.unsqueeze(-2)
        local.diagonal(dim1=-2, dim2=-1).add_(damping)
        factor_b = _factor(local, "local")
        condition["local_cholesky_diagonal_ratio"] = _pivot_ratio(factor_b)
        cross = system.coupling * scale_a[None, :, None] * scale_b[:, None, :]
        rhs = torch.cat((cross.mT, g_b.unsqueeze(-1)), dim=-1)
        whitened = torch.linalg.solve_triangular(factor_b, rhs, upper=False)
        z_a, z_b = whitened[..., :p], whitened[..., p:]
        if p:
            flattened = z_a.flatten(0, 1)
            reduced = reduced - flattened.mT @ flattened
            g_a = g_a - flattened.mT @ z_b.flatten()

    y_a = g_a
    if p:
        factor_a = _factor(reduced, "Schur complement")
        condition["schur_cholesky_diagonal_ratio"] = _pivot_ratio(factor_a)
        y_a = torch.cholesky_solve(-g_a.unsqueeze(-1), factor_a).squeeze(-1)

    y_b = g_b
    if q:
        y_b = torch.linalg.solve_triangular(
            factor_b.mT, -(z_b + z_a @ y_a.unsqueeze(-1)), upper=True,
        ).squeeze(-1)
    step = BlockVector(scale_a * y_a, scale_b * y_b)
    if not torch.isfinite(step.identity).all() or not torch.isfinite(step.local).all():
        raise FloatingPointError("Nonfinite Schur-complement update")
    return step, condition
