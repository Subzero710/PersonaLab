"""FP32 or FP64 block least-squares: eliminate local coordinates in residual space.

No normal matrix or subtractive Schur complement is formed. Damping is an
explicit augmented residual, and prediction uses the actual rounded step.
"""
from dataclasses import dataclass
import math

import torch

from .schur import BlockVector, _scale


@dataclass(frozen=True, slots=True)
class SquareRootBlocks:
    shared: torch.Tensor  # (N, M, P)
    local: torch.Tensor  # (N, M, Q)
    residual: torch.Tensor  # (N, M)
    gradient: BlockVector
    batch_size: int

    @classmethod
    def create(cls, jacobian, residual, width, batch_size):
        if (jacobian.dtype not in (torch.float32, torch.float64)
                or residual.dtype != jacobian.dtype or residual.device != jacobian.device):
            raise ValueError("Square-root requires matching FP32 or FP64 Jacobians and residuals on one device")
        a, b = jacobian[..., :width], jacobian[..., width:]
        gradient = BlockVector(
            (a.mT @ residual.unsqueeze(-1)).squeeze(-1).sum(0),
            (b.mT @ residual.unsqueeze(-1)).squeeze(-1),
        )
        return cls(a, b, residual, gradient, batch_size)

    def scales(self, floor):
        return BlockVector(_scale(self.shared.square().sum((0, 1)), floor),
                           _scale(self.local.square().sum(1), floor))

    def predicted_reduction(self, step):
        delta = (self.shared @ step.identity +
                 (self.local @ step.local.unsqueeze(-1)).squeeze(-1))
        return float((-delta * (2 * self.residual + delta)).sum())


def square_root_step(system, scales, damping, fixed_local):
    """Solve augmented, equilibrated block least squares via Householder QR."""
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("Positive finite damping required")
    a, b, r = system.shared, system.local, system.residual
    n, m, p = a.shape
    q = b.shape[-1]
    scale_b = scales.local.masked_fill(fixed_local, 0)
    root = math.sqrt(damping)
    reduced = []
    local_factors = []
    for start in range(0, n, system.batch_size):
        stop = min(n, start + system.batch_size)
        count = stop - start
        aa = a[start:stop] * scales.identity
        bb = b[start:stop] * scale_b[start:stop, None, :]
        rr = r[start:stop, :, None]
        rhs = torch.cat((aa, rr), dim=-1)
        if q:
            # Local damping rows, with zeros in the shared/residual columns.
            augmented = torch.cat((bb, torch.eye(q, device=b.device, dtype=b.dtype).expand(count, q, q) * root), dim=1)
            rhs = torch.cat((rhs, rhs.new_zeros((count, q, p + 1))), dim=1)
            reflectors, tau = torch.geqrf(augmented)
            rotated = torch.ormqr(reflectors, tau, rhs, left=True, transpose=True)
            local_factors.append((reflectors[:, :q].triu(), rotated[:, :q]))
            rhs = rotated[:, q:]
        reduced.append(rhs.reshape(-1, p + 1))
    # Shared damping occurs once, not once per observation.
    if p:
        shared_damping = torch.cat((torch.eye(p, device=a.device, dtype=a.dtype) * root, a.new_zeros((p, 1))), dim=1)
        augmented = torch.cat((*reduced, shared_damping), dim=0)
        # QR of the augmented residual column as well avoids normal products.
        triangular = torch.linalg.qr(augmented, mode="r").R
        ya = torch.linalg.solve_triangular(triangular[:p, :p], -triangular[:p, p:], upper=True).squeeze(-1)
    else:
        ya = a.new_empty(0)
    local_steps = []
    for triangular, top in local_factors:
        rhs = -(top[..., :p] @ ya + top[..., p]).unsqueeze(-1)
        local_steps.append(torch.linalg.solve_triangular(triangular, rhs, upper=True).squeeze(-1))
    yb = torch.cat(local_steps) if q else b.new_empty((n, 0))
    step = BlockVector(ya * scales.identity, yb * scale_b)
    if not torch.isfinite(step.identity).all() or not torch.isfinite(step.local).all():
        raise FloatingPointError("Nonfinite square-root LM step")
    return step, {}
