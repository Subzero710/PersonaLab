"""Dense least-squares oracles for V7 staged, regularized square-root QR."""
import pytest
import torch

from persona_lab.fitting.schur import BlockVector
from persona_lab.fitting.square_root import SquareRootBlocks, square_root_step


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('p,q', [(4, 3), (0, 3), (4, 0)])
@pytest.mark.parametrize('batch', [1, 3, 5])
def test_regularized_qr_matches_dense(device, p, q, batch):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    n, m = 5, 8
    gen = torch.Generator().manual_seed(71)
    def rand(*shape):
        return torch.randn(shape, generator=gen, dtype=torch.float64).to(device)
    a, b, r = rand(n, m, p), rand(n, m, q), rand(n, m)
    residual = BlockVector(rand(p), rand(n, q))
    slope = BlockVector(rand(p).abs(), rand(n, q).abs())
    if p:
        slope.identity[-1] = residual.identity[-1] = 0
    if q:
        slope.local[:, -1] = residual.local[:, -1] = 0
    system = SquareRootBlocks.create(torch.cat((a, b), -1), r, p, batch,
                                     prior_residual=residual, prior_slope=slope)
    scales = system.scales(1e-12)
    fixed = torch.zeros_like(b[:, 0], dtype=torch.bool)
    if q:
        fixed[0, 0] = True
    damping = 3e-4
    step, _ = square_root_step(system, scales, damping, fixed)
    assert step.identity.dtype == step.local.dtype == torch.float64
    assert torch.equal(step.local[fixed], torch.zeros_like(step.local[fixed]))
    # Independent dense reference, confined to this test.
    matrix = torch.cat((a.flatten(0, 1), torch.block_diag(*b)), dim=1).cpu()
    slope_vector = torch.cat((slope.identity, slope.local.flatten())).cpu()
    matrix = torch.cat((matrix, torch.diag(slope_vector)))
    target = torch.cat((r.flatten(), residual.identity, residual.local.flatten())).cpu()
    actual = torch.cat((step.identity, step.local.flatten())).cpu()
    gradient = torch.cat((system.gradient.identity, system.gradient.local.flatten())).cpu()
    torch.testing.assert_close(gradient, matrix.T @ target, rtol=1e-12, atol=1e-12)
    raw_scale = torch.cat((scales.identity, scales.local.flatten())).cpu()
    torch.testing.assert_close(raw_scale, matrix.square().sum(0).rsqrt(), rtol=1e-12, atol=1e-12)
    scale = torch.cat((scales.identity, scales.local.masked_fill(fixed, 0).flatten())).cpu()
    augmented = torch.cat((matrix * scale, torch.eye(matrix.shape[1], dtype=matrix.dtype) * damping**.5))
    rhs = torch.cat((-target, torch.zeros(matrix.shape[1], dtype=matrix.dtype)))
    expected = torch.linalg.lstsq(augmented, rhs, driver='gels').solution * scale
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    reduction = float(target.square().sum() - (target + matrix @ actual).square().sum())
    assert reduction > 0
    assert system.predicted_reduction(step) == pytest.approx(reduction, rel=1e-11, abs=1e-11)
