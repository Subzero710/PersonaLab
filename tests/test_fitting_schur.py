"""Compare production block algebra to small dense systems."""

from types import SimpleNamespace

import pytest
import torch

from persona_lab.fitting.configuration import LevenbergMarquardtConfig
from persona_lab.fitting.levenberg_marquardt import StageCoordinates, _trial_step, bound_mask
from persona_lab.fitting.parameters import ParameterValues
from persona_lab.fitting.schur import BlockVector, NormalBlocks, column_scales, damped_schur_step


def _problem(p, q, *, rank_deficient=False):
    reference = torch.empty(0, dtype=torch.float64)
    rng = torch.Generator().manual_seed(346)
    n, rows = 3, 6
    a = torch.randn((n, rows, p), generator=rng, dtype=reference.dtype)
    b = torch.randn((n, rows, q), generator=rng, dtype=reference.dtype)
    if rank_deficient:
        a[..., -1] = a[..., 0] * 1e-7
        b[..., -1] = b[..., 0]
    # Each image contributes a different weight to the same identity and
    # has independent nuisance variables.
    weight = reference.new_tensor((0.02, 0.9, 0.08)).sqrt().reshape(n, 1, 1)
    a, b = a * weight, b * weight
    offsets = torch.randn((n, rows, 1), generator=rng, dtype=reference.dtype) * weight
    penalty = 0.0 if rank_deficient else 0.14
    prior = reference.new_full((p,), 0.7)
    global_hessian = (a.mT @ a).sum(0) + penalty * torch.eye(p, dtype=reference.dtype)
    gradient_a = (a.mT @ offsets).sum(0).squeeze(-1) + penalty * prior
    system = NormalBlocks(
        global_hessian, a.mT @ b, b.mT @ b, BlockVector(gradient_a, (b.mT @ offsets).squeeze(-1)),
    )
    dense = reference.new_zeros((n * rows + p, p + n * q))
    for index in range(n):
        view = slice(index * rows, (index + 1) * rows)
        dense[view, :p] = a[index]
        dense[view, p + index * q:p + (index + 1) * q] = b[index]
    dense[n * rows:, :p] = penalty ** 0.5 * torch.eye(p, dtype=reference.dtype)
    residuals = torch.cat((offsets.flatten(), penalty ** 0.5 * prior))
    return system, dense, residuals


@pytest.mark.parametrize("p,q,rank_deficient", [(3, 2, False), (0, 3, False), (3, 0, False), (3, 2, True)])
@pytest.mark.parametrize("damping", [1e-5, 0.3])
def test_schur_matches_dense_damped_normal_system(p, q, rank_deficient, damping):
    system, dense, residuals = _problem(p, q, rank_deficient=rank_deficient)
    scales = column_scales(system, 1e-12)
    fixed = torch.zeros_like(scales.local, dtype=torch.bool)
    if q and not rank_deficient:
        fixed[0, -1] = True
    step, conditions = damped_schur_step(system, scales, damping, fixed)
    scale = torch.cat((scales.identity, scales.local.masked_fill(fixed, 0).flatten()))
    jacobian = dense * scale
    matrix = jacobian.mT @ jacobian + damping * torch.eye(jacobian.shape[1], dtype=jacobian.dtype)
    expected = scale * torch.linalg.solve(matrix, -(jacobian.mT @ residuals))
    actual = torch.cat((step.identity, step.local.flatten()))
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    assert not step.local[fixed].any()
    assert all(value >= 1 for value in conditions.values())
    predicted = residuals.square().sum() - (residuals + dense @ actual).square().sum()
    assert system.predicted_reduction(step) == pytest.approx(float(predicted), abs=2e-12)


def test_rank_zero_block_stays_zero_under_damping():
    dtype = torch.float64
    system = NormalBlocks(
        torch.zeros((2, 2), dtype=dtype), torch.zeros((2, 2, 1), dtype=dtype),
        torch.zeros((2, 1, 1), dtype=dtype),
        BlockVector(torch.zeros(2, dtype=dtype), torch.zeros((2, 1), dtype=dtype)),
    )
    scales = column_scales(system, 1e-12)
    step, _ = damped_schur_step(system, scales, 1e-8, torch.zeros((2, 1), dtype=torch.bool))
    assert not step.identity.any() and not step.local.any()


def test_factorization_failure_is_explicit():
    system, _, _ = _problem(1, 1)
    scales = column_scales(system, 1e-12)
    system.local[0, 0, 0] = -100
    with pytest.raises(FloatingPointError, match="Cholesky factorization failed for local"):
        damped_schur_step(system, scales, 1e-4, torch.zeros_like(scales.local, dtype=torch.bool))


def test_acceptance_uses_projected_step_in_the_model_dtype():
    identity = torch.empty(0, dtype=torch.float32)
    expression = torch.tensor([[0.9]], dtype=torch.float32)
    position = BlockVector(identity, expression)
    step = BlockVector(identity.double(), torch.tensor([[0.8]], dtype=torch.float64))
    blocks = NormalBlocks(
        identity.double().reshape(0, 0), identity.double().reshape(1, 0, 1),
        step.local.new_ones((1, 1, 1)), BlockVector(step.identity, step.local.new_full((1, 1), -5)),
    )
    layout = StageCoordinates(False, ("expression",), (1,))
    base = ParameterValues(None, expression, None, None, None)
    parameters = SimpleNamespace(config=SimpleNamespace(
        expression_limit=1.0, lm=LevenbergMarquardtConfig(),
    ))

    old_coordinate = expression.item()
    target = old_coordinate + 5

    def evaluate(*args, candidate, **kwargs):
        assert candidate.expression.item() == 1.0
        return SimpleNamespace(total=(candidate.expression.item() - target) ** 2), None

    result, reason, (norm, predicted, actual, ratio) = _trial_step(
        None, None, parameters, base, position, layout, blocks, step, slice(0, 1),
        (old_coordinate - target) ** 2, SimpleNamespace(evaluate=evaluate),
    )
    expected = 1.0 - old_coordinate
    assert result is not None and reason == "accepted"
    assert float(base.expression[0, 0]) == old_coordinate
    assert norm == expected
    assert predicted == pytest.approx(10 * expected - expected ** 2)
    assert actual == pytest.approx(predicted)
    assert ratio == pytest.approx(1)


def test_bound_mask_only_freezes_outward_expression_directions():
    identity = torch.empty(0, dtype=torch.float64)
    local = identity.new_tensor([[-1, -1, 1, 1, 0, 1]])
    gradient = identity.new_tensor([[1, -1, 1, -1, 2, -3]])
    mask = bound_mask(BlockVector(identity, local), BlockVector(identity, gradient), slice(0, 5), 1.0)
    assert mask.tolist() == [[True, False, False, True, False, False]]
