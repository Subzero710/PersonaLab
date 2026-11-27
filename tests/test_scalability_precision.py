"""Check the existing Schur implementation in explicitly selected arithmetic."""

import unittest

import torch

from persona_lab.fitting.schur import BlockVector, NormalBlocks, column_scales, damped_schur_step


class ScalabilityPrecisionTests(unittest.TestCase):
    def test_fp32_schur_matches_small_dense_reference(self):
        # Two views, two shared coordinates and two independent local coordinates.
        generator = torch.Generator().manual_seed(42)
        a = torch.randn(2, 12, 2, generator=generator, dtype=torch.float32)
        b = torch.randn(2, 12, 2, generator=generator, dtype=torch.float32)
        r = torch.randn(2, 12, generator=generator, dtype=torch.float32)
        system = NormalBlocks(
            (a.mT @ a).sum(0), a.mT @ b, b.mT @ b,
            BlockVector((a.mT @ r.unsqueeze(-1)).squeeze(-1).sum(0),
                        (b.mT @ r.unsqueeze(-1)).squeeze(-1)),
        )
        scales = column_scales(system, 1e-12)
        damping = 1e-3
        step, _ = damped_schur_step(system, scales, damping, torch.zeros_like(scales.local, dtype=torch.bool))
        self.assertEqual(step.identity.dtype, torch.float32)
        self.assertEqual(step.local.dtype, torch.float32)
        dense = torch.cat((a.flatten(0, 1), torch.block_diag(*b)), dim=1)
        scale = torch.cat((scales.identity, scales.local.flatten()))
        scaled = dense * scale
        expected = scale * torch.linalg.solve(
            scaled.mT @ scaled + damping * torch.eye(6, dtype=torch.float32),
            -(scaled.mT @ r.flatten()),
        )
        torch.testing.assert_close(torch.cat((step.identity, step.local.flatten())), expected,
                                   atol=1e-6, rtol=1e-5)
        self.assertGreater(system.predicted_reduction(step), 0.0)


if __name__ == "__main__":
    unittest.main()
