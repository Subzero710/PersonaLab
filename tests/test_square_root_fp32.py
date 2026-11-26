"""Independent tests of the FP32 square-root block algebra (no GNM asset)."""
import unittest

import torch

from persona_lab.fitting.schur import BlockVector
from persona_lab.fitting.square_root import SquareRootBlocks, square_root_step


class SquareRootTests(unittest.TestCase):
    def _compare(self, device, *, weak=False, fixed=False, batch_size=2):
        gen = torch.Generator().manual_seed(101)
        a = torch.randn(5, 16, 4, generator=gen)
        b = torch.randn(5, 16, 3, generator=gen)
        if weak:
            # A pair of nearly collinear shared columns in the FP32 system.
            a[..., 1] = a[..., 0] + 2e-4 * a[..., 1]
        r = torch.randn(5, 16, generator=gen)
        a, b, r = (x.to(device=device, dtype=torch.float32) for x in (a, b, r))
        system = SquareRootBlocks.create(torch.cat((a, b), dim=-1), r, 4, batch_size)
        scales = system.scales(1e-12)
        mask = torch.zeros_like(scales.local, dtype=torch.bool)
        if fixed:
            mask[1, 0] = True
            mask[3, 2] = True
        damping = 1e-7 if weak else 1e-3
        step, _ = square_root_step(system, scales, damping, mask)
        self.assertEqual(step.identity.dtype, torch.float32)
        self.assertEqual(step.local.device, torch.device(device))
        self.assertTrue(torch.all(step.local[mask] == 0))
        # Independent dense least-squares oracle: used only by the test.
        matrix = torch.cat((a.flatten(0, 1), torch.block_diag(*b)), dim=1).cpu().double()
        scale = torch.cat((scales.identity, scales.local.masked_fill(mask, 0).flatten())).cpu().double()
        m = matrix * scale
        aug = torch.cat((m, torch.eye(m.shape[1], dtype=torch.float64) * damping ** .5))
        rhs = torch.cat((-r.flatten().cpu().double(), torch.zeros(m.shape[1], dtype=torch.float64)))
        expected = torch.linalg.lstsq(aug, rhs).solution * scale
        actual = torch.cat((step.identity, step.local.flatten())).cpu().double()
        torch.testing.assert_close(actual, expected, rtol=3e-3 if weak else 2e-5,
                                   atol=5e-3 if weak else 2e-6)
        expected_reduction = (r.cpu().double().square().sum() -
                              (r.flatten().cpu().double() + matrix @ actual).square().sum()).item()
        self.assertAlmostEqual(system.predicted_reduction(step), expected_reduction, delta=3e-4)

    def test_cpu_masked_and_unmasked(self):
        for fixed in (False, True):
            for batch in (1, 2, 5):
                with self.subTest(fixed=fixed, batch=batch):
                    self._compare('cpu', fixed=fixed, batch_size=batch)

    def test_cpu_weak_direction(self):
        self._compare('cpu', weak=True, fixed=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_masked_and_weak_direction(self):
        self._compare('cuda:0', fixed=True)
        self._compare('cuda:0', weak=True, fixed=True)


if __name__ == '__main__':
    unittest.main()
