"""FP64 QR algebra and explicit protocol selection without a GNM asset."""
from dataclasses import replace
import unittest

import torch

from persona_lab.fitting.configuration import FitConfig, LevenbergMarquardtConfig, Stage, GROUPS
from persona_lab.fitting.square_root import SquareRootBlocks, square_root_step
from persona_lab.benchmark.scalability_cases import case_matrix
from persona_lab.benchmark.scalability_report import _paired_rows


class SquareRootFP64Tests(unittest.TestCase):
    def _compare(self, device, batch):
        rng = torch.Generator().manual_seed(43)
        a = torch.randn(5, 16, 4, generator=rng, dtype=torch.float64)
        b = torch.randn(5, 16, 3, generator=rng, dtype=torch.float64)
        r = torch.randn(5, 16, generator=rng, dtype=torch.float64)
        a[..., 1] = a[..., 0] + 1e-6 * a[..., 1]
        a, b, r = (t.to(device) for t in (a, b, r))
        system = SquareRootBlocks.create(torch.cat((a, b), dim=-1), r, 4, batch)
        scales = system.scales(1e-12)
        fixed = torch.zeros_like(scales.local, dtype=torch.bool)
        fixed[0, 1] = True
        damping = 1e-12
        step, _ = square_root_step(system, scales, damping, fixed)
        self.assertEqual(step.identity.dtype, torch.float64)
        self.assertEqual(step.local.device, torch.device(device))
        self.assertTrue(torch.all(step.local[fixed] == 0))
        # Independent dense augmented QR, test only. No normal equations.
        j = torch.cat((a.flatten(0, 1), torch.block_diag(*b)), dim=1).cpu()
        scale = torch.cat((scales.identity, scales.local.masked_fill(fixed, 0).flatten())).cpu()
        augmented = torch.cat((j * scale, torch.eye(j.shape[1], dtype=j.dtype) * damping**.5))
        rhs = torch.cat((-r.flatten().cpu(), torch.zeros(j.shape[1], dtype=j.dtype)))
        expected = torch.linalg.lstsq(augmented, rhs, driver='gels').solution * scale
        actual = torch.cat((step.identity, step.local.flatten())).cpu()
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-8)
        reference = float(r.square().sum().cpu() - (r.flatten().cpu() + j @ actual).square().sum())
        self.assertAlmostEqual(system.predicted_reduction(step), reference, delta=1e-8)
        self.assertGreater(reference, 0)

    def test_cpu(self):
        for batch in (1, 2, 5):
            with self.subTest(batch=batch):
                self._compare('cpu', batch)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        self._compare('cuda:0', 2)

    def test_mixed_precision_rejected(self):
        with self.assertRaises(ValueError):
            SquareRootBlocks.create(torch.zeros(2, 4, 3, dtype=torch.float64), torch.zeros(2, 4), 1, 2)

    def test_explicit_config_and_roundtrip(self):
        lm = LevenbergMarquardtConfig(fp64_square_root=True)
        config = FitConfig(camera_mode='estimated', camera_model='perspective', focal_mode='per-observation',
                           initial_focal_factor=1.5, identity_weight=0, expression_weight=0,
                           stages=(Stage('joint', 200, GROUPS, .02),), lm=lm)
        self.assertTrue(config.lm.uses_square_root)
        self.assertEqual(config.to_dict()['lm']['arithmetic_dtype'], 'float64')
        self.assertEqual(FitConfig.from_dict(config.to_dict()), config)
        self.assertFalse(FitConfig().lm.uses_square_root)
        self.assertNotIn('fp64_square_root', FitConfig().to_dict()['lm'])
        with self.assertRaises(ValueError):
            replace(lm, arithmetic_dtype='float32')
        with self.assertRaises(ValueError):
            replace(config, identity_weight=1e-6)
        with self.assertRaises(ValueError):
            replace(config, stages=(Stage('pose', 1, ('pose',), .02),))

    def test_batch32_primary_reuse(self):
        rows = case_matrix(seeds=(42,), counts=(100,), devices=('cpu',), batch_size=32, dtype='float64')
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r['dtype'] == 'float64' for r in rows))
        self.assertEqual(rows[-1]['reuse_case'], rows[0]['case'])
        records = [r | {'report_case': 'report.json#/cases/0' if 'reuse_case' in r else f'report.json#/cases/{i}'}
                   for i, r in enumerate(rows)]
        self.assertEqual(len(list(_paired_rows(records))), 5)
