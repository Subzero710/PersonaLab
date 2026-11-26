"""FP32 perspective residual accuracy and smooth Jacobian on either device."""
import unittest

import torch

from persona_lab.fitting.precise_projection import perspective_residual


class PreciseProjectionTests(unittest.TestCase):
    def _near_target(self, device):
        rng = torch.Generator().manual_seed(73)
        # Typical GNM camera-space scales, including cancellation in Y.
        rotated = torch.randn(7, 68, 3, generator=rng) * 0.035
        rotated[..., 1] -= 0.265
        translation = torch.tensor([0.001, 0.265, 0.7]).expand(7, 3).clone()
        focal = torch.linspace(512, 1024, 7)[:, None].expand(7, 2).clone()
        principal = torch.full((7, 2), 255.5)
        rotated, translation, focal, principal = (x.to(device) for x in (rotated, translation, focal, principal))
        # Float64 oracle is confined to tests; production arithmetic stays FP32.
        xyz = rotated.double() + translation.double()[:, None, :]
        exact = xyz[..., :2] / xyz[..., 2:] * focal.double()[:, None, :] + principal.double()[:, None, :]
        observed = exact.float()
        reference = exact - observed.double()
        ordinary_xyz = rotated + translation[:, None, :]
        ordinary = ordinary_xyz[..., :2] / ordinary_xyz[..., 2:] * focal[:, None, :] + principal[:, None, :] - observed
        precise = perspective_residual(rotated, translation, focal, principal, observed)
        self.assertEqual(precise.dtype, torch.float32)
        self.assertEqual(precise.device, rotated.device)
        torch.testing.assert_close(precise.double(), reference, atol=3e-10, rtol=2e-5)
        self.assertLess(float((precise.double() - reference).abs().max()),
                        float((ordinary.double() - reference).abs().max()) / 100)
        return rotated, translation, focal, principal, observed

    def _gradient(self, device):
        rotated, translation, focal, principal, observed = self._near_target(device)
        inputs = tuple(x.detach().requires_grad_() for x in (rotated, translation, focal, principal))
        residual = perspective_residual(*inputs, observed)
        grads = torch.autograd.grad(residual.sum(), inputs)
        doubles = tuple(x.detach().double().requires_grad_() for x in inputs)
        r, t, f, c = doubles
        xyz = r + t[:, None, :]
        oracle = xyz[..., :2] / xyz[..., 2:] * f[:, None, :] + c[:, None, :] - observed.double()
        expected = torch.autograd.grad(oracle.sum(), doubles)
        for g, e in zip(grads, expected, strict=True):
            torch.testing.assert_close(g.double(), e, atol=2e-4, rtol=3e-6)
        with self.assertRaises(ValueError):
            perspective_residual(inputs[0], -inputs[1], inputs[2], inputs[3], observed)
        with self.assertRaises(ValueError):
            perspective_residual(*(x.double() for x in inputs), observed.double())

    def test_cpu_precision_and_derivative(self):
        self._gradient('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_precision_and_derivative(self):
        self._gradient('cuda:0')

    def test_unrounded_translation_cancellation(self):
        xyz = torch.tensor([[[0.05, -0.265, -0.02]]], dtype=torch.float32)
        t = torch.tensor([[0.0, 0.26500003, 0.7]])
        f = torch.tensor([[768., 768.]])
        c = torch.tensor([[256., 256.]])
        exact_xyz = xyz.double() + t.double()[:, None, :]
        projected = exact_xyz[..., :2] * f.double()[:, None, :] / exact_xyz[..., 2:] + c.double()[:, None, :]
        observed = projected.float()
        precise = perspective_residual(xyz, t, f, c, observed)
        torch.testing.assert_close(precise.double(), projected - observed.double(), atol=3e-10, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
