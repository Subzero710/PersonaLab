"""Independent FP64 test oracles for FP32 high/low operations and native GNM math."""
from types import SimpleNamespace
import unittest

import torch

from persona_lab.fitting import precision_geometry as pg


def combined(a):
    return a[0].double() + a[1].double()


def toy(device):
    rng = torch.Generator().manual_seed(82)
    def sample(*shape, scale=1.):
        return (torch.randn(*shape, generator=rng) * scale).to(device)
    n, j, v, i, e = 5, 4, 17, 29, 43
    head = SimpleNamespace(identity_dim=i, num_joints=j, joint_parent_indices=(0, 0, 1, 1),
                           template_joint_positions=sample(j, 3, scale=.1),
                           joint_identity_basis=sample(i, j, 3, scale=.001))
    skin = sample(j, v).softmax(0)
    weights = sample(8, 3).softmax(-1)
    sparse = SimpleNamespace(head=head, template=sample(v, 3, scale=.1),
                             identity_basis=sample(i, v, 3, scale=.001),
                             expression_basis=sample(e, v, 3, scale=.001),
                             pose_regressor=sample(j*9, v*3, scale=.001), skinning_weights=skin,
                             triangles=torch.randint(v, (8, 3), generator=rng).to(device), weights=weights)
    identity, expression = sample(i, scale=.4), sample(n, e, scale=.4)
    rotations = sample(n, j, 3, scale=.3)
    rotations[0] = 0
    root = sample(n, 3, scale=.01)
    return sparse, identity, expression, rotations, root


def rotation_oracle(v):
    x,y,z = v.double().unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*x.shape,3,3)
    return torch.linalg.matrix_exp(k)


def oracle(s, identity, expression, rotations, root):
    head = s.head
    i, e = identity.double(), expression.double()
    v = (s.template.double() + torch.einsum('i,ivc->vc', i, s.identity_basis.double())
         + torch.einsum('ni,ivc->nvc', e, s.expression_basis.double()))
    joints = head.template_joint_positions.double() + torch.einsum('i,ijc->jc', i, head.joint_identity_basis.double())
    matrices = rotation_oracle(rotations)
    features = (matrices - torch.eye(3, dtype=torch.float64, device=i.device)).flatten(1)
    if s.pose_regressor is not None:
        v += (features @ s.pose_regressor.double()).reshape(v.shape)
    r, t = [], []
    for j, parent in enumerate(head.joint_parent_indices):
        if j == 0:
            r.append(matrices[:,j]); t.append(joints[j]+root.double())
        else:
            r.append(r[parent] @ matrices[:,j])
            t.append(t[parent]+(r[parent] @ (joints[j]-joints[parent])[:,None]).squeeze(-1))
    # Independent implementation matching native blend of affine transforms.
    r = torch.stack(r,1); t = torch.stack(t,1) - (r @ joints[...,None]).squeeze(-1)
    rv = torch.einsum('jv,njab->nvab', s.skinning_weights.double(), r)
    tv = torch.einsum('jv,nja->nva', s.skinning_weights.double(), t)
    posed = (rv @ v.unsqueeze(-1)).squeeze(-1)+tv
    return (posed[:,s.triangles] * s.weights.double()[...,None]).sum(-2)


class PrecisionGeometryTests(unittest.TestCase):
    def _check(self, device):
        s,i,e,r,t = toy(device)
        points = pg.landmark_pair(s,i,e,r,t)
        expected = oracle(s,i,e,r,t)
        torch.testing.assert_close(combined(points), expected, atol=2e-13, rtol=0)
        split = [pg.landmark_pair(s,i,e[k:k+1],r[k:k+1],t[k:k+1]) for k in range(len(e))]
        for part in range(2):
            self.assertTrue(torch.equal(points[part], torch.cat([x[part] for x in split])))
        rotation = r[:,0]
        translation = r.new_tensor([0.001,0.265,1.1]).expand(len(e),-1)
        focal = r.new_tensor([512.,768.]).expand(len(e),-1)
        center = r.new_tensor([255.5,255.5]).expand(len(e),-1)
        actual = pg.projected_pair(points,rotation,translation,focal,center)
        xyz = (expected * expected.new_tensor([1,-1,-1])) @ rotation_oracle(rotation).mT + translation.double()[:,None]
        exact = xyz[...,:2]/xyz[...,2:] * focal.double()[:,None]+center.double()[:,None]
        torch.testing.assert_close(combined(actual), exact, atol=4e-10, rtol=0)
        # A high/low archive generated from the same forward replays exactly.
        hi,lo = pg.sub(actual,actual)
        self.assertFalse(bool(hi.any() | lo.any()))
        self.assertEqual(actual[0].dtype, torch.float32)
        self.assertTrue(bool((actual[1].abs() <= (torch.nextafter(actual[0], torch.full_like(actual[0], float('inf'))) - actual[0])*0.5).all()))
        return actual

    def test_cpu_full_forward(self):
        self._check('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_full_forward(self):
        self._check('cuda:0')

    def test_rotation_near_pi_and_invalid_range(self):
        r = torch.tensor([[0.,0.,0.],[0.01,-.02,.03],[3.,-2.,1.],[6.,0.,0.]])
        torch.testing.assert_close(combined(pg.rotation_pair(r)), rotation_oracle(r), atol=4e-13, rtol=0)
        with self.assertRaises(ValueError):
            pg.rotation_pair(torch.tensor([[7.,0.,0.]]))

    def test_compensated_basis_cancellation(self):
        coefficients = torch.tensor([[1.,1.,1.,1.]])
        basis = torch.tensor([[1.],[1e-8],[-1.],[1e-8]])
        actual = pg.linear(pg.pair(coefficients),basis,chunk=4)
        torch.testing.assert_close(combined(actual), coefficients.double() @ basis.double(), atol=1e-15,rtol=0)
