"""Test staged regularized linearizations with a small differentiable test geometry.

The production projection, parameter layout, QR and LM code are used. Only the
geometry is artificial; official GNM integration is covered separately.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from persona_lab.camera import PerspectiveCamera, WeakPerspectiveCamera
from persona_lab.coordinates import ImageSize
from persona_lab.fitting.configuration import FitConfig, LevenbergMarquardtConfig, GROUPS
from persona_lab.fitting.batched_jacobian import batch_jacobian, residual_fp64_batch
from persona_lab.fitting.levenberg_marquardt import StageCoordinates, observation_jacobian, square_root_blocks, normal_blocks, fit_stage
from persona_lab.fitting.losses import evaluate_objective, project_landmarks, _landmark_residuals
from persona_lab.fitting.monitoring import FitMeasurements
from persona_lab.fitting.parameters import FittingParameters, TensorState, ParameterSelection
from persona_lab.fitting.schur import BlockVector


class Data(SimpleNamespace):
    def __len__(self):
        return len(self.ids)


def problem(device, camera_mode, camera_model, focal_mode):
    n, m = 3, 68
    gen = torch.Generator().manual_seed(83)
    def rand(*shape):
        return torch.randn(shape, generator=gen, dtype=torch.float64).to(device)
    ref = rand(1)
    head = SimpleNamespace(identity_names=('i0','i1','i2','i3'), expression_names=('e0','e1','e2'),
                           joint_names=('neck','head','left_eye','right_eye'))
    model = SimpleNamespace(reference=ref, head=head)
    sizes = (ImageSize(512,512), ImageSize(640,480), ImageSize(512,512))
    cameras = tuple(
        WeakPerspectiveCamera(size, ref.new_zeros(3), ref.new_tensor([size.width/2,size.height/2]), ref.new_tensor(500.))
        if camera_model == 'weak-perspective' or camera_mode == 'known' and i == 1 else
        PerspectiveCamera(size, ref.new_zeros(3), ref.new_tensor([.01,.02,1.]),
                          ref.new_tensor([700.,700.]), ref.new_tensor([size.width/2,size.height/2]))
        for i,size in enumerate(sizes)
    )
    initial = TensorState(rand(4)*.1, rand(n,3)*.1, rand(n,4,3)*.01, ref.new_zeros(n,3), cameras)
    config = FitConfig(camera_mode=camera_mode, camera_model=camera_model, focal_mode=focal_mode,
                       initial_focal_factor=None if focal_mode == 'fixed' else 1.5,
                       identity_weight=1e-4, expression_weight=2e-4, pose_weight=3e-4, camera_weight=4e-4,
                       lm=LevenbergMarquardtConfig(fp64_square_root=True, jacobian_rows=32), batch_size=2)
    eligible = torch.ones((n,m), device=device, dtype=torch.bool)
    eligible[0,5] = False; eligible[2,12:17] = False
    weights = (rand(n,m).abs()+.1).masked_fill(~eligible,0)
    data = Data(ids=tuple(str(i) for i in range(n)), short_edges=ref.new_tensor([min(s.width,s.height) for s in sizes]),
                eligible=eligible, weights=weights, weight_sum=weights.sum(), counts=eligible.sum(1),
                gnm_indices=torch.arange(m, device=device).expand(n,-1), pixel_low=None)
    selection = ParameterSelection(('i0','i2','i3'), ('e0','e2'), ('head',), False, ('i1',), ('e1',), ('neck','left_eye','right_eye'))
    params = FittingParameters(model,data,initial,selection,config)
    template, basis_i, basis_e, basis_p = rand(m,3)*.03,rand(4,m,3)*.003,rand(3,m,3)*.003,rand(3,m,3)*.01
    def geometry(identity, expression, rotations, translation):
        return (template + torch.einsum('...i,imk->...mk',identity,basis_i)
                + torch.einsum('...i,imk->...mk',expression,basis_e)
                + torch.einsum('...i,imk->...mk',rotations[:,1],basis_p) + translation[:,None,:])
    params.__dict__['landmark_geometry'] = geometry
    with torch.no_grad():
        params.pose.add_(.015)
        if params.camera_delta is not None:
            params.camera_delta.add_(.02)
        state = params.state(0,n)
        data.pixels = project_landmarks(geometry(state.identity,state.expression,state.rotations,state.translation),state.cameras,data.ids) + rand(n,m,2)*.4
    return model,data,params


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('mode,kind,focal',[('known',None,'fixed'),('estimated','weak-perspective','fixed'),
    ('estimated','perspective','shared'),('estimated','perspective','per-observation')])
def test_staged_regularized_linearization(device,mode,kind,focal):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model,data,params=problem(device,mode,kind,focal)
    base=params.values()
    state=params.state(0,len(data))
    native=project_landmarks(params.landmark_geometry(state.identity,state.expression,state.rotations,state.translation),state.cameras,data.ids)
    expected,_=_landmark_residuals(native,data,0,len(data))
    actual,_=residual_fp64_batch(data,params,0,len(data),base)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-14)
    for groups in [('pose','camera'),('identity','pose','camera'),GROUPS,('identity',)]:
        layout=StageCoordinates.from_parameters(params,groups)
        position=layout.pack(base,model.reference,len(data))
        r,j=batch_jacobian(data,params,layout,position,0,len(data))
        for index in range(len(data)):
            rr,jj=observation_jacobian(data,params,layout,base,position,index)
            mask=data.eligible[index].repeat_interleave(2)
            torch.testing.assert_close(r[index,mask],rr,rtol=1e-11,atol=1e-13)
            torch.testing.assert_close(j[index,mask],jj,rtol=1e-10,atol=1e-12)
            assert not j[index,~mask].count_nonzero()
        system=square_root_blocks(data,params,layout,position)
        # The existing normal-block construction is an independent test oracle
        # for the global prior denominators, active columns and linear model.
        oracle=normal_blocks(model,data,params,layout,base,position)
        torch.testing.assert_close(system.gradient.identity,oracle.gradient.identity,rtol=1e-11,atol=1e-13)
        torch.testing.assert_close(system.gradient.local,oracle.gradient.local,rtol=1e-11,atol=1e-13)
        step=BlockVector(torch.full_like(position.identity,1e-5),torch.full_like(position.local,1e-5))
        assert system.predicted_reduction(step)==pytest.approx(oracle.predicted_reduction(step),rel=1e-10,abs=1e-15)
    with torch.no_grad():
        run=FitMeasurements(model.reference.device)
        values,stats=run.evaluate(model,data,params,least_squares=True)
        for stage in params.config.stages:
            prior=values.total
            values,stats,status,reason=fit_stage(model,data,params,replace(stage,iterations=2),stage.groups,values,stats,run,lambda *args:None)
            assert status in ('converged','budget_exhausted'), reason
            assert values.total <= prior
        assert len(run.square_root_stages)==3
        assert all(s['normal_equations_bytes']==0 for s in run.square_root_stages)
