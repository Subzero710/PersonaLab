"""End-to-end official-asset pilot for the previously failing seed 43."""
import os
from pathlib import Path

import pytest
import torch

from persona_lab.benchmark import fitting_study, focal_scalability, scalability_data
from persona_lab.benchmark.scalability_cases import case_matrix
from persona_lab.benchmark.storage import read_json
from persona_lab.fitting import fit_identity


@pytest.fixture(scope='module')
def native_reference(tmp_path_factory):
    model = Path(os.environ.get('PERSONALAB_GNM_MODEL_FILE', '/cache/gnm/v3_0/gnm_head.npz'))
    if not model.is_file():
        pytest.skip('Pinned official GNM v3.0 Head asset unavailable')
    path = tmp_path_factory.mktemp('v66-reference') / 'seed-43'
    reference = scalability_data.generate_parent(path, 43, model, dtype='float64')
    return model, path, reference


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_seed43_native_fp64_pilot(native_reference, tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model, archive, reference = native_reference
    case = case_matrix(seeds=(43,), counts=(50,), devices=(device,), batch_size=32,
                       primary_only=True, dtype='float64')[0]
    config = fitting_study._scalability_config(32, fp64_square_root=True)
    output = tmp_path / 'case.result.json'
    job = {'case': case, 'config': config.to_dict(), 'model_file': str(model), 'dataset': str(archive),
           'reference': reference, 'spectra': True, 'result': str(output)}
    def blind_fit(model, observations, config, **options):
        assert options['initial'] is None
        assert model.reference.dtype == torch.float64
        assert all(o.known_camera is None and o.camera_initialization is None and o.pixel_low is None
                   and o.landmarks.dtype == torch.float64 and o.landmarks.device.type == device for o in observations)
        return fit_identity(model, observations, config, **options)
    assert focal_scalability.run_worker(job, blind_fit, fitting_study._score_scalability_fit) == 0
    row = read_json(output)
    assert row['failure'] is None and row['dataset_verified']
    assert row['actual_dtype'] == 'float64' and row['actual_execution_device'].split(':')[0] == device
    assert row['pixel_representation'] == 'raster-fp64'
    assert row['termination_status'] == 'converged'
    assert row['recovery']['exact_compatible_recovery']
    assert row['reference_precision']['status'] == 'evaluated'
    assert not row['reference_precision']['used_for_fitting']
    assert row['reference_precision']['truth_linearization']['scaled_projected_gradient_norm'] <= 1e-12
    assert row['optimization']['final_gradient']['status'] == 'evaluated'
    assert row['optimization']['final_gradient']['scaled_projected_gradient_norm'] <= 1e-12
    assert row['optimizer_runtime']['normal_system']['normal_equations_bytes'] == 0
    assert row['optimizer_runtime']['normal_system']['dtype'] == 'torch.float64'
    assert row['final']['fixed_identity_preserved']
    assert all(v['status'] == 'evaluated' for v in row['observability_status'].values())
    assert row['final_identity_observability']['after_all_nuisance']['rank'] == 170


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_fp64_regularized_stages(native_reference, device):
    """Official GNM integration: staged priors and actual CPU/CUDA execution."""
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from dataclasses import replace
    import numpy as np
    from persona_lab.fitting import FitConfig, LevenbergMarquardtConfig
    from persona_lab.fitting.results import FAILED_STATUSES

    path, archive, reference = native_reference
    parent = scalability_data.load_parent(archive, 43, reference)
    dataset, subject, _ = scalability_data.verified_subset(parent, 9)
    model = focal_scalability.load_benchmark_model(path, dtype='float64', device=device)
    observations = dataset.fitting_observations(subject, dtype='float64', device=device,
                                                include_known_cameras=False)
    config = FitConfig(camera_mode='estimated', camera_model='perspective',
                       focal_mode='per-observation', initial_focal_factor=1.5,
                       batch_size=4, lm=LevenbergMarquardtConfig(fp64_square_root=True, jacobian_rows=32),
                       stages=tuple(replace(stage, iterations=3) for stage in FitConfig().stages))
    result = fit_identity(model, observations, config, trace_stages=True)
    assert result.status not in FAILED_STATUSES, result.reason
    assert result.final is not None and result.dtype == 'float64'
    assert result.device.split(':')[0] == device
    assert result.optimization.final_objective.total < result.optimization.initial_objective.total
    assert all(stage.final_loss <= stage.initial_loss for stage in result.stages)
    assert len(result.runtime['square_root_stages']) == 3
    assert [row['shared_parameters'] for row in result.runtime['square_root_stages']] == [0, 170, 170]
    assert [row['local_parameters_per_observation'] for row in result.runtime['square_root_stages']] == [7, 7, 31]
    assert [row['active_prior_coordinates'] for row in result.runtime['square_root_stages']] == [0, 170, 170+9*24]
    assert all(row['normal_equations_bytes'] == 0 for row in result.runtime['square_root_stages'])
    first, second, third = result.stage_diagnostics
    assert np.array_equal(first.before.state.identity, first.after.state.identity)
    assert np.array_equal(second.before.state.expression, second.after.state.expression)
    assert third.after.state is not None
    fixed = [i for i, name in enumerate(model.metadata['identity_names']) if name in result.selection.fixed_identity]
    assert np.array_equal(result.initial.identity[fixed], result.final.identity[fixed])


def test_fp64_real_cli_selection():
    from persona_lab.fit import _parser, _config
    args = _parser().parse_args(['--input','datasets/studio_light.zip','--output','outputs/test',
        '--dtype','float64','--fp64-qr','--jacobian-rows','32','--camera','perspective','--focal-mode','per-observation',
        '--initial-focal-factor','1.5','--max-images','50','--require-images','50','--require-convergence'])
    config = _config(args)
    assert config.lm.fp64_square_root and config.lm.arithmetic_dtype == 'float64'
    assert config.identity_weight > 0 and config.expression_weight > 0
    assert config.lm.jacobian_rows == 32
    assert args.require_images == 50 and args.require_convergence
    args.dtype = 'float32'
    with pytest.raises(ValueError, match='FP64 square-root requires --dtype float64'):
        _config(args)
