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
