import json
import pickle

import numpy as np
import pandas as pd
import pytest

from src import raw_frame_comparison as comparison
from src import modality_diagnostics as modality


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    rng = np.random.default_rng(15)
    matrix = rng.normal(size=(30, 128))
    manifest = pd.DataFrame({'participant_id': [str(i) for i in range(30)],
                             'severity_class': np.tile(np.arange(5), 6),
                             'gender': ['female', 'male', 'unknown']*10,
                             'split': ['train']*20+['validation']*5+['test']*5})
    settings = {'outer_folds': 2, 'inner_folds': 2, 'fold_seeds': [42], 'fit_seed': 42, 'C_candidates': [.1, 1.]}
    x, training = comparison.training_inputs(matrix, manifest)
    folds = modality.nested_folds(training, settings)
    baseline = [modality.run_job(x, training.severity_class.to_numpy(), fold, m, b, settings,
                                tmp_path/'baseline'/f"{fold['fold']}_{m}_{b}")
                for fold in folds for m, b in comparison.VARIANTS]
    monkeypatch.setattr(comparison, 'plot_summary', lambda *args: None)
    return matrix, manifest, x, training, folds, baseline, settings


def test_reserved_rows_never_enter_training_inputs():
    matrix = np.arange(30*128).reshape(30, 128).astype(float)
    manifest = pd.DataFrame({'participant_id': [str(i) for i in range(30)],
                             'severity_class': np.tile(np.arange(5), 6), 'gender': 'unknown',
                             'split': ['train']*20+['validation']*5+['test']*5})
    x, training = comparison.training_inputs(matrix, manifest)
    matrix[20:] += 999999
    manifest.loc[20:, 'severity_class'] = 999
    manifest.loc[20:, 'gender'] = 'changed'
    changed, changed_training = comparison.training_inputs(matrix, manifest)
    np.testing.assert_array_equal(x, changed)
    pd.testing.assert_frame_equal(training, changed_training)


def test_comparison_coverage_checkpoint_format_and_resume(synthetic, tmp_path, monkeypatch):
    _, _, x, training, folds, baseline, settings = synthetic
    out = tmp_path/'result'
    result = comparison.run_comparison(x, training, folds, baseline, settings, out)
    assert result['complete'] and result['total_logistic_fits'] == 40
    assert result['new_outer_jobs'] == 8 and not result['validation_test_predictions_created']
    for path in (out/'oof').glob('*/predictions.csv'):
        frame = pd.read_csv(path, dtype={'participant_id': str})
        assert set(frame.participant_id) == set(training.participant_id) and len(frame) == 20
    checkpoints = list((out/'jobs').rglob('model.pkl'))
    with checkpoints[0].open('rb') as handle:
        state = pickle.load(handle)
    assert state['feature_format'] == comparison.FORMAT
    assert state['purpose'] == comparison.PROTOCOL and len(state['feature_names']) == 128
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    def forbidden(*args):
        raise AssertionError('Completed jobs must not fit or predict')
    monkeypatch.setattr(comparison, 'run_raw_job', forbidden)
    resumed = comparison.run_comparison(x, training, folds, baseline, settings, out, resume=True)
    assert resumed['new_fits_this_invocation'] == 0 and resumed['reused_jobs_this_invocation'] == 8
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    delta = pd.read_csv(out/'paired_partition_differences.csv')
    np.testing.assert_array_equal(delta.macro_f1_raw_minus_cached, np.zeros(4))
    with pytest.raises(FileExistsError):
        comparison.run_comparison(x, training, folds, baseline, settings, out)
    changed = x.copy()
    changed[0, 0] += 1
    with pytest.raises(ValueError, match='changed'):
        comparison.run_comparison(changed, training, folds, baseline, settings, out, resume=True)


def test_interruption_resumes_and_detects_modified_checkpoint(synthetic, tmp_path, monkeypatch):
    _, _, x, training, folds, baseline, settings = synthetic
    original = comparison.run_raw_job
    count = []
    def interrupted(*args):
        count.append(1)
        if len(count) == 2:
            raise KeyboardInterrupt('synthetic interruption')
        return original(*args)
    monkeypatch.setattr(comparison, 'run_raw_job', interrupted)
    out = tmp_path/'result'
    with pytest.raises(KeyboardInterrupt):
        comparison.run_comparison(x, training, folds, baseline, settings, out)
    assert not (out/'completion.json').exists()
    monkeypatch.setattr(comparison, 'run_raw_job', original)
    result = comparison.run_comparison(x, training, folds, baseline, settings, out, resume=True)
    assert result['reused_jobs_this_invocation'] == 1 and result['new_fits_this_invocation'] == 35
    checkpoint = next((out/'jobs').rglob('model.pkl'))
    checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        comparison.run_comparison(x, training, folds, baseline, settings, out, resume=True)


def test_inner_selection_is_verified_from_saved_probabilities(synthetic):
    _, _, _, training, folds, baseline, settings = synthetic
    record = json.loads(json.dumps(baseline[0]))
    comparison.verify_selection(record, folds[0], training.severity_class.to_numpy(), settings)
    record['inner_candidates'][0]['inner_macro_f1'] += .1
    with pytest.raises(ValueError, match='Saved inner score'):
        comparison.verify_selection(record, folds[0], training.severity_class.to_numpy(), settings)


def test_changed_inputs_before_completion_do_not_claim_success(synthetic, tmp_path):
    _, _, x, training, folds, baseline, settings = synthetic
    def changed():
        raise ValueError('inputs changed')
    out = tmp_path/'result'
    with pytest.raises(ValueError, match='inputs changed'):
        comparison.run_comparison(x, training, folds, baseline, settings, out, verify_inputs=changed)
    assert not (out/'completion.json').exists()
