import json
import pickle

import numpy as np
import pandas as pd
import pytest

from src import modality_diagnostics as modality
from src.reference_inference import sha256


@pytest.fixture
def synthetic(monkeypatch):
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(30, 128))
    manifest = pd.DataFrame({'participant_id': [str(i) for i in range(30)],
                             'severity_class': np.tile(np.arange(5), 6),
                             'gender': ['female', 'male', 'unknown']*10,
                             'split': ['train']*20+['validation']*5+['test']*5})
    settings = {'outer_folds': 2, 'inner_folds': 2, 'fold_seeds': [42], 'fit_seed': 42, 'C_candidates': [.1, 1.]}
    monkeypatch.setattr(modality, 'plot_summary', lambda *args: None)
    return matrix, manifest, settings


def test_nested_folds_are_disjoint_and_inner_folds_exclude_outer_holdout(synthetic):
    _, manifest, settings = synthetic
    training = manifest[manifest.split == 'train'].reset_index(drop=True)
    settings = {**settings, 'fold_seeds': [42, 137]}
    folds = modality.nested_folds(training, settings)
    for seed in settings['fold_seeds']:
        counts = np.zeros(len(training), int)
        for fold in [f for f in folds if f['seed'] == seed]:
            fit, heldout = set(fold['fit']), set(fold['heldout'])
            assert fit.isdisjoint(heldout) and fit | heldout == set(range(len(training)))
            inner_counts = dict.fromkeys(fit, 0)
            for inner in fold['inner']:
                a, b = set(inner['fit']), set(inner['heldout'])
                assert a.isdisjoint(b) and a | b == fit
                assert (a | b).isdisjoint(heldout)
                for index in b:
                    inner_counts[index] += 1
            assert set(inner_counts.values()) == {1}
            counts[fold['heldout']] += 1
        assert (counts == 1).all()


def test_modalities_preserve_summary_feature_order():
    np.testing.assert_array_equal(modality.modality_columns('A'), np.arange(78))
    np.testing.assert_array_equal(modality.modality_columns('V'), np.arange(78, 128))
    np.testing.assert_array_equal(modality.modality_columns('AV'), np.arange(128))


def test_outer_labels_and_features_cannot_change_regularization_or_fitted_model(synthetic, tmp_path):
    matrix, manifest, settings = synthetic
    training = manifest[manifest.split == 'train'].reset_index(drop=True)
    x, y = matrix[:20], training.severity_class.to_numpy()
    fold = modality.nested_folds(training, settings)[0]
    first = modality.run_job(x, y, fold, 'AV', True, settings, tmp_path/'first')
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[fold['heldout']] += 1000
    changed_y[fold['heldout']] = (changed_y[fold['heldout']]+2) % 5
    second = modality.run_job(changed_x, changed_y, fold, 'AV', True, settings, tmp_path/'second')
    assert first['selected_C'] == second['selected_C'] and first['inner_candidates'] == second['inner_candidates']
    with (tmp_path/'first/model.pkl').open('rb') as f:
        original = pickle.load(f)['estimator']
    with (tmp_path/'second/model.pkl').open('rb') as f:
        changed = pickle.load(f)['estimator']
    np.testing.assert_array_equal(original.named_steps['logisticregression'].coef_, changed.named_steps['logisticregression'].coef_)
    np.testing.assert_array_equal(original.named_steps['standardscaler'].mean_, changed.named_steps['standardscaler'].mean_)
    assert first['fits'] == len(settings['C_candidates'])*settings['inner_folds']+1


def test_complete_coverage_reserved_data_invariance_and_resume(synthetic, tmp_path, monkeypatch):
    matrix, manifest, settings = synthetic
    out = tmp_path/'result'
    result = modality.run_experiment(matrix, manifest, settings, out)
    assert result['complete'] and result['total_logistic_fits'] == 60
    assert result['outer_jobs'] == 12 and result['all_dataset_participants'] == 30
    assert result['training_participants'] == 20 and not result['validation_test_predictions_created']
    train_ids = set(manifest.loc[manifest.split == 'train', 'participant_id'])
    for path in (out/'oof').glob('*/predictions.csv'):
        table = pd.read_csv(path, dtype={'participant_id': str})
        assert set(table.participant_id) == train_ids and not table.participant_id.duplicated().any()
        assert set(table.split) == {'train_nested_oof'}
    before = {p: (sha256(p), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Resume must not fit or predict completed jobs')
    monkeypatch.setattr(modality, 'run_job', forbidden)
    changed = matrix.copy()
    changed[20:] *= 1000
    changed_manifest = manifest.copy()
    changed_manifest.loc[20:, 'severity_class'] = 999
    changed_manifest.loc[20:, 'gender'] = 'synthetic-changed'
    resumed = modality.run_experiment(changed, changed_manifest, settings, out, resume=True)
    assert resumed['new_logistic_fits_this_invocation'] == 0 and resumed['reused_jobs_this_invocation'] == 12
    assert before == {p: (sha256(p), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        modality.run_experiment(matrix, manifest, settings, out)
    changed[0, 0] += 1
    with pytest.raises(ValueError, match='changed'):
        modality.run_experiment(changed, manifest, settings, out, resume=True)


def test_interrupted_diagnostic_reuses_finished_jobs(synthetic, tmp_path, monkeypatch):
    matrix, manifest, settings = synthetic
    original, calls = modality.run_job, []
    def interrupted(*args):
        calls.append(args[3:5])
        if len(calls) == 2:
            raise KeyboardInterrupt('Synthetic interruption')
        return original(*args)
    monkeypatch.setattr(modality, 'run_job', interrupted)
    out = tmp_path/'result'
    with pytest.raises(KeyboardInterrupt):
        modality.run_experiment(matrix, manifest, settings, out)
    assert not (out/'completion.json').exists()
    assert len(list((out/'jobs').rglob('result.json'))) == 1
    monkeypatch.setattr(modality, 'run_job', original)
    result = modality.run_experiment(matrix, manifest, settings, out, resume=True)
    assert result['reused_jobs_this_invocation'] == 1 and result['new_logistic_fits_this_invocation'] == 55


@pytest.mark.parametrize('filename', ['model.pkl', 'result.json'])
def test_changed_completed_jobs_are_rejected(synthetic, tmp_path, filename):
    matrix, manifest, settings = synthetic
    modality.run_experiment(matrix, manifest, settings, tmp_path/'result')
    path = next((tmp_path/'result/jobs').rglob(filename))
    if filename == 'model.pkl':
        path.write_bytes(b'changed')
    else:
        data = json.loads(path.read_text())
        data['record']['selected_C'] = 999
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='changed|integrity'):
        modality.run_experiment(matrix, manifest, settings, tmp_path/'result', resume=True)


def test_impossible_inner_folds_fail_before_output(synthetic, tmp_path):
    matrix, manifest, settings = synthetic
    with pytest.raises(ValueError, match='classes must support'):
        modality.run_experiment(matrix, manifest, {**settings, 'inner_folds': 3}, tmp_path/'result')
    assert not (tmp_path/'result').exists()


def test_changed_input_callback_prevents_completion(synthetic, tmp_path):
    matrix, manifest, settings = synthetic
    def changed():
        raise ValueError('Dataset changed during synthetic run')
    with pytest.raises(ValueError, match='Dataset changed'):
        modality.run_experiment(matrix, manifest, settings, tmp_path/'result', verify_inputs=changed)
    assert not (tmp_path/'result/completion.json').exists()
