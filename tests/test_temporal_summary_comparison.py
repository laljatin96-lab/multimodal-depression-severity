import json
import pickle

import numpy as np
import pandas as pd
import pytest

from src import temporal_summary_comparison as study
from src.batch_inference import load_record, save_record
from src.modality_diagnostics import nested_folds
from src.temporal_summaries import DIMENSIONS, feature_names, prepare_all, summarize_modality, summarize_participant


def test_missing_bins_are_not_bridged_and_invalid_values_are_ignored():
    x = np.array([0, 1, np.nan, 100, 101.])[:, None]
    vector, quality = summarize_modality(x, np.array([True, True, False, True, True]))
    np.testing.assert_allclose(vector, [50.5, np.std([0, 1, 100, 101]), 50.5, 99.5, 100.4, 1, 1])
    assert quality == {'observed_bins': 4, 'adjacent_pairs': 2, 'undefined_lag_columns': 0}
    x[2] = -1e8
    np.testing.assert_array_equal(vector, summarize_modality(x, np.array([True, True, False, True, True]))[0])


def test_dynamic_statistics_distinguish_same_marginal_distribution():
    a = np.array([0, 1, 2, 3, 4, 5.])[:, None]
    b = a[[0, 5, 1, 4, 2, 3]]
    first, _ = summarize_modality(a, np.ones(6, bool))
    second, _ = summarize_modality(b, np.ones(6, bool))
    np.testing.assert_array_equal(first[:5], second[:5])
    assert first[5] < second[5] and first[6] > second[6]


def test_constant_and_unsupported_lags_have_explicit_support():
    values = np.ones((5, 3))
    constant, quality = summarize_modality(values, np.ones(5, bool))
    np.testing.assert_array_equal(constant[-6:], np.zeros(6))
    assert quality['undefined_lag_columns'] == 3
    result, quality = summarize_modality(values, np.array([True, False, True, False, True]))
    assert quality['adjacent_pairs'] == 0
    np.testing.assert_array_equal(result[-6:], np.zeros(6))


def test_schema_grid_and_all_participant_coverage():
    rng = np.random.default_rng(4)
    seq = {'audio': rng.normal(size=(8, 39)), 'visual': rng.normal(size=(8, 25)),
           'audio_mask': np.ones(8, bool), 'visual_mask': np.ones(8, bool), 'timestamps': np.arange(8)*.5}
    matrices, quality = prepare_all({'one': seq, 'two': seq}, ['one', 'two'])
    for key, dimension in DIMENSIONS.items():
        assert matrices[key].shape == (2, dimension) and len(feature_names()[key]) == dimension
    assert len(quality) == 4
    with pytest.raises(ValueError, match='exactly once'):
        prepare_all({'one': seq}, ['one', 'two'])
    seq['timestamps'][3] += .1
    with pytest.raises(ValueError, match='grid'):
        summarize_participant(seq)


def test_selector_constant_features_perfect_signal_ties_and_serialization():
    y = np.tile(np.arange(5), 5)
    x = np.column_stack([np.ones(len(y)), y, y, np.random.default_rng(9).normal(size=len(y))])
    selector = study.FoldAnovaSelector(k=1).fit(x, y)
    np.testing.assert_array_equal(selector.selected_, [1])
    assert selector.__class__.__module__ == 'src.fold_feature_selection'
    restored = pickle.loads(pickle.dumps(selector))
    np.testing.assert_array_equal(restored.transform(x), x[:, [1]])
    assert len(study.FoldAnovaSelector(k=20).fit(x, y).selected_) == 3
    with pytest.raises(ValueError, match='varying'):
        study.FoldAnovaSelector().fit(np.ones((25, 3)), y)


@pytest.fixture
def synthetic(monkeypatch):
    settings = json.loads(open('configs/temporal_summary_comparison.json').read())
    settings.update(outer_folds=2, inner_folds=2, fold_seeds=[42], C_candidates=[.01, .1],
                    expected_jobs=8, expected_fits=40, bootstrap_repetitions=30, cpu_threads=1)
    manifest = pd.DataFrame({'participant_id': [str(i) for i in range(40)], 'severity_class': np.tile(np.arange(5), 8),
                             'gender': ['female', 'male']*20, 'split': ['train']*30+['validation']*5+['test']*5})
    rng = np.random.default_rng(63)
    matrices = {key: rng.normal(size=(40, d)) for key, d in DIMENSIONS.items()}
    monkeypatch.setattr(study, 'plot_summary', lambda *args: None)
    return matrices, manifest, settings


def test_reserved_rows_do_not_affect_training_inputs(synthetic):
    matrices, manifest, _ = synthetic
    before, training = study.select_training(matrices, manifest)
    for x in matrices.values():
        x[30:] += 1e6
    manifest.loc[30:, 'severity_class'] = 999
    manifest.loc[30:, 'gender'] = 'changed'
    after, same = study.select_training(matrices, manifest)
    pd.testing.assert_frame_equal(training, same)
    for feature in before:
        np.testing.assert_array_equal(before[feature], after[feature])


def test_outer_labels_and_features_cannot_change_fitted_selection(synthetic, tmp_path):
    matrices, manifest, settings = synthetic
    selected, training = study.select_training(matrices, manifest)
    x, y = selected['temporal_visual'], training.severity_class.to_numpy(copy=True)
    fold = nested_folds(training, settings)[0]
    before = study.run_job(x, y, fold, 'temporal_visual', settings, tmp_path/'first')
    x[fold['heldout']] += 999
    y[fold['heldout']] = (y[fold['heldout']]+1) % 5
    after = study.run_job(x, y, fold, 'temporal_visual', settings, tmp_path/'second')
    for key in ['selected_C', 'inner_candidates', 'inner_preprocessing', 'outer_preprocessing', 'model_sha256']:
        assert before[key] == after[key]
    with (tmp_path/'first/model.pkl').open('rb') as handle:
        state = pickle.load(handle)
    assert len(state['selected_feature_names']) == 20
    np.testing.assert_allclose(state['estimator'].named_steps['standardscaler'].mean_, x[fold['fit']].mean(0))


def test_pilot_complete_resume_and_oof_coverage(synthetic, tmp_path, monkeypatch):
    matrices, manifest, settings = synthetic
    out = tmp_path/'experiment'
    args = (matrices, manifest, [], {}, settings, out)
    pilot = study.run_experiment(*args, max_new_jobs=1)
    assert not pilot['complete'] and pilot['completed_fits'] == 5
    assert not (out/'summary.csv').exists()
    complete = study.run_experiment(*args, resume=True)
    assert complete['complete'] and complete['completed_fits'] == 40 and complete['reused_jobs'] == 1
    jobs = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    outputs = {p: p.read_bytes() for p in (out/'oof').glob('*/predictions.csv')}
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed resume must not fit, predict or deserialize')
    monkeypatch.setattr(study, 'run_job', forbidden)
    monkeypatch.setattr(study, 'five_probabilities', forbidden)
    monkeypatch.setattr(study.pickle, 'load', forbidden)
    resumed = study.run_experiment(*args, resume=True)
    assert resumed['new_fits'] == 0 and resumed['reused_jobs'] == 8
    assert jobs == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in jobs}
    assert outputs == {p: p.read_bytes() for p in outputs}
    for path in outputs:
        frame = pd.read_csv(path, dtype={'participant_id': str})
        assert frame.participant_id.tolist() == manifest.loc[manifest.split.eq('train'), 'participant_id'].tolist()
        assert frame.split.eq('train_nested_oof').all()
    assert len(pd.read_csv(out/'paired_comparisons.csv')) == 18
    with pytest.raises(FileExistsError):
        study.run_experiment(*args)
    matrices['cached_visual'][0, 0] += 1
    with pytest.raises(ValueError, match='changed'):
        study.run_experiment(*args, resume=True)


def test_interrupted_run_reuses_completed_jobs(synthetic, tmp_path, monkeypatch):
    matrices, manifest, settings = synthetic
    original, calls = study.run_job, []
    def interrupted(*args):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt('synthetic interruption')
        return original(*args)
    monkeypatch.setattr(study, 'run_job', interrupted)
    args = (matrices, manifest, [], {}, settings, tmp_path/'out')
    with pytest.raises(KeyboardInterrupt):
        study.run_experiment(*args)
    monkeypatch.setattr(study, 'run_job', original)
    result = study.run_experiment(*args, resume=True)
    assert result['new_jobs'] == 7 and result['reused_jobs'] == 1


@pytest.mark.parametrize('corruption', ['model', 'selection', 'matrix'])
def test_saved_corruption_is_rejected(synthetic, tmp_path, corruption):
    matrices, manifest, settings = synthetic
    out = tmp_path/'out'
    args = (matrices, manifest, [], {}, settings, out)
    study.run_experiment(*args, max_new_jobs=1)
    record_path = next((out/'jobs').rglob('result.json'))
    if corruption == 'model':
        path = record_path.with_name('model.pkl')
        path.write_bytes(path.read_bytes()+b'changed')
    elif corruption == 'selection':
        context_hash = json.loads((out/'protocol.json').read_text())['context_hash']
        record = load_record(record_path, context_hash)
        record['outer_preprocessing']['selected_columns'] = [999]
        save_record(record_path, record, context_hash)
    else:
        path = out/'participant_features.npz'
        with np.load(path, allow_pickle=False) as data:
            arrays = dict(data)
        arrays['cached_visual'][0, 0] += 1
        np.savez_compressed(path, **arrays)
    with pytest.raises((ValueError, AssertionError)):
        study.run_experiment(*args, resume=True)


def test_changed_source_before_completion_and_wrong_budget_fail(synthetic, tmp_path):
    matrices, manifest, settings = synthetic
    def changed():
        raise ValueError('source changed')
    with pytest.raises(ValueError, match='source changed'):
        study.run_experiment(matrices, manifest, [], {}, settings, tmp_path/'out', verify_inputs=changed)
    assert not (tmp_path/'out/completion.json').exists()
    settings['expected_fits'] += 1
    with pytest.raises(ValueError, match='budget'):
        study.run_experiment(matrices, manifest, [], {}, settings, tmp_path/'wrong_budget')
