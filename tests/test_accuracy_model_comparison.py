import copy
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from src import accuracy_model_comparison as experiment
from src.batch_inference import load_record, save_record
from src.modality_diagnostics import nested_folds


@pytest.fixture
def synthetic(monkeypatch):
    settings = json.loads(open('configs/accuracy_model_comparison.json').read())
    settings.update(feature_inputs=['cached_visual'], outer_folds=2, inner_folds=2,
                    fold_seeds=[42], bootstrap_repetitions=30)
    for options in settings['models'].values():
        options['C_candidates'] = [.1, 1.]
    settings['budget'] = {'cpu_threads': 1, 'outer_jobs': 4, 'total_estimator_fits': 20}
    training = pd.DataFrame({'participant_id': [str(i) for i in range(30)],
                             'severity_class': np.tile(np.arange(5), 6),
                             'gender': ['male', 'female', 'unknown']*10})
    matrices = {'cached_visual': np.random.default_rng(51).normal(size=(30, 50))}
    monkeypatch.setattr(experiment, 'plot_summary', lambda *args: None)
    return matrices, training, settings


def test_accuracy_is_primary_and_ties_use_macro_f1_then_smaller_c():
    candidates = [{'C': 1, 'inner_correct': 15, 'inner_macro_f1': .3},
                  {'C': .1, 'inner_correct': 14, 'inner_macro_f1': .9},
                  {'C': 10, 'inner_correct': 15, 'inner_macro_f1': .4},
                  {'C': .01, 'inner_correct': 15, 'inner_macro_f1': .4}]
    assert experiment.choose_candidate(candidates)['C'] == .01


def test_reserved_features_labels_and_groups_do_not_enter_training():
    cached = np.random.default_rng(5).normal(size=(30, 128))
    raw = cached.copy()
    raw[:, :78] *= 2
    manifest = pd.DataFrame({'participant_id': [str(i) for i in range(30)],
                             'severity_class': np.tile(np.arange(5), 6), 'gender': 'unknown',
                             'split': ['train']*20+['validation']*5+['test']*5})
    inputs, training = experiment.select_training_features(cached, raw, manifest)
    cached[20:] += 999999
    raw[20:] -= 12345
    manifest.loc[20:, ['severity_class', 'gender']] = [999, 'changed']
    changed, same = experiment.select_training_features(cached, raw, manifest)
    for feature in inputs:
        np.testing.assert_array_equal(inputs[feature], changed[feature])
    pd.testing.assert_frame_equal(training, same)


@pytest.mark.parametrize('family', experiment.FAMILIES)
def test_outer_heldout_data_cannot_affect_tuning_or_fitted_model(synthetic, tmp_path, family):
    matrices, training, settings = synthetic
    x, y = matrices['cached_visual'], training.severity_class.to_numpy()
    fold = nested_folds(training, settings)[0]
    first = experiment.run_job(x, y, fold, family, 'cached_visual', settings, tmp_path/'first')
    changed_x, changed_y = x.copy(), y.copy()
    changed_x[fold['heldout']] += 5000
    changed_y[fold['heldout']] = (changed_y[fold['heldout']]+1) % 5
    second = experiment.run_job(changed_x, changed_y, fold, family, 'cached_visual', settings, tmp_path/'second')
    assert first['inner_candidates'] == second['inner_candidates']
    assert first['selected_C'] == second['selected_C']
    assert first['model_sha256'] == second['model_sha256']
    with (tmp_path/'first/model.pkl').open('rb') as handle:
        estimator = pickle.load(handle)['estimator']
    np.testing.assert_allclose(estimator.named_steps['standardscaler'].mean_, x[fold['fit']].mean(0))


def test_nonlinear_synthetic_smoke_and_score_contract(synthetic):
    _, _, settings = synthetic
    def rings(count, offset):
        angles = 2*np.pi*(np.arange(count)+offset)/count
        x = np.concatenate([radius*np.column_stack([np.cos(angles), np.sin(angles)]) for radius in range(1, 6)])
        return x, np.repeat(np.arange(5), count)
    x, y = rings(80, 0)
    heldout, truth = rings(40, .25)
    svm = experiment.fit_checked(x, y, 'rbf_svm', 100, settings)
    logistic = experiment.fit_checked(x, y, 'logistic_regression', 1, settings)
    predicted, scores = experiment.decisions(svm, heldout)
    linear, _ = experiment.decisions(logistic, heldout)
    assert np.mean(predicted == truth) > .9 and np.mean(linear == truth) < .4
    assert not svm.named_steps['svc'].probA_.size and not svm.named_steps['svc'].probB_.size
    assert not hasattr(svm, 'predict_proba')
    assert not np.allclose(scores.sum(1), 1)


def test_pilot_complete_resume_and_training_oof_outputs(synthetic, tmp_path, monkeypatch):
    matrices, training, settings = synthetic
    out = tmp_path/'experiment'
    pilot = experiment.run_experiment(matrices, training, settings, out, max_new_jobs=1)
    assert not pilot['complete'] and pilot['completed_fits'] == 5
    assert not (out/'summary.csv').exists()
    completed = experiment.run_experiment(matrices, training, settings, out, resume=True)
    assert completed['complete'] and completed['completed_fits'] == 20
    assert completed['reused_jobs_this_invocation'] == 1 and completed['new_fits_this_invocation'] == 15
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    predictions = {p: p.read_bytes() for p in (out/'oof').glob('*/predictions.csv')}
    def forbidden(*args, **kwargs):
        raise AssertionError('Resume must not fit, deserialize or create new predictions')
    monkeypatch.setattr(experiment, 'run_job', forbidden)
    monkeypatch.setattr(experiment.pickle, 'load', forbidden)
    monkeypatch.setattr(experiment, 'decisions', forbidden)
    resumed = experiment.run_experiment(matrices, training, settings, out, resume=True)
    assert resumed['new_fits_this_invocation'] == 0 and resumed['reused_jobs_this_invocation'] == 4
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    assert predictions == {p: p.read_bytes() for p in (out/'oof').glob('*/predictions.csv')}
    for path in predictions:
        frame = pd.read_csv(path, dtype={'participant_id': str})
        assert frame.participant_id.tolist() == training.participant_id.tolist()
        assert frame.split.eq('train_nested_oof').all()
        assert not any(f'p{i}' in frame for i in range(5))
        if not frame.model.eq('majority').all():
            np.testing.assert_array_equal(frame.predicted_label, frame[[f'decision_score_{i}' for i in range(5)]].to_numpy().argmax(1))
    assert len(pd.read_csv(out/'paired_comparisons.csv')) == 9
    with pytest.raises(FileExistsError):
        experiment.run_experiment(matrices, training, settings, out)
    changed = {'cached_visual': matrices['cached_visual']+1}
    with pytest.raises(ValueError, match='changed'):
        experiment.run_experiment(changed, training, settings, out, resume=True)


def test_interruption_preserves_completed_job(synthetic, tmp_path, monkeypatch):
    matrices, training, settings = synthetic
    original, calls = experiment.run_job, []
    def interrupted(*args):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt('synthetic interruption')
        return original(*args)
    monkeypatch.setattr(experiment, 'run_job', interrupted)
    out = tmp_path/'experiment'
    with pytest.raises(KeyboardInterrupt):
        experiment.run_experiment(matrices, training, settings, out)
    assert not (out/'completion.json').exists()
    monkeypatch.setattr(experiment, 'run_job', original)
    result = experiment.run_experiment(matrices, training, settings, out, resume=True)
    assert result['reused_jobs_this_invocation'] == 1 and result['new_jobs_this_invocation'] == 3


@pytest.mark.parametrize('corruption', ['checkpoint', 'record_checksum', 'selection'])
def test_resume_rejects_changed_saved_state(synthetic, tmp_path, corruption):
    matrices, training, settings = synthetic
    out = tmp_path/'experiment'
    experiment.run_experiment(matrices, training, settings, out, max_new_jobs=1)
    path = next((out/'jobs').rglob('result.json'))
    context_hash = json.loads((out/'protocol.json').read_text())['context_hash']
    if corruption == 'checkpoint':
        model = path.with_name('model.pkl')
        model.write_bytes(model.read_bytes()+b'changed')
    elif corruption == 'record_checksum':
        record = json.loads(path.read_text())
        record['record']['selected_C'] = 999
        path.write_text(json.dumps(record))
    else:
        record = load_record(path, context_hash)
        record['selected_C'] = 999
        save_record(path, record, context_hash)
    with pytest.raises(ValueError):
        experiment.run_experiment(matrices, training, settings, out, resume=True)


def test_changed_inputs_prevent_completion(synthetic, tmp_path):
    matrices, training, settings = synthetic
    def changed():
        raise ValueError('inputs changed')
    out = tmp_path/'experiment'
    with pytest.raises(ValueError, match='inputs changed'):
        experiment.run_experiment(matrices, training, settings, out, verify_inputs=changed)
    assert not (out/'completion.json').exists() and not (out/'summary.csv').exists()


def test_declared_budget_and_svm_contract_are_enforced(synthetic, tmp_path):
    matrices, training, settings = synthetic
    for mutation in ['budget', 'probability', 'ties']:
        changed = copy.deepcopy(settings)
        if mutation == 'budget':
            changed['budget']['total_estimator_fits'] += 1
        else:
            changed['models']['rbf_svm']['probability' if mutation == 'probability' else 'break_ties'] = mutation == 'probability'
        with pytest.raises(ValueError):
            experiment.run_experiment(matrices, training, changed, tmp_path/mutation)
        assert not (tmp_path/mutation).exists()
