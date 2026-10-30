import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_limits

from src import baseline_averaging_accuracy as s


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def study():
    old_settings = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    old_settings.update(outer_folds=2, inner_folds=2, fold_seeds=[42])
    settings = json.loads(Path('configs/baseline_averaging_accuracy.json').read_text())
    settings.update(policies={'accuracy': 2, 'top1': 1, 'top3': 3}, expected_jobs=2, cpu_threads=1)
    y = np.repeat(np.arange(5), [24, 12, 12, 12, 8])
    x = {'semantic_av': np.random.default_rng(12).normal(size=(len(y), 20))}
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(len(y))], 'severity_class': y, 'gender': 'unknown'})
    folds = s.nested_folds(frame, old_settings); records = []
    for f in folds:
        rows = [{'candidate': {'family': 'logistic', 'feature': 'semantic_av', 'components': None,
                              'C': c, 'intervention': 'none'}, 'predictions': [0]*len(f['fit']),
                 **s.score(y[f['fit']], np.zeros(len(f['fit']), int))} for c in [.001, .01, .1]]
        records.append({**f, 'candidates': rows,
                        'policies': {'accuracy': {'probabilities': np.tile(np.eye(5)[0], (len(f['heldout']), 1)).tolist()}}})
    return (x, frame, old_settings, records, {}), settings, folds


def test_inner_ranking_has_stable_ties_and_excludes_dummy():
    rows = [{'candidate': {'family': 'logistic'}, 'correct': 40, 'macro_f1': .3},
            {'candidate': {'family': 'majority'}, 'correct': 99, 'macro_f1': .9},
            {'candidate': {'family': 'logistic'}, 'correct': 40, 'macro_f1': .3}]
    assert s.rank(rows) == [0, 2]
    settings = json.loads(Path('configs/baseline_averaging_accuracy.json').read_text())
    assert settings['primary'] == 'accuracy' and settings['policies']['accuracy'] == 5


def test_outer_data_cannot_change_ensemble_members_or_models(study, tmp_path):
    inputs, settings, folds = study; x, frame, old_settings, records, _ = inputs; fold = folds[0]
    first = s.job(x, frame, fold, records[0], old_settings, settings, tmp_path/'first')
    with (tmp_path/'first'/'accuracy.pkl').open('rb') as f:
        state = pickle.load(f)
    expected = np.mean([m['estimator'].predict_proba(x['semantic_av'][fold['heldout']]) for m in state['members']], axis=0)
    np.testing.assert_allclose(expected, first['selected']['accuracy']['probabilities'])
    x['semantic_av'][fold['heldout']] += 100
    frame.loc[fold['heldout'], 'severity_class'] = (frame.loc[fold['heldout'], 'severity_class']+1)%5
    records[0]['policies'] = {'unused_outer_results': 999}
    second = s.job(x, frame, fold, records[0], old_settings, settings, tmp_path/'second')
    assert first['ranking'] == second['ranking']
    for name in settings['policies']:
        assert first['selected'][name]['checkpoint_sha256'] == second['selected'][name]['checkpoint_sha256']


def test_resume_does_not_fit_predict_or_deserialize(study, tmp_path, monkeypatch):
    inputs, settings, _ = study; out = tmp_path/'study'
    assert s.run(inputs, settings, out)['complete']
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out/'jobs').rglob('*') if p.is_file()}
    def forbidden(*a, **kw):
        raise AssertionError('No fitting or prediction in completed resume')
    monkeypatch.setattr(s, 'fit_members', forbidden); monkeypatch.setattr(s, 'predict_state', forbidden); monkeypatch.setattr(s.pickle, 'load', forbidden)
    assert s.run(inputs, settings, out, resume=True)['new_fits'] == 0
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == v for p,v in before.items())
    path = next((out/'jobs').glob('*/*.pkl')); path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError, match='Checkpoint changed'):
        s.run(inputs, settings, out, resume=True)
