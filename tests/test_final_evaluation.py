import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import final_evaluation as final
from src.common import digest
from src.crossvalidated_baselines import PROTOCOL as CV_PROTOCOL, five_probabilities, make_estimator
from src.evaluate import write_report
from src.exploratory_baselines import pool_modalities


def test_frozen_plan_detects_input_and_plan_changes_and_preserves_existing(tmp_path):
    artifact = tmp_path/'model.txt'
    artifact.write_text('initial')
    plan = {'protocol': final.PROTOCOL, 'output': str(tmp_path/'output'),
            'bound_files': {str(artifact): final.sha256(artifact)}}
    path = tmp_path/'plan.json'
    final.seal_plan(path, plan)
    assert final.verify_plan(path) == plan
    with pytest.raises(FileExistsError, match='Preserving'):
        final.seal_plan(path, plan)
    artifact.write_text('changed')
    with pytest.raises(ValueError, match='Frozen input changed'):
        final.verify_plan(path)
    artifact.write_text('initial')
    envelope = json.loads(path.read_text())
    envelope['plan']['output'] = 'different'
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match='integrity mismatch'):
        final.verify_plan(path)


def test_frozen_evaluation_only_loads_models_and_cannot_repeat(tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(25)],
                          'split': ['train']*15+['validation']*5+['test']*5,
                          'gender': ['female', 'male', 'unknown', 'female', 'male']*5,
                          'severity_class': list(range(5))*5})
    sequences = {pid: {'audio': rng.normal(size=(6, 2)).astype('float32'),
                       'visual': rng.normal(size=(6, 1)).astype('float32'),
                       'audio_mask': np.ones(6, bool), 'visual_mask': np.ones(6, bool),
                       'timestamps': np.arange(6)*.5} for pid in frame.participant_id}
    matrix = np.stack([pool_modalities(sequences[pid]) for pid in frame.participant_id])
    config = {'experiment_log': str(tmp_path/'experiments.csv')}
    training, validation = frame.iloc[:15].reset_index(drop=True), frame.iloc[15:20].reset_index(drop=True)
    models, bindings = [], {}
    for gender, gender_only in [(False, False), (True, False), (True, True)]:
        name = 'LR-gender-only' if gender_only else 'CV-LR-AV'+('-gender' if gender else '')+'-balanced'
        state = {'protocol': CV_PROTOCOL, 'gender_input': gender, 'gender_categories': ['female', 'male'],
                 'gender_only': gender_only, 'fit_participant_ids': training.participant_id.tolist()}
        estimator = make_estimator(.1, True, 42)
        estimator.fit(final.summary_inputs(state, matrix[:15], training), training.severity_class)
        state['estimator'] = estimator
        path = tmp_path/name/'model.pkl'
        path.parent.mkdir()
        with path.open('wb') as handle:
            pickle.dump(state, handle)
        probs = five_probabilities(estimator, final.summary_inputs(state, matrix[15:20], validation))
        write_report(validation, probs, path.parent/'validation', name, 42, 'validation')
        predictions = path.parent/'validation/predictions.csv'
        models.append({'model': name, 'family': 'diagnostic' if gender_only else 'primary_logistic', 'kind': 'summary', 'seed': 42,
                       'artifact': str(path), 'validation_predictions': str(predictions)})
        bindings.update({str(p): final.sha256(p) for p in [path, predictions]})
    plan = {'protocol': final.PROTOCOL, 'feature_config': config,
            'manifest_hash': digest(frame.to_dict('records')), 'source_provenance': {},
            'output': str(tmp_path/'final'), 'models': models, 'bound_files': bindings,
            'contrasts': [{'baseline': models[0]['model'], 'candidate': models[1]['model'], 'role': 'primary_gender_effect'}],
            'bootstrap_repetitions': 20, 'bootstrap_seed': 42, 'uncertainty': 'synthetic fixture',
            'frozen_at_utc': 'synthetic fixture', 'limitations': ['Synthetic correctness only'],
            'reference_model_selected_before_test': models[0]['model'], 'after_test': 'Test is consumed'}
    path = tmp_path/'plan.json'
    final.seal_plan(path, plan)
    monkeypatch.setattr(final, 'joined_manifest', lambda config: frame)
    monkeypatch.setattr(final, 'load_audited_sequences', lambda config, manifest: (sequences, {}))
    def forbidden_fit(*args, **kwargs):
        raise AssertionError('Final evaluation must not refit a model')
    monkeypatch.setattr(LogisticRegression, 'fit', forbidden_fit)
    result = final.evaluate(path)
    assert result['models'] == 3
    assert result['test_evaluated']
    assert json.loads((tmp_path/'final/completion.json').read_text())['test_class_counts'] == [1]*5
    for model in models:
        actual = pd.read_csv(tmp_path/'final'/f"{model['model']}_seed42/test/predictions.csv", dtype={'participant_id': str})
        assert actual.participant_id.tolist() == frame.participant_id[20:].tolist()
        assert actual.split.eq('test').all()
    with pytest.raises(FileExistsError, match='already exists'):
        final.evaluate(path)
    assert final.report_final(tmp_path/'final')['models'] == 3
