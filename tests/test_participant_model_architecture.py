"""Alignment, leakage, serialization and fail-closed training-gate tests."""
import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from src.participant_model_data import (FeatureBatch, TrainingTargets, ParticipantFold,
    SCHEMA, verify_repetitions, read_store)
from src import participant_model_runner as r
from src.batch_inference import atomic_json
from src.common import digest
from src.final_evaluation import sha256


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def toy():
    rng = np.random.default_rng(283)
    ids = tuple(f'person{i}' for i in range(75)); base = rng.normal(size=(75, 512))
    base[:, :384] /= np.linalg.norm(base[:, :384], axis=1)[:, None]
    return FeatureBatch(ids, base, rng.normal(size=(75, 16))), TrainingTargets(ids, np.tile(np.arange(5), 15))


def toy_fold(features):
    ids = features.participant_ids; fit, held = ids[:50], ids[50:]
    inner = (ParticipantFold(42, 0, fit[25:], fit[:25]), ParticipantFold(42, 1, fit[:25], fit[25:]))
    return ParticipantFold(42, 0, fit, held, inner)


def test_features_targets_are_separate_strict_archives(toy, tmp_path):
    features, targets = toy
    features.save(tmp_path/'features.npz'); targets.save(tmp_path/'targets.npz')
    assert FeatureBatch.load(tmp_path/'features.npz').participant_ids == features.participant_ids
    assert TrainingTargets.load(tmp_path/'targets.npz').participant_ids == targets.participant_ids
    with pytest.raises(ValueError, match='Inference feature schema'):
        FeatureBatch.load(tmp_path/'targets.npz')
    with pytest.raises(TypeError):
        FeatureBatch(features.participant_ids, features.semantic_av, features.language, severity_class=targets.severity_class)
    with np.load(tmp_path/'features.npz', allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    np.savez_compressed(tmp_path/'leaking.npz', **values, PHQ8_score=targets.severity_class)
    with pytest.raises(ValueError, match='questionnaire'):
        FeatureBatch.load(tmp_path/'leaking.npz')


def test_feature_arrays_are_copied_readonly_and_ids_never_enter_prediction(toy):
    features, targets = toy
    with pytest.raises(ValueError):
        features.language[0, 0] = 999
    model = r.RegularizedParticipantModel(r.ModelSpec(components=4)).fit(features, targets)
    renamed = FeatureBatch(tuple('other'+pid for pid in features.participant_ids), features.semantic_av, features.language)
    np.testing.assert_array_equal(model.predict_proba(renamed), model.predict_proba(features))


def test_duplicate_unknown_and_nonfinite_participants_rejected(toy):
    features, targets = toy
    with pytest.raises(ValueError, match='unique'):
        features.select([features.participant_ids[0]]*2)
    with pytest.raises(ValueError, match='Unknown'):
        targets.select(['outside'])
    with pytest.raises(ValueError, match='finite'):
        FeatureBatch(features.participant_ids, features.semantic_av, features.language+np.nan)
    with pytest.raises(ValueError, match='integer'):
        TrainingTargets(targets.participant_ids, targets.severity_class+.1)


def test_target_reordering_is_rejected_before_any_fit(toy, monkeypatch):
    features, targets = toy
    def forbidden(*a, **k):
        raise AssertionError('Misaligned targets must fail before fitting preprocessing')
    monkeypatch.setattr(r.StandardScaler, 'fit', forbidden)
    reversed_targets = TrainingTargets(tuple(reversed(targets.participant_ids)), targets.severity_class)
    with pytest.raises(ValueError, match='exact order'):
        r.RegularizedParticipantModel().fit(features, reversed_targets)


def test_fitting_statistics_use_only_fitting_people(toy):
    features, targets = toy; fold = toy_fold(features)
    model, record = r.evaluate_fold(features, targets, fold, r.ModelSpec(components=4))
    training = features.select(fold.fit_ids)
    np.testing.assert_allclose(model.base_scaler.mean_, training.semantic_av.mean(0), atol=1e-14)
    np.testing.assert_allclose(model.language_scaler.mean_, training.language.mean(0), atol=1e-14)
    assert model.projection.n_samples_ == 50 and model.estimator.n_features_in_ == 20
    assert record['new_fits'] == 3 and not record['outer_results_used_for_tuning']
    assert all(row['converged'] for row in record['inner_diagnostics'])


def test_outer_features_and_labels_cannot_change_training_or_selection(toy):
    features, targets = toy; fold = toy_fold(features); spec = r.ModelSpec(components=4)
    model, record = r.evaluate_fold(features, targets, fold, spec)
    base, language = features.semantic_av.copy(), features.language.copy()
    base[50:, 384:] += 10000; language[50:] -= 10000
    changed = FeatureBatch(features.participant_ids, base, language)
    labels = targets.severity_class.copy(); labels[50:] = (labels[50:]+1)%5
    altered_model, altered = r.evaluate_fold(changed, TrainingTargets(targets.participant_ids, labels), fold, spec)
    for key in model.parameters:
        np.testing.assert_array_equal(model.parameters[key], altered_model.parameters[key])
    assert record['inner_diagnostics'] == altered['inner_diagnostics']
    assert record['outer_fitting_diagnostics'] == altered['outer_fitting_diagnostics']


def test_numeric_checkpoint_and_inference_round_trip(toy, tmp_path):
    features, targets = toy
    model = r.RegularizedParticipantModel(r.ModelSpec(components=4)).fit(features, targets)
    root = tmp_path/'model'; r.save_model(root, model.parameters, {'kind': 'synthetic_unit_test'})
    loaded, contract = r.load_model(root)
    np.testing.assert_allclose(r.predict_parameters(loaded, features), model.predict_proba(features), atol=1e-14)
    features.save(tmp_path/'features.npz')
    r.predict(root, tmp_path/'features.npz', tmp_path/'predictions.json')
    result = json.loads((tmp_path/'predictions.json').read_text())
    assert not result['questionnaire_inputs'] and not result['new_accuracy_claim']
    with pytest.raises(FileExistsError):
        r.save_model(root, model.parameters, {})
    (root/'weights.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='binding'):
        r.load_model(root)


def test_fold_overlap_and_repeated_owner_errors_are_rejected(toy):
    features, _ = toy; ids = features.participant_ids
    with pytest.raises(ValueError, match='leakage'):
        ParticipantFold(42, 0, ids[:50], ids[49:])
    good = [ParticipantFold(42, 0, ids[25:], ids[:25]), ParticipantFold(42, 1, ids[:25], ids[25:])]
    assert verify_repetitions(good, ids) == [42]
    with pytest.raises(ValueError, match='exactly one'):
        verify_repetitions(good[:1], ids)


def test_current_data_gate_blocks_all_fitting_and_completed_resume(toy, tmp_path, monkeypatch):
    features, targets = toy; fold = toy_fold(features); store = tmp_path/'store'; store.mkdir()
    (store/'manifest.json').write_text('{}')
    manifest = {'training_allowed': False, 'training_blockers': ['No verified speaker annotations; unchanged original features']}
    monkeypatch.setattr(r, 'read_store', lambda root: (features, targets, (fold,), manifest))
    def forbidden(*a, **k):
        raise AssertionError('No real or toy fit may pass this gate')
    monkeypatch.setattr(r.RegularizedParticipantModel, 'fit', forbidden)
    out = tmp_path/'blocked'
    result = r.train(store, out)
    assert result['total_real_data_fits'] == 0 and not result['new_accuracy_claim']
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
    resumed = r.train(store, out, resume=True)
    assert resumed['new_fits_this_call'] == 0 and resumed['rewritten_files'] == 0
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_best_saved_parameter_contract_matches_original_numeric_inference(toy):
    from src.best_observed_inference import load, predict_features
    features, _ = toy; _, parameters = load()
    expected = predict_features(parameters, features.semantic_av[:, :384], features.semantic_av[:, 384:], features.language)
    np.testing.assert_allclose(r.predict_parameters(parameters, features), expected, rtol=0, atol=1e-14)


def test_paired_summary_counts_people_with_repeats_without_promoting_default(toy, tmp_path):
    # Perfect one-hot values are metric fixtures, not predictions from a trained model.
    features, targets = toy; ids = features.participant_ids; folds = []; records = []
    for seed in [42, 137]:
        for number, (fit, held) in enumerate([(ids[25:], ids[:25]), (ids[:25], ids[25:])]):
            fold = ParticipantFold(seed, number, fit, held); folds.append(fold)
            records.append({'fold': fold.as_dict(), 'outer_probabilities': np.eye(5)[targets.select(held).severity_class].tolist()})
    refs = np.zeros((2, len(ids), 5)); refs[:, :, 0] = 1
    np.savez_compressed(tmp_path/'refs.npz', participant_id=np.asarray(ids), seeds=np.asarray([42,137]), original=refs, best_procedure=refs)
    promotion = json.loads(r.CONFIG.read_text())['promotion']
    report = r.summarize_candidate(features, targets, folds, records, tmp_path/'refs.npz', tmp_path, promotion)
    assert report['bootstrap_unit'] == 'participant with all repeated predictions'
    assert report['exploratory_statistical_criteria_met'] and not report['default_replaced']
    changes = report['gains_losses'][0]
    assert changes['gained'] == 2*60 and changes['unique_people_with_gain'] == 60
    assert changes['lost'] == 0 and not report['independent_confirmation']


def test_training_allowed_flag_cannot_contradict_participant_evidence(toy, tmp_path):
    features, targets = toy; ids = features.participant_ids
    features.save(tmp_path/'features.npz'); targets.save(tmp_path/'training_targets.npz')
    rows = [{'participant_id': pid, 'feature_row': i, 'split': 'train',
        'complete_verified_text_review': False, 'features_changed': False} for i,pid in enumerate(ids)]
    atomic_json(tmp_path/'participants.json', rows)
    folds = [ParticipantFold(42, 0, ids[25:], ids[:25]), ParticipantFold(42, 1, ids[:25], ids[25:])]
    atomic_json(tmp_path/'folds.json', [fold.as_dict() for fold in folds])
    np.savez_compressed(tmp_path/'reference_predictions.npz', participant_id=np.asarray(ids))
    manifest = {'schema': SCHEMA, 'participant_ids': list(ids), 'population': len(ids), 'bindings': {},
        'complete_verified_text_reviews': 0, 'changed_feature_participants': 0, 'training_allowed': False,
        'files': {name: sha256(tmp_path/name) for name in ['features.npz','training_targets.npz','participants.json','folds.json','reference_predictions.npz']}}
    atomic_json(tmp_path/'manifest.json', {'manifest': manifest, 'manifest_hash': digest(manifest)})
    assert read_store(tmp_path)[3]['training_allowed'] is False
    manifest['training_allowed'] = True
    atomic_json(tmp_path/'manifest.json', {'manifest': manifest, 'manifest_hash': digest(manifest)})
    with pytest.raises(ValueError, match='gate differs'):
        read_store(tmp_path)
