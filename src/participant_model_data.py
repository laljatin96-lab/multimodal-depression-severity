"""Typed participant-level inputs with physically separate features and targets."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .batch_inference import atomic_json
from .common import digest
from .final_evaluation import sha256
from .age_language_accuracy import LANGUAGE_NAMES

SCHEMA = {'version': 'participant-model-data-v1', 'unit': 'one_original_participant',
    'inference_blocks': {'semantic_av': 512, 'language': 16},
    'semantic_columns': [0, 384], 'av_columns': [384, 512],
    'language_names': list(LANGUAGE_NAMES),
    'classes': [0, 1, 2, 3, 4], 'targets_in_inference_features': False}


def _ids(values):
    result = tuple(values)
    if not result or any(not isinstance(i, str) or not i for i in result) or len(set(result)) != len(result):
        raise ValueError('Nonempty unique string participant IDs required')
    return result


@dataclass(frozen=True)
class FeatureBatch:
    participant_ids: tuple
    semantic_av: np.ndarray
    language: np.ndarray

    def __post_init__(self):
        ids = _ids(self.participant_ids); object.__setattr__(self, 'participant_ids', ids)
        for key, width in SCHEMA['inference_blocks'].items():
            array = np.array(getattr(self, key), dtype=float, copy=True)
            if array.shape != (len(ids), width) or not np.isfinite(array).all():
                raise ValueError(f'Expected finite aligned {key} shape N by {width}')
            array.setflags(write=False); object.__setattr__(self, key, array)
        if not np.allclose(np.linalg.norm(self.semantic_av[:, :384], axis=1), 1, rtol=0, atol=1e-5):
            raise ValueError('Expected the frozen normalized semantic384 block')

    def select(self, participant_ids):
        wanted = _ids(participant_ids); lookup = {pid: i for i, pid in enumerate(self.participant_ids)}
        if set(wanted)-set(lookup):
            raise ValueError('Unknown participant in feature selection')
        positions = [lookup[pid] for pid in wanted]
        return FeatureBatch(wanted, self.semantic_av[positions], self.language[positions])

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError('Preserving feature batch')
        np.savez_compressed(path, participant_id=np.asarray(self.participant_ids),
            semantic_av=self.semantic_av, language=self.language, schema_hash=np.asarray(digest(SCHEMA)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            if (set(archive.files) != {'participant_id', 'semantic_av', 'language', 'schema_hash'}
                    or archive['schema_hash'].shape != () or str(archive['schema_hash']) != digest(SCHEMA)):
                raise ValueError('Inference feature schema differs; targets and questionnaire fields are forbidden')
            return cls(tuple(archive['participant_id'].tolist()), archive['semantic_av'], archive['language'])


@dataclass(frozen=True)
class TrainingTargets:
    participant_ids: tuple
    severity_class: np.ndarray

    def __post_init__(self):
        ids = _ids(self.participant_ids); values = np.asarray(self.severity_class)
        if (values.shape != (len(ids),) or not np.issubdtype(values.dtype, np.number)
                or not np.isfinite(values).all() or not np.equal(values, values.astype(int)).all()
                or not set(values).issubset(set(SCHEMA['classes']))):
            raise ValueError('Aligned integer training classes zero through four required')
        values = np.array(values, dtype=int, copy=True); values.setflags(write=False)
        object.__setattr__(self, 'participant_ids', ids); object.__setattr__(self, 'severity_class', values)

    def select(self, participant_ids):
        wanted = _ids(participant_ids); lookup = {pid: i for i, pid in enumerate(self.participant_ids)}
        if set(wanted)-set(lookup):
            raise ValueError('Unknown participant in target selection')
        return TrainingTargets(wanted, self.severity_class[[lookup[pid] for pid in wanted]])

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError('Preserving training targets')
        np.savez_compressed(path, participant_id=np.asarray(self.participant_ids), severity_class=self.severity_class)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'participant_id', 'severity_class'}:
                raise ValueError('Unexpected training-target schema')
            return cls(tuple(archive['participant_id'].tolist()), archive['severity_class'])


@dataclass(frozen=True)
class ParticipantFold:
    seed: int
    fold: int
    fit_ids: tuple
    heldout_ids: tuple
    inner: tuple = ()

    def __post_init__(self):
        fit, heldout = _ids(self.fit_ids), _ids(self.heldout_ids)
        if set(fit) & set(heldout):
            raise ValueError('Participant leakage between fitting and held-out cohorts')
        object.__setattr__(self, 'fit_ids', fit); object.__setattr__(self, 'heldout_ids', heldout)
        for part in self.inner:
            if set(part.fit_ids+part.heldout_ids) != set(fit):
                raise ValueError('Inner split must contain only its outer fitting participants')
        if self.inner:
            owners = {pid: 0 for pid in fit}
            for part in self.inner:
                for pid in part.heldout_ids:
                    owners[pid] += 1
            if set(owners.values()) != {1}:
                raise ValueError('Each fitting participant requires one inner held-out prediction')

    def as_dict(self):
        return {'seed': self.seed, 'fold': self.fold, 'fit_ids': list(self.fit_ids),
            'heldout_ids': list(self.heldout_ids), 'inner': [part.as_dict() for part in self.inner]}

    @classmethod
    def from_dict(cls, row):
        if set(row) != {'seed', 'fold', 'fit_ids', 'heldout_ids', 'inner'}:
            raise ValueError('Unexpected fold schema')
        return cls(row['seed'], row['fold'], tuple(row['fit_ids']), tuple(row['heldout_ids']),
            tuple(cls.from_dict(part) for part in row['inner']))


def index_folds(folds, ids):
    def one(row):
        return ParticipantFold(row['seed'], row['fold'], tuple(ids[i] for i in row['fit']),
            tuple(ids[i] for i in row['heldout']), tuple(one({**part, 'seed': row['seed'], 'fold': row['fold']}) for part in row.get('inner', [])))
    return tuple(one(row) for row in folds)


def verify_repetitions(folds, participant_ids):
    population = set(_ids(participant_ids)); seeds = sorted({f.seed for f in folds})
    if not folds or len({(f.seed, f.fold) for f in folds}) != len(folds):
        raise ValueError('Unique complete participant folds required')
    for seed in seeds:
        held = []
        for fold in [f for f in folds if f.seed == seed]:
            if set(fold.fit_ids+fold.heldout_ids) != population:
                raise ValueError('Outer fold population differs')
            held.extend(fold.heldout_ids)
        if len(held) != len(population) or set(held) != population:
            raise ValueError('Each participant requires exactly one outer held-out prediction per seed')
    return seeds


def read_store(root):
    root = Path(root); envelope = json.loads((root/'manifest.json').read_text()); manifest = envelope['manifest']
    if digest(manifest) != envelope['manifest_hash'] or manifest['schema'] != SCHEMA:
        raise ValueError('Dataset manifest schema changed')
    if set(manifest['files']) != {'features.npz', 'training_targets.npz', 'participants.json', 'folds.json', 'reference_predictions.npz'}:
        raise ValueError('Dataset file schema differs')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or sha256(root/name) != expected:
            raise ValueError('Dataset file ownership or hash changed')
    for path, expected in manifest['bindings'].items():
        if sha256(path) != expected:
            raise ValueError('Dataset source or implementation changed')
    features = FeatureBatch.load(root/'features.npz'); targets = TrainingTargets.load(root/'training_targets.npz')
    if features.participant_ids != targets.participant_ids or list(features.participant_ids) != manifest['participant_ids']:
        raise ValueError('Feature/target participant order differs')
    participants = json.loads((root/'participants.json').read_text())
    if ([row['participant_id'] for row in participants] != list(features.participant_ids)
            or [row['feature_row'] for row in participants] != list(range(len(participants)))
            or any(row['split'] != 'train' for row in participants)):
        raise ValueError('Participant registry and feature-row ownership differ')
    reviews = sum(row['complete_verified_text_review'] is True for row in participants)
    changed = sum(row['features_changed'] is True for row in participants)
    if (manifest['population'] != len(participants) or reviews != manifest['complete_verified_text_reviews']
            or changed != manifest['changed_feature_participants']
            or manifest['training_allowed'] != (reviews == len(participants) and changed > 0)):
        raise ValueError('Training gate differs from participant evidence')
    folds = tuple(ParticipantFold.from_dict(row) for row in json.loads((root/'folds.json').read_text()))
    verify_repetitions(folds, features.participant_ids)
    return features, targets, folds, manifest


def build_store(output, input_cache='results/participant_speech', resume=False):
    """Reorganize existing verified computations; this function never fits or encodes."""
    from .participant_input_readiness import resume_completed, verify_population
    from .wavlm_accuracy import load_inputs, nested_folds
    from .participant_speech import validate_annotation
    output, cache = Path(output), Path(input_cache)
    if any(output.resolve().is_relative_to(Path(p).resolve()) for p in ['data', 'dataset', 'checkpoints', cache]):
        raise ValueError('Dataset store must be outside original inputs, checkpoints and extraction cache')
    if output.exists():
        if not resume:
            raise FileExistsError('Preserving existing participant dataset')
        return read_store(output)[3]
    if resume:
        raise FileNotFoundError('No completed dataset to resume')
    resume_completed('results/participant_input_readiness_20261029')
    x, frame, settings, _, _ = load_inputs(); ids = tuple(frame.participant_id)
    old_folds = nested_folds(frame, settings); verify_population(frame, old_folds)
    contract = json.loads((cache/'contract.json').read_text()); context = contract['context']
    receipt = json.loads((cache/'completion.json').read_text())
    if (digest(context) != contract['context_hash'] or receipt['context_hash'] != contract['context_hash']
            or context['training_ids'] != list(ids) or not receipt['complete']
            or receipt['targets_used_for_input_selection'] or receipt['validation_test_processed']):
        raise ValueError('Input cache population or provenance differs')
    bindings = {str(cache/'contract.json'): sha256(cache/'contract.json'), str(cache/'completion.json'): sha256(cache/'completion.json'),
        'configs/participant_speech_comparison.json': sha256('configs/participant_speech_comparison.json'),
        str(Path(__file__)): sha256(__file__), 'src/participant_speech.py': sha256('src/participant_speech.py')}
    for path, expected in {**receipt['artifacts'], **context['support_hashes'], **context['annotation_hashes'],
                           **context['provenance']['bindings']}.items():
        if sha256(path) != expected:
            raise ValueError('Input cache dependency changed')
        bindings[path] = expected
    if context['source_code_sha256'] != sha256('src/participant_speech.py'):
        raise ValueError('Participant extraction implementation changed')
    with np.load(cache/'features.npz', allow_pickle=False) as archive:
        if archive['participant_id'].tolist() != list(ids):
            raise ValueError('Input feature order differs')
        features = FeatureBatch(ids, archive['semantic_av'], archive['language'])
    np.testing.assert_array_equal(features.semantic_av[:, 384:], x['semantic_av'][:, 384:])
    annotations = {}
    for path in context['annotation_hashes']:
        row = json.loads(Path(path).read_text()); annotations[row['participant_id']] = row
    participants = []
    for i, source in enumerate(context['sources']):
        pid = source['participant_id']; annotation = annotations.get(pid)
        decision = validate_annotation(annotation, source)
        complete_review = bool(annotation and annotation['status'] == 'reviewed'
            and annotation['transcript_review']['verified'] and annotation['transcript_review']['complete_review'])
        changed = not (np.array_equal(features.semantic_av[i], x['semantic_av'][i]) and np.array_equal(features.language[i], x['language'][i]))
        participants.append({'participant_id': pid, 'feature_row': i, 'split': 'train',
            'complete_verified_text_review': complete_review, 'usable_participant_text': decision['text_verified'],
            'text_reason': decision['text_reason'], 'features_changed': changed,
            'raw_transcript_sha256': source['raw_transcript_sha256'], 'raw_wav_sha256': source['raw_wav_sha256']})
    if [row['participant_id'] for row in participants] != list(ids):
        raise ValueError('Participant registry order differs')
    # Cached references remain outside inference inputs and can never be loaded as FeatureBatch.
    reference_arrays = {'participant_id': np.asarray(ids), 'seeds': np.asarray([42, 137, 2026])}
    for name, root, policy in [('original', 'results/age_language_accuracy', 'without_age'), ('best_procedure', 'results/wavlm_accuracy', 'accuracy')]:
        arrays = []
        for seed in [42, 137, 2026]:
            path = Path(root)/'oof'/f'{policy}_seed{seed}'/'predictions.csv'
            import pandas as pd
            table = pd.read_csv(path, dtype={'participant_id': str})
            if table.participant_id.tolist() != list(ids) or not table.split.eq('train_nested_oof').all():
                raise ValueError('Cached reference ownership differs')
            np.testing.assert_array_equal(table.severity_class, frame.severity_class)
            arrays.append(table[[f'p{c}' for c in range(5)]].to_numpy(float)); bindings[str(path)] = sha256(path)
        reference_arrays[name] = np.stack(arrays)
    output.mkdir(parents=True)
    features.save(output/'features.npz'); TrainingTargets(ids, frame.severity_class.to_numpy(int)).save(output/'training_targets.npz')
    atomic_json(output/'participants.json', participants)
    atomic_json(output/'folds.json', [f.as_dict() for f in index_folds(old_folds, ids)])
    np.savez_compressed(output/'reference_predictions.npz', **reference_arrays)
    blockers = []
    reviews = sum(row['complete_verified_text_review'] for row in participants)
    changed = sum(row['features_changed'] for row in participants)
    if reviews != len(ids):
        blockers.append(f'Complete verified transcript/audio alignment review is available for {reviews}/{len(ids)} participants')
    if not changed:
        blockers.append('All candidate inputs remain identical to the original inputs; repeated fitting would repeat a completed experiment')
    manifest = {'schema': SCHEMA, 'participant_ids': list(ids), 'population': len(ids),
        'files': {name: sha256(output/name) for name in ['features.npz', 'training_targets.npz', 'participants.json', 'folds.json', 'reference_predictions.npz']},
        'bindings': bindings, 'complete_verified_text_reviews': reviews, 'changed_feature_participants': changed,
        'training_allowed': not blockers, 'training_blockers': blockers, 'new_fits': 0, 'new_encoder_runs': 0,
        'validation_test_processed': False, 'targets_stored_separately': True, 'input_cache': str(cache)}
    atomic_json(output/'manifest.json', {'manifest': manifest, 'manifest_hash': digest(manifest)})
    read_store(output)
    return manifest
