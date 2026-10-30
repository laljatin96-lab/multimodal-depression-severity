import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.datasets import fit_scalers
from src.independent_datasets import collate_independent, independent_loader, independent_windows
from src.independent_train import PROTOCOL, predictions, train_independent
from src.models import IndependentAVClassifier
from src.evaluate import metrics
from src.report_independent import paired_mean_bootstrap, point_metrics


def fixture(tmp_path):
    rng = np.random.default_rng(10)
    sequences = {}
    for i in range(25):
        n = 9+i%3
        sequences[str(i)] = {'audio': rng.normal(size=(n, 3)).astype('float32'), 'visual': rng.normal(size=(n, 2)).astype('float32'), 'audio_mask': np.ones(n, bool), 'visual_mask': np.ones(n, bool), 'timestamps': np.arange(n)*.5}
    frame = pd.DataFrame({'participant_id': [str(i) for i in range(25)], 'split': ['train']*15+['validation']*5+['test']*5, 'gender': ['female', 'male', 'unknown', 'female', 'male']*5, 'severity_class': list(range(5))*5})
    config = {'protocol': PROTOCOL, 'allow_unsegmented': True, 'seed': 42, 'gender_categories': ['female', 'male'], 'window_seconds': 2., 'bin_seconds': .5, 'stride_seconds': 2., 'minimum_window_steps': 2, 'minimum_observed_fraction': .5, 'max_windows_per_participant': 3, 'hidden_size': 4, 'dropout': .3, 'batch_size': 5, 'learning_rate': .001, 'weight_decay': .0001, 'gradient_clip': 1., 'epochs': 2, 'patience': 8, 'device': 'cpu', 'experiment_log': str(tmp_path/'experiments.csv')}
    return frame, sequences, config


def test_independent_window_validity_and_shift(tmp_path):
    _, sequences, config = fixture(tmp_path)
    sequence = sequences['0']
    sequence['visual_mask'][:4] = False
    audio = independent_windows(sequence, 'audio', config)
    visual = independent_windows(sequence, 'visual', config)
    assert len(audio) > len(visual)
    sequence['timestamps'] += 1000
    assert visual == independent_windows(sequence, 'visual', config)


def test_independent_pooling_permutation_padding_backward(tmp_path):
    frame, sequences, config = fixture(tmp_path)
    sequences['0']['visual_mask'][:4] = False
    scales = fit_scalers(sequences, ['0','1'])
    loader = independent_loader(frame.iloc[:2], sequences, scales, config)
    batch = next(iter(loader))
    assert len(batch['audio']['values']) != len(batch['visual']['values'])
    model = IndependentAVClassifier(4, 3, hidden_size=4, dropout=0, gender_dim=3).eval()
    expected = model(batch['audio'], batch['visual'], batch['gender'])
    permutation = torch.arange(len(batch['visual']['values'])-1, -1, -1)
    permuted = {key: value[permutation].clone() for key, value in batch['visual'].items()}
    for i, length in enumerate(permuted['lengths']):
        permuted['values'][i, length:] = 1e8
    actual = model(batch['audio'], permuted, batch['gender'])
    torch.testing.assert_close(actual, expected)
    torch.nn.functional.cross_entropy(actual, batch['labels']).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_interrupted_resume_matches_uninterrupted_and_preserves_data(tmp_path, monkeypatch):
    frame, sequences, config = fixture(tmp_path)
    provenance = {'test_fixture': True}
    full = tmp_path/'full'
    completed = train_independent(config, full, frame, sequences, provenance, gender=True, balanced=True)
    original_step = torch.optim.AdamW.step
    counter = [0]
    def interrupt_step(self, *args, **kwargs):
        counter[0] += 1
        if counter[0] == 4:
            raise RuntimeError('simulated interruption')
        return original_step(self, *args, **kwargs)
    interrupted = tmp_path/'interrupted'
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, 'step', interrupt_step)
        with pytest.raises(RuntimeError, match='simulated interruption'):
            train_independent(config, interrupted, frame, sequences, provenance, gender=True, balanced=True)
    resumed = train_independent(config, interrupted, frame, sequences, provenance, gender=True, balanced=True, resume=True)
    assert completed['macro_f1'] == resumed['macro_f1']
    first = torch.load(full/'latest.pt', weights_only=False)
    second = torch.load(interrupted/'latest.pt', weights_only=False)
    for key in first['model']:
        torch.testing.assert_close(first['model'][key], second['model'][key], rtol=0, atol=0)
    assert first['scalers']['fit_participant_ids'] == frame.participant_id[:15].tolist()
    assert not list(full.rglob('test'))
    saved = json.loads((full/'training_provenance.json').read_text())
    assert len(saved['fit_participant_ids']) == 15
    with pytest.raises(ValueError, match='Resume configuration'):
        train_independent({**config, 'seed': 137}, interrupted, frame, sequences, provenance, gender=True, balanced=True, resume=True)


def test_test_labels_and_features_do_not_affect_training(tmp_path):
    frame, sequences, config = fixture(tmp_path)
    config = {**config, 'epochs': 1}
    first = tmp_path/'first'
    train_independent(config, first, frame, sequences, {}, gender=False)
    frame.loc[frame.split == 'test', 'severity_class'] = 4
    for pid in frame.loc[frame.split == 'test', 'participant_id']:
        sequences[pid]['audio'] += 10000
        sequences[pid]['visual'] -= 10000
    second = tmp_path/'second'
    train_independent(config, second, frame, sequences, {}, gender=False)
    a = pd.read_csv(first/'validation/predictions.csv')
    b = pd.read_csv(second/'validation/predictions.csv')
    pd.testing.assert_frame_equal(a, b)


def test_bootstrap_metrics_match_report_and_identical_pairs_have_zero_difference():
    truth = np.array([0,0,1,1,2,4])
    predicted = np.array([0,1,0,1,1,4])
    expected = metrics(truth,np.eye(5)[predicted])
    np.testing.assert_allclose(point_metrics(truth,predicted),[expected[k] for k in ['accuracy','macro_f1','balanced_accuracy']])
    pairs = np.stack([predicted,predicted[::-1]])
    interval = paired_mean_bootstrap(truth,pairs,pairs,repetitions=40)
    assert all(row['mean_seed_delta']==0 and row['paired_participant_ci95']==[0,0] for row in interval.values())
