"""Separate modality window collections for participant-level LSTM fusion."""
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import encode_gender
from .datasets import transform


def independent_windows(sequence, modality, config):
    steps = round(config['window_seconds'] / config['bin_seconds'])
    stride = round(config['stride_seconds'] / config['bin_seconds'])
    cap = config['max_windows_per_participant']
    if steps < 1 or stride < 1 or stride > steps or cap < 1:
        raise ValueError('Invalid window/stride/cap')
    mask = sequence[modality + '_mask']
    windows = []
    for start in range(0, len(mask), stride):
        end = min(start + steps, len(mask))
        if end - start >= config['minimum_window_steps'] and mask[start:end].mean() >= config['minimum_observed_fraction']:
            windows.append((start, end))
    if not windows:
        raise ValueError(f'No usable {modality} windows')
    if len(windows) > cap:
        windows = [windows[i] for i in np.linspace(0, len(windows)-1, cap).astype(int)]
    return windows


class IndependentParticipantDataset(Dataset):
    def __init__(self, manifest, sequences, scalers, config):
        self.rows = manifest.to_dict('records')
        self.items = []
        for row in self.rows:
            sequence = transform(sequences[row['participant_id']], scalers)
            windows = {key: independent_windows(sequence, key, config) for key in ['audio', 'visual']}
            self.items.append((sequence, windows))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sequence, windows = self.items[index]
        return self.rows[index], sequence, windows


def collate_independent(batch, categories):
    result = {'ids': [row['participant_id'] for row, _, _ in batch], 'labels': torch.tensor([row.get('severity_class', -1) for row, _, _ in batch], dtype=torch.long), 'gender': torch.tensor(np.stack([encode_gender(row['gender'], categories) for row, _, _ in batch]))}
    for key in ['audio', 'visual']:
        flat = [(owner, sequence, start, end) for owner, (_, sequence, windows) in enumerate(batch) for start, end in windows[key]]
        lengths = [end-start for _, _, start, end in flat]
        values = np.zeros((len(flat), max(lengths), batch[0][1][key].shape[1]+1), np.float32)
        for i, (_, sequence, start, end) in enumerate(flat):
            values[i, :end-start, :-1] = sequence[key][start:end]
            values[i, :end-start, -1] = sequence[key+'_mask'][start:end]
        result[key] = {'values': torch.from_numpy(values), 'lengths': torch.tensor(lengths), 'owners': torch.tensor([owner for owner, _, _, _ in flat])}
    return result


def independent_loader(manifest, sequences, scalers, config, shuffle=False, generator=None):
    return DataLoader(IndependentParticipantDataset(manifest, sequences, scalers, config), batch_size=config['batch_size'], shuffle=shuffle, generator=generator, num_workers=0, collate_fn=partial(collate_independent, categories=config['gender_categories']))
