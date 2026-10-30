import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import encode_gender, VERSION


def load_sequences(manifest, config):
    sequences, schema, hashes = {}, None, {}
    for row in manifest.itertuples():
        path = Path(config["cache"]) / "participants" / f"{row.participant_id}.npz"
        info = json.loads(path.with_suffix(".json").read_text())
        if info["pilot_only"] or info["version"] != VERSION or info["participant_id"] != row.participant_id:
            raise ValueError("Pilot, stale, or mismatched cache cannot be used for training")
        if any(config[k] != v for k, v in info["settings"].items()):
            raise ValueError("Preprocessing configuration changed; rerun preparation")
        current_schema = (info["audio_columns"], info["visual_columns"])
        if schema is not None and schema != current_schema:
            raise ValueError("Inconsistent participant feature schema/order")
        schema = current_schema
        with np.load(path, allow_pickle=False) as data:
            seq = {k: data[k].copy() for k in data.files}
        n = len(seq["timestamps"])
        if any(len(x) != n for x in seq.values()) or not np.all(np.diff(seq["timestamps"]) > 0):
            raise ValueError("Cached timestamp/sequence length mismatch")
        if not all(np.isfinite(x).all() for x in seq.values()):
            raise ValueError("Nonfinite cache")
        h = hashlib.sha256(seq["audio"].tobytes() + seq["visual"].tobytes()).hexdigest()
        if h in hashes:
            raise ValueError(f"Duplicate modality content: {hashes[h]} and {row.participant_id}")
        hashes[h] = row.participant_id
        sequences[row.participant_id] = seq
    return sequences


def fit_scalers(sequences, train_ids):
    if not train_ids:
        raise ValueError("No training participants")
    result = {}
    for key in ["audio", "visual"]:
        # Each participant has equal weight in preprocessing statistics.
        moments = []
        for pid in train_ids:
            x = sequences[pid][key][sequences[pid][key + "_mask"]]
            if not len(x):
                raise ValueError(f"No observed {key} values: {pid}")
            moments.append((x.mean(0, dtype=np.float64), np.square(x.astype(np.float64)).mean(0)))
        mean = np.mean([m[0] for m in moments], axis=0)
        var = np.mean([m[1] for m in moments], axis=0) - mean ** 2
        result[key] = {"mean": mean.astype(np.float32), "std": np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)}
    result["fit_participant_ids"] = list(train_ids)
    return result


def transform(seq, scalers):
    out = {k: v.copy() for k, v in seq.items()}
    for key in ["audio", "visual"]:
        out[key] = (out[key] - scalers[key]["mean"]) / scalers[key]["std"]
        out[key][~out[key + "_mask"]] = 0
    return out


def window_indices(seq, config):
    steps = round(config["window_seconds"] / config["bin_seconds"])
    stride = round(config["stride_seconds"] / config["bin_seconds"])
    if steps <= 0 or stride <= 0 or stride > steps:
        raise ValueError("Window/stride must be positive, with stride <= window")
    n = len(seq["audio"])
    windows = []
    for start in range(0, n, stride):
        stop = min(n, start + steps)
        if stop - start < config["minimum_window_steps"]:
            continue
        if all(seq[k + "_mask"][start:stop].mean() >= config["minimum_observed_fraction"] for k in ["audio", "visual"]):
            windows.append((start, stop))
    cap = config["max_windows_per_participant"]
    if cap < 1:
        raise ValueError("max_windows_per_participant must be positive")
    if len(windows) > cap:
        windows = [windows[i] for i in np.linspace(0, len(windows) - 1, cap).astype(int)]
    if not windows:
        raise ValueError("No usable windows for participant; inspect modality coverage")
    return windows


class ParticipantDataset(Dataset):
    def __init__(self, manifest, sequences, scalers, config):
        self.rows = manifest.to_dict("records")
        self.items = []
        self.config = config
        for row in self.rows:
            seq = transform(sequences[row["participant_id"]], scalers)
            indices = window_indices(seq, config)
            self.items.append((seq, indices))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, (seq, windows) = self.rows[index], self.items[index]
        return row, seq, windows


def collate_participants(batch, categories):
    flat = [(i, seq, start, end) for i, (_, seq, windows) in enumerate(batch) for start, end in windows]
    max_len = max(end - start for _, _, start, end in flat)
    result = {"owner": torch.tensor([x[0] for x in flat]), "labels": torch.tensor([b[0].get("severity_class", -1) for b in batch]), "gender": torch.tensor(np.stack([encode_gender(b[0]["gender"], categories) for b in batch])), "ids": [b[0]["participant_id"] for b in batch]}
    lengths = []
    for key in ["audio", "visual"]:
        d = flat[0][1][key].shape[1]
        x = np.zeros((len(flat), max_len, d + 1), np.float32)
        for j, (_, seq, start, end) in enumerate(flat):
            length = end - start
            x[j, :length, :d] = seq[key][start:end]
            x[j, :length, d] = seq[key + "_mask"][start:end]
        result[key] = torch.from_numpy(x)
    lengths = [end - start for _, _, start, end in flat]
    result["lengths"] = torch.tensor(lengths, dtype=torch.long)
    return result


def summarize(seq):
    return np.concatenate([f(seq[k][seq[k + "_mask"]], axis=0) for k in ["audio", "visual"] for f in [np.mean, np.std]])
