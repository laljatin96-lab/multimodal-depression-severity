import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.common import encode_gender, joined_manifest, read_metadata, severity
from src.create_splits import split_ids
from src.datasets import collate_participants, fit_scalers, transform, window_indices
from src.evaluate import metrics, paired_bootstrap
from src.models import AVClassifier, aggregate_probabilities
from src.prepare_features import bin_frames, speech_mask, temporal_frame


@pytest.mark.parametrize("score,expected", [(0, 0), (4, 0), (5, 1), (9, 1), (10, 2), (14, 2), (15, 3), (19, 3), (20, 4), (24, 4)])
def test_phq8_boundaries(score, expected):
    assert severity(score) == expected


@pytest.mark.parametrize("score", [-1, 25, 5.5, np.nan, np.inf, ""])
def test_invalid_phq8(score):
    with pytest.raises((ValueError, TypeError)):
        severity(score)


def test_metadata_duplicate_and_label_integrity(tmp_path):
    p = tmp_path / "meta.csv"
    p.write_text("participant_id,gender,phq8_score,severity_class\n306,,5,0\n")
    with pytest.raises(ValueError, match="disagrees"):
        read_metadata(p)
    p.write_text("participant_id,gender,phq8_score\n306,,5\n306,,10\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_metadata(p)


def test_unknown_gender_explicit():
    assert encode_gender("", ["recorded_a", "recorded_b"]).tolist() == [0, 0, 1]
    assert encode_gender("recorded_b", ["recorded_a", "recorded_b"]).tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        encode_gender("a", ["a", "a"])


def test_splits_disjoint_reproducible_and_official():
    meta = pd.DataFrame({"participant_id": [str(i) for i in range(100)], "gender": ["A", "B"] * 50, "severity_class": np.arange(100) % 5})
    a, _ = split_ids(meta, 42)
    b, _ = split_ids(meta, 42)
    assert a.equals(b) and not a.participant_id.duplicated().any()
    groups = {s: set(a[a.split == s].participant_id) for s in a.split.unique()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["test"] & groups["validation"]
    meta = meta.merge(a)
    official, _ = split_ids(meta, 999)
    assert official.equals(meta[["participant_id", "split"]])


def test_joint_sparse_split_fallback():
    meta = pd.DataFrame({"participant_id": [str(i) for i in range(30)], "gender": [str(i) for i in range(30)], "severity_class": np.arange(30) % 5})
    split, notes = split_ids(meta, 42)
    assert len(split) == 30 and "infeasible" in notes[0]


def test_overlap_and_label_matching_fail(tmp_path):
    meta, splits = tmp_path / "meta.csv", tmp_path / "split.csv"
    meta.write_text("participant_id,gender,phq8_score\n306,,4\n487,,10\n")
    splits.write_text("participant_id,split\n306,train\n306,test\n")
    with pytest.raises(ValueError, match="overlapping"):
        joined_manifest({"metadata": meta, "splits": splits})
    splits.write_text("participant_id,split\n306,train\n488,test\n")
    with pytest.raises(ValueError, match="match exactly"):
        joined_manifest({"metadata": meta, "splits": splits})


def seq(values):
    x = np.asarray(values, np.float32).reshape(-1, 1)
    return {"audio": x.copy(), "visual": x.copy(), "audio_mask": np.ones(len(x), bool), "visual_mask": np.ones(len(x), bool), "timestamps": np.arange(len(x)) * .5}


def test_training_only_scalers_exclude_missing():
    train, held = seq([1, 3, 999]), seq([1e6, 1e6])
    train["audio_mask"][-1] = False
    train["visual_mask"][-1] = False
    scales = fit_scalers({"train": train, "test": held}, ["train"])
    assert scales["audio"]["mean"].item() == 2
    transformed = transform(train, scales)
    assert transformed["audio"].ravel().tolist() == [-1, 1, 0]


def test_actual_time_alignment_different_rates():
    edges = np.array([0, .5, 1.0])
    a, am = bin_frames(np.array([0, .1, .5, .9]), np.array([[1], [3], [5], [7]]), np.ones(4, bool), edges)
    v, vm = bin_frames(np.array([.2, .7]), np.array([[20], [40]]), np.ones(2, bool), edges)
    assert a.ravel().tolist() == [2, 6] and v.ravel().tolist() == [20, 40]
    assert am.all() and vm.all()


def test_invalid_tracking_is_missing_not_zero():
    out, mask = bin_frames(np.array([0, .5]), np.array([[4], [0]]), np.array([False, True]), np.array([0, .5, 1]))
    assert out.ravel().tolist() == [0, 0]
    assert mask.tolist() == [False, True]
    with pytest.raises(ValueError, match="Nonfinite"):
        bin_frames(np.array([0]), np.array([[np.nan]]), np.array([False]), np.array([0, 1]))


def test_speech_and_timestamp_validation():
    assert speech_mask(np.array([0, .5, 1]), [[0, 1]]).tolist() == [True, True, False]
    with pytest.raises(ValueError):
        speech_mask(np.array([0]), [[1, .5]])
    with pytest.raises(ValueError, match="Timestamps"):
        temporal_frame(b"timestamp,x\n1,0\n0,1\n", "timestamp", ",")


def test_window_cap_and_missing_data():
    config = {"window_seconds": 2, "bin_seconds": .5, "stride_seconds": 2, "minimum_window_steps": 2, "minimum_observed_fraction": .5, "max_windows_per_participant": 3}
    s = seq(range(40))
    assert len(window_indices(s, config)) == 3
    s["visual_mask"][:] = False
    with pytest.raises(ValueError, match="No usable windows"):
        window_indices(s, config)


def test_padding_masks_backward_checkpoint_and_aggregation(tmp_path):
    torch.manual_seed(42)
    a, b = seq([1, 2, 3, 4]), seq([5, 6])
    batch = collate_participants([({"participant_id": "1", "gender": "A", "severity_class": 0}, a, [(0, 4)]), ({"participant_id": "2", "gender": "", "severity_class": 1}, b, [(0, 2)])], ["A"])
    model = AVClassifier(2, 2, hidden_size=4, dropout=0, gender_dim=2)
    model.eval()
    logits = model(batch["audio"], batch["visual"], batch["lengths"], batch["gender"])
    changed = batch["audio"].clone()
    changed[1, 2:] = 1e8
    assert torch.allclose(logits, model(changed, batch["visual"], batch["lengths"], batch["gender"]))
    torch.nn.functional.cross_entropy(logits, batch["labels"]).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = AVClassifier(2, 2, hidden_size=4, dropout=0, gender_dim=2)
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    assert torch.allclose(logits, restored(batch["audio"], batch["visual"], batch["lengths"], batch["gender"]))
    p = torch.tensor([[1., 0, 0, 0, 0], [0, 1., 0, 0, 0], [0, 0, 1., 0, 0]])
    combined = aggregate_probabilities(p, torch.tensor([0, 0, 1]), 2)
    assert torch.allclose(combined[0], torch.tensor([.5, .5, 0, 0, 0]))


def test_unsupported_class_metrics():
    m = metrics([0, 0], np.array([[1, 0, 0, 0, 0]] * 2))
    assert m["macro_f1"] == .2 and m["balanced_accuracy"] == 1
    assert m["per_class"][4]["recall"] is None


def test_cache_fingerprint_invalidates_source_and_settings(tmp_path):
    from src.smoke_test import create_fixture
    from src.prepare_features import prepare_one
    from src.datasets import load_sequences
    c = create_fixture(tmp_path / "synthetic")
    first = prepare_one(c, "9000")
    c["confidence_threshold"] = .9
    with pytest.raises(ValueError, match="configuration changed"):
        load_sequences(joined_manifest(c), c)
    updated = prepare_one(c, "9000")
    assert first["fingerprint"] != updated["fingerprint"]


def test_end_to_end_resume_prediction_and_baseline(tmp_path):
    from src.smoke_test import create_fixture
    from src.train import baselines, evaluate_saved, train
    from src.predict import predict
    c = create_fixture(tmp_path / "synthetic")
    c["epochs"] = 1
    run = tmp_path / "run"
    report = train(c, run, gender=True, balanced=True)
    assert report["participants"] == 5 and report["data_kind"] == "SYNTHETIC_CORRECTNESS_ONLY"
    a = predict(run, "9025")
    b = predict(run, "9025")
    assert np.allclose(a["probabilities"], b["probabilities"])
    resumed = train(c, run, gender=True, balanced=True, resume=True)
    assert resumed["macro_f1"] == report["macro_f1"]
    baselines(c, tmp_path / "base")
    assert evaluate_saved(tmp_path / "base/logistic", "test")["participants"] == 5
    with pytest.raises(ValueError, match="validation/test"):
        predict(run, "9000")
    ci = paired_bootstrap(run / "validation/predictions.csv", run / "validation/predictions.csv", repetitions=10)
    assert ci["accuracy"]["delta_second_minus_first_ci95"] == [0, 0]
