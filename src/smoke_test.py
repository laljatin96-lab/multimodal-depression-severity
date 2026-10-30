"""SYNTHETIC fixture exercise, never a real depression experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import load_config, write_json
from .prepare_features import AUDIO_COLUMNS, VISUAL_COLUMNS, prepare_one
from .predict import predict
from .train import baselines, evaluate_saved, train


def create_fixture(output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Preserving previous synthetic exercise: {out}")
    out.mkdir(parents=True)
    c = load_config("configs/default.json")
    c.update({"dataset": str(out / "raw_SYNTHETIC"), "metadata": str(out / "participants_SYNTHETIC.csv"), "splits": str(out / "splits_SYNTHETIC.csv"), "cache": str(out / "cache_SYNTHETIC"), "gender_categories": ["fixture_A", "fixture_B"], "timestamp_origins_verified": True, "allow_unsegmented": True, "epochs": 2, "patience": 2, "hidden_size": 4, "batch_size": 5, "window_seconds": 4, "stride_seconds": 4, "device": "cpu", "data_kind": "SYNTHETIC_CORRECTNESS_ONLY", "experiment_log": str(out / "SYNTHETIC_LOG.csv")})
    rng = np.random.default_rng(912)
    metadata = []
    for i in range(30):
        pid, target = str(9000 + i), i % 5
        feature_dir = Path(c["dataset"]) / f"{pid}_P" / "features"
        feature_dir.mkdir(parents=True)
        at, vt = np.arange(0, 12, 0.1), np.arange(0, 12, 0.2)
        audio = pd.DataFrame(rng.normal(size=(len(at), len(AUDIO_COLUMNS))), columns=AUDIO_COLUMNS)
        audio.insert(0, "frameTime", at)
        audio.to_csv(feature_dir / f"{pid}_OpenSMILE2.3.0_mfcc.csv", sep=";", index=False)
        visual = pd.DataFrame(rng.normal(size=(len(vt), len(VISUAL_COLUMNS))), columns=VISUAL_COLUMNS)
        visual["timestamp"], visual["success"], visual["confidence"] = vt, 1, 0.99
        visual.to_csv(feature_dir / f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv", index=False)
        prepare_one(c, pid)
        metadata.append({"participant_id": pid, "gender": "fixture_A" if i % 2 else "fixture_B", "phq8_score": target * 5, "split": "train" if i < 20 else "validation" if i < 25 else "test"})
    pd.DataFrame(metadata).to_csv(c["metadata"], index=False)
    pd.DataFrame(metadata)[["participant_id", "split"]].to_csv(c["splits"], index=False)
    write_json(out / "config_SYNTHETIC.json", c)
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/synthetic_smoke")
    args = p.parse_args()
    c = create_fixture(args.output)
    baselines(c, Path(args.output) / "baselines_SYNTHETIC")
    for gender, balanced in [(False, False), (False, True), (True, False), (True, True)]:
        run = Path(args.output) / (f"AV_g{int(gender)}_b{int(balanced)}_SYNTHETIC")
        train(c, run, gender, balanced)
        evaluate_saved(run, "test")
    demo = predict(Path(args.output) / "AV_g0_b0_SYNTHETIC", "9025")
    demo["data_kind"] = "SYNTHETIC_CORRECTNESS_ONLY"
    write_json(Path(args.output) / "demo_SYNTHETIC.json", demo)
    print(json.dumps({"status": "synthetic end-to-end smoke completed", "output": args.output}))


if __name__ == "__main__":
    main()
