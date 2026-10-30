"""Saved-model held-out participant demo; never infer gender from media."""
import argparse
import json
from pathlib import Path

import torch

from .common import joined_manifest
from .datasets import load_sequences
from .train import choose_device, loader_for, make_model, predict_loader, provenance


def predict(run, pid):
    state = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    c = state["config"]
    manifest = joined_manifest(c)
    if provenance(manifest, c) != state["provenance"]:
        raise ValueError("Data provenance differs from checkpoint")
    frame = manifest[manifest.participant_id == pid].reset_index(drop=True)
    if len(frame) != 1 or frame.iloc[0].split == "train":
        raise ValueError("Demo requires exactly one validation/test participant")
    seqs = load_sequences(frame, c)
    device = choose_device(c["device"])
    model = make_model(c, seqs, state["gender"]).to(device)
    model.load_state_dict(state["model"])
    _, p = predict_loader(model, loader_for(frame, seqs, state["scalers"], c), device)
    return {"participant_id": pid, "split": frame.iloc[0].split, "predicted_class": int(p[0].argmax()), "probabilities": p[0].tolist(), "actual_class": int(frame.iloc[0].severity_class), "interpretation": "Research target; not a clinical diagnosis or calibrated certainty"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--participant", required=True)
    args = p.parse_args()
    print(json.dumps(predict(args.run, args.participant), indent=2))


if __name__ == "__main__":
    main()
