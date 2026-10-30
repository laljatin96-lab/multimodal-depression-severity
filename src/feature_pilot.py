"""Validate provided features numerically without generating clinical labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import load_config, write_json
from .datasets import collate_participants, fit_scalers, transform, window_indices
from .models import AVClassifier
from .prepare_features import prepare_one


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.json")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--ids", nargs="+")
    selection.add_argument("--all-prepared", action="store_true", help="Run an unlabeled numerical probe for every prepared participant")
    p.add_argument("--report", default="results/full_dataset")
    p.add_argument("--output", default="results/audit")
    args = p.parse_args()
    c = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    records, rows = [], []
    if args.all_prepared:
        report = Path(args.report)
        summary = json.loads((report / "summary.json").read_text())
        if not summary.get("complete"):
            raise ValueError("Complete --all-sources preparation first")
        manifest = pd.read_csv(report / "participants.csv", dtype={"participant_id": str}, keep_default_na=False)
        if not manifest.feature_ready.astype(str).str.lower().eq("true").all():
            raise ValueError("Some participants are not feature-ready; inspect unusable_participants.csv")
        records = [(row.participant_id, Path(row.audio_feature_path)) for row in manifest.itertuples()]
    else:
        for pid in args.ids or ["303", "306", "487"]:
            prepare_one(c, pid, pilot=True)
            records.append((pid, Path(c["cache"]) / "unlabeled_pilot" / f"{pid}.npz"))
    for pid, cache_path in records:
        info = json.loads(cache_path.with_suffix(".json").read_text())
        with np.load(cache_path, allow_pickle=False) as z:
            s = {k: z[k] for k in z.files}
        scaler = fit_scalers({pid: s}, [pid])
        normalized = transform(s, scaler)
        indices = window_indices(normalized, c)
        probe_indices = indices if args.all_prepared else indices[:1]
        batch = collate_participants([({"participant_id": pid, "gender": ""}, normalized, probe_indices)], [])
        torch.manual_seed(42)
        model = AVClassifier(s["audio"].shape[1] + 1, s["visual"].shape[1] + 1, hidden_size=4, dropout=0)
        logits = model(batch["audio"], batch["visual"], batch["lengths"])
        # Numerical probe only. No fabricated PHQ-8 targets or severity predictions.
        logits.square().mean().backward()
        rows.append({"participant_id": pid, "audio_frames": info["audio_frames"], "visual_frames": info["visual_frames"], "audio_features": len(info["audio_columns"]), "visual_features": len(info["visual_columns"]), "common_seconds": info["common_seconds"], "tracking_valid_fraction": info["tracking_valid_fraction"], "aligned_bins": len(s["timestamps"]), "selected_windows": len(indices), "probed_windows": len(probe_indices), "finite_forward_backward": bool(torch.isfinite(logits).all() and all(torch.isfinite(param.grad).all() for param in model.parameters()))})
        if args.all_prepared and len(rows) % 25 == 0:
            print(f"Numerically validated {len(rows)}/{len(records)} participants", flush=True)
    pd.DataFrame(rows).to_csv(out / "real_feature_pilot.csv", index=False)
    write_json(out / "real_feature_pilot.json", {"kind": "UNLABELED_FEATURE_AND_NUMERICAL_VALIDATION_ONLY", "participants": rows, "limitations": ["No PHQ-8 targets or recorded gender supplied", "No supervised real-data training, accuracy, or held-out research prediction", "Unsegmented audio; interviewer speech may be present", "Common timestamp origin assumed, not independently verified", "Per-participant scaling here is a numerical probe only; research scalers fit training IDs only"]})
    if not pd.DataFrame(rows).finite_forward_backward.all():
        raise ValueError("Nonfinite numerical model probe; see saved validation report")
    print(f"Validated {len(rows)} participants; report: {out / 'real_feature_pilot.csv'}")


if __name__ == "__main__":
    main()
