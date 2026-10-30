"""Equal-budget four-way ablation; validation-only until explicit evaluation."""
import argparse
from pathlib import Path

import pandas as pd

from .common import load_config
from .train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--output", default="checkpoints/matrix")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2026])
    args = p.parse_args()
    c = load_config(args.config)
    rows = []
    for seed in args.seeds:
        for gender, balanced in [(False, False), (False, True), (True, False), (True, True)]:
            current = {**c, "seed": seed}
            name = "AV" + ("-gender" if gender else "") + ("-balanced" if balanced else "")
            run = Path(args.output) / f"{name}_seed{seed}"
            report = train(current, run, gender, balanced, resume=(run / "latest.pt").exists())
            rows.append({k: report[k] for k in ["model", "seed", "accuracy", "macro_f1", "balanced_accuracy", "participants"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(args.output) / "validation_runs.csv", index=False)
    frame.groupby("model")[["accuracy", "macro_f1", "balanced_accuracy"]].agg(["mean", "std"]).to_csv(Path(args.output) / "validation_summary.csv")


if __name__ == "__main__":
    main()
