import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .common import digest, load_config, read_metadata, validate_dataset_coverage, write_json


def split_ids(meta, seed):
    notes = []
    if "split" in meta:
        if not set(meta.split) <= {"train", "validation", "test"} or set(meta.split) != {"train", "validation", "test"}:
            raise ValueError("Official split column must contain train, validation and test")
        return meta[["participant_id", "split"]].copy(), ["Preserved supplied official/custom split column; provenance must be documented by data owner"]
    if len(meta) < 20:
        raise ValueError("Too few labeled participants for default 70/15/15 split")
    joint = meta.gender + ":" + meta.severity_class.astype(str)
    for strat, name in [(joint, "gender-severity"), (meta.severity_class, "severity"), (None, "unstratified")]:
        try:
            train, held = train_test_split(meta.index, test_size=0.3, random_state=seed, stratify=strat)
            val, test = train_test_split(held, test_size=0.5, random_state=seed, stratify=None if strat is None else strat.loc[held])
            if set(meta.loc[train, "severity_class"]) != set(meta.severity_class):
                raise ValueError("Training misses an observed severity class")
            result = pd.concat([pd.DataFrame({"participant_id": meta.loc[ix, "participant_id"], "split": split}) for ix, split in [(train, "train"), (val, "validation"), (test, "test")]])
            return result, notes + [f"Fixed participant 70/15/15; chosen {name} stratification"]
        except ValueError as exc:
            notes.append(f"{name} infeasible: {exc}")
    raise ValueError("No viable split")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.json")
    args = p.parse_args()
    c = load_config(args.config)
    path = Path(c["splits"])
    if path.exists():
        raise FileExistsError(f"Preserving existing split: {path}")
    meta = read_metadata(c["metadata"])
    validate_dataset_coverage(meta, c)
    split, notes = split_ids(meta, c["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    split.sort_values("participant_id").to_csv(path, index=False)
    write_json(path.with_suffix(".json"), {"seed": c["seed"], "notes": notes, "metadata_hash": digest(meta.to_dict("records")), "participant_overlap": False, "prior_test_use": "unknown; obtain data-owner history before claiming untouched test"})
    meta.drop(columns=["split"], errors="ignore").merge(split).groupby(["split", "gender", "severity_class"]).size().rename("participants").to_csv(path.with_name("counts.csv"))


if __name__ == "__main__":
    main()
