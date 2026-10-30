import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "av-phq8-v1"


def load_config(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def severity(score):
    score = float(score)
    if not np.isfinite(score) or not score.is_integer() or not 0 <= score <= 24:
        raise ValueError(f"PHQ-8 must be an integer in [0,24], got {score}")
    return min(int(score) // 5, 4)


def read_metadata(path, require_labels=True):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing participant metadata: {path}. Required columns: participant_id,gender,phq8_score; optional split, speech_path.")
    df = pd.read_csv(path, dtype={"participant_id": str, "gender": str}, keep_default_na=False)
    needed = {"participant_id", "gender"} | ({"phq8_score"} if require_labels else set())
    if not needed <= set(df):
        raise ValueError(f"Missing metadata columns: {sorted(needed - set(df))}")
    if len(df) == 0 or df.participant_id.eq("").any() or not df.participant_id.str.fullmatch(r"\d+").all():
        raise ValueError("Metadata requires nonempty numeric participant IDs")
    if df.participant_id.duplicated().any():
        raise ValueError("Duplicate/conflicting participant records; resolve explicitly")
    if require_labels:
        labels, invalid = [], []
        for pid, score in zip(df.participant_id, df.phq8_score):
            try:
                labels.append(severity(score))
            except (ValueError, TypeError):
                invalid.append(pid)
                labels.append(None)
        if invalid:
            raise ValueError(f"Missing/invalid PHQ-8 scores for {len(invalid)} participants: {', '.join(invalid[:10])}. Supply original integer questionnaire totals in [0,24].")
        mapped = pd.Series(labels, index=df.index, dtype=int)
        if "severity_class" in df and not np.array_equal(pd.to_numeric(df.severity_class), mapped):
            raise ValueError("severity_class disagrees with PHQ-8")
        df["severity_class"] = mapped
    return df


def validate_dataset_coverage(metadata, config):
    """Preserve all-participant scope when a matching full-dataset audit exists."""
    if not config.get("dataset"):
        return
    root = Path(config.get("full_dataset_report", "results/full_dataset"))
    run_path = root / "run.json"
    if not run_path.exists():
        return
    run = json.loads(run_path.read_text())
    if Path(run["config"]["dataset"]).resolve() != Path(config["dataset"]).resolve():
        return
    summary = json.loads((root / "summary.json").read_text())
    if not summary.get("complete"):
        raise ValueError("Full-dataset audit is incomplete; finish --all-sources before splitting/training")
    expected = set(pd.read_csv(root / "participants.csv", dtype={"participant_id": str}).participant_id)
    actual = set(metadata.participant_id)
    if expected != actual:
        raise ValueError(f"Metadata must cover all {len(expected)} audited participants: {len(expected - actual)} missing IDs, {len(actual - expected)} unknown IDs. Do not silently drop participants without labels.")


def encode_gender(value, categories):
    if len(set(categories)) != len(categories) or "" in categories:
        raise ValueError("Gender categories must be unique explicit nonempty original labels")
    out = np.zeros(len(categories) + 1, dtype=np.float32)
    out[categories.index(value) if value in categories else len(categories)] = 1
    return out


def joined_manifest(config):
    meta = read_metadata(config["metadata"])
    validate_dataset_coverage(meta, config)
    split = pd.read_csv(config["splits"], dtype={"participant_id": str})
    if split.participant_id.duplicated().any() or not set(split.split) <= {"train", "validation", "test"}:
        raise ValueError("Invalid or overlapping split manifest")
    if set(split.participant_id) != set(meta.participant_id):
        raise ValueError("Split and metadata participant IDs must match exactly")
    return meta.merge(split[["participant_id", "split"]], on="participant_id", validate="one_to_one", suffixes=("_metadata", ""))
