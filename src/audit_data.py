"""Inventory only workspace data; do not decompress 108GB during discovery."""
import argparse
from pathlib import Path
import re

import pandas as pd

from .common import read_metadata, write_json


def audit(dataset, output, metadata=None):
    root, out = Path(dataset), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for p in sorted(root.iterdir()):
        match = re.match(r"^(\d+)_P(?: \(\d+\)| \d+)?(?:\.tar\.gz|\.tgz|\.zip|\.tar)?$", p.name)
        if match:
            rows.append({"participant_id": match[1], "path": str(p), "kind": "directory" if p.is_dir() else "archive", "bytes": p.stat().st_size if p.is_file() else None, "contents_verified": p.is_dir()})
    df = pd.DataFrame(rows)
    df.to_csv(out / "inventory.csv", index=False)
    duplicates = df[df.duplicated("participant_id", keep=False)] if len(df) else df
    duplicates.to_csv(out / "duplicate_sources.csv", index=False)
    counts = {"archives": int((df.kind == "archive").sum()) if len(df) else 0, "extracted_directories": int((df.kind == "directory").sum()) if len(df) else 0, "unique_participant_ids": int(df.participant_id.nunique()) if len(df) else 0, "archive_bytes": int(df.bytes.fillna(0).sum()) if len(df) else 0, "metadata_found": bool(metadata and Path(metadata).exists()), "archive_content_audit": "Inventory only; full content checks are reported separately in results/full_dataset/summary.json when generated", "prior_test_use": "Unknown; no experiment history supplied"}
    if metadata and Path(metadata).exists():
        meta = read_metadata(metadata)
        for cols in [["gender"], ["severity_class"], ["gender", "severity_class"]]:
            meta.groupby(cols, dropna=False).size().rename("participants").reset_index().to_csv(out / ("counts_" + "_".join(cols) + ".csv"), index=False)
        counts["labeled_participants"] = len(meta)
    else:
        pd.DataFrame(columns=["participant_id", "gender", "phq8_score", "split", "speech_path"]).to_csv(out / "metadata_template.csv", index=False)
        counts["blocked"] = "PHQ-8 scores, recorded gender, and official split provenance not supplied"
    write_json(out / "audit.json", counts)
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--output", default="results/audit")
    p.add_argument("--metadata", default="data/metadata/participants.csv")
    args = p.parse_args()
    print(audit(args.dataset, args.output, args.metadata))


if __name__ == "__main__":
    main()
