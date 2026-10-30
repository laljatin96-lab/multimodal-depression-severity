"""Reconcile downloaded USC E-DAIC records without guessing questionnaire labels."""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from .common import load_config, severity, validate_dataset_coverage, write_json


def reconcile(root):
    root = Path(root)
    frames = []
    for filename, split in [("train_split.csv", "train"), ("dev_split.csv", "validation"), ("test_split.csv", "test")]:
        frame = pd.read_csv(root / filename, dtype={"Participant_ID": str}, keep_default_na=False)
        frame["split"] = split
        frames.append(frame)
    official = pd.concat(frames, ignore_index=True).set_index("Participant_ID").sort_index()
    detailed = pd.read_csv(root / "detailed_lables.csv", dtype={"Participant": str}, keep_default_na=False).set_index("Participant").sort_index()
    if official.index.duplicated().any() or detailed.index.duplicated().any() or set(official.index) != set(detailed.index):
        raise ValueError("Official records have duplicate, overlapping or missing participant IDs")
    if not official.index.to_series().str.fullmatch(r"\d+").all():
        raise ValueError("Invalid official participant IDs")
    items = detailed.filter(regex=r"^PHQ8_[1-8]_").apply(pd.to_numeric, errors="raise")
    if items.shape[1] != 8 or not np.isfinite(items.to_numpy()).all() or not items.isin([0, 1, 2, 3]).all().all():
        raise ValueError("Invalid eight-item questionnaire responses")
    totals = pd.to_numeric(official.PHQ_Score, errors="raise")
    classes = totals.map(severity)
    if not totals.eq(items.sum(axis=1)).all() or not totals.eq(pd.to_numeric(detailed.Depression_severity)).all():
        raise ValueError("Current official totals disagree with item responses; manual resolution required")
    if not official['split'].eq(detailed['split'].replace({'dev': 'validation'})).all():
        raise ValueError("Official split files disagree")
    left, right = official.Gender.str.strip(), detailed.gender.str.strip()
    known = left.eq(right) & left.isin(["male", "female"])
    issues = []
    for pid in official.index[~known]:
        issues.append({"participant_id": pid, "issue": "recorded_gender_conflict_or_unknown", "resolution": "unknown", "split_value": left[pid], "detailed_value": right[pid]})
    for pid in official.index[pd.to_numeric(official.PHQ_Binary).ne(totals.ge(10).astype(int))]:
        issues.append({"participant_id": pid, "issue": "binary_flag_disagrees_with_score_threshold", "resolution": "binary flag unused; five classes derived from validated total"})
    older_path = root / "Detailed_PHQ8_Labels.csv"
    if older_path.exists():
        older = pd.read_csv(older_path, dtype={"Participant_ID": str}).set_index("Participant_ID")
        if older.index.duplicated().any() or not set(older.index) <= set(official.index):
            raise ValueError("Invalid older questionnaire IDs")
        for pid in older.index[older.PHQ_8Total.ne(totals.loc[older.index])]:
            issues.append({"participant_id": pid, "issue": "older_219_record_total_disagrees", "resolution": "current split total agrees with current 275-record eight-item sum"})
    manifest = pd.DataFrame({"participant_id": official.index, "gender": left.where(known, "unknown").to_numpy(), "phq8_score": totals.astype(int).to_numpy(), "severity_class": classes.to_numpy(), "split": official['split'].to_numpy()})
    return manifest, pd.DataFrame(issues)


def verify_archives(root, dataset, expected_ids):
    rows = re.findall(r"(\d+)_P\.tar\.gz\s*\(([\d,.]+) KB\)", (Path(root) / "source_index.txt").read_text())
    if len(rows) != len(set(pid for pid, _ in rows)) or {pid for pid, _ in rows} != set(expected_ids):
        raise ValueError("Provider archive index does not exactly match participant IDs")
    comparisons = []
    for pid, size in rows:
        path = Path(dataset) / f"{pid}_P.tar.gz"
        local_size = path.stat().st_size
        rounded = (Decimal(local_size) / 1024).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        remote_size = Decimal(size.replace(',', ''))
        if rounded != remote_size:
            raise ValueError(f"Provider archive size mismatch for {pid}")
        comparisons.append({"participant_id": pid, "local_bytes": local_size, "provider_display_kib": str(remote_size), "match": True})
    return comparisons


def import_records(config, root):
    root = Path(root)
    manifest, issues = reconcile(root)
    validate_dataset_coverage(manifest, config)
    comparisons = verify_archives(root, config['dataset'], manifest.participant_id)
    destination = Path(config['metadata'])
    if destination.exists():
        existing = pd.read_csv(destination, keep_default_na=False, dtype=str)
        same = existing.to_csv(index=False) == manifest.astype(str).to_csv(index=False)
        if not same and any(existing.get(col, pd.Series(dtype=str)).ne('').any() for col in ['gender', 'phq8_score', 'split', 'speech_path']):
            raise FileExistsError("Preserving existing populated metadata; reconcile user edits before importing")
        if not same:
            backup = destination.with_name('participants_before_official_import.csv')
            if not backup.exists():
                shutil.copyfile(destination, backup)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination, index=False)
    issues.to_csv(root / 'reconciliation_issues.csv', index=False)
    pd.DataFrame(comparisons).to_csv(root / 'archive_identity_checks.csv', index=False)
    inputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.name in ['train_split.csv', 'dev_split.csv', 'test_split.csv', 'detailed_lables.csv', 'Detailed_PHQ8_Labels.csv', 'source_index.txt']}
    report = {"participants": len(manifest), "splits": manifest['split'].value_counts().to_dict(), "recorded_gender_counts": manifest.gender.value_counts().to_dict(), "issue_counts": issues.issue.value_counts().to_dict(), "all_eight_item_sums_match_current_totals": True, "archive_identity_evidence": "All IDs, published sizes rounded to 0.1 KiB and feature filenames match; provider archive cryptographic hashes unavailable.", "archive_size_matches": len(comparisons), "source": "https://dcapswoz.ict.usc.edu/wwwedaic/index.php?dir=labels", "index_source": "https://dcapswoz.ict.usc.edu/wwwedaic/index.php?dir=data", "acquisition": "Standard browser download links, 2026-10-23; no local dataset uploaded", "input_sha256": inputs, "score_policy": "Current split PHQ_Score, corroborated by all eight current detailed questionnaire items; older 219-record version discrepancies retained", "gender_policy": "Whitespace trimmed; only agreement between current official files retained, conflicts/unknown mapped to unknown; no inference from media", "test_exposure": "Metadata integrity checks only; no test model performance evaluated or used for selection; prior user test use unknown"}
    write_json(root / 'import_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--source', default='data/metadata/official_edaic')
    args = parser.parse_args()
    print(json.dumps(import_records(load_config(args.config), args.source), indent=2))


if __name__ == '__main__':
    main()
