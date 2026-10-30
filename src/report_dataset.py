"""Summarize a completed full-source audit without labels or new training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .common import write_json
from .full_dataset import discover_sources


def report_dataset(input_dir="results/full_dataset", plots=True):
    root = Path(input_dir)
    summary = json.loads((root / "summary.json").read_text())
    if not summary["complete"] or summary["processed_sources"] != summary["expected_sources"]:
        raise ValueError("Full source processing has not completed")
    run = json.loads((root / "run.json").read_text())
    sources = pd.read_csv(root / "sources.csv", dtype={"participant_id": str}, keep_default_na=False)
    current, other = discover_sources(run["config"]["dataset"])
    if set(sources.source) != {str(s["path"]) for s in current}:
        raise ValueError("Source inventory changed; rerun --all-sources to include every current source")
    participants = pd.read_csv(root / "participants.csv", dtype={"participant_id": str}, keep_default_na=False)
    reports = {key: json.loads((root / "sources" / f"{key}.json").read_text()) for key in sources.source_key}
    members, errors, candidates = [], [], []
    for source in reports.values():
        for entry in source.get("members", []):
            row = {"source": source["source"], "participant_id": source["participant_id"], "member": entry["member"], "bytes": entry["bytes"], "sha256": entry.get("sha256", ""), "columns": ";".join(entry.get("columns", []))}
            members.append(row)
            if entry.get("metadata_candidate") or any(re.search(r"phq|gender|severity|depression.*score", c, re.I) for c in entry.get("columns", [])):
                candidates.append(row)
        errors.extend({"source": source["source"], "participant_id": source["participant_id"], "reason": reason} for reason in source["errors"])
    pd.DataFrame(members).to_csv(root / "member_inventory.csv", index=False)
    pd.DataFrame(errors, columns=["source", "participant_id", "reason"]).to_csv(root / "processing_errors.csv", index=False)
    pd.DataFrame(candidates, columns=["source", "participant_id", "member", "bytes", "sha256", "columns"]).to_csv(root / "metadata_candidates.csv", index=False)

    # Audit duplicate processed content across different participant identifiers.
    content_groups = {}
    ready = participants[participants.feature_ready.astype(str).str.lower() == "true"]
    for row in ready.itertuples():
        with np.load(row.audio_feature_path, allow_pickle=False) as data:
            h = hashlib.sha256()
            for key in ["audio", "visual", "audio_mask", "visual_mask", "timestamps"]:
                h.update(data[key].tobytes())
            content_groups.setdefault(h.hexdigest(), []).append(row.participant_id)
    duplicate_content = [{"content_sha256": h, "participant_id": pid, "group_participants": len(ids)} for h, ids in content_groups.items() if len(ids) > 1 for pid in ids]
    pd.DataFrame(duplicate_content, columns=["content_sha256", "participant_id", "group_participants"]).to_csv(root / "duplicate_participant_content.csv", index=False)

    stats = []
    for column in ["audio_frames", "visual_frames", "common_seconds", "tracking_valid_fraction"]:
        values = pd.to_numeric(ready[column])
        stats.append({"feature": column, "participants": len(values), "minimum": float(values.min()) if len(values) else None, "median": float(values.median()) if len(values) else None, "mean": float(values.mean()) if len(values) else None, "maximum": float(values.max()) if len(values) else None})
    pd.DataFrame(stats).to_csv(root / "feature_statistics.csv", index=False)
    counts = {"processed_archives": summary["processed_archives"], "processed_directories": summary["processed_sources"] - summary["processed_archives"], "integrity_checked_sources": summary["integrity_checked_sources"], "participant_ids": len(participants), "mfcc_visual_ready_participants": len(ready), "egemaps_visual_ready_participants": int(ready.egemaps_feature_path.ne("").sum()), "duplicate_participant_groups": int((participants.source_count.astype(int) > 1).sum()), "conflicting_duplicate_groups": summary["duplicate_conflicts"], "duplicate_content_participant_groups": sum(len(ids) > 1 for ids in content_groups.values()), "metadata_candidate_members": len(candidates), "processing_error_reasons": dict(Counter(row["reason"] for row in errors)), "unrecognized_items": other, "data_kind": "PROVIDED_DATASET_UNLABELED", "supervised_metrics": None}
    counts["common_duration_hours"] = float(pd.to_numeric(ready.common_seconds).sum() / 3600)
    counts["total_audio_frames"] = int(pd.to_numeric(ready.audio_frames).sum())
    counts["total_visual_frames"] = int(pd.to_numeric(ready.visual_frames).sum())
    numerical_path = root / "numerical_validation" / "real_feature_pilot.csv"
    if numerical_path.exists():
        numerical = pd.read_csv(numerical_path, dtype={"participant_id": str})
        if numerical.participant_id.duplicated().any() or set(numerical.participant_id) != set(ready.participant_id):
            raise ValueError("Numerical validation participant IDs do not match current prepared manifest")
        counts["numerically_validated_participants"] = int(numerical.finite_forward_backward.astype(str).str.lower().eq("true").sum())
        counts["numerically_probed_windows"] = int(numerical.probed_windows.sum())
    write_json(root / "coverage.json", counts)
    if plots and len(ready):
        plot_quality(ready, root)
    lines = ["# Full supplied-dataset coverage", "", "Every discovered source is accounted for. This is feature validation, not supervised depression detection.", "", f"- Archives read to EOF: {counts['processed_archives']}", f"- Extracted directories checked: {counts['processed_directories']}", f"- Sources passing integrity checks: {counts['integrity_checked_sources']}", f"- Distinct participant IDs: {counts['participant_ids']}", f"- Participants with valid MFCC/OpenFace sequences: {counts['mfcc_visual_ready_participants']}", f"- Participants also having valid temporal eGeMAPS/OpenFace sequences: {counts['egemaps_visual_ready_participants']}", f"- Duplicate source groups: {counts['duplicate_participant_groups']}; conflicting groups: {counts['conflicting_duplicate_groups']}", f"- Matching processed content across different IDs: {counts['duplicate_content_participant_groups']} groups", f"- Possible metadata members: {counts['metadata_candidate_members']}", "", "All per-source errors are listed in processing_errors.csv. A failed optional eGeMAPS sequence does not discard a valid MFCC/OpenFace participant. Repeated copies of an ID are reconciled rather than counted as new participants.", "", "PHQ-8 targets and recorded gender remain unassigned. No labels, split-dependent transforms, severity predictions or performance claims are produced here. Timestamp origin verification and speech-selection limitations remain as recorded in each cache.", "", "## Files", "", "- sources.csv: one row per physical source", "- participants.csv: one row per participant with cache paths and explicit missing labels", "- member_inventory.csv: every archived/directory file inventoried", "- duplicate_resolution.csv: same-ID copy comparison", "- duplicate_participant_content.csv: cross-ID content audit", "- processing_errors.csv: source-specific preparation failures", "- metadata_candidates.csv: filenames/headers that might contain labels", "- feature_statistics.csv and quality.png: sequence duration and tracking summaries", "- sources/*.json: full source checksums, provenance, schemas and detailed modality checks", ""]
    if "numerically_validated_participants" in counts:
        lines += ["## Numerical model checks", "", f"Finite untrained-model forward/backward checks passed for {counts['numerically_validated_participants']} participants across {counts['numerically_probed_windows']} selected windows. These checks use no clinical labels and do not constitute supervised training.", "", "Details: numerical_validation/real_feature_pilot.csv.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    return counts


def plot_quality(frame, output):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].hist(pd.to_numeric(frame.common_seconds) / 60, bins=20, color="#287e8e", edgecolor="white")
    axes[0].set_xlabel("Common audio/visual duration (minutes)")
    axes[0].set_ylabel("Participants")
    axes[1].hist(pd.to_numeric(frame.tracking_valid_fraction) * 100, bins=20, color="#7257a5", edgecolor="white")
    axes[1].set_xlabel("Valid visual tracking (%)")
    axes[1].set_ylabel("Participants")
    fig.suptitle(f"Supplied dataset: {len(frame)} participants with valid MFCC/OpenFace features")
    fig.tight_layout()
    fig.savefig(Path(output) / "quality.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/full_dataset")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    print(json.dumps(report_dataset(args.input, plots=not args.no_plots), indent=2))


if __name__ == "__main__":
    main()
