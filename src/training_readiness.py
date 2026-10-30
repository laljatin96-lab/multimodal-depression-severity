"""Prepare the complete label handoff and report real research prerequisites."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import encode_gender, joined_manifest, load_config, read_metadata, severity, validate_dataset_coverage, write_json
from .full_dataset import discover_sources, source_stamp
from .prepare_features import speech_mask


def audited_participants(config):
    root = Path(config.get("full_dataset_report", "results/full_dataset"))
    run = json.loads((root / "run.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if Path(run["config"]["dataset"]).resolve() != Path(config["dataset"]).resolve() or not summary.get("complete"):
        raise ValueError("A completed full-dataset audit for this dataset is required")
    return root, pd.read_csv(root / "participants.csv", dtype={"participant_id": str}, keep_default_na=False)


def initialize_metadata(config):
    _, participants = audited_participants(config)
    path = Path(config["metadata"])
    if path.exists():
        return "Existing metadata preserved"
    frame = participants[["participant_id"]].copy()
    frame["gender"] = ""
    frame["phq8_score"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects any label file created concurrently.
    with path.open("x") as f:
        frame.to_csv(f, index=False)
    return f"Created {len(frame)} rows at {path}; PHQ-8/gender remain empty"


def check_readiness(config, output="results/training_readiness"):
    root, participants = audited_participants(config)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    blockers, issues = [], []
    expected = set(participants.participant_id)
    # Stat checks reuse the completed checksum audit without rereading archives.
    sources = pd.read_csv(root / "sources.csv", keep_default_na=False)
    current, unrecognized = discover_sources(config["dataset"])
    sources_current = set(sources.source) == {str(s["path"]) for s in current} and not unrecognized
    for key in sources.source_key:
        report = json.loads((root / "sources" / f"{key}.json").read_text())
        if not Path(report["source"]).exists() or source_stamp(Path(report["source"])) != report["source_stamp"]:
            sources_current = False
    if not sources_current:
        blockers.append("Source inventory/content stamps changed; rerun --all-sources")
    feature_paths = participants.egemaps_feature_path if config["audio_kind"] == "egemaps" else participants.audio_feature_path
    feature_count = sum(bool(p) and Path(p).is_file() for p in feature_paths)
    if feature_count != len(participants):
        blockers.append(f"Requested {config['audio_kind']}/OpenFace features unavailable for {len(participants) - feature_count} participants")
    metadata, labels_ok, coverage_ok = None, False, False
    try:
        metadata = read_metadata(config["metadata"], require_labels=False)
        validate_dataset_coverage(metadata, config)
        coverage_ok = True
    except (ValueError, FileNotFoundError) as exc:
        blockers.append(str(exc))
    valid_scores = {}
    if metadata is not None:
        for row in metadata.to_dict("records"):
            pid, score = row["participant_id"], row.get("phq8_score", "")
            try:
                valid_scores[pid] = severity(score)
            except (TypeError, ValueError):
                issues.append({"participant_id": pid, "issue": "missing_phq8" if str(score).strip() == "" else "invalid_phq8", "provided_value": score})
        for pid in sorted(expected - set(metadata.participant_id)):
            issues.append({"participant_id": pid, "issue": "missing_metadata_row", "provided_value": ""})
        try:
            read_metadata(config["metadata"])
            labels_ok = coverage_ok
        except (ValueError, FileNotFoundError) as exc:
            blockers.append(str(exc))
    else:
        issues = [{"participant_id": pid, "issue": "missing_metadata_file", "provided_value": ""} for pid in sorted(expected)]
    origins_ok = bool(config["timestamp_origins_verified"])
    if not origins_ok:
        blockers.append("Confirm/document timestamp origins and offsets before research preparation")
    speech_ok = True
    if metadata is None:
        speech_ok = bool(config["allow_unsegmented"])
    else:
        for row in metadata.to_dict("records"):
            path = row.get("speech_path", "")
            if not path and not config["allow_unsegmented"]:
                speech_ok = False
            if path:
                try:
                    intervals = pd.read_csv(path)[["start", "end"]].to_numpy(float)
                    if not len(intervals):
                        raise ValueError("No intervals")
                    speech_mask(np.empty(0), intervals)
                except (ValueError, OSError, KeyError) as exc:
                    speech_ok = False
                    issues.append({"participant_id": row["participant_id"], "issue": "invalid_speech_intervals", "provided_value": str(exc)})
    if not speech_ok:
        blockers.append("Provide verified participant speech intervals or explicitly document/configure unsegmented use")
    split_ok, cache_ok, split_frame = False, False, None
    if labels_ok:
        try:
            split_frame = joined_manifest(config)
            split_ok = set(split_frame.split) == {"train", "validation", "test"}
            if not split_ok:
                blockers.append("Train, validation and test must all have participants")
            elif split_frame.loc[split_frame.split == "train", "severity_class"].nunique() < 2:
                split_ok = False
                blockers.append("Training requires at least two observed classes for the logistic comparison")
        except (ValueError, FileNotFoundError) as exc:
            blockers.append(f"Participant splits unavailable/invalid: {exc}")
    preprocessing_ok = labels_ok and origins_ok and speech_ok and sources_current and feature_count == len(participants)
    if preprocessing_ok and split_ok:
        try:
            from .datasets import load_sequences
            load_sequences(split_frame, config)
            cache_ok = True
        except (ValueError, FileNotFoundError, KeyError) as exc:
            blockers.append(f"Research cache requires preparation: {exc}")
    genders = metadata.gender if metadata is not None else pd.Series(dtype=str)
    recognized_gender_count = int(genders.isin(config["gender_categories"]).sum())
    recognized_train_gender = int(split_frame.loc[split_frame.split == "train", "gender"].isin(config["gender_categories"]).sum()) if split_frame is not None else 0
    recorded_gender_count = int((genders.ne("") & genders.str.lower().ne("unknown")).sum())
    encoding_ok = True
    try:
        encode_gender("", config["gender_categories"])
    except ValueError as exc:
        encoding_ok = False
        blockers.append(str(exc))
    ready = preprocessing_ok and split_ok and cache_ok and encoding_ok
    result = {"status": "ready" if ready else "blocked", "participants": len(participants), "source_audit_current": sources_current, "participants_with_requested_features": feature_count, "metadata_rows": len(metadata) if metadata is not None else 0, "complete_participant_coverage": coverage_ok, "valid_phq8_scores": sum(pid in expected for pid in valid_scores), "unlabeled_or_invalid_participants": len(expected - set(valid_scores)), "recorded_gender_entries": recorded_gender_count, "recognized_gender_entries": recognized_gender_count, "ready_for_split_creation": labels_ok and sources_current and feature_count == len(participants), "ready_for_research_preparation": preprocessing_ok, "ready_for_av_training": ready, "ready_for_gender_training": ready and recognized_train_gender > 0, "gender_note": "Missing recorded gender does not block AV-only models; gender-aware comparisons require original recorded categories and explicit encoding", "blockers": list(dict.fromkeys(blockers)), "next_command": ".venv/bin/python -m src.training_readiness", "data_kind": "PROVIDED_DATASET; labels must come from original records"}
    pd.DataFrame(issues, columns=["participant_id", "issue", "provided_value"]).to_csv(out / "participant_issues.csv", index=False)
    write_json(out / "readiness.json", result)
    lines = ["# Paired-window model training readiness", "", f"Status: **{result['status']}**", "", f"Prepared participants: {len(participants)}; valid original PHQ-8 totals: {result['valid_phq8_scores']}; missing/invalid totals: {result['unlabeled_or_invalid_participants']}.", "", "## Required before paired-window experiments", ""]
    lines += [f"- {item}" for item in result["blockers"]]
    lines += ["", result["gender_note"], "", "No real training, split creation, label inference or source re-extraction is performed by this check. Exploratory independently pooled summary baselines and LSTMs have separate protocols/reports in results/exploratory_baselines/ and results/independent_lstm/. This check does not report those experiments as incomplete.", "", "After resolving the listed prerequisites, rerun:", "", "```bash", result["next_command"], "```", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--output", default="results/training_readiness")
    p.add_argument("--init-metadata", action="store_true")
    p.add_argument("--require-ready", action="store_true", help="Exit 2 unless AV training prerequisites pass")
    args = p.parse_args()
    c = load_config(args.config)
    if args.init_metadata:
        print(initialize_metadata(c))
    report = check_readiness(c, args.output)
    print(json.dumps(report, indent=2))
    if args.require_ready and not report["ready_for_av_training"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
