"""Promote compatible full-dataset caches after provenance is confirmed.

New speech intervals or temporal settings require source-level reprocessing.
No scaler, windows, split or labels are learned during promotion.
"""
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from .common import VERSION, digest, write_json

TRANSFORM_SETTINGS = ["audio_kind", "audio_time_offset", "visual_time_offset", "confidence_threshold", "bin_seconds", "minimum_window_steps"]


def reuse_full_cache(config, pid):
    report_root = Path(config.get("full_dataset_report", "results/full_dataset"))
    summary_path = report_root / "summary.json"
    if not summary_path.exists() or not json.loads(summary_path.read_text()).get("complete"):
        return None
    run_path = report_root / "run.json"
    if not run_path.exists() or Path(json.loads(run_path.read_text())["config"]["dataset"]).resolve() != Path(config["dataset"]).resolve():
        return None
    if not config["timestamp_origins_verified"] or not config["allow_unsegmented"]:
        return None
    manifest = pd.read_csv(report_root / "participants.csv", dtype={"participant_id": str}, keep_default_na=False)
    rows = manifest[manifest.participant_id == pid]
    if len(rows) != 1:
        return None
    row = rows.iloc[0]
    if row.duplicate_status == "conflicting_features":
        raise ValueError(f"Conflicting source copies for participant {pid}; resolve before training")
    # Values are written as CSV booleans; parsing explicitly avoids bool('False').
    truth = lambda v: str(v).lower() == "true"
    if not truth(row.feature_ready) or not truth(row.all_sources_integrity_checked):
        return None
    sources = [json.loads((report_root / "sources" / f"{key}.json").read_text()) for key in row.source_keys.split(";")]
    from .full_dataset import source_stamp
    for source in sources:
        if source_stamp(Path(source["source"])) != source["source_stamp"]:
            raise ValueError(f"Source changed after full audit: {source['source']}; rerun --all-sources")
    selected = next(s for s in sources if s["source"] == row.selected_source)
    kind = config["audio_kind"]
    if kind not in selected.get("modalities", {}):
        return None
    parent_info = selected["modalities"][kind]
    if parent_info["version"] != VERSION or any(parent_info["settings"][k] != config[k] for k in TRANSFORM_SETTINGS):
        return None
    npz_path = Path(selected["artifacts"][kind + "_npz"])
    if not npz_path.exists():
        return None
    with npz_path.open("rb") as f:
        parent_hash = hashlib.file_digest(f, "sha256").hexdigest()
    current_settings = {key: config[key] for key in parent_info["settings"]}
    info = {**parent_info, "pilot_only": False, "settings": current_settings, "timestamp_origin_status": "user verified", "promoted_from_full_dataset_cache": str(npz_path), "parent_fingerprint": parent_info["fingerprint"], "parent_npz_sha256": parent_hash}
    info["fingerprint"] = digest({"version": VERSION, "parent_fingerprint": parent_info["fingerprint"], "parent_npz_sha256": parent_hash, "settings": current_settings, "pilot": False})
    dest = Path(config["cache"]) / "participants"
    dest.mkdir(parents=True, exist_ok=True)
    info_path = dest / f"{pid}.json"
    if info_path.exists() and (dest / f"{pid}.npz").exists() and json.loads(info_path.read_text()).get("fingerprint") == info["fingerprint"]:
        return json.loads(info_path.read_text())
    temporary = dest / f"{pid}.npz.tmp"
    shutil.copyfile(npz_path, temporary)
    temporary.replace(dest / f"{pid}.npz")
    write_json(info_path, info)
    return info
