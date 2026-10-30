"""Complete, resumable local archive audit and label-free feature preparation.

Every source is represented, including duplicate-named copies. Archives are read
to EOF for integrity checks; WAV/deep features are inventoried, not extracted.
No clinical or demographic targets are derived from media.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import time
import zipfile

import numpy as np
import pandas as pd

from .common import VERSION, digest, load_config, write_json
from .prepare_features import AUDIO_COLUMNS, VISUAL_COLUMNS, cache_feature_blobs

AUDIT_VERSION = "full-archive-v1"
SUFFIXES = (".tar.gz", ".tgz", ".zip", ".tar")
SOURCE_PATTERN = re.compile(r"^(\d+)_P(?: \(\d+\)| \d+)?(?:\.tar\.gz|\.tgz|\.zip|\.tar)?$")
SETTINGS = ["audio_time_offset", "visual_time_offset", "timestamp_origins_verified", "allow_unsegmented", "confidence_threshold", "bin_seconds", "minimum_window_steps"]
MAX_SELECTED_BYTES = 1024 ** 3


class HashingReader:
    def __init__(self, raw):
        self.raw = raw
        self.hasher = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size=-1):
        data = self.raw.read(size)
        self.hasher.update(data)
        self.bytes_read += len(data)
        return data


def discover_sources(dataset):
    root = Path(dataset)
    if not root.is_dir():
        raise FileNotFoundError(root)
    sources, unrecognized = [], []
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        match = SOURCE_PATTERN.fullmatch(path.name)
        is_archive = path.is_file() and path.name.lower().endswith(SUFFIXES)
        if (match and path.is_dir()) or is_archive:
            sources.append({"path": path, "participant_id": match[1] if match else None, "source_key": digest(str(path.resolve()))[:16], "kind": "directory" if path.is_dir() else "archive"})
        elif path.name != ".DS_Store":
            unrecognized.append(str(path))
    if not sources:
        raise ValueError(f"No participant sources found in {root}")
    return sources, unrecognized


def source_stamp(path):
    def stamp(p):
        s = p.stat()
        return {"path": str(p), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns, "inode": s.st_ino}
    if path.is_symlink():
        raise ValueError(f"Symlink sources are not supported: {path}")
    if path.is_file():
        return digest(stamp(path))
    return digest([stamp(p) for p in sorted(path.rglob("*")) if p.is_file()])


def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member name: {name}")
    return p.name


def selected_names(pid):
    return {f"{pid}_OpenSMILE2.3.0_mfcc.csv", f"{pid}_OpenSMILE2.3.0_egemaps.csv", f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"}


def inspect_member(name, size, reader, selected, blobs, inventory):
    base = safe_member(name)
    entry = {"member": name, "bytes": size}
    if base in selected:
        if base in blobs:
            raise ValueError(f"Duplicate selected feature member: {base}")
        if size > MAX_SELECTED_BYTES:
            raise ValueError(f"Selected feature exceeds 1 GiB bound: {name}")
        data = reader.read()
        if len(data) != size:
            raise ValueError(f"Truncated member {name}")
        blobs[base] = data
        entry["sha256"] = hashlib.sha256(data).hexdigest()
    elif base.lower().endswith((".csv", ".tsv", ".txt", ".json")):
        # Capture column names only; never transcript text or inferred labels.
        head = reader.read(min(size, 8192)).split(b"\n", 1)[0].decode("utf-8-sig", errors="replace").strip()
        if base.lower().endswith((".csv", ".tsv")):
            delimiter = ";" if head.count(";") > head.count(",") else "\t" if "\t" in head else ","
            entry["columns"] = next(csv.reader([head], delimiter=delimiter), [])
        if re.search(r"phq|gender|label|split|metadata|participant.*info", base, re.I):
            entry["metadata_candidate"] = True
    inventory.append(entry)


def read_source(source):
    path, pid = source["path"], source["participant_id"]
    if pid is None:
        raise ValueError(f"Archive filename does not identify a participant: {path.name}")
    selected, blobs, inventory = selected_names(pid), {}, []
    if path.is_dir():
        checksums = {}
        for p in sorted(path.rglob("*")):
            if p.is_symlink():
                raise ValueError(f"Symlink inside source directory: {p}")
            if not p.is_file() or p.name == ".DS_Store":
                continue
            with p.open("rb") as raw:
                reader = HashingReader(raw)
                inspect_member(str(p.relative_to(path)), p.stat().st_size, reader, selected, blobs, inventory)
                while reader.read(1024 ** 2):
                    pass
                checksums[str(p.relative_to(path))] = reader.hasher.hexdigest()
        content_hash = digest(checksums)
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                safe_member(member.filename)
                if member.is_dir():
                    continue
                with archive.open(member) as reader:
                    inspect_member(member.filename, member.file_size, reader, selected, blobs, inventory)
                    # Reading to EOF verifies CRC for every ZIP entry.
                    while reader.read(1024 ** 2):
                        pass
        with path.open("rb") as raw:
            content_hash = hashlib.file_digest(raw, "sha256").hexdigest()
    else:
        with path.open("rb") as raw:
            reader = HashingReader(raw)
            compressed = path.name.endswith((".gz", ".tgz"))
            stream = gzip.GzipFile(fileobj=reader, mode="rb") if compressed else reader
            try:
                with tarfile.open(fileobj=stream, mode="r|") as archive:
                    for member in archive:
                        safe_member(member.name)
                        if member.isdir():
                            continue
                        if not member.isfile():
                            raise ValueError(f"Unsupported nonregular tar member: {member.name}")
                        inspect_member(member.name, member.size, archive.extractfile(member), selected, blobs, inventory)
                # Tar stops at end markers; drain gzip to validate its CRC/trailer.
                while stream.read(1024 ** 2):
                    pass
            finally:
                if compressed:
                    stream.close()
            while reader.read(1024 ** 2):
                pass
            if reader.bytes_read != path.stat().st_size:
                raise ValueError("Archive byte coverage mismatch")
            content_hash = reader.hasher.hexdigest()
    return blobs, inventory, content_hash


def process_source(source, config, output, run_hash):
    path, key, pid = source["path"], source["source_key"], source["participant_id"]
    report_path = Path(output) / "sources" / f"{key}.json"
    stamp = source_stamp(path)
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        artifacts = previous.get("artifacts", {})
        if previous.get("source_stamp") == stamp and previous.get("run_hash") == run_hash and previous.get("integrity_checked") and all(Path(p).is_file() for p in artifacts.values()):
            return previous, True
    tick = time.monotonic()
    result = {"source": str(path), "source_key": key, "participant_id": pid, "kind": source["kind"], "source_stamp": stamp, "run_hash": run_hash, "integrity_checked": False, "status": "failed", "errors": [], "artifacts": {}, "modalities": {}}
    try:
        blobs, inventory, content_hash = read_source(source)
        if source_stamp(path) != stamp:
            raise ValueError("Source changed while being read")
        result.update({"integrity_checked": True, "source_sha256": content_hash, "members": inventory, "feature_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}, "metadata_candidates": [x["member"] for x in inventory if x.get("metadata_candidate")]})
        for kind in ["mfcc", "egemaps"]:
            pair = [f"{pid}_OpenSMILE2.3.0_{kind}.csv", f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"]
            if not all(n in blobs for n in pair):
                result["errors"].append(f"{kind}: missing feature pair: {[n for n in pair if n not in blobs]}")
                continue
            try:
                dest = Path(config["cache"]) / "all_sources" / key / kind
                info = cache_feature_blobs({**config, "audio_kind": kind}, pid, {n: blobs[n] for n in pair}, [str(path)], pilot=True, destination=dest)
                result["modalities"][kind] = info
                result["artifacts"][kind + "_npz"] = str(dest / f"{pid}.npz")
                result["artifacts"][kind + "_json"] = str(dest / f"{pid}.json")
            except Exception as exc:
                result["errors"].append(f"{kind}: {type(exc).__name__}: {exc}")
        result["status"] = "ready" if not result["errors"] else "partial" if result["modalities"] else "invalid_features"
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["seconds"] = time.monotonic() - tick
    write_json(report_path.with_suffix(".tmp"), result)
    report_path.with_suffix(".tmp").replace(report_path)
    return result, False


def reconcile_participants(results):
    rows = []
    groups = {}
    for result in results:
        groups.setdefault(result["participant_id"], []).append(result)
    for pid, sources in sorted(groups.items(), key=lambda x: str(x[0])):
        signatures = {digest(s.get("feature_sha256", {})) for s in sources if s.get("integrity_checked")}
        valid = [s for s in sources if "mfcc" in s.get("modalities", {})]
        # Multiple copies of one participant are one participant, never extra samples.
        conflict = len(signatures) > 1
        chosen = next((s for s in valid if Path(s["source"]).name == f"{pid}_P.tar.gz"), valid[0] if valid else None)
        usable = chosen is not None and not conflict
        info = chosen["modalities"]["mfcc"] if usable else {}
        rows.append({"participant_id": pid, "source_count": len(sources), "all_sources_integrity_checked": all(s["integrity_checked"] for s in sources), "duplicate_status": "conflicting_features" if conflict else "identical_selected_features" if len(sources) > 1 else "unique", "feature_ready": usable, "selected_source": chosen["source"] if usable else "", "audio_feature_path": chosen["artifacts"]["mfcc_npz"] if usable else "", "visual_feature_path": chosen["artifacts"]["mfcc_npz"] if usable else "", "egemaps_feature_path": chosen["artifacts"].get("egemaps_npz", "") if usable else "", "audio_frames": info.get("audio_frames"), "visual_frames": info.get("visual_frames"), "audio_dimensions": len(info.get("audio_columns", [])), "visual_dimensions": len(info.get("visual_columns", [])), "common_seconds": info.get("common_seconds"), "tracking_valid_fraction": info.get("tracking_valid_fraction"), "gender": "", "phq8_score": "", "severity_class": "", "label_status": "not_supplied", "research_eligible": False, "source_keys": ";".join(s["source_key"] for s in sources)})
    return rows


def save_reports(results, expected, output, unrecognized, complete):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for s in results:
        row = {k: s[k] for k in ["source", "source_key", "participant_id", "kind", "integrity_checked", "status", "seconds"]}
        row.update({"errors": " | ".join(s["errors"]), "source_sha256": s.get("source_sha256", ""), "members": len(s.get("members", [])), "mfcc_ready": "mfcc" in s["modalities"], "egemaps_ready": "egemaps" in s["modalities"], "metadata_candidates": ";".join(s.get("metadata_candidates", []))})
        source_rows.append(row)
    source_frame = pd.DataFrame(source_rows)
    source_frame.to_csv(out / "sources.csv", index=False)
    participants = pd.DataFrame(reconcile_participants(results)) if results else pd.DataFrame()
    participants.to_csv(out / "participants.csv", index=False)
    if len(participants):
        participants[participants.source_count > 1].to_csv(out / "duplicate_resolution.csv", index=False)
        participants[~participants.feature_ready].to_csv(out / "unusable_participants.csv", index=False)
    if len(source_frame):
        source_frame[source_frame.status != "ready"].to_csv(out / "source_errors.csv", index=False)
    summaries = {"complete": complete, "expected_sources": len(expected), "processed_sources": len(results), "expected_archives": sum(s["kind"] == "archive" for s in expected), "processed_archives": sum(s["kind"] == "archive" for s in results), "integrity_checked_sources": sum(s["integrity_checked"] for s in results), "ready_sources": sum(s["status"] == "ready" for s in results), "sources_with_errors": sum(bool(s["errors"]) for s in results), "unique_participants": len(participants), "mfcc_visual_feature_ready_participants": int(participants.feature_ready.sum()) if len(participants) else 0, "duplicate_conflicts": int(participants.duplicate_status.eq("conflicting_features").sum()) if len(participants) else 0, "metadata_candidate_members": sum(len(s.get("metadata_candidates", [])) for s in results), "unrecognized_items": unrecognized, "supervised_training_ready": False, "labels": "PHQ-8 and recorded gender not supplied; never inferred from media", "limitations": ["Source-level feature preparation only, before any split-dependent scaling/windows/training", "Timestamp origins unverified unless documented in config", "Unsegmented speech; interviewer speech may be present", "Unused deep-feature assets and WAVs inventoried but not added as new model modalities"]}
    write_json(out / "summary.json", summaries)
    return summaries


def run_full_dataset(config, output="results/full_dataset"):
    sources, unrecognized = discover_sources(config["dataset"])
    run_hash = digest({"audit_version": AUDIT_VERSION, "preprocessing_version": VERSION, "settings": {k: config[k] for k in SETTINGS}, "mfcc_columns": AUDIO_COLUMNS, "visual_columns": VISUAL_COLUMNS, "cache": str(Path(config["cache"]).resolve())})
    results = []
    write_json(Path(output) / "run.json", {"status": "running", "sources": [{**s, "path": str(s["path"])} for s in sources], "config": config, "run_hash": run_hash, "resume_command": ".venv/bin/python -m src.prepare_features --all-sources"})
    print(f"Auditing {len(sources)} sources ({sum(s['kind'] == 'archive' for s in sources)} archives); all results local", flush=True)
    for i, source in enumerate(sources, 1):
        result, resumed = process_source(source, config, output, run_hash)
        results.append(result)
        if i % 5 == 0 or result["errors"] or i == len(sources):
            save_reports(results, sources, output, unrecognized, False)
            print(f"{i}/{len(sources)} {source['path'].name}: {result['status']} {'(cached)' if resumed else '(' + str(round(result['seconds'], 1)) + 's)'}; errors so far={sum(bool(s['errors']) for s in results)}", flush=True)
    summary = save_reports(results, sources, output, unrecognized, True)
    write_json(Path(output) / "run.json", {"status": "completed", "run_hash": run_hash, "config": config, "summary": summary})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--output", default="results/full_dataset")
    args = p.parse_args()
    print(json.dumps(run_full_dataset(load_config(args.config), args.output), indent=2))


if __name__ == "__main__":
    main()
