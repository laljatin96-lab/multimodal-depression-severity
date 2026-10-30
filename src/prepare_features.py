"""Timestamp-binned MFCC/eGeMAPS and fixed OpenFace features, with masks."""
import argparse
import hashlib
import io
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from .common import VERSION, digest, load_config, read_metadata, validate_dataset_coverage, write_json

VISUAL_COLUMNS = ["pose_" + x for x in ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]] + ["gaze_angle_x", "gaze_angle_y"] + [f"AU{x:02}_r" for x in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]]
AUDIO_COLUMNS = [f"pcm_fftMag_mfcc{suffix}[{i}]" for suffix in ["", "_de", "_de_de"] for i in range(13)]


def sources(root, pid, kind):
    root = Path(root)
    audio = f"{pid}_OpenSMILE2.3.0_{kind}.csv"
    visual = f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"
    directory = root / f"{pid}_P" / "features"
    if all((directory / n).exists() for n in [audio, visual]):
        return {n: (directory / n).read_bytes() for n in [audio, visual]}, [str(directory / n) for n in [audio, visual]]
    archive = root / f"{pid}_P.tar.gz"
    if not archive.exists():
        raise FileNotFoundError(f"No complete feature pair or canonical archive for participant {pid}")
    found = {}
    # Stream only requested CSVs, never extract raw interviews or unsafe paths.
    with tarfile.open(archive, "r|gz") as tar:
        for entry in tar:
            name = Path(entry.name).name
            if entry.isfile() and name in [audio, visual]:
                if name in found:
                    raise ValueError(f"Duplicate member {name} in {archive}")
                found[name] = tar.extractfile(entry).read()
            if len(found) == 2:
                break
    if len(found) != 2:
        raise ValueError(f"Missing modalities in {archive}")
    return found, [str(archive)]


def temporal_frame(blob, time_column, sep):
    frame = pd.read_csv(io.BytesIO(blob), sep=sep)
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any() or time_column not in frame:
        raise ValueError("Duplicate columns or missing timestamps")
    t = frame[time_column].to_numpy(float, copy=True)
    if len(t) < 2 or not np.isfinite(t).all() or (np.diff(t) <= 0).any() or t[0] < 0:
        raise ValueError("Timestamps must be finite, increasing, nonnegative sequences")
    return frame, t


def speech_mask(t, intervals):
    if intervals is None:
        return np.ones(len(t), bool)
    mask = np.zeros(len(t), bool)
    last = -np.inf
    for start, end in intervals:
        if not np.isfinite([start, end]).all() or start < 0 or end <= start or start < last:
            raise ValueError("Speech intervals must be sorted, disjoint, finite start/end seconds")
        mask |= (t >= start) & (t < end)
        last = end
    return mask


def bin_frames(t, values, valid, edges):
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite source features; fix or explicitly exclude participant")
    indices = np.searchsorted(edges, t, side="right") - 1
    out = np.zeros((len(edges) - 1, values.shape[1]), np.float32)
    counts = np.zeros(len(out), np.int64)
    use = valid & (indices >= 0) & (indices < len(out))
    np.add.at(out, indices[use], values[use])
    np.add.at(counts, indices[use], 1)
    out /= np.maximum(counts, 1)[:, None]
    return out, counts > 0


def align(audio_blob, visual_blob, config, intervals=None):
    a, at = temporal_frame(audio_blob, "frameTime", ";")
    v, vt = temporal_frame(visual_blob, "timestamp", ",")
    ac = AUDIO_COLUMNS if config["audio_kind"] == "mfcc" else [c for c in a if c not in ["name", "frameTime"]]
    if not set(ac) <= set(a) or not set(VISUAL_COLUMNS + ["success", "confidence"]) <= set(v):
        raise ValueError("Required fixed feature columns absent")
    at += config["audio_time_offset"]
    vt += config["visual_time_offset"]
    start, end = max(at[0], vt[0]), min(at[-1], vt[-1])
    dt = config["bin_seconds"]
    if dt <= 0 or end - start < dt * config["minimum_window_steps"]:
        raise ValueError("Insufficient common timestamp interval")
    edges = start + np.arange(int((end - start) // dt) + 1) * dt
    av, am = bin_frames(at, a[ac].to_numpy(float), speech_mask(at, intervals), edges)
    quality = (v.success.to_numpy(float) == 1) & (v.confidence.to_numpy(float) >= config["confidence_threshold"])
    vv, vm = bin_frames(vt, v[VISUAL_COLUMNS].to_numpy(float), quality & speech_mask(vt, intervals), edges)
    if not am.any() or not vm.any():
        raise ValueError("No valid observations for one modality")
    return {"audio": av, "visual": vv, "audio_mask": am, "visual_mask": vm, "timestamps": edges[:-1]}, {"audio_columns": ac, "visual_columns": VISUAL_COLUMNS, "audio_frames": len(a), "visual_frames": len(v), "tracking_valid_fraction": float(quality.mean()), "audio_median_step": float(np.median(np.diff(at))), "visual_median_step": float(np.median(np.diff(vt))), "common_seconds": float(end - start), "audio_observed_bins": int(am.sum()), "visual_observed_bins": int(vm.sum())}


def prepare_one(config, pid, speech_path="", pilot=False):
    if not pilot and not config["timestamp_origins_verified"]:
        raise ValueError("Verify/document timestamp origins and offsets; set timestamp_origins_verified before research preparation")
    intervals = None
    if speech_path:
        speech = pd.read_csv(speech_path)
        intervals = speech[["start", "end"]].to_numpy(float)
    if intervals is None and not config["allow_unsegmented"] and not pilot:
        raise ValueError("Participant-only speech intervals unavailable; supply speech_path or explicitly allow_unsegmented with documented limitation")
    if not pilot and intervals is None:
        from .cache_reuse import reuse_full_cache
        reused = reuse_full_cache(config, pid)
        if reused is not None:
            return reused
    blobs, refs = sources(config["dataset"], pid, config["audio_kind"])
    return cache_feature_blobs(config, pid, blobs, refs, intervals, pilot)


def cache_feature_blobs(config, pid, blobs, refs, intervals=None, pilot=True, destination=None):
    """Cache a validated source pair without rereading its enclosing archive."""
    settings = {k: config[k] for k in ["audio_kind", "audio_time_offset", "visual_time_offset", "timestamp_origins_verified", "allow_unsegmented", "confidence_threshold", "bin_seconds", "minimum_window_steps"]}
    fingerprint = digest({"version": VERSION, "settings": settings, "files": {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}, "speech": intervals.tolist() if intervals is not None else None, "pilot": pilot, "visual_columns": VISUAL_COLUMNS, "audio_columns": AUDIO_COLUMNS if config["audio_kind"] == "mfcc" else "egemaps-source-order"})
    dest = Path(destination) if destination is not None else Path(config["cache"]) / ("unlabeled_pilot" if pilot else "participants")
    dest.mkdir(parents=True, exist_ok=True)
    npz, info_path = dest / f"{pid}.npz", dest / f"{pid}.json"
    import json
    if npz.exists() and info_path.exists() and json.loads(info_path.read_text()).get("fingerprint") == fingerprint:
        return json.loads(info_path.read_text())
    values, info = align(blobs[f"{pid}_OpenSMILE2.3.0_{config['audio_kind']}.csv"], blobs[f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv"], config, intervals)
    np.savez_compressed(npz, **values)
    info.update({"participant_id": pid, "fingerprint": fingerprint, "settings": settings, "sources": refs, "version": VERSION, "pilot_only": pilot, "speech_selection": "provided participant intervals" if intervals is not None else "UNSEGMENTED; interviewer speech may be present", "timestamp_origin_status": "user verified" if config["timestamp_origins_verified"] else "UNVERIFIED zero-offset assumption; pilot only"})
    write_json(info_path, info)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--pilot-ids", nargs="+", help="Unlabeled feature validation only; cannot be used for training")
    p.add_argument("--all-sources", action="store_true", help="Fully audit every archive/directory and prepare unlabeled caches, including duplicate sources")
    p.add_argument("--output", default="results/full_dataset")
    args = p.parse_args()
    c = load_config(args.config)
    if args.all_sources:
        if args.pilot_ids:
            p.error("--all-sources and --pilot-ids are mutually exclusive")
        from .full_dataset import run_full_dataset
        summary = run_full_dataset(c, args.output)
        print(summary)
        return
    rows, exclusions = [], []
    if args.pilot_ids:
        records = [{"participant_id": pid} for pid in args.pilot_ids]
    else:
        metadata = read_metadata(c["metadata"])
        validate_dataset_coverage(metadata, c)
        records = metadata.to_dict("records")
    for row in records:
        pid = row["participant_id"]
        try:
            rows.append(prepare_one(c, pid, row.get("speech_path", ""), bool(args.pilot_ids)))
            print(f"Prepared {pid}", flush=True)
        except Exception as exc:
            exclusions.append({"participant_id": pid, "reason": str(exc)})
    write_json("results/preparation_audit.json", {"participants": rows, "exclusions": exclusions})
    if exclusions:
        raise RuntimeError(f"{len(exclusions)} preparation failures; see results/preparation_audit.json; no silent exclusions")


if __name__ == "__main__":
    main()
