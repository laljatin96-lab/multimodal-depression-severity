import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.common import load_config
from src.full_dataset import discover_sources, read_source, run_full_dataset
from src.prepare_features import AUDIO_COLUMNS, VISUAL_COLUMNS


def feature_files(pid="123"):
    at = np.arange(0, 4, 0.1)
    audio = pd.DataFrame(np.ones((len(at), len(AUDIO_COLUMNS))), columns=AUDIO_COLUMNS)
    audio["frameTime"] = at
    visual = pd.DataFrame(np.ones((len(at), len(VISUAL_COLUMNS))), columns=VISUAL_COLUMNS)
    visual["timestamp"], visual["success"], visual["confidence"] = at, 1, .99
    egemaps = pd.DataFrame({"frameTime": at, "Loudness_sma3": np.ones(len(at))})
    return {f"{pid}_P/features/{pid}_OpenSMILE2.3.0_mfcc.csv": audio.to_csv(sep=";", index=False).encode(), f"{pid}_P/features/{pid}_OpenSMILE2.3.0_egemaps.csv": egemaps.to_csv(sep=";", index=False).encode(), f"{pid}_P/features/{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv": visual.to_csv(index=False).encode(), f"{pid}_P/{pid}_Transcript.csv": b"Start_Time,End_Time,Text,Confidence\n0,1,fixture,1\n"}


def tar_source(path, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def config_for(tmp_path):
    c = load_config("configs/default.json")
    c.update({"dataset": str(tmp_path / "dataset"), "cache": str(tmp_path / "cache")})
    Path(c["dataset"]).mkdir()
    return c


def test_all_archives_directories_duplicate_resolution_and_resume(tmp_path):
    c = config_for(tmp_path)
    root = Path(c["dataset"])
    files = feature_files()
    tar_source(root / "123_P.tar.gz", files)
    tar_source(root / "123_P (1).tar.gz", files)
    for name, data in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    out = tmp_path / "out"
    report = run_full_dataset(c, out)
    assert report["complete"] and report["processed_sources"] == 3
    assert report["processed_archives"] == 2 and report["integrity_checked_sources"] == 3
    assert report["unique_participants"] == report["mfcc_visual_feature_ready_participants"] == 1
    frame = pd.read_csv(out / "participants.csv")
    assert frame.iloc[0].duplicate_status == "identical_selected_features"
    saved = list((out / "sources").glob("*.json"))[0]
    before = saved.stat().st_mtime_ns
    again = run_full_dataset(c, out)
    assert saved.stat().st_mtime_ns == before and again == report
    # Changing a duplicate's features is a conflict, never another participant.
    changed = dict(files)
    key = next(k for k in changed if k.endswith("mfcc.csv"))
    changed[key] = changed[key].replace(b"1.0", b"2.0", 1)
    tar_source(root / "123_P (1).tar.gz", changed)
    conflict = run_full_dataset(c, out)
    assert conflict["duplicate_conflicts"] == 1 and conflict["mfcc_visual_feature_ready_participants"] == 0


def test_zip_entries_metadata_candidates_and_missing_modality(tmp_path):
    c = config_for(tmp_path)
    files = feature_files()
    files["123_P/metadata.csv"] = b"participant_id,phq8_score\n123,5\n"
    with zipfile.ZipFile(Path(c["dataset"]) / "123_P.zip", "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    report = run_full_dataset(c, tmp_path / "out")
    assert report["processed_archives"] == 1 and report["metadata_candidate_members"] == 1
    assert report["ready_sources"] == 1
    # Missing eGeMAPS is explicitly partial, not a silent source exclusion.
    files = {k: v for k, v in files.items() if not k.endswith("egemaps.csv")}
    tar_source(Path(c["dataset"]) / "456_P.tar.gz", files)
    report = run_full_dataset(c, tmp_path / "out")
    assert report["processed_sources"] == 2 and report["sources_with_errors"] == 1


def test_full_gzip_crc_and_unsafe_members(tmp_path):
    c = config_for(tmp_path)
    path = Path(c["dataset"]) / "123_P.tar.gz"
    tar_source(path, feature_files())
    data = bytearray(path.read_bytes())
    data[-8] ^= 1
    path.write_bytes(data)
    sources, _ = discover_sources(c["dataset"])
    with pytest.raises((OSError, EOFError)):
        read_source(sources[0])
    report = run_full_dataset(c, tmp_path / "out")
    assert report["processed_sources"] == 1 and report["integrity_checked_sources"] == 0
    tar_source(path, {"../unsafe.csv": b"a\n1\n"})
    sources, _ = discover_sources(c["dataset"])
    with pytest.raises(ValueError, match="Unsafe"):
        read_source(sources[0])


def test_unrecognized_and_duplicate_feature_members(tmp_path):
    c = config_for(tmp_path)
    root = Path(c["dataset"])
    tar_source(root / "123_P.tar.gz", feature_files())
    (root / "notes.txt").write_text("fixture")
    sources, other = discover_sources(root)
    assert len(sources) == 1 and other == [str(root / "notes.txt")]
    files = feature_files()
    name, data = next(iter(files.items()))
    files["different/" + Path(name).name] = data
    tar_source(root / "123_P.tar.gz", files)
    with pytest.raises(ValueError, match="Duplicate selected"):
        read_source(sources[0])


def test_finder_numbered_extracted_copy(tmp_path):
    c = config_for(tmp_path)
    root = Path(c["dataset"])
    (root / "306_P 2").mkdir()
    sources, other = discover_sources(root)
    assert len(sources) == 1 and sources[0]["participant_id"] == "306"
    assert sources[0]["kind"] == "directory" and not other


def test_promote_cache_only_after_confirmation_and_same_transforms(tmp_path):
    from src.cache_reuse import reuse_full_cache
    from src.prepare_features import prepare_one
    c = config_for(tmp_path)
    root = Path(c["dataset"])
    tar_source(root / "123_P.tar.gz", feature_files())
    c["full_dataset_report"] = str(tmp_path / "out")
    run_full_dataset(c, c["full_dataset_report"])
    assert reuse_full_cache(c, "123") is None
    c.update({"timestamp_origins_verified": True, "allow_unsegmented": True})
    promoted = prepare_one(c, "123")
    assert promoted["pilot_only"] is False
    assert promoted["promoted_from_full_dataset_cache"]
    other_dataset = tmp_path / "other_dataset"
    other_dataset.mkdir()
    assert reuse_full_cache({**c, "dataset": str(other_dataset)}, "123") is None
    c["bin_seconds"] = .25
    assert reuse_full_cache(c, "123") is None
    c["bin_seconds"] = .5
    tar_source(root / "123_P.tar.gz", feature_files())
    with pytest.raises(ValueError, match="Source changed"):
        reuse_full_cache(c, "123")


def test_coverage_report_accounts_for_all_current_sources(tmp_path):
    from src.report_dataset import report_dataset
    c = config_for(tmp_path)
    root = Path(c["dataset"])
    tar_source(root / "123_P.tar.gz", feature_files())
    out = tmp_path / "out"
    run_full_dataset(c, out)
    report = report_dataset(out, plots=False)
    assert report["participant_ids"] == 1 and report["egemaps_visual_ready_participants"] == 1
    assert (out / "member_inventory.csv").exists() and (out / "REPORT.md").exists()
    tar_source(root / "456_P.tar.gz", feature_files("456"))
    with pytest.raises(ValueError, match="inventory changed"):
        report_dataset(out, plots=False)


def test_all_prepared_numerical_probe(tmp_path, monkeypatch):
    from src.feature_pilot import main
    c = config_for(tmp_path)
    tar_source(Path(c["dataset"]) / "123_P.tar.gz", feature_files())
    out = tmp_path / "report"
    run_full_dataset(c, out)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(c))
    numerical = tmp_path / "numerical"
    monkeypatch.setattr("sys.argv", ["feature_pilot", "--config", str(config_path), "--all-prepared", "--report", str(out), "--output", str(numerical)])
    main()
    frame = pd.read_csv(numerical / "real_feature_pilot.csv")
    assert len(frame) == 1 and frame.finite_forward_backward.all()
    assert frame.probed_windows.iloc[0] == frame.selected_windows.iloc[0]
