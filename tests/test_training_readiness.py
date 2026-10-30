import json
from pathlib import Path

import pandas as pd
import pytest

from src.common import joined_manifest, read_metadata, validate_dataset_coverage
from src.full_dataset import run_full_dataset
from src.prepare_features import prepare_one
from src.training_readiness import check_readiness, initialize_metadata
from test_full_dataset import config_for, feature_files, tar_source


def setup_case(tmp_path, ids=("123",)):
    c = config_for(tmp_path)
    c.update({"full_dataset_report": str(tmp_path / "audit"), "metadata": str(tmp_path / "metadata/participants.csv"), "splits": str(tmp_path / "splits.csv")})
    for index, pid in enumerate(ids):
        files = feature_files(pid)
        # Different fixture participants must have distinct feature content.
        key = next(k for k in files if k.endswith("mfcc.csv"))
        files[key] = files[key].replace(b"1.0", str(index + 2.0).encode(), 1)
        tar_source(Path(c["dataset"]) / f"{pid}_P.tar.gz", files)
    run_full_dataset(c, c["full_dataset_report"])
    return c


def test_initialize_all_ids_without_labels_and_preserve_edits(tmp_path):
    c = setup_case(tmp_path, ("123", "124"))
    initialize_metadata(c)
    meta = pd.read_csv(c["metadata"], dtype=str, keep_default_na=False)
    assert meta.participant_id.tolist() == ["123", "124"]
    assert meta.gender.eq("").all() and meta.phq8_score.eq("").all()
    report = check_readiness(c, tmp_path / "readiness")
    assert report["valid_phq8_scores"] == 0 and report["unlabeled_or_invalid_participants"] == 2
    assert report["source_audit_current"] and not report["ready_for_av_training"]
    meta.loc[0, "phq8_score"] = "5"
    meta.to_csv(c["metadata"], index=False)
    before = Path(c["metadata"]).read_bytes()
    initialize_metadata(c)
    assert Path(c["metadata"]).read_bytes() == before
    with pytest.raises(ValueError, match="1 participants: 124"):
        read_metadata(c["metadata"])


def test_partial_metadata_cannot_silently_shrink_training_dataset(tmp_path):
    c = setup_case(tmp_path, ("123", "124"))
    initialize_metadata(c)
    pd.DataFrame({"participant_id": ["123"], "gender": [""], "phq8_score": [5]}).to_csv(c["metadata"], index=False)
    report = check_readiness(c, tmp_path / "readiness")
    assert not report["complete_participant_coverage"] and report["valid_phq8_scores"] == 1
    assert report["unlabeled_or_invalid_participants"] == 1
    pd.DataFrame({"participant_id": ["123"], "split": ["train"]}).to_csv(c["splits"], index=False)
    with pytest.raises(ValueError, match="cover all 2"):
        joined_manifest(c)


def test_invalid_label_and_duplicate_ids_are_reported(tmp_path):
    c = setup_case(tmp_path)
    initialize_metadata(c)
    Path(c["metadata"]).write_text("participant_id,gender,phq8_score\n123,,25\n")
    report = check_readiness(c, tmp_path / "readiness")
    assert report["valid_phq8_scores"] == 0
    issues = pd.read_csv(tmp_path / "readiness/participant_issues.csv")
    assert issues.iloc[0].issue == "invalid_phq8"
    Path(c["metadata"]).write_text("participant_id,gender,phq8_score\n123,,5\n123,,10\n")
    report = check_readiness(c, tmp_path / "readiness")
    assert any("Duplicate" in reason for reason in report["blockers"])


def test_readiness_checks_splits_provenance_and_real_cache_without_gender(tmp_path):
    c = setup_case(tmp_path, ("123", "124", "125", "126"))
    initialize_metadata(c)
    meta = pd.DataFrame({"participant_id": ["123", "124", "125", "126"], "gender": [""] * 4, "phq8_score": [0, 5, 10, 15]})
    meta.to_csv(c["metadata"], index=False)
    assert check_readiness(c, tmp_path / "readiness")["ready_for_split_creation"]
    c.update({"timestamp_origins_verified": True, "allow_unsegmented": True})
    report = check_readiness(c, tmp_path / "readiness")
    assert report["ready_for_research_preparation"] and not report["ready_for_av_training"]
    pd.DataFrame({"participant_id": meta.participant_id, "split": ["train", "train", "validation", "test"]}).to_csv(c["splits"], index=False)
    assert not check_readiness(c, tmp_path / "readiness")["ready_for_av_training"]
    for pid in meta.participant_id:
        prepare_one(c, pid)
    report = check_readiness(c, tmp_path / "readiness")
    assert report["ready_for_av_training"] and not report["ready_for_gender_training"]
    c["audio_time_offset"] = .2
    assert not check_readiness(c, tmp_path / "readiness")["ready_for_av_training"]


def test_coverage_check_is_scoped_to_the_audited_dataset(tmp_path):
    c = setup_case(tmp_path)
    other = {**c, "dataset": str(tmp_path / "different")}
    validate_dataset_coverage(pd.DataFrame({"participant_id": ["999"]}), other)
    with pytest.raises(ValueError, match="unknown IDs"):
        validate_dataset_coverage(pd.DataFrame({"participant_id": ["999"]}), c)
