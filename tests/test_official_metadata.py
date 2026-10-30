import json
from pathlib import Path

import pandas as pd
import pytest

from src.import_official_metadata import import_records, reconcile, verify_archives


def official_fixture(tmp_path):
    root = tmp_path / 'official'
    root.mkdir()
    for pid, name, gender, score in [(1, 'train', 'female ', 0), (2, 'dev', 'male', 5), (3, 'test', 'female', 10)]:
        pd.DataFrame([{'Participant_ID': pid, 'Gender': gender, 'PHQ_Binary': int(score >= 10), 'PHQ_Score': score}]).to_csv(root / f'{name}_split.csv', index=False)
    details = []
    for pid, split, gender, total in [(1, 'train', 'female', 0), (2, 'dev', 'female', 5), (3, 'test', 'female', 10)]:
        row = {'Participant': pid, 'split': split, 'gender': gender, 'Depression_severity': total}
        row.update({f'PHQ8_{j}_item': max(0, min(3, total - 3*(j-1))) for j in range(1, 9)})
        details.append(row)
    pd.DataFrame(details).to_csv(root / 'detailed_lables.csv', index=False)
    return root


def test_official_totals_and_conflicting_gender(tmp_path):
    root = official_fixture(tmp_path)
    manifest, issues = reconcile(root)
    assert manifest.phq8_score.tolist() == [0, 5, 10]
    assert manifest.severity_class.tolist() == [0, 1, 2]
    assert manifest.gender.tolist() == ['female', 'unknown', 'female']
    assert manifest.split.tolist() == ['train', 'validation', 'test']
    assert issues.issue.tolist() == ['recorded_gender_conflict_or_unknown']


def test_score_conflicts_and_overlapping_splits_fail(tmp_path):
    root = official_fixture(tmp_path)
    path = root / 'dev_split.csv'
    frame = pd.read_csv(path)
    frame.PHQ_Score = 6
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='disagree with item'):
        reconcile(root)
    frame.PHQ_Score = 5
    frame.Participant_ID = 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='duplicate, overlapping'):
        reconcile(root)


def test_invalid_item_cannot_hide_in_valid_total(tmp_path):
    root = official_fixture(tmp_path)
    path = root / 'detailed_lables.csv'
    frame = pd.read_csv(path)
    frame.loc[1, 'PHQ8_1_item'] = 4
    frame.loc[1, 'PHQ8_2_item'] = 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Invalid eight-item'):
        reconcile(root)


def test_archive_rounding_and_existing_metadata_protection(tmp_path):
    root = official_fixture(tmp_path)
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    for pid in ['1', '2', '3']:
        (dataset / f'{pid}_P.tar.gz').write_bytes(b'0' * 1280)
    (root / 'source_index.txt').write_text('\n'.join(f'{pid}_P.tar.gz (1.3 KB)' for pid in ['1', '2', '3']))
    assert len(verify_archives(root, dataset, ['1', '2', '3'])) == 3
    destination = tmp_path / 'participants.csv'
    destination.write_text('participant_id,gender,phq8_score\n1,,4\n')
    before = destination.read_bytes()
    config = {'dataset': str(dataset), 'metadata': str(destination)}
    with pytest.raises(FileExistsError, match='Preserving existing populated'):
        import_records(config, root)
    assert destination.read_bytes() == before
    destination.write_text('participant_id,gender,phq8_score\n1,,\n2,,\n3,,\n')
    report = import_records(config, root)
    assert report['participants'] == 3 and report['archive_size_matches'] == 3
    assert Path(destination.with_name('participants_before_official_import.csv')).exists()
    assert import_records(config, root)['participants'] == 3
    assert json.loads((root / 'import_report.json').read_text())['all_eight_item_sums_match_current_totals']
