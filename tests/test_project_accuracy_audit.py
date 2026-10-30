import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import project_accuracy_audit as s


def test_label_audit_detects_source_disagreement():
    metadata = pd.DataFrame({'participant_id': ['a','b','c'], 'phq8_score': [0,5,20],
                             'severity_class': [0,1,4], 'split': ['train','validation','test']})
    detailed = pd.DataFrame([[0]*8, [3,2,0,0,0,0,0,0], [3,3,3,3,3,3,2,0]], index=['a','b','c'],
                            columns=[f'PHQ8_{i}' for i in range(8)])
    assert s.label_checks(metadata, detailed)['class_mismatches'] == 0
    changed = metadata.copy(); changed.loc[1,'severity_class'] = 2
    with pytest.raises(ValueError, match='mapping'):
        s.label_checks(changed, detailed)
    changed = metadata.copy(); changed.loc[1,'phq8_score'] = 7
    with pytest.raises(ValueError, match='Total/item'):
        s.label_checks(changed, detailed)


def test_error_audit_does_not_count_repeats_as_new_people():
    frame = pd.DataFrame({'participant_id': list('abcde'), 'severity_class': range(5)})
    p = np.tile([.8,.05,.05,.05,.05], (3,5,1))
    report, private = s.errors(frame, p)
    assert report['unique_participants'] == 5 and report['repeats'] == 3
    assert report['always_wrong'] == 4 and sum(report['class_errors']) == 12
    assert len(private) == 5 and report['recall'] == [1.,0.,0.,0.,0.]


def test_preservation_checks_hash_and_mtime(tmp_path):
    source = tmp_path/'data'; source.write_text('original')
    snapshot = tmp_path/'before.json'
    snapshot.write_text(json.dumps({str(source): {'sha256': s.sha256(source), 'mtime_ns': source.stat().st_mtime_ns}}))
    assert s.assert_preserved(snapshot) == 1
    source.write_text('modified')
    with pytest.raises(ValueError, match='changed'):
        s.assert_preserved(snapshot)
