import copy
import numpy as np
import pandas as pd
import pytest
from src import semantic_pipeline_audit as audit
from src.evaluate import write_report
from src.modality_diagnostics import nested_folds


def test_fold_audit_rejects_overlap_missing_ownership_and_absent_class():
    y = np.tile(np.arange(5), 10)
    frame = pd.DataFrame({'severity_class':y})
    folds = nested_folds(frame, {'outer_folds':2,'inner_folds':2,'fold_seeds':[42]})
    result = audit.check_folds(folds,y,[42])
    assert min(result['inner_fit_class_min']) >= 2
    changed = copy.deepcopy(folds); changed[0]['inner'][0]['fit'][0] = changed[0]['heldout'][0]
    with pytest.raises(ValueError,match='inner fold'):audit.check_folds(changed,y,[42])
    with pytest.raises(ValueError,match='outer coverage'):audit.check_folds(folds[:1],y,[42])
    changed = y.copy(); changed[changed == 4] = 3
    with pytest.raises(ValueError,match='class is absent'):audit.check_folds(folds,changed,[42])


def test_saved_report_checks_identity_probabilities_metrics_and_groups(tmp_path):
    y = np.tile(np.arange(5),2)
    frame = pd.DataFrame({'participant_id':[str(i) for i in range(len(y))], 'severity_class':y,
                          'gender':['female']*5+['unknown']*5})
    p = np.eye(5)[y]*.8+.04
    write_report(frame,p,tmp_path,'selected',42,'train_nested_oof')
    result = audit.check_saved_report(frame,p,tmp_path)
    assert result['accuracy'] == 1
    path = tmp_path/'predictions.csv'; table = pd.read_csv(path,dtype={'participant_id':str})
    table.loc[0,'severity_class'] = 1; table.to_csv(path,index=False)
    with pytest.raises(AssertionError):audit.check_saved_report(frame,p,tmp_path)


def test_audit_preserves_existing_output(tmp_path):
    out = tmp_path/'existing'; out.mkdir(); (out/'keep').write_text('keep')
    with pytest.raises(FileExistsError):audit.run(out)
    assert (out/'keep').read_text() == 'keep'
