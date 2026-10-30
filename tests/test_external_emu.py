import hashlib
import numpy as np
import pandas as pd
import pytest
from src.external_emu import phq8_labels,import_rows,clean_text


def test_true_ninth_item_subtraction_and_bin_boundaries():
    scores,y=phq8_labels([4,5,9,10,14,15,19,20,27],[0,0,0,0,0,0,0,0,3])
    assert scores.tolist()==[4,5,9,10,14,15,19,20,24]
    assert y.tolist()==[0,1,1,2,2,3,3,4,4]
    assert phq8_labels([20],[3])[1].tolist()==[3]
    for total,ninth in [([2],[3]),([28],[0]),([12],[4]),([1.5],[0]),([np.nan],[0])]:
        with pytest.raises(ValueError):phq8_labels(total,ninth)


def test_missing_labels_are_quarantined_without_changing_other_records():
    t=pd.DataFrame({'ID':['a','b'],'Content':['I feel okay today','I feel tired today']})
    d=pd.DataFrame({'id':['a'],'phq':[12],'q9':[2],'age':[44],'gender':['x']})
    rows,q=import_rows(t,d,[])
    assert len(rows)==1 and rows[0]['phq8_score']==10 and rows[0]['role']=='external_training_only'
    assert len(q)==1 and q[0]['participant_id']=='emu:b'
    assert 'age' not in rows[0] and 'gender' not in rows[0]
    with pytest.raises(ValueError,match='duplicate'):
        import_rows(pd.concat([t,t]),d,[])


def test_contamination_and_score_text_checks():
    t=pd.DataFrame({'ID':['a'],'Content':['I feel okay today']});d=pd.DataFrame({'id':['a'],'phq':[0],'q9':[0]})
    text=clean_text(t.Content.iloc[0]);protected=[{'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest()}]
    with pytest.raises(ValueError,match='overlaps'):import_rows(t,d,protected)
    with pytest.raises(ValueError,match='questionnaire'):clean_text('My PHQ-9 score is 20')
    assert clean_text('I have 4 cats')=='i have cats'
