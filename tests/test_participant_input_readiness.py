"""Guards against participant leakage and rewritten readiness evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from src import participant_input_readiness as readiness
from src.final_evaluation import sha256
from src.batch_inference import atomic_json
from src.common import digest


@pytest.fixture
def frozen_population():
    context=json.loads(Path('results/wavlm_layer_accuracy/protocol.json').read_text())['context']
    return pd.DataFrame(context['training']),deepcopy(context['folds'])


def test_original_frozen_ownership_passes_without_fitting(frozen_population):
    frame,folds=frozen_population
    result=readiness.verify_population(frame,folds)
    assert result['class_supports']==[77,36,26,17,7]
    assert result['unique_original_training_participants']==163


def test_inner_participant_overlap_is_rejected(frozen_population):
    frame,folds=frozen_population
    part=folds[0]['inner'][0]
    part['fit'].append(part['heldout'][0])
    with pytest.raises(ValueError,match='Inner participant separation'):
        readiness.verify_population(frame,folds)


def test_changed_target_mapping_is_rejected(frozen_population):
    frame,folds=frozen_population
    metadata=pd.read_csv('data/metadata/participants.csv',dtype={'participant_id':str})
    metadata.loc[metadata.participant_id.eq(frame.participant_id.iloc[0]),'phq8_score']=24
    with pytest.raises(AssertionError):
        readiness.verify_population(frame,folds,metadata)


def test_duplicate_participant_is_rejected(frozen_population):
    frame,folds=frozen_population
    frame.loc[1,'participant_id']=frame.loc[0,'participant_id']
    with pytest.raises(ValueError,match='unique training participants'):
        readiness.verify_population(frame,folds)


def test_completed_resume_has_zero_writes_and_rejects_changed_source(tmp_path):
    source=tmp_path/'source.txt';source.write_text('frozen input')
    output=tmp_path/'ready';output.mkdir()
    report=output/'REPORT.md';report.write_text('original report')
    receipt=output/'readiness.json'
    receipt.write_text(json.dumps({'protocol':readiness.PROTOCOL,'complete':True,
        'input_bindings':{str(source):sha256(source)},'output_bindings':{str(report):sha256(report)}}))
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in [source,report,receipt]}
    result=readiness.resume_completed(output)
    assert result['new_fits']==result['new_encoder_extractions']==result['new_writes']==0
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before}
    source.write_text('changed input')
    with pytest.raises(ValueError,match='Bound evidence changed'):
        readiness.resume_completed(output)


def test_partial_resume_is_preserved(tmp_path):
    output=tmp_path/'partial';output.mkdir()
    evidence=output/'REPORT.md';evidence.write_text('partial preserved')
    with pytest.raises(ValueError,match='Incomplete readiness output'):
        readiness.resume_completed(output)
    assert evidence.read_text()=='partial preserved'


def test_self_consistent_modified_cache_still_fails_original_feature_comparison(tmp_path):
    root=tmp_path/'cache';root.mkdir()
    frame=pd.DataFrame({'participant_id':['one','two']})
    x={'semantic_av':np.zeros((2,4)),'language':np.zeros((2,16)),'speech':np.zeros((2,1536))}
    context={'targets_used':False,'validation_test_processed':False,'training_ids':frame.participant_id.tolist(),
        'source_code_sha256':sha256('src/participant_speech.py'),'support_hashes':{},
        'provenance':{'bindings':{}},'annotation_hashes':{}}
    contract={'context':context,'context_hash':digest(context)}
    atomic_json(root/'contract.json',contract)
    atomic_json(root/'participant_inputs.json',[{'participant_id':pid,'decision':{'text_verified':False,'audio_verified':False},
        'document':{'fallback':True}} for pid in frame.participant_id])
    def receipt():
        atomic_json(root/'completion.json',{'complete':True,'context_hash':contract['context_hash'],
            'dropped_participants':0,'new_encoder_runs':0,'new_fits':0,
            'artifacts':{str(root/name):sha256(root/name) for name in ['contract.json','features.npz','participant_inputs.json']}})
    np.savez_compressed(root/'features.npz',participant_id=frame.participant_id.to_numpy(str),**x)
    receipt()
    assert readiness.verify_fallback_cache(root/'contract.json',x,frame,{})['all_fallback_features_bit_identical']
    np.savez_compressed(root/'features.npz',participant_id=frame.participant_id.to_numpy(str),**{**x,'speech':x['speech']+1})
    receipt() # Matching new hashes cannot disguise a change against the original arrays.
    with pytest.raises(AssertionError):
        readiness.verify_fallback_cache(root/'contract.json',x,frame,{})
