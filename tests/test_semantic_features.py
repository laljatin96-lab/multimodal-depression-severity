import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src import semantic_features as s
from src.download_sentence_encoder import download
from src.final_evaluation import sha256


def test_token_coverage_and_masked_pooling():
    ids=list(range(1001));chunks=s.token_chunks(ids)
    assert sum(chunks,[])==ids and max(map(len,chunks))==254
    with pytest.raises(ValueError):s.token_chunks([])
    hidden=torch.tensor([[[1.,0.],[0.,1.],[999.,999.]]])
    result=s.pooled_embedding(hidden,torch.tensor([[1,1,0]]))
    np.testing.assert_allclose(result.numpy(),[[2**-.5,2**-.5]],atol=1e-7)
    expected=np.array([2.,1.])/np.sqrt(5)
    np.testing.assert_allclose(s.document_mean([[1.,0.],[0.,1.]],[2,1]),expected)
    with pytest.raises(ValueError):s.document_mean([[1.,0.]],[0])


def test_feature_pilot_resume_and_tamper(tmp_path,monkeypatch):
    docs=[{'participant_id':str(i),'split':'train','text_sha256':str(i),'text':'local confidential interview'} for i in range(3)]
    monkeypatch.setattr(s,'context_for',lambda *a:({'fixture':True},docs))
    monkeypatch.setattr(s,'local_encoder',lambda *a:(None,None))
    embedding=np.ones(384)/np.sqrt(384)
    monkeypatch.setattr(s,'encode_document',lambda *a:{'embedding':embedding.tolist(),'tokens':255,'chunks':2,'tokens_covered':255})
    args=('plan','corpus','encoder',tmp_path/'out')
    assert not s.prepare(*args,max_new=1)['complete']
    assert s.prepare(*args,resume=True)['new_participants']==2
    records,binding=s.load_verified(*args)
    assert len(records)==3
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in (args[-1]/'participants').glob('*.json')}
    def forbidden(*a,**kw):raise AssertionError('No repeated encoding')
    monkeypatch.setattr(s,'local_encoder',forbidden);monkeypatch.setattr(s,'encode_document',forbidden)
    assert s.prepare(*args,resume=True)['new_participants']==0
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before}
    record=next(iter(before));record.write_text(record.read_text()+' ')
    with pytest.raises(ValueError,match='source record'):s.load_verified(*args)


def test_download_reuses_only_matching_assets(tmp_path):
    p=tmp_path/'config.json';p.write_text('{}')
    receipt={'files':{'config.json':{'sha256':sha256(p)}}}
    (tmp_path/'download_receipt.json').write_text(json.dumps(receipt))
    assert download(tmp_path)==receipt
    p.write_text('changed')
    with pytest.raises(ValueError,match='asset'):download(tmp_path)
