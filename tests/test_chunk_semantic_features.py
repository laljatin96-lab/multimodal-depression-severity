import numpy as np
import pytest
from src import chunk_semantic_features as s
from src.semantic_features import document_mean

def test_chunk_aggregation_covers_all_chunks_and_preserves_mean():
    rng=np.random.default_rng(3);v=rng.normal(size=(3,384));v/=np.linalg.norm(v,axis=1,keepdims=True)
    a=rng.normal(size=(16,384));a/=np.linalg.norm(a,axis=1,keepdims=True)
    r=s.aggregate(v,[254,254,40],a)
    np.testing.assert_allclose(r['mean'],document_mean(v,[254,254,40]))
    assert len(r['dispersion'])==768 and len(r['symptom'])==408
    assert min(r['attention'])>0 and sum(r['attention'])==pytest.approx(1)
    uniform=s.aggregate(v,[254,254,40],np.zeros((16,384)))
    np.testing.assert_allclose(uniform['attention'],np.array([254,254,40])/548)
    with pytest.raises(ValueError):s.aggregate(v,[254,0,40],a)

def test_pilot_resume_no_inference_and_corruption(tmp_path,monkeypatch):
    vector=np.ones(384)/np.sqrt(384)
    docs=[{'participant_id':str(i),'split':'train','text_sha256':str(i),'text':'private'} for i in range(3)]
    old=[{'embedding':vector.tolist(),'tokens':255,'chunks':2} for _ in docs]
    monkeypatch.setattr(s,'context',lambda:({'test':True},docs,old))
    monkeypatch.setattr(s.original,'local_encoder',lambda *a:(None,None))
    monkeypatch.setattr(s.original,'encode_document',lambda *a:{'embedding':vector.tolist()})
    monkeypatch.setattr(s,'encode_chunks',lambda *a:([vector.tolist(),vector.tolist()],[254,1]))
    out=tmp_path/'features'
    assert not s.prepare(out,max_new=1)['complete']
    assert s.prepare(out,resume=True)['new_participants']==2
    records,bindings=s.load_verified(out);assert len(records)==3
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in (out/'participants').glob('*.json')}
    def forbidden(*a,**kw):raise AssertionError('No new encoder calls')
    monkeypatch.setattr(s.original,'local_encoder',forbidden);monkeypatch.setattr(s,'encode_chunks',forbidden)
    assert s.prepare(out,resume=True)['new_participants']==0
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before}
    p=next(iter(before));p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='record changed'):s.load_verified(out)
