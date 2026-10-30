import io
import json
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
from src import supplied_deep_features as f


def test_streaming_moments_and_mat_csv():
    a=np.random.default_rng(42).normal(size=(19,3))
    r=f.moments([a[:6],a[6:]],3)
    np.testing.assert_allclose(r['summary'],np.r_[a.mean(0),a.std(0)],rtol=1e-13)
    b=io.BytesIO();savemat(b,{'feature':a});b.seek(0)
    np.testing.assert_allclose(f.summarize(b,'features.mat',3)['summary'],r['summary'])
    table=pd.DataFrame(a,columns=[f'neuron_{i}' for i in range(3)])
    table.insert(0,'timeStamp',np.arange(len(a)));table.insert(0,'name','unknown')
    np.testing.assert_allclose(f.summarize(io.StringIO(table.to_csv(index=False)),'deep.csv',3)['summary'],r['summary'])
    table.loc[5,'timeStamp']=-1
    with pytest.raises(ValueError,match='timestamps'):f.summarize(io.StringIO(table.to_csv(index=False)),'deep.csv',3)
    with pytest.raises(ValueError):f.moments([np.full((2,3),np.nan)],3)


@pytest.fixture
def sources(tmp_path,monkeypatch):
    specs={'densenet':('densenet201.csv',2),'vgg_audio':('vgg16.csv',3),'resnet_face':('CNN_ResNet.mat',4)}
    monkeypatch.setattr(f,'SPECS',specs)
    path=tmp_path/'300_P';path.mkdir();members=[]
    for name,(suffix,width) in specs.items():
        target=path/f'300_{suffix}';a=np.arange(6*width).reshape(6,width).astype(float)
        if suffix.endswith('.mat'):savemat(target,{'feature':a})
        else:
            table=pd.DataFrame(a,columns=[f'neuron_{i}' for i in range(width)])
            table.insert(0,'timeStamp',np.arange(6));table.insert(0,'name','unknown');table.to_csv(target,index=False)
        members.append({'member':target.name,'bytes':target.stat().st_size})
    archive=tmp_path/'300.tar.gz'
    with tarfile.open(archive,'w:gz') as out:
        for m in members:out.add(path/m['member'],arcname=m['member'])
    reports=[{'source':str(p),'source_key':str(i),'participant_id':'300','source_stamp':f.source_stamp(p),'integrity_checked':True,'members':members} for i,p in enumerate([path,archive])]
    ctx={'coverage':[{'participant_id':'300','split':'train'}],'test':True}
    monkeypatch.setattr(f,'context',lambda:(ctx,reports))
    return ctx,reports


def test_all_source_duplicate_reconciliation_and_zero_read_resume(sources,tmp_path,monkeypatch):
    ctx,reports=sources;out=tmp_path/'out'
    assert not f.prepare(out,max_new=1)['complete']
    assert f.prepare(out,resume=True)['complete']
    participants,bindings=f.load_verified(out)
    assert participants[0]['source_keys']==['0','1']
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in (out/'sources').glob('*.json')}
    def forbidden(*a,**kw):raise AssertionError('No source reads')
    monkeypatch.setattr(f,'process',forbidden)
    assert f.prepare(out,resume=True)['new_sources']==0
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before}
    path=out/'participants.json';path.write_text('[]')
    with pytest.raises(ValueError):f.load_verified(out)


def test_conflicting_deep_duplicates_rejected(sources):
    ctx,reports=sources;r=[f.process(p) for p in reports]
    r[1]['features']['densenet']['summary'][0]+=1
    with pytest.raises(ValueError,match='conflict'):f.combine(r,ctx['coverage'])


def test_changed_source_rejected(sources):
    _,reports=sources;p=Path(reports[0]['source'])/reports[0]['members'][0]['member']
    p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='changed source'):f.process(reports[0])
