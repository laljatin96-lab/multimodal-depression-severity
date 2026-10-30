"""Local summaries of supplied DenseNet/VGG audio and ResNet face features."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import time
import zipfile
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from .batch_inference import atomic_json,load_record,save_record,output_lock
from .common import digest
from .final_evaluation import sha256
from .full_dataset import safe_member,source_stamp
from .raw_frame_features import input_state
from .encoder_kernel_accuracy import offline_network

PROTOCOL='supplied-deep-full-source-summary-v1'
SPECS={'densenet':('densenet201.csv',1920),'vgg_audio':('vgg16.csv',4096),'resnet_face':('CNN_ResNet.mat',2048)}
PLAN='results/final_evaluation_plan.json'


def moments(blocks,width):
    n=0;mean=np.zeros(width);m2=np.zeros(width)
    for block in blocks:
        a=np.asarray(block,dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=width or not np.isfinite(a).all() or len(a)==0:raise ValueError('Invalid deep feature block')
        count=len(a);average=a.mean(0);delta=average-mean
        m2+=np.square(a-average).sum(0)+delta**2*n*count/(n+count)
        mean+=delta*count/(n+count);n+=count
    if n<2:raise ValueError('Need at least two observed feature frames')
    return {'frames':n,'summary':np.r_[mean,np.sqrt(np.maximum(m2/n,0))].tolist()}


def summarize(handle,name,width):
    if name.endswith('.mat'):
        a=loadmat(handle,variable_names=['feature'])['feature']
        return moments((a[i:i+2048] for i in range(0,len(a),2048)),width)
    expected=['name','timeStamp']+[f'neuron_{i}' for i in range(width)]
    end=None
    def blocks():
        nonlocal end
        for frame in pd.read_csv(handle,chunksize=256):
            if frame.columns.tolist()!=expected:raise ValueError('Unexpected deep audio schema')
            times=frame.timeStamp.to_numpy(float)
            if not np.isfinite(times).all() or (times<0).any() or (np.diff(times)<0).any() or (end is not None and times[0]<end):raise ValueError('Invalid audio feature timestamps')
            end=times[-1]
            yield frame.iloc[:,2:].to_numpy(float)
    return moments(blocks(),width)


def source_members(report):
    result={}
    for key,(suffix,width) in SPECS.items():
        matches=[m for m in report['members'] if safe_member(m['member'])==f"{report['participant_id']}_{suffix}"]
        if len(matches)!=1 or not 0<matches[0]['bytes']<2_000_000_000:raise ValueError('Missing/ambiguous/big supplied deep feature')
        result[key]=matches[0]
    return result


def context():
    source,coverage,_,_=input_state(PLAN)
    reports=[json.loads((Path(source['feature_config'].get('full_dataset_report','results/full_dataset'))/'sources'/f"{r['source_key']}.json").read_text()) for r in source['sources']]
    for r in reports:source_members(r)
    return {'protocol':PROTOCOL,'source':source,'coverage':coverage.to_dict('records'),'specs':{k:list(v) for k,v in SPECS.items()},
        'code_sha256':sha256(__file__),'manual_sha256':sha256('data/metadata/official_edaic/E-DAIC_Manual.pdf'),
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},
        'definition':'All rows, equal-frame mean and population std, per modality independently; no labels/gender, temporal alignment or trained extraction.',
        'read_mode':'All physical sources; targeted member reads with new per-member hashes. Original full-archive audit plus unchanged stamps retained; no claim of a new full archive hash.'},reports


def process(report):
    start=time.monotonic();path=Path(report['source']);members=source_members(report);found={}
    if not report['integrity_checked'] or source_stamp(path)!=report['source_stamp']:raise ValueError('Unaudited/changed source')
    def consume(key,stream,size):
        if size!=members[key]['bytes']:raise ValueError('Selected member size changed')
        h=hashlib.sha256();count=0
        with tempfile.TemporaryFile() as tmp:
            while True:
                block=stream.read(1024**2)
                if not block:break
                count+=len(block)
                if count>size:raise ValueError('Oversized member')
                h.update(block);tmp.write(block)
            if count!=size:raise ValueError('Truncated member')
            tmp.seek(0);value=summarize(tmp,members[key]['member'],SPECS[key][1])
        found[key]={**value,'member_sha256':h.hexdigest(),'bytes':count}
    wanted={m['member']:key for key,m in members.items()}
    if path.is_dir():
        for key,m in members.items():
            target=path/m['member']
            if any(p.is_symlink() for p in [target,*target.parents]) or not target.resolve().is_relative_to(path.resolve()):raise ValueError('Unsafe source path')
            with target.open('rb') as f:consume(key,f,target.stat().st_size)
    elif path.suffix=='.zip':
        with zipfile.ZipFile(path) as archive:
            for m in archive.infolist():
                if m.filename in wanted:
                    key=wanted[m.filename]
                    if key in found:raise ValueError('Duplicate member')
                    with archive.open(m) as f:consume(key,f,m.file_size)
    else:
        with tarfile.open(path,mode='r|*',bufsize=1024**2) as archive:
            for m in archive:
                safe_member(m.name)
                if m.name in wanted:
                    key=wanted[m.name]
                    if key in found or not m.isfile():raise ValueError('Duplicate/nonregular member')
                    consume(key,archive.extractfile(m),m.size)
                    if len(found)==len(wanted):break
    if set(found)!=set(SPECS) or source_stamp(path)!=report['source_stamp']:raise ValueError('Incomplete/changed source')
    return {**{k:report[k] for k in ['source_key','participant_id','source','source_stamp']},'features':found,'seconds':time.monotonic()-start}


def validate(record,report):
    if any(record[k]!=report[k] for k in ['source_key','participant_id','source','source_stamp']):raise ValueError('Source identity changed')
    if set(record['features'])!=set(SPECS):raise ValueError('Missing modalities')
    for name,(_,width) in SPECS.items():
        r=record['features'][name];a=np.asarray(r['summary'])
        if a.shape!=(width*2,) or not np.isfinite(a).all() or (a[width:]<0).any() or r['frames']<2 or r['bytes']!=source_members(report)[name]['bytes'] or len(r['member_sha256'])!=64:raise ValueError('Invalid feature summary')


def combine(records,coverage):
    result=[]
    for item in coverage:
        copies=[r for r in records if r['participant_id']==item['participant_id']]
        if not copies:raise ValueError('Missing participant')
        chosen=copies[0]
        if any(r['features']!=chosen['features'] for r in copies):raise ValueError('Duplicate deep features conflict')
        result.append({**item,'source_keys':[r['source_key'] for r in copies],'features':chosen['features']})
    return result


def prepare(output='results/supplied_deep_features',resume=False,max_new=None,workers=2):
    out=Path(output)
    if out.exists() and not resume:raise FileExistsError('Preserving existing features')
    if resume and not (out/'protocol.json').exists():raise ValueError('Resume needs protocol')
    if any(out.resolve().is_relative_to(Path(p).resolve()) for p in ['dataset','data','checkpoints']):raise ValueError('Output overlaps original inputs')
    if workers not in [1,2] or (max_new is not None and max_new<1):raise ValueError('Invalid bounded run settings')
    ctx,reports=context();ch=digest(ctx);out.mkdir(parents=True,exist_ok=True)
    with output_lock(out),offline_network():
        envelope={'context':ctx,'context_hash':ch}
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text())!=envelope:raise ValueError('Feature context changed')
        else:atomic_json(out/'protocol.json',envelope)
        records={};pending=[]
        for report in reports:
            p=out/'sources'/f"{report['source_key']}.json"
            if p.exists():
                r=load_record(p,ch);validate(r,report);records[report['source_key']]=r
            else:pending.append(report)
        reused=len(records);pending=pending if max_new is None else pending[:max_new]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(process,r):r for r in pending}
            for future in as_completed(futures):
                report=futures[future];r=future.result();validate(r,report)
                save_record(out/'sources'/f"{report['source_key']}.json",r,ch);records[report['source_key']]=r
                print(f"Deep sources: {len(records)}/{len(reports)}; last {r['seconds']:.1f}s",flush=True)
        if context()[0]!=ctx:raise ValueError('Source changed during feature pass')
        complete=len(records)==len(reports)
        if complete:atomic_json(out/'participants.json',combine([records[r['source_key']] for r in reports],ctx['coverage']))
        receipt={'complete':complete,'context_hash':ch,'sources':len(records),'expected_sources':len(reports),'participants':len(ctx['coverage']) if complete else None,
            'new_sources':len(records)-reused,'reused_sources':reused,'participants_sha256':sha256(out/'participants.json') if complete else None,
            'records':{k:sha256(out/'sources'/f'{k}.json') for k in records},'new_full_archive_hash':False,'labels_used':False}
        atomic_json(out/'completion.json',receipt)
    return receipt


def load_verified(output='results/supplied_deep_features'):
    out=Path(output);ctx,reports=context();ch=digest(ctx)
    if json.loads((out/'protocol.json').read_text())!={'context':ctx,'context_hash':ch}:raise ValueError('Changed feature context')
    done=json.loads((out/'completion.json').read_text())
    if not done['complete'] or done['context_hash']!=ch or set(done['records'])!={r['source_key'] for r in reports} or done['participants_sha256']!=sha256(out/'participants.json'):raise ValueError('Incomplete/changed features')
    records=[]
    for report in reports:
        p=out/'sources'/f"{report['source_key']}.json"
        if sha256(p)!=done['records'][report['source_key']]:raise ValueError('Changed source record')
        r=load_record(p,ch);validate(r,report);records.append(r)
    participants=json.loads((out/'participants.json').read_text())
    if combine(records,ctx['coverage'])!=participants:raise ValueError('Participant coverage/values changed')
    return participants,{str(out/n):sha256(out/n) for n in ['protocol.json','completion.json','participants.json']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true');p.add_argument('--max-new-sources',type=int)
    a=p.parse_args();r=prepare(resume=a.resume,max_new=a.max_new_sources)
    print(json.dumps({k:v for k,v in r.items() if k!='records'},indent=2))
