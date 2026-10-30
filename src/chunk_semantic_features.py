"""Label-free local chunk dispersion and symptom-contrast pooling, with complete token coverage."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import semantic_features as original
from .batch_inference import atomic_json,load_record,save_record,output_lock
from .common import digest
from .encoder_kernel_accuracy import offline_network
from .final_evaluation import sha256

PROTOCOL='local-complete-chunk-symptom-contrast-v1'
PLAN='results/final_evaluation_plan.json'
CORPUS='results/transcript_corpus'
MODEL='checkpoints/pretrained_minilm'
OLD='results/semantic_features'
QUERIES=[
 'I have lost interest and enjoyment in my usual activities.',
 'I feel depressed, sad, or hopeless.',
 'I have difficulty sleeping or I sleep far too much.',
 'I feel exhausted and have very little energy.',
 'I have little appetite or I eat far more than usual.',
 'I feel worthless, guilty, or that I have let people down.',
 'I find it difficult to concentrate on reading or watching television.',
 'I move and speak unusually slowly, or feel restless and unable to sit still.',
 'I still enjoy my usual activities.',
 'I generally feel cheerful and hopeful.',
 'I sleep well and feel rested after a normal amount of sleep.',
 'I have enough energy for my daily activities.',
 'My appetite is normal and I eat regular meals.',
 'I feel worthwhile and satisfied with myself.',
 'I can concentrate well on reading or watching television.',
 'I move and speak at my usual pace and feel calm while sitting.'
]


def aggregate(vectors,lengths,anchors):
    v,w,a=np.asarray(vectors,float),np.asarray(lengths,float),np.asarray(anchors,float)
    if v.ndim!=2 or a.shape!=(16,v.shape[1]) or w.shape!=(len(v),) or not np.isfinite(v).all() or not np.isfinite(a).all() or not np.isfinite(w).all() or (w<=0).any():
        raise ValueError('Invalid complete chunk vectors')
    mean=np.average(v,axis=0,weights=w)
    std=np.sqrt(np.average((v-mean)**2,axis=0,weights=w))
    contrast=v@(a[:8]-a[8:]).T
    average=np.average(contrast,axis=0,weights=w)
    deviation=np.sqrt(np.average((contrast-average)**2,axis=0,weights=w))
    logits=contrast.max(1)/.1
    attention=w*np.exp(logits-logits.max());attention/=attention.sum()
    attended=original.document_mean(v,attention)
    return {'mean':original.document_mean(v,w).tolist(),'dispersion':np.r_[std,v.max(0)].tolist(),
            'symptom':np.r_[attended,average,deviation,contrast.max(0)].tolist(),
            'attention':attention.tolist()}


def encode_chunks(text,tokenizer,model):
    ids=tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)['input_ids']
    chunks=original.token_chunks(ids);vectors=[]
    with torch.inference_mode():
        for start in range(0,len(chunks),8):
            inputs=[{'input_ids':tokenizer.build_inputs_with_special_tokens(c)} for c in chunks[start:start+8]]
            encoded=tokenizer.pad(inputs,padding=True,return_tensors='pt')
            hidden=model(**encoded).last_hidden_state
            vectors.extend(original.pooled_embedding(hidden,encoded['attention_mask']).cpu().numpy().astype(float).tolist())
    return vectors,list(map(len,chunks))


def context():
    upstream,docs=original.context_for(PLAN,CORPUS,MODEL)
    records,bindings=original.load_verified(PLAN,CORPUS,MODEL,OLD)
    return {'protocol':PROTOCOL,'upstream':upstream,'original_features':bindings,'queries':QUERIES,
            'temperature':.1,'code_sha256':sha256(__file__),'labels_used':False,'uploads':False},docs,records


def validate(record,item,old,anchors):
    if any(record[k]!=item[k] for k in ['participant_id','split','text_sha256']):raise ValueError('Participant identity changed')
    lengths=record['lengths'];vectors=np.asarray(record['vectors'])
    if vectors.shape!=(len(lengths),384) or lengths!=[254]*(old['chunks']-1)+[old['tokens']-254*(old['chunks']-1)]:
        raise ValueError('Chunk coverage differs from original')
    np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,rtol=0,atol=1e-6)
    values=aggregate(vectors,lengths,anchors)
    for name in values:np.testing.assert_allclose(record[name],values[name],rtol=0,atol=1e-12)
    np.testing.assert_allclose(record['mean'],old['embedding'],rtol=0,atol=2e-7)


def prepare(output='results/chunk_semantic_features',resume=False,max_new=None):
    out=Path(output)
    if out.exists() and not resume:raise FileExistsError('Preserving existing features')
    if resume and not (out/'protocol.json').exists():raise ValueError('Resume needs protocol')
    if any(out.resolve().is_relative_to(Path(p).resolve()) for p in ['data','dataset','checkpoints',OLD,CORPUS]):raise ValueError('Output overlaps inputs')
    if max_new is not None and max_new<1:raise ValueError('Positive pilot count required')
    source,docs,old_records=context();ch=digest(source);out.mkdir(parents=True,exist_ok=True)
    with output_lock(out),offline_network():
        envelope={'context':source,'context_hash':ch}
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text())!=envelope:raise ValueError('Feature context changed')
        else:atomic_json(out/'protocol.json',envelope)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);encoder=None
        if (out/'anchors.json').exists():anchors=load_record(out/'anchors.json',ch)['vectors']
        else:
            encoder=original.local_encoder(MODEL)
            anchors=[original.encode_document(q,*encoder)['embedding'] for q in QUERIES]
            save_record(out/'anchors.json',{'vectors':anchors},ch)
        if (out/'completion.json').exists() and json.loads((out/'completion.json').read_text())['anchors_sha256']!=sha256(out/'anchors.json'):
            raise ValueError('Anchor assets changed')
        records=[];new=reused=0
        for item,old in zip(docs,old_records):
            path=out/'participants'/f"{item['participant_id']}.json"
            if path.exists():record=load_record(path,ch);reused+=1
            elif max_new is not None and new>=max_new:continue
            else:
                if encoder is None:encoder=original.local_encoder(MODEL)
                vectors,lengths=encode_chunks(item['text'],*encoder)
                record={**{k:item[k] for k in ['participant_id','split','text_sha256']},
                        'vectors':vectors,'lengths':lengths,**aggregate(vectors,lengths,anchors)}
                validate(record,item,old,anchors);save_record(path,record,ch);new+=1
            validate(record,item,old,anchors);records.append(record)
            if len(records)%25==0:print(f'Chunk features: {len(records)}/{len(docs)}',flush=True)
        if context()[0]!=source:raise ValueError('Sources changed during extraction')
        complete=len(records)==len(docs)
        if complete:atomic_json(out/'features.json',records)
        result={'complete':complete,'context_hash':ch,'participants':len(records),'new_participants':new,'reused_participants':reused,
                'content_tokens':sum(sum(r['lengths']) for r in records),'all_original_means_reproduced':True,
                'anchors_sha256':sha256(out/'anchors.json'),'features_sha256':sha256(out/'features.json') if complete else None,
                'records':{r['participant_id']:sha256(out/'participants'/f"{r['participant_id']}.json") for r in records},
                'encoder_finetuned':False,'network_blocked':True}
        atomic_json(out/'completion.json',result)
    return result


def load_verified(output='results/chunk_semantic_features'):
    out=Path(output);source,docs,old=context();ch=digest(source)
    if json.loads((out/'protocol.json').read_text())!={'context':source,'context_hash':ch}:raise ValueError('Chunk context changed')
    done=json.loads((out/'completion.json').read_text())
    if not done['complete'] or done['context_hash']!=ch or done['features_sha256']!=sha256(out/'features.json') or done['anchors_sha256']!=sha256(out/'anchors.json'):raise ValueError('Incomplete/changed chunk features')
    records=json.loads((out/'features.json').read_text());anchors=load_record(out/'anchors.json',ch)['vectors']
    if len(records)!=len(docs) or set(done['records'])!={r['participant_id'] for r in docs}:raise ValueError('Incomplete source coverage')
    for record,item,previous in zip(records,docs,old):
        validate(record,item,previous,anchors);path=out/'participants'/f"{item['participant_id']}.json"
        if done['records'][item['participant_id']]!=sha256(path) or load_record(path,ch)!=record:raise ValueError('Chunk record changed')
    return records,{str(out/n):sha256(out/n) for n in ['protocol.json','completion.json','features.json','anchors.json']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-new-participants',type=int);args=parser.parse_args()
    result=prepare(resume=args.resume,max_new=args.max_new_participants)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
