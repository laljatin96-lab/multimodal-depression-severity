"""Offline, frozen sentence embeddings covering every supplied interview."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .batch_inference import atomic_json, load_record, output_lock, save_record
from .common import digest
from .final_evaluation import sha256
from .transcript_corpus import load_verified as corpus

PROTOCOL = 'offline-mpnet-full-document-chunk-mean-v1'


def token_chunks(ids, size=254):
    if size < 1 or not ids:
        raise ValueError('Nonempty token sequence and positive chunk size required')
    return [ids[i:i+size] for i in range(0,len(ids),size)]


def pooled_embedding(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden*weights).sum(1)/weights.sum(1).clamp_min(1), p=2, dim=1)


def document_mean(embeddings, lengths):
    values, lengths = np.asarray(embeddings,dtype=np.float64), np.asarray(lengths,dtype=np.float64)
    if values.ndim!=2 or len(values)!=len(lengths) or not np.isfinite(values).all() or (lengths<=0).any():
        raise ValueError('Invalid chunk embeddings/weights')
    mean = np.average(values,axis=0,weights=lengths)
    norm = np.linalg.norm(mean)
    if norm <= 0:
        raise ValueError('Empty semantic representation')
    return mean/norm


def context_for(plan, corpus_root, model_root):
    documents, bindings = corpus(plan,corpus_root)
    root=Path(model_root)
    receipt=json.loads((root/'download_receipt.json').read_text())
    if receipt['model']!='sentence-transformers/all-mpnet-base-v2':
        raise ValueError('Unexpected public model')
    for name,expected in receipt['files'].items():
        if sha256(root/name)!=expected['sha256']:
            raise ValueError('Encoder asset changed')
    import transformers
    context={'protocol':PROTOCOL,'corpus':bindings,'encoder':receipt,'model_root':str(root),
             'coverage':[{k:r[k] for k in ['participant_id','split','text_sha256']} for r in documents],
             'chunk_content_tokens':254,'batch_size':8,'cpu_threads':2,'encoder_finetuned':False,
             'uploads':False,'code_sha256':sha256(__file__),'torch':torch.__version__,'transformers':transformers.__version__}
    return context,documents


def local_encoder(root):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    from transformers import AutoModel, AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(root,local_files_only=True,trust_remote_code=False,use_safetensors=True).to('cpu').eval()
    return tokenizer,model


def encode_document(text,tokenizer,model):
    ids=tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)['input_ids']
    chunks=token_chunks(ids)
    embeddings=[]
    with torch.inference_mode():
        for start in range(0,len(chunks),8):
            batch=chunks[start:start+8]
            inputs=[{'input_ids':tokenizer.build_inputs_with_special_tokens(chunk)} for chunk in batch]
            encoded=tokenizer.pad(inputs,padding=True,return_tensors='pt')
            hidden=model(**encoded).last_hidden_state
            embeddings.extend(pooled_embedding(hidden,encoded['attention_mask']).cpu().numpy())
    value=document_mean(embeddings,[len(c) for c in chunks])
    return {'embedding':value.tolist(),'tokens':len(ids),'chunks':len(chunks),'tokens_covered':sum(map(len,chunks))}


def validate_record(record,item):
    if any(record[k]!=item[k] for k in ['participant_id','split','text_sha256']):
        raise ValueError('Participant/document identity changed')
    vector=np.asarray(record['embedding'])
    if vector.shape!=(768,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1) or record['tokens']!=record['tokens_covered'] or record['tokens']<=0 or record['chunks']!=(record['tokens']+253)//254:
        raise ValueError('Invalid embedding/token coverage')


def prepare(plan, corpus_root, model_root, output, resume=False, max_new=None):
    out=Path(output)
    if any(out.resolve().is_relative_to(Path(p).resolve()) for p in ['dataset','data','checkpoints',corpus_root]):
        raise ValueError('Feature output must be outside inputs')
    if out.exists() and not resume:raise FileExistsError('Preserving existing semantic features; use --resume')
    if resume and not (out/'protocol.json').exists():raise ValueError('Resume requires protocol')
    if max_new is not None and max_new<1:raise ValueError('Positive pilot count required')
    context,documents=context_for(plan,corpus_root,model_root);ch=digest(context)
    out.mkdir(parents=True,exist_ok=True)
    with output_lock(out):
        envelope={'context':context,'context_hash':ch}
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text())!=envelope:raise ValueError('Feature inputs/code changed')
        else:atomic_json(out/'protocol.json',envelope)
        records=[];new=0;reused=0;encoder=None
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        for item in documents:
            path=out/'participants'/f"{item['participant_id']}.json"
            if path.exists():
                record=load_record(path,ch);reused+=1
            elif max_new is not None and new>=max_new:continue
            else:
                if encoder is None:encoder=local_encoder(model_root)
                record={**{k:item[k] for k in ['participant_id','split','text_sha256']},**encode_document(item['text'],*encoder)}
                save_record(path,record,ch);new+=1
            validate_record(record,item);records.append(record)
            if len(records)%25==0:print(f"Semantic embeddings: {len(records)}/{len(documents)} participants",flush=True)
        current,_=context_for(plan,corpus_root,model_root)
        if current!=context:raise ValueError('Embedding inputs changed during preparation')
        complete=len(records)==len(documents)
        if complete:
            atomic_json(out/'embeddings.json',records)
        result={'complete':complete,'participants':len(records),'expected_participants':len(documents),
                'new_participants':new,'reused_participants':reused,'context_hash':ch,
                'encoded_tokens':sum(r['tokens'] for r in records),'encoder_finetuned':False,'uploads':False,
                'embeddings_sha256':sha256(out/'embeddings.json') if complete else None,
                'records':{r['participant_id']:sha256(out/'participants'/f"{r['participant_id']}.json") for r in records}}
        atomic_json(out/'completion.json',result)
        return result


def load_verified(plan,corpus_root,model_root,output):
    out=Path(output);context,documents=context_for(plan,corpus_root,model_root);ch=digest(context)
    if json.loads((out/'protocol.json').read_text())!={'context':context,'context_hash':ch}:raise ValueError('Semantic protocol changed')
    receipt=json.loads((out/'completion.json').read_text())
    if not receipt['complete'] or receipt['context_hash']!=ch or receipt['embeddings_sha256']!=sha256(out/'embeddings.json'):raise ValueError('Incomplete or changed embeddings')
    records=json.loads((out/'embeddings.json').read_text())
    if len(records)!=len(documents) or set(receipt['records'])!={r['participant_id'] for r in documents}:raise ValueError('Semantic coverage changed')
    for record,item in zip(records,documents):
        validate_record(record,item)
        path=out/'participants'/f"{item['participant_id']}.json"
        if sha256(path)!=receipt['records'][item['participant_id']] or load_record(path,ch)!=record:raise ValueError('Semantic source record changed')
    return records,{'protocol':sha256(out/'protocol.json'),'completion':sha256(out/'completion.json'),'embeddings':sha256(out/'embeddings.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen-plan',default='results/final_evaluation_plan.json');p.add_argument('--corpus',default='results/transcript_corpus')
    p.add_argument('--model',default='checkpoints/pretrained_mpnet');p.add_argument('--output',default='results/mpnet_features')
    p.add_argument('--resume',action='store_true');p.add_argument('--max-new-participants',type=int)
    a=p.parse_args();r=prepare(a.frozen_plan,a.corpus,a.model,a.output,a.resume,a.max_new_participants)
    print(json.dumps({k:v for k,v in r.items() if k!='records'},indent=2))
