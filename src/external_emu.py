"""Provenance-checked EMU transcript import; original E-DAIC splits are untouched."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .age_language_accuracy import language_features
from .batch_inference import atomic_json
from .common import digest
from .encoder_kernel_accuracy import offline_network
from .final_evaluation import sha256
from .semantic_features import local_encoder, encode_document
from .transcript_corpus import clean_transcript

SOURCE = Path('data/external/emu_20261028')
OUTPUT = Path('results/external_emu_features')
REVISION = '552f3185e0671a8c382e4c92505b9f711a149b41'
PROTOCOL = 'author-emu-phq8-text-training-only-v1'


def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Missing original transcript')
    # Reuse identical label-free cleaning; timestamps/confidence are not model features.
    blob = pd.DataFrame([{'Start_Time':0,'End_Time':1,'Text':text,'Confidence':1}]).to_csv(index=False).encode()
    result = clean_transcript(blob)
    if not result['text'] or result['explicit_score_rows_removed']:
        raise ValueError('Empty transcript or explicit questionnaire/score content')
    return result['text']


def phq8_labels(total, ninth):
    total, ninth = np.asarray(total, float), np.asarray(ninth, float)
    if total.shape != ninth.shape or not np.isfinite(total).all() or not np.isfinite(ninth).all():
        raise ValueError('Missing or inconsistent questionnaire targets')
    if not np.array_equal(total,total.astype(int)) or not np.array_equal(ninth,ninth.astype(int)):
        raise ValueError('Questionnaire targets must be integers')
    score = total-ninth
    if ((total<0)|(total>27)|(ninth<0)|(ninth>3)|(score<0)|(score>24)).any():
        raise ValueError('Invalid PHQ-9/item9 values; never guess missing labels')
    score=score.astype(int)
    return score,np.minimum(score//5,4)


def import_rows(transcripts, demographics, protected_documents):
    if set(transcripts.columns) != {'ID','Content'} or not {'id','phq','q9'} <= set(demographics.columns):
        raise ValueError('Unexpected source schema')
    if transcripts.ID.isna().any() or demographics.id.isna().any() or transcripts.ID.duplicated().any() or demographics.id.duplicated().any():
        raise ValueError('Missing or duplicate participant identity')
    # Only identifiers and targets are read from metadata. Demographics never become features.
    metadata=demographics[['id','phq','q9']].set_index('id')
    protected={d['text_sha256'] for d in protected_documents}
    protected_shingles=set()
    for d in protected_documents:
        words=d['text'].split()
        protected_shingles.update(tuple(words[i:i+12]) for i in range(len(words)-11))
    rows=[]; quarantine=[]; seen={}
    for record in transcripts.itertuples(index=False):
        pid=f'emu:{record.ID}'
        if record.ID not in metadata.index:
            quarantine.append({'participant_id':pid,'reason':'No matching author-provided questionnaire label'});continue
        try:
            text=clean_text(record.Content)
        except ValueError as e:
            quarantine.append({'participant_id':pid,'reason':str(e)});continue
        fingerprint=hashlib.sha256(text.encode()).hexdigest(); words=text.split()
        if fingerprint in protected or any(tuple(words[i:i+12]) in protected_shingles for i in range(len(words)-11)):
            raise ValueError('External transcript overlaps an original protected participant')
        if fingerprint in seen:
            raise ValueError('Duplicate external text: participant independence needs reconciliation')
        seen[fingerprint]=pid
        total,ninth=metadata.loc[record.ID,['phq','q9']].to_numpy()
        scores,classes=phq8_labels([total],[ninth])
        rows.append({'participant_id':pid,'source_id':str(record.ID),'dataset':'EMU','role':'external_training_only',
                     'text':text,'text_sha256':fingerprint,'words':len(words),'phq9_total':int(total),'item9':int(ninth),
                     'phq8_score':int(scores[0]),'severity_class':int(classes[0])})
    return rows,quarantine


def inputs(source=SOURCE):
    source=Path(source);manifest=json.loads((source/'download_manifest.json').read_text())
    tree=json.loads((source/'source_tree.json').read_text())
    if manifest['revision'] != REVISION or tree['sha'] != REVISION or tree.get('truncated'):
        raise ValueError('Source revision changed')
    bindings={str(source/'download_manifest.json'):sha256(source/'download_manifest.json'),str(source/'source_tree.json'):sha256(source/'source_tree.json')}
    for r in manifest['files']:
        path=Path(r['path'])
        if not path.resolve().is_relative_to(source.resolve()) or path.is_symlink():
            raise ValueError('Unexpected source path')
        b=path.read_bytes(); entry=next(x for x in tree['tree'] if x['path']==str(path.relative_to(source)))
        if sha256(path)!=r['sha256'] or len(b)!=entry['size'] or hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()!=entry['sha']:
            raise ValueError('Author source checksum changed')
        bindings[str(path)]=r['sha256']
    # This reads no original held-out labels: original texts are used only for contamination detection.
    corpus=Path('results/transcript_corpus/participants.json'); documents=json.loads(corpus.read_text())
    bindings[str(corpus)]=sha256(corpus)
    transcripts=pd.read_csv(source/'EMU_data/text_features/transcripts_audio_open.csv',encoding='cp1252',dtype={'ID':str})
    demographics=pd.read_csv(source/'EMU_data/demographicsEMU.csv',dtype={'id':str})
    rows,quarantine=import_rows(transcripts,demographics,documents)
    return rows,quarantine,bindings


def prepare(output=OUTPUT):
    output=Path(output)
    if output.exists(): raise FileExistsError('Preserving existing external features')
    rows,quarantine,bindings=inputs()
    from .best_accuracy_inference import load_bundle
    contract,_=load_bundle(); encoder_root=Path(contract['encoder_root'])
    for name,expected in contract['encoder']['files'].items():
        if sha256(encoder_root/name)!=expected['sha256']: raise ValueError('Original encoder changed')
        bindings[str(encoder_root/name)]=expected['sha256']
    for name in ['external_emu.py','semantic_features.py','transcript_corpus.py','age_language_accuracy.py']:
        p=Path('src')/name;bindings[str(p)]=sha256(p)
    context={'protocol':PROTOCOL,'revision':REVISION,'source_bindings':bindings,
             'cleaning':'Same original label-free cleaning; CP1252 decoded before normalization',
             'label_derivation':'PHQ8 = author PHQ9 total minus author ninth item; original five-bin cutoffs',
             'external_roles':'All imported EMU people are external training only; no generated participants',
             'overlap_check':'Exact normalized text and any shared consecutive twelve-word span against all 275 original documents; no held-out labels accessed',
             'identity_limit':'Independent documented source cohort; text checks cannot conclusively exclude an unidentified person present in both source studies',
             'cross_domain_limit':'Short unscripted mobile responses differ from long E-DAIC interviews',
             'encoder_finetuned':False,'uploads':False}
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    features=[]
    with threadpool_limits(limits=2),offline_network():
        tokenizer,encoder=local_encoder(encoder_root)
        for row in rows:
            encoded=encode_document(row['text'],tokenizer,encoder)
            features.append({'participant_id':row['participant_id'],'semantic':encoded['embedding'],
                             'language':language_features(row['text']).tolist(),'tokens':encoded['tokens']})
    output.mkdir(parents=True)
    atomic_json(output/'protocol.json',{'context':context,'context_hash':digest(context)})
    atomic_json(output/'participants.json',rows);atomic_json(output/'quarantine.json',quarantine)
    atomic_json(output/'features.json',features)
    result={'complete':True,'context_hash':digest(context),'source_transcripts':len(rows)+len(quarantine),
            'imported_participants':len(rows),'quarantined_participants':len(quarantine),
            'class_supports':np.bincount([r['severity_class'] for r in rows],minlength=5).tolist(),
            'original_train_participants':163,'original_participants_removed':0,'roles':'external_training_only',
            'files':{str(output/n):sha256(output/n) for n in ['protocol.json','participants.json','quarantine.json','features.json']}}
    atomic_json(output/'completion.json',result)
    return result


def load_verified(output=OUTPUT):
    output=Path(output);r=json.loads((output/'completion.json').read_text());env=json.loads((output/'protocol.json').read_text())
    if not r['complete'] or digest(env['context'])!=env['context_hash'] or r['context_hash']!=env['context_hash']:
        raise ValueError('Incomplete external feature cache')
    for path,h in {**r['files'],**env['context']['source_bindings']}.items():
        if sha256(path)!=h:raise ValueError('External provenance or feature checksum changed')
    rows=json.loads((output/'participants.json').read_text());features=json.loads((output/'features.json').read_text())
    if [x['participant_id'] for x in rows]!=[x['participant_id'] for x in features] or len(rows)!=r['imported_participants'] or len({x['participant_id'] for x in rows})!=len(rows):
        raise ValueError('External participant ownership changed')
    scores,y=phq8_labels([x['phq9_total'] for x in rows],[x['item9'] for x in rows])
    if [x['phq8_score'] for x in rows]!=scores.tolist() or [x['severity_class'] for x in rows]!=y.tolist() or any(x['role']!='external_training_only' for x in rows):
        raise ValueError('External target derivation or training role changed')
    x={k:np.asarray([r[k] for r in features],float) for k in ['semantic','language']}
    if x['semantic'].shape!=(len(rows),384) or x['language'].shape!=(len(rows),16) or not all(np.isfinite(v).all() for v in x.values()) or not np.allclose(np.linalg.norm(x['semantic'],axis=1),1):
        raise ValueError('External inference feature contract changed')
    return x,y,[r['participant_id'] for r in rows],{**r['files'],str(output/'completion.json'):sha256(output/'completion.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','verify']);a=p.parse_args()
    print(json.dumps(prepare() if a.action=='prepare' else {'verified_participants':len(load_verified()[1])},indent=2))
