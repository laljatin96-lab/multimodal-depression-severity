"""Diagnose all severe training participants using the verified 49.08% saved OOF run."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from . import semantic_severe_repair as baseline
from .batch_inference import load_record, atomic_json
from .common import digest
from .final_evaluation import sha256
from .modality_diagnostics import nested_folds

ROOT = Path('results/semantic_severe_repair_v2')


def load_baseline():
    settings = json.loads(Path('configs/semantic_severe_repair.json').read_text())
    x, frame, provenance = baseline.load_inputs(settings)
    envelope = json.loads((ROOT/'protocol.json').read_text())
    context = {'protocol':baseline.PROTOCOL,'settings':settings,'training':frame.to_dict('records'),
        'folds':nested_folds(frame,settings),'provenance':provenance,
        'feature_hashes':{k:digest(np.asarray(x[k]).tolist()) for k in settings['feature_inputs']},
        'code_sha256':sha256(baseline.__file__),
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
        'support_code_sha256':{n:sha256(Path(baseline.__file__).with_name(n)) for n in
            ['semantic_pipeline_audit.py','semantic_accuracy_target.py','encoder_kernel_accuracy.py',
             'crossvalidated_baselines.py','modality_diagnostics.py','evaluate.py','raw_frame_uncertainty.py']}}
    if envelope != {'context':context,'context_hash':digest(context)}:
        raise ValueError('49.08% baseline context changed')
    completion = json.loads((ROOT/'completion.json').read_text())
    if not completion['complete'] or completion['context_hash'] != envelope['context_hash']:
        raise ValueError('Incomplete baseline')
    records = []
    for fold in context['folds']:
        job = ROOT/'jobs'/f"seed{fold['seed']}_fold{fold['fold']}"
        record = load_record(job/'result.json',envelope['context_hash'])
        baseline.validate_job(record,fold,settings,job,frame.severity_class.to_numpy(int))
        baseline.validate_controls(record,next(r for r in provenance['control_records'] if r['seed']==fold['seed'] and r['fold']==fold['fold']))
        records.append(record)
    bindings = {str(p):sha256(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}
    return x,frame,settings,records,{'baseline_context_hash':envelope['context_hash'],'bindings':bindings}


def diagnose(output='results/severe_detection_diagnosis'):
    out = Path(output)
    if out.exists():raise FileExistsError('Preserving existing diagnosis')
    x,frame,settings,records,provenance = load_baseline()
    y = frame.severity_class.to_numpy(int)
    metadata = pd.read_csv('data/metadata/participants.csv',dtype={'participant_id':str}).set_index('participant_id')
    detail = pd.read_csv('data/metadata/official_edaic/detailed_lables.csv',dtype={'Participant':str}).set_index('Participant')
    corpus = {r['participant_id']:r for r in json.loads(Path('results/transcript_corpus/participants.json').read_text())}
    aliases = {frame.participant_id.iloc[i]:f'S{n+1}' for n,i in enumerate(np.flatnonzero(y==4))}
    rows = []
    for record in records:
        result = record['policies']['accuracy']; candidate = result['candidate']
        values = x[candidate['feature']]
        scaler = StandardScaler().fit(values[record['fit']])
        fit_x = scaler.transform(values[record['fit']])
        probabilities = np.asarray(result['probabilities'])
        for row,index in enumerate(record['heldout']):
            if y[index] != 4:continue
            pid = frame.participant_id.iloc[index]; p = probabilities[row]
            z = scaler.transform(values[index:index+1])[0]
            distances = np.square(fit_x-z).mean(1)
            neighbors = np.argsort(distances,kind='stable')[:5]
            item_sum = int(pd.to_numeric(detail.loc[pid].filter(regex='^PHQ8_[1-8]_')).sum())
            score = int(metadata.loc[pid,'phq8_score'])
            if not 20 <= score <= 24 or item_sum != score:raise ValueError('Severe PHQ labels inconsistent')
            rows.append({'case':aliases[pid],'participant_id':pid,'seed':record['seed'],'fold':record['fold'],
                'phq8':score,'item_sum':item_sum,'predicted_class':int(p.argmax()),'severe_probability':float(p[4]),
                'winning_probability':float(p.max()),'severe_rank':int(1+(p>p[4]).sum()),
                'words':corpus[pid]['words'],'finite_features':bool(np.isfinite(values[index]).all()),
                'rms_standardized_feature':float(np.sqrt(np.mean(z*z))),
                'nearest_fitting_labels':json.dumps(y[np.asarray(record['fit'])[neighbors]].tolist()),
                **{f'p{k}':float(p[k]) for k in range(5)}})
    out.mkdir(parents=True)
    table = pd.DataFrame(rows).sort_values(['case','seed']);table.to_csv(out/'private_case_diagnosis.csv',index=False)
    public = table.drop(columns=['participant_id']);public.to_csv(out/'case_diagnosis.csv',index=False)
    report = {'passed':True,'training_counts':np.bincount(y,minlength=5).tolist(),'severe_participants':len(aliases),
        'all_severe_labels_match_items':True,'all_severe_features_finite':bool(table.finite_features.all()),
        'severe_probability_range':[float(table.severe_probability.min()),float(table.severe_probability.max())],
        'severe_predictions_correct':0,'baseline_accuracy':0.49079754601226994,
        'case_aliases_private':True,'provenance':provenance,
        'limitations':['whole-interview speaker identity unverified','no new raw archive byte reread',
                       'current label consistency does not independently authenticate provider identities',
                       'prior exposure and repeated training searches prevent independent confirmation']}
    atomic_json(out/'diagnosis.json',report)
    print(public[['case','seed','phq8','predicted_class','severe_probability','severe_rank','words']].to_string(index=False))
    return report

if __name__ == '__main__':
    diagnose()
