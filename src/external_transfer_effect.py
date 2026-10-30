"""Participant-paired source/no-source comparison; never refits or tunes models."""
from pathlib import Path
import pandas as pd,numpy as np,json
from src.raw_frame_uncertainty import participant_counts,weighted_metrics


def run():
    out=Path('results/external_transfer_accuracy');predictions={}
    if (out/'external_data_effect.json').exists():
        raise FileExistsError('Preserving completed source-effect analysis')
    for name in ['transfer_only','no_external']:
     tables=[pd.read_csv(out/'oof'/f'{name}_seed{s}/predictions.csv',dtype={'participant_id':str}) for s in [42,137,2026]]
     for t in tables:assert t.participant_id.tolist()==tables[0].participant_id.tolist()
     predictions[name]=np.stack([t.predicted_label.to_numpy(int) for t in tables])
    y=tables[0].severity_class.to_numpy(int)
    counts=participant_counts(len(y),5000,20261025)
    boot={k:weighted_metrics(y,p,counts) for k,p in predictions.items()}
    point={k:weighted_metrics(y,p,np.ones((1,len(y)),int))[0] for k,p in predictions.items()}
    limits=np.percentile(boot['transfer_only']-boot['no_external'],[2.5,97.5],axis=0)
    rows=[{'metric':m,'candidate':'transfer_only','reference':'no_external','difference':float(point['transfer_only'][i]-point['no_external'][i]),'ci95_low':float(limits[0,i]),'ci95_high':float(limits[1,i])} for i,m in enumerate(['accuracy','macro_f1','balanced_accuracy'])]
    pd.DataFrame(rows).to_csv(out/'external_data_effect.csv',index=False)
    np.savez_compressed(out/'external_data_effect_bootstrap.npz',participant_counts=counts,differences=boot['transfer_only']-boot['no_external'])
    a=predictions['transfer_only']==y;b=predictions['no_external']==y
    result={'reference':'no_external','candidate':'transfer_only','gained':int((a&~b).sum()),'lost':int((b&~a).sum()),'correct':int(a.sum()),'reference_correct':int(b.sum()),'participants_net_better':int((a.sum(0)>b.sum(0)).sum()),'participants_net_worse':int((a.sum(0)<b.sum(0)).sum()),'independent_confirmation':False}
    (out/'external_data_effect.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'effect':result,'intervals':rows},indent=2))


if __name__=="__main__":
    run()
