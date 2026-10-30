"""Verify saved exploratory LSTMs and report every seed without test exposure."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import digest, joined_manifest, write_json
from .datasets import fit_scalers
from .exploratory_baselines import load_audited_sequences
from .independent_datasets import independent_loader
from .independent_train import PROTOCOL, build_model, predictions
from .train import choose_device


def point_metrics(y, pred):
    cm = np.bincount(5*np.asarray(y, int)+np.asarray(pred, int), minlength=25).reshape(5,5)
    true = cm.sum(1)
    predicted = cm.sum(0)
    correct = cm.diagonal()
    f1 = np.divide(2*correct, true+predicted, out=np.zeros(5), where=(true+predicted)>0)
    recall = np.divide(correct, true, out=np.zeros(5), where=true>0)
    return np.array([correct.sum()/cm.sum(), f1.mean(), recall[true>0].mean()])


def paired_mean_bootstrap(y, baseline, candidate, repetitions=2000, seed=42):
    """Resample participants jointly across models/seeds; average seed metrics."""
    y, baseline, candidate = np.asarray(y), np.asarray(baseline), np.asarray(candidate)
    if baseline.shape != candidate.shape or baseline.ndim != 2 or baseline.shape[1] != len(y):
        raise ValueError('Paired seed predictions must have matching participant dimensions')
    def difference(indices):
        return np.mean([point_metrics(y[indices], c[indices])-point_metrics(y[indices], b[indices]) for b,c in zip(baseline,candidate)], axis=0)
    rng = np.random.default_rng(seed)
    draws = np.array([difference(rng.integers(0,len(y),len(y))) for _ in range(repetitions)])
    point = difference(np.arange(len(y)))
    intervals = np.percentile(draws,[2.5,97.5],axis=0)
    return {key:{'mean_seed_delta':float(point[i]),'paired_participant_ci95':intervals[:,i].tolist()} for i,key in enumerate(['accuracy','macro_f1','balanced_accuracy'])}


def generate_report(matrix_root='checkpoints/independent_lstm', output='results/independent_lstm'):
    torch.set_num_threads(2)
    root, out = Path(matrix_root), Path(output)
    completion = json.loads((root/'completion.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    if not completion['complete'] or protocol['pilot'] or protocol['protocol'] != PROTOCOL:
        raise ValueError('A completed full independent LSTM matrix is required')
    config = protocol['config']
    manifest = joined_manifest(config)
    sequences, source_provenance = load_audited_sequences(config, manifest)
    provenance = {'protocol': PROTOCOL, 'manifest_hash': digest(manifest.to_dict('records')), 'features': source_provenance}
    if provenance != protocol['provenance']:
        raise ValueError('Data changed since training')
    runs = pd.read_csv(root/'validation_runs.csv')
    if len(runs) != len(protocol['seeds'])*len(protocol['variants']) or runs.duplicated(['model','seed']).any():
        raise ValueError('Incomplete or duplicated matrix runs')
    validation = manifest[manifest.split == 'validation'].reset_index(drop=True)
    train_ids = manifest.loc[manifest.split == 'train','participant_id'].tolist()
    fresh_scalers = fit_scalers(sequences,train_ids)
    predictions_by_model, groups, verification, histories, per_class = {}, [], [], [], []
    for row in runs.itertuples():
        run = Path(row.run)
        state = torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        if state['provenance'] != provenance or state['scalers']['fit_participant_ids'] != train_ids:
            raise ValueError('Checkpoint provenance or scaler participants changed')
        for modality in ['audio','visual']:
            for key in ['mean','std']:
                np.testing.assert_allclose(state['scalers'][modality][key],fresh_scalers[modality][key],rtol=0,atol=0)
        device = choose_device(state['config']['device'])
        model = build_model(sequences,state['config'],state['gender']).to(device)
        model.load_state_dict(state['model'])
        ids, probs = predictions(model,independent_loader(validation,sequences,state['scalers'],state['config']),device)
        saved = pd.read_csv(run/'validation/predictions.csv',dtype={'participant_id':str})
        if ids != saved.participant_id.tolist() or ids != validation.participant_id.tolist():
            raise ValueError('Saved prediction IDs differ from validation manifest')
        np.testing.assert_allclose(probs,saved[[f'p{i}' for i in range(5)]].to_numpy(),rtol=1e-5,atol=1e-7)
        np.testing.assert_array_equal(saved.severity_class,validation.severity_class)
        if (run/'test').exists():
            raise ValueError('Unexpected test evaluation artifacts')
        predictions_by_model.setdefault(row.model,{})[row.seed] = probs.argmax(1)
        report = json.loads((run/'validation/metrics.json').read_text())
        per_class.extend({'model':row.model,'seed':row.seed,**values} for values in report['per_class'])
        for group, values in report['group_metrics'].items():
            groups.append({'model':row.model,'seed':row.seed,'gender':group,**{key:values[key] for key in ['participants','accuracy','macro_f1','balanced_accuracy']}})
        history = pd.read_csv(run/'history.csv')
        histories.append({'model':row.model,'seed':row.seed,'epochs_run':len(history),'selected_epoch':int(row.selected_epoch),'seconds':float(history.seconds.sum()),'training_parameters':sum(p.numel() for p in model.parameters())})
        verification.append({'model':row.model,'seed':row.seed,'checkpoint_replay_matches':True,'train_only_scalers_match':True,'test_evaluated':False})
    out.mkdir(parents=True,exist_ok=True)
    summary = runs.groupby('model')[['accuracy','macro_f1','balanced_accuracy']].agg(['mean','std'])
    summary.to_csv(out/'validation_summary.csv')
    runs.to_csv(out/'validation_runs.csv',index=False)
    group_frame = pd.DataFrame(groups)
    group_frame.to_csv(out/'validation_groups.csv',index=False)
    group_frame.groupby(['model','gender'])[['accuracy','macro_f1','balanced_accuracy']].agg(['mean','std']).to_csv(out/'validation_group_summary.csv')
    pd.DataFrame(per_class).to_csv(out/'validation_per_class.csv',index=False)
    pd.DataFrame(histories).to_csv(out/'compute.csv',index=False)
    write_json(out/'verification.json',{'runs':verification,'fit_participants':len(train_ids),'validation_participants':len(validation),'all_dataset_participants':len(manifest),'test_evaluated':False})
    baseline_name = 'Independent-AV'
    seed_order = protocol['seeds']
    baseline = np.stack([predictions_by_model[baseline_name][seed] for seed in seed_order])
    comparisons = {}
    for name in sorted(predictions_by_model):
        if name != baseline_name:
            candidate = np.stack([predictions_by_model[name][seed] for seed in seed_order])
            comparisons[name] = paired_mean_bootstrap(validation.severity_class.to_numpy(),baseline,candidate)
    balanced_baseline = np.stack([predictions_by_model['Independent-AV-balanced'][seed] for seed in seed_order])
    balanced_gender = np.stack([predictions_by_model['Independent-AV-gender-balanced'][seed] for seed in seed_order])
    gender_effect = paired_mean_bootstrap(validation.severity_class.to_numpy(),balanced_baseline,balanced_gender)
    write_json(out/'paired_validation_bootstrap.json',{'baseline':baseline_name,'repetitions':2000,'seed':42,'resampling_unit':'validation participant, jointly across all paired seeds/models','limitations':'Conditional on three fitted seeds and validation-selected checkpoints; no correction for model selection; not test-set inference or a full estimate of training uncertainty','comparisons':comparisons,'gender_effect_with_balancing':{'baseline':'Independent-AV-balanced','candidate':'Independent-AV-gender-balanced','metrics':gender_effect}})
    lines = ['# Exploratory independent-modality LSTM results','',f"All {len(manifest)} supplied participants are accounted for: {len(train_ids)} training, {len(validation)} validation and {completion['test_participants_reserved']} reserved test. Every run used the fixed official split. No test predictions or performance were computed.",'','Separate audio/visual LSTMs encode 30-second windows. Hidden embeddings are averaged independently within each participant/modality before fusion and one five-class participant loss. Audio and visual window counts can differ. Recorded gender and inverse training-class weighting are varied under the same training budget.','',f"Seeds: {', '.join(map(str,seed_order))}. Maximum {config['epochs']} epochs, early-stopping patience {config['patience']}, hidden size {config['hidden_size']}. All runs use validation macro-F1 for checkpoint selection. Values below are mean ± sample standard deviation across seeds.",'','| Model | Accuracy | Five-class macro-F1 | Balanced accuracy |','|---|---:|---:|---:|']
    for name,row in summary.iterrows():
        cells = [f"{row[(key,'mean')]:.4f} ± {row[(key,'std')]:.4f}" for key in ['accuracy','macro_f1','balanced_accuracy']]
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    best_name = summary[('macro_f1','mean')].idxmax()
    best_mean = summary.loc[best_name,('macro_f1','mean')]
    lines += ['',f"Highest mean neural macro-F1: {best_name}, {best_mean:.4f}. The current neural matrix does not exceed the earlier AV logistic macro-F1 of 0.3509. Plain and gender-only unbalanced LSTMs mostly favor the majority class; weighting improves minority-class coverage. None of these LSTM runs correctly identifies the sole severe-class validation participant. This is a weak exploratory result, not evidence of a reliable detector."] if best_mean < .3509 else []
    lines += ['','The previously selected AV logistic summary baseline had validation macro-F1 0.3509 and accuracy 0.3571; the majority predictor had macro-F1 0.1268 and accuracy 0.4643. Logistic regularization and neural epochs were selected on this same validation set, so comparisons are exploratory.','', '## Paired comparison with independent AV','', '| Variant | Mean macro-F1 difference | Paired participant bootstrap 95% interval |','|---|---:|---|']
    for name,report in comparisons.items():
        r = report['macro_f1']; lo,hi = r['paired_participant_ci95']
        lines.append(f"| {name} | {r['mean_seed_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    effect = gender_effect['macro_f1']
    lo,hi = effect['paired_participant_ci95']
    lines += ['',f"Isolating gender within balanced training gives mean macro-F1 difference {effect['mean_seed_delta']:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}]. This comparison separates gender input from class weighting; the combined variant's difference from plain AV cannot establish a gender benefit."]
    lines += ['','Intervals jointly resample the same 56 validation participants across paired models and all three seeds. They are conditional on the fitted models and selected checkpoints; they do not correct validation selection optimism.','', '## Practical limits','', '- Event-level AV synchronization remains unverified. This architecture does not pair modality windows, but retains the original cached common-interval crop and 0.5-second averaging.', '- Unsegmented audio may contain interviewer speech. No participant-only isolation claim is made.', '- Validation class support is 26/15/8/6/1. Severe-class and subgroup results are unstable; no clinical-validity or fairness claim is justified.', '- No test tuning, random target creation, gender inference, new source extraction or participant exclusion was used. This is a documented extension, not a faithful paper reproduction.','',f"All {len(verification)} saved best checkpoints reproduce their stored validation probabilities, and their scalers match training-only statistics. Saved checkpoints, optimizer/RNG state, histories and individual group/per-class reports are in {root}. Protocol, source hashes and participant window coverage are recorded there.",'', 'Outputs: validation_runs.csv, validation_summary.csv, validation_groups.csv, validation_group_summary.csv, compute.csv, paired_validation_bootstrap.json, verification.json and comparison.png.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    plot_summary(summary,out)
    return {'runs':len(runs),'summary':str(out/'validation_summary.csv'),'report':str(out/'REPORT.md')}


def plot_summary(summary,out):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(Path('results/.matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [name.replace('Independent-','').replace('-gender',' + gender').replace('-balanced','\nbalanced') for name in summary.index]
    fig,axes = plt.subplots(1,2,figsize=(10,4.4))
    logistic = json.loads(Path('results/exploratory_baselines/LR-AV/C_1/validation/metrics.json').read_text())
    for axis,metric,reference in [(axes[0],'macro_f1',logistic['macro_f1']),(axes[1],'accuracy',logistic['accuracy'])]:
        axis.bar(np.arange(len(summary)),summary[(metric,'mean')],yerr=summary[(metric,'std')],capsize=4,color=['#386a90','#8eb0c5','#577b66','#a0b89b'])
        axis.axhline(reference,color='#a35c35',linestyle='--',label='AV logistic baseline')
        axis.set_xticks(np.arange(len(summary)),labels,fontsize=9)
        axis.set_ylim(0,.65)
        axis.set_ylabel('Validation '+('five-class macro-F1' if metric=='macro_f1' else 'accuracy'))
        axis.legend(fontsize=8)
    fig.suptitle('Exploratory independent-modality LSTMs: mean ± SD across three seeds')
    fig.tight_layout()
    fig.savefig(out/'comparison.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix',default='checkpoints/independent_lstm')
    parser.add_argument('--output',default='results/independent_lstm')
    args = parser.parse_args()
    print(json.dumps(generate_report(args.matrix,args.output),indent=2))
