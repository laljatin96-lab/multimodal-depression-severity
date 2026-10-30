"""Freeze existing models before a single final held-out evaluation; never train."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from .common import digest, encode_gender, joined_manifest, write_json
from .crossvalidated_baselines import PROTOCOL as CV_PROTOCOL, VARIANTS, five_probabilities, variant_name
from .datasets import fit_scalers
from .evaluate import write_report
from .exploratory_baselines import load_audited_sequences, pool_modalities
from .independent_datasets import independent_loader
from .independent_train import PROTOCOL as LSTM_PROTOCOL, build_model, predictions
from .report_independent import paired_mean_bootstrap
from .train import choose_device, record_result

PROTOCOL = 'frozen-final-five-class-evaluation-v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal_plan(path, plan):
    path = Path(path)
    if path.exists() or Path(plan['output']).exists():
        raise FileExistsError('Preserving existing final plan/evaluation')
    write_json(path, {'plan': plan, 'plan_sha256': digest(plan)})


def verify_plan(path):
    envelope = json.loads(Path(path).read_text())
    plan = envelope['plan']
    if plan['protocol'] != PROTOCOL or envelope['plan_sha256'] != digest(plan):
        raise ValueError('Frozen plan integrity mismatch')
    for filename, expected in plan['bound_files'].items():
        if sha256(filename) != expected:
            raise ValueError(f'Frozen input changed: {filename}')
    return plan


def freeze(plan_path, output, cv_root='results/crossvalidated_baselines',
           baseline_root='results/exploratory_baselines', neural_root='checkpoints/independent_lstm'):
    if Path(plan_path).exists() or Path(output).exists():
        raise FileExistsError('Preserving existing final plan/evaluation')
    cv, baselines, neural = Path(cv_root), Path(baseline_root), Path(neural_root)
    cv_protocol = json.loads((cv/'protocol.json').read_text())
    if cv_protocol['protocol'] != CV_PROTOCOL or not json.loads((cv/'completion.json').read_text())['complete']:
        raise ValueError('Completed training-CV experiment required')
    config = cv_protocol['feature_config']
    manifest = joined_manifest(config)
    _, sources = load_audited_sequences(config, manifest)
    manifest_hash = digest(manifest.to_dict('records'))
    if cv_protocol['manifest_hash'] != manifest_hash or cv_protocol['source_provenance'] != sources:
        raise ValueError('Training-CV data changed')
    baseline_protocol = json.loads((baselines/'protocol.json').read_text())
    if baseline_protocol['manifest_hash'] != manifest_hash or json.loads((baselines/'feature_provenance.json').read_text()) != sources:
        raise ValueError('Diagnostic baseline data changed')
    neural_protocol = json.loads((neural/'protocol.json').read_text())
    provenance = {'protocol': LSTM_PROTOCOL, 'manifest_hash': manifest_hash, 'features': sources}
    if neural_protocol['pilot'] or neural_protocol['provenance'] != provenance or not json.loads((neural/'completion.json').read_text())['complete']:
        raise ValueError('Completed compatible neural matrix required')
    models = []
    bindings = [Path(config['metadata']), Path(config['splits']), cv/'protocol.json', cv/'frozen_selections.json',
                baselines/'protocol.json', baselines/'feature_provenance.json', neural/'protocol.json']
    for gender, balanced in VARIANTS:
        name = variant_name(gender, balanced)
        models.append({'model': name, 'family': 'primary_logistic', 'kind': 'summary',
                       'seed': cv_protocol['settings']['fit_seed'], 'artifact': str(cv/name/'model.pkl'),
                       'validation_predictions': str(cv/name/'validation/predictions.csv')})
    baseline_rows = pd.read_csv(baselines/'validation_summary.csv').set_index('model')
    for name in ['majority', 'LR-gender-only']:
        root = Path(baseline_rows.loc[name, 'run'])
        models.append({'model': name, 'family': 'diagnostic', 'kind': 'summary',
                       'seed': baseline_protocol['seed'], 'artifact': str(root/'model.pkl'),
                       'validation_predictions': str(root/'validation/predictions.csv')})
    for seed in neural_protocol['seeds']:
        for gender, balanced in neural_protocol['variants']:
            name = 'Independent-AV' + ('-gender' if gender else '') + ('-balanced' if balanced else '')
            root = neural/f'{name}_seed{seed}'
            models.append({'model': name, 'family': 'secondary_lstm', 'kind': 'independent_lstm',
                           'seed': seed, 'artifact': str(root/'best.pt'),
                           'validation_predictions': str(root/'validation/predictions.csv')})
    for item in models:
        bindings.extend([Path(item['artifact']), Path(item['validation_predictions'])])
    contrasts = [
        {'baseline': 'CV-LR-AV-balanced', 'candidate': 'CV-LR-AV-gender-balanced', 'role': 'primary_gender_effect'},
        {'baseline': 'CV-LR-AV', 'candidate': 'CV-LR-AV-gender', 'role': 'supporting_unbalanced_gender_effect'},
        {'baseline': 'CV-LR-AV', 'candidate': 'CV-LR-AV-balanced', 'role': 'supporting_class_weight_effect'},
        {'baseline': 'CV-LR-AV', 'candidate': 'CV-LR-AV-gender-balanced', 'role': 'supporting_combined_effect'},
        {'baseline': 'Independent-AV-balanced', 'candidate': 'Independent-AV-gender-balanced', 'role': 'secondary_neural_gender_effect'},
    ]
    plan = {'protocol': PROTOCOL, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'output': str(output),
            'feature_config': config, 'manifest_hash': manifest_hash, 'source_provenance': sources,
            'models': models, 'bound_files': {str(path): sha256(path) for path in bindings},
            'primary_metric': 'Participant-level macro-F1 over all five declared classes; undefined F1 contributes zero',
            'contrasts': contrasts, 'bootstrap_repetitions': 2000, 'bootstrap_seed': 42,
            'reference_model_selected_before_test': 'CV-LR-AV',
            'selection_rationale': 'Highest mean training OOF macro-F1 among the four matched logistic variants; prior validation also favors AV summaries over LSTMs',
            'test_rule': 'Evaluate every frozen model, argmax probabilities, no refitting, threshold changes, ensembling or best-seed selection',
            'neural_reporting': 'Mean and sample SD across all frozen paired seeds; no test-selected seed',
            'uncertainty': 'Paired participant bootstrap, jointly across neural seeds; conditional on fitted models; supporting comparisons unadjusted for multiplicity',
            'scope': 'All audited supplied sources; one record per participant; official train/validation/test unchanged',
            'limitations': ['Prior test use outside this project is unknown', 'Validation informed architecture and protocol decisions',
                           'Unverified timing origins; inherited common-time crop and 0.5-second averaging',
                           'Participant-only speech is unverified; unsegmented audio may include interviewer speech',
                           'Sparse classes/subgroups; questionnaire research targets, no clinical or fairness validation'],
            'after_test': 'The test set is consumed for final reporting. Further model development must disclose reuse and seek an independent holdout.'}
    seal_plan(plan_path, plan)
    return {'plan': str(plan_path), 'models_frozen': len(models), 'test_evaluated': False}


def summary_inputs(state, matrix, frame):
    if state.get('gender_input'):
        gender = np.stack([encode_gender(value, state['gender_categories']) for value in frame.gender])
        return gender if state.get('gender_only') else np.column_stack([matrix, gender])
    return matrix


def replay_check(probabilities, frame, saved_path):
    saved = pd.read_csv(saved_path, dtype={'participant_id': str})
    if saved.participant_id.tolist() != frame.participant_id.tolist():
        raise ValueError('Stored validation participant order mismatch')
    np.testing.assert_array_equal(saved.severity_class, frame.severity_class)
    np.testing.assert_allclose(probabilities, saved[[f'p{i}' for i in range(5)]].to_numpy(), rtol=1e-5, atol=1e-7)


def evaluate(plan_path):
    plan = verify_plan(plan_path)
    out = Path(plan['output'])
    if out.exists():
        raise FileExistsError('Final evaluation already exists; preserve it and use report-only to regenerate summaries')
    config = plan['feature_config']
    manifest = joined_manifest(config)
    sequences, sources = load_audited_sequences(config, manifest)
    if digest(manifest.to_dict('records')) != plan['manifest_hash'] or sources != plan['source_provenance']:
        raise ValueError('Frozen dataset changed')
    training = manifest[manifest.split.eq('train')].reset_index(drop=True)
    validation = manifest[manifest.split.eq('validation')].reset_index(drop=True)
    test = manifest[manifest.split.eq('test')].reset_index(drop=True)
    if not len(test) or not set(test.participant_id).isdisjoint(set(training.participant_id) | set(validation.participant_id)):
        raise ValueError('Invalid held-out partition')
    train_ids = training.participant_id.tolist()
    fresh_scalers = fit_scalers(sequences, train_ids)
    matrices = {split: np.stack([pool_modalities(sequences[pid]) for pid in frame.participant_id])
                for split, frame in [('train', training), ('validation', validation), ('test', test)]}
    torch.set_num_threads(2)
    loaded, checks = [], []
    # Validate every model before producing any test predictions.
    for item in plan['models']:
        if item['kind'] == 'summary':
            with Path(item['artifact']).open('rb') as handle:
                state = pickle.load(handle)  # Only trusted locally produced artifacts.
            if state['fit_participant_ids'] != train_ids:
                raise ValueError('Summary fit participants differ from official training')
            if item['family'] == 'primary_logistic' and state['protocol'] != CV_PROTOCOL:
                raise ValueError('Unexpected summary protocol')
            estimator = state['estimator']
            features = summary_inputs(state, matrices['train'], training)
            if hasattr(estimator, 'named_steps'):
                scaler = estimator.named_steps['standardscaler']
                np.testing.assert_allclose(scaler.mean_, features.mean(0, dtype=np.float64), rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(scaler.var_, features.var(0, dtype=np.float64), rtol=1e-12, atol=1e-12)
            probs = five_probabilities(estimator, summary_inputs(state, matrices['validation'], validation))
            replay_check(probs, validation, item['validation_predictions'])
            loaded.append((item, state, estimator, None))
        else:
            state = torch.load(item['artifact'], map_location='cpu', weights_only=False)
            expected = {'protocol': LSTM_PROTOCOL, 'manifest_hash': plan['manifest_hash'], 'features': sources}
            if state['protocol'] != LSTM_PROTOCOL or state['provenance'] != expected or state['scalers']['fit_participant_ids'] != train_ids:
                raise ValueError('Neural training provenance mismatch')
            for modality in ['audio', 'visual']:
                for key in ['mean', 'std']:
                    np.testing.assert_array_equal(state['scalers'][modality][key], fresh_scalers[modality][key])
            device = choose_device(state['config']['device'])
            model = build_model(sequences, state['config'], state['gender']).to(device)
            model.load_state_dict(state['model'])
            ids, probs = predictions(model, independent_loader(validation, sequences, state['scalers'], state['config']), device)
            if ids != validation.participant_id.tolist():
                raise ValueError('Neural validation prediction order changed')
            replay_check(probs, validation, item['validation_predictions'])
            loaded.append((item, state, model, device))
        checks.append({'model': item['model'], 'seed': item['seed'], 'train_only_preprocessing_verified': True,
                       'stored_validation_reproduced': True, 'frozen_artifact_sha256': sha256(item['artifact'])})
    out.mkdir(parents=True)
    write_json(out/'frozen_protocol.json', {'plan': plan, 'plan_sha256': digest(plan)})
    # This receipt is written before the first prediction so partial exposure is explicit.
    write_json(out/'test_access.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
                                       'test_set_exposed': True, 'purpose': 'frozen final reporting',
                                       'policy': plan['after_test'], 'plan_sha256': digest(plan)})
    manifest[['participant_id', 'split']].to_csv(out/'participant_coverage.csv', index=False)
    rows = []
    for item, state, model, device in loaded:
        if item['kind'] == 'summary':
            probs = five_probabilities(model, summary_inputs(state, matrices['test'], test))
        else:
            ids, probs = predictions(model, independent_loader(test, sequences, state['scalers'], state['config']), device)
            if ids != test.participant_id.tolist():
                raise ValueError('Test prediction order mismatch')
        if not np.isfinite(probs).all() or (probs < 0).any():
            raise ValueError('Invalid model probabilities')
        np.testing.assert_allclose(probs.sum(1), 1, rtol=1e-5, atol=1e-6)
        run = out/f"{item['model']}_seed{item['seed']}"
        report = write_report(test, probs, run/'test', item['model'], item['seed'], 'test')
        rows.append({'model': item['model'], 'family': item['family'], 'seed': item['seed'], 'run': str(run),
                     **{key: report[key] for key in ['accuracy', 'macro_f1', 'balanced_accuracy', 'participants']}})
        record_result(run, {**report, 'model': 'FINAL-'+item['model']}, {**config, 'seed': item['seed']})
        print(f"{item['model']} seed={item['seed']}: test macro-F1={report['macro_f1']:.4f}, accuracy={report['accuracy']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(out/'test_runs.csv', index=False)
    verify_plan(plan_path)  # Catch model/config changes during evaluation too.
    write_json(out/'verification.json', {'models': checks, 'plan_sha256': digest(plan), 'models_refit': 0, 'test_evaluated': True})
    write_json(out/'completion.json', {'complete': True, 'models': len(rows), 'all_dataset_participants': len(manifest),
                                       'train_participants': len(training), 'validation_participants': len(validation),
                                       'test_participants': len(test), 'test_evaluated': True,
                                       'test_class_counts': np.bincount(test.severity_class, minlength=5).tolist(),
                                       'test_gender_counts': test.gender.value_counts().to_dict(),
                                       'completed_at_utc': datetime.now(timezone.utc).isoformat()})
    return report_final(out)


def report_final(output):
    out = Path(output)
    plan = verify_plan(out/'frozen_protocol.json')
    completion = json.loads((out/'completion.json').read_text())
    if not completion['complete']:
        raise ValueError('Final evaluation incomplete')
    runs = pd.read_csv(out/'test_runs.csv')
    if len(runs) != len(plan['models']) or runs.duplicated(['model', 'seed']).any():
        raise ValueError('Missing or duplicated final models')
    metrics_columns = ['accuracy', 'macro_f1', 'balanced_accuracy']
    summary = runs.groupby(['family', 'model'])[metrics_columns].agg(['mean', 'std'])
    summary.to_csv(out/'test_summary.csv')
    predictions_by_model, groups, per_class = {}, [], []
    labels, participant_ids = None, None
    for row in runs.itertuples():
        root = Path(row.run)/'test'
        frame = pd.read_csv(root/'predictions.csv', dtype={'participant_id': str})
        if participant_ids is None:
            participant_ids, labels = frame.participant_id.tolist(), frame.severity_class.to_numpy()
        if frame.participant_id.tolist() != participant_ids or not np.array_equal(labels, frame.severity_class) or not frame.split.eq('test').all():
            raise ValueError('Final predictions are not paired on the same test participants')
        predictions_by_model.setdefault(row.model, {})[row.seed] = frame.predicted_label.to_numpy()
        report = json.loads((root/'metrics.json').read_text())
        for gender, values in report['group_metrics'].items():
            groups.append({'model': row.model, 'seed': row.seed, 'gender': gender,
                           **{key: values[key] for key in ['participants', *metrics_columns]}})
        per_class.extend({'model': row.model, 'seed': row.seed, **values} for values in report['per_class'])
    pd.DataFrame(groups).to_csv(out/'test_groups.csv', index=False)
    pd.DataFrame(groups).groupby(['model', 'gender'])[['participants', *metrics_columns]].agg(['mean', 'std']).to_csv(out/'test_group_summary.csv')
    pd.DataFrame(per_class).to_csv(out/'test_per_class.csv', index=False)
    comparisons = []
    for contrast in plan['contrasts']:
        baseline, candidate = predictions_by_model[contrast['baseline']], predictions_by_model[contrast['candidate']]
        if set(baseline) != set(candidate):
            raise ValueError('Contrast requires paired identical seeds')
        seeds = sorted(baseline)
        value = paired_mean_bootstrap(labels, np.stack([baseline[s] for s in seeds]), np.stack([candidate[s] for s in seeds]),
                                      repetitions=plan['bootstrap_repetitions'], seed=plan['bootstrap_seed'])
        changes = [{'seed': seed, 'changed_predictions': int((baseline[seed] != candidate[seed]).sum()),
                    'correct_baseline': int((baseline[seed] == labels).sum()),
                    'correct_candidate': int((candidate[seed] == labels).sum()),
                    'changed_both_wrong': int(((baseline[seed] != candidate[seed]) & (baseline[seed] != labels) & (candidate[seed] != labels)).sum())}
                   for seed in seeds]
        comparisons.append({**contrast, 'seeds': seeds, 'metrics': value, 'prediction_changes': changes})
    write_json(out/'paired_test_bootstrap.json', {'comparisons': comparisons, 'uncertainty_definition': plan['uncertainty'],
                                                'repetitions': plan['bootstrap_repetitions'], 'seed': plan['bootstrap_seed']})
    lines = ['# Final held-out research evaluation', '',
             f"All {completion['all_dataset_participants']} supplied participants are accounted for: {completion['train_participants']} training, {completion['validation_participants']} validation and {completion['test_participants']} final test. Every supplied archive remains covered by the original audit; duplicate copies represent one participant.", '',
             f"The model list, artifact hashes, reference model, metric, and contrasts were frozen at {plan['frozen_at_utc']} before test predictions. All {completion['models']} models were evaluated without refitting or test-time tuning. Test class supports: {completion['test_class_counts']}; recorded-gender supports: {completion['test_gender_counts']}.", '',
             'CV logistic models were selected using training-only repeated folds. LSTM checkpoints were previously selected on validation macro-F1. Neural results below summarize all frozen seeds (mean ± sample SD), not an ensemble or a test-selected best seed. Values without SD represent one deterministic fitted model.', '',
             '| Family | Model | Test accuracy | Five-class macro-F1 | Balanced accuracy |',
             '|---|---|---:|---:|---:|']
    for (family, name), row in summary.iterrows():
        cells = [f"{row[(key, 'mean')]:.4f}" + (f" ± {row[(key, 'std')]:.4f}" if pd.notna(row[(key, 'std')]) else '') for key in metrics_columns]
        lines.append('| '+family+' | '+name+' | '+' | '.join(cells)+' |')
    lines += ['', f"The reference selected before test is **{plan['reference_model_selected_before_test']}**. The model with the largest observed test score is not automatically promoted to a new selected model.", '',
              '## Prespecified paired comparisons', '', '| Role / candidate minus baseline | Macro-F1 difference | Participant bootstrap 95% interval |', '|---|---:|---|']
    for contrast in comparisons:
        value = contrast['metrics']['macro_f1']; lo, hi = value['paired_participant_ci95']
        lines.append(f"| {contrast['role']}: {contrast['candidate']} minus {contrast['baseline']} | {value['mean_seed_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    primary_comparison = next(row for row in comparisons if row['role'] == 'primary_gender_effect')
    primary = primary_comparison['metrics']['macro_f1']
    lo, hi = primary['paired_participant_ci95']
    finding = 'does not establish a gender benefit' if lo <= 0 <= hi else ('supports a positive difference conditional on these fitted models' if lo > 0 else 'shows a negative difference conditional on these fitted models')
    lines += ['', f"The primary balanced logistic comparison {finding}. It estimates the effect of adding recorded gender under a matched training-CV selection budget. The combined intervention contrast does not isolate gender."]
    if len(primary_comparison['prediction_changes']) == 1:
        change = primary_comparison['prediction_changes'][0]
        lines += ['', f"In that primary comparison, {change['changed_predictions']} test prediction(s) changed; {change['changed_both_wrong']} changed from one incorrect class to another. Correct classifications were {change['correct_baseline']} without gender and {change['correct_candidate']} with gender out of {len(labels)}. The macro-F1 difference can reflect redistribution of false positives even when the count of correct predictions is unchanged."]
    severe = [row for row in per_class if row['class'] == 4 and row['support'] > 0]
    if severe and all(row['recall'] == 0 for row in severe):
        lines += ['', f"None of the {len(runs)} frozen models correctly classified either severe-class test participant." if completion['test_class_counts'][4] == 2 else f"None of the frozen models correctly classified a severe-class test participant (support {completion['test_class_counts'][4]})."]
    lines += ['',
              'Bootstrap intervals resample participants jointly across models/seeds, condition on fitted models, and are not calibrated clinical certainty. Supporting comparisons are unadjusted for multiplicity; sparse classes/subgroups and model selection constrain inference. Group metrics and per-class supports are reported separately, without a fairness claim.', '',
              '## Scope and next use', '', *['- '+item for item in plan['limitations']], '',
              '**The test set has now been used for final reporting. Do not tune further on these results and call the same test set untouched. Future changes need an independent evaluation set or explicit reuse disclosure.**', '',
              'All predictions, models, features and participant identifiers remain local. Frozen protocol, file hashes, preprocessing checks, test-access receipt, complete seed/group/per-class tables, and paired intervals accompany this report. No additional training was performed during final evaluation.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    plot_summary(summary, out)
    return {'report': str(out/'REPORT.md'), 'models': len(runs), 'test_evaluated': True}


def plot_summary(summary, out):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(Path('results/.matplotlib').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = [('primary_logistic', variant_name(g, b)) for g, b in VARIANTS]
    order += [('secondary_lstm', 'Independent-AV'+('-gender' if g else '')+('-balanced' if b else '')) for g, b in VARIANTS]
    order = [item for item in order if item in summary.index]
    values = summary.loc[order]
    labels = [name.replace('CV-LR-', 'LR ').replace('Independent-', 'LSTM ').replace('-gender', ' + gender').replace('-balanced', '\nweighted') for _, name in order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric, title in [(axes[0], 'macro_f1', 'Five-class macro-F1'), (axes[1], 'accuracy', 'Accuracy')]:
        means = values[(metric, 'mean')].to_numpy()
        errors = values[(metric, 'std')].fillna(0).to_numpy()
        axis.barh(np.arange(len(order)), means, xerr=errors, capsize=3, color=['#376a8c']*4+['#638366']*4)
        axis.set_yticks(np.arange(len(order)), labels, fontsize=9)
        axis.invert_yaxis()
        axis.set_xlim(0, max(.6, float(np.max(means+errors))+.05))
        axis.set_xlabel(title)
        if ('diagnostic', 'majority') in summary.index:
            axis.axvline(summary.loc[('diagnostic', 'majority'), (metric, 'mean')], color='#a15b38', linestyle='--', label='Majority baseline')
            axis.legend(fontsize=8, loc='lower right')
    fig.suptitle('Frozen final test evaluation — neural error bars show SD across frozen seeds')
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'evaluate', 'report-only'])
    parser.add_argument('--plan', default='results/final_evaluation_plan.json')
    parser.add_argument('--output', default='results/final_test')
    args = parser.parse_args()
    result = freeze(args.plan, args.output) if args.action == 'freeze' else evaluate(args.plan) if args.action == 'evaluate' else report_final(args.output)
    print(json.dumps(result, indent=2))
