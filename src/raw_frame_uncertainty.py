"""Paired participant uncertainty and error changes from saved training OOF outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .batch_inference import load_record
from .common import VERSION, digest, write_json
from .final_evaluation import sha256
from .modality_diagnostics import variant_name
from .raw_frame_comparison import VARIANTS
from .verify_raw_frame_stage import verify_stage

METRICS = ['accuracy', 'macro_f1', 'balanced_accuracy']
PROTOCOL = 'post-final-paired-training-participant-bootstrap-v1'


def participant_counts(n, repetitions, seed):
    if n < 1 or not 1 <= repetitions <= 50000:
        raise ValueError('Positive participants and 1–50000 bootstrap repetitions required')
    draws = np.random.default_rng(seed).integers(0, n, size=(repetitions, n))
    counts = np.zeros((repetitions, n), dtype=np.int32)
    np.add.at(counts, (np.arange(repetitions)[:, None], draws), 1)
    return counts


def weighted_metrics(y, predictions, counts):
    """Average partition metrics; each row of counts resamples whole participants."""
    y, predictions, counts = np.asarray(y), np.asarray(predictions), np.asarray(counts)
    if (y.ndim != 1 or not len(y) or predictions.ndim != 2 or predictions.shape[1] != len(y)
            or predictions.shape[0] < 1 or counts.ndim != 2 or counts.shape[1] != len(y)):
        raise ValueError('Mismatched participant/partition dimensions')
    for values in [y, predictions]:
        if not np.isfinite(values).all() or not np.equal(values, values.astype(int)).all() or (values < 0).any() or (values > 4).any():
            raise ValueError('Classes must be integers from zero through four')
    if (not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, counts.astype(int)).all()
            or not np.equal(counts.sum(1), len(y)).all()):
        raise ValueError('Each participant resample must contain exactly N nonnegative integer counts')
    result = np.zeros((len(counts), 3))
    weights = counts.astype(np.float64)
    for prediction in predictions.astype(int):
        indicators = np.eye(25)[5*y.astype(int)+prediction]
        cm = (weights@indicators).reshape(-1, 5, 5)
        actual, predicted = cm.sum(2), cm.sum(1)
        correct = np.diagonal(cm, axis1=1, axis2=2)
        f1 = np.divide(2*correct, actual+predicted, out=np.zeros_like(correct), where=actual+predicted > 0)
        recall = np.divide(correct, actual, out=np.zeros_like(correct), where=actual > 0)
        result += np.column_stack([correct.sum(1)/len(y), f1.mean(1), recall.sum(1)/(actual > 0).sum(1)])
    return result/len(predictions)


def checked_predictions(path, training, expected, seed, model):
    frame = pd.read_csv(path, dtype={'participant_id': str}, keep_default_na=False)
    if (frame.participant_id.tolist() != training.participant_id.tolist() or frame.participant_id.duplicated().any()
            or not np.array_equal(frame.severity_class, training.severity_class)
            or not frame.split.eq('train_nested_oof').all() or not frame.seed.eq(seed).all()
            or not frame.model_version.eq(VERSION+':'+model).all()):
        raise ValueError('Saved predictions do not match training IDs/targets/model/partition')
    probabilities = frame[[f'p{i}' for i in range(5)]].to_numpy(float)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Invalid saved probabilities')
    np.testing.assert_allclose(probabilities, expected, rtol=0, atol=1e-12)
    if not np.array_equal(frame.predicted_label, probabilities.argmax(1)):
        raise ValueError('Saved classes differ from probabilities')
    return frame.predicted_label.to_numpy(int)


def load_verified(settings):
    comparison_settings = json.loads(Path(settings['comparison_config']).read_text())
    root = Path(settings['comparison_directory'])
    verification = verify_stage(comparison_settings, root)
    if not verification['passed']:
        raise ValueError('Raw-frame stage must pass verification before uncertainty analysis')
    envelope = json.loads((root/'protocol.json').read_text())
    context = envelope['context']
    training = pd.DataFrame(context['training_participants'])
    seeds, folds = context['settings']['fold_seeds'], context['folds']
    cached = Path(comparison_settings['cached_diagnostic'])
    bindings = {settings['comparison_config']: sha256(settings['comparison_config'])}
    arrays = {}
    for representation, folder in [('cached', cached), ('raw', root)]:
        protocol = json.loads((folder/'protocol.json').read_text())
        bindings[str(folder/'protocol.json')] = sha256(folder/'protocol.json')
        for modality, balanced in VARIANTS:
            name = variant_name(modality, balanced)
            predictions = []
            for seed in seeds:
                probabilities = np.full((len(training), 5), np.nan)
                ownership = np.zeros(len(training), dtype=int)
                for fold in [f for f in folds if f['seed'] == seed]:
                    path = folder/'jobs'/f"seed{seed}_fold{fold['fold']}"/name/'result.json'
                    record = load_record(path, protocol['context_hash'])
                    if record['heldout'] != fold['heldout'] or record['model'] != name or record['seed'] != seed:
                        raise ValueError('Saved job ownership differs from verified folds')
                    probabilities[record['heldout']] = record['outer_probabilities']
                    ownership[record['heldout']] += 1
                    bindings[str(path)] = sha256(path)
                if not np.isfinite(probabilities).all() or not (ownership == 1).all():
                    raise ValueError('Each participant needs one OOF prediction per partition')
                path = folder/'oof'/f'{name}_seed{seed}'/'predictions.csv'
                model = name.replace('Nested-LR', 'RawFrame-LR') if representation == 'raw' else name
                predictions.append(checked_predictions(path, training, probabilities, seed, model))
                bindings[str(path)] = sha256(path)
            arrays[(representation, modality, balanced)] = np.stack(predictions)
    return training, seeds, arrays, bindings, context['coverage'], verification


def error_changes(training, seeds, old, new, modality, balanced):
    y = training.severity_class.to_numpy()
    transitions, classes = [], []
    for i, seed in enumerate(seeds):
        a, b = old[i] == y, new[i] == y
        transitions.append({'modality': modality, 'balanced': balanced, 'seed': seed, 'participants': len(y),
                            'both_correct': int((a & b).sum()), 'raw_only_correct': int((~a & b).sum()),
                            'cached_only_correct': int((a & ~b).sum()), 'both_wrong': int((~a & ~b).sum())})
        for label in range(5):
            select = y == label
            classes.append({'modality': modality, 'balanced': balanced, 'seed': seed, 'class': label,
                            'participants': int(select.sum()), 'cached_correct': int((a & select).sum()),
                            'raw_correct': int((b & select).sum()),
                            'cached_recall': float(a[select].mean()) if select.any() else None,
                            'raw_recall': float(b[select].mean()) if select.any() else None})
    people = [{'participant_id': pid, 'severity_class': int(y[i]), 'modality': modality, 'balanced': balanced,
               'partitions': len(seeds), 'cached_correct_partitions': int((old[:, i] == y[i]).sum()),
               'raw_correct_partitions': int((new[:, i] == y[i]).sum())} for i, pid in enumerate(training.participant_id)]
    return transitions, classes, people


def analyze(training, seeds, arrays, settings):
    y = training.severity_class.to_numpy()
    counts = participant_counts(len(training), settings['bootstrap_repetitions'], settings['bootstrap_seed'])
    point_counts = np.ones((1, len(training)), dtype=int)
    intervals, transitions, classes, people, draws = [], [], [], [], []
    for modality, balanced in VARIANTS:
        old, new = arrays[('cached', modality, balanced)], arrays[('raw', modality, balanced)]
        if old.shape != new.shape or old.shape != (len(seeds), len(y)):
            raise ValueError('Paired predictions must cover the same participants and partitions')
        old_point, new_point = weighted_metrics(y, old, point_counts)[0], weighted_metrics(y, new, point_counts)[0]
        delta = weighted_metrics(y, new, counts)-weighted_metrics(y, old, counts)
        draws.append(delta)
        limits = np.percentile(delta, [2.5, 97.5], axis=0)
        for i, metric in enumerate(METRICS):
            intervals.append({'modality': modality, 'balanced': balanced, 'metric': metric,
                              'cached_mean': old_point[i], 'raw_mean': new_point[i], 'raw_minus_cached': new_point[i]-old_point[i],
                              'ci95_low': limits[0, i], 'ci95_high': limits[1, i],
                              'interval_includes_zero': bool(limits[0, i] <= 0 <= limits[1, i])})
        t, c, p = error_changes(training, seeds, old, new, modality, balanced)
        transitions.extend(t); classes.extend(c); people.extend(p)
    return pd.DataFrame(intervals), pd.DataFrame(transitions), pd.DataFrame(classes), pd.DataFrame(people), counts, np.stack(draws)


def plot_intervals(intervals, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = intervals[intervals.metric.eq('macro_f1')]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for i, row in enumerate(table.itertuples()):
        ax.plot([row.ci95_low, row.ci95_high], [i, i], color='#477b94', linewidth=3)
        ax.scatter([row.raw_minus_cached], [i], color='#a95625', zorder=3, s=45)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_yticks(np.arange(len(table)), [r.modality+(' + weights' if r.balanced else '') for r in table.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel('Raw − cached five-class macro-F1')
    ax.set_title('Paired participant bootstrap: saved training predictions\nPointwise 95% intervals, conditional on fitted models')
    ax.grid(axis='x', alpha=.2)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run(settings, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError('Preserving existing uncertainty analysis; choose a fresh output directory')
    if not 1 <= settings['cpu_threads'] <= 2:
        raise ValueError('Use one or two CPU threads')
    training, seeds, arrays, bindings, coverage, verified = load_verified(settings)
    comparison_settings = json.loads(Path(settings['comparison_config']).read_text())
    raw_context = json.loads((Path(comparison_settings['raw_features'])/'protocol.json').read_text())['context']
    protected = [comparison_settings['raw_features'], comparison_settings['cached_diagnostic'],
                 settings['comparison_directory'], raw_context['feature_config']['dataset']]
    if any(out.resolve().is_relative_to(Path(p).resolve()) for p in protected):
        raise ValueError('Output must be outside input dataset/experiment directories')
    code_hashes = {name: sha256(Path(__file__).with_name(name)) for name in
                   ['raw_frame_uncertainty.py', 'verify_raw_frame_stage.py', 'raw_frame_comparison.py', 'common.py']}
    protocol = {'format': PROTOCOL, 'settings': settings, 'training_participants': training.to_dict('records'),
                'partitions': seeds, 'coverage': coverage, 'input_bindings': bindings, 'implementation_sha256': code_hashes,
                'numpy_version': np.__version__, 'pandas_version': pd.__version__, 'model_fitting': False,
                'validation_test_scoring': False, 'model_promotion': False}
    out.mkdir(parents=True)
    write_json(out/'protocol.json', {'protocol': protocol, 'protocol_sha256': digest(protocol)})
    write_json(out/'input_verification.json', verified)
    with threadpool_limits(limits=settings['cpu_threads']):
        intervals, transitions, classes, people, counts, draws = analyze(training, seeds, arrays, settings)
    # Recheck all source/code bindings before reporting a completed analysis.
    _, _, _, current, current_coverage, after = load_verified(settings)
    if current != bindings or current_coverage != coverage or any(sha256(Path(__file__).with_name(n)) != h for n, h in code_hashes.items()):
        raise ValueError('Inputs or implementation changed during analysis')
    for name, table in [('paired_intervals', intervals), ('correctness_transitions', transitions),
                        ('per_class_changes', classes), ('participant_error_consistency', people)]:
        table.to_csv(out/(name+'.csv'), index=False)
    np.savez_compressed(out/'bootstrap_draws.npz', participant_counts=counts, deltas=draws,
                        participant_ids=training.participant_id.to_numpy(dtype=str), metrics=np.asarray(METRICS),
                        variant_names=np.asarray([variant_name(m, b) for m, b in VARIANTS]))
    plot_intervals(intervals, out/'paired_intervals.png')
    main = intervals[intervals.metric.eq('macro_f1')]
    sources = next(c['sources'] for c in verified['checks'] if c['check'] == 'source_and_input_bindings')
    lines = ['# Paired uncertainty of the raw-frame comparison', '',
             f"{len(training)} official training participants; {len(seeds)} fixed overlapping partitions; "
             f"{settings['bootstrap_repetitions']:,} paired participant bootstrap resamples. All {len(coverage)} dataset participants and {sources} sources remain accounted for by the verified upstream stage.", '',
             '| Input | Weights | Mean macro-F1 change | Pointwise 95% interval |', '|---|---|---:|---|']
    for row in main.itertuples():
        lines.append(f"| {row.modality} | {'balanced' if row.balanced else 'ordinary'} | {row.raw_minus_cached:+.4f} | [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |")
    lines += ['', 'All four primary intervals include zero; these data do not establish a consistent representation benefit.' if main.interval_includes_zero.all()
              else 'Some pointwise intervals exclude zero; this remains an exploratory, unadjusted comparison requiring independent confirmation.', '',
              'One participant resample is shared by both representations, all variants and all partitions. Each statistic averages per-partition metrics; participants are not multiplied by the number of folds or repeats. Five-class macro-F1 always includes all five classes. Balanced accuracy averages recall over classes present in each resample.', '',
              'Intervals condition on the existing fitted models and observed training participants. They do not capture model-refitting uncertainty, remove previous dataset exposure, or provide independent test evidence. Comparisons are not corrected for multiplicity. No new predictions, model fitting, validation/test scoring, label inference or model promotion occurred.', '',
              'See correctness_transitions.csv and per_class_changes.csv for paired error counts. Participant-level error consistency stays local. Bootstrap counts and all metric differences are saved in bootstrap_draws.npz for reproducibility. Original sources/models/results were preserved.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    artifacts = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
    result = {'complete': True, 'protocol_sha256': digest(protocol), 'training_participants': len(training),
              'partitions': len(seeds), 'paired_comparisons': len(VARIANTS), 'bootstrap_repetitions': len(counts),
              'all_macro_f1_intervals_include_zero': bool(main.interval_includes_zero.all()),
              'models_fitted': 0, 'new_model_predictions': 0, 'validation_test_scoring': False,
              'upstream_verification_passed_before_and_after': verified['passed'] and after['passed'], 'artifacts': artifacts}
    write_json(out/'completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/raw_frame_uncertainty.json')
    parser.add_argument('--output', default='results/raw_frame_uncertainty')
    args = parser.parse_args()
    result = run(json.loads(Path(args.config).read_text()), args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}, indent=2))


if __name__ == '__main__':
    main()
