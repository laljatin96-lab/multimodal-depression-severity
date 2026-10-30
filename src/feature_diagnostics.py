"""Label-free cache geometry and raw-to-binned acoustic variance diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import digest, write_json
from .exploratory_baselines import load_audited_sequences, pool_modalities
from .final_evaluation import sha256, verify_plan
from .full_dataset import source_stamp
from .prepare_features import AUDIO_COLUMNS, VISUAL_COLUMNS, bin_frames, temporal_frame

FAMILIES = ['MFCC', 'delta', 'delta_delta']
SUMMARY_NAMES = ([f'audio_mean:{c}' for c in AUDIO_COLUMNS] + [f'audio_std:{c}' for c in AUDIO_COLUMNS]
                 + [f'visual_mean:{c}' for c in VISUAL_COLUMNS] + [f'visual_std:{c}' for c in VISUAL_COLUMNS])


def cache_statistics(sequence):
    audio = np.asarray(sequence['audio'], dtype=np.float64)
    mask = sequence['audio_mask']
    if audio.ndim != 2 or audio.shape[1] != 39 or mask.shape != (len(audio),) or mask.dtype != bool or not mask.any():
        raise ValueError('Expected 39 audio features and nonempty boolean observation mask')
    if not np.isfinite(audio).all():
        raise ValueError('Nonfinite cached audio')
    observed = audio[mask]
    adjacent = mask[:-1] & mask[1:]
    pairs = int(adjacent.sum())
    rows = []
    for i, name in enumerate(AUDIO_COLUMNS):
        x, y = audio[:-1, i][adjacent], audio[1:, i][adjacent]
        correlation = float(np.corrcoef(x, y)[0, 1]) if pairs > 1 and x.std() > 0 and y.std() > 0 else None
        rows.append({'feature': name, 'family': FAMILIES[i//13], 'mean': float(observed[:, i].mean()),
                     'std': float(observed[:, i].std()), 'exact_zero_fraction': float((observed[:, i] == 0).mean()),
                     'constant_over_observed_bins': bool(np.ptp(observed[:, i]) == 0),
                     'adjacent_bin_correlation': correlation})
    quality = {'bins': len(audio), 'audio_observed_bins': int(mask.sum()), 'audio_observed_fraction': float(mask.mean()),
               'visual_observed_fraction': float(sequence['visual_mask'].mean()), 'adjacent_observed_pairs': pairs,
               'identical_adjacent_audio_vector_fraction': float(np.all(audio[:-1][adjacent] == audio[1:][adjacent], axis=1).mean()) if pairs else None,
               'constant_audio_columns': sum(row['constant_over_observed_bins'] for row in rows)}
    return quality, rows


def training_geometry(matrix, coverage):
    """Reference statistics use official training rows only; no target columns."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (len(coverage), 128) or not np.isfinite(matrix).all():
        raise ValueError('Expected finite 128-feature summaries')
    training = matrix[coverage.split.eq('train').to_numpy()]
    if len(training) < 2:
        raise ValueError('At least two training participants required')
    results, pairs = [], []
    for modality, indices in [('audio', np.arange(78)), ('visual', np.arange(78, 128)), ('combined', np.arange(128))]:
        x = training[:, indices]
        scale = x.std(0)
        keep = scale > 0
        z = (x[:, keep]-x[:, keep].mean(0))/scale[keep]
        singular = np.linalg.svd(z, compute_uv=False)
        energy = singular**2
        positive = energy.sum() > 0
        cumulative = np.cumsum(energy)/energy.sum() if positive else np.array([])
        tolerance = max(z.shape)*np.finfo(float).eps*singular[0] if len(singular) else 0
        correlation = z.T@z/len(z)
        names = [SUMMARY_NAMES[i] for i in indices[keep]]
        for a, b in zip(*np.where(np.triu(np.abs(correlation) >= .95, k=1))):
            pairs.append({'modality': modality, 'first_feature': names[a], 'second_feature': names[b],
                          'training_correlation': float(correlation[a, b])})
        results.append({'modality': modality, 'training_participants': len(training), 'features': len(indices),
                        'constant_training_columns': int((~keep).sum()), 'numerical_rank': int((singular > tolerance).sum()),
                        'components_for_95_percent_variance': int(np.searchsorted(cumulative, .95)+1) if positive else 0,
                        'components_for_99_percent_variance': int(np.searchsorted(cumulative, .99)+1) if positive else 0,
                        'absolute_correlation_at_least_095_pairs': sum(p['modality'] == modality for p in pairs)})
    return results, pairs


def variance_retention(audio_blob, sequence, bin_seconds, time_offset=0):
    """Weighted between-bin variance / raw variance over the identical crop.

    Counts weight each bin, so irregular sampling cannot change the population
    being compared. This measures averaging loss, not clinical information.
    """
    frame, times = temporal_frame(audio_blob, 'frameTime', ';')
    times += time_offset
    values = frame[AUDIO_COLUMNS].to_numpy(float)
    edges = np.r_[sequence['timestamps'], sequence['timestamps'][-1]+bin_seconds]
    cached, observed = bin_frames(times, values, np.ones(len(times), bool), edges)
    if not np.array_equal(cached, sequence['audio']) or not np.array_equal(observed, sequence['audio_mask']):
        raise ValueError('Raw audio does not exactly reproduce the audited cache')
    indices = np.searchsorted(edges, times, side='right')-1
    use = (indices >= 0) & (indices < len(cached))
    selected, indices = values[use], indices[use]
    counts = np.bincount(indices, minlength=len(cached))
    sums = np.zeros_like(cached, dtype=np.float64)
    np.add.at(sums, indices, selected)
    means = sums[counts > 0]/counts[counts > 0, None]
    raw_mean, raw_variance = selected.mean(0), selected.var(0)
    between = ((means-raw_mean)**2 * counts[counts > 0, None]).sum(0)/len(selected)
    defined = raw_variance > 0
    retained = np.divide(between, raw_variance, out=np.zeros_like(between), where=defined)
    if (retained[defined] > 1+1e-10).any():
        raise ValueError('Variance decomposition exceeds raw variance')
    rows = [{'feature': name, 'family': FAMILIES[i//13], 'raw_variance': float(raw_variance[i]),
             'weighted_between_bin_variance': float(between[i]),
             'retained_variance_fraction': float(np.clip(retained[i], 0, 1)) if defined[i] else None,
             'raw_constant': bool(not defined[i])} for i, name in enumerate(AUDIO_COLUMNS)]
    return rows, {'raw_frames_total': len(values), 'raw_frames_in_cached_interval': len(selected),
                  'observed_bins': int(observed.sum()), 'raw_median_step_seconds': float(np.median(np.diff(times))),
                  'cached_audio_reproduced_exactly': True, 'raw_variance_definition': 'Population variance within the exact cached time interval',
                  'retained_variance_definition': 'Between-bin variance of float64 bin means weighted by raw frame counts / raw frame variance'}


def extracted_comparisons(audit, sequences, config):
    table = pd.read_csv(Path(audit)/'sources.csv', dtype=str, keep_default_na=False)
    rows, checks, seen = [], [], set()
    for source in table[table.kind.eq('directory')].itertuples():
        pid = source.participant_id
        if pid in seen:
            continue
        report = json.loads((Path(audit)/'sources'/f'{source.source_key}.json').read_text())
        name = f'{pid}_OpenSMILE2.3.0_mfcc.csv'
        paths = list(Path(source.source).rglob(name))
        if len(paths) != 1 or paths[0].is_symlink() or not paths[0].resolve().is_relative_to(Path(source.source).resolve()):
            raise ValueError('Expected one safe extracted MFCC file')
        blob = paths[0].read_bytes()
        if hashlib.sha256(blob).hexdigest() != report['feature_sha256'][name]:
            raise ValueError('Extracted MFCC checksum differs from the original audit')
        values, check = variance_retention(blob, sequences[pid], config['bin_seconds'], config['audio_time_offset'])
        rows.extend({'participant_id': pid, **row} for row in values)
        checks.append({'participant_id': pid, 'source': source.source, 'feature_sha256': report['feature_sha256'][name], **check})
        seen.add(pid)
    return rows, checks


def plot_retention(table, output):
    if table.empty:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grouped = table.groupby(['participant_id', 'family']).retained_variance_fraction.median().unstack().reindex(columns=FAMILIES)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    width = .8/len(grouped)
    for i, (pid, values) in enumerate(grouped.iterrows()):
        positions = np.arange(3)-.4+width*(i+.5)
        bars = ax.bar(positions, values.to_numpy()*100, width=width, label=f'Extracted interview {i+1}')
        ax.bar_label(bars, labels=[f'{v*100:.3f}%' if pd.notna(v) else 'undefined' for v in values], fontsize=9, padding=3)
    ax.set_xticks(np.arange(3), ['MFCC', 'Delta', 'Delta-delta'])
    ax.set_ylabel('Median retained temporal variance (%)')
    ax.set_ylim(0, 110)
    ax.set_title(f'0.5-second averaging: {len(grouped)} extracted interviews only\nFrame-weighted variance over identical time intervals')
    ax.legend(frameon=False)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run_diagnostic(plan_path, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError('Preserving existing feature diagnostic; choose a new output directory')
    plan = verify_plan(plan_path)
    config = plan['feature_config']
    if out.resolve().is_relative_to(Path(config['dataset']).resolve()):
        raise ValueError('Output must be outside the dataset')
    coverage = pd.read_csv(config['splits'], usecols=['participant_id', 'split'], dtype=str, keep_default_na=False)
    if coverage.participant_id.duplicated().any() or set(coverage.participant_id) != set(plan['source_provenance']):
        raise ValueError('Split coverage differs from frozen participants')
    if set(coverage.split) != {'train', 'validation', 'test'}:
        raise ValueError('Expected complete official split annotations')
    sequences, provenance = load_audited_sequences(config, coverage)
    if provenance != plan['source_provenance']:
        raise ValueError('Feature caches differ from the frozen plan')
    audit = Path(config.get('full_dataset_report', 'results/full_dataset'))
    sources = pd.read_csv(audit/'sources.csv', dtype=str, keep_default_na=False)
    audit_hash = sha256(audit/'sources.csv')
    quality, features, matrix = [], [], []
    for row in coverage.itertuples():
        profile, details = cache_statistics(sequences[row.participant_id])
        quality.append({'participant_id': row.participant_id, 'split': row.split, **profile})
        features.extend({'participant_id': row.participant_id, 'split': row.split, **item} for item in details)
        matrix.append(pool_modalities(sequences[row.participant_id]))
    with threadpool_limits(limits=2):
        geometry, pairs = training_geometry(matrix, coverage)
    retention, raw_checks = extracted_comparisons(audit, sequences, config)
    # Verify before writing a completion claim; original artifacts are read only.
    verify_plan(plan_path)
    if sha256(audit/'sources.csv') != audit_hash or any(sha256(item['cache']) != item['npz_sha256'] for item in provenance.values()):
        raise ValueError('Audit/cache input changed during diagnostics')
    for row in sources.itertuples():
        report = json.loads((audit/'sources'/f'{row.source_key}.json').read_text())
        if source_stamp(Path(report['source'])) != report['source_stamp']:
            raise ValueError('Physical source changed during diagnostics')
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(quality).to_csv(out/'participant_quality.csv', index=False)
    pd.DataFrame(features).to_csv(out/'cached_audio_statistics.csv', index=False)
    pd.DataFrame(geometry).to_csv(out/'training_summary_geometry.csv', index=False)
    pd.DataFrame(pairs, columns=['modality', 'first_feature', 'second_feature', 'training_correlation']).to_csv(out/'training_correlated_features.csv', index=False)
    retention_table = pd.DataFrame(retention, columns=['participant_id', 'feature', 'family', 'raw_variance',
                   'weighted_between_bin_variance', 'retained_variance_fraction', 'raw_constant'])
    retention_table.to_csv(out/'raw_to_cache_variance.csv', index=False)
    write_json(out/'raw_source_checks.json', raw_checks)
    coverage.to_csv(out/'participant_coverage.csv', index=False)
    completion = {'complete': True, 'sources_accounted_for': len(sources),
                  'archives_accounted_for': int(sources.kind.eq('archive').sum()),
                  'folders_accounted_for': int(sources.kind.eq('directory').sum()), 'unique_participants_profiled': len(coverage),
                  'training_participants_for_geometry': int(coverage.split.eq('train').sum()),
                  'raw_comparison_participants': len(raw_checks), 'raw_cache_matches': all(c['cached_audio_reproduced_exactly'] for c in raw_checks),
                  'questionnaire_values_used': False, 'gender_values_used': False, 'predictive_models_fitted': 0,
                  'severity_predictions_created': False, 'performance_metrics_computed': False, 'archives_decompressed': 0,
                  'frozen_plan_sha256': digest(plan), 'source_table_sha256': audit_hash,
                  'implementation_sha256': sha256(__file__)}
    plot_retention(retention_table, out/'variance_retention.png')
    lines = ['# Acoustic feature representation diagnostic', '',
             f"Profiled all {len(coverage)} participants using verified caches; all {completion['archives_accounted_for']} archives and {completion['folders_accounted_for']} folders remain accounted for. "
             'No questionnaire/gender values, predictions, model fitting or performance scoring were used.', '',
             '## Training-summary geometry', '',
             '| Modality | Columns | Constant columns | Numerical rank | Components for 95% variance | Highly correlated pairs |',
             '|---|---:|---:|---:|---:|---:|']
    for row in geometry:
        lines.append(f"| {row['modality']} | {row['features']} | {row['constant_training_columns']} | {row['numerical_rank']} | {row['components_for_95_percent_variance']} | {row['absolute_correlation_at_least_095_pairs']} |")
    lines += ['', 'Geometry uses standardized summaries from official training participants only. High correlation means absolute Pearson correlation >=0.95. '
              'These are descriptive redundancy checks, not feature selection or performance claims.', '', '## Raw-to-cache spot checks', '',
              f'{len(raw_checks)} unique interviews had already-extracted MFCC CSVs. Their raw file hashes match the original audit, '
              'and recomputed audio bins match the frozen caches exactly. Other archives were not decompressed again. '
              'Median retained variance below is across 13 coefficients per family within each checked interview.', '',
              '| Extracted interview | MFCC | Delta | Delta-delta |', '|---|---:|---:|---:|']
    if len(raw_checks):
        table = retention_table.groupby(['participant_id', 'family']).retained_variance_fraction.median().unstack().reindex(columns=FAMILIES)
        for i, (_, values) in enumerate(table.iterrows(), 1):
            lines.append('| '+str(i)+' | '+' | '.join(f'{v*100:.3f}%' if pd.notna(v) else 'undefined' for v in values)+' |')
    lines += ['', 'Retained variance = raw-frame-count-weighted between-bin variance divided by raw variance over the identical cached interval. '
              'Constant raw coefficients have undefined ratios and are marked explicitly. This quantifies temporal variation removed by averaging; '
              'it does not measure depression-related information or prove the cause of weak model performance.', '',
              'All-participant cache statistics include validation/test inputs as descriptive data-quality checks only. Training geometry uses training rows exclusively. '
              'No participant is excluded or cache rewritten. Zero values and fully observed bins do not establish silence, speech isolation or signal validity. '
              'The available raw-feature spot checks are a convenience sample and cannot establish dataset-wide raw-variance loss. '
              'Speech identity and event synchrony remain unverified. The original final test is already used; new model experiments require explicit reuse disclosure or independent data.', '',
              'See participant_quality.csv, cached_audio_statistics.csv, training_summary_geometry.csv, training_correlated_features.csv, '
              'raw_to_cache_variance.csv and raw_source_checks.json. All participant-level artifacts remain local.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    write_json(out/'completion.json', completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='results/final_evaluation_plan.json')
    parser.add_argument('--output', default='results/feature_diagnostics')
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.plan, args.output), indent=2))


if __name__ == '__main__':
    main()
