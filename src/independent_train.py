"""Bounded exploratory LSTM experiments without cross-modal frame matching."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from .common import digest, joined_manifest, load_config, write_json
from .datasets import fit_scalers
from .evaluate import metrics, write_report
from .exploratory_baselines import load_audited_sequences
from .independent_datasets import independent_loader, independent_windows
from .models import IndependentAVClassifier
from .train import choose_device, plot_history, record_result, seed_all, validate_gender

PROTOCOL = 'exploratory-independent-modality-lstm-v1'


def forward(model, batch, device):
    streams = [{key: value.to(device) if key != 'lengths' else value for key, value in batch[modality].items()} for modality in ['audio', 'visual']]
    return model(*streams, batch['gender'].to(device))


def predictions(model, loader, device):
    model.eval()
    ids, probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            ids.extend(batch['ids'])
            probabilities.append(forward(model, batch, device).softmax(-1).cpu().numpy())
    return ids, np.concatenate(probabilities)


def build_model(sequences, config, gender):
    first = next(iter(sequences.values()))
    return IndependentAVClassifier(first['audio'].shape[1]+1, first['visual'].shape[1]+1, config['hidden_size'], config['dropout'], len(config['gender_categories'])+1 if gender else 0)


def evaluate_checkpoint(run, manifest, sequences, provenance):
    root = Path(run)
    # Only load locally generated, trusted checkpoints.
    state = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    if state['protocol'] != PROTOCOL or state['provenance'] != provenance:
        raise ValueError('Checkpoint protocol/data mismatch')
    config = state['config']
    device = choose_device(config['device'])
    model = build_model(sequences, config, state['gender']).to(device)
    model.load_state_dict(state['model'])
    validation = manifest[manifest.split == 'validation'].reset_index(drop=True)
    loader = independent_loader(validation, sequences, state['scalers'], config)
    ids, probabilities = predictions(model, loader, device)
    if ids != validation.participant_id.tolist():
        raise ValueError('Prediction order differs from participant manifest')
    report = write_report(validation, probabilities, root/'validation', state['model_name'], config['seed'], 'validation')
    report.update({'data_kind': 'EXPLORATORY_REAL_DATA', 'protocol': PROTOCOL, 'selected_epoch': state['epoch']+1, 'test_evaluated': False})
    write_json(root/'validation/metrics.json', report)
    record_result(root, report, config)
    return report


def train_independent(config, run, manifest, sequences, provenance, gender=False, balanced=False, resume=False):
    root = Path(run)
    if root.exists() and not resume:
        raise FileExistsError(f'Preserving existing run: {root}')
    if config.get('protocol') != PROTOCOL or not config.get('allow_unsegmented'):
        raise ValueError('Independent experiments require the explicit exploratory unsegmented protocol')
    if config['epochs'] < 1 or config['patience'] < 1:
        raise ValueError('Epoch and patience limits must be positive')
    seed_all(config['seed'])
    torch.set_num_threads(2)
    validate_gender(config, manifest, gender)
    training = manifest[manifest.split == 'train'].reset_index(drop=True)
    validation = manifest[manifest.split == 'validation'].reset_index(drop=True)
    if not len(training) or not len(validation) or training.severity_class.nunique() < 2:
        raise ValueError('Training requires multiple classes and validation participants')
    scalers = fit_scalers(sequences, training.participant_id.tolist())
    device = choose_device(config['device'])
    model = build_model(sequences, config, gender).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(config['seed'])
    counts = np.bincount(training.severity_class, minlength=5)
    weights = np.ones(5, np.float32)
    if balanced:
        observed = counts > 0
        weights[observed] = len(training) / (observed.sum()*counts[observed])
        weights[~observed] = 0
    class_weights = torch.tensor(weights, device=device)
    model_name = 'Independent-AV' + ('-gender' if gender else '') + ('-balanced' if balanced else '')
    epoch_start, best, stale, history = 0, -1.0, 0, []
    if resume:
        state = torch.load(root/'latest.pt', map_location='cpu', weights_only=False)
        if state['config'] != config or state['provenance'] != provenance or state['protocol'] != PROTOCOL or state['gender'] != gender or state['balanced'] != balanced:
            raise ValueError('Resume configuration/data mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        for opt_state in optimizer.state.values():
            for key, value in opt_state.items():
                if torch.is_tensor(value):
                    opt_state[key] = value.to(device)
        epoch_start, best, stale, history = state['epoch']+1, state['best_metric'], state['stale'], state['history']
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        generator.set_state(state['rng']['loader'])
        if device.type == 'cuda' and state['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
        if device.type == 'mps' and state['rng']['mps'] is not None:
            torch.mps.set_rng_state(state['rng']['mps'])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'config.json', config)
    train_loader = independent_loader(training, sequences, scalers, config, True, generator)
    validation_loader = independent_loader(validation, sequences, scalers, config)
    write_json(root/'training_provenance.json', {'fit_participant_ids': training.participant_id.tolist(), 'validation_participant_ids': validation.participant_id.tolist(), 'class_counts': counts.tolist(), 'class_weights': weights.tolist(), 'loss': 'One cross-entropy per participant after independent modality embedding aggregation', 'contributions': {row['participant_id']: {'audio_windows': len(windows['audio']), 'visual_windows': len(windows['visual']), 'base_participant_weight': 1/len(training)} for row, _, windows in train_loader.dataset}})
    for epoch in range(epoch_start, config['epochs']):
        if stale >= config['patience']:
            break
        started = time.monotonic()
        model.train()
        loss_total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = forward(model, batch, device)
            labels = batch['labels'].to(device)
            losses = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
            loss = (losses*class_weights[labels]).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            loss_total += float(loss.detach())*len(labels)
        ids, probabilities = predictions(model, validation_loader, device)
        if ids != validation.participant_id.tolist():
            raise ValueError('Validation ID order mismatch')
        report = metrics(validation.severity_class, probabilities)
        val_loss = float(-np.log(np.maximum(probabilities[np.arange(len(validation)), validation.severity_class.to_numpy()], 1e-12)).mean())
        improved = report['macro_f1'] > best
        best = max(best, report['macro_f1'])
        stale = 0 if improved else stale+1
        history.append({'epoch': epoch, 'train_loss': loss_total/len(training), 'validation_loss': val_loss, 'validation_macro_f1': report['macro_f1'], 'validation_accuracy': report['accuracy'], 'seconds': time.monotonic()-started})
        state = {'protocol': PROTOCOL, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config, 'gender': gender, 'balanced': balanced, 'model_name': model_name, 'epoch': epoch, 'best_metric': best, 'stale': stale, 'history': history, 'scalers': scalers, 'provenance': provenance, 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'loader': generator.get_state(), 'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None, 'mps': torch.mps.get_rng_state() if device.type == 'mps' else None}}
        temporary = root/'latest.pt.tmp'
        torch.save(state, temporary)
        temporary.replace(root/'latest.pt')
        if improved:
            temporary = root/'best.pt.tmp'
            torch.save(state, temporary)
            temporary.replace(root/'best.pt')
        pd.DataFrame(history).to_csv(root/'history.csv', index=False)
        print(f"{model_name} seed={config['seed']} epoch={epoch+1} val_macro_f1={report['macro_f1']:.4f} seconds={history[-1]['seconds']:.1f}", flush=True)
    plot_history(history, root)
    return evaluate_checkpoint(root, manifest, sequences, provenance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/independent_lstm.json')
    parser.add_argument('--output', default='checkpoints/independent_lstm')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 137, 2026])
    parser.add_argument('--pilot', action='store_true', help='Two-epoch AV-only compute/correctness pilot, separate output required')
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get('protocol') != PROTOCOL or config['timestamp_origins_verified']:
        raise ValueError('Use the explicit independent-modality exploratory configuration')
    if args.pilot:
        config = {**config, 'epochs': 2, 'experiment_log': str(Path(args.output)/'pilot_experiments.csv')}
    manifest = joined_manifest(config)
    sequences, sources = load_audited_sequences(config, manifest)
    provenance = {'protocol': PROTOCOL, 'manifest_hash': digest(manifest.to_dict('records')), 'features': sources}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [args.seeds[0]] if args.pilot else args.seeds
    variants = [(False, False)] if args.pilot else [(False, False), (False, True), (True, False), (True, True)]
    protocol = {'protocol': PROTOCOL, 'config': config, 'seeds': seeds, 'variants': variants, 'pilot': args.pilot, 'provenance': provenance, 'selection': 'Best participant validation macro-F1; no test evaluation', 'architecture': 'Separate LSTM window encoders; mean hidden embeddings independently within each participant/modality; concatenate; optional recorded gender; one five-logit head and participant-level CE', 'limitations': ['Cross-modal event alignment unverified and unused', 'Original cached overlap crop retained', 'Unsegmented audio; interviewer contamination not ruled out', '0.5-second averaging discards faster dynamics', 'Exploratory results; no faithful reproduction or clinical validity claim']}
    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError('Preserving existing matrix with a different protocol')
    write_json(output/'protocol.json', protocol)
    coverage = []
    for row in manifest.itertuples():
        counts = {key+'_windows': len(independent_windows(sequences[row.participant_id], key, config)) for key in ['audio', 'visual']}
        coverage.append({'participant_id': row.participant_id, 'split': row.split, **counts})
    pd.DataFrame(coverage).to_csv(output/'participant_coverage.csv', index=False)
    rows = []
    for seed in seeds:
        for gender, balanced in variants:
            current = {**config, 'seed': seed}
            name = 'Independent-AV' + ('-gender' if gender else '') + ('-balanced' if balanced else '')
            run = output/f'{name}_seed{seed}'
            report = train_independent(current, run, manifest, sequences, provenance, gender, balanced, resume=(run/'latest.pt').exists())
            rows.append({'run': str(run), **{key: report[key] for key in ['model', 'seed', 'accuracy', 'macro_f1', 'balanced_accuracy', 'participants', 'selected_epoch']}})
            pd.DataFrame(rows).to_csv(output/'validation_runs.csv', index=False)
    pd.DataFrame(rows).groupby('model')[['accuracy', 'macro_f1', 'balanced_accuracy']].agg(['mean', 'std']).to_csv(output/'validation_summary.csv')
    write_json(output/'completion.json', {'complete': True, 'runs': len(rows), 'participants': len(manifest), 'fit_participants': int(manifest.split.eq('train').sum()), 'validation_participants': int(manifest.split.eq('validation').sum()), 'test_participants_reserved': int(manifest.split.eq('test').sum()), 'test_evaluated': False})


if __name__ == '__main__':
    main()
