"""Portable entry point for the unchanged Project_V2 retained checkpoint."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


@contextmanager
def package_directory():
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcript', type=Path)
    parser.add_argument('--audio-csv', type=Path)
    parser.add_argument('--visual-csv', type=Path)
    parser.add_argument('--features', type=Path,
                        help='Exact-schema NPZ; see FEATURE_SCHEMA.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ['transcript', 'audio_csv', 'visual_csv', 'features', 'output']:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    raw = [args.transcript, args.audio_csv, args.visual_csv]
    if (args.features is not None and any(raw)) or (args.features is None and not all(raw)):
        parser.error('Provide --features OR all three source CSVs')
    if args.output.exists():
        parser.error('Output already exists; choose a new filename')
    if not args.output.parent.is_dir():
        parser.error('Output parent folder must exist')

    from verify_release import verify
    verify(ROOT)
    with package_directory():
        from src.best_observed_inference import BUNDLE, load, predict, predict_features
        from src.checked_inference import validate_identity
        from src.common import digest
        contract, parameters = load(BUNDLE)
        if args.features is None:
            participant = validate_identity(*raw)
            result = predict(BUNDLE, args.output, *raw)
            result['participant_id'] = participant
        else:
            import numpy as np
            from threadpoolctl import threadpool_limits
            with np.load(args.features, allow_pickle=False) as archive:
                if set(archive.files) != {'semantic', 'av', 'language', 'contract_hash'}:
                    raise ValueError('Unexpected feature fields; targets and questionnaire fields are forbidden')
                if (archive['contract_hash'].shape != ()
                        or str(archive['contract_hash']) != digest(contract)):
                    raise ValueError('Features must declare the exact packaged feature contract')
                with threadpool_limits(limits=2):
                    probabilities = predict_features(parameters, archive['semantic'],
                                                     archive['av'], archive['language'])
            output = {'predicted_class': probabilities.argmax(1).tolist(),
                      'class_order': contract['classes'], 'class_names': contract['class_names'],
                      'decision_scores': probabilities.tolist(),
                      'source_model_sha256': contract['source_model']['sha256'],
                      'feature_contract_hash': digest(contract),
                      'score_interpretation': 'Uncalibrated research outputs',
                      'checkpoint_associated_cv_accuracy': 241 / 489,
                      'independent_test_accuracy': None}
            with args.output.open('x') as handle:
                handle.write(json.dumps(output, indent=2, allow_nan=False) + '\n')
            result = {'output': str(args.output), 'participants': len(probabilities)}
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
