"""Create a whitelisted code/model ZIP; never include participant data."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'Project_V2_Approx50_20261030'
EVIDENCE = ['summary.csv', 'matched_comparison.csv', 'language_paired_comparisons.csv',
            'language_error_changes.csv', 'per_class_precision_recall.csv',
            'delivery_confusion_matrices.json', 'repeat_scores.csv']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build(release_name=NAME):
    if not release_name.replace('_', '').isalnum():
        raise ValueError('Release names may contain only letters, numbers and underscores')
    release = ROOT / 'releases' / release_name
    archive = release.with_suffix('.zip')
    if release.exists() or archive.exists():
        raise FileExistsError('Preserving completed releases; use a new version')
    release.mkdir(parents=True)
    copied = {}

    def copy(source, target=None):
        source = ROOT / source
        target = release / (target or source.relative_to(ROOT))
        if source.is_symlink() or not source.is_file():
            raise ValueError('Only explicit regular source files may be packaged')
        target.parent.mkdir(parents=True, exist_ok=True)
        original = sha(source)
        shutil.copy2(source, target)
        if sha(target) != original or sha(source) != original:
            raise ValueError('Source or copied bytes changed')
        copied[str(source.relative_to(ROOT))] = original

    for directory, glob in [('src', '*.py'), ('configs', '*.json'), ('tests', '*.py')]:
        for path in sorted((ROOT / directory).glob(glob)):
            copy(path.relative_to(ROOT))
    for path in sorted(ROOT.glob('requirements*.txt')):
        copy(path.relative_to(ROOT), Path('historical_requirements') / path.name)
    for path in sorted((ROOT / 'checkpoints/best_observed_4928').iterdir()):
        copy(path.relative_to(ROOT))
    for path in sorted((ROOT / 'checkpoints/pretrained_minilm').iterdir()):
        copy(path.relative_to(ROOT))
    for name in EVIDENCE:
        copy(Path('results/wavlm_accuracy') / name, Path('evidence') / name)
    for name in ['predict.py', 'verify_release.py', 'README.md']:
        copy(Path('release_support') / name, Path(name))
    copy(Path(__file__).relative_to(ROOT), Path('packaging_source/package_accuracy_release.py'))
    apache = '.venv/lib/python3.12/site-packages/packaging-26.3.dist-info/licenses/LICENSE.APACHE'
    copy(apache, Path('licenses/Apache-2.0.txt'))
    write(release / 'ACCURACY.json', {
        'best_observed_selection_procedure': {
            'accuracy': 244 / 489, 'correct': 244, 'repeated_predictions': 489,
            'unique_participants': 163, 'repeats': 3,
            'rounded_whole_percent': 50, 'at_least_50_percent': False,
            'macro_f1': 0.2570610369251744, 'balanced_accuracy': 0.26876576582458933},
        'retained_checkpoint_associated_procedure': {
            'accuracy': 241 / 489, 'correct': 241, 'repeated_predictions': 489,
            'source_checkpoint_sha256': '599f1ec55d82a4a61f199aace525a23dd7f13f0e6afa77625086a4c4bc20b660'},
        'majority_class_accuracy': 231 / 489,
        'gain_over_retained': {'percentage_points': 0.613496932515345,
            'participant_paired_ci95_percentage_points': [-0.8179959100204526, 2.0449897750511314],
            'repeated_predictions_gained': 8, 'repeated_predictions_lost': 5,
            'statistically_supported_accuracy_gain': False},
        'severe_recall': 0, 'independent_test_accuracy': None,
        'new_training_performed': False, 'new_accuracy_claim': False,
        'questionnaire_inference_inputs': False,
        'scope': 'Exploratory, extensively reused original training cohort; independent confirmation unavailable'})
    contract = json.loads((release / 'checkpoints/best_observed_4928/model.json').read_text())
    import sys
    sys.path.insert(0, str(ROOT))
    from src.common import digest
    write(release / 'FEATURE_SCHEMA.json', {
        'npz_fields': {'semantic': ['N', 384], 'av': ['N', 128], 'language': ['N', 16],
                       'contract_hash': 'scalar Unicode array'},
        'contract_hash': digest(contract['contract']),
        'semantic': 'L2-normalized frozen MiniLM complete-document embedding',
        'av_feature_names': contract['contract']['original_feature_contract']['av_feature_names'],
        'language_names': contract['contract']['language_names'],
        'feature_generation': 'The packaged original source CSV extraction; no substitute features',
        'forbidden': ['PHQ scores', 'questionnaire answers', 'severity labels', 'target-derived fields']})
    packages = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'threadpoolctl', 'torch',
                'matplotlib', 'transformers', 'huggingface-hub', 'tokenizers', 'safetensors']
    versions = {name: importlib.metadata.version(name) for name in packages}
    (release / 'requirements-inference.txt').write_text(
        '# Direct inference dependencies from the tested Python3.12 environment\n'
        + ''.join(f'{name}=={version}\n' for name, version in versions.items()))
    write(release / 'environment-tested.json', {'python': sys.version, 'packages': versions,
        'fresh_install_tested': False, 'platform': sys.platform})
    (release / 'THIRD_PARTY_NOTICES.md').write_text(
        '# Bundled public encoder\n\n'
        'sentence-transformers/all-MiniLM-L6-v2, revision '
        '`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.\n\n'
        'Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/\n\n'
        'The unchanged provider model card declares Apache-2.0. The license text is '
        'in `licenses/Apache-2.0.txt`; the original model card and byte hashes are '
        'bundled beside the weights. No encoder fine-tuning occurred.\n\n'
        'Original project source is supplied to the project owner without assigning '
        'a new license. Python dependencies are installed separately under their '
        'respective licenses. No licensed participant dataset is redistributed.\n')
    files = {str(path.relative_to(release)): {'sha256': sha(path), 'bytes': path.stat().st_size}
             for path in sorted(release.rglob('*')) if path.is_file()}
    write(release / 'RELEASE_MANIFEST.json', {'name': release_name, 'format': 'project-v2-code-release-v1',
        'files': files, 'participant_data_included': False, 'numeric_model_only_no_pickle': True,
        'pretrained_minilm_included': True, 'wavlm_encoder_included': False})
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(release.rglob('*')):
            if path.is_file():
                info = zipfile.ZipInfo(str(Path(release_name) / path.relative_to(release)),
                                       date_time=(2026, 10, 30, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                output.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                                compresslevel=6)
    if any(sha(ROOT / path) != expected for path, expected in copied.items()):
        raise ValueError('Original source preservation failed')
    receipt = {'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
               'source_files_preserved': len(copied), 'packaged_files': len(files) + 1,
               'new_fits': 0, 'new_accuracy_claim': False}
    write(archive.with_suffix('.build.json'), receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default=NAME)
    args = parser.parse_args()
    print(json.dumps(build(args.name), indent=2))
