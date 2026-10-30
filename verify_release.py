"""Check packaged files without importing the project or loading participant data."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text())
    for name, entry in manifest['files'].items():
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or '..' in relative.parts or path.is_symlink():
            raise ValueError(f'Unsafe packaged path: {name}')
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing packaged file: {name}')
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if path.stat().st_size != entry['bytes'] or actual != entry['sha256']:
            raise ValueError(f'Packaged file changed: {name}')
    return {'passed': True, 'files_checked': len(manifest['files']),
            'new_fits': 0, 'new_accuracy_claim': False}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
