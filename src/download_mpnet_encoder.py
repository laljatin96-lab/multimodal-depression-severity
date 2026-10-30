"""Download public sentence-encoder assets only; never reads interview data."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from .common import write_json

MODEL = 'sentence-transformers/all-mpnet-base-v2'
FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'vocab.txt', 'model.safetensors', 'sentence_bert_config.json', 'README.md']


def download(output):
    root = Path(output)
    receipt_path = root/'download_receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for name, value in receipt['files'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != value['sha256']:
                raise ValueError('Existing encoder asset changed')
        return receipt
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL}?blobs=true', timeout=60) as response:
        metadata = json.load(response)
    revision = metadata['sha']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Unexpected model revision')
    members = {m['rfilename']: m for m in metadata['siblings']}
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in FILES:
        member = members[name]
        path = root/name
        temporary = path.with_suffix(path.suffix+'.tmp')
        with urllib.request.urlopen(f'https://huggingface.co/{MODEL}/resolve/{revision}/{name}', timeout=60) as response, temporary.open('wb') as handle:
            total = 0
            while block := response.read(1024**2):
                total += len(block)
                if total > 500*1024**2:
                    raise ValueError('Unexpectedly large encoder asset')
                handle.write(block)
        value = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if total != member['size'] or ('lfs' in member and value != member['lfs']['sha256']):
            raise ValueError('Public model asset checksum/size mismatch')
        temporary.replace(path)
        files[name] = {'sha256': value, 'bytes': total}
        print('Downloaded public asset:', name, total, flush=True)
    receipt = {'model': MODEL, 'revision': revision, 'files': files, 'dataset_access': False,
               'source': f'https://huggingface.co/{MODEL}/tree/{revision}'}
    write_json(receipt_path, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='checkpoints/pretrained_mpnet')
    args = parser.parse_args()
    result = download(args.output)
    print(json.dumps({'model': result['model'], 'revision': result['revision'], 'files': len(result['files'])}))
