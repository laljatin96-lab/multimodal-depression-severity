"""Download public TabPFN-v2 assets only; never access participant data."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from .common import write_json

MODEL = 'Prior-Labs/TabPFN-v2-clf'
WEIGHTS = 'tabpfn-v2-classifier-v2_default.ckpt'
FILES = [WEIGHTS, 'README.md']


def download(output):
    root=Path(output); receipt_path=root/'download_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['model']!=MODEL or set(receipt['files'])!=set(FILES):raise ValueError('Unexpected model receipt')
        for name,expected in receipt['files'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected['sha256']:raise ValueError('Existing public model asset changed')
        return receipt
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL}?blobs=true',timeout=60) as response:
        metadata=json.load(response)
    revision=metadata['sha']
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):raise ValueError('Invalid model revision')
    members={m['rfilename']:m for m in metadata['siblings']}
    root.mkdir(parents=True,exist_ok=True);files={}
    for name in FILES:
        item=members[name];path=root/name;temporary=path.with_suffix(path.suffix+'.tmp')
        if name==WEIGHTS and 'lfs' not in item:raise ValueError('Provider weight digest required')
        with urllib.request.urlopen(f'https://huggingface.co/{MODEL}/resolve/{revision}/{name}',timeout=60) as response,temporary.open('wb') as handle:
            total=0
            while block:=response.read(1024**2):
                total+=len(block)
                if total>500*1024**2:raise ValueError('Unexpected model asset size')
                handle.write(block)
        actual=hashlib.sha256(temporary.read_bytes()).hexdigest()
        if total!=item['size'] or ('lfs' in item and actual!=item['lfs']['sha256']):raise ValueError('Provider asset hash/size mismatch')
        temporary.replace(path);files[name]={'sha256':actual,'bytes':total}
        print(f'Downloaded public model asset: {name}, {total} bytes',flush=True)
    receipt={'model':MODEL,'revision':revision,'files':files,'dataset_access':False,
             'source':f'https://huggingface.co/{MODEL}/tree/{revision}',
             'attribution':'TabPFN by Prior Labs; Hollmann et al., Nature 2025, DOI 10.1038/s41586-024-08328-6',
             'license_url':'https://priorlabs.ai/license'}
    write_json(receipt_path,receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='checkpoints/pretrained_tabpfn_v2')
    r=download(p.parse_args().output)
    print(json.dumps({'model':r['model'],'revision':r['revision'],'files':len(r['files'])}))
