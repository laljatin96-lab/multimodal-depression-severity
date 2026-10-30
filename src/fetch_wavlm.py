"""Fetch only pinned public model data; no private data is transmitted."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path('checkpoints/pretrained_wavlm_base_plus')
REVISION='4c66d4806a428f2e922ccfa1a962776e232d487b'
FILES={'README.md':(3904,'1192671605727eff94113d3b599161892688b9d5'),
    'config.json':(2232,'b7b4e5f6c410f71283a59e26250e40855ca99310'),
    'preprocessor_config.json':(215,'10f6def8c83d70a2b087a567dcf523b75152a80b'),
    'pytorch_model.bin':(377617425,'3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51')}


def fetch():
    ROOT.mkdir(parents=True,exist_ok=True);records={}
    for name,(size,expected) in FILES.items():
        path=ROOT/name;url=f'https://huggingface.co/microsoft/wavlm-base-plus/resolve/{REVISION}/{name}'
        if not path.exists():
            tmp=path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(url,timeout=90) as response,tmp.open('wb') as handle:
                count=0;last=0
                while chunk:=response.read(1024**2):
                    count+=len(chunk)
                    if count>size:raise ValueError('Oversized model file')
                    handle.write(chunk)
                    if count-last>32*1024**2:
                        print(f'{name}: {count}/{size} bytes',flush=True);last=count
            if tmp.stat().st_size!=size:raise ValueError('Incomplete model file')
            tmp.rename(path)
        blob=path.read_bytes()
        actual=hashlib.sha256(blob).hexdigest() if name.endswith('.bin') else hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
        if len(blob)!=size or actual!=expected:raise ValueError('Pinned publisher file checksum differs')
        records[name]={'url':url,'bytes':size,'sha256':hashlib.sha256(blob).hexdigest(),'publisher_object':expected}
        print(f'Verified {name}',flush=True)
    license_path=ROOT/'LICENSE.txt'
    license_url='https://raw.githubusercontent.com/microsoft/UniSpeech/main/LICENSE'
    if not license_path.exists():
        with urllib.request.urlopen(license_url,timeout=30) as response:blob=response.read(100000)
        if b'Attribution-ShareAlike 3.0 Unported' not in blob:raise ValueError('Unexpected publisher-linked license')
        license_path.write_bytes(blob)
    records['LICENSE.txt']={'url':license_url,'sha256':hashlib.sha256(license_path.read_bytes()).hexdigest(),'bytes':license_path.stat().st_size}
    notice=ROOT/'MODEL_NOTICE.md'
    if not notice.exists():notice.write_text('Microsoft WavLM Base Plus, original weights unchanged.\nModel: https://huggingface.co/microsoft/wavlm-base-plus\nRevision: '+REVISION+'\nThe model card links to the included Attribution-ShareAlike 3.0 Unported license. Preserve attribution and applicable terms on redistribution. This project uses local frozen feature extraction, not a depression-pretrained checkpoint. No participant recordings or text were uploaded.\n')
    records['MODEL_NOTICE.md']={'sha256':hashlib.sha256(notice.read_bytes()).hexdigest(),'bytes':notice.stat().st_size}
    manifest={'model':'microsoft/wavlm-base-plus','revision':REVISION,'files':records,'private_data_uploaded':False,'remote_code_executed':False}
    path=ROOT/'download_manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest:raise ValueError('Download manifest differs')
    else:path.write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':fetch()
