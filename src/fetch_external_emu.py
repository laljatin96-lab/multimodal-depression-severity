"""Reproduce the public author-source download; no repository code is executed."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION='552f3185e0671a8c382e4c92505b9f711a149b41'
FILES=['README.md','EMU_data/demographicsEMU.csv','EMU_data/text_features/transcripts_audio_open.csv',
       'EMU_data/text_features/textfeatures2021.csv','TlachacEMU.pdf',
       'EMU_data/text_features/TextFeatureEngineering.ipynb','audio_experiments/get_demographic.py','audio_experiments/get_phq9.py']


def download(url, maximum):
    request=urllib.request.Request(url,headers={'User-Agent':'Project-V2-research-data-import'})
    with urllib.request.urlopen(request,timeout=30) as response:
        if response.geturl().split('/')[2] not in {'api.github.com','raw.githubusercontent.com'}:
            raise ValueError('Unexpected download redirect host')
        content=response.read(maximum+1)
    if len(content)>maximum:raise ValueError('Download exceeds explicit byte budget')
    return content


def fetch(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    index=output/'source_tree.json'
    if index.exists():blob=index.read_bytes()
    else:
        blob=download(f'https://api.github.com/repos/mltlachac/EMU/git/trees/{REVISION}?recursive=1',4_000_000)
    tree=json.loads(blob)
    if tree['sha']!=REVISION or tree.get('truncated'):raise ValueError('Unpinned or incomplete source tree')
    if not index.exists():index.write_bytes(blob)
    receipts=[];downloaded=0
    for name in FILES:
        entry=next(x for x in tree['tree'] if x['path']==name)
        if entry['type']!='blob' or not 0<entry['size']<=2_000_000:raise ValueError('Unexpected source file')
        url=f'https://raw.githubusercontent.com/mltlachac/EMU/{REVISION}/{name}'
        path=output/name
        if path.exists():data=path.read_bytes()
        else:data=download(url,entry['size'])
        if len(data)!=entry['size'] or hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()!=entry['sha']:
            raise ValueError('Source hash mismatch; refusing to overwrite evidence')
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);downloaded+=1
        receipts.append({'url':url,'path':str(path),'bytes':len(data),'git_blob_sha1':entry['sha'],'sha256':hashlib.sha256(data).hexdigest()})
    manifest=output/'download_manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['revision']!=REVISION or old['files']!=receipts:raise ValueError('Existing manifest differs')
    else:
        from datetime import datetime,timezone
        manifest.write_text(json.dumps({'repository':'https://github.com/mltlachac/EMU','revision':REVISION,
            'retrieved_date':datetime.now(timezone.utc).date().isoformat(),'files':receipts},indent=2)+'\n')
    return {'verified_files':len(receipts),'new_downloads':downloaded,'manifest':str(manifest)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='data/external/emu_20261028');args=parser.parse_args()
    print(json.dumps(fetch(args.output),indent=2))
