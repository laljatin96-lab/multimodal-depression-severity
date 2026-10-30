import io
from pathlib import Path
import tarfile
import zipfile
import pytest
from src import transcript_corpus as corpus
from src.full_dataset import read_source, source_stamp


def report_for(tmp_path, kind, pid='101', words='I enjoy music and walking'):
    blob=f'Start_Time,End_Time,Text,Confidence\n0,1,"{words}",0.9\n'.encode()
    name=f'{pid}_Transcript.csv'
    path=tmp_path/f'{pid}_P'
    if kind=='directory':
        path.mkdir()
        (path/name).write_bytes(blob)
    elif kind=='zip':
        path=path.with_suffix('.zip')
        with zipfile.ZipFile(path,'w') as archive: archive.writestr(name,blob)
    else:
        path=Path(str(path)+'.tar.gz')
        with tarfile.open(path,'w:gz') as archive:
            info=tarfile.TarInfo(name); info.size=len(blob)
            archive.addfile(info,io.BytesIO(blob))
    source={'path':path,'participant_id':pid}
    _,members,sha=read_source(source)
    return {'source':str(path),'source_key':pid+kind,'participant_id':pid,'kind':'directory' if kind=='directory' else 'archive',
            'members':members,'integrity_checked':True,'source_stamp':source_stamp(path),'source_sha256':sha}


@pytest.mark.parametrize('kind',['directory','tar','zip'])
def test_local_reader_full_source_binding_and_cleaning(tmp_path,kind):
    report=report_for(tmp_path,kind)
    record=corpus.process(report)
    corpus.validate_record(record,report)
    assert record['text']=='i enjoy music and walking' and record['full_source_hash_verified']
    assert record['speaker_field_available'] is False
    report['source_sha256']='0'*64
    with pytest.raises(ValueError,match='hash'): corpus.read_transcript(report)


def test_explicit_target_rows_and_numbers_excluded():
    blob=b'Start_Time,End_Time,Text,Confidence\n0,1,"PHQ-8 score 23",.9\n1,2,"I walked 15 miles in 2025",.9\n'
    result=corpus.clean_transcript(blob)
    assert result['text']=='i walked miles in' and result['explicit_score_rows_removed']==1
    assert result['invalid_time_rows']==0


def test_unknown_schema_and_unsafe_members_fail(tmp_path):
    with pytest.raises(ValueError,match='schema'):
        corpus.clean_transcript(b'ID,Label,Text\n101,4,words\n')
    report=report_for(tmp_path,'directory')
    report['members'][0]['member']='../101_Transcript.csv'
    with pytest.raises(ValueError,match='Unsafe'): corpus.transcript_member(report)


def test_complete_coverage_duplicates_resume_and_integrity(tmp_path,monkeypatch):
    a=report_for(tmp_path,'tar')
    b=report_for(tmp_path,'directory')
    c=report_for(tmp_path,'zip','102','I prefer reading books')
    context={'coverage':[{'participant_id':'101','split':'train'},{'participant_id':'102','split':'test'}]}
    monkeypatch.setattr(corpus,'build_context',lambda *a:(context,[a_report,b,c]))
    a_report=a
    out=tmp_path/'corpus'
    first=corpus.prepare('unused',out,workers=1)
    assert first['participants']==2 and first['physical_sources']==3 and first['new_sources']==3
    docs,_=corpus.load_verified('unused',out)
    assert docs[0]['copies']==2 and docs[1]['copies']==1
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in (out/'sources').glob('*.json')}
    def forbidden(*args): raise AssertionError('Resume must reuse verified transcript records')
    monkeypatch.setattr(corpus,'read_transcript',forbidden)
    second=corpus.prepare('unused',out,resume=True)
    assert second['reused_sources']==3 and second['new_sources']==0
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before}
    path=next((out/'sources').glob('*.json'))
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='record changed'): corpus.load_verified('unused',out)
