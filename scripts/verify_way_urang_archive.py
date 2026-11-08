"""Independent byte verification of archive v1; accepts valid LF/CRLF checksum line."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json,csv,zipfile,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';FOLDER=OUT/'arsip_way_urang_v1';ZIP=OUT/'arsip_way_urang_v1.zip'
RECEIPT=OUT/'arsip_way_urang_v1.verifikasi.json'

def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    manifest_bytes=(FOLDER/'manifest.json').read_bytes();mh=sha256(manifest_bytes).hexdigest();m=json.loads(manifest_bytes)
    control=(FOLDER/'manifest.sha256').read_bytes();assert control.decode('ascii').splitlines()==[f'{mh}  manifest.json']
    entries=m['entries'];expected={r['archive_path'] for r in entries}|{'manifest.json','manifest.sha256'}
    assert len({r['archive_path'] for r in entries})==len(entries)
    assert {p.relative_to(FOLDER).as_posix() for p in FOLDER.rglob('*') if p.is_file()}==expected
    copied=0;total_bytes=0
    for r in entries:
        rel=Path(r['archive_path']);assert not rel.is_absolute() and '..' not in rel.parts
        p=FOLDER/rel;assert p.stat().st_size==r['size_bytes'] and digest(p)==r['sha256'],r['archive_path']
        total_bytes+=p.stat().st_size
        if r['origin']=='copy_existing':
            original=ROOT/r['project_source_path'];assert original.stat().st_size==r['size_bytes'] and digest(original)==r['sha256'],str(original)
            copied+=1
    external=m['external_source_references']
    for r in external:
        p=ROOT/r['project_path'];assert p.stat().st_size==r['size_bytes'] and digest(p)==r['sha256']
        assert not any(x['project_source_path']==r['project_path'] for x in entries)
    excluded_files=0
    with (FOLDER/'berkas_tidak_disalin.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['Tipe']=='file':
                p=ROOT/row['Lokasi'];assert p.stat().st_size==int(row['Ukuran byte']) and digest(p)==row['SHA256'];excluded_files+=1
    prefix=FOLDER.name+'/'
    with zipfile.ZipFile(ZIP) as z:
        names=z.namelist();assert len(names)==len(set(names)) and not any(n.endswith('/') for n in names)
        assert set(names)=={prefix+s for s in expected}
        assert z.read(prefix+'manifest.json')==manifest_bytes and z.read(prefix+'manifest.sha256')==control
        for i,r in enumerate(entries,1):
            name=prefix+r['archive_path'];info=z.getinfo(name);assert info.file_size==r['size_bytes'];h=sha256()
            with z.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            assert h.hexdigest()==r['sha256'],name
            if i%100==0:print('ZIP cocok ukuran/SHA256:',i,flush=True)
        assert z.testzip() is None
    inv=json.loads((FOLDER/'inventaris_tahap.json').read_text(encoding='utf-8'))
    checkpoint_n=sum(x['Tersedia'] for x in inv['Checkpoint LSTM']);history_n=sum(x['Tersedia'] for x in inv['Riwayat permodel'])
    assert checkpoint_n==sum(r['archive_path'].endswith('.pt') for r in entries)
    result={'archive_path':str(FOLDER),'zip_path':str(ZIP),'zip_size_bytes':ZIP.stat().st_size,'zip_sha256':digest(ZIP),
        'manifest_sha256':mh,'copied_existing_files':copied,'generated_metadata_entries':sum(r['origin']=='archive_metadata' for r in entries),
        'manifest_entries':len(entries),'control_files_verified':2,'total_files_folder_and_zip':len(expected),'payload_bytes_uncompressed':total_bytes,
        'folder_exact_file_set':True,'zip_exact_file_set':True,'payload_sizes_sha256_all_match':True,'zip_crc_pass':True,
        'manifest_controls_match':'Manifest bytes hash matches checksum line; LF/CRLF accepted for parsing only; ZIP control bytes equal folder exactly.',
        'original_copied_files_unchanged':True,'external_source_files_unchanged':True,'external_source_count':len(external),
        'source_excel_copied':False,'excluded_internal_log_files_hashes_unchanged':excluded_files,
        'checkpoint_files_existing_copied':checkpoint_n,'training_histories_existing_copied':history_n,
        'notebook_count':sum(r['archive_path'].endswith('.ipynb') for r in entries),'stage_required_missing':inv['Berkas wajib hilang'],
        'notebook03_available':any('/notebooks/03_' in r['archive_path'] for r in entries),
        'new_model_training':False,'new_experiment_results':False,'test_evaluation':False,
        'verified_utc':datetime.now(timezone.utc).isoformat(),'verifier_script':str(Path(__file__).resolve()),'verifier_sha256':digest(Path(__file__))}
    if RECEIPT.exists():
        prior=json.loads(RECEIPT.read_text(encoding='utf-8'));assert prior['zip_sha256']==result['zip_sha256'];print('Receipt lama dipertahankan.')
    else:RECEIPT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
