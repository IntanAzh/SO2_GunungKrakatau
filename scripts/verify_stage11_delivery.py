"""Final read-only verification of stage 11 deliverables and old artifacts."""
from pathlib import Path
import hashlib,json,math
import openpyxl

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/perbandingan_algoritma_tahap11'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def verify():
    cfg=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'))
    tables=json.loads((RUN/'tabel_hasil.json').read_text(encoding='utf-8'))
    independent=json.loads((RUN/'independent_checks.json').read_text(encoding='utf-8'))
    assert independent['new_models_verified_without_fit']==24
    assert independent['old_files_sha256_identical']==len(cfg['initial_hashes'])
    for relative,h in cfg['initial_hashes'].items():assert sha(ROOT/relative)==h,relative
    assert sha(ROOT/'outputs/lstm_antarstasiun_tahap10/config_locked.json')==cfg['stage10_config_sha256']
    assert sha(ROOT/'outputs/lstm_antarstasiun_tahap10/manifest_tahap10.json')==cfg['stage10_manifest_sha256']
    notebook=ROOT/'notebooks/11_perbandingan_algoritma.ipynb'
    cells=json.loads(notebook.read_text(encoding='utf-8'))['cells']
    code=[c for c in cells if c['cell_type']=='code']
    assert len(code)==5 and all(c.get('execution_count') is not None for c in code)
    assert not any(o.get('output_type')=='error' for c in code for o in c.get('outputs',[]))
    wb=openpyxl.load_workbook(RUN/'hasil_perbandingan_algoritma.xlsx',read_only=True,data_only=True)
    assert wb.sheetnames==list(tables)
    cells_verified=0
    for name,rows in tables.items():
        ws=wb[name];assert ws.max_row is None or ws.max_row==len(rows)+1
        keys=list(dict.fromkeys(k for row in rows for k in row)) if rows else []
        iterator=ws.iter_rows(values_only=True)
        if keys:assert list(next(iterator)[:len(keys)])==keys
        counted=0
        for i,(row,values) in enumerate(zip(rows,iterator),2):
            counted+=1
            for j,key in enumerate(keys,1):
                expected=row.get(key);actual=values[j-1]
                if expected is None:assert actual is None,(name,i,key,actual)
                elif isinstance(expected,bool):assert actual is expected,(name,i,key,actual)
                elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
                    assert isinstance(actual,(int,float)) and math.isclose(expected,actual,rel_tol=1e-11,abs_tol=1e-10),(name,i,key,expected,actual)
                else:assert str(actual)==str(expected),(name,i,key,expected,actual)
                cells_verified+=1
        assert counted==len(rows) and next(iterator,None) is None,(name,counted,len(rows))
    wb.close()
    models=list(RUN.glob('Fold_*/GRU_seed*/complete.json'))+list(RUN.glob('Fold_*/TCN_seed*/complete.json'))
    assert len(models)==24
    required=['config_locked.json','daftar_fitur.json','tabel_hasil.json','checks.json','independent_checks.json','hasil_perbandingan_algoritma.xlsx','ringkasan_perbandingan_algoritma.md','prediksi_oof_seluruh_seed.csv','timestamp_semua_bagian.csv']
    assert all((RUN/r).exists() for r in required)
    assert (ROOT/'outputs/STATUS_PROYEK.md').read_text(encoding='utf-8').count('## Tahap 11')==1
    paths=[p for p in RUN.rglob('*') if p.is_file() and 'previews' not in p.parts and p.name!='manifest_tahap11.json']
    paths+= [notebook,ROOT/'outputs/STATUS_PROYEK.md',ROOT/'scripts/algorithms_stage11.py',ROOT/'scripts/summarize_algorithms_stage11.py',ROOT/'scripts/report_algorithms_stage11.py',ROOT/'scripts/verify_algorithms_stage11.py',ROOT/'scripts/verify_stage11_delivery.py',ROOT/'scripts/build_stage11_notebook.py']
    manifest={'purpose':'Stage 11 delivery integrity','created_after_all_models':True,'old_files_verified':len(cfg['initial_hashes']),'workbook_cells_verified':cells_verified,
      'files':[{'path':p.relative_to(ROOT).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(paths))]}
    (RUN/'manifest_tahap11.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in manifest['files']:
        p=ROOT/row['path'];assert p.stat().st_size==row['size_bytes'] and sha(p)==row['sha256']
    print(json.dumps({'files_verified':len(manifest['files']),'workbook_sheets':len(wb.sheetnames),'workbook_cells_verified':cells_verified,'old_files_identical':len(cfg['initial_hashes']),'new_models':len(models),'test_evaluated':False},indent=2))

if __name__=='__main__':verify()
