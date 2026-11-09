"""Read-only workbook verification and stage10 manifest, after notebook export."""
from pathlib import Path
from datetime import datetime
import json,math
import openpyxl
from audit_quality import hashfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'lstm_antarstasiun_tahap10'
def verify():
    tables=json.loads((RUN/'tabel_hasil.json').read_text(encoding='utf-8'))
    numeric=json.loads((RUN/'independent_checks.json').read_text(encoding='utf-8'))
    assert numeric['tables_sha256']==hashfile(RUN/'tabel_hasil.json') and numeric['checkpoint_models_verified_without_refit']==24
    wbpath=RUN/'hasil_lstm_antarstasiun.xlsx';w=openpyxl.load_workbook(wbpath,read_only=True,data_only=True);assert w.sheetnames==list(tables)
    cells=0
    for name,records in tables.items():
        keys=list(dict.fromkeys(k for r in records for k in r));rows=w[name].iter_rows(values_only=True);headers=next(rows)
        assert list(headers)==keys,(name,headers,keys)
        for row,r in zip(rows,records,strict=True):
            for k,a in zip(keys,row):
                b=r.get(k);cells+=1
                if b is None:assert a is None,(name,k,a,b)
                elif isinstance(a,datetime):assert a.strftime('%Y-%m-%d %H:%M:%S')==str(b)[:19],(name,k,a,b)
                elif isinstance(b,(int,float)) and not isinstance(b,bool):assert isinstance(a,(int,float)) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(name,k,a,b)
                else:assert a==b or (a is None and b==''),(name,k,a,b)
    w.close()
    nbpath=ROOT/'notebooks/10_lstm_antarstasiun.ipynb';nb=json.loads(nbpath.read_text(encoding='utf-8'));code=[c for c in nb['cells'] if c['cell_type']=='code']
    assert all(c['execution_count'] is not None for c in code) and not any(o['output_type']=='error' for c in code for o in c['outputs'])
    for row in tables['Integritas_SHA256']:assert hashfile(ROOT/row['File'])==row['SHA256 sebelum'],row['File']
    result={'xlsx_sheets_verified':len(tables),'xlsx_cells_verified':cells,'executed_notebook_cells':len(code),'completed_models':24,
        'old_files_identical':len(tables['Integritas_SHA256']),'test_loaded':False,'test_evaluated':False,'xlsx_sha256':hashfile(wbpath),'model_training_in_verifier':False}
    (RUN/'delivery_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    files=[p for p in RUN.rglob('*') if p.is_file() and p.name!='manifest_tahap10.json' and 'previews' not in p.relative_to(RUN).parts]
    files += [nbpath,OUT/'STATUS_PROYEK.md']+list((ROOT/'scripts').glob('*stage10*.py'))
    manifest=[{'File':p.relative_to(ROOT).as_posix(),'Bytes':p.stat().st_size,'SHA256':hashfile(p)} for p in sorted(set(files))]
    (RUN/'manifest_tahap10.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for r in manifest:assert hashfile(ROOT/r['File'])==r['SHA256'] and (ROOT/r['File']).stat().st_size==r['Bytes']
    print(json.dumps({**result,'manifest_files_verified':len(manifest)},indent=2))
if __name__=='__main__':verify()
