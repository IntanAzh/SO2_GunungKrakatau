from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json,math
import openpyxl
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'lstm_preprocessing'
t=json.loads((OUT/'tabel_lstm_preprocessing.json').read_text(encoding='utf-8'))
audit=json.loads((RUN/'independent_model_checks.json').read_text(encoding='utf-8'))
assert sha256((OUT/'tabel_lstm_preprocessing.json').read_bytes()).hexdigest()==audit['tables_sha256']
w=openpyxl.load_workbook(OUT/'hasil_lstm_preprocessing.xlsx',read_only=True,data_only=False)
for name,records in t.items():
    rows=list(w[name].iter_rows(values_only=True));assert len(rows)==len(records)+1,(name,len(rows),len(records))
    for actual,expected in zip(rows[1:],records):
        for key,value in zip(rows[0],actual):
            exp=expected.get(key)
            if exp=='' and value is None:continue
            if isinstance(value,datetime):assert value.isoformat(sep=' ')[:19]==str(exp).replace('T',' ')[:19]
            elif isinstance(value,(int,float)) and not isinstance(value,bool) and isinstance(exp,(int,float)):assert math.isclose(value,exp,rel_tol=1e-10,abs_tol=1e-8)
            else:assert value==exp,(name,key,value,exp)
w.close()
for row in t['Integritas_SHA256']:assert sha256((ROOT/row['File']).read_bytes()).hexdigest()==row['SHA256 sesudah']
resume=json.loads((RUN/'resume_initial_hashes.json').read_text(encoding='utf-8'))
for path,digest in resume.items():assert sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
nb=json.loads((ROOT/'notebooks/07_lstm_preprocessing.ipynb').read_text(encoding='utf-8'))
assert all(c['execution_count'] is not None and not any(o['output_type']=='error' for o in c['outputs']) for c in nb['cells'] if c['cell_type']=='code')
print(f'Lolos: {len(t)}worksheet seluruh sel cocok, notebook selesai, {len(resume)}artefak sebelumresume identik; {audit["completed_checkpoints_verified"]}checkpoint dan skor diverifikasi tanpa fit/test.')
