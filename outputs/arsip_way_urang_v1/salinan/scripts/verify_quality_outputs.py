"""Bandingkan setiap nilai workbook ekspor dengan tabel audit perantara."""
from pathlib import Path
import json
import math
import openpyxl
from hashlib import sha256

ROOT=Path(__file__).resolve().parents[1]
tables=json.loads((ROOT/'outputs/audit_kualitas_tables.json').read_text(encoding='utf-8'))
wb=openpyxl.load_workbook(ROOT/'outputs/audit_kualitas_data.xlsx',read_only=True,data_only=False)
checks=[]
for name, records in tables.items():
    ws=wb[name]; rows=list(ws.iter_rows(values_only=True))
    assert len(rows)-1==(len(records) if records else 1),(name,len(rows),len(records))
    if records:
        for actual,expected in zip(rows[1:],records):
            for key,value in zip(rows[0],actual):
                exp=expected.get(key)
                if hasattr(value,'isoformat'):
                    rendered=value.isoformat(sep=' ')[:19]
                    assert rendered==str(exp).replace('T',' ')[:19] or value.strftime('%Y-%m-%d')==exp,(name,key,value,exp)
                elif isinstance(value,(int,float)) and not isinstance(value,bool) and isinstance(exp,(int,float)):
                    assert math.isclose(value,exp,rel_tol=1e-12,abs_tol=1e-10),(name,key,value,exp)
                else:
                    assert value==exp,(name,key,value,exp)
    checks.append({'Sheet':name,'Baris':ws.max_row,'Kolom':ws.max_column})
wb.close()
nbpath=ROOT/'notebooks/02_audit_kualitas_data.ipynb'
nb=json.loads(nbpath.read_text(encoding='utf-8'))
assert all(c['execution_count'] is not None and not any(o['output_type']=='error' for o in c['outputs'])
           for c in nb['cells'] if c['cell_type']=='code')
assert len(list((ROOT/'outputs/figures').glob('*audit_so2.png')))==8
for r in tables['Sumber_SHA256']:
    actual=sha256((ROOT/r['File sumber']).read_bytes()).hexdigest()
    assert actual==r['SHA256 sebelum']==r['SHA256 sesudah']
for s in tables['Ringkasan']:
    assert s['SO2 teramati slot']+s['Tanpa baris slot']+s['Baris ada SO2 kosong slot']==s['Denominator slot grid']
    gaps=[g for g in tables['Gap_SO2'] if g['Stasiun']==s['Stasiun']]
    assert sum(g['Slot hilang'] for g in gaps)==s['SO2 hilang slot']
    assert all(g['Durasi menit']==30*g['Slot hilang'] for g in gaps)
(ROOT/'outputs/.audit_preview/verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(f'Lolos: seluruh nilai dan timestamp pada {len(checks)} sheet, notebook tanpa error, 8 grafik, rekonsiliasi gap/grid, serta SHA256 sumber.')
