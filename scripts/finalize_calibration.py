"""Finalize report and validate exported workbook, without fitting or imputation."""
from types import SimpleNamespace
from pathlib import Path
from hashlib import sha256
from datetime import datetime
import json,math,sys
import openpyxl
from summarize_calibration import write_report
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'sensitivitas_kalibrasi'
t=json.loads((OUT/'tabel_sensitivitas_kalibrasi.json').read_text(encoding='utf-8'));cfg=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'))
checks=json.loads((RUN/'checks.json').read_text(encoding='utf-8'));audit=json.loads((RUN/'independent_checks.json').read_text(encoding='utf-8'))
assert sha256((OUT/'tabel_sensitivitas_kalibrasi.json').read_bytes()).hexdigest()==audit['tables_sha256']
e=SimpleNamespace(cfg=cfg,config_hash=checks['config_sha256'],metrics=t['Metrik_per_fold'],pooled=t['Ringkasan'],variation=t['Variasi_antar_fold'],
    deltas=t['Vs_persistence'],ranks=t['Peringkat'],legacy=t['Audit90_vs_tahap6'],composition=t['Tambahan_train_vs90'],input_checks=t['Kesamaan_input_evaluasi'],integrity=t['Integritas_SHA256'],cases=[])
for flow in t['Alur_sampel']:
    select=lambda rows:next(r for r in rows if r['Fold']==flow['Fold'] and r['Kalibrasi hari']==flow['Kalibrasi hari'])
    e.cases.append(SimpleNamespace(flow=flow,dist=select(t['Train_distribusi']),timing=select(t['Waktu_komputasi'])))
write_report(e)
nbpath=ROOT/'notebooks/08_sensitivitas_kalibrasi_ridge.ipynb';nb=json.loads(nbpath.read_text(encoding='utf-8'))
for output in [c for c in nb['cells'] if c['cell_type']=='code'][-1]['outputs']:
    if output.get('output_type')=='display_data' and 'text/markdown' in output.get('data',{}):output['data']['text/markdown']=(OUT/'ringkasan_sensitivitas_kalibrasi.md').read_text(encoding='utf-8')
assert all(c['execution_count'] is not None and not any(o['output_type']=='error' for o in c['outputs']) for c in nb['cells'] if c['cell_type']=='code')
nbpath.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
if '--report-only' in sys.argv:
    print('Laporan diperbarui tanpa fit, perubahan data hasil, atau ekspor ulang.');sys.exit(0)
w=openpyxl.load_workbook(OUT/'hasil_sensitivitas_kalibrasi.xlsx',read_only=True,data_only=False)
for name,records in t.items():
    rows=list(w[name].iter_rows(values_only=True));assert len(rows)==len(records)+1
    for row,expected in zip(rows[1:],records):
        for key,value in zip(rows[0],row):
            exp=expected.get(key)
            if exp=='' and value is None:continue
            if isinstance(value,datetime):assert value.isoformat(sep=' ')[:19]==str(exp).replace('T',' ')[:19]
            elif isinstance(value,(int,float)) and not isinstance(value,bool) and isinstance(exp,(int,float)):assert math.isclose(value,exp,rel_tol=1e-10,abs_tol=1e-8)
            else:assert value==exp,(name,key,value,exp)
w.close()
for row in t['Integritas_SHA256']:assert sha256((ROOT/row['File']).read_bytes()).hexdigest()==row['SHA256 sebelum']==row['SHA256 sesudah'],row['File']
from PIL import Image,ImageDraw
preview=OUT/'.audit_preview_kalibrasi';files=[p for p in preview.glob('*.png') if p.name!='contact_all.png']
canvas=Image.new('RGB',(1980,250*((len(files)+2)//3)),'#dddddd');draw=ImageDraw.Draw(canvas)
for i,p in enumerate(files):
    im=Image.open(p).convert('RGB');im.thumbnail((650,220));x=(i%3)*660;y=(i//3)*250;canvas.paste(im,(x,y+25));draw.text((x+5,y+5),p.stem,fill='black')
canvas.save(preview/'contact_all.png')
print(f'Lolos:{len(t)}sheet semua sel cocok, notebook selesai, {len(t["Integritas_SHA256"])}artefak lama identik;12Ridge/102prefix telah diverifikasi terpisah, tanpa deep learning/test.')
