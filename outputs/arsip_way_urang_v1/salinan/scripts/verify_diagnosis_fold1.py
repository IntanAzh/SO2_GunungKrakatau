"""Verify diagnosis outputs without any estimator fit or test-value access."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json,math
import numpy as np
import pandas as pd
import openpyxl
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
t=json.loads((OUT/'tabel_diagnosis_fold1.json').read_text(encoding='utf-8'))
assert all(r['Lulus'] for r in t['Pemeriksaan'])
assert all(r['Ketidaksesuaian nilai']==0 for r in t['Cocok_sumber_variabel'])
p=pd.read_csv(OUT/'diagnosis_fold1/prediksi_dan_origin.csv',parse_dates=['Origin','Timestamp target'])
old=pd.read_csv(OUT/'backtesting_way_urang/Fold_01/prediksi_validasi.csv',parse_dates=['Origin','Timestamp target'])
for col in old.columns:
    if pd.api.types.is_numeric_dtype(old[col]):assert np.allclose(p[col],old[col],equal_nan=True)
    else:assert p[col].equals(old[col])
assert p['Timestamp target'].max()==pd.Timestamp('2025-08-12 13:30:00')
methods=['P0_median','P1_ffill_L4','P2_knn_k10','Persistence'];common=p['Irisan semua metode'].to_numpy()
for row in t['Skor']:
    within=np.ones(len(p),bool) if row['Periode']=='seluruh fold' else p['Timestamp target'].dt.to_period('M').astype(str).eq(row['Periode']).to_numpy()
    valid=common if row['Domain']=='bersama' else np.isfinite(p[row['Pipeline']]).to_numpy()
    mask=within&valid;e=(p.loc[mask,row['Pipeline']]-p.loc[mask,'SO2 target asli']).to_numpy()
    assert len(e)==row['N'] and int(within.sum())==row['Denominator target asli']
    assert np.allclose([np.abs(e).mean(),np.sqrt(np.mean(e**2)),e.mean()],[row['MAE'],row['RMSE'],row['Bias']])
    assert math.isclose(row['Cakupan persen'],100*len(e)/within.sum())
    assert sha256('\n'.join(p.loc[mask,'Timestamp target'].astype(str)).encode()).hexdigest()==row['SHA256 timestamp']
groups=['SO2 historis','Polutan lain','Meteorologi','Indikator SO2 missing','Indikator kovariat missing']
dec=pd.DataFrame(t['Kontribusi_per_prediksi']);dates=pd.to_datetime(dec['Timestamp target'])
for method in methods[:3]:
    g=dec[dec.Pipeline==method];s=p[p['Timestamp target'].isin(pd.to_datetime(g['Timestamp target']))]
    assert np.allclose(g.Prediksi,g.Intercept+g[groups].sum(axis=1))
    assert np.allclose(g.Prediksi,s[method])
    assert np.allclose(g.Residual,g.Prediksi-g['SO2 target asli'])
    cor=pd.DataFrame([r for r in t['Koefisien_scaler'] if r['Pipeline']==method]);assert len(cor)==34
    assert cor['Dikenai penalti'].all() and cor['Alpha L2'].eq(1).all()
for r in t['Integritas_SHA256']:
    assert sha256((ROOT/r['File']).read_bytes()).hexdigest()==r['SHA256 sebelum']==r['SHA256 sesudah'],r['File']
w=openpyxl.load_workbook(OUT/'diagnosis_fold1.xlsx',read_only=True,data_only=False)
for name,records in t.items():
    rows=list(w[name].iter_rows(values_only=True));assert len(rows)==len(records)+1,(name,len(rows),len(records))
    for actual,expected in zip(rows[1:],records):
        for key,value in zip(rows[0],actual):
            exp=expected.get(key)
            if exp=='' and value is None:continue
            if isinstance(value,datetime):assert value.isoformat(sep=' ')[:19]==str(exp).replace('T',' ')[:19]
            elif isinstance(value,(int,float)) and not isinstance(value,bool) and isinstance(exp,(int,float)):assert math.isclose(value,exp,rel_tol=1e-11,abs_tol=1e-8)
            else:assert value==exp,(name,key,value,exp)
w.close()
from PIL import Image,ImageDraw
files=[f for f in (OUT/'.audit_preview_diagnosis_fold1').glob('*.png') if f.name!='contact_all.png']
canvas=Image.new('RGB',(1980,250*((len(files)+2)//3)),'#dddddd');draw=ImageDraw.Draw(canvas)
for i,file in enumerate(files):
    im=Image.open(file).convert('RGB');im.thumbnail((650,220));x=(i%3)*660;y=(i//3)*250;canvas.paste(im,(x,y+25));draw.text((x+5,y+5),file.stem,fill='black')
canvas.save(OUT/'.audit_preview_diagnosis_fold1/contact_all.png')
print(f'Lolos: {len(t["Pemeriksaan"])} checks, {len(t)} sheet seluruh sel cocok, skor dan dekomposisi cocok, {len(t["Integritas_SHA256"])} hash sumber/artefak lama identik; tanpa fit, tanpa test.')
