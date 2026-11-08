"""Verifikasi independen ledger skor, kausalitas P1, sampel KNN dan workbook."""
from pathlib import Path
from datetime import datetime
import json,math
import numpy as np
import pandas as pd
import openpyxl
from audit_quality import hashfile

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'pilot_way_urang'
t=json.loads((OUT/'pilot_way_urang_tables.json').read_text(encoding='utf-8'))
cfg=json.loads((RUN/'config.json').read_text(encoding='utf-8'))
mask=pd.read_csv(RUN/'lokasi_masking.csv',parse_dates=['Timestamp'])
truth=pd.read_csv(RUN/'truth_penilaian_saja.csv')
pred=pd.read_csv(RUN/'prediksi_penilaian.csv')
grid=pd.read_csv(RUN/'grid_dan_penanda_asli.csv',parse_dates=['Timestamp'])
a=t['Pembagian_waktu'][0]['Jumlah slot'];b=a+t['Pembagian_waktu'][1]['Jumlah slot']
y=grid['SO2 numerik'].to_numpy(dtype=float)
assert not cfg['test_evaluated'] and len(grid)==sum(r['Jumlah slot'] for r in t['Pembagian_waktu'])
assert len(mask)==8205 and mask.Timestamp.max()<pd.Timestamp(t['Pembagian_waktu'][2]['Awal'])
assert len(mask.Seed.unique())==3 and set(mask.Seed)=={11,23,47}
assert not mask['Mask ID'].duplicated().any() and not pred[['Mask ID','Metode']].duplicated().any()
joined=pred.merge(mask,on=['Mask ID','Skenario'],validate='many_to_one').merge(truth,on='Mask ID',validate='many_to_one')
assert len(joined)==len(pred)
assert np.allclose(joined['SO2 asli untuk skor saja'],y[a+joined['Index validasi'].to_numpy()])
assert np.isfinite(joined['SO2 asli untuk skor saja']).all()
assert np.array_equal(joined['Berhasil finite'].to_numpy(),np.isfinite(joined.Prediksi))
for sid,m in mask.groupby('Skenario'):
    indices=m['Index validasi'].to_numpy(dtype=int)
    assert (indices>=0).all() and (indices<b-a).all()
    if m.Pola.iloc[0]=='titik':assert np.all(np.diff(np.sort(indices))>1)
    else:
        for _,event in m.groupby('Kejadian'):
            ids=np.sort(event['Index validasi'].to_numpy());assert len(ids)==m['Panjang slot'].iloc[0]
            assert np.all(np.diff(ids)==1)
    z=y[:b].copy();z[a+indices]=np.nan
    good=np.flatnonzero(np.isfinite(z))
    for name,g in joined[joined.Skenario==sid].groupby('Metode'):
        assert set(g['Mask ID'])==set(m['Mask ID'])
        if name=='P0':assert np.allclose(g.Prediksi,np.nanmedian(y[:a]))
        if name.startswith('P1_L'):
            limit=int(name.split('_L')[1]);loc=a+g['Index validasi'].to_numpy(dtype=int)
            donor=good[np.searchsorted(good,loc,side='left')-1]
            expected=np.where(loc-donor<=limit,z[donor],np.nan)
            assert np.allclose(g.Prediksi,expected,equal_nan=True),name
    all_ok=joined[joined.Skenario==sid].pivot(index='Mask ID',columns='Metode',values='Berhasil finite').all(axis=1)
    recorded=[r for r in t['Irisan_mask'] if r['Skenario']==sid][0]
    assert all_ok.sum()==recorded['Irisan semua metode']
for r in t['Hasil_per_seed']:
    g=joined[(joined.Seed==r['Seed'])&(joined.Pola==r['Pola'])&(joined['Panjang slot']==r['Panjang slot'])&(joined.Metode==r['Metode'])]
    keep=g['Berhasil finite'] if r['Domain']=='seluruh berhasil' else g['Irisan semua metode']
    err=g.loc[keep,'Prediksi']-g.loc[keep,'SO2 asli untuk skor saja']
    assert len(err)==r['Jumlah dinilai'] and math.isclose(100*len(err)/len(g),r['Cakupan persen'],abs_tol=1e-10)
    for name,expected in [('MAE',np.mean(np.abs(err))),('RMSE',np.sqrt(np.mean(err**2))),('Bias',np.mean(err))]:
        assert math.isclose(r[name],expected,rel_tol=1e-10,abs_tol=1e-10)
per_seed=pd.DataFrame(t['Hasil_per_seed'])
for r in t['Ringkasan']:
    g=per_seed[(per_seed.Metode==r['Metode'])&(per_seed.Pola==r['Pola'])&(per_seed['Panjang slot']==r['Panjang slot'])&(per_seed.Domain==r['Domain'])]
    for col in ['MAE','RMSE','Bias','Cakupan persen','Jumlah dinilai','Transform detik']:
        assert math.isclose(r[col+' rata-rata'],g[col].mean(),rel_tol=1e-10,abs_tol=1e-10)
        assert math.isclose(r[col+' SD antar-seed'],g[col].std(ddof=1),rel_tol=1e-10,abs_tol=1e-10)
# Skala/median adalah dari train, bukan validasi atau test.
features=cfg['common_covariates']
matrix=np.column_stack([y[:b]]+[grid[f'{c} numerik'].shift(1).iloc[:b].to_numpy(dtype=float) for c in features])
means=np.nanmean(matrix[:a],axis=0);scales=np.nanstd(matrix[:a],axis=0)
assert np.allclose(means,[r['Mean train'] for r in t['Scaler_train']])
assert np.allclose(scales,[r['Skala train'] for r in t['Scaler_train']])
scaled=(matrix-means)/scales;donors=scaled[:a][np.isfinite(y[:a])]
# Cek silang KNN manual pada beberapa query tanpa tie di batas k.
checked=0
for _,r in joined[joined.Metode=='P2_k10'].drop_duplicates('Index validasi').iloc[::101].iterrows():
    if not r['Berhasil finite']:continue
    q=scaled[a+int(r['Index validasi'])].copy();q[0]=np.nan
    overlap=np.isfinite(donors)&np.isfinite(q);n=overlap.sum(axis=1)
    d=np.sqrt(np.divide(np.nansum((donors-q)**2,axis=1)*len(q),n,out=np.full(len(n),np.inf),where=n>0))
    order=np.argsort(d);best=order[:10]
    if d[order[9]]==d[order[10]]:continue
    weights=(d[best]==0).astype(float) if np.any(d[best]==0) else 1/d[best]
    expected=np.sum(weights*donors[best,0])/weights.sum()*scales[0]+means[0]
    assert math.isclose(expected,r.Prediksi,rel_tol=1e-8,abs_tol=1e-8)
    checked+=1
assert checked>=5
for r in t['Integritas_SHA256']:assert hashfile(ROOT/r['File'])==r['SHA256 sebelum']==r['SHA256 sesudah']
# Seluruh nilai workbook disamakan dengan tabel eksperimen.
w=openpyxl.load_workbook(OUT/'pilot_imputasi_way_urang.xlsx',read_only=True,data_only=False)
for name,records in t.items():
    rows=list(w[name].iter_rows(values_only=True));assert len(rows)==len(records)+1,(name,len(rows),len(records))
    for actual,expected in zip(rows[1:],records):
        for key,v in zip(rows[0],actual):
            exp=expected.get(key)
            if exp=='' and v is None:continue
            if isinstance(v,datetime):assert v.isoformat(sep=' ')[:19]==str(exp).replace('T',' ')[:19]
            elif isinstance(v,(int,float)) and not isinstance(v,bool) and isinstance(exp,(int,float)):
                assert math.isclose(v,exp,rel_tol=1e-11,abs_tol=1e-10),(name,key,v,exp)
            else:assert v==exp,(name,key,v,exp)
w.close()
nb=json.loads((ROOT/'notebooks/04_pilot_imputasi_way_urang.ipynb').read_text(encoding='utf-8'))
assert all(c['execution_count'] is not None and not any(o['output_type']=='error' for o in c['outputs']) for c in nb['cells'] if c['cell_type']=='code')
print(f'Lolos: {len(t)} sheet, seluruh skor dihitung ulang, P1 kausal, {checked} sampel KNN manual, scaler train, mask validasi identik, test tidak dievaluasi, dan SHA256 sumber.')
