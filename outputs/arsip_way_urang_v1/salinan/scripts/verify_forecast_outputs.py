"""Verifikasi kausalitas histori, fitur, purge, prediksi Ridge dan seluruh skor."""
from pathlib import Path
from datetime import datetime
import json,math
import numpy as np
import pandas as pd
import openpyxl
from audit_quality import hashfile

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'forecasting_way_urang'
t=json.loads((OUT/'forecasting_way_urang_tables.json').read_text(encoding='utf-8'))
cfg=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'))
checks=json.loads((RUN/'checks.json').read_text(encoding='utf-8'))
assert hashfile(RUN/'config_locked.json')==checks['config_sha256']
assert not cfg['test_loaded'] and not cfg['test_evaluated']
a=cfg['split'][0]['Jumlah slot'];b=a+cfg['split'][1]['Jumlah slot']
d=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=b,parse_dates=['Timestamp'])
times=pd.DatetimeIndex(d.Timestamp);y=d['SO2 numerik'].to_numpy(dtype=float);n=b-1
m=np.column_stack([y]+[d[f'{c} numerik'].shift(1).to_numpy(dtype=float) for c in cfg['covariates']])[:n]
names=cfg['pipelines'];hist={};X={}
for name in names:
    h=pd.read_csv(RUN/f'histori_kausal_{name}.csv')
    hist[name]=h.iloc[:,1:13].to_numpy(dtype=float)
    assert hist[name].shape==m.shape
    known=np.isfinite(m);assert np.allclose(hist[name][known],m[known],rtol=1e-12,atol=1e-10)
    f=pd.read_csv(RUN/f'fitur_{name}.csv');X[name]=f.iloc[:,1:].to_numpy(dtype=float)
    expected=np.column_stack([pd.Series(hist[name][:,0]).shift(lag) for lag in cfg['so2_lags']]+[hist[name][:,j] for j in range(1,12)])
    raw=np.column_stack([pd.Series(m[:,0]).shift(lag) for lag in cfg['so2_lags']]+[m[:,j] for j in range(1,12)])
    assert np.allclose(X[name][:,:18],expected,equal_nan=True,rtol=1e-11,atol=1e-10)
    assert np.array_equal(X[name][:,18:],~np.isfinite(raw))
last=np.full(12,-1,int);values=np.full(12,np.nan)
for u,row in enumerate(m):
    known=np.isfinite(row);last[known]=u;values[known]=row[known]
    if u<cfg['warmup_slots']:continue
    eligible=(~known)&(last>=0)&(u-last<=4)
    expected=row.copy();expected[eligible]=values[eligible]
    assert np.allclose(hist['P1_ffill_L4'][u],expected,equal_nan=True,rtol=1e-11,atol=1e-10)
manual_knn=0
for log in t['Prefix_imputer_kausal']:
    start=log['Jumlah baris prefix'];stop=times.get_loc(pd.Timestamp(log['Akhir query']))+1
    assert start<=a and pd.Timestamp(log['Timestamp donor maksimum'])<pd.Timestamp(log['Awal query'])
    assert pd.Timestamp(log['Akhir query'])<pd.Timestamp(cfg['split'][2]['Awal'])
    prefix=m[:start];med=np.nanmedian(prefix,axis=0)
    expected=np.where(np.isfinite(m[start:stop]),m[start:stop],med)
    assert np.allclose(hist['P0_median'][start:stop],expected,equal_nan=True,rtol=1e-11,atol=1e-10)
    mean=np.nanmean(prefix,axis=0);scale=np.nanstd(prefix,axis=0);scale=np.where(scale==0,1,scale)
    donors=(prefix[np.isfinite(prefix[:,0])]-mean)/scale
    candidate=np.flatnonzero((~np.isfinite(m[start:stop])).any(axis=1)&(np.isfinite(m[start:stop,1:]).sum(axis=1)>=2))
    for local in candidate[:10]:
        q=(m[start+local]-mean)/scale;overlap=np.isfinite(q)&np.isfinite(donors);count=overlap.sum(axis=1)
        dist=np.sqrt(np.divide(np.nansum((donors-q)**2,axis=1)*len(q),count,out=np.full(len(count),np.inf),where=count>0))
        col=int(np.flatnonzero(~np.isfinite(q))[0]);valid=np.flatnonzero(np.isfinite(donors[:,col])&np.isfinite(dist))
        if len(valid)<11:continue
        order=valid[np.argsort(dist[valid])];near=order[:10]
        if dist[order[9]]==dist[order[10]]:continue
        weights=(dist[near]==0).astype(float) if np.any(dist[near]==0) else 1/dist[near]
        val=np.sum(weights*donors[near,col])/weights.sum()*scale[col]+mean[col]
        assert math.isclose(val,hist['P2_knn_k10'][start+local,col],rel_tol=1e-7,abs_tol=1e-7)
        manual_knn+=1;break
assert manual_knn>=3
train=pd.read_csv(RUN/'timestamp_train_bersama.csv',parse_dates=['Origin','Target train'])
ids=train['Index origin'].to_numpy(dtype=int)
assert (ids+1<a).all() and ids.min()>=cfg['warmup_slots']+max(cfg['so2_lags'])
assert np.isfinite(y[ids+1]).all()
eligible=np.arange(cfg['warmup_slots']+max(cfg['so2_lags']),a-1)
common=np.isfinite(y[eligible+1])
for matrix in X.values():common &= np.isfinite(matrix[eligible]).all(axis=1)
assert np.array_equal(ids,eligible[common])
pred=pd.read_csv(RUN/'prediksi_validasi.csv',parse_dates=['Origin','Timestamp target'])
origin=pred['Index origin'].to_numpy(dtype=int);target=pred['Index target'].to_numpy(dtype=int)
assert np.array_equal(origin+1,target)
assert np.array_equal(target,np.flatnonzero(np.isfinite(y[a:b]))+a)
assert (pred['Timestamp target']-pred.Origin==pd.Timedelta(minutes=30)).all()
assert pred['Timestamp target'].max()<pd.Timestamp(cfg['split'][2]['Awal'])
assert np.allclose(pred['SO2 target asli'],y[target])
for name in names:
    coefficients=[r for r in t['Koefisien_Ridge'] if r['Pipeline']==name]
    mean=np.array([r['Mean fitur train'] for r in coefficients]);scale=np.array([r['Skala fitur train'] for r in coefficients])
    assert np.allclose(mean,np.mean(X[name][ids],axis=0),rtol=1e-11,atol=1e-10)
    expected_scale=np.std(X[name][ids],axis=0);expected_scale[expected_scale==0]=1
    assert np.allclose(scale,expected_scale,rtol=1e-10,atol=1e-10)
    ok=np.isfinite(X[name][origin]).all(axis=1);expected=np.full(len(origin),np.nan)
    beta=np.array([r['Koefisien fitur standar'] for r in coefficients])
    ztrain=(X[name][ids]-mean)/scale
    target_train=y[ids+1]
    analytical=np.linalg.solve(ztrain.T@ztrain+cfg['ridge']['alpha']*np.eye(ztrain.shape[1]),ztrain.T@(target_train-target_train.mean()))
    assert np.allclose(beta,analytical,rtol=1e-7,atol=1e-8)
    assert math.isclose(coefficients[0]['Intercept'],target_train.mean(),rel_tol=1e-10,abs_tol=1e-10)
    expected[ok]=((X[name][origin[ok]]-mean)/scale)@beta+coefficients[0]['Intercept']
    assert np.allclose(pred[name],expected,equal_nan=True,rtol=1e-10,atol=1e-10)
last=np.nan;raw_last=[]
for val in y[:n]:
    if np.isfinite(val):last=val
    raw_last.append(last)
assert np.allclose(pred.Persistence,np.array(raw_last)[origin])
common=np.isfinite(pred[names+['Persistence']]).all(axis=1)
assert np.array_equal(common,pred['Irisan semua metode'])
def verify_score(row,within):
    good=np.isfinite(pred[row['Metode']]) if row['Domain']=='seluruh berhasil' else common
    keep=good&within;err=pred.loc[keep,row['Metode']]-pred.loc[keep,'SO2 target asli']
    assert len(err)==row['Jumlah prediksi']
    assert math.isclose(100*len(err)/int(within.sum()),row['Cakupan persen'],abs_tol=1e-10)
    for key,value in [('MAE',np.mean(np.abs(err))),('RMSE',np.sqrt(np.mean(err**2))),('Bias',np.mean(err))]:
        assert math.isclose(row[key],value,rel_tol=1e-10,abs_tol=1e-10)
for row in t['Ringkasan']:verify_score(row,np.ones(len(pred),bool))
for row in t['Performa_bulanan']:verify_score(row,pred['Timestamp target'].dt.to_period('M').astype(str).eq(row['Bulan target']).to_numpy())
for r in t['Integritas_SHA256']:assert hashfile(ROOT/r['File'])==r['SHA256 sebelum']==r['SHA256 sesudah']
w=openpyxl.load_workbook(OUT/'forecasting_acuan_way_urang.xlsx',read_only=True,data_only=False)
for name,records in t.items():
    rows=list(w[name].iter_rows(values_only=True));assert len(rows)==len(records)+1,(name,len(rows),len(records))
    for actual,expected in zip(rows[1:],records):
        for key,value in zip(rows[0],actual):
            exp=expected.get(key)
            if exp=='' and value is None:continue
            if isinstance(value,datetime):assert value.isoformat(sep=' ')[:19]==str(exp).replace('T',' ')[:19]
            elif isinstance(value,(int,float)) and not isinstance(value,bool) and isinstance(exp,(int,float)):
                assert math.isclose(value,exp,rel_tol=1e-11,abs_tol=1e-10)
            else:assert value==exp,(name,key,value,exp)
w.close()
nb=json.loads((ROOT/'notebooks/05_forecasting_acuan_way_urang.ipynb').read_text(encoding='utf-8'))
assert all(c['execution_count'] is not None and not any(o['output_type']=='error' for o in c['outputs']) for c in nb['cells'] if c['cell_type']=='code')
print(f'Lolos: histori P0/P1 kausal, {manual_knn} prefix KNN manual, fitur lag/indikator, train bersama/purge, seluruh prediksi dan skor, {len(t)} sheet, konfigurasi terkunci, test tidak dimuat, SHA256 identik.')
