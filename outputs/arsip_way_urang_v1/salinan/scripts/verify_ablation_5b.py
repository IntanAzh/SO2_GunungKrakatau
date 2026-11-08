"""Verifikasi 5B tanpa melatih model: koefisien, histori, kohort, skor dan hash."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json,math
import numpy as np
import pandas as pd
import openpyxl
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'ablasi_curah_hujan_5B';OLD=OUT/'forecasting_way_urang'
t=json.loads((OUT/'tabel_5B_ablasi_curah_hujan.json').read_text(encoding='utf-8'))
cfg=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'));old_cfg=json.loads((OLD/'config_locked.json').read_text(encoding='utf-8'))
checks=json.loads((RUN/'checks.json').read_text(encoding='utf-8'));gate=json.loads((RUN/'gate_sebelum_Ridge.json').read_text(encoding='utf-8'))
assert sha256((RUN/'config_locked.json').read_bytes()).hexdigest()==checks['config_sha256']==gate['Konfigurasi SHA256']
assert gate['Ridge fit count saat pemeriksaan']==0 and checks['ridge_fit_count']==6
assert gate['Target tinggi kembali masuk']>0 and gate['Timestamp lama tidak lagi layak']==0
assert cfg['covariates']==[c for c in old_cfg['covariates'] if c!='Curah Hujan']
for r in t['Konfigurasi_tetap']:assert cfg[r['Parameter']]==old_cfg[r['Parameter']] and r['Sama dengan tahap 5']
assert not cfg['test_loaded'] and not cfg['test_evaluated']
a=cfg['split'][0]['Jumlah slot'];b=a+cfg['split'][1]['Jumlah slot'];n=b-1
d=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=b,parse_dates=['Timestamp'])
times=pd.DatetimeIndex(d.Timestamp);y=d['SO2 numerik'].to_numpy(dtype=float)
raw=np.column_stack([y]+[d[f'{c} numerik'].shift(1).to_numpy(dtype=float) for c in cfg['covariates']])[:n]
X={};hist={};feature_names=None
for method in cfg['pipelines']:
    f=pd.read_csv(RUN/f'fitur_{method}.csv');h=pd.read_csv(RUN/f'histori_kausal_{method}.csv')
    feature_names=list(f.columns[1:]);assert len(feature_names)==34 and all('Curah Hujan' not in c for c in feature_names)
    assert all('Curah Hujan' not in c for c in h.columns)
    X[method]=f.iloc[:,1:].to_numpy(dtype=float);hist[method]=h.iloc[:,1:12].to_numpy(dtype=float)
    known=np.isfinite(raw);assert np.allclose(hist[method][known],raw[known],rtol=1e-12,atol=1e-10)
    lagged=np.column_stack([pd.Series(hist[method][:,0]).shift(lag) for lag in cfg['so2_lags']]+[hist[method][:,j] for j in range(1,11)])
    original=np.column_stack([pd.Series(raw[:,0]).shift(lag) for lag in cfg['so2_lags']]+[raw[:,j] for j in range(1,11)])
    assert np.allclose(X[method][:,:17],lagged,equal_nan=True,rtol=1e-11,atol=1e-10)
    assert np.array_equal(X[method][:,17:],~np.isfinite(original))
    if method!='P2_knn_k10':
        old=pd.read_csv(OLD/f'histori_kausal_{method}.csv').iloc[:,1:12].to_numpy(dtype=float)
        assert np.allclose(hist[method],old,equal_nan=True,rtol=1e-11,atol=1e-10)
ids_by_arm={}
for arm in ['5B_train_diperluas','5B_train_lama_tetap']:
    f=pd.read_csv(RUN/f'train_{arm}.csv');ids_by_arm[arm]=f['Index origin'].to_numpy(dtype=int)
    ids=ids_by_arm[arm];assert (ids+1<a).all() and ids.min()>=cfg['warmup_slots']+max(cfg['so2_lags'])
    assert np.allclose(f['SO2 target asli'],y[ids+1])
    assert all(np.isfinite(matrix[ids]).all() for matrix in X.values())
old_ids=pd.read_csv(OLD/'timestamp_train_bersama.csv')['Index origin'].to_numpy(dtype=int)
assert np.array_equal(ids_by_arm['5B_train_lama_tetap'],old_ids)
candidate=np.arange(cfg['warmup_slots']+max(cfg['so2_lags']),a-1);ok=np.isfinite(y[candidate+1])
for matrix in X.values():ok &= np.isfinite(matrix[candidate]).all(axis=1)
new_ids=ids_by_arm['5B_train_diperluas'];assert np.array_equal(new_ids,candidate[ok])
assert np.isin(old_ids,new_ids).all()
assert len(new_ids)==gate['Train bersama baru']==16530
assert (y[new_ids+1]>2.14).sum()==gate['Target tinggi baru']==3786
manual_knn=0
for log in t['Prefix_imputer']:
    start=log['Jumlah baris prefix'];stop=times.get_loc(pd.Timestamp(log['Akhir query']))+1
    assert start<=a and pd.Timestamp(log['Timestamp donor maksimum'])<pd.Timestamp(log['Awal query'])
    prefix=raw[:start];med=np.nanmedian(prefix,axis=0)
    expected=np.where(np.isfinite(raw[start:stop]),raw[start:stop],med)
    assert np.allclose(hist['P0_median'][start:stop],expected,equal_nan=True,rtol=1e-11,atol=1e-10)
    mean=np.nanmean(prefix,axis=0);scale=np.nanstd(prefix,axis=0);scale[scale==0]=1
    donors=(prefix[np.isfinite(prefix[:,0])]-mean)/scale
    possible=np.flatnonzero((~np.isfinite(raw[start:stop])).any(axis=1)&(np.isfinite(raw[start:stop,1:]).sum(axis=1)>=2))
    for local in possible[:20]:
        q=(raw[start+local]-mean)/scale;overlap=np.isfinite(q)&np.isfinite(donors);count=overlap.sum(axis=1)
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
frames={'Tahap5_dengan_hujan':pd.read_csv(OLD/'prediksi_validasi.csv',parse_dates=['Origin','Timestamp target'])}
for arm,ids in ids_by_arm.items():
    frame=pd.read_csv(RUN/f'prediksi_{arm}.csv',parse_dates=['Origin','Timestamp target']);frames[arm]=frame
    origin=frame['Index origin'].to_numpy(dtype=int);target=frame['Index target'].to_numpy(dtype=int)
    assert np.array_equal(target,np.flatnonzero(np.isfinite(y[a:b]))+a)
    assert np.array_equal(origin+1,target) and target.max()<b
    assert np.allclose(frame['SO2 target asli'],y[target])
    for method,matrix in X.items():
        c=[r for r in t['Koefisien_Ridge'] if r['Lengan']==arm and r['Pipeline']==method]
        assert [r['Fitur'] for r in c]==feature_names
        mean=np.array([r['Mean train'] for r in c]);scale=np.array([r['Skala train'] for r in c]);beta=np.array([r['Koefisien fitur standar'] for r in c])
        assert np.allclose(mean,np.mean(matrix[ids],axis=0),rtol=1e-11,atol=1e-10)
        sd=np.std(matrix[ids],axis=0);sd[sd==0]=1;assert np.allclose(scale,sd,rtol=1e-10,atol=1e-10)
        z=(matrix[ids]-mean)/scale;truth=y[ids+1]
        analytic=np.linalg.solve(z.T@z+cfg['ridge']['alpha']*np.eye(z.shape[1]),z.T@(truth-truth.mean()))
        assert np.allclose(beta,analytic,rtol=1e-6,atol=1e-7)
        valid=np.isfinite(matrix[origin]).all(axis=1);pred=np.full(len(origin),np.nan)
        pred[valid]=((matrix[origin[valid]]-mean)/scale)@beta+c[0]['Intercept']
        assert np.allclose(pred,frame[method],equal_nan=True,rtol=1e-10,atol=1e-9)
    assert np.allclose(frame.Persistence,frames['Tahap5_dengan_hujan'].Persistence)
    assert np.array_equal(frame['Persistence umur donor slot'],frames['Tahap5_dengan_hujan']['Persistence umur donor slot'])
methods=cfg['pipelines']+['Persistence'];common=np.ones(4524,bool)
for frame in frames.values():common &= np.isfinite(frame[methods]).all(axis=1).to_numpy()
assert common.sum()==checks['global_eval_count']==4487
ledger=pd.read_csv(RUN/'timestamp_evaluasi_identik.csv');assert np.array_equal(common,ledger['Irisan seluruh perbandingan'])
for row in t['Ringkasan']+t['Hasil_5B_cakupan']+t['Bulanan_identik']+t['Bulanan_5B_cakupan']:
    frame=frames[row['Lengan']];method=row['Pipeline']
    if row['Domain'].startswith('irisan identik'):keep=common.copy()
    elif row['Domain']=='irisan internal 5B':keep=np.isfinite(frame[methods]).all(axis=1).to_numpy()
    else:keep=np.isfinite(frame[method]).to_numpy()
    month_mask=np.ones(len(frame),bool)
    if 'Bulan target' in row:month_mask=frame['Timestamp target'].dt.to_period('M').astype(str).eq(row['Bulan target']).to_numpy()
    keep=keep&month_mask;err=frame.loc[keep,method]-frame.loc[keep,'SO2 target asli']
    assert len(err)==row['Jumlah prediksi']
    assert math.isclose(100*len(err)/month_mask.sum(),row['Cakupan persen'],abs_tol=1e-10)
    assert np.allclose([np.abs(err).mean(),np.sqrt(np.mean(err**2)),err.mean()],[row['MAE'],row['RMSE'],row['Bias']],rtol=1e-10,atol=1e-10)
    digest=sha256('\n'.join(frame.loc[keep,'Timestamp target'].astype(str)).encode()).hexdigest()
    assert digest==row['SHA256 timestamp target']
assert len({r['SHA256 timestamp target'] for r in t['Ringkasan']})==1
for r in t['Integritas_SHA256']:
    assert sha256((ROOT/r['File']).read_bytes()).hexdigest()==r['SHA256 sebelum']==r['SHA256 sesudah'],r['File']
w=openpyxl.load_workbook(OUT/'hasil_5B_ablasi_curah_hujan.xlsx',read_only=True,data_only=False)
for name,records in t.items():
    rows=list(w[name].iter_rows(values_only=True));assert len(rows)==len(records)+1,(name,len(rows),len(records))
    for actual,expected in zip(rows[1:],records):
        for key,value in zip(rows[0],actual):
            exp=expected.get(key)
            if exp=='' and value is None:continue
            if isinstance(value,datetime):assert value.isoformat(sep=' ')[:19]==str(exp).replace('T',' ')[:19]
            elif isinstance(value,(int,float)) and not isinstance(value,bool) and isinstance(exp,(int,float)):
                assert math.isclose(value,exp,rel_tol=1e-11,abs_tol=1e-9)
            else:assert value==exp,(name,key,value,exp)
w.close()
nb=json.loads((ROOT/'notebooks/05B_ablasi_curah_hujan.ipynb').read_text(encoding='utf-8'))
assert all(c['execution_count'] is not None and not any(o['output_type']=='error' for o in c['outputs']) for c in nb['cells'] if c['cell_type']=='code')
print(f'Lolos: {len(t)} sheet, 6 Ridge via persamaan analitik, {manual_knn} prefix KNN manual, gate nol Ridge, train lama persis, train baru 16.530, 3.786 target tinggi, seluruh skor/timestamp sama, {len(t["Integritas_SHA256"])} hash lama identik, test tidak dimuat.')
