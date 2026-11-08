"""Independent checks from saved arrays; no new model training or test access."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json,math
import numpy as np
import pandas as pd
import openpyxl
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'backtesting_way_urang'
t=json.loads((OUT/'tabel_backtesting_preprocessing.json').read_text(encoding='utf-8'))
cfg=json.loads((RUN/'fold_config_locked.json').read_text(encoding='utf-8'))
old=json.loads((OUT/'ablasi_curah_hujan_5B/config_locked.json').read_text(encoding='utf-8'))
checks=json.loads((RUN/'checks.json').read_text(encoding='utf-8'))
gates=json.loads((RUN/'all_gates_before_models.json').read_text(encoding='utf-8'))
assert gates['global_ridge_fit_count']==0 and gates['all_gates_passed'] and checks['ridge_fit_count']==12
assert sha256((RUN/'fold_config_locked.json').read_bytes()).hexdigest()==checks['config_sha256']==gates['config_sha256']
assert not cfg['test_loaded'] and not cfg['test_evaluated']
assert cfg['fold_cuts']==[5760,10732,15703,20674,25645]
for row in t['Parameter_tetap_5B']:assert cfg[row['Parameter']]==old[row['Parameter']]
n=cfg['fold_cuts'][-1];methods=['P0_median','P1_ffill_L4','P2_knn_k10'];allmethods=methods+['Persistence']
d=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=n,parse_dates=['Timestamp'])
times=pd.DatetimeIndex(d.Timestamp);y=d['SO2 numerik'].to_numpy(dtype=float)
raw=np.column_stack([y]+[d[f'{c} numerik'].shift(1).to_numpy(dtype=float) for c in cfg['covariates']])
p=pd.read_csv(RUN/'prediksi_oof_development.csv',parse_dates=['Timestamp target','Origin'])
manual_knn=0
assert p['Timestamp target'].is_unique and p['Timestamp target'].is_monotonic_increasing
assert p['Timestamp target'].max()<pd.Timestamp(cfg['split'][2]['Awal'])
assert np.array_equal(p['Index target'],np.flatnonzero(np.isfinite(y)&(np.arange(n)>=5760)))
assert np.allclose(p['SO2 target asli'],y[p['Index target']])
assert np.array_equal(p['Irisan semua metode'],np.isfinite(p[allmethods]).all(axis=1))
for number,(a,b) in enumerate(zip(cfg['fold_cuts'][:-1],cfg['fold_cuts'][1:]),1):
    fold=f'Fold_{number:02d}';folder=RUN/fold
    ids=pd.read_csv(folder/'timestamp_train_bersama.csv')['Index origin'].to_numpy()
    cand=np.arange(cfg['warmup_slots']+max(cfg['so2_lags']),a-1)
    shared=np.isfinite(y[cand+1]);frame=p[p.Fold==fold];origins=frame['Index origin'].to_numpy()
    X={}
    for method in methods:
        f=pd.read_csv(folder/f'fitur_{method}.csv');X[method]=f.iloc[:,1:].to_numpy(dtype=float)
        assert X[method].shape==(b-1,34) and all('Curah Hujan' not in c for c in f.columns)
        shared=shared&np.isfinite(X[method][cand]).all(axis=1)
        h=pd.read_csv(folder/f'histori_kausal_{method}.csv').iloc[:,1:12].to_numpy(dtype=float)
        obs=np.isfinite(raw[:b-1]);assert np.allclose(h[obs],raw[:b-1][obs])
        numerical=np.column_stack([pd.Series(h[:,0]).shift(l).to_numpy() for l in cfg['so2_lags']]+[h[:,j] for j in range(1,11)])
        original=np.column_stack([pd.Series(raw[:b-1,0]).shift(l).to_numpy() for l in cfg['so2_lags']]+[raw[:b-1,j] for j in range(1,11)])
        assert np.allclose(X[method][:,:17],numerical,equal_nan=True)
        assert np.array_equal(X[method][:,17:],~np.isfinite(original))
        if method=='P0_median':
            for r in [r for r in t['Prefix_imputer'] if r['Fold']==fold]:
                start=times.get_loc(pd.Timestamp(r['Awal query']));end=times.get_loc(pd.Timestamp(r['Akhir query']))+1
                med=np.nanmedian(raw[:start],axis=0);expected=np.where(np.isfinite(raw[start:end]),raw[start:end],med)
                assert np.allclose(h[start:end],expected,equal_nan=True)
                assert pd.Timestamp(r['Timestamp donor maksimum'])<times[start] and start<=a
        if method=='P1_ffill_L4':
            last=np.full(11,-1);value=np.full(11,np.nan)
            for u,row in enumerate(raw[:b-1]):
                seen=np.isfinite(row);last[seen]=u;value[seen]=row[seen]
                expected=row.copy()
                if u>=cfg['warmup_slots']:
                    fill=(~seen)&(last>=0)&(u-last<=4);expected[fill]=value[fill]
                assert np.allclose(h[u],expected,equal_nan=True)
        if method=='P2_knn_k10':
            for log in [r for r in t['Prefix_imputer'] if r['Fold']==fold]:
                start=log['Jumlah baris prefix'];stop=times.get_loc(pd.Timestamp(log['Akhir query']))+1
                prefix=raw[:start];mean=np.nanmean(prefix,axis=0);scale=np.nanstd(prefix,axis=0);scale[scale==0]=1
                assert np.isfinite(mean).all()
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
                    value=np.sum(weights*donors[near,col])/weights.sum()*scale[col]+mean[col]
                    assert math.isclose(value,h[start+local,col],rel_tol=1e-7,abs_tol=1e-7)
                    manual_knn+=1;break
        co=pd.DataFrame([r for r in t['Koefisien_Ridge'] if r['Fold']==fold and r['Pipeline']==method])
        good=np.isfinite(X[method][origins]).all(axis=1)
        z=(X[method][origins[good]]-co['Mean fitur train'].to_numpy())/co['Skala fitur train'].to_numpy()
        pred=z@co['Koefisien fitur standar'].to_numpy()+co.Intercept.iloc[0]
        assert np.allclose(frame.loc[good,method],pred,rtol=1e-10,atol=1e-8)
    assert np.array_equal(ids,cand[shared]) and (ids+1<a).all() and len(ids)>=340
    for method in methods:
        co=pd.DataFrame([r for r in t['Koefisien_Ridge'] if r['Fold']==fold and r['Pipeline']==method])
        xt=X[method][ids];mean=xt.mean(axis=0);scale=xt.std(axis=0);scale[scale<1e-12]=1
        assert np.allclose(co['Mean fitur train'],mean) and np.allclose(co['Skala fitur train'],scale)
        z=(xt-mean)/scale;yc=y[ids+1]-np.mean(y[ids+1]);zc=z-z.mean(axis=0)
        beta=np.linalg.solve(zc.T@zc+cfg['ridge']['alpha']*np.eye(34),zc.T@yc)
        assert np.allclose(beta,co['Koefisien fitur standar'],atol=2e-6,rtol=1e-6)
    last=np.maximum.accumulate(np.where(np.isfinite(y[:b-1]),np.arange(b-1),-1))
    assert np.allclose(frame.Persistence,y[last[origins]])
    assert np.array_equal(frame['Usia donor persistence slot'],origins-last[origins])

assert manual_knn>=4
for key in ['Ringkasan_per_fold','Ringkasan','Skor_bulanan_fold','Skor_bulanan_gabungan','Strata_SO2_validasi']:
    for r in t[key]:
        frame=p if r['Fold']=='Gabungan_OOF' else p[p.Fold==r['Fold']]
        part=np.ones(len(frame),bool)
        if 'Bulan target' in r:part=frame['Timestamp target'].dt.to_period('M').astype(str).eq(r['Bulan target']).to_numpy()
        if 'Strata' in r:part=(frame['SO2 target asli'].to_numpy()>2.14) if r['Strata']=='SO2 lebih 2.14' else (frame['SO2 target asli'].to_numpy()<=2.14)
        mask=part&(frame['Irisan semua metode'].to_numpy() if r['Domain']=='irisan semua metode' else np.isfinite(frame[r['Pipeline']]).to_numpy())
        e=frame.loc[mask,r['Pipeline']].to_numpy()-frame.loc[mask,'SO2 target asli'].to_numpy()
        assert len(e)==r['Jumlah prediksi'] and int(part.sum())==r['Denominator target asli']
        if len(e):assert np.allclose([np.abs(e).mean(),np.sqrt(np.mean(e**2)),e.mean()],[r['MAE'],r['RMSE'],r['Bias']],rtol=1e-9,atol=1e-9)
        else:assert r['MAE'] is None and r['RMSE'] is None and r['Bias'] is None
        if part.sum():assert math.isclose(100*len(e)/part.sum(),r['Cakupan persen'],abs_tol=1e-10)
        assert sha256('\n'.join(frame.loc[mask,'Timestamp target'].astype(str)).encode()).hexdigest()==r['SHA256 timestamp target']
for r in t['Variasi_antar_fold']:
    rows=[x for x in t['Ringkasan_per_fold'] if x['Pipeline']==r['Pipeline'] and x['Domain']==r['Domain']]
    for col in ['MAE','RMSE','Bias','Cakupan persen','Jumlah prediksi']:
        assert math.isclose(np.mean([x[col] for x in rows]),r[col+' rata-rata fold'],abs_tol=1e-10)
        assert math.isclose(np.std([x[col] for x in rows],ddof=1),r[col+' SD fold'],abs_tol=1e-10)
for r in t['Integritas_SHA256']:assert sha256((ROOT/r['File']).read_bytes()).hexdigest()==r['SHA256 sebelum']==r['SHA256 sesudah']
w=openpyxl.load_workbook(OUT/'backtesting_preprocessing.xlsx',read_only=True,data_only=False)
for name,records in t.items():
    rows=list(w[name].iter_rows(values_only=True));assert len(rows)==len(records)+1
    for actual,expected in zip(rows[1:],records):
        for key,value in zip(rows[0],actual):
            exp=expected.get(key)
            if exp=='' and value is None:continue
            if isinstance(value,datetime):assert value.isoformat(sep=' ')[:19]==str(exp).replace('T',' ')[:19]
            elif isinstance(value,(int,float)) and not isinstance(value,bool) and isinstance(exp,(int,float)):assert math.isclose(value,exp,rel_tol=1e-11,abs_tol=1e-8)
            else:assert value==exp,(name,key,value,exp)
w.close()
nb=json.loads((ROOT/'notebooks/06_backtesting_preprocessing.ipynb').read_text(encoding='utf-8'))
assert all(c['execution_count'] is not None and not any(o['output_type']=='error' for o in c['outputs']) for c in nb['cells'] if c['cell_type']=='code')
print(f'Lolos: 4 fold kalender, 12 Ridge diverifikasi persamaan analitik, {manual_knn} prefix KNN manual, histori kausal P0/P1, indikator/lag/purge/irisan, seluruh skor fold/bulan/gabungan, {len(p)} target OOF unik, {len(t)} sheet, {len(t["Integritas_SHA256"])} hash identik. Test akhir tidak dimuat.')
