"""Independent numeric verification, never trains models or loads final-test values."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json,math,sys
import numpy as np
import pandas as pd
import openpyxl
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'sensitivitas_kalibrasi'
t=json.loads((OUT/'tabel_sensitivitas_kalibrasi.json').read_text(encoding='utf-8'));cfg=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'))
checks=json.loads((RUN/'checks.json').read_text(encoding='utf-8'));gate=json.loads((RUN/'gate_semua_sebelum_fit.json').read_text(encoding='utf-8'))
assert checks['fit_count']==12 and gate['fit_count']==0 and all(g['Ridge fit sebelum gate']==0 and g['Lulus'] for g in gate['gates'])
assert sha256((RUN/'config_locked.json').read_bytes()).hexdigest()==checks['config_sha256']==gate['config_sha256']
assert not cfg['test_loaded'] and not cfg['test_evaluated'] and not cfg['deep_learning_trained']
assert cfg['calibration_days']==[30,60,90] and cfg['fold_cuts']==[5760,10732,15703,20674,25645]
old=json.loads((OUT/'backtesting_way_urang/fold_config_locked.json').read_text(encoding='utf-8'))
for k in ['ridge','covariates','so2_lags','availability','quality']:assert cfg[k]==old[k]
assert cfg['buffer_slots']==48 and cfg['prefix_update_slots']==1440 and cfg['target_scaling'] is False
d=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=25645,parse_dates=['Timestamp']);times=pd.DatetimeIndex(d.Timestamp);y=d['SO2 numerik'].to_numpy(dtype=float)
assert times.max()<pd.Timestamp(cfg['test_start']) and (times[1:]-times[:-1]==pd.Timedelta(minutes=30)).all()
raw=np.column_stack([y]+[d[f'{c} numerik'].shift(1).to_numpy(dtype=float) for c in cfg['covariates']])
pred=pd.read_csv(RUN/'prediksi_evaluasi_semua_konfigurasi.csv',parse_dates=['Origin','Timestamp target'])
assert pred['Timestamp target'].is_unique and pred['Timestamp target'].is_monotonic_increasing
methods=['C30_median','C60_median','C90_median','Persistence'];prefix_checks=0
for number,(a,b) in enumerate(zip(cfg['fold_cuts'][:-1],cfg['fold_cuts'][1:]),1):
    fold=f'Fold_{number:02d}';frame=pred[pred.Fold==fold];origin=frame['Index origin'].to_numpy()
    assert np.array_equal(frame['Index target'],np.flatnonzero(np.isfinite(y[a:b]))+a) and np.array_equal(origin+1,frame['Index target'])
    assert np.array_equal(frame.Origin.to_numpy(),times[origin].values) and np.array_equal(frame['Timestamp target'].to_numpy(),times[origin+1].values)
    assert np.allclose(frame['SO2 target asli'],y[origin+1]) and (frame['Timestamp target']-frame.Origin).eq(pd.Timedelta(minutes=30)).all()
    bank={}
    for days in [30,60,90]:
        folder=RUN/fold/f'C{days}_median';z=np.load(folder/'inputs_dan_sampel.npz');X=z['X'];h=z['filled'];ids=z['train_origins'];w=days*48;bank[days]=X
        assert X.shape==(b-1,34) and h.shape==(b-1,11)
        obs=np.isfinite(raw[:b-1]);assert np.allclose(h[obs],raw[:b-1][obs]) and np.allclose(h[:w],raw[:w],equal_nan=True)
        for start in list(range(w,a,1440))+[a]:
            stop=min(start+1440,a) if start<a else b-1
            if start>=stop:continue
            med=np.nanmedian(raw[:start],axis=0);expected=np.where(np.isfinite(raw[start:stop]),raw[start:stop],med)
            assert np.allclose(h[start:stop],expected,equal_nan=True,rtol=1e-12,atol=1e-10);prefix_checks+=1
            logs=[r for r in t['Prefix_median'] if r['Fold']==fold and r['Kalibrasi hari']==days and r['Prefix N']==start]
            assert len(logs)==11 and all(pd.Timestamp(r['Cutoff prefix eksklusif'])==times[start] for r in logs)
            for j,r in enumerate(logs):
                if np.isfinite(med[j]):assert np.isclose(r['Median prefix'],med[j])
                else:assert r['Median prefix'] is None
                assert r['Observasi prefix']==int(np.isfinite(raw[:start,j]).sum())
        numerical=np.column_stack([pd.Series(h[:,0]).shift(l).to_numpy() for l in cfg['so2_lags']]+[h[:,j] for j in range(1,11)])
        original=np.column_stack([pd.Series(raw[:b-1,0]).shift(l).to_numpy() for l in cfg['so2_lags']]+[raw[:b-1,j] for j in range(1,11)])
        assert np.allclose(X[:,:17],numerical,equal_nan=True) and np.array_equal(X[:,17:],~np.isfinite(original))
        cand=np.arange(w+48,a-1);keep=np.isfinite(y[cand+1])&np.isfinite(X[cand]).all(axis=1)
        assert np.array_equal(ids,cand[keep]) and (ids+1<a).all()
        stored=pd.read_csv(folder/'timestamp_train.csv');assert np.array_equal(stored['Index origin'],ids) and np.allclose(stored['SO2 target asli'],y[ids+1])
        ledger=pd.read_csv(folder/'penanda_sampel_train.csv');assert len(ledger)==a and np.array_equal(ledger['Digunakan'],np.isin(np.arange(a),ids))
        assert np.array_equal(ledger['Alasan utama'].eq('digunakan'),ledger['Digunakan']) and pd.isna(ledger['SO2 target asli'].iloc[-1])
        co=pd.DataFrame([r for r in t['Parameter_Ridge_scaler'] if r['Fold']==fold and r['Kalibrasi hari']==days]);assert co.Fitur.tolist()==cfg['features']
        mean=X[ids].mean(axis=0);sd=X[ids].std(axis=0);sd[sd<1e-12]=1
        assert np.allclose(mean,co['Mean train']) and np.allclose(sd,co['Scale train'])
        beta=co['Koefisien standar'].to_numpy();intercept=float(co.Intercept.iloc[0]);Z=(X[ids]-mean)/sd;res=intercept+Z@beta-y[ids+1]
        assert np.max(np.abs(Z.T@res+cfg['ridge']['alpha']*beta))<1e-4 and abs(res.sum())<1e-5
        good=np.isfinite(X[origin]).all(axis=1);reconstructed=((X[origin[good]]-mean)/sd)@beta+intercept
        assert np.allclose(frame.loc[good,f'C{days}_median'],reconstructed,rtol=1e-10,atol=1e-8)
        assert frame.loc[~good,f'C{days}_median'].isna().all()
        if days==90:
            oldX=pd.read_csv(OUT/f'backtesting_way_urang/{fold}/fitur_P0_median.csv').iloc[:,1:].to_numpy(dtype=float)
            assert np.allclose(X,oldX,equal_nan=True,rtol=1e-11,atol=1e-10)
    assert np.allclose(bank[30][origin],bank[90][origin],equal_nan=True) and np.allclose(bank[60][origin],bank[90][origin],equal_nan=True)
    last=np.maximum.accumulate(np.where(np.isfinite(y[:b-1]),np.arange(b-1),-1))
    assert np.allclose(frame.Persistence,y[last[origin]]) and np.array_equal(frame['Usia donor slot'],origin-last[origin])
assert np.array_equal(pred['Evaluasi bersama'],np.isfinite(pred[methods]).all(axis=1))
for name in ['Metrik_per_fold','Ringkasan','Skor_bulanan']:
    for r in t[name]:
        frame=pred if r['Fold']=='Gabungan_OOF' else pred[pred.Fold==r['Fold']]
        part=np.ones(len(frame),bool)
        if 'Bulan target' in r:part=frame['Timestamp target'].dt.to_period('M').astype(str).eq(r['Bulan target']).to_numpy()
        good=np.isfinite(frame[r['Konfigurasi']]).to_numpy() if r['Domain']=='mandiri' else frame['Evaluasi bersama'].to_numpy()
        keep=good&part;err=frame.loc[keep,r['Konfigurasi']].to_numpy()-frame.loc[keep,'SO2 target asli'].to_numpy()
        assert len(err)==r['N'] and part.sum()==r['Denominator target asli']
        assert np.allclose([np.abs(err).mean(),np.sqrt(np.mean(err**2)),err.mean()],[r['MAE'],r['RMSE'],r['Bias']],atol=1e-9)
        assert math.isclose(r['Cakupan persen'],100*len(err)/part.sum(),abs_tol=1e-10)
        assert sha256('\n'.join(frame.loc[keep,'Timestamp target'].astype(str)).encode()).hexdigest()==r['SHA256 timestamp']
for r in t['Variasi_antar_fold']:
    rows=[x for x in t['Metrik_per_fold'] if x['Konfigurasi']==r['Konfigurasi'] and x['Domain']==r['Domain']]
    for col in ['MAE','RMSE','Bias','Cakupan persen','N']:
        assert math.isclose(np.mean([x[col] for x in rows]),r[col+' mean fold'],abs_tol=1e-10)
        assert math.isclose(np.std([x[col] for x in rows],ddof=1),r[col+' SD fold'],abs_tol=1e-10)
for r in t['Integritas_SHA256']:assert sha256((ROOT/r['File']).read_bytes()).hexdigest()==r['SHA256 sebelum']==r['SHA256 sesudah'],r['File']
if '--audit-only' in sys.argv:
    result={'ridge_count_verified':12,'median_prefixes_verified':prefix_checks,'causal_inputs_verified':True,'test_loaded':False,
        'oof_targets':len(pred),'old_files_identical':len(t['Integritas_SHA256']),'tables_sha256':sha256((OUT/'tabel_sensitivitas_kalibrasi.json').read_bytes()).hexdigest()}
    (RUN/'independent_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result);sys.exit(0)
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
nb=json.loads((ROOT/'notebooks/08_sensitivitas_kalibrasi_ridge.ipynb').read_text(encoding='utf-8'))
assert all(c['execution_count'] is not None and not any(o['output_type']=='error' for o in c['outputs']) for c in nb['cells'] if c['cell_type']=='code')
print(f'Lolos:12Ridge via kondisi penalti tanpa fitulang, {prefix_checks}medianprefix, fitur/target/scaler/purge/ledger/persistence, semua skor dan{len(t)}sheet, {len(t["Integritas_SHA256"])}hash lama identik; tidak ada deep learning atau test.')
