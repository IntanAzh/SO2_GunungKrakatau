"""Independent numerical verification of stored stage9 results; never fits models."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from audit_quality import hashfile

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; RUN=OUT/'kontribusi_antarstasiun'
def verify():
    t=json.loads((OUT/'tabel_kontribusi_antarstasiun.json').read_text(encoding='utf-8'))
    c=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'))
    checks=json.loads((RUN/'checks.json').read_text())
    assert checks['ridge_fit_count']==16 and not checks['test_loaded'] and not checks['test_evaluated']
    assert hashfile(RUN/'config_locked.json')==checks['config_sha256']
    gate=json.loads((RUN/'gate_semua_sebelum_fit.json').read_text()); assert gate['fit_count']==0
    n=c['fold_cuts'][-1]
    way=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=n)
    times=pd.DatetimeIndex(pd.to_datetime(way.Timestamp)); y=way['SO2 numerik'].to_numpy(float)
    raw=np.column_stack([y]+[way[f'{k} numerik'].shift(1).to_numpy(float) for k in c['covariates']])
    donor={d:pd.read_csv(RUN/f'grid_dev_{d}.csv')['SO2 numerik untuk grafik'].to_numpy(float) for d in ['Pulomerak','Talang']}
    def fill(x,cut,b):
        result=x[:b-1].copy()
        for start in list(range(1440,cut,1440))+[cut]:
            stop=min(start+1440,cut) if start<cut else b-1
            med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in x[:start].T])
            result[start:stop]=np.where(np.isfinite(x[start:stop]),x[start:stop],med)
        return result
    p=pd.DataFrame(t['Prediksi']); p['Timestamp target']=pd.to_datetime(p['Timestamp target'])
    assert not p['Timestamp target'].duplicated().any() and p['Timestamp target'].max()<pd.Timestamp(c['test_start'])
    assert np.array_equal(p['Index target'],p['Index origin']+1)
    assert np.allclose(p['SO2 target asli'],y[p['Index target']])
    co=pd.DataFrame(t['Ridge_scaler']); max_gradient=0.; model_checks=0
    for number,(a,b) in enumerate(zip(c['fold_cuts'][:-1],c['fold_cuts'][1:]),1):
        label=f'Fold_{number:02d}'; folder=RUN/label; z=np.load(folder/'inputs_train_eval.npz')
        f=fill(raw,a,b); assert np.allclose(z['way_filled'],f,equal_nan=True)
        base_num=np.column_stack([pd.Series(f[:,0]).shift(l) for l in c['so2_lags']]+[f[:,j] for j in range(1,11)])
        base_raw=np.column_stack([pd.Series(raw[:b-1,0]).shift(l) for l in c['so2_lags']]+[raw[:b-1,j] for j in range(1,11)])
        base=np.column_stack([base_num,(~np.isfinite(base_raw)).astype(float)])
        blocks={}
        for d,x in donor.items():
            f=fill(x[:,None],a-1,b)[:,0]; assert np.allclose(z[d+'_filled'],f,equal_nan=True)
            vals=np.column_stack([pd.Series(f).shift(k) for k in [1,2,3,7,13,25,49]])
            missing=np.column_stack([pd.Series(x[:b-1]).shift(k) for k in [1,2,3,7,13,25,49]])
            blocks[d]=np.column_stack([vals,(~np.isfinite(missing)).astype(float)])
        own={}; shared={}
        for part,ids in [('train',np.arange(1488,a-1)),('evaluation',np.arange(a-1,b-1))]:
            for s,ds in c['scenarios'].items():
                X=np.column_stack([base]+[blocks[d] for d in ds]); assert np.allclose(z[s],X,equal_nan=True)
                own[s,part]=ids[np.isfinite(y[ids+1])&np.isfinite(X[ids]).all(1)&(ids>=(1488 if s=='S0' else 1489))]
            shared[part]=own['S0',part]
            for s in c['scenarios']:shared[part]=np.intersect1d(shared[part],own[s,part])
            assert np.array_equal(shared[part],z[part])
        train=shared['train']; sub=p[p.Fold==label]; origin=sub['Index origin'].to_numpy(int)
        assert np.array_equal(shared['evaluation'],origin)
        for s in c['scenarios']:
            coef=co[(co.Fold==label)&(co.Skenario==s)]; mean=coef['Mean train'].to_numpy(); scale=coef['Scale train'].to_numpy()
            beta=coef['Koefisien standar'].to_numpy(); intercept=coef.Intercept.iloc[0]; X=z[s]
            assert np.allclose(mean,X[train].mean(0),atol=1e-9)
            expected=X[train].std(0); expected[expected<1e-12]=1
            assert np.allclose(scale,expected,atol=1e-8)
            Z=(X[train]-mean)/scale; residual=intercept+Z@beta-y[train+1]
            grad=np.max(np.abs(Z.T@residual+c['ridge']['alpha']*beta)); max_gradient=max(max_gradient,float(grad))
            assert grad<1e-3 and abs(residual.mean())<1e-8
            assert np.allclose(sub[s],intercept+((X[origin]-mean)/scale)@beta,atol=1e-8)
            model_checks+=1
        last=np.maximum.accumulate(np.where(np.isfinite(y[:b-1]),np.arange(b-1),-1))[origin]
        assert np.allclose(sub.Persistence,y[last]) and np.array_equal(sub['Persistence usia slot'],origin-last)
    for d,x in donor.items():
        src=pd.read_csv(RUN/f'sumber_dev_{d}.csv',parse_dates=['Timestamp']); original=pd.DatetimeIndex(src.loc[src['SO2 numerik'].notna()&src.Timestamp.notna(),'Timestamp'])
        last=np.maximum.accumulate(np.where(np.isfinite(x),np.arange(n),-1))
        origins=p['Index origin'].to_numpy(int); cutoff=origins-1
        assert np.allclose(p[d+' Usia asli menit grid'],(origins-last[cutoff])*30)
        oi=original.searchsorted(times[cutoff],side='right')-1
        age=(times[origins]-original[oi]).total_seconds()/60
        assert np.allclose(p[d+' Usia asli menit semua'],age) and np.min(age)>=30
    used_cache={}
    for r in t['Rentetan_konstan']:
        x=donor[r['Donor']]; start=times.get_loc(pd.Timestamp(r['Run mulai']));stop=times.get_loc(pd.Timestamp(r['Run akhir']))+1
        assert stop-start==r['Jumlah slot run'] and (stop-start)*30==r['Durasi run menit']
        assert np.isfinite(x[start:stop]).all() and np.all(x[start:stop]==r['Nilai SO2 asli'])
        key=(r['Fold'],r['Bagian sampel model'])
        if key not in used_cache:
            f=np.load(RUN/r['Fold']/'inputs_train_eval.npz'); ids=f[r['Bagian sampel model']]
            used_cache[key]=np.unique((ids[:,None]-np.array([1,2,3,7,13,25,49])).ravel())
        used=used_cache[key]
        assert int(((used>=start)&(used<stop)).sum())==r['Timestamp sumber unik masuk fitur']
    for r in t['Skor_per_fold']+t['Ringkasan']+t['Skor_bulanan']:
        q=p if r['Fold']=='Gabungan_OOF' else p[p.Fold==r['Fold']]
        month=r.get('Bulan target',r.get('Bulan'))
        if month is not None:q=q[q['Timestamp target'].dt.strftime('%Y-%m')==month]
        err=q[r['Skenario']].to_numpy(float)-q['SO2 target asli'].to_numpy(float); err=err[np.isfinite(err)]
        assert len(err)==r['N']
        assert np.allclose([r['MAE'],r['RMSE'],r['Bias']],[np.abs(err).mean(),np.sqrt(np.mean(err**2)),err.mean()],atol=1e-9)
        assert np.isclose(r['Cakupan persen'],100*len(err)/r['Denominator target asli'])
    for r in t['Vs_S0_persistence']+t['Vs_S0_bulanan']:
        q=p if r['Fold']=='Gabungan_OOF' else p[p.Fold==r['Fold']]
        if 'Bulan target' in r:q=q[q['Timestamp target'].dt.strftime('%Y-%m')==r['Bulan target']]
        error=lambda s:q[s].to_numpy()-q['SO2 target asli'].to_numpy()
        for metric,fun in [('MAE',lambda z:np.abs(z).mean()),('RMSE',lambda z:np.sqrt(np.mean(z**2)))]:
            own=fun(error(r['Skenario']));base=fun(error('S0'));per=fun(error('Persistence'))
            assert np.isclose(r[f'Delta {metric} vs S0'],own-base)
            assert np.isclose(r[f'Perbaikan {metric} persen'],100*(base-own)/base)
            assert np.isclose(r[f'Delta {metric} vs persistence'],own-per)
    for rel,sha in c['initial_hashes'].items():assert hashfile(ROOT/rel)==sha,rel
    result={'model_equations_verified_without_refit':model_checks,'maximum_ridge_normal_equation_residual':max_gradient,'original_targets_verified':len(p),'old_hashes_identical':len(c['initial_hashes']),'source_and_features_development_only':True,'donor_delays_and_ages_verified':True,'causal_prefix_medians_verified':True,'all_fold_monthly_pooled_metrics_verified':True,'tables_sha256':hashfile(OUT/'tabel_kontribusi_antarstasiun.json')}
    (RUN/'independent_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
