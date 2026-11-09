"""Independent stage11 checks; reconstruct saved predictions without training."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import torch
from audit_quality import hashfile
from algorithms_stage11 import GRUModel,TCNModel,causality_gate,ROOT,OUT,OLD,RUN,SEEDS

def verify(partial=False):
    cfg=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'))
    gates=json.loads((RUN/'gate_semua_sebelum_fit.json').read_text(encoding='utf-8'))
    ch=hashfile(RUN/'config_locked.json');assert gates['config_sha256']==ch and gates['fit_count']==0
    assert hashfile(ROOT/'scripts/algorithms_stage11.py')==cfg['implementation_sha256']
    assert hashfile(OLD/'config_locked.json')==cfg['stage10_config_sha256'] and hashfile(OLD/'manifest_tahap10.json')==cfg['stage10_manifest_sha256']
    assert not cfg['test_loaded'] and not cfg['test_evaluated'] and cfg['planned_new_fits']==24
    assert len(gates['folds'])==4 and all(r['Lulus'] for r in gates['folds'])
    assert causality_gate()==cfg['causal_test'] and cfg['architecture']['TCN']['receptive_field_slots']==61
    torch.set_num_threads(1)
    n=cfg['fold_cuts'][-1];data=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=n,parse_dates=['Timestamp'])
    times=pd.DatetimeIndex(data.Timestamp);y=data['SO2 numerik'].to_numpy(float)
    checked=0;checked_samples=0;foldinfo=[];model_shapes=[]
    for i,(a,b) in enumerate(zip(cfg['fold_cuts'][:-1],cfg['fold_cuts'][1:]),1):
        fold=f'Fold_{i:02d}';new=RUN/fold;old=OLD/fold
        z=np.load(old/'inputs_sekuens.npz');x=z['S0'];assert x.shape==(b-1,22)
        for part,key in [('inner_train','train'),('inner_validation','internal'),('evaluation','evaluation')]:
            ids=z[key];f=pd.read_csv(new/f'timestamp_{part}.csv',parse_dates=['Timestamp target'])
            oldf=pd.read_csv(old/f'timestamp_{part}_bersama.csv')
            assert np.array_equal(f['Index origin'],oldf['Index origin']) and np.array_equal(f['Index origin'],ids)
            assert np.array_equal(f['SO2 target asli'],y[ids+1])
            assert np.array_equal(f['Timestamp target'].to_numpy(),times[ids+1].to_numpy())
        tr=z['train'];val=z['internal'];ev=z['evaluation'];c=int(.85*a)
        assert (tr+1<c).all() and ((val+1>=c)&(val+1<a)).all() and ((ev+1>=a)&(ev+1<b)).all()
        scale=json.loads((old/'parameter_scaler.json').read_text(encoding='utf-8'))
        sc=[r for r in scale if r['Skenario']=='S0'];target_mean=sc[0]['Target mean'];target_scale=sc[0]['Target scale']
        assert np.isclose(target_mean,y[tr+1].mean()) and np.isclose(target_scale,y[tr+1].std())
        slots=np.unique((tr[:,None]-np.arange(48)).ravel());assert np.array_equal(z['scale_slots'],slots) and slots.max()<c
        assert np.isfinite(x[ev[:,None]-np.arange(47,-1,-1)]).all()
        ref={seed:pd.read_csv(old/f'S0_seed{seed}/prediksi.csv',parse_dates=['Timestamp target']) for seed in SEEDS}
        for seed in SEEDS:
            f=ref[seed];assert np.array_equal(f['Index origin'],ev)
            assert np.array_equal(f['SO2 target asli'],y[ev+1]) and f['Timestamp target'].max()<pd.Timestamp(cfg['test_start'])
            for method,klass in [('GRU',GRUModel),('TCN',TCNModel)]:
                modeldir=new/f'{method}_seed{seed}';donefile=modeldir/'complete.json'
                if partial and not donefile.exists():continue
                done=json.loads(donefile.read_text(encoding='utf-8'));assert done['config_sha256']==ch
                assert done['Jumlah parameter']==cfg['architecture'][method]['parameters']
                for name,h in done['files_sha256'].items():assert hashfile(modeldir/name)==h
                hist=json.loads((modeldir/'history.json').read_text(encoding='utf-8'))
                best=min(hist,key=lambda r:r['Internal MSE standar'])
                assert len(hist)==done['Epoch dijalankan']<=50 and best['Epoch']==done['Epoch terbaik']
                if len(hist)<50:assert len(hist)-best['Epoch']==8
                q=pd.read_csv(modeldir/'prediksi.csv',parse_dates=['Origin','Timestamp target'])
                assert np.array_equal(q['Index origin'],ev) and np.array_equal(q['Index target'],ev+1)
                assert np.array_equal(q['SO2 target asli'],f['SO2 target asli']) and np.array_equal(q.Persistence,f.Persistence)
                assert np.array_equal(q['Timestamp target'].to_numpy(),times[ev+1].to_numpy())
                assert np.array_equal(q['Origin'].to_numpy(),times[ev].to_numpy())
                assert (q['Timestamp target']-q.Origin==pd.Timedelta(minutes=30)).all()
                assert q['Timestamp target'].max()<pd.Timestamp(cfg['test_start'])
                model=klass();model.load_state_dict(torch.load(modeldir/'best.pt',map_location='cpu',weights_only=True));model.eval()
                selected=np.unique(np.r_[np.arange(0,len(ev),336),np.arange(min(3,len(ev))),np.arange(max(0,len(ev)-3),len(ev))])
                if i==1:
                    around=np.flatnonzero(q['Timestamp target'].between(pd.Timestamp('2025-06-22')-pd.Timedelta(days=1),pd.Timestamp('2025-06-22')+pd.Timedelta(days=1)))
                    selected=np.unique(np.r_[selected,around[::12]])
                seq=torch.from_numpy(x[ev[selected,None]-np.arange(47,-1,-1)])
                with torch.no_grad():p=model(seq).numpy()*target_scale+target_mean
                assert np.allclose(p,q.Prediksi.iloc[selected],atol=3e-5,rtol=3e-6,equal_nan=True)
                if method=='TCN':
                    with torch.no_grad():
                        xx=seq[:2].clone();full=model.backbone(xx)
                        for length in [1,7,17,47]:
                            changed=xx.clone();changed[:,length:]=999
                            assert torch.allclose(model.backbone(changed)[:,:,:length],full[:,:,:length],atol=2e-5,rtol=2e-6)
                checked+=1;checked_samples+=len(selected)
                model_shapes.append({'Fold':fold,'Algoritma':method,'Seed':seed,'Parameter':sum(v.numel() for v in model.parameters()),
                    'Checkpoint direkonstruksi':len(selected),'Best epoch':best['Epoch']})
        foldinfo.append({'Fold':fold,'Train N':len(tr),'Internal N':len(val),'Evaluasi N':len(ev),'Causality and source verified':True})
    for rel,h in cfg['initial_hashes'].items():assert hashfile(ROOT/rel)==h,rel
    result={'config_sha256':ch,'new_models_verified_without_fit':checked,'checkpoint_predictions_recomputed':checked_samples,
        'old_files_sha256_identical':len(cfg['initial_hashes']),'folds':foldinfo,'model_shapes':model_shapes,
        'causal_tcn_prefix_tests_passed':True,'test_loaded':False,'test_evaluated':False}
    if not partial:
        assert checked==24
        tables=json.loads((RUN/'tabel_hasil.json').read_text(encoding='utf-8'))
        pred=pd.read_csv(RUN/'prediksi_oof_seluruh_seed.csv',parse_dates=['Timestamp target'])
        assert not pred[['Seed','Timestamp target']].duplicated().any() and pred['Timestamp target'].max()<pd.Timestamp(cfg['test_start'])
        for r in tables['Skor_fold_seed']+tables['Gabungan_per_seed']+tables['Skor_bulanan']:
            f=pred[pred.Seed.eq(r['Seed'])]
            if r['Fold']!='Gabungan_OOF':f=f[f.Fold.eq(r['Fold'])]
            if 'Bulan target' in r:f=f[f['Timestamp target'].dt.strftime('%Y-%m').eq(r['Bulan target'])]
            den=len(f);mask=f['Timestamp bersama'] if r['Domain']=='bersama' else np.isfinite(f[r['Algoritma']]);f=f.loc[mask]
            error=f[r['Algoritma']].to_numpy(float)-f['SO2 target asli'].to_numpy(float)
            assert len(f)==r['N'] and den==r['Denominator target asli']
            assert np.allclose([r['MAE'],r['RMSE'],r['Bias']],[np.abs(error).mean(),np.sqrt(np.mean(error**2)),error.mean()],atol=1e-8)
            assert r['Prediksi negatif']==int((f[r['Algoritma']]<0).sum()) and np.isclose(r['Cakupan persen'],100*len(f)/den)
        for r in tables['Vs_persistence']:
            f=pred[pred.Seed.eq(r['Seed'])]
            if r['Fold']!='Gabungan_OOF':f=f[f.Fold.eq(r['Fold'])]
            f=f[f['Timestamp bersama']]
            err=lambda m:f[m].to_numpy(float)-f['SO2 target asli'].to_numpy(float)
            a=err(r['Algoritma']);p=err('Persistence')
            assert len(f)==r['N identik']
            assert np.isclose(r['Delta MAE vs persistence'],np.abs(a).mean()-np.abs(p).mean())
            assert np.isclose(r['Delta RMSE vs persistence'],np.sqrt(np.mean(a*a))-np.sqrt(np.mean(p*p)))
        result['metrics_fold_month_pooled_recomputed']=True;result['tables_sha256']=hashfile(RUN/'tabel_hasil.json')
    path=RUN/('partial_checkpoint_checks.json' if partial else 'independent_checks.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['model_shapes','folds']},indent=2))
if __name__=='__main__':verify('--partial' in sys.argv)
