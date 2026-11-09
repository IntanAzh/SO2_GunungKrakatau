"""Independent checks, including checkpoint inference; does not fit any model."""
from pathlib import Path
import sys,json,math
import numpy as np
import pandas as pd
from audit_quality import hashfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'lstm_antarstasiun_tahap10'

def verify(preflight_only=False,completed_only=False):
    cfg=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'));gate=json.loads((RUN/'gate_semua_sebelum_fit.json').read_text(encoding='utf-8'))
    assert hashfile(RUN/'config_locked.json')==gate['config_sha256']
    assert hashfile(ROOT/'scripts/lstm_interstation_stage10.py')==cfg['implementation_sha256']
    assert gate['fit_count']==0 and len(gate['gates'])==4 and all(x['Lulus'] for x in gate['gates'])
    assert cfg['seeds']==[11,23,47] and cfg['warmup_slots']==1440 and cfg['sequence_length']==48
    assert not cfg['test_loaded'] and not cfg['test_evaluated']
    n=cfg['fold_cuts'][-1];data=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=n)
    times=pd.DatetimeIndex(pd.to_datetime(data.Timestamp));y=data['SO2 numerik'].to_numpy(float)
    donor=pd.read_csv(OUT/'kontribusi_antarstasiun/grid_dev_Pulomerak.csv',nrows=n)['SO2 numerik untuk grafik'].to_numpy(float)
    raw=np.column_stack([y]+[data[k+' numerik'].shift(1).to_numpy(float) for k in cfg['covariates']]+[pd.Series(donor).shift(1)])
    last=np.maximum.accumulate(np.where(np.isfinite(donor),np.arange(n),-1))
    result={'config_sha256':gate['config_sha256'],'preflight_only':preflight_only,'model_fits_in_verifier':0,'folds':[]}
    frames=[];verified_models=0
    if not preflight_only:
        import torch
        from torch import nn
        torch.set_num_threads(1)
        class CheckModel(nn.Module):
            def __init__(self,channels):
                super().__init__();self.lstm=nn.LSTM(channels,32,num_layers=1,batch_first=True);self.output=nn.Linear(32,1)
            def forward(self,x):return self.output(self.lstm(x)[0][:,-1,:]).squeeze(-1)
    for i,(a,b) in enumerate(zip(cfg['fold_cuts'][:-1],cfg['fold_cuts'][1:]),1):
        label=f'Fold_{i:02d}';c=int(.85*a);folder=RUN/label;z=np.load(folder/'inputs_sekuens.npz');h=raw[:b-1].copy()
        for start in list(range(1440,c,1440))+[c]:
            stop=min(start+1440,c) if start<c else b-1
            med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in raw[:start].T])
            h[start:stop]=np.where(np.isfinite(raw[start:stop]),raw[start:stop],med)
        assert np.allclose(z['filled'],h,equal_nan=True) and np.array_equal(z['raw_flags'],(~np.isfinite(raw[:b-1])).astype(float))
        observed=np.isfinite(raw[:b-1]);assert np.array_equal(z['filled'][observed],raw[:b-1][observed])
        parts={'train':np.arange(1488,c-1),'internal':np.arange(max(1488,c-1),a-1),'evaluation':np.arange(a-1,b-1)};common={};own={}
        for part,ids in parts.items():
            good=np.isfinite(y[ids+1])
            for s,d in [('S0',11),('S1',12)]:
                complete=np.isfinite(h[ids[:,None]-np.arange(48),:d]).all(axis=(1,2))
                own[s,part]=ids[good&complete]
            common[part]=np.intersect1d(own['S0',part],own['S1',part]);assert np.array_equal(z[part],common[part])
        tr=common['train'];assert (tr+1<c).all() and ((common['internal']+1>=c)&(common['internal']+1<a)).all()
        slots=np.unique((tr[:,None]-np.arange(48)).ravel());assert np.array_equal(z['scale_slots'],slots) and slots.max()<c
        scales=json.loads((folder/'parameter_scaler.json').read_text(encoding='utf-8'));target_mean=y[tr+1].mean();target_scale=y[tr+1].std() or 1.
        for s,d in [('S0',11),('S1',12)]:
            r=[x for x in scales if x['Skenario']==s];mean=np.array([x['Mean'] for x in r]);scale=np.array([x['Scale'] for x in r])
            std=h[slots,:d].std(0);std[std<1e-12]=1
            assert np.allclose(mean,h[slots,:d].mean(0)) and np.allclose(scale,std)
            assert np.allclose(z[s][:,:d],(h[:,:d]-mean)/scale,equal_nan=True,atol=2e-6)
            assert np.array_equal(z[s][:,d:],(~np.isfinite(raw[:b-1,:d])).astype(float))
            assert all(np.isclose(x['Target mean'],target_mean) and np.isclose(x['Target scale'],target_scale) for x in r)
            if preflight_only:continue
            for seed in cfg['seeds']:
                modeldir=folder/f'{s}_seed{seed}'
                if completed_only and not (modeldir/'complete.json').exists():continue
                done=json.loads((modeldir/'complete.json').read_text(encoding='utf-8'));assert done['config_sha256']==gate['config_sha256']
                for name,sha in done['files_sha256'].items():assert hashfile(modeldir/name)==sha
                history=json.loads((modeldir/'history.json').read_text(encoding='utf-8'));best=min(history,key=lambda r:r['Internal MSE standar'])
                assert best['Epoch']==done['Epoch terbaik'] and len(history)==done['Epoch dijalankan'] and len(history)<=50
                if len(history)<50:assert len(history)-best['Epoch']==8
                assert done['Fit early stopping detik']>0 and done['Predict detik']>0
                frame=pd.read_csv(modeldir/'prediksi.csv',parse_dates=['Origin','Timestamp target']);ids=frame['Index origin'].to_numpy(int);target=frame['Index target'].to_numpy(int)
                assert np.array_equal(ids+1,target) and np.array_equal(frame['SO2 target asli'],y[target])
                assert np.array_equal(frame['Timestamp target'].to_numpy(),times[target].to_numpy())
                assert np.array_equal(frame['Origin'].to_numpy(),times[ids].to_numpy()) and (frame['Timestamp target']-frame.Origin==pd.Timedelta(minutes=30)).all()
                assert frame['Timestamp target'].max()<pd.Timestamp(cfg['test_start'])
                expected=own[s,'evaluation'];assert np.array_equal(ids[np.isfinite(frame.Prediksi)],expected)
                assert np.allclose(frame['Pulomerak usia asli origin menit'],(ids-last[ids-1])*30)
                missing=(~np.isfinite(donor[ids[:,None]-np.arange(1,49)])).sum(1)
                assert np.array_equal(frame['Pulomerak posisi missing asli dari48'],missing)
                assert np.allclose(frame['Pulomerak proporsi posisi diimputasi'],missing/48)
                lastway=np.maximum.accumulate(np.where(np.isfinite(y),np.arange(n),-1))[ids]
                assert np.allclose(frame.Persistence,y[lastway]) and np.array_equal(frame['Persistence usia slot'],ids-lastway)
                model=CheckModel(d*2);model.load_state_dict(torch.load(modeldir/'best.pt',map_location='cpu',weights_only=True));model.eval()
                # Verify representative first/last/every-seventh-day plus transition samples directly from saved channels.
                selected=np.unique(np.r_[np.arange(0,len(ids),336),np.arange(min(3,len(ids))),np.arange(max(0,len(ids)-3),len(ids))])
                selected=selected[np.isfinite(frame.Prediksi.iloc[selected])];ix=ids[selected]
                seq=z[s][ix[:,None]-np.arange(47,-1,-1)]
                with torch.no_grad():pred=model(torch.from_numpy(seq)).numpy()*target_scale+target_mean
                assert np.allclose(pred,frame.Prediksi.iloc[selected],atol=2e-5,rtol=2e-6)
                verified_models+=1;frames.append(frame)
        # Every source donor timestamp in a sequence is strictly before the associated step.
        assert min(common['evaluation'])-48>=0
        result['folds'].append({'Fold':label,'causal_prefix_verified':True,'train_scalers_verified':True,'sequence_and_original_flags_verified':True,'train_common_N':len(tr),'internal_common_N':len(common['internal']),'evaluation_common_N':len(common['evaluation'])})
    for rel,sha in cfg['initial_hashes'].items():assert hashfile(ROOT/rel)==sha,rel
    result['old_files_identical']=len(cfg['initial_hashes'])
    previous=json.loads((OUT/'kontribusi_antarstasiun/config_locked.json').read_text(encoding='utf-8'))['initial_hashes']
    for rel,sha in previous.items():assert hashfile(ROOT/rel)==sha,rel
    result['old_files_union_with_stage9_baseline']=len(set(previous)|set(cfg['initial_hashes']))
    if not preflight_only:
        result['checkpoint_models_verified_without_refit']=verified_models
    if not preflight_only and not completed_only:
        assert verified_models==24
        t=json.loads((RUN/'tabel_hasil.json').read_text(encoding='utf-8'));p=pd.read_csv(RUN/'prediksi_oof_seluruh_seed.csv',parse_dates=['Timestamp target'])
        for r in t['Skor_fold_seed']+t['Gabungan_per_seed']+t['Skor_bulanan']:
            f=p[p.Seed.eq(r['Seed'])];f=f if r['Fold']=='Gabungan_OOF' else f[f.Fold.eq(r['Fold'])]
            if 'Bulan target' in r:f=f[f['Timestamp target'].dt.strftime('%Y-%m').eq(r['Bulan target'])]
            den=len(f);f=f[f['Bersama digunakan']] if r['Domain']=='bersama' else f[np.isfinite(f[r['Skenario']])]
            err=f[r['Skenario']].to_numpy()-f['SO2 target asli'].to_numpy()
            assert len(f)==r['N'] and den==r['Denominator target asli']
            assert np.allclose([r['MAE'],r['RMSE'],r['Bias']],[np.abs(err).mean(),np.sqrt(np.mean(err**2)),err.mean()],atol=1e-8)
            assert r['Prediksi negatif']==int((f[r['Skenario']]<0).sum()) and np.isclose(r['Cakupan persen'],100*len(f)/den)
        result['metrics_fold_monthly_pooled_verified']=True;result['tables_sha256']=hashfile(RUN/'tabel_hasil.json')
    output=RUN/('preflight_independent_checks.json' if preflight_only else 'partial_checkpoint_checks.json' if completed_only else 'independent_checks.json')
    output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':verify('--preflight' in sys.argv,'--completed-only' in sys.argv)
