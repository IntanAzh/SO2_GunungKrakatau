"""Verify saved sequence experiment; inference only, never training or final test."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json,math
import sys
import numpy as np
import pandas as pd
import torch
import openpyxl
from lstm_stage7 import SimpleLSTM,METHODS,ALL
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'lstm_preprocessing'
cfg=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'));checks=json.loads((RUN/'checks.json').read_text(encoding='utf-8'))
gate=json.loads((RUN/'all_gates_before_training.json').read_text(encoding='utf-8'));t=json.loads((OUT/'tabel_lstm_preprocessing.json').read_text(encoding='utf-8'))
assert not cfg['test_loaded'] and not cfg['test_evaluated'] and checks['fit_count']==36 and gate['fit_count']==0
assert all(g['Lulus'] and g['LSTM fit count sebelum gate']==0 for g in gate['gates'])
assert sha256((RUN/'config_locked.json').read_bytes()).hexdigest()==checks['config_sha256']==gate['config_sha256']
assert cfg['sequence_length']==48 and len(cfg['sequence_features'])==22 and cfg['seeds']==[11,23,47]
assert not any('Curah Hujan' in x for x in cfg['sequence_features'])
assert cfg['fold_cuts']==[5760,10732,15703,20674,25645]
oldcfg=json.loads((OUT/'backtesting_way_urang/fold_config_locked.json').read_text(encoding='utf-8'))
for key in ['covariates','horizon_slots','covariate_lag_slots','ffill_limit_slots','knn','warmup_slots','prefix_update_slots','availability','quality']:
    assert cfg[key]==oldcfg[key]
d=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=cfg['fold_cuts'][-1],parse_dates=['Timestamp'])
times=pd.DatetimeIndex(d.Timestamp);y=d['SO2 numerik'].to_numpy(dtype=float)
assert (times[1:]-times[:-1]==pd.Timedelta(minutes=30)).all() and times[-1]<pd.Timestamp(cfg['split'][2]['Awal'])
raw=np.column_stack([y]+[d[f'{c} numerik'].shift(1).to_numpy(dtype=float) for c in cfg['covariates']])
pred=pd.read_csv(RUN/'prediksi_oof_seluruh_seed.csv',parse_dates=['Origin','Timestamp target'])
torch.set_num_threads(1);manual_knn=0;inference_count=0
for number,(a,b) in enumerate(zip(cfg['fold_cuts'][:-1],cfg['fold_cuts'][1:]),1):
    label=f'Fold_{number:02d}';folder=RUN/label;c=int(np.floor(.85*a))
    parts={'inner_train':np.arange(4368,c-1),'inner_validation':np.arange(max(4368,c-1),a-1),'evaluation':np.arange(a-1,b-1)}
    shared={part:ids[np.isfinite(y[ids+1])] for part,ids in parts.items()};arrays={};eligible={}
    records=[r for r in t['Prefix_imputer'] if r['Fold']==label]
    assert all(r['Jumlah baris prefix']<=c and pd.Timestamp(r['Timestamp donor maksimum'])<pd.Timestamp(r['Awal query']) for r in records)
    for method in METHODS:
        arr=np.load(folder/f'sekuens_inputs_{method}.npz');arrays[method]=arr
        h=arr['filled'];features=arr['features'];assert h.shape==(b-1,11) and features.shape==(b-1,22)
        obs=np.isfinite(raw[:b-1]);assert np.allclose(h[obs],raw[:b-1][obs])
        assert np.array_equal(arr['raw_flags'],~np.isfinite(raw[:b-1])) and np.array_equal(features[:,11:],arr['raw_flags'])
        good=np.isfinite(h).all(axis=1);prefix=np.r_[0,np.cumsum(~good)];complete=np.zeros(b-1,bool)
        ids=np.arange(47,b-1);complete[ids]=(prefix[ids+1]-prefix[ids-47])==0
        eligible[method]={}
        for part,ids in parts.items():
            own=ids[np.isfinite(y[ids+1])&complete[ids]];eligible[method][part]=own;shared[part]=np.intersect1d(shared[part],own)
        if method=='P0_median':
            for r in records:
                start=r['Jumlah baris prefix'];stop=times.get_loc(pd.Timestamp(r['Akhir query']))+1
                expected=np.where(np.isfinite(raw[start:stop]),raw[start:stop],np.nanmedian(raw[:start],axis=0))
                assert np.allclose(h[start:stop],expected,equal_nan=True)
        if method=='P1_ffill_L4':
            last=np.full(11,-1);value=np.full(11,np.nan)
            for u,row in enumerate(raw[:b-1]):
                seen=np.isfinite(row);last[seen]=u;value[seen]=row[seen];expected=row.copy()
                if u>=4320:
                    fill=(~seen)&(last>=0)&(u-last<=4);expected[fill]=value[fill]
                assert np.allclose(h[u],expected,equal_nan=True)
        if method=='P2_knn_k10':
            for r in records:
                start=r['Jumlah baris prefix'];stop=times.get_loc(pd.Timestamp(r['Akhir query']))+1;past=raw[:start]
                mean=np.nanmean(past,axis=0);scale=np.nanstd(past,axis=0);scale[scale==0]=1;assert np.isfinite(mean).all()
                donors=(past[np.isfinite(past[:,0])]-mean)/scale
                poss=np.flatnonzero((~np.isfinite(raw[start:stop])).any(axis=1)&(np.isfinite(raw[start:stop,1:]).sum(axis=1)>=2))
                for local in poss[:20]:
                    q=(raw[start+local]-mean)/scale;overlap=np.isfinite(q)&np.isfinite(donors);count=overlap.sum(axis=1)
                    distance=np.sqrt(np.divide(np.nansum((donors-q)**2,axis=1)*len(q),count,out=np.full(len(count),np.inf),where=count>0))
                    col=int(np.flatnonzero(~np.isfinite(q))[0]);valid=np.flatnonzero(np.isfinite(donors[:,col])&np.isfinite(distance))
                    if len(valid)<11:continue
                    order=valid[np.argsort(distance[valid])];near=order[:10]
                    if distance[order[9]]==distance[order[10]]:continue
                    weights=(distance[near]==0).astype(float) if np.any(distance[near]==0) else 1/distance[near]
                    expected=np.sum(weights*donors[near,col])/weights.sum()*scale[col]+mean[col]
                    assert math.isclose(expected,h[start+local,col],rel_tol=1e-7,abs_tol=1e-7);manual_knn+=1;break
    for part,ids in shared.items():
        stored=pd.read_csv(folder/f'timestamp_{part}.csv');assert np.array_equal(stored['Index origin'],ids)
        assert np.allclose(stored['SO2 target asli'],y[ids+1])
        for arr in arrays.values():assert np.array_equal(arr[{'inner_train':'train_origins','inner_validation':'internal_origins','evaluation':'eval_origins'}[part]],ids)
    train=shared['inner_train'];assert (train+1<c).all() and ((shared['inner_validation']+1>=c)&(shared['inner_validation']+1<a)).all()
    slots=np.unique((train[:,None]-np.arange(48)[None,:]).ravel());assert slots.max()<c
    target_mean=y[train+1].mean();target_scale=y[train+1].std()
    for method in METHODS:
        arr=arrays[method];assert np.array_equal(arr['scale_slots'],slots)
        feature_values=arr['features']
        mean=arr['filled'][slots].mean(axis=0);scale=arr['filled'][slots].std(axis=0);scale[scale<1e-12]=1
        pars=pd.DataFrame([r for r in t['Scaler'] if r['Fold']==label and r['Pipeline']==method])
        assert np.allclose(mean,pars.Mean) and np.allclose(scale,pars.Scale)
        assert np.allclose(target_mean,pars['Target mean']) and np.allclose(target_scale,pars['Target scale'])
        assert np.allclose(feature_values[:,:11],(arr['filled']-mean)/scale,equal_nan=True,atol=1e-5,rtol=1e-6)
        for seed in cfg['seeds']:
            frame=pred[(pred.Fold==label)&(pred.Seed==seed)].sort_values('Timestamp target');origin=frame['Index origin'].to_numpy()
            assert np.array_equal(frame['Index target'],np.flatnonzero(np.isfinite(y[a:b]))+a)
            assert np.array_equal(origin+1,frame['Index target']) and np.allclose(frame['SO2 target asli'],y[origin+1])
            assert np.array_equal(frame['Irisan LSTM persistence'],np.isin(origin,shared['evaluation']))
            last=np.maximum.accumulate(np.where(np.isfinite(y[:b-1]),np.arange(b-1),-1))
            assert np.allclose(frame.Persistence,y[last[origin]]) and np.array_equal(frame['Usia donor persistence slot'],origin-last[origin])
            rows=next(r for r in t['Training_ringkas'] if r['Fold']==label and r['Pipeline']==method and r['Seed']==seed)
            history=[r for r in t['Training_epoch'] if r['Fold']==label and r['Pipeline']==method and r['Seed']==seed]
            best=int(np.argmin([r['Internal MSE standar'] for r in history]))+1
            assert best==rows['Epoch terbaik'] and len(history)==rows['Epoch dijalankan'] and len(history)<=50
            assert len(history)==50 or len(history)-best==8
            checkpoint=folder/f'LSTM_{method}_seed{seed}.pt';assert sha256(checkpoint.read_bytes()).hexdigest()==rows['Checkpoint SHA256']
            model=SimpleLSTM();assert sum(p.numel() for p in model.parameters())==7201
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
            internal_ids=shared['inner_validation'];internal_y=torch.tensor((y[internal_ids+1]-target_mean)/target_scale,dtype=torch.float32)
            val_sse=0.0
            with torch.no_grad():
                for start in range(0,len(internal_ids),256):
                    ix=internal_ids[start:start+256];vx=feature_values[ix[:,None]-np.arange(47,-1,-1)[None,:]]
                    val_sse+=float(torch.sum((model(torch.from_numpy(vx))-internal_y[start:start+256])**2))
            assert math.isclose(val_sse/len(internal_ids),rows['Internal MSE standar terbaik'],rel_tol=1e-5,abs_tol=1e-9)
            good=np.isin(origin,eligible[method]['evaluation']);valid=np.flatnonzero(good);reconstructed=[]
            with torch.no_grad():
                for start in range(0,len(valid),256):
                    origins=origin[valid[start:start+256]];x=feature_values[origins[:,None]-np.arange(47,-1,-1)[None,:]]
                    reconstructed.extend((model(torch.from_numpy(x)).numpy()*target_scale+target_mean).tolist())
            assert np.allclose(reconstructed,frame.loc[good,method],atol=1e-5,rtol=1e-6)
            assert frame.loc[~good,method].isna().all();inference_count+=1
assert manual_knn>=4 and inference_count==36
for records in [t['Metrik_fold_seed'],t['Ringkasan'],t['Ridge_timestamp_identik'],t['Bulanan']]:
    for r in records:
        frame=pred[pred.Seed==r['Seed']];frame=frame if r['Fold']=='Gabungan_OOF' else frame[frame.Fold==r['Fold']]
        part=np.ones(len(frame),bool)
        if 'Bulan' in r:part=frame['Timestamp target'].dt.to_period('M').astype(str).eq(r['Bulan']).to_numpy()
        mask=np.isfinite(frame[r['Pipeline']]).to_numpy() if r['Domain']=='mandiri' else frame['Irisan LSTM persistence' if r['Domain']=='LSTM bersama' else 'Irisan LSTM Ridge persistence'].to_numpy()
        mask=mask&part;e=frame.loc[mask,r['Pipeline']].to_numpy()-frame.loc[mask,'SO2 target asli'].to_numpy()
        assert len(e)==r['N'] and int(part.sum())==r['Denominator target asli']
        assert np.allclose([np.abs(e).mean(),np.sqrt(np.mean(e**2)),e.mean()],[r['MAE'],r['RMSE'],r['Bias']],atol=1e-8)
        assert int((frame.loc[mask,r['Pipeline']]<0).sum())==r['Prediksi negatif']
        assert sha256('\n'.join(frame.loc[mask,'Timestamp target'].astype(str)).encode()).hexdigest()==r['SHA256 timestamp']
for r in t['Variasi_seed']:
    source=t['Metrik_fold_seed'] if r['Jenis']=='fold' else t['Ringkasan'] if r['Jenis']=='gabungan' else t['Ridge_timestamp_identik']
    rows=[x for x in source if x['Fold']==r['Fold'] and x['Pipeline']==r['Pipeline'] and x['Domain']==r['Domain']]
    for col in ['MAE','RMSE','Bias','Cakupan persen','N','Prediksi negatif']:
        assert math.isclose(np.mean([x[col] for x in rows]),r[col+' mean seed'],abs_tol=1e-10)
        assert math.isclose(np.std([x[col] for x in rows],ddof=1),r[col+' SD seed'],abs_tol=1e-10)
discrepancies=json.loads((RUN/'resume_existing_discrepancies.json').read_text(encoding='utf-8')) if (RUN/'resume_existing_discrepancies.json').exists() else {}
for r in t['Integritas_SHA256']:
    actual=sha256((ROOT/r['File']).read_bytes()).hexdigest();assert actual==r['SHA256 sesudah'],r['File']
    if r['Identik']:assert actual==r['SHA256 sebelum'],r['File']
    else:assert r['File'] in discrepancies and actual==discrepancies[r['File']]['SHA256 sebelum resume']
if '--audit-only' in sys.argv:
    result={'completed_checkpoints_verified':inference_count,'manual_knn_prefix_checks':manual_knn,
        'sequence_shape_verified':[48,22],'causal_preprocessing_and_inner_scalers':True,'early_stopping_best_weights_verified':True,
        'all_metrics_recomputed':True,'target_inverse_verified':True,'test_loaded':False,
        'old_hashes_identical':sum(r['Identik'] for r in t['Integritas_SHA256']),'preexisting_notebook_discrepancies':len(discrepancies),
        'tables_sha256':sha256((OUT/'tabel_lstm_preprocessing.json').read_bytes()).hexdigest()}
    (RUN/'independent_model_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result));sys.exit(0)
w=openpyxl.load_workbook(OUT/'hasil_lstm_preprocessing.xlsx',read_only=True,data_only=False)
for name,records in t.items():
    rows=list(w[name].iter_rows(values_only=True));assert len(rows)==len(records)+1,(name,len(rows),len(records))
    for actual,expected in zip(rows[1:],records):
        for key,value in zip(rows[0],actual):
            exp=expected.get(key)
            if exp=='' and value is None:continue
            if isinstance(value,datetime):assert value.isoformat(sep=' ')[:19]==str(exp).replace('T',' ')[:19]
            elif isinstance(value,(int,float)) and not isinstance(value,bool) and isinstance(exp,(int,float)):assert math.isclose(value,exp,rel_tol=1e-10,abs_tol=1e-8)
            else:assert value==exp,(name,key,value,exp)
w.close()
nb=json.loads((ROOT/'notebooks/07_lstm_preprocessing.ipynb').read_text(encoding='utf-8'))
assert all(c['execution_count'] is not None and not any(o['output_type']=='error' for o in c['outputs']) for c in nb['cells'] if c['cell_type']=='code')
print(f'Lolos: 36checkpoint, 48slot×22kanal, innertrain-only scaler, {manual_knn}prefixKNN manual, purge/irisan/persistence, seluruh skor/inverse/SDseed, {len(t)}sheet; {sum(r["Identik"] for r in t["Integritas_SHA256"])}hash lama identik, {len(discrepancies)}perbedaan notebook sebelumresume stabil; test tidak dimuat.')
