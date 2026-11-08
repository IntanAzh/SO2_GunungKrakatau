"""Resume interrupted fixed experiment; never refit completed models or preprocessing."""
import json,copy,random
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from lstm_stage7 import ROOT,OUT,RUN,METHODS,ALL,Experiment,SequenceFold,SimpleLSTM,dist
from audit_quality import hashfile
from pilot_way_urang import dump

def restore_experiment():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    e=Experiment.__new__(Experiment);e.cfg=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'))
    e.config_hash=hashfile(RUN/'config_locked.json');e.cuts=e.cfg['fold_cuts'];e.before=e.cfg['initial_hashes'];e.refs=[ROOT/p for p in e.before]
    e.preexisting_diffs={p:{'SHA256 awal eksperimen':v,'SHA256 sebelum resume':hashfile(ROOT/p)} for p,v in e.before.items() if hashfile(ROOT/p)!=v}
    assert set(e.preexisting_diffs).issubset({'notebooks\\02_audit_kualitas_data.ipynb'}),e.preexisting_diffs
    discrepancy_path=RUN/'resume_existing_discrepancies.json'
    if discrepancy_path.exists():assert json.loads(discrepancy_path.read_text(encoding='utf-8'))==e.preexisting_diffs
    else:dump(discrepancy_path,e.preexisting_diffs)
    e.data=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=e.cuts[-1],parse_dates=['Timestamp'])
    e.times=pd.DatetimeIndex(e.data.Timestamp);e.y=e.data['SO2 numerik'].to_numpy(dtype=float)
    e.matrix=np.column_stack([e.y]+[e.data[f'{c} numerik'].shift(1).to_numpy(dtype=float) for c in e.cfg['covariates']])
    e.plan=json.loads((RUN/'rencana_fold_sebelum_training.json').read_text(encoding='utf-8'));e.fit_count=0;e.folds=[]
    e.restored=True
    resume_hash_path=RUN/'resume_initial_hashes.json'
    if resume_hash_path.exists():e.resume_hashes=json.loads(resume_hash_path.read_text(encoding='utf-8'))
    else:
        e.resume_hashes={str(p.relative_to(ROOT)):hashfile(p) for p in RUN.glob('Fold_*/*') if p.is_file()}
        dump(resume_hash_path,e.resume_hashes)
    assert all(hashfile(ROOT/p)==v for p,v in e.resume_hashes.items())
    for number,(a,b) in enumerate(zip(e.cuts[:-1],e.cuts[1:]),1):
        f=SequenceFold(e,number,a,b);f.filled={};f.features={};f.eligible={};f.shared={};f.parts={'inner_train':np.arange(4368,f.a-1),
            'inner_validation':np.arange(max(4368,f.a-1),a-1),'evaluation':np.arange(a-1,b-1)}
        f.flow=[];f.losses=[];f.loss_detail=[];f.distributions=[];f.monthly_dist=[]
        for method in METHODS:
            z=np.load(f.folder/f'sekuens_inputs_{method}.npz');f.filled[method]=z['filled'];f.features[method]=z['features'];f.raw_flags=z['raw_flags'];f.scale_slots=z['scale_slots']
            complete=pd.Series(np.isfinite(f.filled[method]).all(axis=1)).rolling(48,min_periods=48).sum().eq(48).to_numpy()
            f.eligible[method]={part:ids[np.isfinite(f.y[ids+1])&complete[ids]] for part,ids in f.parts.items()}
        for part,ids in f.parts.items():
            obs=ids[np.isfinite(f.y[ids+1])];shared=obs
            for method in METHODS:shared=np.intersect1d(shared,f.eligible[method][part])
            stored=pd.read_csv(f.folder/f'timestamp_{part}.csv');assert np.array_equal(shared,stored['Index origin']);f.shared[part]=shared
            for method in METHODS:
                own=f.eligible[method][part];lost=np.setdiff1d(own,shared)
                f.flow.append({'Fold':f.label,'Bagian':part,'Pipeline':method,'Kandidat origin setelah buffer purge':len(ids),'Target asli teramati':len(obs),
                    'Sekuens finite mandiri':len(own),'Sampel bersama':len(shared),'Keluar karena irisan':len(lost),'Tidak layak karena sekuens':len(obs)-len(own)})
                f.losses.append({'Fold':f.label,'Bagian':part,'Pipeline':method,**dist(f.y[lost+1])})
                for mo in f.times[ids+1].to_period('M').unique():
                    sub=lost[f.times[lost+1].to_period('M')==mo];f.monthly_dist.append({'Fold':f.label,'Bagian':part,'Domain':'keluar irisan '+method,'Bulan':str(mo),**dist(f.y[sub+1])})
            for domain,selected in [('target asli kandidat',obs),('bersama digunakan',shared)]:
                f.distributions.append({'Fold':f.label,'Bagian':part,'Domain':domain,'Awal target':f.times[selected[0]+1] if len(selected) else None,
                    'Akhir target':f.times[selected[-1]+1] if len(selected) else None,**dist(f.y[selected+1])})
                for mo in f.times[ids+1].to_period('M').unique():
                    sub=selected[f.times[selected+1].to_period('M')==mo];f.monthly_dist.append({'Fold':f.label,'Bagian':part,'Domain':domain,'Bulan':str(mo),**dist(f.y[sub+1])})
        f.gate=json.loads((f.folder/'gate_sebelum_LSTM.json').read_text(encoding='utf-8'));f.scaler_records=json.loads((f.folder/'parameter_scaler.json').read_text(encoding='utf-8'))
        f.target_mean=f.scaler_records[0]['Target mean'];f.target_scale=f.scaler_records[0]['Target scale'];f.impute_seconds=None;f.prefix_logs=[]
        for start in list(range(4320,f.a,1440))+[f.a]:
            stop=min(start+1440,f.a) if start<f.a else f.origstop
            if start>=stop:continue
            donors=np.flatnonzero(np.isfinite(f.matrix[:start,0]));query=f.matrix[start:stop]
            f.prefix_logs.append({'Awal query':f.times[start],'Akhir query':f.times[stop-1],'Fit sampai eksklusif':f.times[start],
                'Timestamp donor maksimum':f.times[donors[-1]],'Jumlah baris prefix':start,'Jumlah donor SO2 asli':len(donors),'Jumlah query':stop-start,
                'Query KNN ada sel kosong dan dukungan':int(((np.isfinite(query[:,1:]).sum(axis=1)>=2)&(~np.isfinite(query)).any(axis=1)).sum()),
                'Fit imputer dan scaler detik':None,'Transform KNN detik':None,'Periode':'train prefix' if start<f.a else 'validation frozen train',
                'Median SO2 prefix':float(np.nanmedian(f.matrix[:start,0])),'Provenansi':'Cutoff/count direkonstruksi dari konfigurasi dan input tersimpan, tanpa fit; timing hilang saat proses berhenti.'})
        e.folds.append(f)
    e.fit_count=len(list(RUN.glob('Fold_*/*.pt')))
    print('Restore tanpa fit. Checkpoint selesai:',e.fit_count,'; konfigurasi',e.config_hash,flush=True)
    return e

def continue_models(e):
    started=e.fit_count
    for f in e.folds:
        if (f.folder/'training_summary.json').exists():
            f.training=json.loads((f.folder/'training_summary.json').read_text(encoding='utf-8'))
            f.history=[r for seed in e.cfg['seeds'] for m in METHODS for r in json.loads((f.folder/f'history_{m}_seed{seed}.json').read_text(encoding='utf-8'))]
            f.predictions=[pd.read_csv(f.folder/f'prediksi_seed{seed}.csv',parse_dates=['Origin','Timestamp target']) for seed in e.cfg['seeds']]
            print('Tidak fit ulang:',f.label,flush=True);continue
        f.training=[];f.history=[];f.predictions=[];train=f.shared['inner_train'];inner=f.shared['inner_validation']
        targetids=np.arange(f.outer_a,f.b);targetids=targetids[np.isfinite(f.y[targetids])];origins=targetids-1
        last=np.maximum.accumulate(np.where(np.isfinite(f.y[:f.origstop]),np.arange(f.origstop),-1))
        for seed in e.cfg['seeds']:
            frame=pd.DataFrame({'Fold':f.label,'Seed':seed,'Origin':f.times[origins],'Timestamp target':f.times[targetids],'Index origin':origins,
                'Index target':targetids,'SO2 target asli':f.y[targetids],'SO2 asli origin':f.y[origins],'Persistence':f.y[last[origins]],'Usia donor persistence slot':origins-last[origins]})
            for method in METHODS:
                checkpoint=f.folder/f'LSTM_{method}_seed{seed}.pt';histpath=f.folder/f'history_{method}_seed{seed}.json'
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);model=SimpleLSTM()
                reused=checkpoint.exists() and histpath.exists()
                if reused:
                    history=json.loads(histpath.read_text(encoding='utf-8'));best_epoch=int(np.argmin([r['Internal MSE standar'] for r in history]))+1
                    epoch=len(history);best=history[best_epoch-1]['Internal MSE standar'];fit_seconds=None
                    assert epoch==50 or epoch-best_epoch==8
                    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));print('Checkpoint dipertahankan:',checkpoint.name,flush=True)
                else:
                    optimizer=torch.optim.Adam(model.parameters(),lr=.001);lossfn=nn.MSELoss()
                    X=torch.from_numpy(f.sequences(method,train));Y=torch.tensor((f.y[train+1]-f.target_mean)/f.target_scale,dtype=torch.float32)
                    VX=torch.from_numpy(f.sequences(method,inner));VY=torch.tensor((f.y[inner+1]-f.target_mean)/f.target_scale,dtype=torch.float32)
                    best=float('inf');state=None;best_epoch=0;stale=0;history=[];tick=perf_counter()
                    for epoch in range(1,51):
                        model.train();order=torch.from_numpy(np.random.default_rng(seed+epoch*100003).permutation(len(train)));sumloss=0.0
                        for begin in range(0,len(train),64):
                            ix=order[begin:begin+64];optimizer.zero_grad(set_to_none=True);prediction=model(X[ix]);loss=lossfn(prediction,Y[ix]);loss.backward();optimizer.step();sumloss+=float(loss.detach())*len(ix)
                        model.eval();vsum=0.0
                        with torch.no_grad():
                            for begin in range(0,len(inner),256):
                                z=model(VX[begin:begin+256]);vsum+=float(torch.sum((z-VY[begin:begin+256])**2))
                        vloss=vsum/len(inner)
                        if not np.isfinite(vloss):raise RuntimeError('Nonfinite loss, tidak ada fallback.')
                        history.append({'Fold':f.label,'Pipeline':method,'Seed':seed,'Epoch':epoch,'Train MSE standar':sumloss/len(train),'Internal MSE standar':vloss})
                        if vloss<best:best=vloss;state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
                        else:stale+=1
                        if stale>=8:break
                    fit_seconds=perf_counter()-tick;model.load_state_dict(state);torch.save(state,checkpoint);dump(histpath,history);e.fit_count+=1
                f.history.extend(history);model.eval();good=np.isin(origins,f.eligible[method]['evaluation']);pred=np.full(len(origins),np.nan);eval_ids=np.flatnonzero(good);tick=perf_counter()
                with torch.no_grad():
                    for begin in range(0,len(eval_ids),256):
                        rows=eval_ids[begin:begin+256];out=model(torch.from_numpy(f.sequences(method,origins[rows]))).numpy();pred[rows]=out*f.target_scale+f.target_mean
                predict_seconds=perf_counter()-tick;frame[method]=pred
                f.training.append({'Fold':f.label,'Pipeline':method,'Seed':seed,'Epoch dijalankan':epoch,'Epoch terbaik':best_epoch,'Internal MSE standar terbaik':best,
                    'Fit dan early stopping detik':fit_seconds,'Predict detik':predict_seconds,'N train bersama':len(train),'N early stopping bersama':len(inner),'Jumlah parameter':7201,
                    'Checkpoint SHA256':hashfile(checkpoint),'Resume':'checkpoint selesai sebelum interupsi; waktu fit tidak tersimpan' if reused else 'diselesaikan dengan konfigurasi terkunci setelah interupsi'})
                print('Selesai',f.label,method,seed,'epoch',epoch,'best',best_epoch,'fit detik',fit_seconds,flush=True)
            frame['Irisan LSTM persistence']=np.isfinite(frame[ALL]).all(axis=1);frame.to_csv(f.folder/f'prediksi_seed{seed}.csv',index=False);f.predictions.append(frame)
        dump(f.folder/'training_summary.json',f.training)
    assert e.fit_count==36 and hashfile(RUN/'config_locked.json')==e.config_hash
    for path,digest in e.resume_hashes.items():assert hashfile(ROOT/path)==digest,path
    e.pred=pd.concat([frame for f in e.folds for frame in f.predictions],ignore_index=True);e.pred.to_csv(RUN/'prediksi_oof_seluruh_seed.csv',index=False)
    e.history=[r for f in e.folds for r in f.history];e.training=[r for f in e.folds for r in f.training]
    dump(RUN/'training_history.json',e.history);dump(RUN/'training_summary.json',e.training)
    if not (RUN/'resume_log.json').exists():
        dump(RUN/'resume_log.json',{'checkpoint_before':started,'checkpoint_after':e.fit_count,'models_newly_completed':36-started,
            'all_pre_resume_artifacts_identical':True,'config_sha256':e.config_hash,'timing_unavailable':'Preprocessing4fold dan fit1model checkpointFold4P0seed11 hilang bersama memori proses; tidak diestimasi atau diisi0.'})
    return e

if __name__=='__main__':
    e=restore_experiment();continue_models(e);e.summarize();e.export_excel()
