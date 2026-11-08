"""Fixed single-LSTM experiment. Development only, inner chronological stopping."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from importlib.metadata import version
import json,os,sys,platform,random,subprocess,copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import forecast_ridge_way_urang as histbase
from audit_quality import hashfile
from pilot_way_urang import dump
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'lstm_preprocessing';FIG=OUT/'figures/lstm_preprocessing'
METHODS=['P0_median','P1_ffill_L4','P2_knn_k10'];ALL=METHODS+['Persistence']
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def dist(values):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    return {'N':len(x),'Minimum':float(x.min()) if len(x) else None,'Median':float(np.median(x)) if len(x) else None,
        'Maksimum':float(x.max()) if len(x) else None,'Mean':float(x.mean()) if len(x) else None,
        'SD':float(x.std(ddof=1)) if len(x)>1 else None,'SO2 lebih2.14':int((x>2.14).sum())}

def metric(frame,method,mask,fold,seed,domain,den):
    from hashlib import sha256
    e=frame.loc[mask,method].to_numpy()-frame.loc[mask,'SO2 target asli'].to_numpy();n=len(e)
    return {'Fold':fold,'Seed':seed,'Pipeline':method,'Domain':domain,'N':n,'Denominator target asli':int(den),
        'Cakupan persen':100*n/den if den else None,'MAE':float(np.abs(e).mean()) if n else None,
        'RMSE':float(np.sqrt(np.mean(e**2))) if n else None,'Bias':float(e.mean()) if n else None,
        'Prediksi negatif':int((frame.loc[mask,method]<0).sum()),
        'SHA256 timestamp':sha256('\n'.join(frame.loc[mask,'Timestamp target'].astype(str)).encode()).hexdigest()}

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__();self.lstm=nn.LSTM(22,32,num_layers=1,batch_first=True);self.output=nn.Linear(32,1)
    def forward(self,x):
        out,_=self.lstm(x);return self.output(out[:,-1,:]).squeeze(-1)

class SequenceFold(histbase.Forecast):
    def __init__(self,parent,number,a,b):
        self.parent=parent;self.number=number;self.label=f'Fold_{number:02d}';self.outer_a=a;self.b=b
        self.a=int(np.floor(.85*a));self.folder=RUN/self.label;self.folder.mkdir(exist_ok=True)
        self.times=parent.times[:b];self.y=parent.y[:b];self.matrix=parent.matrix[:b].copy();self.origstop=b-1
        self.cov=parent.cfg['covariates'];self.lags=parent.cfg['so2_lags'];self.warmup=4320;self.batch=1440;self.first_origin=4368
    def prepare_sequences(self):
        previous=histbase.RUN
        try:
            histbase.RUN=self.folder
            # Reuse audited causal history algorithm. Tabular X is NOT sequence input.
            with threadpool_limits(limits=1):super().impute_history()
        finally:histbase.RUN=previous
        self.X.clear();self.features={};self.eligible={};self.scalers={};self.scaler_records=[]
        self.parts={'inner_train':np.arange(self.first_origin,self.a-1),
            'inner_validation':np.arange(max(self.first_origin,self.a-1),self.outer_a-1),
            'evaluation':np.arange(self.outer_a-1,self.b-1)}
        self.shared={};self.flow=[];self.losses=[];self.loss_detail=[];self.distributions=[];self.monthly_dist=[]
        self.raw_flags=(~np.isfinite(self.matrix[:self.origstop])).astype(float)
        for method in METHODS:
            h=self.filled[method];good=np.isfinite(h).all(axis=1)
            # A 48-slot window cannot cross an unresolved slot. The clock is intact.
            total=np.r_[0,np.cumsum(~good)];complete=np.zeros(len(h),bool)
            end=np.arange(47,len(h));complete[end]=(total[end+1]-total[end-47])==0
            self.eligible[method]={}
            for part,ids in self.parts.items():self.eligible[method][part]=ids[np.isfinite(self.y[ids+1])&complete[ids]]
        for part,ids in self.parts.items():
            observed=ids[np.isfinite(self.y[ids+1])];shared=observed
            for method in METHODS:shared=np.intersect1d(shared,self.eligible[method][part])
            self.shared[part]=shared
            for method in METHODS:
                own=self.eligible[method][part];lost=np.setdiff1d(own,shared)
                self.flow.append({'Fold':self.label,'Bagian':part,'Pipeline':method,'Kandidat origin setelah buffer purge':len(ids),
                    'Target asli teramati':len(observed),'Sekuens finite mandiri':len(own),'Sampel bersama':len(shared),
                    'Keluar karena irisan':len(lost),'Tidak layak karena sekuens':len(observed)-len(own)})
                self.losses.append({'Fold':self.label,'Bagian':part,'Pipeline':method,**dist(self.y[lost+1])})
                for idx in lost:self.loss_detail.append({'Fold':self.label,'Bagian':part,'Pipeline':method,'Index origin':int(idx),
                    'Origin':self.times[idx],'Timestamp target':self.times[idx+1],'SO2 target asli':self.y[idx+1]})
                for mo in self.times[ids+1].to_period('M').unique():
                    sub=lost[self.times[lost+1].to_period('M')==mo]
                    self.monthly_dist.append({'Fold':self.label,'Bagian':part,'Domain':'keluar irisan '+method,'Bulan':str(mo),**dist(self.y[sub+1])})
            for domain,selected in [('target asli kandidat',observed),('bersama digunakan',shared)]:
                row={'Fold':self.label,'Bagian':part,'Domain':domain,'Awal target':self.times[selected[0]+1] if len(selected) else None,
                    'Akhir target':self.times[selected[-1]+1] if len(selected) else None,**dist(self.y[selected+1])}
                self.distributions.append(row)
                for mo in self.times[ids+1].to_period('M').unique():
                    sub=selected[self.times[selected+1].to_period('M')==mo]
                    self.monthly_dist.append({'Fold':self.label,'Bagian':part,'Domain':domain,'Bulan':str(mo),**dist(self.y[sub+1])})
            pd.DataFrame({'Index origin':shared,'Origin':self.times[shared],'Timestamp target':self.times[shared+1],
                'SO2 target asli':self.y[shared+1]}).to_csv(self.folder/f'timestamp_{part}.csv',index=False)
        train=self.shared['inner_train'];val=self.shared['inner_validation']
        self.gate={'Fold':self.label,'Sampel train bersama':len(train),'Sampel early stopping bersama':len(val),
            'Sampel evaluasi bersama':len(self.shared['evaluation']),'Lulus':bool(len(train)>=64 and len(val)>0 and len(self.shared['evaluation'])>0),
            'LSTM fit count sebelum gate':self.parent.fit_count,'Internal train batas eksklusif':self.times[self.a],
            'Origin purge internal':self.times[self.a-1],'Target purge internal':self.times[self.a],
            'Origin purge outer train':self.times[self.outer_a-1],'Target purge outer train':self.times[self.outer_a]}
        assert (train+1<self.a).all() and ((val+1>=self.a)&(val+1<self.outer_a)).all()
        # Fit scaler on UNIQUE input slots used in common inner-training windows.
        slots=np.unique((train[:,None]-np.arange(48)[None,:]).ravel());assert slots.max()<self.a
        self.scale_slots=slots
        self.target_mean=float(self.y[train+1].mean());self.target_scale=float(self.y[train+1].std());self.target_scale=self.target_scale or 1.0
        for method in METHODS:
            h=self.filled[method];scaler=StandardScaler().fit(h[slots]);scaled=scaler.transform(h)
            self.features[method]=np.column_stack([scaled,self.raw_flags]).astype(np.float32);self.scalers[method]=scaler
            for j,name in enumerate(self.parent.cfg['sequence_numeric_features']):
                self.scaler_records.append({'Fold':self.label,'Pipeline':method,'Fitur':name,'Mean':float(scaler.mean_[j]),
                    'Scale':float(scaler.scale_[j]),'Jumlah slot input unik train':len(slots),'Slot maksimum scaler':int(slots.max()),
                    'Timestamp maksimum scaler':self.times[slots.max()],'Target mean':self.target_mean,'Target scale':self.target_scale})
            np.savez_compressed(self.folder/f'sekuens_inputs_{method}.npz',features=self.features[method],scale_slots=slots,
                filled=h,raw_flags=self.raw_flags,train_origins=train,internal_origins=val,eval_origins=self.shared['evaluation'])
        dump(self.folder/'gate_sebelum_LSTM.json',self.gate)
        dump(self.folder/'parameter_scaler.json',self.scaler_records)
        pd.DataFrame(self.loss_detail).to_csv(self.folder/'timestamp_keluar_irisan.csv',index=False)
        print('Gate',self.gate,flush=True)
        return self
    def sequences(self,method,origins):
        return self.features[method][origins[:,None]-np.arange(47,-1,-1)[None,:]]
    def train_models(self):
        train=self.shared['inner_train'];inner=self.shared['inner_validation']
        self.predictions=[];self.history=[];self.training=[]
        targetids=np.arange(self.outer_a,self.b);targetids=targetids[np.isfinite(self.y[targetids])];origins=targetids-1
        last=np.maximum.accumulate(np.where(np.isfinite(self.y[:self.origstop]),np.arange(self.origstop),-1))
        for seed in self.parent.cfg['seeds']:
            frame=pd.DataFrame({'Fold':self.label,'Seed':seed,'Origin':self.times[origins],'Timestamp target':self.times[targetids],
                'Index origin':origins,'Index target':targetids,'SO2 target asli':self.y[targetids],
                'SO2 asli origin':self.y[origins],'Persistence':self.y[last[origins]],'Usia donor persistence slot':origins-last[origins]})
            for method in METHODS:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                model=SimpleLSTM();assert sum(p.numel() for p in model.parameters())==7201
                optimizer=torch.optim.Adam(model.parameters(),lr=.001);lossfn=nn.MSELoss()
                X=torch.from_numpy(self.sequences(method,train));Y=torch.tensor((self.y[train+1]-self.target_mean)/self.target_scale,dtype=torch.float32)
                VX=torch.from_numpy(self.sequences(method,inner));VY=torch.tensor((self.y[inner+1]-self.target_mean)/self.target_scale,dtype=torch.float32)
                best=float('inf');state=None;best_epoch=0;stale=0;tick=perf_counter()
                for epoch in range(1,51):
                    model.train();order=torch.from_numpy(np.random.default_rng(seed+epoch*100003).permutation(len(train)))
                    sumloss=0.0
                    for begin in range(0,len(train),64):
                        ix=order[begin:begin+64];optimizer.zero_grad(set_to_none=True);prediction=model(X[ix]);loss=lossfn(prediction,Y[ix]);loss.backward();optimizer.step()
                        sumloss+=float(loss.detach())*len(ix)
                    model.eval();vsum=0.0
                    with torch.no_grad():
                        for begin in range(0,len(inner),256):
                            z=model(VX[begin:begin+256]);vsum+=float(torch.sum((z-VY[begin:begin+256])**2))
                    vloss=vsum/len(inner)
                    if not np.isfinite(vloss):raise RuntimeError(f'Nonfinite loss {self.label} {method} {seed}; tidak ada fallback')
                    self.history.append({'Fold':self.label,'Pipeline':method,'Seed':seed,'Epoch':epoch,'Train MSE standar':sumloss/len(train),'Internal MSE standar':vloss})
                    if vloss<best:best=vloss;state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
                    else:stale+=1
                    if stale>=8:break
                fit_seconds=perf_counter()-tick;model.load_state_dict(state);model.eval()
                checkpoint=self.folder/f'LSTM_{method}_seed{seed}.pt';torch.save(state,checkpoint)
                good=np.isin(origins,self.eligible[method]['evaluation']);pred=np.full(len(origins),np.nan)
                eval_ids=np.flatnonzero(good);tick=perf_counter()
                with torch.no_grad():
                    for begin in range(0,len(eval_ids),256):
                        rows=eval_ids[begin:begin+256];out=model(torch.from_numpy(self.sequences(method,origins[rows]))).numpy()
                        pred[rows]=out*self.target_scale+self.target_mean
                predict_seconds=perf_counter()-tick;frame[method]=pred
                self.training.append({'Fold':self.label,'Pipeline':method,'Seed':seed,'Epoch dijalankan':epoch,'Epoch terbaik':best_epoch,
                    'Internal MSE standar terbaik':best,'Fit dan early stopping detik':fit_seconds,'Predict detik':predict_seconds,
                    'N train bersama':len(train),'N early stopping bersama':len(inner),'Jumlah parameter':7201,
                    'Checkpoint SHA256':hashfile(checkpoint)})
                self.parent.fit_count+=1
                dump(self.folder/f'history_{method}_seed{seed}.json',[r for r in self.history if r['Pipeline']==method and r['Seed']==seed])
                print('Selesai',self.label,method,'seed',seed,'epoch',epoch,'best',best_epoch,'detik',round(fit_seconds,2),flush=True)
            frame['Irisan LSTM persistence']=np.isfinite(frame[ALL]).all(axis=1)
            assert np.array_equal(origins[frame['Irisan LSTM persistence']],self.shared['evaluation'])
            frame.to_csv(self.folder/f'prediksi_seed{seed}.csv',index=False);self.predictions.append(frame)
        dump(self.folder/'training_summary.json',self.training)
        return self

class Experiment:
    def prepare(self):
        RUN.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        self.prior=json.loads((OLD:=OUT/'backtesting_way_urang/fold_config_locked.json').read_text(encoding='utf-8'))
        self.prior_results=json.loads((OUT/'tabel_backtesting_preprocessing.json').read_text(encoding='utf-8'))
        self.diagnosis=json.loads((OUT/'tabel_diagnosis_fold1.json').read_text(encoding='utf-8'))
        self.read_reports=[(OUT/f).read_text(encoding='utf-8') for f in ['ringkasan_backtesting_preprocessing.md','diagnosis_fold1.md']]
        self.cuts=self.prior['fold_cuts'];self.data=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=self.cuts[-1],parse_dates=['Timestamp'])
        self.times=pd.DatetimeIndex(self.data.Timestamp);self.y=self.data['SO2 numerik'].to_numpy(dtype=float)
        self.matrix=np.column_stack([self.y]+[self.data[f'{c} numerik'].shift(1).to_numpy(dtype=float) for c in self.prior['covariates']])
        assert self.times.max()<pd.Timestamp(self.prior['split'][2]['Awal'])
        self.refs=[p for p in OUT.rglob('*') if p.is_file() and not any(v.startswith('.') for v in p.relative_to(OUT).parts)
            and 'lstm_preprocessing' not in p.relative_to(OUT).parts and not p.name.startswith(('hasil_lstm_preprocessing','ringkasan_lstm_preprocessing','tabel_lstm_preprocessing')) and not p.name.endswith('.inspect.ndjson')]
        self.refs+=list((ROOT/'data_raw').glob('*.xlsx'))+[p for p in (ROOT/'notebooks').glob('*.ipynb') if not p.name.startswith('07_')]
        self.refs += [ROOT/'requirements.txt',ROOT/'.venv/pyvenv.cfg',ROOT/'scripts/backtesting_stage6.py',ROOT/'scripts/forecast_ridge_way_urang.py']
        self.before={str(p.relative_to(ROOT)):hashfile(p) for p in self.refs}
        numeric=['SO2_origin']+[f'{c}_lag1' for c in self.prior['covariates']]
        self.cfg={**self.prior,'experiment':'Tahap7 satu LSTM32, development validation','created_utc':datetime.now(timezone.utc).isoformat(),
            'framework':'PyTorch','device':'cpu','torch_threads':1,'deterministic_algorithms':True,'seeds':[11,23,47],
            'sequence_length':48,'sequence_range':'origin-47 sampai origin inklusif; durasi bentang23.5jam,48slot30menit',
            'sequence_numeric_features':numeric,'sequence_features':numeric+['missing_asli_'+c for c in numeric],
            'tabular_so2_lags_used_as_sequence_channels':False,'internal_split':'floor(0.85*slot_train_fold) menjadi batas target validasi internal; tanpa memilih konsentrasi',
            'imputer_fit_scope':'Prefix kausal inner train; setelah batas inner train dibekukan termasuk evaluasi outer. Ffill tetap memakai pengamatan asli terdahulu yang sudah tersedia.',
            'feature_scaling':'StandardScaler11kanal numerik pada slot input UNIK yang dipakai sekuens train bersama internal;11indikator tetap0/1.',
            'target_scaling':'z=(target-mean_train_bersama_internal)/SDpopulasi_train; MSE training di satuan standar; prediksi di-inverse ke satuan asli sebelum semua skor. Tidak mengisi target.',
            'target_scaling_reason':'Satu aturan tetap untuk optimisasi lintas fold yang rentang targetnya berbeda, ditetapkan sebelum fitting; bukan tuning hasil.',
            'architecture':{'lstm_layers':1,'units':32,'dense_output':1,'output_activation':'linear','bidirectional':False,'dropout':0,'stateful':False,'parameters':7201},
            'optimization':{'loss':'MSE standardized original target','optimizer':'Adam','lr':.001,'betas':[.9,.999],'eps':1e-8,'weight_decay':0,'batch_size':64,'max_epochs':50,
                'patience':8,'min_delta':0,'restore_best_weights':True,'epoch_shuffle':'same permutation seed+epoch*100003 across pipelines','gradient_clipping':False,'prediction_clipping':False},
            'adequacy':'Sebelum semua LSTM, >=64 train bersama, >=1 internal validasi dan >=1 evaluasi bersama. Gate komputasional, bukan jaminan representasi/generalization.',
            'ridge_comparison':'Prediksi Ridge tahap6 disimpan; skor dihitung pada irisan timestamp LSTM3+persistence+Ridge3. Ridge train lebih panjang (tanpa holdback15%internal), input tabular lag berbeda; bukan isolasi arsitektur.',
            'test_loaded':False,'test_evaluated':False,'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                'processor':os.environ.get('PROCESSOR_IDENTIFIER',platform.processor()),'logical_cpus':os.cpu_count(),'cuda_available':torch.cuda.is_available(),
                'torch_cuda_build':torch.version.cuda,'versions':{k:version(k) for k in ['torch','numpy','pandas','scikit-learn','scipy','matplotlib','ipykernel']}},
            'initial_hashes':self.before,'sources_documentation':['https://pytorch.org/get-started/locally/']}
        dump(RUN/'config_locked.json',self.cfg);self.config_hash=hashfile(RUN/'config_locked.json');self.fit_count=0;self.folds=[]
        self.plan=[]
        for number,(a,b) in enumerate(zip(self.cuts[:-1],self.cuts[1:]),1):
            c=int(np.floor(.85*a));self.plan.append({'Fold':f'Fold_{number:02d}','Inner train awal':self.times[0],'Inner train akhir':self.times[c-1],
                'Validasi internal awal':self.times[c],'Validasi internal akhir':self.times[a-1],'Evaluasi awal':self.times[a],'Evaluasi akhir':self.times[b-1],
                'Slot inner train':c,'Slot internal validation':a-c,'Slot outer train':a,'Slot evaluasi':b-a})
        dump(RUN/'rencana_fold_sebelum_training.json',self.plan)
        print('Konfigurasi dikunci',self.config_hash,'LSTM fit=0',flush=True);print(self.cfg['environment'],flush=True)
        return self
    def prepare_all_folds(self):
        for number,(a,b) in enumerate(zip(self.cuts[:-1],self.cuts[1:]),1):self.folds.append(SequenceFold(self,number,a,b).prepare_sequences())
        gates=[f.gate for f in self.folds];dump(RUN/'all_gates_before_training.json',{'fit_count':self.fit_count,'config_sha256':self.config_hash,'gates':gates})
        assert self.fit_count==0 and all(g['Lulus'] for g in gates),'Tidak cukup sampel; belum melatih LSTM, tidak ubah split.'
        return self
    def train(self):
        assert self.fit_count==0 and hashfile(RUN/'config_locked.json')==self.config_hash
        for fold in self.folds:fold.train_models()
        assert self.fit_count==36
        self.pred=pd.concat([frame for fold in self.folds for frame in fold.predictions],ignore_index=True)
        self.pred.to_csv(RUN/'prediksi_oof_seluruh_seed.csv',index=False)
        self.history=[r for f in self.folds for r in f.history];self.training=[r for f in self.folds for r in f.training]
        dump(RUN/'training_history.json',self.history);dump(RUN/'training_summary.json',self.training)
        return self
    def summarize(self):
        from summarize_lstm_stage7 import summarize
        summarize(self);return self
    def export_excel(self):
        node=Path(r'C:\Users\INTAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
        completed=subprocess.run([str(node),str(ROOT/'scripts/export_quality.mjs'),'tabel_lstm_preprocessing.json','hasil_lstm_preprocessing.xlsx','.audit_preview_lstm'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        print(completed.stdout)
        if completed.returncode:raise RuntimeError(completed.stderr)
