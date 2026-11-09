"""Locked C30 interstation LSTM experiment; CPU, development-only, resumable."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from importlib.metadata import version,distributions
import os,sys,json,copy,random,platform,subprocess
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from audit_quality import hashfile
from pilot_way_urang import dump
from calibration_sensitivity import distribution

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; RUN=OUT/'lstm_antarstasiun_tahap10'; FIG=RUN/'figures'
SCENARIOS=['S0','S1']; SEEDS=[11,23,47]
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))

def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');dump(tmp,value);os.replace(tmp,path)
def atomic_torch(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);os.replace(tmp,path)
def atomic_csv(path,frame):
    tmp=path.with_suffix(path.suffix+'.tmp');frame.to_csv(tmp,index=False);os.replace(tmp,path)

class SimpleLSTM(nn.Module):
    def __init__(self,channels):
        super().__init__();self.lstm=nn.LSTM(channels,32,num_layers=1,batch_first=True);self.output=nn.Linear(32,1)
    def forward(self,x):
        h,_=self.lstm(x);return self.output(h[:,-1,:]).squeeze(-1)

def causal_history(raw,cut,stop,times,names):
    h=raw[:stop].copy();logs=[]
    for start in list(range(1440,cut,1440))+[cut]:
        end=min(start+1440,cut) if start<cut else stop
        if start>=end:continue
        med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in raw[:start].T])
        h[start:end]=np.where(np.isfinite(raw[start:end]),raw[start:end],med)
        for j,name in enumerate(names):logs.append({'Kanal':name,'Query awal':times[start],'Query akhir':times[end-1],
            'Prefix cutoff eksklusif':times[start],'Prefix slot':start,'Original prefix N':int(np.isfinite(raw[:start,j]).sum()),
            'Median':med[j],'Beku setelah train internal':start==cut,
            'Timestamp sumber terakhir diizinkan':times[start-2] if name.endswith('_lag1') else times[start-1]})
    return h,logs

class Fold:
    def __init__(self,e,number,a,b):
        self.e=e;self.label=f'Fold_{number:02d}';self.a=a;self.b=b;self.c=int(np.floor(.85*a));self.folder=RUN/self.label;self.folder.mkdir(exist_ok=True)
    def prepare(self):
        tick=perf_counter();e=self.e;c=self.c;a=self.a;b=self.b
        h,logs=causal_history(e.raw,c,b-1,e.times,e.numeric);self.filled=h;self.prefix=[{'Fold':self.label,**r} for r in logs]
        self.flags=(~np.isfinite(e.raw[:b-1])).astype(float);self.own={};self.shared={};self.flow=[];self.dist=[];self.monthly=[];self.lost=[];self.availability=[]
        self.parts={'inner_train':np.arange(1488,c-1),'inner_validation':np.arange(max(1488,c-1),a-1),'evaluation':np.arange(a-1,b-1)}
        self.bounds={'inner_train':(0,c),'inner_validation':(c,a),'evaluation':(a,b)}
        complete={}
        for s,d in [('S0',11),('S1',12)]:
            missing=~np.isfinite(h[:,:d]).all(1);cs=np.r_[0,np.cumsum(missing)];ok=np.zeros(b-1,bool);ix=np.arange(47,b-1)
            ok[ix]=(cs[ix+1]-cs[ix-47])==0;complete[s]=ok
        for part,ids in self.parts.items():
            obs=ids[np.isfinite(e.y[ids+1])];common=obs
            for s in SCENARIOS:
                self.own[s,part]=obs[complete[s][obs]];common=np.intersect1d(common,self.own[s,part])
            self.shared[part]=common
            lo,hi=self.bounds[part];alltargets=np.arange(lo,hi);allobs=alltargets[np.isfinite(e.y[alltargets])]
            for s,d in [('S0',11),('S1',12)]:
                own=self.own[s,part];lost=np.setdiff1d(own,common)
                self.flow.append({'Fold':self.label,'Bagian':part,'Skenario':s,'Slot kalender target':hi-lo,'Target asli seluruh bagian':len(allobs),
                    'Kandidat origin setelah warmup buffer purge':len(ids),'Target asli sebelum sekuens':len(obs),
                    'Origin finite setelah imputasi':int(np.isfinite(h[obs,:d]).all(1).sum()),'Sekuens lengkap 48 slot':len(own),
                    'Sesudah irisan':len(common),'Keluar karena sekuens':len(obs)-len(own),'Keluar karena irisan':len(lost)})
                self.lost.append({'Fold':self.label,'Bagian':part,'Skenario':s,'Alasan':'irisan antar-skenario',**distribution(e.y[lost+1])})
                ledger=pd.DataFrame({'Index origin':ids,'Origin':e.times[ids],'Timestamp target':e.times[ids+1],'SO2 target asli':e.y[ids+1],
                    'Target kosong':~np.isfinite(e.y[ids+1]),'Origin input kosong setelah imputasi':~np.isfinite(h[ids,:d]).all(1),
                    'Sekuens tidak lengkap':~complete[s][ids],'Layak mandiri':np.isin(ids,own),'Bersama digunakan':np.isin(ids,common)})
                atomic_csv(self.folder/f'ledger_{part}_{s}.csv',ledger)
            for domain,selected in [('target asli kalender',allobs-1),('target asli sebelum sekuens',obs),('bersama digunakan',common)]:
                self.dist.append({'Fold':self.label,'Bagian':part,'Domain':domain,'Target awal':e.times[selected[0]+1] if len(selected) else None,
                    'Target akhir':e.times[selected[-1]+1] if len(selected) else None,**distribution(e.y[selected+1])})
                for month in e.times[alltargets].to_period('M').unique():
                    sub=selected[e.times[selected+1].to_period('M')==month]
                    self.monthly.append({'Fold':self.label,'Bagian':part,'Domain':domain,'Bulan':str(month),**distribution(e.y[sub+1])})
            atomic_csv(self.folder/f'timestamp_{part}_bersama.csv',pd.DataFrame({'Index origin':common,'Origin':e.times[common],
                'Timestamp target':e.times[common+1],'SO2 target asli':e.y[common+1]}))
            if len(common):
                slots=common[:,None]-np.arange(47,-1,-1)[None,:];flag=self.flags[slots,11]
                self.availability.append({'Fold':self.label,'Bagian':part,'N sekuens bersama':len(common),'Posisi fitur donor':flag.size,
                    'Posisi donor missing asli diimputasi':int(flag.sum()),'Proporsi posisi donor diimputasi persen':100*flag.mean(),
                    'Sekuens dengan donor diimputasi N':int(flag.any(1).sum()),'Sekuens dengan donor diimputasi persen':100*flag.any(1).mean(),
                    'Usia donor origin median menit':float(np.median(e.age[common])),'Usia donor origin P95 menit':float(np.percentile(e.age[common],95)),
                    'Usia donor origin maksimum menit':float(np.max(e.age[common]))})
        tr=self.shared['inner_train'];val=self.shared['inner_validation'];ev=self.shared['evaluation']
        self.gate={'Fold':self.label,'Train bersama':len(tr),'Validasi internal bersama':len(val),'Evaluasi bersama':len(ev),'Fit count sebelum gate':0,
            'Lulus':bool(len(tr)>=64 and len(val)>0 and len(ev)>0),'Train internal akhir kalender':e.times[c-1],
            'Validasi internal awal kalender':e.times[c],'Validasi internal akhir kalender':e.times[a-1],
            'Evaluasi awal kalender':e.times[a],'Evaluasi akhir kalender':e.times[b-1]}
        assert self.gate['Lulus'],'Kecukupan gagal sebelum training; jangan mengubah split berdasarkan skor.'
        assert (tr+1<c).all() and ((val+1>=c)&(val+1<a)).all() and ((ev+1>=a)&(ev+1<b)).all()
        slots=np.unique((tr[:,None]-np.arange(48)).ravel());assert slots.max()<c
        self.slots=slots;self.target_mean=float(e.y[tr+1].mean());self.target_scale=float(e.y[tr+1].std()) or 1.
        self.features={};self.scale_records=[]
        for s,d in [('S0',11),('S1',12)]:
            scaler=StandardScaler().fit(h[slots,:d]);numeric=scaler.transform(h[:,:d])
            self.features[s]=np.column_stack([numeric,self.flags[:,:d]]).astype(np.float32)
            for j,name in enumerate(e.numeric[:d]):self.scale_records.append({'Fold':self.label,'Skenario':s,'Kanal':name,'Mean':scaler.mean_[j],
                'Scale':scaler.scale_[j],'Slot scaler N':len(slots),'Slot scaler terakhir':e.times[slots.max()],
                'Target mean':self.target_mean,'Target scale':self.target_scale})
        np.savez_compressed(self.folder/'inputs_sekuens.npz',S0=self.features['S0'],S1=self.features['S1'],filled=h,raw_flags=self.flags,
            scale_slots=slots,train=tr,internal=val,evaluation=ev)
        atomic_json(self.folder/'parameter_scaler.json',self.scale_records);atomic_json(self.folder/'gate_pra_training.json',self.gate)
        self.preprocess_seconds=perf_counter()-tick
        atomic_json(self.folder/'preprocessing.json',{'detik':self.preprocess_seconds,'flow':self.flow,'distribusi':self.dist,
            'bulanan':self.monthly,'irisan_keluar':self.lost,'ketersediaan':self.availability,'prefix':self.prefix})
        print('GATE',self.gate,flush=True);return self
    def sequences(self,s,ids):return self.features[s][ids[:,None]-np.arange(47,-1,-1)[None,:]]
    def model(self,s,seed):
        e=self.e; folder=self.folder/f'{s}_seed{seed}';folder.mkdir(exist_ok=True);done=folder/'complete.json'
        if done.exists():
            result=json.loads(done.read_text(encoding='utf-8'));assert result['config_sha256']==e.config_hash
            for name,sha in result['files_sha256'].items():assert hashfile(folder/name)==sha
            print('LEWATI model lengkap',self.label,s,seed,flush=True);return
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=SimpleLSTM(22 if s=='S0' else 24);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        lossfn=nn.MSELoss();train=self.shared['inner_train'];val=self.shared['inner_validation']
        X=torch.from_numpy(self.sequences(s,train));Y=torch.tensor((e.y[train+1]-self.target_mean)/self.target_scale,dtype=torch.float32)
        VX=torch.from_numpy(self.sequences(s,val));VY=torch.tensor((e.y[val+1]-self.target_mean)/self.target_scale,dtype=torch.float32)
        lastfile=folder/'last_epoch.pt';best=float('inf');beststate=None;bestepoch=0;stale=0;history=[];elapsed=0.;start=1
        if lastfile.exists():
            saved=torch.load(lastfile,map_location='cpu',weights_only=False);assert saved['config_sha256']==e.config_hash
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng'])
            best=saved['best'];beststate=saved['best_state'];bestepoch=saved['best_epoch'];stale=saved['stale'];history=saved['history'];elapsed=saved['fit_seconds'];start=saved['epoch']+1
            print('RESUME epoch',self.label,s,seed,start,flush=True)
        for epoch in range(start,51):
            if stale>=8:break
            tick=perf_counter();model.train();order=torch.from_numpy(np.random.default_rng(seed+epoch*100003).permutation(len(train)));total=0.
            for begin in range(0,len(train),64):
                ix=order[begin:begin+64];optimizer.zero_grad(set_to_none=True);pred=model(X[ix]);loss=lossfn(pred,Y[ix]);loss.backward();optimizer.step();total+=float(loss.detach())*len(ix)
            model.eval();vsum=0.
            with torch.no_grad():
                for begin in range(0,len(val),256):vsum+=float(torch.sum((model(VX[begin:begin+256])-VY[begin:begin+256])**2))
            vloss=vsum/len(val)
            if not np.isfinite(vloss):raise RuntimeError(f'Nonfinite loss {self.label} {s} seed{seed}; tidak ada clipping/fallback.')
            history.append({'Fold':self.label,'Skenario':s,'Seed':seed,'Epoch':epoch,'Train MSE standar':total/len(train),'Internal MSE standar':vloss})
            if vloss<best:best=vloss;beststate=copy.deepcopy(model.state_dict());bestepoch=epoch;stale=0
            else:stale+=1
            elapsed+=perf_counter()-tick
            atomic_torch(lastfile,{'config_sha256':e.config_hash,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'torch_rng':torch.get_rng_state(),'best':best,'best_state':beststate,'best_epoch':bestepoch,'stale':stale,'history':history,'fit_seconds':elapsed})
            atomic_json(folder/'history.json',history)
            atomic_json(folder/'progress.json',{'config_sha256':e.config_hash,'epoch':epoch,'best_epoch':bestepoch,'fit_seconds':elapsed,'state':'training'})
            if epoch%5==0:print(self.label,s,'seed',seed,'epoch',epoch,'best',bestepoch,'fit',round(elapsed,1),flush=True)
        assert beststate is not None
        model.load_state_dict(beststate);model.eval();atomic_torch(folder/'best.pt',beststate)
        target=np.arange(self.a,self.b);target=target[np.isfinite(e.y[target])];origin=target-1;good=np.isin(origin,self.own[s,'evaluation'])
        pred=np.full(len(origin),np.nan);ids=np.flatnonzero(good);tick=perf_counter()
        with torch.no_grad():
            for begin in range(0,len(ids),256):
                ix=ids[begin:begin+256];pred[ix]=model(torch.from_numpy(self.sequences(s,origin[ix]))).numpy()*self.target_scale+self.target_mean
        inference=perf_counter()-tick
        last=np.maximum.accumulate(np.where(np.isfinite(e.y[:self.b-1]),np.arange(self.b-1),-1))[origin]
        flags=self.flags[origin[:,None]-np.arange(47,-1,-1),11]
        frame=pd.DataFrame({'Fold':self.label,'Skenario':s,'Seed':seed,'Index origin':origin,'Origin':e.times[origin],
            'Index target':target,'Timestamp target':e.times[target],'SO2 target asli':e.y[target],'SO2 asli origin':e.y[origin],
            'Prediksi':pred,'Persistence':e.y[last],'Persistence usia slot':origin-last,
            'Bersama digunakan':np.isin(origin,self.shared['evaluation']),'Pulomerak usia asli origin menit':e.age[origin],
            'Pulomerak timestamp asli terakhir':e.times[np.maximum(e.last_donor[origin],0)],
            'Pulomerak posisi missing asli dari48':flags.sum(1),'Pulomerak proporsi posisi diimputasi':flags.mean(1)})
        atomic_csv(folder/'prediksi.csv',frame)
        result={'Fold':self.label,'Skenario':s,'Seed':seed,'config_sha256':e.config_hash,'Epoch dijalankan':len(history),'Epoch terbaik':bestepoch,
            'Internal MSE standar terbaik':best,'Fit early stopping detik':elapsed,'Predict detik':inference,
            'Waktu fit definisi':'Jumlah durasi epoch train+validasi internal; tidak termasuk penulisan checkpoint, persiapan tensor, dan jeda resume.',
            'N train bersama':len(train),'N validasi internal bersama':len(val),'Jumlah parameter':sum(p.numel() for p in model.parameters()),
            'files_sha256':{name:hashfile(folder/name) for name in ['best.pt','last_epoch.pt','history.json','prediksi.csv']}}
        atomic_json(done,result);print('SELESAI',self.label,s,seed,'epoch',len(history),'best',bestepoch,'fit',round(elapsed,2),flush=True)

class Experiment:
    def prepare(self):
        RUN.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
        torch.set_num_threads(1)
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
        torch.use_deterministic_algorithms(True)
        self.stage7=json.loads((OUT/'lstm_preprocessing/config_locked.json').read_text(encoding='utf-8'))
        self.stage8=json.loads((OUT/'sensitivitas_kalibrasi/config_locked.json').read_text(encoding='utf-8'))
        self.stage9=json.loads((OUT/'kontribusi_antarstasiun/config_locked.json').read_text(encoding='utf-8'))
        self.reports={name:(OUT/name).read_text(encoding='utf-8') for name in ['ringkasan_lstm_preprocessing.md','ringkasan_sensitivitas_kalibrasi.md','ringkasan_kontribusi_antarstasiun.md','diagnosis_fold1.md']}
        self.cuts=self.stage9['fold_cuts'];n=self.cuts[-1]
        self.data=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=n,parse_dates=['Timestamp'])
        self.times=pd.DatetimeIndex(self.data.Timestamp);self.y=self.data['SO2 numerik'].to_numpy(float)
        donor=pd.read_csv(OUT/'kontribusi_antarstasiun/grid_dev_Pulomerak.csv',nrows=n,parse_dates=['Timestamp'])
        assert self.times.equals(pd.DatetimeIndex(donor.Timestamp)) and self.times[-1]<pd.Timestamp(self.stage9['test_start'])
        assert (self.times[1:]-self.times[:-1]==pd.Timedelta(minutes=30)).all()
        self.donor=donor['SO2 numerik untuk grafik'].to_numpy(float);self.cov=self.stage9['covariates']
        assert self.cov==self.stage7['covariates'] and 'Curah Hujan' not in self.cov
        self.numeric=['SO2_origin']+[k+'_lag1' for k in self.cov]+['Pulomerak_SO2_lag1']
        self.raw=np.column_stack([self.y]+[self.data[k+' numerik'].shift(1).to_numpy(float) for k in self.cov]+[pd.Series(self.donor).shift(1)])
        pos=np.arange(n);last=np.maximum.accumulate(np.where(np.isfinite(self.donor),pos,-1));self.last_donor=np.r_[-1,last[:-1]]
        self.age=np.where(self.last_donor>=0,(pos-self.last_donor)*30,np.nan)
        self.names={s:self.numeric[:d]+['missing_asli_'+k for k in self.numeric[:d]] for s,d in [('S0',11),('S1',12)]}
        self.fit_count=0;self.folds=[]
        cfgfile=RUN/'config_locked.json'
        if cfgfile.exists():
            self.cfg=json.loads(cfgfile.read_text(encoding='utf-8'))
            if self.cfg['implementation_sha256']!=hashfile(Path(__file__)):
                assert not list(RUN.glob('Fold_*/S*_seed*/*.pt')) and not (RUN/'gate_semua_sebelum_fit.json').exists(),'Implementasi training berubah setelah gate/model.'
                archived=RUN/'config_prefit_attempt_01.json';assert not archived.exists()
                archived.write_bytes(cfgfile.read_bytes())
                self.cfg['preflight_environment_note']='Attempt pertama berhenti sebelum gate/fit: pip tidak tersedia. Pencatatan paket diganti importlib.metadata, tanpa perubahan rancangan/model.'
                self.cfg['implementation_sha256']=hashfile(Path(__file__));atomic_json(cfgfile,self.cfg)
            for rel,sha in self.cfg['initial_hashes'].items():assert hashfile(ROOT/rel)==sha,rel
        else:
            refs=[]
            for dirname in ['data_raw','notebooks','scripts','outputs']:
                for p in (ROOT/dirname).rglob('*'):
                    rel=p.relative_to(ROOT)
                    if not p.is_file() or RUN in p.parents or p==RUN or p.name.startswith('10_') or p.name.endswith('.inspect.ndjson'):continue
                    if any(v.startswith('.') or v in ['node_modules','__pycache__'] for v in rel.parts):continue
                    if dirname=='scripts' and ('stage10' in p.name):continue
                    refs.append(p)
            refs += [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.txt','.md']]
            self.cfg={'experiment':'Tahap10 LSTM kontribusi Pulomerak C30','created_utc':datetime.now(timezone.utc).isoformat(),
                'scenarios':{'S0':'Way Urang saja','S1':'Way Urang + SO2 Pulomerak lag1 tiap langkah'},'target':'SO2 Way Urang original','horizon_minutes':30,
                'fold_cuts':self.cuts,'folds':self.stage9['folds'],'test_start':self.stage9['test_start'],'internal_split':'floor(0.85 * slot kalender outer train); target kronologis',
                'warmup_slots':1440,'prefix_update_slots':1440,'first_origin':1488,'sequence_length':48,'sequence_order':'origin−47 sampai origin; clock30menit utuh',
                'covariates':self.cov,'numeric_features':self.numeric,'features':self.names,'numeric_channels':{'S0':11,'S1':12},'total_channels':{'S0':22,'S1':24},
                'no_tabular_lag_duplication':True,'donor_latency_slots':1,'availability':'Way SO2(u) tersedia u; kovariat Way(u−1) tersedia u; Pulomerak SO2(u−1) tersedia u, asumsi belum terkonfirmasi.',
                'alignment_assumption':self.stage9['alignment_assumption'],'quality':self.stage9['quality'],'units_literal':self.stage9['units_literal'],
                'offgrid_policy':'Exact 00/30 saja, ledger development tahap9 disalin tanpa rounding; tidak dipakai untuk imputasi/fitur.',
                'imputation':'Median raw kanal tertunda [:prefix_start], update30hari selama train internal; frozen pada cutoff inner_train, tidak memakai target atau kovariat internal validation/evaluasi untuk estimasi. Observasi baru yang tersedia tetap boleh menjadi input.',
                'donor_median_detail':'Kanal Pulomerak(u−1) disusun dahulu, lalu median prefix langkah [:start], sehingga sumber terbaru di bank adalah start−2. Ini prefix kanal sekuens tahap7 dengan warmup C30, bukan duplikasi lag tabular tahap9.',
                'numeric_scaling':'StandardScaler pada slot langkah UNIK yang dipakai sekuens train internal bersama, terpisah per skenario; indikator missing tidak diskalakan.',
                'target_scaling':'Mean dan SD populasi target asli train internal bersama; SD nol diganti1 eksplisit; inverse sebelum evaluasi.',
                'architecture':{'lstm_layers':1,'units':32,'dense_output':1,'output_activation':'linear','bidirectional':False,'dropout':0,'stateful':False},
                'optimization':self.stage7['optimization'],'seeds':SEEDS,'planned_models':24,'gate':'sebelum semua fit >=64 train bersama dan >=1 internal/evaluasi bersama',
                'persistence':'SO2 Way Urang original terakhir pada/sebelum origin; tanpa batas usia; usia dilaporkan.',
                'comparison':'S0–S1 seed sama dan timestamp train/internal/evaluasi identik; tidak memilih seed. Ridge9 hanya prediksi tersimpan pada timestamp identik, representasi dan holdback train berbeda.',
                'resume':'Checkpoint optimizer/model/RNG setiap epoch; complete.json ditulis terakhir setelah checkpoint best, history, prediksi, waktu. Model lengkap tidak dilatih ulang.',
                'timing':'Fit=sum durasi epoch train+early-stopping, tanpa checkpoint I/O/tensor setup/jeda; predict inferensi+inverse. Preprocessing dicatat sekali per fold, bukan per seed.',
                'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'processor':os.environ.get('PROCESSOR_IDENTIFIER'),
                    'device':'cpu','cuda_available':torch.cuda.is_available(),'torch_cuda_build':torch.version.cuda,'threads':1,'deterministic_algorithms':True,
                    'versions':{k:version(k) for k in ['torch','numpy','pandas','scikit-learn','scipy','matplotlib','ipykernel']}},
                'test_loaded':False,'test_evaluated':False,'implementation_sha256':hashfile(Path(__file__)),
                'initial_hashes':{p.relative_to(ROOT).as_posix():hashfile(p) for p in sorted(set(refs))}}
            atomic_json(cfgfile,self.cfg)
            atomic_json(RUN/'daftar_fitur.json',self.names)
        if not (RUN/'dependensi_freeze.txt').exists():
            (RUN/'dependensi_freeze.txt').write_text('\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in distributions()))+'\n',encoding='utf-8')
        if not (RUN/'ledger_offgrid_development.csv').exists():
            atomic_csv(RUN/'ledger_offgrid_development.csv',pd.read_csv(OUT/'kontribusi_antarstasiun/ledger_offgrid_dev.csv').query("Stasiun in ['Way Urang','Pulomerak']"))
        self.config_hash=hashfile(cfgfile);print('KONFIGURASI TERKUNCI',self.config_hash,'models=0',flush=True);return self
    def prepare_all(self):
        self.folds=[Fold(self,i,a,b).prepare() for i,(a,b) in enumerate(zip(self.cuts[:-1],self.cuts[1:]),1)]
        atomic_json(RUN/'gate_semua_sebelum_fit.json',{'fit_count':0,'config_sha256':self.config_hash,'gates':[f.gate for f in self.folds]})
        return self
    def train(self):
        assert all(f.gate['Lulus'] for f in self.folds) and hashfile(RUN/'config_locked.json')==self.config_hash
        for f in self.folds:
            for seed in SEEDS:
                for s in SCENARIOS:f.model(s,seed)
        complete=list(RUN.glob('Fold_*/S*_seed*/complete.json'));assert len(complete)==24
        print('SEMUA 24 MODEL LENGKAP',flush=True);return self

if __name__=='__main__':Experiment().prepare().prepare_all().train()
