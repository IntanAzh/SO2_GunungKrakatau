"""Locked GRU/TCN comparison using the verified S0 tensor from stage 10."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from importlib.metadata import version
import json,os,sys,platform,random,copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from audit_quality import hashfile
from pilot_way_urang import dump
from lstm_interstation_stage10 import atomic_json,atomic_torch,atomic_csv

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OLD=OUT/'lstm_antarstasiun_tahap10';RUN=OUT/'perbandingan_algoritma_tahap11';FIG=RUN/'figures'
SEEDS=[11,23,47];NEW=['GRU','TCN']

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__();self.gru=nn.GRU(22,32,num_layers=1,batch_first=True);self.output=nn.Linear(32,1)
    def forward(self,x):
        h,_=self.gru(x);return self.output(h[:,-1,:]).squeeze(-1)

class CausalConv(nn.Module):
    def __init__(self,inside,outside,dilation):
        super().__init__();self.pad=2*dilation;self.conv=nn.Conv1d(inside,outside,3,padding=self.pad,dilation=dilation)
    def forward(self,x):
        return self.conv(x)[...,:x.size(-1)]

class ResidualBlock(nn.Module):
    def __init__(self,inside,outside,dilation):
        super().__init__();self.conv1=CausalConv(inside,outside,dilation);self.conv2=CausalConv(outside,outside,dilation)
        self.relu=nn.ReLU();self.drop=nn.Dropout(0);self.skip=nn.Conv1d(inside,outside,1) if inside!=outside else nn.Identity()
    def forward(self,x):
        h=self.drop(self.relu(self.conv1(x)));h=self.drop(self.relu(self.conv2(h)))
        return self.relu(h+self.skip(x))

class TCNModel(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList([ResidualBlock(22 if i==0 else 16,16,d) for i,d in enumerate([1,2,4,8])]);self.output=nn.Linear(16,1)
    def backbone(self,x):
        h=x.transpose(1,2)
        for block in self.blocks:h=block(h)
        return h
    def forward(self,x):return self.output(self.backbone(x)[:,:,-1]).squeeze(-1)

def model_for(name):return GRUModel() if name=='GRU' else TCNModel() if name=='TCN' else None

def causality_gate():
    torch.manual_seed(9973);model=TCNModel().eval()
    x=torch.randn(2,48,22)
    with torch.no_grad():
        full=model.backbone(x)
        for length in [1,2,7,17,31,47]:
            shorter=model.backbone(x[:,:length])
            changed=x.clone();changed[:,length:]=1000*torch.randn_like(changed[:,length:])
            perturbed=model.backbone(changed)
            assert torch.allclose(shorter,full[:,:,:length],atol=2e-6,rtol=2e-6),(length,'truncation')
            assert torch.allclose(perturbed[:,:,:length],full[:,:,:length],atol=2e-6,rtol=2e-6),(length,'future perturbation')
    return {'prefix_lengths_checked':[1,2,7,17,31,47],'future_perturbation_no_effect':True,
        'receptive_field_slots':61,'available_history_slots':48,'causal_padding_right_trim':True}

class Fold:
    def __init__(self,e,number,a,b):
        self.e=e;self.label=f'Fold_{number:02d}';self.a=a;self.b=b;self.folder=RUN/self.label;self.folder.mkdir(exist_ok=True)
    def prepare(self):
        e=self.e;old=OLD/self.label
        self.old=old;self.npz=old/'inputs_sekuens.npz';z=np.load(self.npz)
        self.features=z['S0'];assert self.features.dtype==np.float32 and self.features.shape==(self.b-1,22)
        self.train=z['train'];self.inner=z['internal'];self.eval=z['evaluation'];self.scaled=z['filled'];self.scale_slots=z['scale_slots']
        assert len(self.train)>=64 and len(self.inner)>0 and len(self.eval)>0
        c=int(np.floor(.85*self.a))
        assert (self.train+1<c).all() and ((self.inner+1>=c)&(self.inner+1<self.a)).all() and ((self.eval+1>=self.a)&(self.eval+1<self.b)).all()
        assert self.scale_slots.max()<c and np.isfinite(self.features[self.eval[:,None]-np.arange(47,-1,-1)]).all()
        assert set(np.unique(self.features[:,11:]))<=set([0.,1.])
        scaler=json.loads((old/'parameter_scaler.json').read_text(encoding='utf-8'))
        ref=[r for r in scaler if r['Skenario']=='S0'];assert len(ref)==11
        means=np.array([r['Mean'] for r in ref]);scales=np.array([r['Scale'] for r in ref]);self.target_mean=ref[0]['Target mean'];self.target_scale=ref[0]['Target scale']
        assert all(r['Target mean']==self.target_mean and r['Target scale']==self.target_scale for r in ref)
        assert np.allclose(means,self.scaled[self.scale_slots,:11].mean(0))
        std=self.scaled[self.scale_slots,:11].std(0);std[std<1e-12]=1.;assert np.allclose(scales,std)
        assert np.allclose(self.features[:,:11],(self.scaled[:,:11]-means)/scales,equal_nan=True,atol=2e-6)
        assert np.isclose(self.target_mean,e.y[self.train+1].mean()) and np.isclose(self.target_scale,e.y[self.train+1].std())
        self.ref={}
        for seed in SEEDS:
            f=pd.read_csv(old/f'S0_seed{seed}/prediksi.csv',parse_dates=['Origin','Timestamp target'])
            assert np.array_equal(f['Index origin'],self.eval) and np.array_equal(f['Index target'],self.eval+1)
            assert np.array_equal(f['SO2 target asli'],e.y[self.eval+1]) and f['Timestamp target'].max()<e.test_start
            assert np.array_equal(f['Timestamp target'].to_numpy(),e.times[self.eval+1].to_numpy())
            self.ref[seed]=f
        for seed in SEEDS[1:]:
            assert np.array_equal(self.ref[seed]['Timestamp target'],self.ref[SEEDS[0]]['Timestamp target'])
            assert np.array_equal(self.ref[seed].Persistence,self.ref[SEEDS[0]].Persistence)
        for part,ids in [('inner_train',self.train),('inner_validation',self.inner),('evaluation',self.eval)]:
            prior=pd.read_csv(old/f'timestamp_{part}_bersama.csv')
            assert np.array_equal(prior['Index origin'],ids) and np.array_equal(prior['SO2 target asli'],e.y[ids+1])
            atomic_csv(self.folder/f'timestamp_{part}.csv',prior)
        oldgate=json.loads((old/'gate_pra_training.json').read_text(encoding='utf-8'))
        self.gate={'Fold':self.label,'Train N':len(self.train),'Internal N':len(self.inner),'Evaluasi N':len(self.eval),
            'Stage10 S0 train cocok':oldgate['Train bersama']==len(self.train),
            'Stage10 S0 internal cocok':oldgate['Validasi internal bersama']==len(self.inner),
            'Stage10 S0 evaluasi cocok':oldgate['Evaluasi bersama']==len(self.eval),
            'Fitur kanal':self.features.shape[1],'Fitur terverifikasi':True,'Target asli':True,'Scaler train internal':True,
            'Fit baru sebelum gate':0,'Lulus':True,'Train akhir kalender':e.times[c-1],
            'Internal mulai kalender':e.times[c],'Internal akhir kalender':e.times[self.a-1],
            'Evaluasi mulai kalender':e.times[self.a],'Evaluasi akhir kalender':e.times[self.b-1]}
        atomic_json(self.folder/'gate_pra_fit.json',self.gate)
        return self
    def sequences(self,origins):return self.features[origins[:,None]-np.arange(47,-1,-1)[None,:]]
    def fit(self,method,seed):
        e=self.e;folder=self.folder/f'{method}_seed{seed}';folder.mkdir(exist_ok=True);done=folder/'complete.json'
        if done.exists():
            r=json.loads(done.read_text(encoding='utf-8'));assert r['config_sha256']==e.config_hash
            for name,h in r['files_sha256'].items():assert hashfile(folder/name)==h
            print('LEWATI model lengkap',self.label,method,seed,flush=True);return
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=model_for(method);optimizer=torch.optim.Adam(model.parameters(),lr=.001);lossfn=nn.MSELoss()
        train=self.train;inner=self.inner
        X=torch.from_numpy(self.sequences(train));Y=torch.tensor((e.y[train+1]-self.target_mean)/self.target_scale,dtype=torch.float32)
        VX=torch.from_numpy(self.sequences(inner));VY=torch.tensor((e.y[inner+1]-self.target_mean)/self.target_scale,dtype=torch.float32)
        last=folder/'last_epoch.pt';best=float('inf');best_state=None;best_epoch=0;stale=0;history=[];elapsed=0.;start=1
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=False);assert saved['config_sha256']==e.config_hash
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng'])
            best=saved['best'];best_state=saved['best_state'];best_epoch=saved['best_epoch'];stale=saved['stale'];history=saved['history'];elapsed=saved['fit_seconds'];start=saved['epoch']+1
            print('LANJUT epoch',self.label,method,seed,start,flush=True)
        for epoch in range(start,51):
            if stale>=8:break
            tick=perf_counter();model.train();order=torch.from_numpy(np.random.default_rng(seed+epoch*100003).permutation(len(train)));total=0.
            for begin in range(0,len(train),64):
                ix=order[begin:begin+64];optimizer.zero_grad(set_to_none=True);pred=model(X[ix]);loss=lossfn(pred,Y[ix]);loss.backward();optimizer.step();total+=float(loss.detach())*len(ix)
            model.eval();vsum=0.
            with torch.no_grad():
                for begin in range(0,len(inner),256):vsum+=float(torch.sum((model(VX[begin:begin+256])-VY[begin:begin+256])**2))
            vloss=vsum/len(inner)
            if not np.isfinite(vloss):raise RuntimeError(f'Loss nonfinite {self.label} {method} {seed}; tidak ada fallback.')
            history.append({'Fold':self.label,'Algoritma':method,'Seed':seed,'Epoch':epoch,'Train MSE standar':total/len(train),'Internal MSE standar':vloss})
            if vloss<best:best=vloss;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
            else:stale+=1
            elapsed+=perf_counter()-tick
            atomic_torch(last,{'config_sha256':e.config_hash,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'torch_rng':torch.get_rng_state(),'best':best,'best_state':best_state,'best_epoch':best_epoch,'stale':stale,'history':history,'fit_seconds':elapsed})
            atomic_json(folder/'history.json',history)
            atomic_json(folder/'progress.json',{'config_sha256':e.config_hash,'epoch':epoch,'best_epoch':best_epoch,'fit_seconds':elapsed,'state':'training'})
            if epoch%5==0:print(self.label,method,seed,'epoch',epoch,'best',best_epoch,'fit',round(elapsed,2),flush=True)
        assert best_state is not None
        model.load_state_dict(best_state);model.eval();atomic_torch(folder/'best.pt',best_state)
        ids=self.eval;tick=perf_counter();p=np.full(len(ids),np.nan)
        with torch.no_grad():
            for begin in range(0,len(ids),256):
                q=ids[begin:begin+256];p[begin:begin+256]=model(torch.from_numpy(self.sequences(q))).numpy()*self.target_scale+self.target_mean
        predict_seconds=perf_counter()-tick
        frame=self.ref[seed][['Fold','Index origin','Origin','Index target','Timestamp target','SO2 target asli','Persistence','Persistence usia slot']].copy()
        frame.insert(1,'Algoritma',method);frame.insert(2,'Seed',seed);frame['Prediksi']=p
        atomic_csv(folder/'prediksi.csv',frame)
        result={'Fold':self.label,'Algoritma':method,'Seed':seed,'config_sha256':e.config_hash,'Epoch dijalankan':len(history),'Epoch terbaik':best_epoch,
            'Internal MSE standar terbaik':best,'Fit early stopping detik':elapsed,'Predict detik':predict_seconds,
            'N train':len(train),'N internal':len(inner),'N evaluasi target asli':len(ids),'Jumlah parameter':sum(v.numel() for v in model.parameters()),
            'Waktu definisi':'Fit=sum durasi epoch training dan validasi internal; checkpoint I/O, persiapan tensor, dan jeda resume tidak termasuk. Predict=inferensi dan inverse target scaling.',
            'files_sha256':{name:hashfile(folder/name) for name in ['best.pt','last_epoch.pt','history.json','prediksi.csv']}}
        atomic_json(done,result);print('SELESAI',self.label,method,seed,'epoch',len(history),'best',best_epoch,'fit',round(elapsed,2),flush=True)

class Experiment:
    def prepare(self):
        RUN.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
        torch.set_num_threads(1)
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
        torch.use_deterministic_algorithms(True)
        self.prior=json.loads((OLD/'config_locked.json').read_text(encoding='utf-8'));self.cuts=self.prior['fold_cuts'];n=self.cuts[-1];self.test_start=pd.Timestamp(self.prior['test_start'])
        self.data=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=n,parse_dates=['Timestamp']);self.times=pd.DatetimeIndex(self.data.Timestamp);self.y=self.data['SO2 numerik'].to_numpy(float)
        assert len(self.data)==n and self.times[-1]<self.test_start and (self.times[1:]-self.times[:-1]==pd.Timedelta(minutes=30)).all()
        assert self.prior['scenarios']['S0']=='Way Urang saja' and self.prior['warmup_slots']==1440 and self.prior['sequence_length']==48 and self.prior['seeds']==SEEDS
        assert 'Curah Hujan' not in self.prior['covariates'] and self.prior['features']['S0'][-1]=='missing_asli_Sol.Rad_lag1'
        self.causal_check=causality_gate();self.folds=[];self.fit_count=0
        cfgfile=RUN/'config_locked.json'
        if cfgfile.exists():
            self.cfg=json.loads(cfgfile.read_text(encoding='utf-8'))
            assert self.cfg['implementation_sha256']==hashfile(Path(__file__)),'Implementasi training berubah setelah kunci.'
            assert hashfile(OLD/'config_locked.json')==self.cfg['stage10_config_sha256']
        else:
            oldmanifest=json.loads((OLD/'manifest_tahap10.json').read_text(encoding='utf-8'))
            for row in oldmanifest:assert hashfile(ROOT/row['File'])==row['SHA256'],row['File']
            baseline={row['File']:row['SHA256'] for row in oldmanifest if row['File']!='outputs/STATUS_PROYEK.md'}
            for source in [OUT/'kontribusi_antarstasiun/config_locked.json',OLD/'config_locked.json']:
                for rel,h in json.loads(source.read_text(encoding='utf-8'))['initial_hashes'].items():
                    if rel in baseline:assert baseline[rel]==h,rel
                    baseline[rel]=h
            status=OUT/'STATUS_PROYEK.md';status_before=hashfile(status)
            self.cfg={'experiment':'Tahap11 perbandingan algoritma forecasting SO2 Way Urang S0 C30','created_utc':datetime.now(timezone.utc).isoformat(),
                'stage10_config_sha256':hashfile(OLD/'config_locked.json'),'stage10_manifest_sha256':hashfile(OLD/'manifest_tahap10.json'),
                'fold_cuts':self.cuts,'folds':self.prior['folds'],'test_start':str(self.test_start),'inner_split':self.prior['internal_split'],
                'target':'SO2 Way Urang original pada t+30menit','input':'S0 Way Urang saja, tanpa Curah Hujan','horizon_minutes':30,'warmup_slots':1440,'sequence_length':48,
                'features':self.prior['features']['S0'],'sequence_channels':22,'missing_indicators_original':11,
                'preprocessing':'Reuse exact stage10 S0 tensor, scaler, original targets, train/internal/eval timestamp; no new imputer/scaler fit; stage10 prefix median train-internal only.',
                'lstm_reference':'Stage10 S0 12 saved checkpoints/predictions, not refit or selected by seed.',
                'new_models':['GRU','TCN'],'seeds':SEEDS,'planned_new_fits':24,'architecture':{
                    'GRU':{'layers':1,'units':32,'output':'Linear(32,1)','bidirectional':False,'dropout':0,'parameters':sum(v.numel() for v in GRUModel().parameters())},
                    'TCN':{'blocks':4,'channels':16,'convolutions_per_block':2,'kernel':3,'dilations':[1,2,4,8],'causal_padding':'symmetric pad 2*d then trim right 2*d, output at t uses input<=t',
                        'activation':'ReLU after each convolution and residual addition','dropout':0,'first_block_skip_projection':'1x1 Conv1d 22→16 to match residual channels',
                        'output':'Linear(16,1) from last time step','receptive_field_slots':61,'parameters':sum(v.numel() for v in TCNModel().parameters())}},
                'causal_test':self.causal_check,'optimization':self.prior['optimization'],
                'target_scaling':self.prior['target_scaling'],'quality':self.prior['quality'],'alignment_assumption':self.prior['alignment_assumption'],
                'comparison':'Same S0 stage10 train/internal/eval, same target and persistence timestamps, three fixed seeds. Equal max epochs/patience not equal parameter count, compute, convergence or optimization state.',
                'resume':'Atomic last_epoch checkpoint each epoch (model+optimizer+RNG+best+time); complete.json written last per model. Completed models never refit.',
                'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'device':'cpu',
                    'cuda_available':torch.cuda.is_available(),'torch_cuda_build':torch.version.cuda,'torch_threads':1,'deterministic_algorithms':True,
                    'versions':{k:version(k) for k in ['torch','numpy','pandas','scikit-learn','scipy','matplotlib','ipykernel']}},
                'status_before_sha256':status_before,'status_update_authorized':True,
                'test_loaded':False,'test_evaluated':False,'implementation_sha256':hashfile(Path(__file__)),'initial_hashes':baseline}
            atomic_json(cfgfile,self.cfg)
            atomic_json(RUN/'daftar_fitur.json',{'S0':self.cfg['features'],'input_shape_per_sample':[48,22],
                'flags_original':self.cfg['features'][11:],'numeric_scaled':self.cfg['features'][:11]})
        self.config_hash=hashfile(cfgfile);print('KONFIG TERKUNCI',self.config_hash,'fit baru=0',flush=True);return self
    def prepare_all(self):
        self.folds=[Fold(self,i,a,b).prepare() for i,(a,b) in enumerate(zip(self.cuts[:-1],self.cuts[1:]),1)]
        assert all(f.gate['Lulus'] for f in self.folds) and self.fit_count==0
        atomic_json(RUN/'gate_semua_sebelum_fit.json',{'config_sha256':self.config_hash,'fit_count':0,
            'folds':[f.gate for f in self.folds],'causality':self.causal_check})
        print('SEMUA GATE LULUS; model baru belum dilatih',flush=True);return self
    def train(self):
        assert hashfile(RUN/'config_locked.json')==self.config_hash
        for f in self.folds:
            for seed in SEEDS:
                for method in NEW:f.fit(method,seed)
        assert len(list(RUN.glob('Fold_*/GRU_seed*/complete.json')))==12
        assert len(list(RUN.glob('Fold_*/TCN_seed*/complete.json')))==12
        print('24 MODEL BARU SELESAI',flush=True);return self

if __name__=='__main__':Experiment().prepare().prepare_all().train()
