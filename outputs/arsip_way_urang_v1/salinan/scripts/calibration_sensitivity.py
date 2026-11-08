"""Ridge/median sensitivity: calendar development folds, no final-test access."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from importlib.metadata import version
import os,sys,json,subprocess
from hashlib import sha256
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from audit_quality import hashfile
from pilot_way_urang import dump
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'sensitivitas_kalibrasi';FIG=OUT/'figures/sensitivitas_kalibrasi'
DAYS=[30,60,90];METHODS=[f'C{d}_median' for d in DAYS];ALL=METHODS+['Persistence']
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def distribution(v):
    x=np.asarray(v,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return {'N':0,'Minimum':None,'P1':None,'P25':None,'Median':None,'P75':None,'P99':None,'Maksimum':None,'Mean':None,'SD':None,'SO2 lebih2.14':0}
    q=np.quantile(x,[0,.01,.25,.5,.75,.99,1])
    return dict(zip(['Minimum','P1','P25','Median','P75','P99','Maksimum'],map(float,q)),N=len(x),Mean=float(x.mean()),SD=float(x.std(ddof=1)) if len(x)>1 else None,**{'SO2 lebih2.14':int((x>2.14).sum())})

def score(frame,method,keep,fold,domain,den):
    e=frame.loc[keep,method].to_numpy()-frame.loc[keep,'SO2 target asli'].to_numpy();n=len(e)
    return {'Fold':fold,'Konfigurasi':method,'Domain':domain,'N':n,'Denominator target asli':int(den),'Cakupan persen':100*n/den if den else None,
        'MAE':float(np.abs(e).mean()) if n else None,'RMSE':float(np.sqrt(np.mean(e**2))) if n else None,'Bias':float(e.mean()) if n else None,
        'Prediksi negatif':int((frame.loc[keep,method]<0).sum()),'SHA256 timestamp':sha256('\n'.join(frame.loc[keep,'Timestamp target'].astype(str)).encode()).hexdigest()}

class Case:
    def __init__(self,e,fold,days,a,b):
        self.e=e;self.fold=fold;self.days=days;self.method=f'C{days}_median';self.a=a;self.b=b;self.w=days*48;self.first=self.w+48
        self.folder=RUN/fold/self.method;self.folder.mkdir(parents=True,exist_ok=True)
    def prepare(self):
        tick=perf_counter();raw=self.e.raw[:self.b-1];filled=raw.copy();self.prefix=[]
        for start in list(range(self.w,self.a,1440))+[self.a]:
            stop=min(start+1440,self.a) if start<self.a else self.b-1
            if start>=stop:continue
            counts=np.isfinite(raw[:start]).sum(axis=0);med=np.full(11,np.nan)
            for j in range(11):
                if counts[j]:med[j]=np.nanmedian(raw[:start,j])
            filled[start:stop]=np.where(np.isfinite(raw[start:stop]),raw[start:stop],med)
            for j,name in enumerate(self.e.matrix_names):
                self.prefix.append({'Fold':self.fold,'Kalibrasi hari':self.days,'Awal query':self.e.times[start],'Akhir query':self.e.times[stop-1],
                    'Cutoff prefix eksklusif':self.e.times[start],'Prefix N':start,'Variabel':name,'Observasi prefix':int(counts[j]),
                    'Median prefix':float(med[j]),'Median tersedia':bool(counts[j]>0),'Periode':'train prefix' if start<self.a else 'evaluasi frozen train'})
        f=pd.DataFrame(filled);r=pd.DataFrame(raw)
        values=np.column_stack([f[0].shift(l).to_numpy() for l in self.e.cfg['so2_lags']]+[f[j].to_numpy() for j in range(1,11)])
        original=np.column_stack([r[0].shift(l).to_numpy() for l in self.e.cfg['so2_lags']]+[r[j].to_numpy() for j in range(1,11)])
        self.X=np.column_stack([values,(~np.isfinite(original)).astype(float)]);self.filled=filled
        ids=np.arange(self.first,self.a-1);observed=np.isfinite(self.e.y[ids+1]);complete=np.isfinite(self.X[ids]).all(axis=1)
        self.train=ids[observed&complete];assert len(self.train)>=340 and (self.train+1<self.a).all()
        self.preprocess_seconds=perf_counter()-tick
        # One row per original train origin; booleans may overlap, primary reasons do not.
        origins=np.arange(self.a);purge=origins==self.a-1;warmup=origins<self.w;buffer=(origins>=self.w)&(origins<self.first)
        missing=np.zeros(self.a,bool);missing[:-1]=~np.isfinite(self.e.y[1:self.a]);nonfinite=~np.isfinite(self.X[origins]).all(axis=1)
        eligible=np.isin(origins,self.train)
        primary=np.select([warmup,buffer,purge,missing,nonfinite],['warmup','buffer lag48','purge target masuk evaluasi','target asli kosong','fitur tetap nonfinite'],default='digunakan')
        ledger=pd.DataFrame({'Index origin':origins,'Origin':self.e.times[origins],'Timestamp target':self.e.times[origins+1],
            'SO2 target asli':np.r_[self.e.y[1:self.a],np.nan],'Warmup':warmup,'Buffer':buffer,'Purge':purge,'Target kosong':missing,
            'Fitur nonfinite':nonfinite,'Histori sebelum awal seri':origins<48,'Digunakan':eligible,'Alasan utama':primary})
        assert np.array_equal(primary=='digunakan',eligible)
        ledger.to_csv(self.folder/'penanda_sampel_train.csv',index=False)
        pd.DataFrame({'Index origin':self.train,'Origin':self.e.times[self.train],'Timestamp target':self.e.times[self.train+1],
            'SO2 target asli':self.e.y[self.train+1]}).to_csv(self.folder/'timestamp_train.csv',index=False)
        self.flow={'Fold':self.fold,'Kalibrasi hari':self.days,'Slot origin train':self.a,'Warmup dikeluarkan':int(warmup.sum()),
            'Buffer lag48':int(buffer.sum()),'Purge batas':1,'Kandidat setelah warmup buffer purge':len(ids),'Target asli kandidat':int(observed.sum()),
            'Target kosong kandidat':int((~observed).sum()),'Fitur nonfinite kandidat target teramati':int((observed&~complete).sum()),'Train dipakai':len(self.train),
            'Jumlah tersisih':self.a-len(self.train)}
        self.reasons=[];self.reason_month=[]
        month=self.e.times[origins].to_period('M')
        for reason in pd.unique(primary):
            keep=primary==reason
            self.reasons.append({'Fold':self.fold,'Kalibrasi hari':self.days,'Alasan utama':reason,'Jumlah origin':int(keep.sum()),**distribution(ledger.loc[keep,'SO2 target asli'])})
            for mo in pd.unique(month):
                sub=keep&(month==mo);self.reason_month.append({'Fold':self.fold,'Kalibrasi hari':self.days,'Bulan origin':str(mo),'Alasan utama':reason,
                    'Jumlah origin':int(sub.sum()),**distribution(ledger.loc[sub,'SO2 target asli'])})
        self.overlap={'Fold':self.fold,'Kalibrasi hari':self.days,'Target kosong semua origin dalam train':int(missing.sum()),
            'Fitur nonfinite semua origin train':int(nonfinite.sum()),'Target kosong dan fitur nonfinite':int((missing&nonfinite).sum()),
            'Warmup dan target kosong':int((warmup&missing).sum()),'Warmup dan fitur nonfinite':int((warmup&nonfinite).sum())}
        self.dist={'Fold':self.fold,'Kalibrasi hari':self.days,'Origin awal':self.e.times[self.train[0]],'Origin akhir':self.e.times[self.train[-1]],
            'Target awal':self.e.times[self.train[0]+1],'Target akhir':self.e.times[self.train[-1]+1],**distribution(self.e.y[self.train+1])}
        self.monthly=[]
        for mo in pd.unique(self.e.times[self.train+1].to_period('M')):
            selected=self.train[self.e.times[self.train+1].to_period('M')==mo]
            self.monthly.append({'Fold':self.fold,'Kalibrasi hari':self.days,'Bulan target':str(mo),**distribution(self.e.y[selected+1])})
        self.gate={'Fold':self.fold,'Kalibrasi hari':self.days,'Ridge fit sebelum gate':self.e.fit_count,'Train N':len(self.train),'Fitur':34,
            'Prefix tanpa median':sum(not r['Median tersedia'] for r in self.prefix),'Lulus':len(self.train)>=340}
        np.savez_compressed(self.folder/'inputs_dan_sampel.npz',X=self.X,filled=filled,train_origins=self.train)
        dump(self.folder/'gate_pra_fit.json',self.gate)
        return self
    def fit_predict(self):
        assert self.e.all_gates and hashfile(RUN/'config_locked.json')==self.e.config_hash
        ids=self.train;X=self.X;target_ids=np.arange(self.a,self.b);target_ids=target_ids[np.isfinite(self.e.y[target_ids])];origin=target_ids-1
        with threadpool_limits(limits=1):
            tick=perf_counter();scaler=StandardScaler().fit(X[ids]);z=scaler.transform(X[ids]);scale_seconds=perf_counter()-tick
            tick=perf_counter();model=Ridge(**self.e.cfg['ridge']).fit(z,self.e.y[ids+1]);fit_seconds=perf_counter()-tick
            good=np.isfinite(X[origin]).all(axis=1);pred=np.full(len(origin),np.nan);tick=perf_counter()
            pred[good]=model.predict(scaler.transform(X[origin[good]]));predict_seconds=perf_counter()-tick
        self.e.fit_count+=1
        self.coefs=[{'Fold':self.fold,'Kalibrasi hari':self.days,'Fitur':name,'Mean train':float(scaler.mean_[j]),'Scale train':float(scaler.scale_[j]),
            'Koefisien standar':float(model.coef_[j]),'Intercept':float(model.intercept_)} for j,name in enumerate(self.e.feature_names)]
        self.timing={'Fold':self.fold,'Kalibrasi hari':self.days,'Preprocessing detik':self.preprocess_seconds,'Scaler fit detik':scale_seconds,
            'Ridge fit detik':fit_seconds,'Predict detik':predict_seconds}
        frame=pd.DataFrame({'Fold':self.fold,'Origin':self.e.times[origin],'Timestamp target':self.e.times[target_ids],
            'Index origin':origin,'Index target':target_ids,'SO2 target asli':self.e.y[target_ids],self.method:pred})
        frame.to_csv(self.folder/'prediksi_evaluasi.csv',index=False);dump(self.folder/'parameter_ridge_scaler.json',self.coefs)
        self.pred=frame;return self

class Sensitivity:
    def prepare(self):
        RUN.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
        old=json.loads((OUT/'backtesting_way_urang/fold_config_locked.json').read_text(encoding='utf-8'))
        self.old_results=json.loads((OUT/'tabel_backtesting_preprocessing.json').read_text(encoding='utf-8'))
        self.old_explanation=(OUT/'diagnosis_fold1.md').read_text(encoding='utf-8')
        self.cuts=old['fold_cuts'];self.data=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=self.cuts[-1],parse_dates=['Timestamp'])
        self.times=pd.DatetimeIndex(self.data.Timestamp);self.y=self.data['SO2 numerik'].to_numpy(dtype=float)
        self.raw=np.column_stack([self.y]+[self.data[f'{c} numerik'].shift(1).to_numpy(dtype=float) for c in old['covariates']])
        assert self.times[-1]<pd.Timestamp(old['split'][2]['Awal']) and (self.times[1:]-self.times[:-1]==pd.Timedelta(minutes=30)).all()
        self.matrix_names=['SO2_origin']+[f'{c}_lag1' for c in old['covariates']]
        names=[f'SO2_lag{lag}' for lag in old['so2_lags']]+[f'{c}_lag1' for c in old['covariates']]
        self.feature_names=names+['missing_asli_'+c for c in names]
        self.refs=[p for p in OUT.rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.relative_to(OUT).parts)
            and 'sensitivitas_kalibrasi' not in p.relative_to(OUT).parts and not p.name.startswith(('hasil_sensitivitas_kalibrasi','ringkasan_sensitivitas_kalibrasi','tabel_sensitivitas_kalibrasi')) and not p.name.endswith('.inspect.ndjson')]
        self.refs+=list((ROOT/'data_raw').glob('*.xlsx'))+[p for p in (ROOT/'notebooks').glob('*.ipynb') if not p.name.startswith('08_')]
        self.refs+=[p for p in (ROOT/'scripts').glob('*') if p.is_file() and p.suffix in ['.py','.mjs'] and not ('calibration' in p.name or 'kalibrasi' in p.name)]
        self.refs += [ROOT/'requirements.txt',ROOT/'requirements-lstm-lock.txt',ROOT/'.venv/pyvenv.cfg',ROOT/'.venv-lstm/pyvenv.cfg']
        self.before={str(p.relative_to(ROOT)):hashfile(p) for p in self.refs}
        self.cfg={'created_utc':datetime.now(timezone.utc).isoformat(),'experiment':'Sensitivitas warmup median/Ridge30/60/90hari, bukan kalibrasi alat',
            'station':'Way Urang pilot sementara','calibration_days':DAYS,'fold_cuts':self.cuts,'folds':old['folds'],'test_start':old['split'][2]['Awal'],
            'so2_lags':old['so2_lags'],'covariates':old['covariates'],'covariate_lag_slots':1,'horizon_slots':1,'horizon_minutes':30,'features':self.feature_names,
            'ridge':old['ridge'],'buffer_slots':48,'prefix_update_slots':1440,'threads':1,'target_scaling':False,
            'warmup_definition':'Data awal disimpan untuk median prefix; tidak diimputasi sebelum warmup dan tidak jadi origin train. Origin train mulai warmup+48, target<awal evaluasi.',
            'imputation':'Median tiap kolom dari raw prefix eksklusif query-start, blok train30hari; evaluasi beku seluruh trainfold. Semua prefix hanya masa lalu. Jika prefix tidak memiliki nilai, median tetapNaN, tanpa fallback.',
            'scaling':'StandardScaler34fitur fit hanya train eligible konfigurasi/fold. Tidak fit pada evaluasi atau seluruh development.',
            'training_cohort':'Mandiri median per konfigurasi, bukan irisan P1/KNN tahap6 atau irisan antar-durasi; komposisi train boleh berubah sesuai tujuan sensitivitas.',
            'evaluation':'Semua target asli teramati pada tanggalfold tetap; irisan3konfigurasi+persistence, domainmandiri juga dilaporkan. Target kosong tidak diimputasi.',
            'interpretation':'Efek gabungan warmup, histori imputasi awaltrain, komposisi dan scalingtrain; bukan pengaruh imputasi terisolasi.',
            'availability':old['availability'],'quality':old['quality'],'development_reuse':old['development_reuse'],
            'test_loaded':False,'test_evaluated':False,'deep_learning_trained':False,'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','scipy']},
            'python':sys.version,'initial_hashes':self.before}
        dump(RUN/'config_locked.json',self.cfg);self.config_hash=hashfile(RUN/'config_locked.json');self.fit_count=0;self.cases=[];self.all_gates=False
        print('Konfigurasi dikunci sebelum12fit:',self.config_hash,flush=True);return self
    def prepare_all(self):
        for number,(a,b) in enumerate(zip(self.cuts[:-1],self.cuts[1:]),1):
            for days in DAYS:self.cases.append(Case(self,f'Fold_{number:02d}',days,a,b).prepare())
        assert self.fit_count==0
        self.all_gates=all(c.gate['Lulus'] for c in self.cases);assert self.all_gates
        self.composition=[];self.legacy=[];self.input_checks=[]
        for number in range(1,5):
            cases=[c for c in self.cases if c.fold==f'Fold_{number:02d}'];c90=cases[-1]
            oldx=pd.read_csv(OUT/f'backtesting_way_urang/{c90.fold}/fitur_P0_median.csv').iloc[:,1:].to_numpy(dtype=float)
            assert np.allclose(c90.X,oldx,equal_nan=True,atol=1e-10,rtol=1e-11)
            oldids=pd.read_csv(OUT/f'backtesting_way_urang/{c90.fold}/timestamp_train_bersama.csv')['Index origin'].to_numpy()
            added=np.setdiff1d(c90.train,oldids);assert np.isin(oldids,c90.train).all()
            self.legacy.append({'Fold':c90.fold,'C90 fitur identik P0 tahap6':True,'Train P0 tahap6 irisan3':len(oldids),'C90 median mandiri':len(c90.train),
                'Tambahan karena tidak memakai irisan P1 KNN':len(added),**distribution(self.y[added+1])})
            for c in cases:
                add=np.setdiff1d(c.train,c90.train);lost=np.setdiff1d(c90.train,c.train)
                self.composition.append({'Fold':c.fold,'Kalibrasi hari':c.days,'Domain':'tambahan dibanding C90',**distribution(self.y[add+1])})
                assert not len(lost)
                origin=np.arange(c.a-1,c.b-1);same=np.allclose(c.X[origin],c90.X[origin],equal_nan=True,atol=1e-10,rtol=1e-11)
                self.input_checks.append({'Fold':c.fold,'Kalibrasi hari':c.days,'C90 fitur vs tahap6 cocok':True,'Fitur evaluasi belum scaling identik antar kalibrasi':same,
                    'Origin evaluasi':len(origin),'Selisih maksimum finite':float(np.nanmax(np.abs(c.X[origin]-c90.X[origin])))})
        dump(RUN/'gate_semua_sebelum_fit.json',{'fit_count':0,'config_sha256':self.config_hash,'gates':[c.gate for c in self.cases]})
        print(pd.DataFrame([c.dist for c in self.cases]).to_string(index=False),flush=True);return self
    def fit_and_summarize(self):
        assert self.all_gates
        for case in self.cases:case.fit_predict()
        assert self.fit_count==12
        from summarize_calibration import summarize
        summarize(self);return self
    def export_excel(self):
        node=Path(r'C:\Users\INTAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
        done=subprocess.run([str(node),str(ROOT/'scripts/export_quality.mjs'),'tabel_sensitivitas_kalibrasi.json','hasil_sensitivitas_kalibrasi.xlsx','.audit_preview_kalibrasi'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        print(done.stdout)
        if done.returncode:raise RuntimeError(done.stderr)
