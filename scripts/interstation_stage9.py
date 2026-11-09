"""Fixed interstation contribution experiment. Development only; no deep learning."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from importlib.metadata import version
import json,subprocess,os,sys
import numpy as np
import pandas as pd
import openpyxl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from audit_quality import hashfile,stamp
from pilot_way_urang import dump
from calibration_sensitivity import distribution
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'kontribusi_antarstasiun';FIG=OUT/'figures/kontribusi_antarstasiun'
SCENARIOS={'S0':[],'S1':['Pulomerak'],'S2':['Talang'],'S3':['Pulomerak','Talang']};OFFSETS=[1,2,3,7,13,25,49];ALL=list(SCENARIOS)+['Persistence']
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def runs(x):
    result=[];start=0
    for stop in range(1,len(x)+1):
        if stop<len(x) and np.isfinite(x[stop]) and np.isfinite(x[stop-1]) and x[stop]==x[stop-1]:continue
        if stop-start>=2 and np.isfinite(x[start]):result.append((start,stop,float(x[start])))
        start=stop
    return result

def source_prefix(path,sheet,end,times,grid_values):
    """Timestamp-only cutoff pass, then SO2-only pass bounded strictly before test."""
    wb=openpyxl.load_workbook(path,read_only=True,data_only=True);ws=wb[sheet]
    units=list(ws.iter_rows(min_row=9,max_row=9,min_col=5,max_col=5,values_only=True))[0][0]
    rows=[];previous=None;last_row=9
    for rownum,row in enumerate(ws.iter_rows(min_row=10,min_col=2,max_col=2,values_only=True),10):
        t=stamp(row[0])
        if pd.notna(t):
            if previous is not None:assert t>=previous,'Urutan sumber berubah; jangan berhenti diam-diam.'
            if t>=end:break # Read boundary timestamp only; no test SO2 accessed.
            previous=t
        rows.append((rownum,t,str(row[0])));last_row=rownum
    values=[row[0] for row in ws.iter_rows(min_row=10,max_row=last_row,min_col=5,max_col=5,values_only=True)]
    wb.close();assert len(rows)==len(values)
    src=pd.DataFrame({'Baris Excel':[r[0] for r in rows],'Timestamp':[r[1] for r in rows],
        'Timestamp literal':[r[2] for r in rows],'SO2 literal':values})
    src['SO2 numerik']=pd.to_numeric(src['SO2 literal'],errors='coerce');src.loc[~np.isfinite(src['SO2 numerik']),'SO2 numerik']=np.nan
    src['On grid']=src.Timestamp.notna()&src.Timestamp.isin(times)
    expected=pd.Series(np.nan,index=times)
    for t,g in src[src['On grid']].groupby('Timestamp',sort=False):
        v=g['SO2 numerik'].to_numpy();expected.loc[t]=v[0] if np.isfinite(v).all() and np.all(v==v[0]) else np.nan
    assert np.allclose(expected.to_numpy(),grid_values,equal_nan=True,atol=1e-10,rtol=1e-11),sheet
    return src,units

def median_history(raw,cut,b,times,label):
    filled=raw[:b-1].copy();logs=[]
    for start in list(range(1440,cut,1440))+[cut]:
        stop=min(start+1440,cut) if start<cut else b-1
        if start>=stop:continue
        med=np.full(raw.shape[1],np.nan)
        for j in range(raw.shape[1]):
            v=raw[:start,j];v=v[np.isfinite(v)]
            if len(v):med[j]=np.median(v)
            logs.append({'Seri':label,'Awal query':times[start],'Akhir query':times[stop-1],'Cutoff eksklusif':times[start],
                'Prefix N':start,'Kolom':j,'Observasi prefix':len(v),'Median':float(med[j]),'Beku evaluasi':start==cut})
        filled[start:stop]=np.where(np.isfinite(raw[start:stop]),raw[start:stop],med)
    return filled,logs

class Fold:
    def __init__(self,e,number,a,b):
        self.e=e;self.label=f'Fold_{number:02d}';self.a=a;self.b=b;self.folder=RUN/self.label;self.folder.mkdir(exist_ok=True)
    def prepare(self):
        tick=perf_counter();e=self.e;a=self.a;b=self.b
        way,logs=median_history(e.way_matrix,a,b,e.times,'Way Urang');self.way_filled=way;self.prefix=logs
        f=pd.DataFrame(way);r=pd.DataFrame(e.way_matrix[:b-1])
        numeric=np.column_stack([f[0].shift(l) for l in e.cfg['so2_lags']]+[f[j] for j in range(1,11)])
        original=np.column_stack([r[0].shift(l) for l in e.cfg['so2_lags']]+[r[j] for j in range(1,11)])
        base=np.column_stack([numeric,(~np.isfinite(original)).astype(float)])
        old=np.load(OUT/f'sensitivitas_kalibrasi/{self.label}/C30_median/inputs_dan_sampel.npz')
        assert np.allclose(base,old['X'],equal_nan=True,atol=1e-10,rtol=1e-11)
        self.donor_filled={};blocks={};self.names={'S0':list(e.cfg['base_features'])};self.X={'S0':base}
        for donor in ['Pulomerak','Talang']:
            # Available at first origin=a-1 only through donor source index a-2.
            filled,logs=median_history(e.donor[donor][:,None],a-1,b,e.times,donor);self.donor_filled[donor]=filled[:,0]
            self.prefix += logs
            numerical=np.column_stack([pd.Series(filled[:,0]).shift(k) for k in OFFSETS])
            raw=np.column_stack([pd.Series(e.donor[donor][:b-1]).shift(k) for k in OFFSETS])
            blocks[donor]=np.column_stack([numerical,(~np.isfinite(raw)).astype(float)])
        for scenario,donors in SCENARIOS.items():
            if scenario=='S0':continue
            self.X[scenario]=np.column_stack([base]+[blocks[d] for d in donors])
            self.names[scenario]=list(e.cfg['base_features'])
            for donor in donors:
                names=[f'{donor}_SO2_origin_minus{k}' for k in OFFSETS]
                self.names[scenario] += names+['missing_asli_'+n for n in names]
        self.own={};self.shared={};self.flow=[];self.lost=[];self.lost_month=[];self.dist=[];self.month_dist=[]
        self.parts={'train':np.arange(1488,a-1),'evaluation':np.arange(a-1,b-1)}
        for part,ids in self.parts.items():
            target_obs=ids[np.isfinite(e.y[ids+1])];shared=target_obs
            for scenario,X in self.X.items():
                buffer_ok=ids>=(1488 if scenario=='S0' else 1489)
                valid=np.isfinite(e.y[ids+1])&np.isfinite(X[ids]).all(axis=1)&buffer_ok
                self.own[scenario,part]=ids[valid];shared=np.intersect1d(shared,ids[valid])
                self.flow.append({'Fold':self.label,'Bagian':part,'Skenario':scenario,'Kandidat setelah warmup buffer dasar purge':len(ids),
                    'Target asli teramati':len(target_obs),'Buffer tambahan donor49 pada target teramati':int((~buffer_ok&np.isfinite(e.y[ids+1])).sum()),
                    'Fitur nonfinite pada target teramati':int((~np.isfinite(X[ids]).all(axis=1)&np.isfinite(e.y[ids+1])).sum()),'Layak mandiri':int(valid.sum())})
            self.shared[part]=shared
            for scenario in SCENARIOS:
                own=self.own[scenario,part];lost=np.setdiff1d(own,shared)
                row=next(r for r in self.flow if r['Bagian']==part and r['Skenario']==scenario)
                row['Irisan digunakan']=len(shared);row['Keluar karena irisan']=len(lost)
                self.lost.append({'Fold':self.label,'Bagian':part,'Skenario':scenario,**distribution(e.y[lost+1])})
                for mo in e.times[ids+1].to_period('M').unique():
                    sub=lost[e.times[lost+1].to_period('M')==mo];self.lost_month.append({'Fold':self.label,'Bagian':part,'Skenario':scenario,'Bulan target':str(mo),**distribution(e.y[sub+1])})
            for domain,selected in [('target asli kandidat',target_obs),('irisan digunakan',shared)]:
                self.dist.append({'Fold':self.label,'Bagian':part,'Domain':domain,'Awal target':e.times[selected[0]+1] if len(selected) else None,
                    'Akhir target':e.times[selected[-1]+1] if len(selected) else None,**distribution(e.y[selected+1])})
                for mo in e.times[ids+1].to_period('M').unique():
                    sub=selected[e.times[selected+1].to_period('M')==mo];self.month_dist.append({'Fold':self.label,'Bagian':part,'Domain':domain,'Bulan target':str(mo),**distribution(e.y[sub+1])})
            ledger=pd.DataFrame({'Index origin':ids,'Origin':e.times[ids],'Timestamp target':e.times[ids+1],'Target asli':e.y[ids+1],'Irisan digunakan':np.isin(ids,shared)})
            for scenario in SCENARIOS:ledger[scenario+'_layak_mandiri']=np.isin(ids,self.own[scenario,part])
            ledger.to_csv(self.folder/f'ledger_sampel_{part}.csv',index=False)
            pd.DataFrame({'Index origin':shared,'Origin':e.times[shared],'Timestamp target':e.times[shared+1],'SO2 target asli':e.y[shared+1]}).to_csv(self.folder/f'timestamp_{part}_bersama.csv',index=False)
        self.gate={'Fold':self.label,'Fit count sebelum gate':e.fit_count,'Train bersama':len(self.shared['train']),
            'Evaluasi bersama':len(self.shared['evaluation']),'Minimum train komputasional':620,'S0 train layak mandiri':len(self.own['S0','train']),
            'S0 hilang akibat irisan':len(np.setdiff1d(self.own['S0','train'],self.shared['train'])),
            'Lulus':len(self.shared['train'])>=620 and len(self.shared['evaluation'])>0,'S0 fitur identik C30 lama':True}
        self.constant_usage=[]
        for donor in ['Pulomerak','Talang']:
            for part,ids in self.shared.items():
                used=np.unique((ids[:,None]-np.array(OFFSETS)[None,:]).ravel());assert used.min()>=0
                for run_id,(start,stop,value) in enumerate(e.runs[donor],1):
                    selected=used[(used>=start)&(used<stop)]
                    if not len(selected):continue
                    self.constant_usage.append({'Fold':self.label,'Donor':donor,'Bagian sampel model':part,'Run ID':run_id,'Nilai SO2 asli':value,
                        'Run mulai':e.times[start],'Run akhir':e.times[stop-1],'Jumlah slot run':stop-start,'Durasi run menit':(stop-start)*30,
                        'Timestamp sumber unik masuk fitur':len(selected),'Sumber dipakai awal':e.times[selected[0]],'Sumber dipakai akhir':e.times[selected[-1]],
                        'Terpotong batas development':stop==len(e.times)})
        self.preprocess_seconds=perf_counter()-tick
        np.savez_compressed(self.folder/'inputs_train_eval.npz',**{s:self.X[s] for s in SCENARIOS},way_filled=way,
            Pulomerak_filled=self.donor_filled['Pulomerak'],Talang_filled=self.donor_filled['Talang'],train=self.shared['train'],evaluation=self.shared['evaluation'])
        dump(self.folder/'nama_fitur.json',self.names);dump(self.folder/'gate_pra_fit.json',self.gate)
        print('Gate sebelum semua model:',self.gate,flush=True);return self
    def fit(self):
        assert self.e.gates_pass and hashfile(RUN/'config_locked.json')==self.e.config_hash
        e=self.e;train=self.shared['train'];target_ids=np.arange(self.a,self.b);target_ids=target_ids[np.isfinite(e.y[target_ids])];origin=target_ids-1
        self.pred=pd.DataFrame({'Fold':self.label,'Origin':e.times[origin],'Timestamp target':e.times[target_ids],'Index origin':origin,
            'Index target':target_ids,'SO2 target asli':e.y[target_ids]})
        self.coefficients=[];self.timing=[]
        with threadpool_limits(limits=1):
            for scenario,X in self.X.items():
                tick=perf_counter();scaler=StandardScaler().fit(X[train]);Z=scaler.transform(X[train]);st=perf_counter()-tick
                tick=perf_counter();model=Ridge(**e.cfg['ridge']).fit(Z,e.y[train+1]);ft=perf_counter()-tick;e.fit_count+=1
                good=np.isin(origin,self.own[scenario,'evaluation']);p=np.full(len(origin),np.nan);tick=perf_counter()
                p[good]=model.predict(scaler.transform(X[origin[good]]));pt=perf_counter()-tick;self.pred[scenario]=p
                self.timing.append({'Fold':self.label,'Skenario':scenario,'Train N':len(train),'Fitur N':X.shape[1],'Scaler fit detik':st,'Ridge fit detik':ft,'Predict detik':pt})
                self.coefficients += [{'Fold':self.label,'Skenario':scenario,'Fitur':name,'Mean train':float(scaler.mean_[j]),'Scale train':float(scaler.scale_[j]),
                    'Koefisien standar':float(model.coef_[j]),'Intercept':float(model.intercept_)} for j,name in enumerate(self.names[scenario])]
        last=np.maximum.accumulate(np.where(np.isfinite(e.y[:self.b-1]),np.arange(self.b-1),-1))
        self.pred['Persistence']=e.y[last[origin]];self.pred['Persistence usia slot']=origin-last[origin]
        self.pred['Irisan semua skenario']=np.isfinite(self.pred[ALL]).all(axis=1)
        assert np.array_equal(origin[self.pred['Irisan semua skenario']],self.shared['evaluation'])
        for donor in ['Pulomerak','Talang']:
            age=e.ages[donor].iloc[origin].reset_index(drop=True)
            for col in age.columns:
                if col!='Origin':self.pred[donor+' '+col]=age[col].to_numpy()
        self.pred.to_csv(self.folder/'prediksi.csv',index=False);dump(self.folder/'parameter_ridge_scaler.json',self.coefficients)
        return self

class Experiment:
    def prepare(self):
        RUN.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
        base=json.loads((OUT/'sensitivitas_kalibrasi/config_locked.json').read_text(encoding='utf-8'));self.base=base;self.cuts=base['fold_cuts'];self.end=pd.Timestamp(base['test_start'])
        self.data=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=self.cuts[-1],parse_dates=['Timestamp']);self.times=pd.DatetimeIndex(self.data.Timestamp)
        self.y=self.data['SO2 numerik'].to_numpy(dtype=float);self.way_matrix=np.column_stack([self.y]+[self.data[f'{c} numerik'].shift(1).to_numpy(dtype=float) for c in base['covariates']])
        assert self.times[-1]<self.end and (self.times[1:]-self.times[:-1]==pd.Timedelta(minutes=30)).all()
        self.donor={};self.source={};self.grid={};self.ages={};self.runs={};self.offgrid=[];self.source_checks=[];self.quality=[]
        source=next((ROOT/'data_raw').glob('*.xlsx'));self.source_sha_before=hashfile(source)
        specs={'Way Urang':('02','Kab. Lampung Selatan Way Urang'),'Pulomerak':('03','Kota Cilegon Pulomerak'),'Talang':('01','Bandar Lampung Talang')}
        self.units={}
        for name,(number,sheet) in specs.items():
            g=pd.read_csv(OUT/f'audit_grids/{number}_grid_audit.csv',nrows=self.cuts[-1],parse_dates=['Timestamp'])
            assert pd.DatetimeIndex(g.Timestamp).equals(self.times)
            x=g['SO2 numerik untuk grafik'].to_numpy(dtype=float)
            if name=='Way Urang':assert np.allclose(x,self.y,equal_nan=True)
            src,unit=source_prefix(source,sheet,self.end,self.times,x);self.source[name]=src;self.units[name]=unit;self.grid[name]=g
            src.to_csv(RUN/f'sumber_dev_{name.replace(" ","_")}.csv',index=False);g.to_csv(RUN/f'grid_dev_{name.replace(" ","_")}.csv',index=False)
            off=src[~src['On grid']]
            for row in off.to_dict('records'):self.offgrid.append({'Stasiun':name,**row})
            self.source_checks.append({'Stasiun':name,'Sheet':sheet,'Satuan literal':unit,'Slot development':len(x),'Baris sumber sebelum test':len(src),
                'Nilai numerik grid cocok sumber':True,'Offgrid baris':len(off),'Offgrid SO2 numerik nonkosong':int(off['SO2 numerik'].notna().sum())})
            self.quality.append({'Stasiun':name,**distribution(x),'Slot development':len(x),'Slot SO2 kosong/ambigu/nonnumerik':int((~np.isfinite(x)).sum()),
                'Ada baris':int(g['Ada baris'].sum()),'Ambigu duplikat':int(g['SO2 ambigu duplikat'].sum()),'Nol':int((x==0).sum()),'Negatif':int((x<0).sum())})
            if name=='Way Urang':continue
            self.donor[name]=x;self.runs[name]=runs(x)
            positions=np.arange(len(x));last=np.maximum.accumulate(np.where(np.isfinite(x),positions,-1));cut=positions-1
            validcut=cut>=0;gridlast=np.full(len(x),-1);gridlast[validcut]=last[cut[validcut]];age_grid=np.full(len(x),np.nan)
            seen=gridlast>=0;age_grid[seen]=(positions[seen]-gridlast[seen])*30
            observed=src[src['SO2 numerik'].notna()&src.Timestamp.notna()].sort_values('Timestamp');ot=pd.DatetimeIndex(observed.Timestamp)
            bounds=self.times-pd.Timedelta(minutes=30);ix=ot.searchsorted(bounds,side='right')-1;has=ix>=0
            dt=pd.Series(pd.NaT,index=positions,dtype='datetime64[ns]');dt.loc[has]=ot[ix[has]].values
            age_all=(pd.Series(self.times,index=positions)-dt).dt.total_seconds()/60
            self.ages[name]=pd.DataFrame({'Origin':self.times,'Batas donor terbaru':bounds,'Timestamp asli terakhir semua':dt,
                'Usia asli menit semua':age_all,'Timestamp asli terakhir grid':[self.times[i] if i>=0 else pd.NaT for i in gridlast],
                'Usia asli menit grid':age_grid,'Usia tambahan setelah cutoff menit grid':age_grid-30})
            self.ages[name].to_csv(RUN/f'usia_ketersediaan_{name}.csv',index=False)
        pd.DataFrame(self.offgrid).to_csv(RUN/'ledger_offgrid_dev.csv',index=False)
        tracked=subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
        self.refs=[ROOT/p for p in tracked if p and not p.startswith(('notebooks/09_','outputs/kontribusi_antarstasiun/','outputs/figures/kontribusi_antarstasiun/'))]
        self.before={p.relative_to(ROOT).as_posix():hashfile(p) for p in self.refs}
        self.cfg={'created_utc':datetime.now(timezone.utc).isoformat(),'experiment':'Tahap9 kontribusi antarstasiun Ridge/median C30','target':'SO2 Way Urang asli','scenarios':SCENARIOS,
            'fold_cuts':self.cuts,'folds':base['folds'],'test_start':str(self.end),'horizon_minutes':30,'covariates':base['covariates'],'so2_lags':base['so2_lags'],
            'base_features':base['features'],'donor_relative_lags':[0,1,2,6,12,24,48],'donor_delay_slots':1,'donor_origin_offsets':OFFSETS,'feature_counts':{'S0':34,'S1':48,'S2':48,'S3':62},
            'warmup_slots':1440,'prefix_update_slots':1440,'base_buffer_slots':48,'donor_buffer_slots':49,'ridge':base['ridge'],'target_scaling':False,'threads':1,
            'way_imputation':'C30 identik tahap8: median rawprefix sebelumquery, update30hari; frozen rawWay[:a] untukquery>=a.',
            'donor_imputation':'Median seri donor terpisah; trainprefix kausal. Frozen rawdonor[:a-1] (maks timestampa-2) tersedia pada origin evaluasi pertamaa-1; tidak memakai seluruhdevelopment.',
            'alignment_assumption':'Timestamp tertulis tanpa zona waktu dianggap pada clock yang sebanding; grid00/30 exact. Tidak konversi zona, rounding offgrid, backwardfill atau interpolasi dua arah. Sinkronisasi fisik belum dikonfirmasi.',
            'latency_assumption':'SO2 donor timestampv diasumsikan tersedia pada originv+1slot; bukan latensi operasional terverifikasi. Semua fitur donor <=origin-30menit.',
            'missing_policy':'Missing asli tiaplag donor menjadi fitur; usia asli hanya diagnostik, bukan fitur. Median tanpa donor tersedia tetapNaN, tanpa fallback.',
            'age_definition':'Usia menit=origin−timestamp original numerik terakhir <=origin−30menit; grid dan semuaoriginaltermasukoffgrid dilaporkan terpisah. Tidak menyatakan pengamatan asli valid otomatis.',
            'train_policy':'Irisan S0–S3, targetasli finite, fit scaler dan Ridge per scenario di timestamptrain sama; purge target>=awal evaluasi.',
            'evaluation_policy':'Target WayUrang asli, irisan4skenario+persistence dan domainmandiri; cakupan memakai seluruh target asli fold. Tidak ada donor untuk mengisi target.',
            'quality':base['quality'],'constant_runs':'Kesamaan numerik tepat >=2slot, missing memutus; durasiNslot×30menit. Statistik penggunaan berdasarkan sumberunik yang masukfitur train/eval bersama; bukan label kerusakan.',
            'units_literal':self.units,'test_loaded':False,'test_evaluated':False,'deep_learning_trained':False,
            'source_reading':'Baca timestamp saja sampai batas test; SO2 sumber dibaca hanya baris sebelumnya. Griddibatasi25645slot. Tidak mengakses nilaiSO2 test.',
            'source_sha256_before_read':self.source_sha_before,
            'versions':{k:version(k) for k in ['numpy','pandas','scikit-learn','scipy','openpyxl']},'initial_hashes':self.before}
        dump(RUN/'config_locked.json',self.cfg);self.config_hash=hashfile(RUN/'config_locked.json');self.fit_count=0;self.folds=[];self.gates_pass=False
        print('Rancangan terkunci sebelum fit:',self.config_hash,flush=True);return self
    def prepare_all(self):
        for i,(a,b) in enumerate(zip(self.cuts[:-1],self.cuts[1:]),1):self.folds.append(Fold(self,i,a,b).prepare())
        self.gates_pass=all(f.gate['Lulus'] for f in self.folds)
        dump(RUN/'gate_semua_sebelum_fit.json',{'fit_count':self.fit_count,'config_sha256':self.config_hash,'gates':[f.gate for f in self.folds]})
        assert self.fit_count==0 and self.gates_pass,'Sampel tidak cukup; jangan mengubah split untuk mengejar skor.'
        return self
    def train(self):
        for f in self.folds:f.fit()
        assert self.fit_count==16
        from summarize_interstation_stage9 import summarize
        summarize(self);return self
    def export_excel(self):
        node=Path(r'C:\Users\INTAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
        p=subprocess.run([str(node),str(ROOT/'scripts/export_quality.mjs'),'tabel_kontribusi_antarstasiun.json','hasil_kontribusi_antarstasiun.xlsx','.audit_preview_antarstasiun'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        print(p.stdout)
        if p.returncode:raise RuntimeError(p.stderr)
