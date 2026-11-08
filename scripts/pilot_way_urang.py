"""Pilot rekonstruksi SO2 pada validasi; tidak mengevaluasi test/forecasting."""
from pathlib import Path
from datetime import datetime,date
from time import perf_counter
from importlib.metadata import version
import os,json,warnings,platform,subprocess
import numpy as np
import pandas as pd
import openpyxl
from audit_quality import hashfile,stamp,empty
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer,IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
RUN=OUT/'pilot_way_urang'; FIG=OUT/'figures/pilot_way_urang'
STATION='Kab. Lampung Selatan Way Urang'
SEEDS=[11,23,47]
CANDIDATES=['PM10','PM2.5','CO','O3','NO2','Kec.Angin','Kelembaban','Suhu','Tek.Udara','Sol.Rad','Curah Hujan']

def jsonsafe(v):
    if isinstance(v,(datetime,pd.Timestamp)):return v.isoformat(sep=' ')
    if isinstance(v,date):return v.isoformat()
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.bool_):return bool(v)
    if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
    return v

def dump(path,obj):
    # Pandas -> JSON handles NaN without writing nonstandard JSON literals.
    def recurse(v):
        if isinstance(v,dict):return {k:recurse(x) for k,x in v.items()}
        if isinstance(v,list):return [recurse(x) for x in v]
        return jsonsafe(v)
    path.write_text(json.dumps(recurse(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def missing_runs(y):
    result=[]; i=0
    while i<len(y):
        if np.isfinite(y[i]):i+=1;continue
        start=i
        while i<len(y) and not np.isfinite(y[i]):i+=1
        result.append((start,i-1,i-start))
    return result

class Pilot:
    def prepare(self):
        RUN.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
        self.source=next((ROOT/'data_raw').glob('*.xlsx'))
        self.references=[self.source,OUT/'rancangan_eksperimen_preprocessing.md',OUT/'audit_kualitas_tables.json',
            OUT/'pemeriksaan_lanjutan_tables.json',OUT/'audit_grids/02_grid_audit.csv']
        self.before={str(p.relative_to(ROOT)):hashfile(p) for p in self.references}
        # Baca rancangan dan struktur audit, tanpa mengubahnya.
        design=(OUT/'rancangan_eksperimen_preprocessing.md').read_text(encoding='utf-8')
        assert all(tag in design for tag in ['P0','P1','P2','P3'])
        audit=json.loads((OUT/'audit_kualitas_tables.json').read_text(encoding='utf-8'))
        self.quality_reference=[r for r in audit['Ringkasan'] if r['Stasiun']==STATION][0]
        follow=json.loads((OUT/'pemeriksaan_lanjutan_tables.json').read_text(encoding='utf-8'))
        wb=openpyxl.load_workbook(self.source,read_only=True,data_only=False)
        try:
            ws=wb[STATION]; preview=list(ws.iter_rows(min_row=1,max_row=9,values_only=True))
            header=list(preview[7][1:15]); assert header[0]=='Waktu' and header[3]=='SO2'
            raw=pd.DataFrame([r[1:15] for r in ws.iter_rows(min_row=10,max_col=15,values_only=True)],columns=header,dtype=object)
        finally:wb.close()
        raw.insert(0,'Baris Excel',np.arange(10,10+len(raw)))
        ts=pd.Series([stamp(v) for v in raw['Waktu']],dtype='datetime64[ns]')
        assert ts.notna().all() and ts.is_unique and ts.is_monotonic_increasing
        raw['Timestamp audit']=ts
        off=ts.ne(ts.dt.floor('30min')); self.offgrid=raw.loc[off].copy()
        assert len(self.offgrid)==1 and self.offgrid['SO2'].map(empty).all()
        self.offgrid.to_csv(RUN/'offgrid_asli.csv',index=False)
        # Salinan arsip untuk semua kolom dan baris; tidak mengganti sumber.
        raw.to_csv(RUN/'salinan_baris_asli.csv',index=False)
        frame=raw.loc[~off].set_index('Timestamp audit')
        grid=pd.date_range(ts.min(),ts.max(),freq='30min')
        data=frame.reindex(grid); data.index.name='Timestamp'
        data['Ada baris sumber']=data['Baris Excel'].notna()
        data['SO2 kosong asli']=data['SO2'].map(empty)
        data['SO2 rentetan konstan audit']=False
        for r in audit['Rentetan_SO2']:
            if r['Stasiun']==STATION:
                data.loc[pd.Timestamp(r['Awal']):pd.Timestamp(r['Akhir']),'SO2 rentetan konstan audit']=True
        for col in header[1:]:
            values=pd.to_numeric(data[col],errors='coerce').astype(float)
            data[f'{col} numerik']=values
            data[f'{col} nol asli']=values.eq(0)
            data[f'{col} negatif asli']=values.lt(0)
        data['Kelembaban di luar 0-100']=data['Kelembaban numerik'].lt(0)|data['Kelembaban numerik'].gt(100)
        data['Arah Angin di luar 0-360']=data['Arah Angin numerik'].lt(0)|data['Arah Angin numerik'].gt(360)
        n=len(grid); a=int(np.floor(.7*n)); b=int(np.floor(.85*n))
        self.a=a;self.b=b;self.grid=grid;self.n=n
        data['Split']=np.where(np.arange(n)<a,'train',np.where(np.arange(n)<b,'validation','test'))
        # Penanda kualitas dan nilai asli terpisah dari masukan algoritma.
        data.to_csv(RUN/'grid_dan_penanda_asli.csv')
        self.split=[]
        for label,lo,hi in [('train',0,a),('validation',a,b),('test',b,n)]:
            part=data.iloc[lo:hi]
            self.split.append({'Split':label,'Awal':grid[lo],'Akhir':grid[hi-1],'Jumlah slot':hi-lo,
                'Jumlah baris sumber':int(part['Ada baris sumber'].sum()),
                'SO2 nonkosong asli':int((~part['SO2 kosong asli']).sum()),
                'Status':'Hanya inventaris, tidak dievaluasi' if label=='test' else 'Digunakan sesuai kontrak'})
        # Tidak memberikan nilai test ke fit/transform/masking/evaluasi.
        work=data.iloc[:b].copy()
        self.y=work['SO2 numerik'].to_numpy(copy=True)
        self.times=grid[:b];self.val_times=grid[a:b]
        self.constant=work['SO2 rentetan konstan audit'].to_numpy(copy=True)
        cov=pd.DataFrame({c:work[f'{c} numerik'] for c in CANDIDATES}).shift(1)
        self.features=[];self.feature_info=[]
        for c in CANDIDATES:
            train=cov[c].iloc[:a];use=int(train.notna().sum())>=2 and train.nunique()>1
            self.feature_info.append({'Fitur':c,'Lag slot':1,'Nonkosong train':int(train.notna().sum()),
                'Denominator slot train':a,'Train teramati persen':100*train.notna().sum()/a,
                'Nilai unik train':int(train.nunique()),'Dipakai P2 dan P3':use,
                'Alasan':'Numerik, bervariasi dan tersedia pada train; hanya t-30 menit.' if use else 'Kosong/konstan pada train; tidak informatif.'})
            if use:self.features.append(c)
        self.matrix=np.column_stack([self.y,cov[self.features].to_numpy(dtype=float)])
        assert len(self.matrix)==b and self.times[a-1]<self.val_times[0]
        self.train_gaps=[]
        for first,last,length in missing_runs(self.y[:a]):
            self.train_gaps.append({'Awal':grid[first],'Akhir':grid[last],'Jumlah slot':length,'Durasi jam':length*.5})
        lengths=[r['Jumlah slot'] for r in self.train_gaps]
        long=[v for v in lengths if v>4]
        self.block_lengths=sorted(set([4]+[int(v) for v in np.quantile(long,[0,.5,.9,1],method='nearest')]))
        assert all(v in lengths for v in self.block_lengths)
        self.limits=[1]+self.block_lengths
        self.config={'stasiun':STATION,'status':'pilot sementara, bukan stasiun final',
            'split':'floor(0.70*N) slot train; floor(0.85*N) kumulatif validation; sisanya test',
            'seeds':SEEDS,'point_fraction':.10,'block_lengths_slots':self.block_lengths,
            'block_events':'max(1,min(30,floor(0.10*N_observed_validation/L)))',
            'block_selection':'truth tersedia seluruh blok, buffer 1 slot teramati, tidak overlap dalam skenario',
            'p1_limits_slots':self.limits,'p2_k':[3,5,10],'p2_weights':'distance','p2_metric':'nan_euclidean',
            'p3':{'estimator':'BayesianRidge','max_iter':20,'tol':.001,'initial_strategy':'median',
                  'imputation_order':'ascending','random_state':42,'sample_posterior':False,'skip_complete':False},
            'common_covariates':self.features,'covariate_lag_slots':1,'query_min_observed_covariates':2,
            'scaling':'StandardScaler fit train only, NaN retained, same matrix for P2/P3',
            'no_raw_time':True,'wind_direction_excluded':True,'test_evaluated':False,
            'availability_assumption':'Pengukuran bertimestamp <= t-30 menit diasumsikan telah tersedia pada t; latensi aktual belum dikonfirmasi.',
            'quality_policy':'Nol, lonjakan, konstan dan ekstrem dipertahankan. Off-grid terpisah tanpa pembulatan.',
            'scope':'Mask SO2 saja; kovariat lag tetap pengukuran asli, bukan simulasi semua sensor hilang.',
            'thread_limit':1,'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','scipy']},
            'platform':platform.platform(),'python':platform.python_version()}
        dump(RUN/'config.json',self.config)
        self.quality=[{'Keputusan':'Off-grid','Status':'Satu baris 2025-05-22 19:01:00, SO2 kosong. Ledger dipertahankan, tidak dipetakan.'},
            {'Keputusan':'Ketersediaan kovariat','Status':self.config['availability_assumption']},
            {'Keputusan':'Validitas sensor','Status':'Belum dikonfirmasi: kode/nilai 0.01, lonjakan, rentetan konstan, satuan/ambang dan latensi. Semua nilai asli dipertahankan.'},
            {'Keputusan':'Kebenaran evaluasi','Status':'Nilai asli teramati, bukan jaminan lolos validasi sensor. Missing alami bukan truth.'},
            {'Keputusan':'Test','Status':'Sudah tercakup audit tahap sebelumnya; pada pilot ini hanya inventaris, tidak dihitung skor atau ditransformasi.'}]
        print('Persiapan selesai:',self.split,flush=True)
        return self

    def make_masks(self):
        val=self.y[self.a:self.b]; available=np.isfinite(val); count=int(available.sum())
        self.scenarios=[];self.mask_ledger=[];self.truth_ledger=[];self.events=[]
        for seed in SEEDS:
            rng=np.random.default_rng(seed)
            for pattern,length in [('titik',1)]+[('blok',v) for v in self.block_lengths]:
                ids=[]; blocks=[]; occupied=np.zeros(len(val),bool)
                if pattern=='titik':
                    desired=int(np.floor(.1*count));candidates=rng.permutation(np.flatnonzero(available))
                    for k in candidates:
                        if not occupied[max(0,k-1):k+2].any():
                            ids.append(int(k));blocks.append([int(k)]);occupied[k]=True
                            if len(ids)==desired:break
                    assert len(ids)==desired
                else:
                    desired=max(1,min(30,int(np.floor(.1*count/length))))
                    candidates=[j for j in range(1,len(val)-length) if available[j-1:j+length+1].all()]
                    for j in rng.permutation(candidates):
                        if not occupied[j-1:j+length+1].any():
                            group=list(range(int(j),int(j)+length));ids+=group;blocks.append(group)
                            occupied[j:j+length]=True
                            if len(blocks)==desired:break
                sid=f'{seed}_{pattern}_{length}';ids=np.array(sorted(ids),dtype=int)
                assert len(ids)>0 and available[ids].all()
                self.scenarios.append({'id':sid,'seed':seed,'pattern':pattern,'length':length,'ids':ids})
                for e,group in enumerate(blocks,1):
                    self.events.append({'Skenario':sid,'Seed':seed,'Pola':pattern,'Panjang slot':length,
                        'Kejadian':e,'Awal':self.val_times[group[0]],'Akhir':self.val_times[group[-1]],'Jumlah slot':len(group)})
                    for i in group:
                        mid=f'{sid}_{i}'
                        self.mask_ledger.append({'Mask ID':mid,'Skenario':sid,'Seed':seed,'Pola':pattern,'Panjang slot':length,
                            'Kejadian':e,'Index validasi':i,'Timestamp':self.val_times[i],
                            'SO2 konstan audit':bool(self.constant[self.a+i])})
                        self.truth_ledger.append({'Mask ID':mid,'SO2 asli untuk skor saja':float(val[i])})
                assert len(blocks)==desired,(sid,len(blocks),desired)
        pd.DataFrame(self.mask_ledger).to_csv(RUN/'lokasi_masking.csv',index=False)
        pd.DataFrame(self.truth_ledger).to_csv(RUN/'truth_penilaian_saja.csv',index=False)
        pd.DataFrame(self.events).to_csv(RUN/'kejadian_masking.csv',index=False)
        self.union_ids=np.unique(np.concatenate([s['ids'] for s in self.scenarios]))
        print('Masking:',len(self.scenarios),'skenario;',len(self.mask_ledger),'sel; union',len(self.union_ids),flush=True)
        return self

    def fit(self):
        self.fit_times={};self.cache={};self.failures=[];self.params=[];self.warn=[]
        train=self.matrix[:self.a].copy(); assert len(train)==self.a
        tick=perf_counter();self.median=float(np.nanmedian(train[:,0]));self.fit_times['P0']=perf_counter()-tick
        self.params.append({'Parameter':'median SO2 train','Nilai':self.median})
        for limit in self.limits:self.fit_times[f'P1_L{limit}']=0.0
        tick=perf_counter();self.scaler=StandardScaler().fit(train);self.scaler_time=perf_counter()-tick
        scaled=self.scaler.transform(train)
        self.scaling=[]
        for i,c in enumerate(['SO2']+self.features):
            self.scaling.append({'Fitur':c,'Mean train':self.scaler.mean_[i],'Skala train':self.scaler.scale_[i],
                'Jumlah nilai untuk scaler':int(np.isfinite(train[:,i]).sum())})
        self.donors=scaled[np.isfinite(train[:,0])].copy()
        self.models={}
        with threadpool_limits(limits=1):
            for k in [3,5,10]:
                name=f'P2_k{k}';tick=perf_counter()
                self.models[name]=KNNImputer(n_neighbors=k,weights='distance',metric='nan_euclidean').fit(self.donors)
                self.fit_times[name]=perf_counter()-tick
            tick=perf_counter()
            p3=IterativeImputer(estimator=BayesianRidge(),**{k:v for k,v in self.config['p3'].items() if k!='estimator'})
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');p3.fit(scaled)
                self.models['P3']=p3
                self.warn += [{'Metode':'P3','Tahap':'fit','Kategori':type(w.message).__name__,'Pesan':str(w.message)} for w in caught]
                self.params.append({'Parameter':'P3 iterasi nyata','Nilai':int(p3.n_iter_)})
            except Exception as exc:
                self.models['P3']=None;self.failures.append({'Metode':'P3','Tahap':'fit','Pesan':repr(exc)})
            self.fit_times['P3']=perf_counter()-tick
        self.train_min=float(np.nanmin(train[:,0]));self.train_max=float(np.nanmax(train[:,0]))
        self.params += [{'Parameter':'SO2 train minimum','Nilai':self.train_min},{'Parameter':'SO2 train maksimum','Nilai':self.train_max},
            {'Parameter':'Jumlah donor KNN SO2 asli train','Nilai':len(self.donors)}]
        dump(RUN/'parameter_train.json',{'parameter':self.params,'scaler':self.scaling,'fit_seconds':self.fit_times,
            'scaler_fit_seconds':self.scaler_time,'train_end':str(self.times[self.a-1]),'validation_start':str(self.val_times[0])})
        print('Fit selesai; seluruh parameter dari train, tidak ada test.',flush=True)
        return self

    def transform_multivariate(self,ids):
        # Query SO2 SELALU NaN, tidak pernah menyalin truth validasi.
        query=np.full((len(ids),self.matrix.shape[1]),np.nan)
        query[:,1:]=self.matrix[self.a+ids,1:]
        assert np.isnan(query[:,0]).all()
        qscaled=self.scaler.transform(query)
        query_known=np.isfinite(query[:,1:]).sum(axis=1)
        donor_has=np.isfinite(self.donors[:,1:]).any(axis=0)
        supported=(query_known>=2)&np.any(np.isfinite(query[:,1:])&donor_has,axis=1)
        self.query_support={'Jumlah query skenario terakhir':len(ids),'Kurang dari 2 kovariat asli':int((query_known<2).sum()),
            'Dengan dukungan':int(supported.sum()),'Denominator kovariat':len(self.features)}
        predictions={};times={}
        with threadpool_limits(limits=1):
            for name,model in self.models.items():
                values=np.full(len(ids),np.nan);elapsed=np.zeros(len(ids));reason=np.full(len(ids),'kovariat tidak cukup',dtype=object)
                if model is None:reason[:]='fit gagal'
                else:
                    eligible=np.flatnonzero(supported)
                    for begin in range(0,len(eligible),128):
                        chunk=eligible[begin:begin+128];tick=perf_counter()
                        try:
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always');z=model.transform(qscaled[chunk])[:,0]
                            values[chunk]=z*self.scaler.scale_[0]+self.scaler.mean_[0]
                            reason[chunk]=np.where(np.isfinite(values[chunk]),'berhasil finite','hasil nonfinite')
                            self.warn += [{'Metode':name,'Tahap':'transform','Kategori':type(w.message).__name__,'Pesan':str(w.message)} for w in caught]
                        except Exception as exc:
                            reason[chunk]='transform gagal';self.failures.append({'Metode':name,'Tahap':'transform','Pesan':repr(exc)})
                        elapsed[chunk]=(perf_counter()-tick)/len(chunk)
                predictions[name]=(values,reason)
                times[name]=float(elapsed.sum())
        return predictions,times

    def evaluate(self):
        self.metrics=[];self.predictions=[];self.intersections=[];self.cached_elapsed={m:0. for m in self.models}
        truth_by_id={r['Mask ID']:r['SO2 asli untuk skor saja'] for r in self.truth_ledger}
        for scenario in self.scenarios:
            sid=scenario['id'];ids=scenario['ids'];masked=self.y.copy();masked[self.a+ids]=np.nan
            assert np.isnan(masked[self.a+ids]).all()
            # Nilai yang disembunyikan tidak dipulihkan ke state forward fill.
            last=-1;last_value=np.nan;last_idx=np.full(self.b-self.a,-1,dtype=int);last_vals=np.full(len(last_idx),np.nan)
            tick=perf_counter()
            for i,v in enumerate(masked):
                if np.isfinite(v):last=i;last_value=v
                if i>=self.a:last_idx[i-self.a]=last;last_vals[i-self.a]=last_value
            p1_state_time=perf_counter()-tick
            results={};times={};reasons={}
            tick=perf_counter();results['P0']=np.full(len(ids),self.median);times['P0']=perf_counter()-tick;reasons['P0']=np.full(len(ids),'berhasil finite',object)
            for limit in self.limits:
                name=f'P1_L{limit}';tick=perf_counter();age=self.a+ids-last_idx[ids]
                good=(last_idx[ids]>=0)&(age<=limit)
                results[name]=np.where(good,last_vals[ids],np.nan)
                reasons[name]=np.where(good,'berhasil finite','umur donor melebihi batas atau belum ada donor')
                times[name]=perf_counter()-tick+p1_state_time
            multivariate,actual_times=self.transform_multivariate(ids)
            for name,(values,why) in multivariate.items():
                results[name]=values;times[name]=actual_times[name];reasons[name]=why
                self.cached_elapsed[name]+=actual_times[name]
            common=np.logical_and.reduce([np.isfinite(v) for v in results.values()])
            self.intersections.append({'Skenario':sid,'Seed':scenario['seed'],'Pola':scenario['pattern'],
                'Panjang slot':scenario['length'],'Jumlah mask':len(ids),'Irisan semua metode':int(common.sum()),
                'Irisan persen':100*common.sum()/len(ids)})
            truth=np.array([truth_by_id[f'{sid}_{i}'] for i in ids])
            for name,pred in results.items():
                ok=np.isfinite(pred)
                for domain,keep in [('seluruh berhasil',ok),('irisan semua metode',common)]:
                    err=pred[keep]-truth[keep];n=len(err)
                    self.metrics.append({'Metode':name,'Seed':scenario['seed'],'Pola':scenario['pattern'],
                        'Panjang slot':scenario['length'],'Panjang jam':scenario['length']/2,'Domain':domain,
                        'Jumlah mask':len(ids),'Jumlah dinilai':n,'Cakupan persen':100*n/len(ids),
                        'MAE':float(np.mean(np.abs(err))) if n else None,'RMSE':float(np.sqrt(np.mean(err**2))) if n else None,
                        'Bias':float(np.mean(err)) if n else None,'Fit detik':self.fit_times[name],
                        'Scaler fit detik bersama':self.scaler_time if name.startswith(('P2','P3')) else 0.,
                        'Transform detik':times[name],
                        'Prediksi negatif':int((pred[ok]<0).sum()),
                        'Di luar rentang SO2 train':int(((pred[ok]<self.train_min)|(pred[ok]>self.train_max)).sum())})
                for j,i in enumerate(ids):
                    self.predictions.append({'Mask ID':f'{sid}_{i}','Skenario':sid,'Metode':name,
                        'Prediksi':pred[j],'Berhasil finite':bool(ok[j]),'Irisan semua metode':bool(common[j]),
                        'Status':reasons[name][j],'Prediksi negatif':bool(ok[j] and pred[j]<0),
                        'Di luar rentang train':bool(ok[j] and (pred[j]<self.train_min or pred[j]>self.train_max))})
        pd.DataFrame(self.predictions).to_csv(RUN/'prediksi_penilaian.csv',index=False)
        metrics=pd.DataFrame(self.metrics)
        group=['Metode','Pola','Panjang slot','Domain']
        self.summary=[]
        for keys,frame in metrics.groupby(group,sort=False):
            row=dict(zip(group,keys));row['Jumlah seed']=len(frame)
            for col in ['MAE','RMSE','Bias','Cakupan persen','Jumlah dinilai','Transform detik']:
                vals=frame[col].astype(float)
                row[col+' rata-rata']=vals.mean();row[col+' SD antar-seed']=vals.std(ddof=1)
                if col in ['MAE','Cakupan persen']:row[col+' minimum']=vals.min();row[col+' maksimum']=vals.max()
            self.summary.append(row)
        self._plots()
        self.tables={'Ringkasan':self.summary,'Hasil_per_seed':self.metrics,'Irisan_mask':self.intersections,
            'Pembagian_waktu':self.split,'Distribusi_gap_train':self.train_gaps,'Fitur_P2_P3':self.feature_info,
            'Scaler_train':self.scaling,'Parameter_train':self.params,'Kejadian_masking':self.events,
            'Lokasi_masking':self.mask_ledger,'Keputusan_kualitas':self.quality,
            'Offgrid_asli':self.offgrid.to_dict('records'),
            'Waktu_fit':[{'Metode':m,'Fit detik':v,'Scaler fit detik bersama':self.scaler_time if m.startswith(('P2','P3')) else 0.} for m,v in self.fit_times.items()],
            'Waktu_transform_total':[{'Metode':m,'Transform semua skenario detik':v,'Jumlah query diproses':len(self.mask_ledger),'Timestamp query unik':len(self.union_ids)} for m,v in self.cached_elapsed.items()],
            'Peringatan':self.warn or [{'Pesan':'Tidak ada peringatan pustaka.'}],
            'Kegagalan':self.failures or [{'Pesan':'Tidak ada exception fit/transform; penolakan kovariat/batas donor tercatat per prediksi.'}],
            'Definisi':self.definitions()}
        self.integrity=[]
        for p in self.references:
            key=str(p.relative_to(ROOT));after=hashfile(p);assert self.before[key]==after
            self.integrity.append({'File':key,'SHA256 sebelum':self.before[key],'SHA256 sesudah':after,'Identik':True})
        self.tables['Integritas_SHA256']=self.integrity
        dump(OUT/'pilot_way_urang_tables.json',self.tables)
        dump(RUN/'run_checks.json',{'test_evaluated':False,'fit_max_timestamp':str(self.times[self.a-1]),
            'query_min_timestamp':str(self.val_times[self.union_ids.min()]),'query_max_timestamp':str(self.val_times[self.union_ids.max()]),
            'mask_count':len(self.mask_ledger),'query_support':self.query_support,'source_identical':True,
            'query_so2_always_nan':True,'training_donors_only':True,'common_covariates':self.features})
        from write_pilot_report import write_report
        write_report(self)
        return self

    def definitions(self):
        return [{'Istilah':k,'Definisi':v} for k,v in [
            ('MAE/RMSE/Bias','e=prediksi−SO2 asli disembunyikan; MAE=mean(abs(e)); RMSE=sqrt(mean(e²)); Bias=mean(e). Satuan µg/m³.'),
            ('Cakupan persen','100 × jumlah prediksi finite pada domain / jumlah SO2 asli sengaja disembunyikan dalam skenario. Missing alami tidak menjadi denominator.'),
            ('Irisan','Sel mask dengan prediksi finite pada SEMUA konfigurasi P0, P1 semua L, P2 semua k dan P3. N=0: skor kosong, bukan nol.'),
            ('Domain seluruh berhasil','Metrik pada seluruh sel yang berhasil diisi masing-masing konfigurasi; cakupan dapat berbeda.'),
            ('SD antar-seed','Simpangan baku sampel (ddof=1) dari tiga seed; bukan confidence interval independen karena mask dapat overlap.'),
            ('Fit detik','Fit satu kali pada train, digunakan ulang untuk semua mask. Nilai ini bukan biaya tambahan setiap skenario.'),
            ('Scaler fit detik bersama','Fit scaler train diukur terpisah dan dipakai bersama P2/P3. Tidak dijumlah berulang antar skenario.'),
            ('Transform detik','Transform dijalankan dan diukur terpisah tiap skenario. P2/P3 dijumlah dari batch query aktual; scaling query merupakan tahap bersama di luar timer model. P1 termasuk pembangunan state kausal seluruh train+validasi per skenario.'),
            ('Dukungan kovariat','Minimal 2 kovariat lag asli teramati per query untuk P2/P3. KNN menggunakan jarak nan_euclidean pada fitur yang overlap; donor train wajib SO2 asli finite. Tidak ada donor validasi.'),
            ('Kovariat kosong','NaN dipertahankan saat scaling. KNN jarak memakai overlap; P3 mengestimasi kovariat sementara memakai model train pada baris itu. Tidak backfill/rolling terpusat.'),
            ('Rentang train','Nilai di luar min/max train hanya kandidat pemeriksaan, bukan batas sensor; tidak di-clipping. Nilai negatif dilaporkan terpisah.'),
            ('Truth','Hanya ledger penilaian terpisah; seluruh SO2 query P2/P3 selalu NaN. P1 tidak mereset donor dengan nilai mask atau hasil imputasi.'),
            ('Split','70/15/15 memakai jumlah slot waktu reguler: floor(.70*N), floor(.85*N) kumulatif; bukan pemilahan acak jumlah pengamatan tersedia.'),
            ('Pengujian akhir','Hanya inventaris; tidak transform atau evaluasi. Sudah diaudit pada tahap sebelumnya, bukan sama sekali belum pernah dilihat.'),
            ('Kualitas','Nol, ekstrem, konstan dan kode ambigu dipertahankan. Pilot sementara, belum pemilihan stasiun atau penilaian forecasting.')]]

    def _plots(self):
        df=pd.DataFrame(self.summary);sel=df[(df['Domain']=='seluruh berhasil')&(df['Pola']=='blok')]
        fig,axes=plt.subplots(1,2,figsize=(15,5.5),layout='constrained')
        for method,g in sel.groupby('Metode',sort=False):
            g=g.sort_values('Panjang slot');x=g['Panjang slot'].to_numpy()/2
            axes[0].errorbar(x,g['MAE rata-rata'],yerr=g['MAE SD antar-seed'].fillna(0),marker='.',label=method,linewidth=1)
            axes[1].plot(x,g['Cakupan persen rata-rata'],marker='.',label=method,linewidth=1)
        axes[0].set(ylabel='MAE (µg/m³), mean ± SD 3 seed',title='Error pada seluruh titik yang berhasil')
        axes[1].set(ylabel='Cakupan (%)',ylim=(-2,102),title='Cakupan terhadap seluruh sel mask')
        for ax in axes:ax.set_xscale('log');ax.set_xlabel('Panjang blok (jam, skala log)');ax.grid(alpha=.2)
        axes[1].legend(fontsize=8,ncol=2)
        fig.suptitle('Pilot Way Urang — validasi saja; domain keberhasilan berbeda')
        fig.savefig(FIG/'error_dan_cakupan_blok.png',dpi=150);plt.close(fig)
        common=df[(df['Domain']=='irisan semua metode')&(df['Pola']=='blok')]
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        for method,g in common.groupby('Metode',sort=False):
            g=g.sort_values('Panjang slot');ax.plot(g['Panjang slot']/2,g['MAE rata-rata'],marker='.',label=method)
        ax.set(xscale='log',xlabel='Panjang blok (jam)',ylabel='MAE (µg/m³)',title='Irisan keberhasilan SEMUA konfigurasi — cek jumlah sel pada tabel')
        ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
        fig.savefig(FIG/'error_irisan_semua_metode.png',dpi=150);plt.close(fig)

    def export_excel(self):
        node=Path(r'C:\Users\INTAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
        result=subprocess.run([str(node),str(ROOT/'scripts/export_quality.mjs'),'pilot_way_urang_tables.json',
            'pilot_imputasi_way_urang.xlsx','.audit_preview_pilot'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        print(result.stdout)
        if result.returncode:raise RuntimeError(result.stderr)
