"""Forecasting terkontrol t+1 dengan Ridge tunggal per pipeline, tanpa test."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from importlib.metadata import version
import json,os,subprocess
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from audit_quality import hashfile
from pilot_way_urang import dump
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'forecasting_way_urang';FIG=OUT/'figures/forecasting_way_urang'
METHODS=['P0_median','P1_ffill_L4','P2_knn_k10']

class Forecast:
    def prepare(self):
        RUN.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
        self.source=next((ROOT/'data_raw').glob('*.xlsx'))
        refs=[self.source,OUT/'pilot_way_urang_tables.json',OUT/'ringkasan_pilot_imputasi_way_urang.md',
            OUT/'pilot_way_urang/config.json',OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',
            ROOT/'notebooks/04_pilot_imputasi_way_urang.ipynb',OUT/'rancangan_eksperimen_preprocessing.md']
        self.refs=refs;self.before={str(p.relative_to(ROOT)):hashfile(p) for p in refs}
        # Seluruh tabel hasil/konfigurasi/penjelasan pilot dibaca; nilai test tidak dimuat.
        self.pilot=json.loads((OUT/'pilot_way_urang_tables.json').read_text(encoding='utf-8'))
        self.pilot_report=(OUT/'ringkasan_pilot_imputasi_way_urang.md').read_text(encoding='utf-8')
        prior=json.loads((OUT/'pilot_way_urang/config.json').read_text(encoding='utf-8'))
        self.split=self.pilot['Pembagian_waktu'];self.a=self.split[0]['Jumlah slot'];self.b=self.a+self.split[1]['Jumlah slot']
        self.data=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=self.b,parse_dates=['Timestamp'])
        assert len(self.data)==self.b and self.data.Timestamp.max()<pd.Timestamp(self.split[2]['Awal'])
        self.times=pd.DatetimeIndex(self.data.Timestamp);self.y=self.data['SO2 numerik'].to_numpy(dtype=float)
        self.cov=prior['common_covariates'];self.lags=[0,1,2,6,12,24,48]
        self.warmup=90*48;self.batch=30*48;self.first_origin=self.warmup+max(self.lags)
        self.matrix=np.column_stack([self.y]+[self.data[f'{c} numerik'].shift(1).to_numpy(dtype=float) for c in self.cov])
        self.origstop=self.b-1 # origin terakhir b-2, target terakhir b-1 (validasi).
        self.pilot_seed=[r for r in self.pilot['Hasil_per_seed'] if r['Metode'] in ['P0','P2_k10']]
        self.pilot_summary=[r for r in self.pilot['Ringkasan'] if r['Metode'] in ['P0','P2_k10']]
        self.monthly=[];self.distribution=[]
        for label,lo,hi in [('train',0,self.a),('validation',self.a,self.b)]:
            series=pd.Series(self.y[lo:hi],index=self.times[lo:hi])
            groups=[('seluruh split',series)]+[(str(m),g) for m,g in series.groupby(series.index.to_period('M'))]
            for month,g in groups:
                obs=g.dropna();q=obs.quantile([.01,.25,.5,.75,.99])
                row={'Split':label,'Bulan':month,'Awal':g.index.min(),'Akhir':g.index.max(),'Slot grid':len(g),
                    'Pengamatan asli':len(obs),'Minimum':obs.min(),'P1':q.loc[.01],'P25':q.loc[.25],'Median':q.loc[.5],
                    'P75':q.loc[.75],'P99':q.loc[.99],'Maksimum':obs.max(),'Mean':obs.mean(),'SD':obs.std(ddof=1),
                    'IQR':q.loc[.75]-q.loc[.25]}
                (self.distribution if month=='seluruh split' else self.monthly).append(row)
        self.config={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'Dikunci sebelum forecasting; tanpa tuning hasil forecasting',
            'station':'Way Urang pilot sementara','horizon_slots':1,'horizon_minutes':30,'so2_lags':self.lags,
            'covariates':self.cov,'covariate_lag_slots':1,'pipelines':METHODS,'ffill_limit_slots':4,
            'ffill_reason':'Gap pendek train 1 dan 4 slot; pilot L4 mencakup blok 4 slot seluruhnya, L1 hanya 25%. Bukan klaim batas sensor.',
            'knn':{'k':10,'weights':'distance','metric':'nan_euclidean','minimum_original_covariates':2},
            'ridge':{'alpha':1.0,'fit_intercept':True,'solver':'svd'},'ridge_scaler':'StandardScaler train common timestamps per pipeline',
            'warmup_slots':self.warmup,'prefix_update_slots':self.batch,'train_input_history':'expanding prefix, fit cutoff eksklusif awal blok',
            'validation_imputer':'frozen seluruh train; histori masa lalu tidak dihitung ulang',
            'covariate_missing_policy':'P0 median prefix, P1 ffill L4 tanpa fallback, P2 KNN joint matrix [SO2(u), cov(u-1)] dengan guard dan tanpa fallback tersembunyi',
            'train_origins':'origin>=warmup+48; target<validation_start; seluruh fitur finite untuk SEMUA pipeline; identik antar Ridge',
            'validation_targets':'semua target asli finite di validasi, origin=target-1; irisan empat metode serta domain masing-masing',
            'missing_indicators':'18 original-missing flags untuk SO2 lags dan kovariat; dimasukkan pada seluruh Ridge',
            'persistence':'last ORIGINAL available SO2 at origin or before, tanpa batas umur, tanpa nilai imputasi',
            'availability':'SO2(u) diasumsikan tersedia pada origin u; kovariat(u-1) diasumsikan tersedia pada u. Latensi aktual belum diketahui.',
            'quality':'nol, lonjakan, konstan dan ekstrem dipertahankan; ledger off-grid tahap 4 tidak dipetakan',
            'P3':'ditunda karena belum konvergen dan hasil negatif pada pilot','test_loaded':False,'test_evaluated':False,
            'threads':1,'seed':42,'versions':{p:version(p) for p in ['pandas','numpy','scikit-learn','scipy']},
            'split':self.split,'initial_input_hashes':self.before}
        dump(RUN/'config_locked.json',self.config)
        self.config_hash=hashfile(RUN/'config_locked.json')
        self.locked_protocol=[{'Keputusan':k,'Nilai':('UTC: '+str(v)) if k=='created_utc' else str(v)} for k,v in self.config.items() if k not in ['initial_input_hashes','versions','split']]
        print('Konfigurasi dikunci:',self.config_hash,'; belum ada Ridge fit.',flush=True)
        return self

    def impute_history(self):
        tick=perf_counter();m=self.matrix[:self.origstop].copy()
        self.filled={name:m.copy() for name in METHODS};self.prefix_logs=[]
        # Forward fill memakai pengamatan asli saja; hasil imputasi tidak mereset umur.
        last_idx=np.full(m.shape[1],-1,dtype=int);last_val=np.full(m.shape[1],np.nan)
        for u,row in enumerate(m):
            seen=np.isfinite(row);last_idx[seen]=u;last_val[seen]=row[seen]
            if u>=self.warmup:
                use=(~seen)&(last_idx>=0)&(u-last_idx<=4)
                self.filled['P1_ffill_L4'][u,use]=last_val[use]
        with threadpool_limits(limits=1):
            starts=list(range(self.warmup,self.a,self.batch))+[self.a]
            for start in starts:
                stop=min(start+self.batch,self.a) if start<self.a else self.origstop
                if start>=stop:continue
                prefix=self.matrix[:start];query=m[start:stop].copy()
                assert start<=self.a and self.times[start-1]<self.times[start]
                fit_tick=perf_counter();med=np.nanmedian(prefix,axis=0)
                self.filled['P0_median'][start:stop]=np.where(np.isfinite(query),query,med)
                scaler=StandardScaler().fit(prefix)
                active=np.isfinite(scaler.mean_);active[0]=True
                scaled=scaler.transform(prefix)[:,active]
                donor_rows=np.flatnonzero(np.isfinite(prefix[:,0]))
                donors=scaled[donor_rows]
                knn=KNNImputer(n_neighbors=10,weights='distance',metric='nan_euclidean').fit(donors)
                fit_secs=perf_counter()-fit_tick
                transform_tick=perf_counter();out=query.copy()
                eligible=np.flatnonzero((np.isfinite(query[:,1:]).sum(axis=1)>=2)&(~np.isfinite(query)).any(axis=1))
                for begin in range(0,len(eligible),128):
                    ids=eligible[begin:begin+128];q=query[ids];z=scaler.transform(q)[:,active]
                    estimated=knn.transform(z)
                    # Guard per kolom: ada donor dengan nilai kolom itu dan overlap fitur asli.
                    original_known=np.isfinite(z)
                    for j,col in enumerate(np.flatnonzero(active)):
                        bank=donors[np.isfinite(donors[:,j])]
                        support=np.any(original_known & np.isfinite(bank).any(axis=0),axis=1) if len(bank) else np.zeros(len(ids),bool)
                        use=(~np.isfinite(q[:,col]))&support
                        out[ids[use],col]=estimated[use,j]*scaler.scale_[col]+scaler.mean_[col]
                self.filled['P2_knn_k10'][start:stop]=out
                self.prefix_logs.append({'Awal query':self.times[start],'Akhir query':self.times[stop-1],
                    'Fit sampai eksklusif':self.times[start],'Timestamp donor maksimum':self.times[donor_rows[-1]],
                    'Jumlah baris prefix':start,'Jumlah donor SO2 asli':len(donor_rows),
                    'Jumlah query':stop-start,'Query KNN ada sel kosong dan dukungan':len(eligible),
                    'Fit imputer dan scaler detik':fit_secs,'Transform KNN detik':perf_counter()-transform_tick,
                    'Periode':'train prefix' if start<self.a else 'validation frozen train',
                    'Median SO2 prefix':med[0]})
                print('Histori kausal:',self.times[start],'-',self.times[stop-1],flush=True)
        self.impute_seconds=perf_counter()-tick
        self.feature_names=[f'SO2_lag{lag}' for lag in self.lags]+[f'{c}_lag1' for c in self.cov]
        self.feature_names += [f'missing_asli_{c}' for c in self.feature_names]
        self.X={};self.input_flags=None
        raw=pd.DataFrame(m)
        for name,filled in self.filled.items():
            # Imputasi historis selesai dahulu, baru membentuk lag; tidak ada lead.
            f=pd.DataFrame(filled)
            values=np.column_stack([f[0].shift(lag).to_numpy() for lag in self.lags]+[f[c].to_numpy() for c in range(1,f.shape[1])])
            original=np.column_stack([raw[0].shift(lag).to_numpy() for lag in self.lags]+[raw[c].to_numpy() for c in range(1,raw.shape[1])])
            flags=~np.isfinite(original)
            self.X[name]=np.column_stack([values,flags.astype(float)])
            self.input_flags=flags
            hist=pd.DataFrame(filled,columns=['SO2']+[f'{c}_lag1' for c in self.cov],index=self.times[:self.origstop])
            for j,col in enumerate(hist.columns):
                hist[f'{col}_missing_asli']=~np.isfinite(m[:,j]);hist[f'{col}_diimputasi']=~np.isfinite(m[:,j])&np.isfinite(filled[:,j])
            hist.index.name='Origin';hist.to_csv(RUN/f'histori_kausal_{name}.csv')
        self.input_counts=[]
        for name,X in self.X.items():
            self.input_counts.append({'Pipeline':name,'Jumlah fitur':X.shape[1],
                'Slot SO2 hilang asli setelah warmup':int((~np.isfinite(m[self.warmup:,0])).sum()),
                'Slot SO2 terisi':int((~np.isfinite(m[self.warmup:,0]) & np.isfinite(self.filled[name][self.warmup:,0])).sum()),
                'Origin pasca warmup dengan seluruh fitur finite':int(np.isfinite(X[self.first_origin:]).all(axis=1).sum())})
        return self

    def fit_predict(self):
        assert hashfile(RUN/'config_locked.json')==self.config_hash
        train_orig=np.arange(self.first_origin,self.a-1) # terakhir a-2; a-1 dipurge.
        observed=np.isfinite(self.y[train_orig+1])
        shared=observed.copy()
        for X in self.X.values():shared &= np.isfinite(X[train_orig]).all(axis=1)
        self.train_origins=train_orig[shared]
        assert (self.train_origins+1<self.a).all()
        assert len(self.train_origins)>len(self.feature_names)
        train_targets=pd.Series(self.y[self.train_origins+1],index=self.times[self.train_origins+1])
        self.actual_train_distribution=[]
        for label,part in [('seluruh train bersama',train_targets)]+[(str(m),v) for m,v in train_targets.groupby(train_targets.index.to_period('M'))]:
            self.actual_train_distribution.append({'Bagian':label,'Awal target':part.index.min(),'Akhir target':part.index.max(),
                'Jumlah target asli':len(part),'Minimum':part.min(),'Median':part.median(),'Maksimum':part.max(),
                'Mean':part.mean(),'SD':part.std(ddof=1)})
        target_ids=np.arange(self.a,self.b);target_ids=target_ids[np.isfinite(self.y[target_ids])]
        origins=target_ids-1;truth=self.y[target_ids]
        self.pred=pd.DataFrame({'Origin':self.times[origins],'Timestamp target':self.times[target_ids],
            'Index origin':origins,'Index target':target_ids,'SO2 target asli':truth,
            'SO2 historis missing count':self.input_flags[origins,:len(self.lags)].sum(axis=1),
            'Kovariat missing count':self.input_flags[origins,len(self.lags):].sum(axis=1)})
        self.models={};self.model_scalers={};self.coefficients=[];self.timing=[];self.train_detail=[]
        self.training_features={}
        with threadpool_limits(limits=1):
            for name,X in self.X.items():
                tick=perf_counter();scaler=StandardScaler().fit(X[self.train_origins]);Z=scaler.transform(X[self.train_origins])
                scaler_secs=perf_counter()-tick
                tick=perf_counter();model=Ridge(**self.config['ridge']).fit(Z,self.y[self.train_origins+1]);fit_secs=perf_counter()-tick
                self.models[name]=model;self.model_scalers[name]=scaler
                eligible=np.isfinite(X[origins]).all(axis=1);p=np.full(len(origins),np.nan)
                tick=perf_counter();p[eligible]=model.predict(scaler.transform(X[origins[eligible]]));predict_secs=perf_counter()-tick
                self.pred[name]=p
                self.timing.append({'Pipeline':name,'Scaler fitur fit detik':scaler_secs,'Ridge fit detik':fit_secs,'Predict detik':predict_secs})
                for i,col in enumerate(self.feature_names):
                    self.coefficients.append({'Pipeline':name,'Fitur':col,'Koefisien fitur standar':model.coef_[i],
                        'Mean fitur train':scaler.mean_[i],'Skala fitur train':scaler.scale_[i],'Intercept':model.intercept_})
                self.train_detail.append({'Pipeline':name,'Origin train awal':self.times[self.train_origins[0]],
                    'Origin train akhir':self.times[self.train_origins[-1]],'Jumlah train kandidat target asli':int(observed.sum()),
                    'Jumlah train fitur lengkap pipeline':int((observed & np.isfinite(X[train_orig]).all(axis=1)).sum()),
                    'Jumlah train bersama digunakan':len(self.train_origins),
                    'Origin dipurge':self.times[self.a-1],'Target origin dipurge':self.times[self.a],
                    'Target train maksimum':self.times[self.train_origins[-1]+1]})
                pd.DataFrame(X,columns=self.feature_names,index=self.times[:self.origstop]).to_csv(RUN/f'fitur_{name}.csv',index_label='Origin')
        # Persistence menggunakan SO2 asli, bukan hasil salah satu pipeline.
        last=np.full(self.origstop,np.nan);last_time=np.full(self.origstop,-1,int);value=np.nan;idx=-1
        for i,v in enumerate(self.y[:self.origstop]):
            if np.isfinite(v):value=v;idx=i
            last[i]=value;last_time[i]=idx
        self.pred['Persistence']=last[origins]
        self.pred['Persistence umur donor slot']=origins-last_time[origins]
        names=METHODS+['Persistence'];self.common=np.isfinite(self.pred[names]).all(axis=1).to_numpy()
        self.pred['Irisan semua metode']=self.common
        self.metrics=[];self.month_metrics=[];self.strata=[]
        def score(name,keep,domain,total):
            e=self.pred.loc[keep,name].to_numpy()-self.pred.loc[keep,'SO2 target asli'].to_numpy();n=len(e)
            return {'Metode':name,'Domain':domain,'Jumlah prediksi':n,'Denominator target asli validasi':total,
                'Cakupan persen':100*n/total if total else None,'MAE':np.mean(np.abs(e)) if n else None,
                'RMSE':np.sqrt(np.mean(e**2)) if n else None,'Bias':np.mean(e) if n else None,
                'Prediksi negatif':int((self.pred.loc[keep,name]<0).sum())}
        month=self.pred['Timestamp target'].dt.to_period('M').astype(str)
        for name in names:
            ok=np.isfinite(self.pred[name]).to_numpy()
            self.metrics += [score(name,ok,'seluruh berhasil',len(self.pred)),score(name,self.common,'irisan semua metode',len(self.pred))]
            for mo in month.unique():
                inmonth=month.eq(mo).to_numpy()
                for domain,keep in [('seluruh berhasil',ok&inmonth),('irisan semua metode',self.common&inmonth)]:
                    self.month_metrics.append({'Bulan target':mo,**score(name,keep,domain,int(inmonth.sum()))})
            for label,mask in [('semua input asli lengkap',(self.input_flags[origins].sum(axis=1)==0)),
                ('ada SO2 historis missing',self.pred['SO2 historis missing count'].to_numpy()>0),
                ('ada kovariat missing',self.pred['Kovariat missing count'].to_numpy()>0)]:
                self.strata.append({'Strata':label,**score(name,self.common&mask,'irisan semua metode',int(mask.sum()))})
        self.pred.to_csv(RUN/'prediksi_validasi.csv',index=False)
        self.pred[['Origin','Timestamp target','Index origin','Index target','Irisan semua metode']].to_csv(RUN/'timestamp_evaluasi.csv',index=False)
        pd.DataFrame({'Index origin':self.train_origins,'Origin':self.times[self.train_origins],
            'Target train':self.times[self.train_origins+1]}).to_csv(RUN/'timestamp_train_bersama.csv',index=False)
        self._plots()
        self.integrity=[]
        for p in self.refs:
            key=str(p.relative_to(ROOT));after=hashfile(p);assert after==self.before[key]
            self.integrity.append({'File':key,'SHA256 sebelum':self.before[key],'SHA256 sesudah':after,'Identik':True})
        assert hashfile(RUN/'config_locked.json')==self.config_hash
        self.tables={'Ringkasan':self.metrics,'Performa_bulanan':self.month_metrics,'Strata_missing':self.strata,
            'Distribusi_train_Ridge':self.actual_train_distribution,
            'SO2_bulanan_train_validasi':self.monthly,'SO2_distribusi_split':self.distribution,
            'Pilot_P0_KNN_per_seed':self.pilot_seed,'Pilot_P0_KNN_variasi':self.pilot_summary,
            'Pilot_waktu_fit':[r for r in self.pilot['Waktu_fit'] if r['Metode'] in ['P0','P2_k10']],
            'Rancangan_dikunci':self.locked_protocol,'Train_bersama_purge':self.train_detail,
            'Prefix_imputer_kausal':self.prefix_logs,'Kelengkapan_input':self.input_counts,
            'Waktu_Ridge':self.timing,'Koefisien_Ridge':self.coefficients,
            'Prediksi_validasi':self.pred.to_dict('records'),'Definisi':self.definitions(),
            'Integritas_SHA256':self.integrity,'Pembagian_waktu':self.split}
        dump(OUT/'forecasting_way_urang_tables.json',self.tables)
        dump(RUN/'checks.json',{'test_loaded':False,'test_evaluated':False,'config_sha256':self.config_hash,
            'n_train_common':len(self.train_origins),'n_valid_observed_targets':len(self.pred),'n_common_evaluation':int(self.common.sum()),
            'max_loaded_time':str(self.times[-1]),'max_target_time':str(self.pred['Timestamp target'].max()),
            'purged_train_origin':str(self.times[self.a-1]),'source_unchanged':True})
        from write_forecast_report import write_report
        write_report(self)
        return self

    def definitions(self):
        return [{'Istilah':k,'Definisi':v} for k,v in [
            ('Horizon','Satu langkah t+30 menit; penyaringan awal, bukan horizon final penelitian.'),
            ('MAE/RMSE/Bias','e=prediksi−target SO2 asli; MAE=mean(abs(e)); RMSE=sqrt(mean(e²)); Bias=mean(e). Satuan µg/m³.'),
            ('Cakupan persen','Jumlah prediksi finite / seluruh target validasi SO2 asli teramati ×100%. Bulanan memakai denominator target asli pada bulan itu.'),
            ('Irisan semua metode','Timestamp target dengan prediksi finite P0, P1 L4, P2 k10 dan persistence. Tidak mengisi target kosong.'),
            ('Kausalitas histori','Setelah 90 hari kalibrasi, imputer train fit prefix sebelum blok 30 hari; pada validasi fit seluruh train dan dibekukan. Histori lama tidak diperbaiki ulang.'),
            ('Matriks rekonstruksi','Baris u=[SO2(u),11 kovariat(u−1)]. Kovariat telah dilag sebelum rekonstruksi; SO2 historis dilag sesudah rekonstruksi kausal.'),
            ('Fitur','7 lag SO2 +11 kovariat lag1 +18 indikator missing asli. Tidak ada waktu mentah, arah angin linear, lead atau target masa depan.'),
            ('P1','Forward fill maksimum 4 slot dari nilai asli sebelumnya, untuk input numerik; tidak ada median fallback.'),
            ('P2','Donor original SO2 finite dari prefix train saja; KNN joint matrix k10 distance, scaler prefix, guard minimum 2 kovariat asli dan dukungan donor per kolom. Known SO2(u) dapat membantu mengisi kovariat(u−1) saat tersedia pada u, tidak memakai target(u+1).'),
            ('Persistence','SO2 asli terakhir pada/sebelum origin, tanpa batas umur; bukan SO2 hasil imputasi.'),
            ('Train bersama','Tiga Ridge memakai origin train/target asli yang sama, fitur lengkap semua pipeline. Scaler fitur fit hanya baris train bersama masing-masing pipeline.'),
            ('Batas split','Origin a−1 yang targetnya awal validasi dikeluarkan dari train dan menjadi origin prediksi validasi pertama.'),
            ('Kalibrasi','90 hari pertama tidak menjadi sampel Ridge; tambahan 48 slot setelahnya untuk lag maksimum. Batas split tetap, sampel berkurang karena prosedur kausal.'),
            ('Test','Tidak dimuat nilainya atau dievaluasi. Batas/jumlah inventaris hanya dibaca dari hasil pilot, yang telah diaudit sebelumnya.'),
            ('Kualitas','Nilai asli, nol, ekstrem dan konstan dipertahankan. Tidak ada clipping prediksi. Ketersediaan SO2 saat origin dan kovariat lag1 masih asumsi latensi yang perlu konfirmasi.'),
            ('Strata missing','Strata dapat overlap. Skor pada irisan semua metode dalam strata; denominator cakupan adalah target asli strata itu.'),
            ('Stabilitas','Distribusi deskriptif train/validasi; rentang/SD lebih kecil bukan bukti stasioneritas atau jaminan hasil pada konsentrasi tinggi.')]]

    def _plots(self):
        monthly=pd.DataFrame(self.month_metrics);g=monthly[monthly.Domain=='irisan semua metode']
        fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
        for name,part in g.groupby('Metode',sort=False):
            axes[0].plot(part['Bulan target'],part.MAE,marker='.',label=name)
            axes[1].plot(part['Bulan target'],part.RMSE,marker='.',label=name)
        for name,part in monthly[monthly.Domain=='seluruh berhasil'].groupby('Metode',sort=False):
            axes[2].plot(part['Bulan target'],part['Cakupan persen'],marker='.',label=name)
        for ax,title in zip(axes,['MAE pada timestamp bersama','RMSE pada timestamp bersama','Cakupan terhadap target asli']):
            ax.set_title(title);ax.tick_params(axis='x',rotation=25);ax.grid(alpha=.2)
        axes[0].set_ylabel('µg/m³');axes[1].set_ylabel('µg/m³');axes[2].set_ylabel('%');axes[2].set_ylim(0,105);axes[2].legend(fontsize=8)
        fig.suptitle('Way Urang — Ridge t+30 menit, validasi saja')
        fig.savefig(FIG/'performa_bulanan.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
        for label,lo,hi in [('train',0,self.a),('validation',self.a,self.b)]:
            vals=self.y[lo:hi];vals=vals[np.isfinite(vals)]
            axes[0].hist(vals,bins=60,density=True,histtype='step',label=label)
        axes[0].set(xlabel='SO2 asli (µg/m³)',ylabel='Densitas',title='Distribusi train dan validasi');axes[0].legend()
        monthly=pd.DataFrame(self.monthly)
        for label,g in monthly.groupby('Split',sort=False):
            x=np.arange(len(g));axes[1].errorbar(g['Bulan'],g.Median,yerr=[g.Median-g.Minimum,g.Maksimum-g.Median],fmt='.',label=label)
        axes[1].tick_params(axis='x',rotation=90,labelsize=8);axes[1].set(title='Median dan rentang bulanan',ylabel='SO2 asli (µg/m³)');axes[1].legend()
        fig.savefig(FIG/'distribusi_train_validasi.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(15,5),layout='constrained')
        # Seluruh validasi, tanpa menyambung celah pada pengamatan asli.
        truth=pd.Series(self.y[self.a:self.b],index=self.times[self.a:self.b]);ax.plot(truth.index,truth,label='SO2 asli',color='black',lw=.6)
        for name in METHODS+['Persistence']:
            s=self.pred.set_index('Timestamp target')[name].reindex(truth.index)
            ax.plot(s.index,s,label=name,lw=.6,alpha=.8)
        ax.set(title='Prediksi satu langkah pada validasi — tanpa test',ylabel='SO2 (µg/m³)');ax.legend(ncol=3,fontsize=8)
        fig.savefig(FIG/'prediksi_validasi.png',dpi=150);plt.close(fig)

    def export_excel(self):
        node=Path(r'C:\Users\INTAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
        done=subprocess.run([str(node),str(ROOT/'scripts/export_quality.mjs'),'forecasting_way_urang_tables.json',
            'forecasting_acuan_way_urang.xlsx','.audit_preview_forecast'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        print(done.stdout)
        if done.returncode:raise RuntimeError(done.stderr)
