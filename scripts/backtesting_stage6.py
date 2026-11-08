"""Expanding-window development backtest, fixed 5B settings, no final test."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from copy import deepcopy
import json,os,subprocess
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import forecast_ridge_way_urang as base
from audit_quality import hashfile
from pilot_way_urang import dump
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'backtesting_way_urang';FIG=OUT/'figures/backtesting_way_urang'
METHODS=['P0_median','P1_ffill_L4','P2_knn_k10'];ALL=METHODS+['Persistence']

def values_stats(values):
    x=np.asarray(values,dtype=float);v=x[np.isfinite(x)]
    return {'Jumlah slot atau sampel':len(x),'Target asli teramati':len(v),'Minimum SO2':float(v.min()) if len(v) else None,
        'P25 SO2':float(np.quantile(v,.25)) if len(v) else None,'Median SO2':float(np.median(v)) if len(v) else None,
        'P75 SO2':float(np.quantile(v,.75)) if len(v) else None,'Maksimum SO2':float(v.max()) if len(v) else None,
        'Mean SO2':float(v.mean()) if len(v) else None,'SD SO2':float(v.std(ddof=1)) if len(v)>1 else None,
        'Target lebih 2.14':int((v>2.14).sum())}

class Fold(base.Forecast):
    def __init__(self,manager,number,a,b):
        self.manager=manager;self.number=number;self.label=f'Fold_{number:02d}';self.folder=RUN/self.label;self.folder.mkdir(exist_ok=True)
        self.a=a;self.b=b;self.origstop=b-1;self.times=manager.times[:b];self.y=manager.y[:b].copy()
        self.cov=list(manager.cfg['covariates']);self.lags=list(manager.cfg['so2_lags'])
        self.warmup=manager.cfg['warmup_slots'];self.batch=manager.cfg['prefix_update_slots'];self.first_origin=self.warmup+max(self.lags)
        self.matrix=manager.matrix[:b].copy()
        assert self.times[-1]<manager.test_start

    def preprocess_gate(self):
        previous=base.RUN
        try:
            base.RUN=self.folder
            super().impute_history()
        finally:base.RUN=previous
        assert len(self.feature_names)==34 and all('Curah Hujan' not in c for c in self.feature_names)
        candidates=np.arange(self.first_origin,self.a-1);observed=np.isfinite(self.y[candidates+1]);shared=observed.copy()
        self.flow=[];self.lost=[];self.monthly_lost=[];self.individual={}
        for method,X in self.X.items():
            own=observed&np.isfinite(X[candidates]).all(axis=1);self.individual[method]=candidates[own];shared &= own
            self.flow.append({'Fold':self.label,'Pipeline':method,'Seluruh origin train':self.a,
                'Setelah kalibrasi buffer purge':len(candidates),'Target train asli':int(observed.sum()),
                'Setelah fitur finite mandiri':int(own.sum())})
            pd.DataFrame(X,columns=self.feature_names,index=self.times[:self.origstop]).to_csv(self.folder/f'fitur_{method}.csv',index_label='Origin')
        self.train_ids=candidates[shared]
        for method in METHODS:
            ids=np.setdiff1d(self.individual[method],self.train_ids)
            self.lost.append({'Fold':self.label,'Pipeline':method,'Layak mandiri':len(self.individual[method]),
                'Aktual bersama':len(self.train_ids),'Keluar karena irisan':len(ids),**values_stats(self.y[ids+1])})
            for month in self.times[candidates+1].to_period('M').unique():
                sub=ids[self.times[ids+1].to_period('M')==month]
                self.monthly_lost.append({'Fold':self.label,'Pipeline':method,'Bulan target':str(month),**values_stats(self.y[sub+1])})
            self.flow[METHODS.index(method)]['Final train bersama']=len(self.train_ids)
        threshold=self.manager.cfg['minimum_common_train_samples']
        self.gate={'Fold':self.label,'Ridge fit count global sebelum gate':self.manager.fit_count,
            'Jumlah fitur':len(self.feature_names),'Minimum train bersama disyaratkan':threshold,
            'Jumlah train bersama':len(self.train_ids),'Target unik train bersama':int(np.unique(self.y[self.train_ids+1]).size),
            'Cukup sampel':bool(len(self.train_ids)>=threshold and np.unique(self.y[self.train_ids+1]).size>=2),
            'Kalibrasi mulai':self.times[0],'Kalibrasi sampai':self.times[self.warmup-1],
            'Origin dipurge':self.times[self.a-1],'Target origin dipurge':self.times[self.a]}
        dump(self.folder/'gate_sebelum_Ridge.json',self.gate)
        self.train_distribution=[{'Fold':self.label,'Bulan target':'seluruh train bersama',**values_stats(self.y[self.train_ids+1])}]
        for month in self.times[self.train_ids+1].to_period('M').unique():
            sub=self.train_ids[self.times[self.train_ids+1].to_period('M')==month]
            self.train_distribution.append({'Fold':self.label,'Bulan target':str(month),**values_stats(self.y[sub+1])})
        pd.DataFrame({'Index origin':self.train_ids,'Origin':self.times[self.train_ids],
            'Timestamp target train':self.times[self.train_ids+1],'SO2 target asli':self.y[self.train_ids+1]}).to_csv(self.folder/'timestamp_train_bersama.csv',index=False)
        self.audit=[]
        # Original (pre-imputation) missing flags, reconstructed before selecting cohorts.
        for method,X in self.X.items():
            rawmiss=X[candidates,17:].astype(bool)
            post=np.isfinite(X[candidates]).all(axis=1)
            self.audit.append({'Fold':self.label,'Pipeline':method,'Kandidat target teramati':int(observed.sum()),
                'Ada SO2 historis asli kosong':int((observed&rawmiss[:,:7].any(axis=1)).sum()),
                'Ada kovariat asli kosong':int((observed&rawmiss[:,7:].any(axis=1)).sum()),
                'Fitur tetap nonfinite':int((observed&~post).sum()),
                'Target tinggi keluar karena fitur':int((observed&~post&(self.y[candidates+1]>2.14)).sum())})
        print('Gate',self.label,self.gate,flush=True)
        return self

    def fit_predict_fold(self):
        assert self.manager.all_gates_passed and self.gate['Cukup sampel']
        ids=self.train_ids;targets=np.arange(self.a,self.b);targets=targets[np.isfinite(self.y[targets])];origin=targets-1
        frame=pd.DataFrame({'Fold':self.label,'Origin':self.times[origin],'Timestamp target':self.times[targets],
            'Index origin':origin,'Index target':targets,'SO2 target asli':self.y[targets]})
        self.timing=[];self.coefficients=[]
        with threadpool_limits(limits=self.manager.cfg['threads']):
            for method in METHODS:
                X=self.X[method];tick=perf_counter();scaler=StandardScaler().fit(X[ids]);Z=scaler.transform(X[ids]);scale_secs=perf_counter()-tick
                tick=perf_counter();model=Ridge(**self.manager.cfg['ridge']).fit(Z,self.y[ids+1]);fit_secs=perf_counter()-tick
                self.manager.fit_count+=1
                good=np.isfinite(X[origin]).all(axis=1);p=np.full(len(origin),np.nan)
                tick=perf_counter();p[good]=model.predict(scaler.transform(X[origin[good]]));predict_secs=perf_counter()-tick
                frame[method]=p
                self.timing.append({'Fold':self.label,'Pipeline':method,'Jumlah train':len(ids),'Scaler fitur fit detik':scale_secs,
                    'Ridge fit detik':fit_secs,'Predict detik':predict_secs})
                for j,col in enumerate(self.feature_names):
                    self.coefficients.append({'Fold':self.label,'Pipeline':method,'Fitur':col,
                        'Koefisien fitur standar':float(model.coef_[j]),'Mean fitur train':float(scaler.mean_[j]),
                        'Skala fitur train':float(scaler.scale_[j]),'Intercept':float(model.intercept_)})
        last=np.maximum.accumulate(np.where(np.isfinite(self.y[:self.origstop]),np.arange(self.origstop),-1))
        frame['Persistence']=self.y[last[origin]];frame['Usia donor persistence slot']=origin-last[origin]
        frame['Irisan semua metode']=np.isfinite(frame[ALL]).all(axis=1)
        self.pred=frame;self.metrics=[];self.month_metrics=[];self.strata=[]
        month=frame['Timestamp target'].dt.to_period('M').astype(str)
        for method in ALL:
            valid=np.isfinite(frame[method]).to_numpy();common=frame['Irisan semua metode'].to_numpy()
            for domain,keep in [('seluruh berhasil',valid),('irisan semua metode',common)]:
                self.metrics.append(score(frame,method,keep,self.label,domain,len(frame)))
                for mo in month.unique():
                    within=month.eq(mo).to_numpy()
                    self.month_metrics.append({'Bulan target':mo,**score(frame,method,keep&within,self.label,domain,int(within.sum()))})
            for label,mask in [('SO2 lebih 2.14',frame['SO2 target asli'].to_numpy()>2.14),('SO2 tidak lebih 2.14',frame['SO2 target asli'].to_numpy()<=2.14)]:
                self.strata.append({'Strata':label,**score(frame,method,common&mask,self.label,'irisan semua metode',int(mask.sum()))})
        frame.to_csv(self.folder/'prediksi_validasi.csv',index=False)
        frame[['Fold','Origin','Timestamp target','Index origin','Index target','Irisan semua metode']].to_csv(self.folder/'timestamp_evaluasi.csv',index=False)
        dump(self.folder/'metrics.json',self.metrics)
        return self

def timestamp_hash(frame,mask):
    from hashlib import sha256
    return sha256('\n'.join(frame.loc[mask,'Timestamp target'].astype(str)).encode()).hexdigest()

def score(frame,method,mask,fold,domain,den):
    pred=frame.loc[mask,method].to_numpy();truth=frame.loc[mask,'SO2 target asli'].to_numpy();error=pred-truth;n=len(error)
    return {'Fold':fold,'Pipeline':method,'Domain':domain,'Jumlah prediksi':n,'Denominator target asli':den,
        'Cakupan persen':100*n/den if den else None,'MAE':float(np.abs(error).mean()) if n else None,
        'RMSE':float(np.sqrt(np.mean(error**2))) if n else None,'Bias':float(error.mean()) if n else None,
        'Jumlah absolute error':float(np.abs(error).sum()),'Jumlah squared error':float(np.sum(error**2)),
        'Jumlah signed error':float(error.sum()),'Prediksi negatif':int((pred<0).sum()),
        'SHA256 timestamp target':timestamp_hash(frame,mask)}

class Backtest:
    def prepare(self):
        RUN.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
        prior=json.loads((OUT/'ablasi_curah_hujan_5B/config_locked.json').read_text(encoding='utf-8'))
        self.prior=prior;self.dev_n=sum(r['Jumlah slot'] for r in prior['split'][:2]);self.test_start=pd.Timestamp(prior['split'][2]['Awal'])
        cols=['Timestamp','SO2 numerik']+[f'{c} numerik' for c in prior['covariates']]
        self.data=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',usecols=cols,nrows=self.dev_n,parse_dates=['Timestamp'])
        self.times=pd.DatetimeIndex(self.data.Timestamp);self.y=self.data['SO2 numerik'].to_numpy(dtype=float)
        self.matrix=np.column_stack([self.y]+[self.data[f'{c} numerik'].shift(1).to_numpy(dtype=float) for c in prior['covariates']])
        assert len(self.times)==self.dev_n and self.times[-1]+pd.Timedelta(minutes=30)==self.test_start
        initial_time=self.times[0]+pd.Timedelta(days=120);first=self.times.get_loc(initial_time)
        q,r=divmod(self.dev_n-first,4);self.cuts=[first]
        for i in range(4):self.cuts.append(self.cuts[-1]+q+(i<r))
        assert first==120*48 and self.cuts[-1]==self.dev_n and max(np.diff(self.cuts))-min(np.diff(self.cuts))<=1
        self.plan=[];self.distributions=[];self.distributions_month=[]
        for number,(a,b) in enumerate(zip(self.cuts[:-1],self.cuts[1:]),1):
            fold=f'Fold_{number:02d}'
            self.plan.append({'Fold':fold,'Train mulai':self.times[0],'Train akhir':self.times[a-1],
                'Validasi mulai':self.times[a],'Validasi akhir':self.times[b-1],'Validasi akhir eksklusif':self.times[b-1]+pd.Timedelta(minutes=30),
                'Slot train':a,'Slot validasi':b-a,'Durasi validasi jam':(b-a)/2,
                'Index awal validasi':a,'Index akhir validasi eksklusif':b})
            for label,lo,hi in [('train',0,a),('validation',a,b)]:
                self.distributions.append({'Fold':fold,'Bagian':label,**values_stats(self.y[lo:hi])})
                months=self.times[lo:hi].to_period('M')
                for mo in months.unique():self.distributions_month.append({'Fold':fold,'Bagian':label,'Bulan target':str(mo),**values_stats(self.y[lo:hi][months==mo])})
        self.refs=[]
        for p in OUT.rglob('*'):
            if not p.is_file():continue
            parts=p.relative_to(OUT).parts
            if any(v.startswith('.') for v in parts) or p.name.endswith('.inspect.ndjson'):continue
            if 'backtesting_way_urang' in parts or p.name.startswith(('backtesting_preprocessing','ringkasan_backtesting_preprocessing','tabel_backtesting')):continue
            self.refs.append(p)
        self.refs += list((ROOT/'data_raw').rglob('*.xlsx'))
        self.refs += [p for p in (ROOT/'notebooks').glob('*.ipynb') if p.name!='06_backtesting_preprocessing.ipynb']
        self.refs += [ROOT/'scripts/forecast_ridge_way_urang.py']
        self.refs=list(dict.fromkeys(self.refs));self.before={str(p.relative_to(ROOT)):hashfile(p) for p in self.refs}
        self.cfg=deepcopy(prior)
        # A fixed adequacy rule; no outcome-based respecification.
        self.cfg.update({'created_utc':datetime.now(timezone.utc).isoformat(),'experiment':'Tahap6 expanding-window development validation',
            'initial_train_days':120,'n_folds':4,'fold_cuts':self.cuts,'folds':self.plan,
            'minimum_common_train_samples':10*34,'minimum_distinct_train_targets':2,
            'adequacy_rule':'Minimal 10×34 fitur=340 contoh bersama dan >=2 nilai target berbeda; pemeriksaan komputasional, bukan jaminan daya statistik.',
            'calendar_rule':'Mulai validasi setelah120hari; bagi sisa slot waktu development menjadi4bagian, sisa1slot diberikan ke fold pertama.',
            'development_reuse':'Development=train+validasi lama, sudah digunakan untuk eksperimen/seleksi; bukan holdout independen.',
            'refit':'Seluruh imputer/scaler dan Ridge dibuat baru pada setiapfold, dari data trainfold saja; tidak reuse fitted parameters5B/seluruhdevelopment.',
            'adequacy_adjustment':'Tidak ada, kecuali gate pra-Ridge gagal; penyesuaian harus disimpan dan berbasis kecukupan saja.',
            'test_loaded':False,'test_evaluated':False,'initial_input_hashes':self.before})
        unchanged=['covariates','horizon_slots','horizon_minutes','so2_lags','covariate_lag_slots','ffill_limit_slots','knn','ridge',
            'warmup_slots','prefix_update_slots','availability','quality','threads','seed']
        self.invariants=[{'Parameter':k,'Sama dengan5B':self.cfg[k]==prior[k],'Nilai':str(self.cfg[k])} for k in unchanged]
        assert all(r['Sama dengan5B'] for r in self.invariants)
        dump(RUN/'fold_config_locked.json',self.cfg);self.config_hash=hashfile(RUN/'fold_config_locked.json')
        self.fit_count=0;self.all_gates_passed=False;self.folds=[]
        print('Fold kalender dikunci sebelum fitting:',self.plan,flush=True)
        return self

    def preprocess_and_check_all(self):
        for number,(a,b) in enumerate(zip(self.cuts[:-1],self.cuts[1:]),1):
            fold=Fold(self,number,a,b).preprocess_gate();self.folds.append(fold)
        self.all_gates_passed=all(f.gate['Cukup sampel'] for f in self.folds)
        dump(RUN/'all_gates_before_models.json',{'global_ridge_fit_count':self.fit_count,'all_gates_passed':self.all_gates_passed,
            'gates':[f.gate for f in self.folds],'config_sha256':self.config_hash})
        assert self.fit_count==0
        if not self.all_gates_passed:raise RuntimeError('Kecukupan sampel tidak lolos; belum ada Ridge dilatih. Perlu rencana penyesuaian berbasis data, bukan skor.')
        print('Semua gate lulus sebelum Ridge; tidak ada penyesuaian fold.',flush=True)
        return self

    def run_models(self):
        assert self.all_gates_passed and hashfile(RUN/'fold_config_locked.json')==self.config_hash
        for fold in self.folds:
            fold.fit_predict_fold();print('Ridge selesai:',fold.label,flush=True)
        assert self.fit_count==12
        self.pred=pd.concat([f.pred for f in self.folds],ignore_index=True)
        assert self.pred['Timestamp target'].is_unique and self.pred['Timestamp target'].is_monotonic_increasing
        assert self.pred['Timestamp target'].max()<self.test_start
        self.metrics=[r for f in self.folds for r in f.metrics];self.month_metrics=[r for f in self.folds for r in f.month_metrics]
        self.pooled=[];self.summary=[];self.ranks=[];self.deltas=[];self.strata=[r for f in self.folds for r in f.strata]
        common=self.pred['Irisan semua metode'].to_numpy()
        for method in ALL:
            ok=np.isfinite(self.pred[method]).to_numpy()
            for domain,keep in [('seluruh berhasil',ok),('irisan semua metode',common)]:
                self.pooled.append(score(self.pred,method,keep,'Gabungan_OOF',domain,len(self.pred)))
                rows=[r for r in self.metrics if r['Pipeline']==method and r['Domain']==domain]
                row={'Pipeline':method,'Domain':domain,'Jumlah fold':len(rows)}
                for col in ['MAE','RMSE','Bias','Cakupan persen','Jumlah prediksi']:
                    values=np.array([r[col] for r in rows],dtype=float)
                    row[col+' rata-rata fold']=float(np.mean(values));row[col+' SD fold']=float(np.std(values,ddof=1))
                    row[col+' minimum']=float(np.min(values));row[col+' maksimum']=float(np.max(values))
                self.summary.append(row)
        for f in self.folds:
            rows=pd.DataFrame([r for r in f.metrics if r['Domain']=='irisan semua metode']).set_index('Pipeline')
            rank_mae=rows.loc[METHODS,'MAE'].rank(method='min');rank_rmse=rows.loc[METHODS,'RMSE'].rank(method='min')
            for method in METHODS:
                self.ranks.append({'Fold':f.label,'Pipeline':method,'Peringkat MAE':float(rank_mae[method]),
                    'Peringkat RMSE':float(rank_rmse[method]),'MAE':rows.loc[method,'MAE'],'RMSE':rows.loc[method,'RMSE']})
                baseline=rows.loc['Persistence'];own=rows.loc[method]
                self.deltas.append({'Fold':f.label,'Pipeline':method,'Jumlah timestamp sama':int(own['Jumlah prediksi']),
                    'Delta MAE vs persistence':float(own.MAE-baseline.MAE),'Delta RMSE vs persistence':float(own.RMSE-baseline.RMSE),
                    'Delta bias vs persistence':float(own.Bias-baseline.Bias),
                    'Perbaikan MAE persen':float(100*(baseline.MAE-own.MAE)/baseline.MAE) if baseline.MAE else None,
                    'Perbaikan RMSE persen':float(100*(baseline.RMSE-own.RMSE)/baseline.RMSE) if baseline.RMSE else None})
        self.pooled_month=[]
        months=self.pred['Timestamp target'].dt.to_period('M').astype(str)
        for method in ALL:
            for mo in months.unique():
                within=months.eq(mo).to_numpy()
                for domain,mask in [('seluruh berhasil',np.isfinite(self.pred[method]).to_numpy()),('irisan semua metode',common)]:
                    self.pooled_month.append({'Bulan target':mo,**score(self.pred,method,mask&within,'Gabungan_OOF',domain,int(within.sum()))})
        self.timing=[r for f in self.folds for r in f.timing]
        self.preprocess_time=[{'Fold':f.label,'Pembentukan histori semua pipeline detik':f.impute_seconds,
            'Jumlah prefix refit':len(f.prefix_logs),'Fit imputer median scaler KNN detik':sum(r['Fit imputer dan scaler detik'] for r in f.prefix_logs),
            'Transform KNN detik':sum(r['Transform KNN detik'] for r in f.prefix_logs)} for f in self.folds]
        self.pred.to_csv(RUN/'prediksi_oof_development.csv',index=False)
        self.pred[['Fold','Origin','Timestamp target','Index origin','Index target','Irisan semua metode']].to_csv(RUN/'timestamp_oof_evaluasi.csv',index=False)
        self._plots()
        self.integrity=[]
        for p in self.refs:
            key=str(p.relative_to(ROOT));after=hashfile(p);assert after==self.before[key],key
            self.integrity.append({'File':key,'SHA256 sebelum':self.before[key],'SHA256 sesudah':after,'Identik':True})
        assert hashfile(RUN/'fold_config_locked.json')==self.config_hash
        self.tables={'Ringkasan_per_fold':self.metrics,'Ringkasan':self.pooled,'Variasi_antar_fold':self.summary,
            'Peringkat_deskriptif':self.ranks,'Vs_persistence_identik':self.deltas,'Skor_bulanan_fold':self.month_metrics,
            'Skor_bulanan_gabungan':self.pooled_month,'Distribusi_train_validasi':self.distributions,
            'Distribusi_target_bulanan':self.distributions_month,'Distribusi_train_aktual':[r for f in self.folds for r in f.train_distribution],
            'Strata_SO2_validasi':self.strata,'Rencana_fold':self.plan,'Gate_sebelum_Ridge':[f.gate for f in self.folds],
            'Alur_sampel':[r for f in self.folds for r in f.flow],'Keluar_irisan_train':[r for f in self.folds for r in f.lost],
            'Keluar_irisan_bulanan':[r for f in self.folds for r in f.monthly_lost],'Penanda_missing_train':[r for f in self.folds for r in f.audit],
            'Parameter_tetap_5B':self.invariants,'Waktu_preprocessing':self.preprocess_time,'Waktu_Ridge':self.timing,
            'Prefix_imputer':[{'Fold':f.label,**r} for f in self.folds for r in f.prefix_logs],
            'Koefisien_Ridge':[r for f in self.folds for r in f.coefficients],
            'Prediksi_OOF':self.pred.to_dict('records'),'Definisi':self.definitions(),'Integritas_SHA256':self.integrity}
        dump(OUT/'tabel_backtesting_preprocessing.json',self.tables)
        dump(RUN/'checks.json',{'test_loaded':False,'test_evaluated':False,'config_sha256':self.config_hash,
            'all_gates_before_ridge':True,'ridge_fit_count':self.fit_count,'oof_target_unique':True,
            'oof_count':len(self.pred),'common_oof_count':int(common.sum()),'old_artifacts_unchanged':True})
        from write_backtesting_report import write_report
        write_report(self)
        return self

    def definitions(self):
        return [{'Istilah':k,'Definisi':v} for k,v in [
            ('Validasi pengembangan','Development=train+validasi lama yang sudah digunakan untuk eksperimen. Hasil bukan validasi final independen; test akhir tetap dikunci.'),
            ('Fold kalender','Train awal120hari, sisa slot development dibagi4jendela target nonoverlap. Durasi berbeda maksimal1slot30menit. Batas tidak ditentukan konsentrasi/skor.'),
            ('Kecukupan','Sebelum semua Ridge: >=340 contoh train bersama (10×34 fitur) dan >=2 nilai target berbeda. Pemeriksaan komputasional, bukan perhitungan daya statistik.'),
            ('Refit dari nol','Setiapfold membuat estimator imputer, scaler dan Ridge baru. Tidak memuat parameterfit5B atau fitted model foldlain. Data validasi foldlama boleh menjadi trainfoldberikut setelah waktunya tersedia.'),
            ('Histori kausal','Kalibrasi90hari pertama trainfold, buffer48slot; prefix refit30hari hanya masa lalu. Imputer validasi beku trainfold. Histori lalu tidak memakai data setelah origin.'),
            ('Fitur','Tanpa CurahHujan; SO2lag0/1/2/6/12/24/48 +10kovariatlag1 +17indikator missing asli. Guard KNN2, k10 distance; ffillL4 tanpa fallback; Ridgealpha1svd tetap.'),
            ('Cakupan persen','Jumlah prediksi finite / seluruh target asli teramati pada validasi fold ×100%. Bulanan denominator target asli bulan+fold. Gabungan denominator seluruh target OOF asli, unik.'),
            ('MAE/RMSE/Bias','e=prediksi−target asli; MAE=mean(abs(e)), RMSE=sqrt(mean(e²)), bias=mean(e), satuanµg/m³. Target kosong tidak diisi.'),
            ('RMSE gabungan','sqrt(total squared error seluruh prediksi nonoverlap / jumlah prediksi), bukan rata-rata RMSE fold. MAE gabungan=totalabs/N, bias gabungan=totalsigned/N.'),
            ('Variasi antar-fold','Mean dan SD sampel ddof1 dari empatfold; bukan intervalkepercayaan atau bukti signifikansi. Train beririsan dan rangkaian waktu berkorelasi.'),
            ('Irisan evaluasi','Timestamp target dengan prediksi finite ketiga Ridge dan persistence dalam fold yang sama. Hash daftar timestamp identik untuk perbandingan.'),
            ('Seleksi train','Ketiga Ridge dalam setiapfold memakai irisan train yang sama. Tabelkeluaririsan berisi jumlah/rentangdistribusi yang sebenarnya layak pada pipeline sendiri.'),
            ('Target lebih2.14','Hanya penelusuran seperti tahap5, bukan batas bahaya/sensor. Foldvalidasi tanpa nilai tersebut tidak diberi klaim evaluasi konsentrasi tinggi.'),
            ('Vs persistence','Delta=Ridge−persistence; perbaikan%=100×(baseline−Ridge)/baseline. Semua pada timestamp sama; negatif berarti memburuk.'),
            ('Peringkat','Urutan deskriptif tiapfold dari MAE/RMSE, bukan uji ekuivalensi/signifikansi. Tidak dipakai untuk mengubah parameter dalam runini.'),
            ('Persistence','SO2 asli terakhir pada/sebelum origin, tanpa batas umur; tidak memakai SO2imputasi. Asumsi latensi sama5B, belum dikonfirmasi.'),
            ('Kualitas','Nol/lonjakan/konstan dipertahankan, tanpa clipping. Pengukuran asli teramati belum otomatis valid secara sensor. Test tidak dimuat nilainya/dievaluasi.')]]

    def _plots(self):
        m=pd.DataFrame(self.metrics);shared=m[m.Domain=='irisan semua metode']
        fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
        for method in ALL:
            g=shared[shared.Pipeline==method]
            axes[0].plot(g.Fold,g.MAE,marker='.',label=method);axes[1].plot(g.Fold,g.RMSE,marker='.',label=method)
            coverage=m[(m.Domain=='seluruh berhasil')&(m.Pipeline==method)]
            axes[2].plot(coverage.Fold,coverage['Cakupan persen'],marker='.',label=method)
        for ax,title in zip(axes,['MAE timestamp bersama','RMSE timestamp bersama','Cakupan terhadap target asli']):
            ax.set_title(title);ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25)
        axes[0].set_ylabel('µg/m³');axes[1].set_ylabel('µg/m³');axes[2].set_ylabel('%');axes[2].set_ylim(0,105);axes[2].legend(fontsize=8)
        fig.suptitle('Expanding-window — validasi pengembangan, bukan test akhir')
        fig.savefig(FIG/'metrik_dan_cakupan_fold.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
        for part in ['train','validation']:
            g=pd.DataFrame(self.distributions);g=g[g.Bagian==part]
            axes[0].errorbar(g.Fold,g['Median SO2'],yerr=[g['Median SO2']-g['Minimum SO2'],g['Maksimum SO2']-g['Median SO2']],fmt='.',label=part)
        actual=pd.DataFrame([f.train_distribution[0] for f in self.folds]);axes[1].bar(actual.Fold,actual['Jumlah slot atau sampel'],label='train aktual bersama')
        axes[0].set(title='Median dan rentang SO2 asli',ylabel='µg/m³');axes[0].legend();axes[1].set(title='Kecukupan train sesudah kalibrasi/buffer',ylabel='Jumlah target asli');axes[1].legend()
        for ax in axes:ax.tick_params(axis='x',rotation=25)
        fig.savefig(FIG/'distribusi_dan_train_fold.png',dpi=150);plt.close(fig)
        frame=pd.DataFrame(self.pooled_month);g=frame[frame.Domain=='irisan semua metode']
        fig,ax=plt.subplots(figsize=(15,5),layout='constrained')
        for method in ALL:
            part=g[g.Pipeline==method];ax.plot(part['Bulan target'],part.MAE,marker='.',label=method)
        ax.set(title='MAE bulanan OOF pada timestamp bersama',ylabel='µg/m³');ax.tick_params(axis='x',rotation=90);ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(FIG/'mae_bulanan_oof.png',dpi=150);plt.close(fig)

    def export_excel(self):
        node=Path(r'C:\Users\INTAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
        result=subprocess.run([str(node),str(ROOT/'scripts/export_quality.mjs'),'tabel_backtesting_preprocessing.json',
            'backtesting_preprocessing.xlsx','.audit_preview_backtesting'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        print(result.stdout)
        if result.returncode:raise RuntimeError(result.stderr)
