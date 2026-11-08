"""5B: ablasi penuh Curah Hujan, dua kohort Ridge, tanpa memuat test."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from copy import deepcopy
from importlib.metadata import version
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

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OLD=OUT/'forecasting_way_urang'
RUN=OUT/'ablasi_curah_hujan_5B';FIG=OUT/'figures/ablasi_curah_hujan_5B'
MAIN='5B_train_diperluas';FIXED='5B_train_lama_tetap';LEGACY='Tahap5_dengan_hujan'

class Ablation(base.Forecast):
    def prepare(self):
        RUN.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
        self.old_cfg=json.loads((OLD/'config_locked.json').read_text(encoding='utf-8'))
        self.old_tables=json.loads((OUT/'forecasting_way_urang_tables.json').read_text(encoding='utf-8'))
        self.old_report=(OUT/'ringkasan_forecasting_acuan_way_urang.md').read_text(encoding='utf-8')
        self.trace_report=(OUT/'penelusuran_sampel_tahap5.md').read_text(encoding='utf-8')
        self.methods=self.old_cfg['pipelines'];assert self.methods==base.METHODS
        self.split=deepcopy(self.old_cfg['split']);self.a=self.split[0]['Jumlah slot'];self.b=self.a+self.split[1]['Jumlah slot']
        self.cov=[c for c in self.old_cfg['covariates'] if c!='Curah Hujan']
        self.lags=list(self.old_cfg['so2_lags']);self.warmup=self.old_cfg['warmup_slots'];self.batch=self.old_cfg['prefix_update_slots']
        self.first_origin=self.warmup+max(self.lags);self.origstop=self.b-1
        cols=['Timestamp','SO2 numerik']+[f'{c} numerik' for c in self.cov]
        self.data=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',usecols=cols,nrows=self.b,parse_dates=['Timestamp'])
        self.times=pd.DatetimeIndex(self.data.Timestamp);self.y=self.data['SO2 numerik'].to_numpy(dtype=float)
        assert self.times[-1]<pd.Timestamp(self.split[2]['Awal'])
        self.matrix=np.column_stack([self.y]+[self.data[f'{c} numerik'].shift(1).to_numpy(dtype=float) for c in self.cov])
        assert len(self.cov)==10 and self.matrix.shape==(self.b,11)
        self.old_ids=pd.read_csv(OLD/'timestamp_train_bersama.csv')['Index origin'].to_numpy(dtype=int)
        self.old_pred=pd.read_csv(OLD/'prediksi_validasi.csv',parse_dates=['Origin','Timestamp target'])
        # Hash seluruh artefak lama yang terlihat pengguna, bukan log inspect/cache internal.
        self.refs=[]
        for p in OUT.rglob('*'):
            if not p.is_file():continue
            parts=p.relative_to(OUT).parts
            if any(v.startswith('.') for v in parts) or p.name.endswith('.inspect.ndjson'):continue
            if 'ablasi_curah_hujan_5B' in parts or p.name.startswith(('hasil_5B','ringkasan_5B','tabel_5B')):continue
            self.refs.append(p)
        self.refs += list((ROOT/'data_raw').rglob('*.xlsx'))
        self.refs += [p for p in (ROOT/'notebooks').glob('*.ipynb') if p.name!='05B_ablasi_curah_hujan.ipynb']
        self.refs += [ROOT/'scripts/forecast_ridge_way_urang.py']
        self.refs=list(dict.fromkeys(self.refs));self.before={str(p.relative_to(ROOT)):hashfile(p) for p in self.refs}
        self.config=deepcopy(self.old_cfg)
        self.config.update({'created_utc':datetime.now(timezone.utc).isoformat(),'experiment':'Tahap 5B ablasi penuh Curah Hujan',
            'covariates':self.cov,'missing_indicators':'17 indikator missing asli, tanpa Curah Hujan dan turunannya',
            'arms':[MAIN,FIXED],'legacy_config_sha256':hashfile(OLD/'config_locked.json'),
            'allowed_change':'Hanya pengeluaran Curah Hujan dari matriks imputer, scaler, fitur Ridge dan indikator; fit ulang.',
            'fixed_cohort':'Timestamp train lama persis; imputer tanpa hujan sama dengan lengan utama.',
            'gate':'Sebelum Ridge: common baru bertambah dan target >2.14 kembali masuk, semua timestamp train lama tetap finite.',
            'comparison':'Irisan timestamp validasi finite untuk seluruh tiga kondisi dan persistence; skor bulanan dengan mask identik.',
            'quality_threshold_2_14':'Hanya penelusuran kohort lama, bukan batas kualitas/bahaya.',
            'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','scipy']},
            'initial_input_hashes':self.before})
        unchanged=['split','horizon_slots','horizon_minutes','so2_lags','covariate_lag_slots','ffill_limit_slots','knn','ridge',
            'warmup_slots','prefix_update_slots','train_input_history','validation_imputer','availability','quality','threads','seed','pipelines','P3']
        self.invariants=[{'Parameter':k,'Sama dengan tahap 5':self.config[k]==self.old_cfg[k],'Nilai':str(self.config[k])} for k in unchanged]
        assert all(r['Sama dengan tahap 5'] for r in self.invariants)
        dump(RUN/'config_locked.json',self.config);self.config_hash=hashfile(RUN/'config_locked.json')
        self.ridge_fit_count=0
        print('5B dikunci. Curah Hujan tidak ada pada matriks/daftar kovariat; belum ada Ridge dilatih.',flush=True)
        return self

    def impute_and_gate(self):
        # Pakai algoritma asli tanpa mengubah file/globals permanen atau jalur keluaran lama.
        previous_run=base.RUN
        try:
            base.RUN=RUN
            super().impute_history()
        finally:base.RUN=previous_run
        assert all('Curah Hujan' not in c for c in self.feature_names) and len(self.feature_names)==34
        candidates=np.arange(self.first_origin,self.a-1);original=np.isfinite(self.y[candidates+1])
        self.flow=[];self.common_mask=original.copy();self.eligible={}
        for method,X in self.X.items():
            finite=np.isfinite(X[candidates]).all(axis=1)
            eligible=original&finite;self.common_mask &= eligible;self.eligible[method]=candidates[eligible]
            origin_all=np.arange(self.a);obs_all=np.isfinite(self.y[origin_all+1]);all_finite=np.isfinite(X[:self.a]).all(axis=1)
            sequence=[('01 Seluruh slot origin train',origin_all),('02 Target t+1 asli teramati',origin_all[obs_all]),
                ('03 Setelah lag struktural tersedia',origin_all[obs_all&(origin_all>=max(self.lags))]),
                ('04 Setelah kalibrasi 90 hari',origin_all[obs_all&(origin_all>=self.warmup)]),
                ('05 Setelah buffer lag 48 slot',origin_all[obs_all&(origin_all>=self.first_origin)]),
                ('06 Setelah imputasi, tanpa membuang sisa NaN',origin_all[obs_all&(origin_all>=self.first_origin)]),
                ('07 Setelah pemeriksaan 34 fitur finite',origin_all[obs_all&(origin_all>=self.first_origin)&all_finite]),
                ('08 Setelah purge target validasi',candidates[eligible])]
            for stage,ids in sequence:self.flow.append(self.cohort_stats(ids,Pipeline=method,Tahap=stage))
        self.new_ids=candidates[self.common_mask]
        self.added=np.setdiff1d(self.new_ids,self.old_ids);self.lost=np.setdiff1d(self.old_ids,self.new_ids)
        self.gate={
            'Ridge fit count saat pemeriksaan':self.ridge_fit_count,'Train bersama lama':len(self.old_ids),
            'Train bersama baru':len(self.new_ids),'Tambahan timestamp':len(self.added),'Timestamp lama tidak lagi layak':len(self.lost),
            'Target tinggi lama':int((self.y[self.old_ids+1]>2.14).sum()),
            'Target tinggi baru':int((self.y[self.new_ids+1]>2.14).sum()),
            'Target tinggi kembali masuk':int((self.y[self.added+1]>2.14).sum()),
            'Minimum target baru':float(self.y[self.new_ids+1].min()),'Median target baru':float(np.median(self.y[self.new_ids+1])),
            'Maksimum target baru':float(self.y[self.new_ids+1].max()),
            'Konfigurasi SHA256':self.config_hash,'Test dimuat':False,'Test dievaluasi':False}
        assert self.ridge_fit_count==0 and len(self.new_ids)>len(self.old_ids)
        assert self.gate['Target tinggi kembali masuk']>0 and len(self.lost)==0
        for method in self.methods:
            assert np.isfinite(self.X[method][self.old_ids]).all()
            self.flow.append(self.cohort_stats(self.new_ids,Pipeline=method,Tahap='09 Final bersama baru, gate sebelum Ridge'))
            pd.DataFrame(self.X[method],columns=self.feature_names,index=self.times[:self.origstop]).to_csv(RUN/f'fitur_{method}.csv',index_label='Origin')
        self.train_distribution=[]
        for arm,ids in [(LEGACY,self.old_ids),(FIXED,self.old_ids),(MAIN,self.new_ids)]:
            self.train_distribution.append(self.cohort_stats(ids,Lengan=arm,Bulan='seluruh kohort'))
            for month in self.times[ids+1].to_period('M').unique():
                sub=ids[self.times[ids+1].to_period('M')==month]
                self.train_distribution.append(self.cohort_stats(sub,Lengan=arm,Bulan=str(month)))
        self.monthly_added=[]
        for month in self.times[self.new_ids+1].to_period('M').unique():
            old=self.old_ids[self.times[self.old_ids+1].to_period('M')==month]
            new=self.new_ids[self.times[self.new_ids+1].to_period('M')==month]
            self.monthly_added.append({'Bulan target':str(month),'Train lama':len(old),'Train baru':len(new),
                'Perubahan jumlah':len(new)-len(old),'Target lebih 2.14 lama':int((self.y[old+1]>2.14).sum()),
                'Target lebih 2.14 baru':int((self.y[new+1]>2.14).sum())})
        for arm,ids in [(MAIN,self.new_ids),(FIXED,self.old_ids)]:
            pd.DataFrame({'Index origin':ids,'Origin':self.times[ids],'Timestamp target train':self.times[ids+1],
                'SO2 target asli':self.y[ids+1],'Lebih tinggi dari 2.14':self.y[ids+1]>2.14}).to_csv(RUN/f'train_{arm}.csv',index=False)
        dump(RUN/'gate_sebelum_Ridge.json',self.gate)
        # Pastikan pemakaian algoritma bersama tidak menulis ke folder eksperimen lama.
        for p in self.refs:assert hashfile(p)==self.before[str(p.relative_to(ROOT))]
        print('GATE LULUS sebelum Ridge:',self.gate,flush=True)
        return self

    def cohort_stats(self,ids,**meta):
        values=self.y[ids+1];v=values[np.isfinite(values)]
        return {**meta,'Jumlah sampel':len(ids),'Target asli teramati':len(v),'Origin awal':self.times[ids[0]] if len(ids) else None,
            'Origin akhir':self.times[ids[-1]] if len(ids) else None,'Target awal':self.times[ids[0]+1] if len(ids) else None,
            'Target akhir':self.times[ids[-1]+1] if len(ids) else None,'Minimum SO2':float(v.min()) if len(v) else None,
            'Median SO2':float(np.median(v)) if len(v) else None,'Maksimum SO2':float(v.max()) if len(v) else None,
            'Mean SO2':float(v.mean()) if len(v) else None,'SD SO2':float(v.std(ddof=1)) if len(v)>1 else None,
            'Target lebih 2.14':int((v>2.14).sum())}

    def run_models(self):
        assert hashfile(RUN/'config_locked.json')==self.config_hash
        gate=json.loads((RUN/'gate_sebelum_Ridge.json').read_text(encoding='utf-8'))
        assert gate['Ridge fit count saat pemeriksaan']==0 and gate['Target tinggi kembali masuk']>0
        target_ids=np.arange(self.a,self.b);target_ids=target_ids[np.isfinite(self.y[target_ids])];origin=target_ids-1
        self.origin=origin;self.target_ids=target_ids
        assert np.array_equal(target_ids,self.old_pred['Index target'].to_numpy())
        self.preds={};self.timing=[];self.coefficients=[]
        last=np.maximum.accumulate(np.where(np.isfinite(self.y[:self.origstop]),np.arange(self.origstop),-1))
        persistence=self.y[last[origin]];ages=origin-last[origin]
        assert np.allclose(persistence,self.old_pred.Persistence) and np.array_equal(ages,self.old_pred['Persistence umur donor slot'])
        with threadpool_limits(limits=self.config['threads']):
            for arm,ids in [(MAIN,self.new_ids),(FIXED,self.old_ids)]:
                frame=pd.DataFrame({'Origin':self.times[origin],'Timestamp target':self.times[target_ids],
                    'Index origin':origin,'Index target':target_ids,'SO2 target asli':self.y[target_ids],
                    'Persistence':persistence,'Persistence umur donor slot':ages})
                for method in self.methods:
                    X=self.X[method]
                    tick=perf_counter();scaler=StandardScaler().fit(X[ids]);Z=scaler.transform(X[ids]);scale_time=perf_counter()-tick
                    tick=perf_counter();model=Ridge(**self.config['ridge']).fit(Z,self.y[ids+1]);fit_time=perf_counter()-tick
                    self.ridge_fit_count+=1
                    valid=np.isfinite(X[origin]).all(axis=1);pred=np.full(len(origin),np.nan)
                    tick=perf_counter();pred[valid]=model.predict(scaler.transform(X[origin[valid]]));pred_time=perf_counter()-tick
                    frame[method]=pred
                    self.timing.append({'Lengan':arm,'Pipeline':method,'Jumlah train':len(ids),
                        'Scaler fitur fit detik':scale_time,'Ridge fit detik':fit_time,'Predict detik':pred_time})
                    for j,col in enumerate(self.feature_names):
                        self.coefficients.append({'Lengan':arm,'Pipeline':method,'Fitur':col,
                            'Koefisien fitur standar':float(model.coef_[j]),'Mean train':float(scaler.mean_[j]),
                            'Skala train':float(scaler.scale_[j]),'Intercept':float(model.intercept_)})
                frame['Irisan empat metode']=np.isfinite(frame[self.methods+['Persistence']]).all(axis=1)
                self.preds[arm]=frame
        assert self.ridge_fit_count==6
        self.legacy=self.old_pred.copy();self.preds[LEGACY]=self.legacy
        self.global_common=np.ones(len(origin),bool)
        for frame in self.preds.values():self.global_common &= np.isfinite(frame[self.methods+['Persistence']]).all(axis=1).to_numpy()
        self.metrics=[];self.month_metrics=[];self.compare=[];self.compare_month=[];self.deltas=[]
        month=self.times[target_ids].to_period('M').astype(str)
        def score(arm,method,keep,domain,total):
            p=self.preds[arm].loc[keep,method].to_numpy();truth=self.y[target_ids[keep]];e=p-truth;n=len(e)
            return {'Lengan':arm,'Pipeline':method,'Domain':domain,'Jumlah prediksi':n,'Denominator target asli':total,
                'Cakupan persen':100*n/total if total else None,'MAE':float(np.abs(e).mean()) if n else None,
                'RMSE':float(np.sqrt(np.mean(e**2))) if n else None,'Bias':float(e.mean()) if n else None,
                'Prediksi negatif':int((p<0).sum()),'SHA256 timestamp target':self.timestamp_hash(keep)}
        for arm in [LEGACY,FIXED,MAIN]:
            frame=self.preds[arm];within=np.isfinite(frame[self.methods+['Persistence']]).all(axis=1).to_numpy()
            for method in self.methods+['Persistence']:
                ok=np.isfinite(frame[method]).to_numpy()
                if arm!=LEGACY:
                    for domain,mask in [('seluruh berhasil',ok),('irisan internal 5B',within)]:
                        self.metrics.append(score(arm,method,mask,domain,len(origin)))
                        for mo in np.unique(month):
                            inmonth=month==mo
                            self.month_metrics.append({'Bulan target':mo,**score(arm,method,mask&inmonth,domain,int(inmonth.sum()))})
                self.compare.append(score(arm,method,self.global_common,'irisan identik lama dan kedua 5B',len(origin)))
                for mo in np.unique(month):
                    inmonth=month==mo
                    self.compare_month.append({'Bulan target':mo,**score(arm,method,self.global_common&inmonth,'irisan identik lama dan kedua 5B',int(inmonth.sum()))})
            if arm!=LEGACY:
                frame['Irisan identik dengan tahap5']=self.global_common
                frame.to_csv(RUN/f'prediksi_{arm}.csv',index=False)
        compare=pd.DataFrame(self.compare)
        for label,before,after in [('Ablasi pada train tetap',LEGACY,FIXED),('Penambahan train tanpa hujan',FIXED,MAIN),('Gabungan ablasi dan train baru',LEGACY,MAIN)]:
            for method in self.methods:
                first=compare[(compare.Lengan==before)&(compare.Pipeline==method)].iloc[0]
                end=compare[(compare.Lengan==after)&(compare.Pipeline==method)].iloc[0]
                self.deltas.append({'Perbandingan':label,'Pipeline':method,'Sebelum':before,'Sesudah':after,
                    'Jumlah target identik':int(first['Jumlah prediksi']),'Delta MAE':end.MAE-first.MAE,
                    'Delta RMSE':end.RMSE-first.RMSE,'Delta bias':end.Bias-first.Bias,
                    'Catatan':'Delta=sesudah−sebelum; bukan klaim pengaruh murni tanpa batasan.'})
        ledger=pd.DataFrame({'Origin':self.times[origin],'Timestamp target':self.times[target_ids],
            'Index origin':origin,'Index target':target_ids,'SO2 target asli':self.y[target_ids],
            'Irisan lama':np.isfinite(self.legacy[self.methods+['Persistence']]).all(axis=1),
            'Irisan 5B':self.preds[MAIN]['Irisan empat metode'],'Irisan seluruh perbandingan':self.global_common})
        ledger.to_csv(RUN/'timestamp_evaluasi_identik.csv',index=False)
        self._plots_5b()
        self.integrity=[]
        for p in self.refs:
            key=str(p.relative_to(ROOT));after=hashfile(p);assert after==self.before[key],key
            self.integrity.append({'File':key,'SHA256 sebelum':self.before[key],'SHA256 sesudah':after,'Identik':True})
        assert hashfile(RUN/'config_locked.json')==self.config_hash
        self.tables={'Ringkasan':self.compare,'Hasil_5B_cakupan':self.metrics,'Bulanan_identik':self.compare_month,
            'Bulanan_5B_cakupan':self.month_metrics,'Delta_sensitivitas':self.deltas,
            'Gate_sebelum_Ridge':[self.gate],'Alur_sebelum_Ridge':self.flow,
            'Distribusi_train':self.train_distribution,'Perubahan_train_bulanan':self.monthly_added,
            'Konfigurasi_tetap':self.invariants,'Prefix_imputer':self.prefix_logs,
            'Waktu_Ridge':self.timing,'Koefisien_Ridge':self.coefficients,
            'Prediksi_utama':self.preds[MAIN].to_dict('records'),'Prediksi_train_lama':self.preds[FIXED].to_dict('records'),
            'Timestamp_evaluasi':ledger.to_dict('records'),'Definisi':self.definitions_5b(),'Integritas_SHA256':self.integrity}
        dump(OUT/'tabel_5B_ablasi_curah_hujan.json',self.tables)
        dump(RUN/'checks.json',{'test_loaded':False,'test_evaluated':False,'ridge_fit_count':self.ridge_fit_count,
            'config_sha256':self.config_hash,'gate_passed_before_ridge':True,'old_ids_subset_new':True,
            'global_eval_count':int(self.global_common.sum()),'global_eval_hash':self.timestamp_hash(self.global_common),
            'rain_in_features':False,'rain_in_imputer_matrix':False,'source_and_legacy_unchanged':True,
            'impute_history_total_seconds':self.impute_seconds})
        from write_ablation_5b_report import write_report
        write_report(self)
        return self

    def timestamp_hash(self,mask):
        from hashlib import sha256
        return sha256('\n'.join(self.times[self.target_ids[mask]].astype(str)).encode()).hexdigest()

    def definitions_5b(self):
        return [{'Istilah':k,'Definisi':v} for k,v in [
            ('Lengan utama','5B_train_diperluas: tanpa Curah Hujan, train bersama baru ketiga pipeline.'),
            ('Sensitivitas','5B_train_lama_tetap: tanpa Curah Hujan dengan timestamp Ridge train lama persis. Histori imputer tanpa hujan sama dengan lengan utama.'),
            ('Ablasi penuh','Curah Hujan dikeluarkan dari matriks imputer/scaler jarak, fitur Ridge dan indikator missing. P2 dapat berubah karena jarak KNN berbeda, bukan sekadar menghapus koefisien Ridge.'),
            ('Parameter tetap','Split, horizon, lag SO2, lag kovariat, Ridge alpha1 svd, warmup90hari, buffer48, pembaruanprefix30hari, guard2kovariat, ffillL4, donor, kualitas dan tanpa fallback sama.'),
            ('Histori kausal','Imputer train fit prefix masa lalu; validasi memakai parameter seluruh train beku; tidak memperbaiki histori lama dengan informasi masa depan. Tidak ada nilai test dimuat.'),
            ('Fitur','7 lag SO2 +10 kovariat lag1 +17 indikator missing asli =34. Tidak ada turunan Curah Hujan atau waktu mentah/arah angin linear.'),
            ('Target lebih2.14','Hitungan penelusuran kohort lama; bukan batas kualitas atau bahaya. Train mencakupnya tidak membuktikan kemampuan prediksi pada konsentrasi tinggi.'),
            ('MAE/RMSE/bias','Error=prediksi−target asli; MAE=mean(abs(e)), RMSE=sqrt(mean(e²)), bias=mean(e). Satuan µg/m³. Target tidak diimputasi.'),
            ('Cakupan persen','Jumlah prediksi finite / seluruh4524 target validasi asli ×100%; bulanan memakai target asli bulan itu. Pada perbandinganidentik, pembilang hanya irisan seluruh kondisi.'),
            ('Irisan identik','Timestamp finite pada ketiga pipeline +persistence dalam hasil lama, 5B utama dan 5B trainlama. Hash timestamp sama; domain ini memisahkan perubahan eligibility evaluasi.'),
            ('Pengaruh utama','Lama versus5B diperluas menggabungkan pengeluaran fitur dan perubahan komposisi train. Tidak mengisolasi pengaruh murni Curah Hujan.'),
            ('Pengaruh sampel tetap','Lama versus5B trainlama menilai ablasi fitur termasuk refit preprocessing pada timestamp Ridge train tetap. Penambahantrain dibandingkan antara dua lengan tanpa hujan.'),
            ('Waktu komputasi','Histori/imputer tanpa hujan disiapkan sekali dan dipakai bersama dua lengan, bukan didobel. Fit scaler fitur, Ridge dan predict dicatat per lengan/pipeline.'),
            ('Persistence','SO2 asli terakhir tersedia pada/sebelum origin; umur donor sama dengan tahap5. Tidak memakai hasil imputasi.'),
            ('Asumsi latensi','SO2 pada origin dianggap tersedia, kovariat lag1 dianggap tiba dalam30menit. Belum dikonfirmasi penyedia data; nol/lonjakan/konstan dipertahankan, tanpa clipping.')]]

    def _plots_5b(self):
        comp=pd.DataFrame(self.compare);fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
        x=np.arange(3);width=.23
        for k,arm in enumerate([LEGACY,FIXED,MAIN]):
            rows=comp[(comp.Lengan==arm)&comp.Pipeline.isin(self.methods)].set_index('Pipeline').loc[self.methods]
            axes[0].bar(x+(k-1)*width,rows.MAE,width,label=arm)
            axes[1].bar(x+(k-1)*width,rows.RMSE,width,label=arm)
            axes[2].bar(x+(k-1)*width,rows.Bias,width,label=arm)
        for ax,title in zip(axes,['MAE — timestamp identik','RMSE — timestamp identik','Bias — timestamp identik']):
            ax.set_title(title);ax.set_xticks(x,['P0 median','P1 ffill L4','P2 KNN10']);ax.set_ylabel('µg/m³');ax.grid(axis='y',alpha=.15)
        axes[2].legend(fontsize=7);fig.suptitle('5B: fitur dan komposisi train dibedakan melalui sensitivitas')
        fig.savefig(FIG/'perbandingan_identik.png',dpi=150);plt.close(fig)
        df=pd.DataFrame(self.compare_month);fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
        for ax,method in zip(axes,self.methods):
            for arm in [LEGACY,FIXED,MAIN]:
                g=df[(df.Lengan==arm)&(df.Pipeline==method)];ax.plot(g['Bulan target'],g.MAE,marker='.',label=arm)
            persistence=df[(df.Lengan==LEGACY)&(df.Pipeline=='Persistence')]
            ax.plot(persistence['Bulan target'],persistence.MAE,linestyle='--',color='black',label='Persistence')
            ax.set(title=method,ylabel='MAE (µg/m³)');ax.tick_params(axis='x',rotation=30);ax.grid(alpha=.2)
        axes[-1].legend(fontsize=7);fig.suptitle('MAE bulanan pada irisan timestamp lama dan baru')
        fig.savefig(FIG/'bulanan_identik.png',dpi=150);plt.close(fig)
        rows=pd.DataFrame(self.monthly_added);fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
        x=np.arange(len(rows));axes[0].bar(x-.2,rows['Train lama'],.4,label='lama');axes[0].bar(x+.2,rows['Train baru'],.4,label='5B diperluas')
        axes[1].bar(x,rows['Target lebih 2.14 baru'],label='target >2.14 kembali tersedia')
        for ax in axes:ax.set_xticks(x,rows['Bulan target'],rotation=90);ax.set_ylabel('Jumlah target asli');ax.legend(fontsize=8)
        axes[0].set_title('Komposisi train bersama');axes[1].set_title('Penelusuran konsentrasi lebih tinggi, bukan ambang kualitas')
        fig.savefig(FIG/'perluasan_train.png',dpi=150);plt.close(fig)

    def export_excel(self):
        node=Path(r'C:\Users\INTAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
        result=subprocess.run([str(node),str(ROOT/'scripts/export_quality.mjs'),'tabel_5B_ablasi_curah_hujan.json',
            'hasil_5B_ablasi_curah_hujan.xlsx','.audit_preview_5B'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        print(result.stdout)
        if result.returncode:raise RuntimeError(result.stderr)
