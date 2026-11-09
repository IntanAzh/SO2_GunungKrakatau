import numpy as np
import pandas as pd
from hashlib import sha256
from interstation_stage9 import ROOT,OUT,RUN,FIG,SCENARIOS,ALL,OFFSETS,plt
from calibration_sensitivity import distribution
from audit_quality import hashfile
from pilot_way_urang import dump

def score(frame,scenario,keep,fold,domain,den):
    p=frame.loc[keep,scenario].to_numpy();y=frame.loc[keep,'SO2 target asli'].to_numpy();err=p-y;n=len(err)
    return {'Fold':fold,'Skenario':scenario,'Domain':domain,'N':n,'Denominator target asli':den,'Cakupan persen':100*n/den if den else None,
        'MAE':float(np.abs(err).mean()) if n else None,'RMSE':float(np.sqrt(np.mean(err**2))) if n else None,'Bias':float(err.mean()) if n else None,
        'Prediksi negatif':int((p<0).sum()),'SHA256 timestamp':sha256('\n'.join(frame.loc[keep,'Timestamp target'].astype(str)).encode()).hexdigest()}

def summarize(e):
    e.pred=pd.concat([f.pred for f in e.folds],ignore_index=True);assert e.pred['Timestamp target'].is_unique and e.pred['Timestamp target'].is_monotonic_increasing
    e.pred.to_csv(RUN/'prediksi_oof.csv',index=False)
    e.pred[['Fold','Origin','Timestamp target','Index origin','Index target','Irisan semua skenario']].to_csv(RUN/'timestamp_evaluasi_bersama.csv',index=False)
    e.metrics=[];e.month_metrics=[];e.pooled=[];e.variation=[];e.deltas=[];e.month_deltas=[];e.age_stats=[];e.age_month=[];e.s0_audit=[]
    for fold in [f.label for f in e.folds]+['Gabungan_OOF']:
        frame=e.pred if fold=='Gabungan_OOF' else e.pred[e.pred.Fold==fold]
        months=frame['Timestamp target'].dt.to_period('M').astype(str);common=frame['Irisan semua skenario'].to_numpy();records=[]
        for scenario in ALL:
            for domain,keep in [('mandiri',np.isfinite(frame[scenario]).to_numpy()),('bersama',common)]:
                row=score(frame,scenario,keep,fold,domain,len(frame));records.append(row)
                if fold=='Gabungan_OOF':e.pooled.append(row)
                else:e.metrics.append(row)
                for mo in months.unique():
                    within=months.eq(mo).to_numpy();e.month_metrics.append({'Bulan target':mo,**score(frame,scenario,keep&within,fold,domain,int(within.sum()))})
        baseline=next(r for r in records if r['Skenario']=='S0' and r['Domain']=='bersama')
        pers=next(r for r in records if r['Skenario']=='Persistence' and r['Domain']=='bersama')
        for row in records:
            if row['Domain']!='bersama' or row['Skenario']=='Persistence':continue
            e.deltas.append({'Fold':fold,'Skenario':row['Skenario'],'N identik':row['N'],
                'Delta MAE vs S0':row['MAE']-baseline['MAE'],'Delta RMSE vs S0':row['RMSE']-baseline['RMSE'],
                'Perbaikan MAE persen':100*(baseline['MAE']-row['MAE'])/baseline['MAE'],
                'Perbaikan RMSE persen':100*(baseline['RMSE']-row['RMSE'])/baseline['RMSE'],
                'Delta MAE vs persistence':row['MAE']-pers['MAE'],'Delta RMSE vs persistence':row['RMSE']-pers['RMSE']})
    for (scenario,domain),g in pd.DataFrame(e.metrics).groupby(['Skenario','Domain'],sort=False):
        row={'Skenario':scenario,'Domain':domain,'Fold N':4}
        for col in ['MAE','RMSE','Bias','Cakupan persen','N']:
            row[col+' mean fold']=float(g[col].mean());row[col+' SD fold']=float(g[col].std(ddof=1));row[col+' min fold']=float(g[col].min());row[col+' max fold']=float(g[col].max())
        e.variation.append(row)
    monthly=pd.DataFrame([r for r in e.month_metrics if r['Domain']=='bersama'])
    for (fold,month),g in monthly.groupby(['Fold','Bulan target'],sort=False):
        g=g.set_index('Skenario');base=g.loc['S0'];pers=g.loc['Persistence']
        for scenario in SCENARIOS:
            row=g.loc[scenario]
            e.month_deltas.append({'Fold':fold,'Bulan target':month,'Skenario':scenario,'N identik':int(row.N),
                'Delta MAE vs S0':float(row.MAE-base.MAE),'Delta RMSE vs S0':float(row.RMSE-base.RMSE),
                'Perbaikan MAE persen':float(100*(base.MAE-row.MAE)/base.MAE),'Perbaikan RMSE persen':float(100*(base.RMSE-row.RMSE)/base.RMSE),
                'Delta MAE vs persistence':float(row.MAE-pers.MAE),'Delta RMSE vs persistence':float(row.RMSE-pers.RMSE)})
    for f in e.folds:
        old_ids=pd.read_csv(OUT/f'sensitivitas_kalibrasi/{f.label}/C30_median/timestamp_train.csv')['Index origin'].to_numpy()
        same=np.array_equal(old_ids,f.shared['train']);oldp=pd.read_csv(OUT/f'sensitivitas_kalibrasi/{f.label}/C30_median/prediksi_evaluasi.csv')
        delta=np.max(np.abs(oldp.C30_median.to_numpy()-f.pred.S0.to_numpy())) if same else None
        if same:assert np.allclose(oldp.C30_median,f.pred.S0,atol=1e-8,rtol=1e-10)
        e.s0_audit.append({'Fold':f.label,'S0 fitur identik C30 lama':True,'S0 dilatih ulang':True,'Train lama':len(old_ids),
            'Train bersama baru':len(f.shared['train']),'Cohort identik':same,'Maks selisih prediksi jika cohort sama':float(delta) if delta is not None else None})
        for donor in ['Pulomerak','Talang']:
            for part,ids in f.shared.items():
                a=e.ages[donor].iloc[ids];row={'Fold':f.label,'Donor':donor,'Bagian sampel':part,'N origin':len(ids)}
                for col,suffix in [('Usia asli menit grid','grid'),('Usia asli menit semua','semua')]:
                    v=a[col].dropna().to_numpy();q=np.quantile(v,[0,.5,.9,.95,.99,1]) if len(v) else [np.nan]*6
                    row.update({f'{key} menit {suffix}':float(value) for key,value in zip(['Minimum','Median','P90','P95','P99','Maksimum'],q)})
                    row['Tanpa pengamatan asli '+suffix]=int(a[col].isna().sum())
                e.age_stats.append(row)
                month=e.times[ids+1].to_period('M')
                for mo in month.unique():
                    sub=a.iloc[np.flatnonzero(month==mo)]['Usia asli menit grid'];e.age_month.append({'Fold':f.label,'Donor':donor,'Bagian sampel':part,'Bulan target':str(mo),
                        'N origin':len(sub),'Usia median menit grid':float(sub.median()),'Usia maksimum menit grid':float(sub.max()),'Tanpa pengamatan asli':int(sub.isna().sum())})
    e.constant=[r for f in e.folds for r in f.constant_usage]
    e.constant_summary=[]
    for (fold,donor,part),g in pd.DataFrame(e.constant).groupby(['Fold','Donor','Bagian sampel model'],sort=False):
        e.constant_summary.append({'Fold':fold,'Donor':donor,'Bagian sampel model':part,'Jumlah rentetan terkait':len(g),
            'Run terpanjang slot':int(g['Jumlah slot run'].max()),'Durasi run terpanjang menit':int(g['Durasi run menit'].max()),
            'Timestamp sumber konstan unik digunakan':int(g['Timestamp sumber unik masuk fitur'].sum())})
    e.integrity=[]
    for path in e.refs:
        key=path.relative_to(ROOT).as_posix();after=hashfile(path);assert after==e.before[key],key
        e.integrity.append({'File':key,'SHA256 sebelum':e.before[key],'SHA256 sesudah':after,'Identik':True})
    assert hashfile(RUN/'config_locked.json')==e.config_hash
    e.tables={'Ringkasan':e.pooled,'Skor_per_fold':e.metrics,'Vs_S0_persistence':e.deltas,'Vs_S0_bulanan':e.month_deltas,'Variasi_antar_fold':e.variation,'Skor_bulanan':e.month_metrics,
        'Alur_sampel':[r for f in e.folds for r in f.flow],'Target_distribusi':[r for f in e.folds for r in f.dist],
        'Target_bulanan':[r for f in e.folds for r in f.month_dist],'Keluar_irisan':[r for f in e.folds for r in f.lost],
        'Keluar_irisan_bulanan':[r for f in e.folds for r in f.lost_month],'Audit_S0_C30':e.s0_audit,'Sumber_penyelarasan':e.source_checks,
        'Offgrid_development':e.offgrid,'Kualitas_SO2_dev':e.quality,'Usia_donor':e.age_stats,'Usia_bulanan':e.age_month,
        'Rentetan_konstan':e.constant,'Ringkas_konstan':e.constant_summary,'Gate_pra_fit':[f.gate for f in e.folds],
        'Prefix_median':[{'Fold':f.label,**r} for f in e.folds for r in f.prefix],
        'Ridge_scaler':[r for f in e.folds for r in f.coefficients],'Waktu_Ridge':[r for f in e.folds for r in f.timing],
        'Waktu_preprocessing':[{'Fold':f.label,'Detik seluruh skenario':f.preprocess_seconds} for f in e.folds],
        'Prediksi':e.pred.to_dict('records'),
        'Konfigurasi':[{'Parameter':k,'Nilai':('UTC: '+v if k=='created_utc' else str(v))} for k,v in e.cfg.items() if k!='initial_hashes'],
        'Definisi':[{'Istilah':k,'Definisi':v} for k,v in [
            ('Wilayah','Development saja, sudah digunakan eksperimen sebelumnya. Target WayUrang tetap asli. Tidak memuat nilaiSO2 test; timestampbatas dibaca hanya untuk menghentikan pembacaan sumber.'),
            ('Donor','Stasiun terpisah, bukan digabung satu seri. SO2 origin−1 lalu lag0/1/2/6/12/24/48 terhadapdonor, totaloffset1/2/3/7/13/25/49. Tidak ada meteorologi donor.'),
            ('Waktu','Grid exact00/30, clocknaive dianggap sebanding tanpa konversizona. Delay30menit asumsi, belum terverifikasi. Offgrid dicatat, tidak masuk fitur/statistikmedian, tidak dibulatkan.'),
            ('Median','Warmup1440slot; prefixtrain hanya data asli masa lalu, update1440slot. Wayfreeze[:a], donorfreeze[:a−1] tersedia pada originpertamaa−1. Prefix tanpa nilai tetapNaN; tidak ada donor untuk target.'),
            ('Scaler','StandardScaler seluruh fitur tiapscenario hanya timestamptrain bersama. 34/48/48/62fitur termasukmissingasli. Target tidak diskalakan. Usia donor bukan fitur.'),
            ('Cohort','S0firstorigin1488; S1/2/3first1489 untukmaxlag49. Semua train target<awal evaluasi. Irisan4scenario dengan targetasli; ledger alasan/kandidat/mandiri/common disimpan tanpa menghapus raw.'),
            ('Usia','Origin−timestamp original numerik terakhir<=origin−30menit; minimum30menit bila cutoffteramati. Grid dansemua sumber (termasukoffgrid) terpisah. Tambahanumursetelahcutoff=umurgrid−30. Tidak menyatakan nilaiasli valid otomatis.'),
            ('Konstan','Kesamaan tepat ≥2slot asli grid, NaNmemutus. Durasi=Nslot×30menit; bukanakhir−awal. Penggunaan dihitung pada timestamp sumberunik semua7lag yang masuk sampelbersama. Peristiwa antar-fold dapat berulang; bukan jumlahkejadian independen.'),
            ('Cakupan','100×prediksi finite/seluruh targetWayUrang asli teramati dalamjendela. Domainbersama memakai N irisan dengan denominatorasli yang sama. Bulanan denominator targetasli bulan+fold.'),
            ('Skor','e=prediksi−asli; MAE=Σ|e|/N; RMSE=√Σe²/N; bias=Σe/N (µg/m³). Gabungan dari seluruherror OOF unik, bukan rata-rataRMSEfold.'),
            ('Perbaikan','100×(errorS0−errorskenario)/errorS0; positifberarti errorlebih kecil. Deltaskenario−baseline negatifberarti errorlebih kecil. Tidak cukup menilai dari satu metrik.'),
            ('Variasi','SDsampel antar4fold ddof1; expandingtrain salingterkait. Tidak mengklaim signifikansi atau hubungan sebab-akibat dari selisih mean.'),
            ('Kualitas','Nol,ekstrem,konstan danqualityflags asli dipertahankan; tidak clipping hasil. Kelengkapan tinggi setelah median tidak menjamin informasi donor baru/segar atau keabsahan sensor.'),
            ('Interpretasi','Manfaat prediktif kondisional pada desain/clock/delay dancohort; bukan bukti transportpolusi, kausalitas spasial atau sumbervulkanik. Tidak memilihstasiunfinal/algoritmaDLterbaik.'),
        ]],'Integritas_SHA256':e.integrity}
    dump(OUT/'tabel_kontribusi_antarstasiun.json',e.tables)
    dump(RUN/'checks.json',{'ridge_fit_count':e.fit_count,'test_loaded':False,'test_evaluated':False,'deep_learning_trained':False,
        'config_sha256':e.config_hash,'oof_targets':len(e.pred),'common_targets':int(e.pred['Irisan semua skenario'].sum()),'old_hashes_identical':len(e.integrity)})
    plots(e)
    from write_interstation_stage9 import report
    report(e)

def plots(e):
    common=pd.DataFrame([r for r in e.metrics if r['Domain']=='bersama']);fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
    for scenario in ALL:
        g=common[common.Skenario==scenario]
        for ax,col in zip(axes,['MAE','RMSE','Bias']):ax.plot(g.Fold,g[col],marker='o',label=scenario);ax.set(title=col+' pada timestamp bersama',ylabel='µg/m³');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.savefig(FIG/'skor_per_fold.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    d=pd.DataFrame([r for r in e.deltas if r['Fold']!='Gabungan_OOF' and r['Skenario']!='S0'])
    for scenario in ['S1','S2','S3']:
        g=d[d.Skenario==scenario]
        for ax,col in zip(axes,['Perbaikan MAE persen','Perbaikan RMSE persen']):ax.plot(g.Fold,g[col],marker='o',label=scenario);ax.set(title=col,ylabel='%');ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.2)
    axes[-1].legend();fig.savefig(FIG/'perubahan_vs_S0.png',dpi=150);plt.close(fig)
    t=pd.Timestamp('2025-06-22 17:30:00');g=e.pred[e.pred.Fold=='Fold_01'].set_index('Timestamp target').reindex(pd.date_range(t-pd.Timedelta(hours=72),t+pd.Timedelta(hours=72),freq='30min'))
    fig,ax=plt.subplots(figsize=(15,4),layout='constrained')
    for col in ['SO2 target asli']+ALL:ax.plot(g.index,g[col],label=col,lw=1)
    ax.axvline(t,color='black',ls='--');ax.set(title='Transisi fold1 — semua skenario, tanpa clipping',ylabel='µg/m³');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2)
    fig.savefig(FIG/'transisi_22_juni.png',dpi=150);plt.close(fig)
