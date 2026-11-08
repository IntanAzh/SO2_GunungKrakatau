"""Stored LSTM and Ridge prediction comparisons, no fitting."""
import json
import numpy as np
import pandas as pd
from lstm_stage7 import OUT,RUN,FIG,METHODS,ALL,metric,dist,plt
from audit_quality import hashfile
from pilot_way_urang import dump

def summarize(e):
    e.accounting=[]
    e.target_quantiles=[]
    for f in e.folds:
        spans={'inner_train':(0,f.a),'inner_validation':(f.a,f.outer_a),'evaluation':(f.outer_a,f.b)}
        for part in ['inner_train','inner_validation']:
            ids=f.shared[part];v=f.y[ids+1];q=np.quantile(v,[.01,.25,.5,.75,.99])
            e.target_quantiles.append({'Fold':f.label,'Bagian':part,'N':len(v),'Minimum':float(v.min()),
                'P1':float(q[0]),'P25':float(q[1]),'Median':float(q[2]),'P75':float(q[3]),'P99':float(q[4]),
                'Maksimum':float(v.max()),'Mean':float(v.mean()),'SD':float(v.std(ddof=1))})
        for row in f.flow:
            lo,hi=spans[row['Bagian']]
            e.accounting.append({'Fold':f.label,'Bagian':row['Bagian'],'Pipeline':row['Pipeline'],
                'Slot target kalender sebelum seleksi':hi-lo,'Target asli teramati kalender':int(np.isfinite(f.y[lo:hi]).sum()),
                'Kandidat setelah kalibrasi buffer purge':row['Kandidat origin setelah buffer purge'],
                'Target asli sebelum pembentukan sekuens':row['Target asli teramati'],
                'Setelah sekuens48 finite mandiri':row['Sekuens finite mandiri'],'Setelah irisan pipeline':row['Sampel bersama'],
                'Target kosong kandidat':row['Kandidat origin setelah buffer purge']-row['Target asli teramati'],
                'Sekuens gagal finite setelah imputasi':row['Tidak layak karena sekuens'],'Keluar irisan tambahan':row['Keluar karena irisan']})
    f=e.folds[0];e.fold1_sequence_breaks=[]
    invalid=np.flatnonzero(~np.isfinite(f.filled['P2_knn_k10']).all(axis=1))
    for part in ['inner_train','inner_validation']:
        ids=f.parts[part];observed=ids[np.isfinite(f.y[ids+1])]
        for step in invalid:
            affected=observed[(observed>=step)&(observed<=step+47)]
            if len(affected):e.fold1_sequence_breaks.append({'Bagian':part,'Index langkah input':int(step),'Timestamp langkah input':f.times[step],
                'Timestamp sumber kovariat lag1':f.times[step-1],'SO2 asli pada langkah':f.y[step],
                'Kovariat asli tersedia':int(np.isfinite(f.matrix[step,1:]).sum()),'Kovariat tetap kosong':int((~np.isfinite(f.filled['P2_knn_k10'][step,1:])).sum()),
                'Target asli kandidat terdampak':len(affected),'Awal origin terdampak':f.times[affected[0]],'Akhir origin terdampak':f.times[affected[-1]]})
    e.calendar_distribution=[]
    for f in e.folds:
        for label,lo,hi in [('inner_train',0,f.a),('inner_validation',f.a,f.outer_a),('evaluation',f.outer_a,f.b)]:
            e.calendar_distribution.append({'Fold':f.label,'Bagian':label,'Awal target kalender':e.times[lo],
                'Akhir target kalender':e.times[hi-1],'Slot kalender':hi-lo,**dist(e.y[lo:hi])})
    old=pd.read_csv(OUT/'backtesting_way_urang/prediksi_oof_development.csv',parse_dates=['Origin','Timestamp target'])
    ridge_names=['Ridge_'+c for c in METHODS];r=old[['Fold','Timestamp target','SO2 target asli','Persistence']+METHODS].rename(columns={c:'Ridge_'+c for c in METHODS}|{'SO2 target asli':'Ridge_target','Persistence':'Ridge_persistence'})
    e.pred=e.pred.merge(r,on=['Fold','Timestamp target'],how='left',validate='many_to_one')
    assert np.allclose(e.pred['SO2 target asli'],e.pred.Ridge_target) and np.allclose(e.pred.Persistence,e.pred.Ridge_persistence)
    e.pred['Irisan LSTM Ridge persistence']=np.isfinite(e.pred[ALL+ridge_names]).all(axis=1)
    e.pred.to_csv(RUN/'prediksi_oof_seluruh_seed.csv',index=False)
    e.metrics=[];e.pooled=[];e.monthly=[];e.ridge_comparison=[];e.ranks=[];e.deltas=[];e.transition=[]
    for seed in e.cfg['seeds']:
        seedframe=e.pred[e.pred.Seed==seed].sort_values('Timestamp target')
        assert seedframe['Timestamp target'].is_unique
        for fold,frame in seedframe.groupby('Fold',sort=False):
            common=frame['Irisan LSTM persistence'].to_numpy();same=frame['Irisan LSTM Ridge persistence'].to_numpy()
            months=frame['Timestamp target'].dt.to_period('M').astype(str)
            for method in ALL:
                for domain,mask in [('mandiri',np.isfinite(frame[method]).to_numpy()),('LSTM bersama',common)]:
                    e.metrics.append(metric(frame,method,mask,fold,seed,domain,len(frame)))
                    for mo in months.unique():
                        within=months.eq(mo).to_numpy();e.monthly.append({'Bulan':mo,**metric(frame,method,mask&within,fold,seed,domain,int(within.sum()))})
            for method in ALL+ridge_names:
                e.ridge_comparison.append(metric(frame,method,same,fold,seed,'LSTM Ridge bersama',len(frame)))
            rows=pd.DataFrame([x for x in e.metrics if x['Fold']==fold and x['Seed']==seed and x['Domain']=='LSTM bersama']).set_index('Pipeline')
            for method in METHODS:
                e.ranks.append({'Fold':fold,'Seed':seed,'Pipeline':method,'Rank MAE':float(rows.loc[METHODS,'MAE'].rank(method='min')[method]),
                    'Rank RMSE':float(rows.loc[METHODS,'RMSE'].rank(method='min')[method])})
                e.deltas.append({'Fold':fold,'Seed':seed,'Pipeline':method,'N sama':int(rows.loc[method,'N']),
                    'Delta MAE vs persistence':float(rows.loc[method,'MAE']-rows.loc['Persistence','MAE']),
                    'Delta RMSE vs persistence':float(rows.loc[method,'RMSE']-rows.loc['Persistence','RMSE'])})
            if fold=='Fold_01':
                low=(frame['SO2 asli origin']<=2.14).to_numpy()&common
                after=(frame['Timestamp target']>=pd.Timestamp('2025-06-29 17:30:00')).to_numpy()&common
                for method in ALL+ridge_names:
                    for label,mask in [('origin asli rendah <=2.14',low),('>=7hari setelah transisi',after)]:
                        keep=mask&np.isfinite(frame[method]).to_numpy()
                        e.transition.append({'Bagian':label,'Target median':float(frame.loc[keep,'SO2 target asli'].median()),
                            'Origin asli median':float(frame.loc[keep,'SO2 asli origin'].median()),'Prediksi median':float(frame.loc[keep,method].median()),
                            **metric(frame,method,keep,fold,seed,'bersama untuk diagnosis',int(mask.sum()))})
        for method in ALL:
            for domain,mask in [('mandiri',np.isfinite(seedframe[method]).to_numpy()),('LSTM bersama',seedframe['Irisan LSTM persistence'].to_numpy())]:
                e.pooled.append(metric(seedframe,method,mask,'Gabungan_OOF',seed,domain,len(seedframe)))
        for method in ALL+ridge_names:
            e.ridge_comparison.append(metric(seedframe,method,seedframe['Irisan LSTM Ridge persistence'].to_numpy(),'Gabungan_OOF',seed,'LSTM Ridge bersama',len(seedframe)))
    e.variation=[]
    for label,records in [('fold',e.metrics),('gabungan',e.pooled),('Ridge identik',e.ridge_comparison)]:
        frame=pd.DataFrame(records)
        for (fold,method,domain),g in frame.groupby(['Fold','Pipeline','Domain'],sort=False):
            row={'Jenis':label,'Fold':fold,'Pipeline':method,'Domain':domain,'Seed N':len(g)}
            for col in ['MAE','RMSE','Bias','Cakupan persen','N','Prediksi negatif']:
                row[col+' mean seed']=float(g[col].mean());row[col+' SD seed']=float(g[col].std(ddof=1));row[col+' min seed']=float(g[col].min());row[col+' max seed']=float(g[col].max())
            e.variation.append(row)
    e.sensitivity=[]
    for fold in [f'Fold_{i:02d}' for i in range(1,5)]:
        for seed in e.cfg['seeds']:
            vals=[r['MAE'] for r in e.metrics if r['Fold']==fold and r['Seed']==seed and r['Domain']=='LSTM bersama' and r['Pipeline'] in METHODS]
            e.sensitivity.append({'Fold':fold,'Seed':seed,'MAE maksimum minus minimum pipeline':max(vals)-min(vals),
                'Median SD MAE antar seed pipeline':float(np.median([r['MAE SD seed'] for r in e.variation if r['Jenis']=='fold' and r['Fold']==fold and r['Domain']=='LSTM bersama' and r['Pipeline'] in METHODS]))})
    def time_sum(records,key):
        return sum(r[key] for r in records) if all(r[key] is not None for r in records) else None
    e.timeprep=[{'Fold':f.label,'Histori imputasi seluruh pipeline detik':f.impute_seconds,'Prefix fit imputer scaler detik':time_sum(f.prefix_logs,'Fit imputer dan scaler detik'),
        'Transform KNN detik':time_sum(f.prefix_logs,'Transform KNN detik'),'Jumlah prefix fit':len(f.prefix_logs)} for f in e.folds]
    e.integrity=[]
    for p in e.refs:
        key=str(p.relative_to(OUT.parent));after=hashfile(p);same=after==e.before[key]
        if not same:
            assert key in e.preexisting_diffs and after==e.preexisting_diffs[key]['SHA256 sebelum resume'],key
        e.integrity.append({'File':key,'SHA256 sebelum':e.before[key],'SHA256 sesudah':after,'Identik':same,
            'Status':'Identik sejak awal eksperimen' if same else 'Sudah berbeda saat resume; hash tetap sama sepanjang kelanjutan; penyebab belum diketahui'})
    assert hashfile(RUN/'config_locked.json')==e.config_hash
    first=e.pred[e.pred.Seed==e.cfg['seeds'][0]].copy()
    wide=first[['Fold','Origin','Timestamp target','Index origin','Index target','SO2 target asli','SO2 asli origin','Persistence','Usia donor persistence slot','Irisan LSTM persistence','Irisan LSTM Ridge persistence']+ridge_names].copy()
    for seed in e.cfg['seeds']:
        g=e.pred[e.pred.Seed==seed][['Fold','Timestamp target']+METHODS].rename(columns={m:f'{m}_seed{seed}' for m in METHODS})
        wide=wide.merge(g,on=['Fold','Timestamp target'],validate='one_to_one')
    wide.to_csv(RUN/'prediksi_oof_wide.csv',index=False)
    legacy={'ridge','ridge_scaler','so2_lags','missing_indicators','train_origins','validation_targets','validation_imputer',
        'minimum_common_train_samples','minimum_distinct_train_targets','adequacy_rule','adequacy_adjustment','refit'}
    configrows=[{'Parameter':k,'Lingkup':'Konteks Ridge tahap6; bukan spesifikasi operasional LSTM' if k in legacy else 'LSTM atau aturan bersama yang dipertahankan',
        'Nilai':str(v)} for k,v in e.cfg.items() if k not in ['initial_hashes','initial_input_hashes']]
    dump(RUN/'klarifikasi_lingkup_konfigurasi.json',{'locked_config_sha256':e.config_hash,
        'catatan':'Config terkunci menyalin metadata Ridge untuk konteks. Field legacy berikut bukan jumlah kanal/indikator atau aturan fit LSTM. Konfigurasi terkunci dan pelaksanaan tidak diubah.',
        'legacy_ridge_only':sorted(legacy),'LSTM_numeric_channels':11,'LSTM_missing_indicators':11,'LSTM_total_channels':22,
        'LSTM_scaler_scope':'inner train common unique sequence input slots','LSTM_targets':'original only, inner chronological split, standardized for loss then inverse for metrics'})
    definitions=[{'Istilah':k,'Definisi':v} for k,v in [
        ('Wilayah','Development train+validasi lama sudah dipakai eksperimen; validasi pengembangan, bukan holdout independen. Test akhir terkunci.'),
        ('Sekuens','48 slot berurutan origin−47..origin, SO2(u),10kovariat(u−1),11indikator missing sebelum imputasi. Tidak kompres waktu, tidak gandakan lag tabular Ridge.'),
        ('Inner split','15% akhir slot train fold untuk early stopping, floor85% awal inner train. Target yang melintasi batas tidak dilatih. Konteks sebelumnya tersedia boleh dipakai.'),
        ('Imputasi kausal','Kalibrasi90hari dan prefix30hari hanya inner train. Median/scalerKNN/donorKNN beku sesudah inner train, termasuk evaluasi outer. Ffill online memakai original terdahulu max4slot tanpa fallback.'),
        ('Scaler','11fitur numerik StandardScaler pada slot inputunik sekuens innertrain bersama;11indikator tetap0/1. Targetmean danSDpopulasi dari targetasli innertrain bersama; inverse sebelum skor.'),
        ('Early stopping','Hanya valinternal bersama: patience8 min_delta0 max50epoch, restorebestweights. MSE training targetstandar; tidak melihat error outerfold untuk pilihan epoch/arsitektur.'),
        ('Cakupan persen','N prediksi finite / seluruh target SO2 asli teramati jendela×100. Domainbersama membatasi timestamp tanpa mengganti denominator. Tidak mengisi target.'),
        ('Metrik','e=prediksi−asli; MAE=Σ|e|/N; RMSE=√(Σe²/N); bias=Σe/N. Gabungan dari errorOOF unik perseed, bukan rata-rataRMSE fold.'),
        ('Variasi','SD sampel antar3seed ddof1; bukan confidenceinterval atau bukti signifikansi. Seed bukan3replikasi waktu independen.'),
        ('Ridge','Skor dihitungulang prediksi lama pada irisan semua3LSTM+3Ridge+persistence. Inputsekuens vs lagtabular, fiturindikator, targetscaling/optimisasi dan innerholdback berbeda; bukan efekarsitektur murni.'),
        ('Persistence','Pengamatan SO2 asli terakhir pada atau sebelumorigin, tanpa imputasi/batasumur; umur slot disimpan. Targetvalidasi tetap asli.'),
        ('Kualitas','Nol,lonjakan,konstan,ekstrem dan offgrid auditdipertahankan. Tidak clipping prediksi, negatif dihitung. Latensi aktual/zonawaktu/penyebabtransisi belum dikonfirmasi.'),
        ('Transisi','Tanggal22Juni2025 17:30 dipilih sebelumnya melalui penurunanSO2 asli terbesar fold1; jendela±72jam. Bandgrafik minmax3seed, bukanconfidenceinterval.'),
        ('Kehilangan irisan','Layakfinite mandiri tetapi tidak masuk semua3pipeline; distribusi target dan bulan dilaporkan. Jendela48slot memperluas dampak unresolvedgap relatif lagtabular.'),
    ]]
    e.tables={'Ringkasan':e.pooled,'Metrik_fold_seed':e.metrics,'Variasi_seed':e.variation,'Ridge_timestamp_identik':e.ridge_comparison,
        'Peringkat':e.ranks,'Sensitivitas_seed_metode':e.sensitivity,'Vs_persistence':e.deltas,'Bulanan':e.monthly,'Diagnosis_transisi':e.transition,
        'Rencana_fold':e.plan,'Gate_pra_training':[f.gate for f in e.folds],'Alur_rinci':e.accounting,'Fold1_sekuens_terputus':e.fold1_sequence_breaks,'Alur_sampel':[r for f in e.folds for r in f.flow],
        'Target_kalender':e.calendar_distribution,'Target_kuantil':e.target_quantiles,'Target_distribusi':[r for f in e.folds for r in f.distributions],'Target_bulanan':[r for f in e.folds for r in f.monthly_dist],
        'Keluar_irisan':[r for f in e.folds for r in f.losses],
        'Training_epoch':e.history,'Training_ringkas':e.training,'Waktu_preprocessing':e.timeprep,
        'Scaler':[r for f in e.folds for r in f.scaler_records],'Prefix_imputer':[{'Fold':f.label,**r} for f in e.folds for r in f.prefix_logs],
        'Prediksi_semua_seed':wide.to_dict('records'),'Konfigurasi':configrows,'Definisi':definitions,'Integritas_SHA256':e.integrity}
    dump(OUT/'tabel_lstm_preprocessing.json',e.tables)
    dump(RUN/'checks.json',{'test_loaded':False,'test_evaluated':False,'fit_count':e.fit_count,'config_sha256':e.config_hash,
        'old_hashes_unchanged':sum(r['Identik'] for r in e.integrity),'old_hashes_preexisting_difference':sum(not r['Identik'] for r in e.integrity),'oof_targets_per_seed':len(first),'common_lstm_per_seed':int(first['Irisan LSTM persistence'].sum()),
        'common_ridge_per_seed':int(first['Irisan LSTM Ridge persistence'].sum())})
    plots(e)
    from write_lstm_stage7 import report
    report(e)

def plots(e):
    summary=pd.DataFrame([r for r in e.variation if r['Jenis']=='fold' and r['Domain']=='LSTM bersama'])
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for method in ALL:
        g=summary[summary.Pipeline==method]
        axes[0].errorbar(g.Fold,g['MAE mean seed'],yerr=g['MAE SD seed'],marker='o',capsize=4,label=method)
        own=pd.DataFrame([r for r in e.metrics if r['Pipeline']==method and r['Domain']=='mandiri']).groupby('Fold',sort=False)['Cakupan persen'].mean()
        axes[1].plot(own.index,own.values,marker='o',label=method)
    axes[0].set(title='MAE bersama: mean ± SD tiga seed',ylabel='µg/m³');axes[1].set(title='Cakupan mandiri / target asli',ylabel='%');axes[1].set_ylim(0,105)
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(FIG/'mae_cakupan_fold.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(15,5),layout='constrained');t=pd.Timestamp('2025-06-22 17:30:00')
    first=e.pred[e.pred.Fold=='Fold_01'];base=first[first.Seed==e.cfg['seeds'][0]].set_index('Timestamp target')
    grid=pd.date_range(t-pd.Timedelta(hours=72),t+pd.Timedelta(hours=72),freq='30min');base=base.reindex(grid)
    for col in ['SO2 target asli','SO2 asli origin','Persistence','Ridge_P2_knn_k10']:ax.plot(base.index,base[col],label=col,lw=1.1)
    for method in METHODS:
        z=first.pivot(index='Timestamp target',columns='Seed',values=method).reindex(grid)
        line=ax.plot(z.index,z.mean(axis=1),label=method+' mean3seed')[0];ax.fill_between(z.index,z.min(axis=1),z.max(axis=1),color=line.get_color(),alpha=.12)
    ax.axvline(t,color='black',ls='--');ax.legend(ncol=3,fontsize=8);ax.set(title='Fold1 — transisi22Juni2025, tanpa clipping; band=min–max seed',ylabel='µg/m³');ax.grid(alpha=.2)
    fig.savefig(FIG/'transisi_fold1.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
    for ax,fold in zip(axes,[f'Fold_{i:02d}' for i in range(1,5)]):
        for method in ALL:
            g=pd.DataFrame([r for r in e.metrics if r['Fold']==fold and r['Pipeline']==method and r['Domain']=='LSTM bersama']);ax.plot(g.Seed,g.MAE,marker='o',label=method)
        ax.set(title=fold,xlabel='Seed',ylabel='MAE µg/m³');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7);fig.savefig(FIG/'mae_antar_seed.png',dpi=150);plt.close(fig)
