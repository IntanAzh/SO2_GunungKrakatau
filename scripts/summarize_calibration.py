import numpy as np
import pandas as pd
from calibration_sensitivity import ROOT,OUT,RUN,FIG,DAYS,METHODS,ALL,score,plt
from audit_quality import hashfile
from pilot_way_urang import dump
from write_pilot_report import table

def summarize(e):
    frames=[];e.metrics=[];e.month_metrics=[];e.deltas=[];e.ranks=[]
    for number in range(1,5):
        label=f'Fold_{number:02d}';cases=[c for c in e.cases if c.fold==label];frame=cases[0].pred.copy()
        for c in cases[1:]:frame=frame.merge(c.pred[['Index target',c.method]],on='Index target',validate='one_to_one')
        origins=frame['Index origin'].to_numpy();last=np.maximum.accumulate(np.where(np.isfinite(e.y[:cases[0].b-1]),np.arange(cases[0].b-1),-1))
        frame['Persistence']=e.y[last[origins]];frame['Usia donor slot']=origins-last[origins]
        frame['Evaluasi bersama']=np.isfinite(frame[ALL]).all(axis=1);frames.append(frame)
        months=frame['Timestamp target'].dt.to_period('M').astype(str)
        for method in ALL:
            for domain,keep in [('mandiri',np.isfinite(frame[method]).to_numpy()),('bersama',frame['Evaluasi bersama'].to_numpy())]:
                e.metrics.append(score(frame,method,keep,label,domain,len(frame)))
                for mo in months.unique():
                    within=months.eq(mo).to_numpy();e.month_metrics.append({'Bulan target':mo,**score(frame,method,keep&within,label,domain,int(within.sum()))})
        common=pd.DataFrame([r for r in e.metrics if r['Fold']==label and r['Domain']=='bersama']).set_index('Konfigurasi')
        for method in METHODS:
            e.deltas.append({'Fold':label,'Konfigurasi':method,'N sama':int(common.loc[method,'N']),
                'Delta MAE vs persistence':float(common.loc[method,'MAE']-common.loc['Persistence','MAE']),
                'Delta RMSE vs persistence':float(common.loc[method,'RMSE']-common.loc['Persistence','RMSE']),
                'Delta Bias vs persistence':float(common.loc[method,'Bias']-common.loc['Persistence','Bias'])})
            e.ranks.append({'Fold':label,'Konfigurasi':method,'Rank MAE':float(common.loc[METHODS,'MAE'].rank(method='min')[method]),
                'Rank RMSE':float(common.loc[METHODS,'RMSE'].rank(method='min')[method])})
    e.pred=pd.concat(frames,ignore_index=True);assert e.pred['Timestamp target'].is_unique and e.pred['Timestamp target'].is_monotonic_increasing
    e.pred.to_csv(RUN/'prediksi_evaluasi_semua_konfigurasi.csv',index=False)
    e.pred[['Fold','Origin','Timestamp target','Index origin','Index target','Evaluasi bersama']].to_csv(RUN/'timestamp_evaluasi_bersama.csv',index=False)
    e.pooled=[];e.variation=[]
    for method in ALL:
        for domain,keep in [('mandiri',np.isfinite(e.pred[method]).to_numpy()),('bersama',e.pred['Evaluasi bersama'].to_numpy())]:
            e.pooled.append(score(e.pred,method,keep,'Gabungan_OOF',domain,len(e.pred)))
            rows=[r for r in e.metrics if r['Konfigurasi']==method and r['Domain']==domain];row={'Konfigurasi':method,'Domain':domain,'Fold N':4}
            for col in ['MAE','RMSE','Bias','Cakupan persen','N']:
                v=np.array([r[col] for r in rows]);row[col+' mean fold']=float(v.mean());row[col+' SD fold']=float(v.std(ddof=1));row[col+' min fold']=float(v.min());row[col+' max fold']=float(v.max())
            e.variation.append(row)
    e.integrity=[]
    for path in e.refs:
        key=str(path.relative_to(ROOT));after=hashfile(path);assert after==e.before[key],key
        e.integrity.append({'File':key,'SHA256 sebelum':e.before[key],'SHA256 sesudah':after,'Identik':True})
    assert hashfile(RUN/'config_locked.json')==e.config_hash
    definitions=[{'Istilah':k,'Definisi':v} for k,v in [
        ('Kalibrasi','Warmup statistik untuk menyediakan prefix median; bukan kalibrasi sensor dan bukan ambang validitas. Semua data asli tetap disimpan. Tidak membuktikan30/60/90hari cukup secara domain.'),
        ('Awal train','Origin>=warmup+48slot; target=origin+1slot harus<awal evaluasi. Warmup+buffer dikeluarkan dari fitting, digunakan sebagai histori/statistik masa lalu. Purge origin terakhirtrain.'),
        ('Imputasi','Median rawprefix eksklusif awalquery; blok train1440slot/30hari. Setelah trainfreeze median seluruhtrain. Histori lama tidak dihitung ulang dengan informasi baru. Median tidak tersedia=>tetapNaN tanpa fallback.'),
        ('Fitur dan scaler','7lagSO2:0/1/2/6/12/24/48;10kovariatlag1 tanpaCurahHujan;17indikator missingasli. StandardScaler34kolom fit di traineligible konfigurasi/fold. Target tidak diskalakan atau diisi.'),
        ('Cohort','Median mandiri tiapdurasi; tidak irisan P1/KNN tahap6 atau irisantrain antar-durasi. Efek mencakup perubahan komposisi, historiimputasi dan scalertrain, bukan imputasi terisolasi.'),
        ('Alasan','Ledger perorigin mempertahankan semua slot. Alasanutama eksklusif: warmup→buffer→purge→targetkosong→fiturnonfinite→dipakai. Boolean alasan dapat tumpangtindih; hitunganinklusif tidak dijumlahkan.'),
        ('N tersisih','Seluruh origintrain−origin dipakai. Target purge tidak termasuk targettrain dan nilainya kosong di ledger; target0 memerlukan originsebelumseri dan tidak menjadi sampel model.'),
        ('Persen cakupan','100×Nprediksi finite/seluruh targetasli numerik teramati di jendelaevaluasi. Domainbersama memakai irisan3konfigurasi+persistence dengan denominator sama. Bulanan denominator targetasli bulan/fold.'),
        ('Metrik','e=prediksi−asli; MAE=Σ|e|/N, RMSE=√(Σe²/N), bias=Σe/N; satuan µg/m³. Gabungan dihitung dari errorOOF unik, bukan merata-rataRMSEfold.'),
        ('Variasi','Mean/SDsampel ddof1 antar4fold, deskriptif; fold berkaitan karena trainexpanding. Tidak menjadi bukti signifikansi statistik.'),
        ('Persistence','SO2 original terakhir pada/sebelumorigin, tidak memakai imputasi, donorusia takdibatasi seperti tahap6. Usia slot disimpan. Semua perbandingan skor domainbersama memakai timestampidentik.'),
        ('Distribusi','Targettrain asli, minimum/P1/P25/median/P75/P99/maksimum/mean/SDsampel; bukan nilaiimputasi. Angka2.14 hanya pelacak hasil lama, bukan batas kualitas/bahaya.'),
        ('Integritas','Baseline diambil saat sensitivitasini dimulai. Perbedaan notebook02 sebelum tahap7resume sudah menjadi kondisi awal yang dipertahankan, bukan direparasi. Hashingbyte bukan akses nilai test.'),
        ('Keterbatasan','Development telah digunakan eksperimen sebelumnya. Latensi aktual/zonawaktu/kodekualitas dan penyebab transisi tidak ditebak. Nol,lonjakan,konstan tidak dihapus; prediksi tidak di-clipping.'),
    ]]
    e.tables={'Ringkasan':e.pooled,'Metrik_per_fold':e.metrics,'Variasi_antar_fold':e.variation,'Vs_persistence':e.deltas,'Peringkat':e.ranks,
        'Train_distribusi':[c.dist for c in e.cases],'Train_bulanan':[r for c in e.cases for r in c.monthly],
        'Alur_sampel':[c.flow for c in e.cases],'Alasan_utama':[r for c in e.cases for r in c.reasons],'Alasan_bulanan':[r for c in e.cases for r in c.reason_month],
        'Alasan_tumpang_tindih':[c.overlap for c in e.cases],'Tambahan_train_vs90':e.composition,'Audit90_vs_tahap6':e.legacy,
        'Kesamaan_input_evaluasi':e.input_checks,'Skor_bulanan':e.month_metrics,'Gate_sebelum_fit':[c.gate for c in e.cases],
        'Parameter_Ridge_scaler':[r for c in e.cases for r in c.coefs],'Prefix_median':[r for c in e.cases for r in c.prefix],
        'Waktu_komputasi':[c.timing for c in e.cases],'Prediksi':e.pred.to_dict('records'),
        'Konfigurasi':[{'Parameter':k,'Nilai':('UTC: '+v if k=='created_utc' else str(v))} for k,v in e.cfg.items() if k!='initial_hashes'],
        'Definisi':definitions,'Integritas_SHA256':e.integrity}
    dump(OUT/'tabel_sensitivitas_kalibrasi.json',e.tables)
    dump(RUN/'checks.json',{'fit_count':e.fit_count,'test_loaded':False,'test_evaluated':False,'deep_learning_trained':False,
        'config_sha256':e.config_hash,'oof_target_n':len(e.pred),'common_target_n':int(e.pred['Evaluasi bersama'].sum()),'old_files_identical':len(e.integrity)})
    plots(e);write_report(e)

def plots(e):
    f=pd.DataFrame([r for r in e.metrics if r['Domain']=='bersama']);fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
    for method in ALL:
        g=f[f.Konfigurasi==method]
        for ax,col in zip(axes,['MAE','RMSE','Bias']):ax.plot(g.Fold,g[col],marker='o',label=method);ax.set(title=col+' timestamp bersama',ylabel='µg/m³');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.savefig(FIG/'skor_antar_fold.png',dpi=150);plt.close(fig)
    train=pd.DataFrame([c.dist for c in e.cases]);fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for days in DAYS:
        g=train[train['Kalibrasi hari']==days];axes[0].plot(g.Fold,g.N,marker='o',label=f'{days}hari')
        axes[1].errorbar(g.Fold,g.Median,yerr=[g.Median-g.Minimum,g.Maksimum-g.Median],marker='o',capsize=3,label=f'{days}hari')
    axes[0].set(title='Jumlah train yang benar-benar dipakai',ylabel='N target asli');axes[1].set(title='Median dan rentang target train',ylabel='µg/m³')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(FIG/'komposisi_train.png',dpi=150);plt.close(fig)
    t=pd.Timestamp('2025-06-22 17:30:00');g=e.pred[e.pred.Fold=='Fold_01'].set_index('Timestamp target').reindex(pd.date_range(t-pd.Timedelta(hours=72),t+pd.Timedelta(hours=72),freq='30min'))
    fig,ax=plt.subplots(figsize=(15,4),layout='constrained')
    for col in ['SO2 target asli']+ALL:ax.plot(g.index,g[col],label=col,lw=1)
    ax.axvline(t,color='black',ls='--');ax.set(title='Fold1: transisi terpilih pada diagnosis sebelumnya, tanpa mengubah fold',ylabel='µg/m³');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2)
    fig.savefig(FIG/'transisi_fold1.png',dpi=150);plt.close(fig)

def write_report(e):
    common=[r for r in e.metrics if r['Domain']=='bersama'];pooled=[r for r in e.pooled if r['Domain']=='bersama']
    lines=['# Sensitivitas periode kalibrasi — median dan Ridge Way Urang','',
    '**Satu metode imputasi dan satu model Ridge.** Tidak melatih deep learning, tidak mengubah empat fold kalender, tidak memuat atau mengevaluasi test akhir. Development set train+validasi lama telah dipakai eksperimen sebelumnya, sehingga hasil merupakan validasi pengembangan.','',
    '## Fungsi kalibrasi sebelum eksperimen','',
    'Implementasi `Forecast.impute_history` sebelumnya memakai90hari pertama sebagai **warmup statistik**, bukan kalibrasi alat. Data awal itu tetap mentah dan menyediakan prefix median. Imputasi histori baru dimulai sesudah warmup; median train diperbarui setiap30hari menggunakan rawprefix yang berakhir sebelum awal query. Sampel Ridge dimulai setelah tambahan48slot untuk lag terpanjang. Saat evaluasi, median dibekukan dari seluruh trainfold. Scaler Ridge dan koefisien diestimasi hanya dari train yang layak.','',
    'Pilihan90hari adalah keputusan rancangan terdahulu, bukan hasil dokumentasi sensor atau bukti kebutuhan90hari. Mempersingkatnya mengubah tanggal pertama sampel model, komposisi target serta representasi histori awal train. Penjelasan ini disampaikan sebelum model sensitivitas dijalankan.','',
    '## Rancangan terkunci','',
    'Kalibrasi30/60/90hari, buffer48slot, prefixupdate30hari; horizon30menit; SO₂lag0/1/2/6/12/24/48, sepuluh kovariatlag1 tanpa CurahHujan, serta17indikator missing asli. Scaler StandardScaler pada seluruh34fitur, Ridgealpha1, interceptaktif, solverSVD, CPU1thread; target tidak diskalakan. Tidak ada tuning, clipping atau fallback. Konfigurasi dikunci sebelum12fit, SHA256: '+e.config_hash+'.','',
    table(['Fold','Train mulai','Train akhir','Validasi mulai','Validasi akhir'],e.cfg['folds']),'',
    'Test akhir tetap mulai2026-06-19 06:30. Pembacaan grid dibatasi25.645slot development. Data asli, nol, lonjakan, nilai konstan dan penanda kualitas dipertahankan; off-grid tidak dibulatkan. Ketersediaan SO₂di origin dan kovariatlag1 mengikuti asumsi tahap6; latensi aktual dan zona waktu tidak ditebak.','',
    'Median tiapkolom berasal dari rawprefix eksklusif awalquery; bila belum ada nilai, median tetapNaN dan kegagalannya dicatat. Fitur train historis tidak diisi ulang dengan informasi dari akhirtrain. StandardScaler untuk regresi memakai seluruh traineligible yang sudah tersedia sebelum origin evaluasi pertama; tidak menggunakan data evaluasi atau parameter seluruhdevelopment.','',
    '**Train mandiri per durasi:** uji ini tidak memaksakan irisan train30/60/90hari, karena perubahan sampel adalah bagian tujuan. Tidak pula memakai irisan ffill/KNN tahap6. Karena itu C90baru dapat berbeda dari median tahap6 meskipun matriks fitur90hari identik. Audit perbedaannya:','',
    table(['Fold','Train P0 tahap6 irisan3','C90 median mandiri','Tambahan karena tidak memakai irisan P1 KNN','Minimum','Median','Maksimum'],e.legacy),'',
    'Tanggal evaluasi tetap, tetapi tahap6 membatasi skor utamanya pada irisan median/ffill/KNN. Uji median saja ini mengevaluasi semua targetasli pada tanggal yang sama dan kemudian mengambil irisan30/60/90+persistence. Karena cakupan dan cohorttrain berbeda, skor C90baru tidak boleh dibandingkan langsung dengan skor utama lama sebagai efek perubahan kalibrasi. Semua perbandingan utama di sini berada pada timestampidentik antarketiga durasi.','',
    '## Sampel train dan alasan tersisih','',
    table(['Fold','Kalibrasi hari','Slot origin train','Warmup dikeluarkan','Buffer lag48','Purge batas','Kandidat setelah warmup buffer purge','Target kosong kandidat','Fitur nonfinite kandidat target teramati','Train dipakai','Jumlah tersisih'],[c.flow for c in e.cases]),'',
    'Ledger CSV percase menyimpan **semua origin**, bukan menghapus baris sumber. Alasanutama eksklusif berurutan: warmup, buffer, purge, targetkosong, fiturnonfinite, digunakan; jumlahnya tepat totalorigin train. Boolean alasan dan hitungan tumpang-tindih disimpan terpisah dan tidak dijumlahkan sebagai kategori independen. Hanya34fitur dan targetasli yang diuji, tidak ada dropna pada kolom lain. Nilai target yang melintasi batas tidak masuk statistik train.','',
    table(['Fold','Kalibrasi hari','Target awal','Target akhir','N','Minimum','P1','P25','Median','P75','P99','Maksimum','Mean','SD'],[c.dist for c in e.cases]),'',
    'Distribusi bulanan dan distribusi target menurut alasan tersisih ada di workbook. Target selalu nilai asli numerik yang teramati, bukan hasil imputasi atau pernyataan validitas sensor. Angka2,14µg/m³ hanya pelacak hasil lama, bukan ambang kualitas/bahaya.','',
    'Tambahan sampel dibanding train90hari:','',table(['Fold','Kalibrasi hari','N','Minimum','Median','Maksimum','SO2 lebih2.14'],e.composition),'',
    '## Performa pada timestamp identik','',
    table(['Fold','Konfigurasi','N','Denominator target asli','Cakupan persen','MAE','RMSE','Bias','Prediksi negatif'],common),'',
    'e=prediksi−pengamatan asli. MAE=Σ|e|/N, RMSE=√(Σe²/N), bias=Σe/N dalamµg/m³. Cakupan(%)=Nprediksi finite / seluruh targetasli teramati jendela×100. Domainbersama memakai irisan3durasi+persistence dengan denominator sama. Cakupan mandiri juga disimpan terhadap semua target asli. Targetkosong tidak pernah diisi untuk penilaian.','',
    table(['Fold','Konfigurasi','N','Cakupan persen','MAE','RMSE','Bias'],[r for r in e.metrics if r['Domain']=='mandiri']),'',
    'Persistence memakai pengamatan SO₂asli terakhir yang tersedia pada atau sebelumorigin, bukan hasil imputasi; donor boleh lebih lama tanpa batasumur seperti tahap6. Usia donor dicatat perprediksi. Semua perbandingan bersama memiliki digesttimestamp identik.','',
    table(['Fold','Konfigurasi','Delta MAE vs persistence','Delta RMSE vs persistence','Delta Bias vs persistence'],e.deltas),'',
    'Delta negatif berarti lebih rendah daripada persistence, tetapi bias harus dibaca kedekatannya terhadapnol.','',
    '## Gabungan dan kestabilan','',table(['Konfigurasi','N','Cakupan persen','MAE','RMSE','Bias'],pooled),'',
    'Gabungan dihitung ulang dari error semua prediksiOOF unik, bukan rata-rataRMSEfold. Mean danSDsampel antar4fold (ddof1) disajikan terpisah:','',
    table(['Konfigurasi','MAE mean fold','MAE SD fold','RMSE mean fold','RMSE SD fold','Bias mean fold'],[r for r in e.variation if r['Domain']=='bersama']),'',
    table(['Fold','Konfigurasi','Rank MAE','Rank RMSE'],e.ranks),'',
    'Tidak mengklaim signifikansi statistik dari selisih rata-rata. Fold saling terkait karena trainexpanding dan development telah dipakai sebelumnya. Skor perbulan tersimpan untuk memeriksa perubahan performa di dalam jendela.','',
    '## Apa yang sebenarnya berubah','',table(['Fold','Kalibrasi hari','Fitur evaluasi belum scaling identik antar kalibrasi','Selisih maksimum finite'],e.input_checks),'',
    'Jika seluruh fitur evaluasi sebelum scaling identik, hasil itu menunjukkan bahwa jadwal prefix30hari yang sama sesudah hari90 dan evaluasi mulai sesudah hari120 menghasilkan masukan yang sama pada saat evaluasi. Namun scaler regresi, koefisien, distribusi train dan histori imputasi sebelum hari90 dapat berbeda. Hasil bukan efek median saja, bukan kalibrasi alat, dan bukan bukti sebab fisik perubahan konsentrasi.','',
    '![Komposisi train](figures/sensitivitas_kalibrasi/komposisi_train.png)','',
    '![Skor antar-fold](figures/sensitivitas_kalibrasi/skor_antar_fold.png)','',
    '![Transisi fold1](figures/sensitivitas_kalibrasi/transisi_fold1.png)','',
    '## Penilaian terbatas','']
    for method in METHODS:
        rows=[r for r in e.deltas if r['Konfigurasi']==method];wins=sum(r['Delta MAE vs persistence']<0 and r['Delta RMSE vs persistence']<0 for r in rows)
        lines.append(f'- {method}: MAE danRMSE sekaligus lebih rendah dari persistence pada {wins}/4fold; baca bersama bias, cakupan dan keterwakilan distribusi train.')
    lines+=['',
    'Temuan aktual: C60 memiliki MAE/RMSE gabungan terendah di antara ketiga durasi, namun keuntungan itu terutama dari fold 1. Peringkat per periode berubah: C60 terendah pada fold 1, C90 pada fold 2, C30 pada fold 3 dan 4. Tidak ada konfigurasi yang mengungguli persistence pada MAE dan RMSE sekaligus di satu fold pun. Pada fold 3/4, MAE Ridge bisa lebih rendah tetapi RMSE masih lebih tinggi daripada persistence.','',
    'Pada fold 1, warmup 30 hari menambah train menjadi 3.539 target (rentang 1,24–80,71 µg/m³), termasuk 707 target ≤2,14 sebagai pelacak rezim rendah; C60 menggunakan 2.712 target (43,31–80,71), C90 menggunakan 1.385 target (43,53–47,43). Meski train C30 lebih beragam, MAE 17,245739 dan RMSE 40,229392 masih jauh di atas persistence (0,430562 dan 0,885256). Jadi cakupan rentang train lebih luas tidak otomatis memperbaiki generalisasi. Konsentrasi itu tidak dihapus atau dinyatakan salah.','',
    'Semua durasi memiliki 100% cakupan terhadap 19.797 target asli teramati, dan fitur evaluasi sebelum scaling identik (selisih maksimum 0). Oleh karena itu perbedaan skor antar-durasi di sini bukan akibat titik evaluasi atau nilai median evaluasi yang berbeda. Perubahan cohort, histori imputasi awal train dan estimasi scaler/Ridge tetap terlibat. C30 menghasilkan 1.321 prediksi negatif gabungan, C60 sebanyak 98 dan C90 sebanyak 116, tanpa clipping; pola keluaran ini perlu ditelusuri sebelum memilih warmup.','',
    'Tidak memilih konfigurasi hanya dari rerata error terendah. Warmup pendek dapat membuka periode train lebih beragam sekaligus mengubah representasi missing; warmup panjang dapat menghasilkan cohort sempit dan menyingkirkan rezim yang penting. Semua hasil perlu dibaca perperiode dan metrik dengan cakupan yang sepadan. Tidak membuka test untuk memilih durasi dan tidak melatih arsitektur lain.','',
    '## Reproduksibilitas dan integritas','',
    table(['Fold','Kalibrasi hari','Preprocessing detik','Scaler fit detik','Ridge fit detik','Predict detik'],[c.timing for c in e.cases]),'',
    f'{len(e.integrity)} sumber/artefak lama identikSHA256 terhadap baseline saat uji ini dimulai, termasuk seluruh hasil LSTM dan checkpointnya. Perbedaan notebook02 yang telah dicatat pada tahap7 menjadi kondisi awal dan tidak diperbaiki. Tidak ada filelama ditimpa. Hashingbyte untuk integritas tidak membaca nilai test secara semantik.','',
    'Notebook: `notebooks/08_sensitivitas_kalibrasi_ridge.ipynb`. Konfigurasi, gate pra-fit, input/fitur, timestamptrain, ledger alasan, parameter Ridge/scaler dan prediksi tersimpan di `outputs/sensitivitas_kalibrasi/`. Seluruh tabel pada `hasil_sensitivitas_kalibrasi.xlsx`; prediksi semua konfigurasi tersedia sebagaiCSV; grafik di `figures/sensitivitas_kalibrasi/`.','']
    from polish_lstm_report import polish
    text=polish('\n'.join(lines))
    corrections={'identikSHA256':'identik SHA256','notebook02':'notebook 02','tahap7':'tahap 7','tahap6':'tahap 6',
        'filelama':'file lama','Hashingbyte':'Hashing byte','timestamptrain':'timestamp train','sebagaiCSV':'sebagai CSV',
        'seluruhdevelopment':'seluruh development','rawprefix':'raw prefix','awalquery':'awal query','akhirtrain':'akhir train',
        'traineligible':'train eligible','cohorttrain':'cohort train','C90baru':'C90 baru','CPU1thread':'CPU 1 thread',
        'Ridgealpha1':'Ridge alpha 1','interceptaktif':'intercept aktif','solverSVD':'solver SVD','fiturnonfinite':'fitur nonfinite',
        'sebelumorigin':'sebelum origin','perprediksi':'per prediksi','terhadapnol':'terhadap nol','prediksiOOF':'prediksi OOF',
        'trainexpanding':'train expanding','slot/30hari':'slot/30 hari','prefixupdate30':'pembaruan prefix 30',
        'konfigurasi30/60/90hari':'konfigurasi 30/60/90 hari','train30/60/90hari':'train 30/60/90 hari',
        'train90hari':'train 90 hari','fitur90hari':'fitur 90 hari','mediantrain':'median train'}
    for old,new in corrections.items():text=text.replace(old,new)
    (OUT/'ringkasan_sensitivitas_kalibrasi.md').write_text(text,encoding='utf-8')
