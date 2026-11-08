import pandas as pd
from write_pilot_report import table
from lstm_stage7 import OUT,RUN,METHODS,ALL

def report(e):
    gates=[f.gate for f in e.folds];common=[r for r in e.metrics if r['Domain']=='LSTM bersama']
    summaries=[r for r in e.variation if r['Jenis']=='fold' and r['Domain']=='LSTM bersama']
    pooled=[r for r in e.pooled if r['Domain']=='LSTM bersama']
    rank=pd.DataFrame(e.ranks);delta=pd.DataFrame(e.deltas)
    wins=[]
    for method in METHODS:
        for fold in rank.Fold.unique():
            r=rank[(rank.Pipeline==method)&(rank.Fold==fold)];d=delta[(delta.Pipeline==method)&(delta.Fold==fold)]
            wins.append({'Fold':fold,'Pipeline':method,'Rank MAE seed11/23/47':', '.join(str(int(x)) for x in r['Rank MAE']),
                'Rank RMSE seed11/23/47':', '.join(str(int(x)) for x in r['Rank RMSE']),
                'Seed unggul MAE dan RMSE vs persistence':int(((d['Delta MAE vs persistence']<0)&(d['Delta RMSE vs persistence']<0)).sum())})
    full_target=[r for f in e.folds for r in f.distributions if r['Domain']=='bersama digunakan']
    comp=[r for r in e.ridge_comparison if r['Fold']=='Gabungan_OOF']
    fold_comparison=[]
    for (fold,pipeline),g in pd.DataFrame(e.ridge_comparison).groupby(['Fold','Pipeline'],sort=False):
        if fold=='Gabungan_OOF':continue
        fold_comparison.append({'Fold':fold,'Pipeline':pipeline,'N timestamp sama':int(g.N.iloc[0]),
            'MAE mean seed':float(g.MAE.mean()),'RMSE mean seed':float(g.RMSE.mean()),'Bias mean seed':float(g.Bias.mean())})
    env=e.cfg['environment']
    lines=['# Tahap 7 — verifikasi preprocessing dengan satu LSTM Way Urang','',
    '**Validasi pengembangan**, bukan pengujian independen: development train+validasi lama sudah digunakan dalam eksperimen sebelumnya. Semua empat fold tahap6 termasuk fold pertama dipertahankan. Test akhir mulai2026-06-19 06:30 tidak dimuat, ditransformasi atau dievaluasi. Tidak ada pencarian arsitektur/parameter terbaik. Way Urang masih pilot sementara.','',
    '## Lingkungan','',
    'Environment lama `.venv` menggunakan Python3.15.2 tanpa framework. Dibuat `.venv-lstm` terpisah dengan Python3.12.14, tanpa mengubah environment lama atau dependensinya. PyTorchCPU resmi dipasang setelah pemeriksaan kompatibilitas; environment Python3.12 didukung pada Windows menurut [dokumentasi PyTorch](https://pytorch.org/get-started/locally/).','',
    f"Python: {env['python']}. Executable: `{env['executable']}`. Platform: {env['platform']}. CPU: {env['processor']}; logicalCPU={env['logical_cpus']}. Perangkat pelatihan **CPU satu thread**, deterministic algorithms=True; CUDA tersedia={env['cuda_available']}, CUDA build={env['torch_cuda_build']}. GPU tidak digunakan. Pemeriksaan Win32/CIM tidak diberi akses; identitas CPU memakai informasi proses/OS, bukan inventaris hardware penuh.", '',
    table(['Dependensi','Versi'],[{'Dependensi':k,'Versi':v} for k,v in env['versions'].items()]),'',
    'Lock dependensi lengkap ada pada `requirements-lstm-lock.txt` di akar proyek; torch berasal dari indexCPU resmi `https://download.pytorch.org/whl/cpu`. Kernel `so2-lstm` terpasang pada prefix environment baru. Tidak ada lingkungan sebelumnya dihapus.','',
    '## Rancangan dikunci sebelum training','',
    'SHA256 konfigurasi: '+e.config_hash+'. Konfigurasi dan rencana fold disimpan sebelum preprocessing/model; semua gate dihitung ketika jumlahfitLSTM=0.','',
    'Konfigurasi terkunci juga membawa metadata Ridge tahap6 sebagai konteks. Field lama `so2_lags`, `missing_indicators` (17indikator Ridge), `ridge_scaler`, serta aturan train/validasi Ridge tidak menentukan kanal atau fitting LSTM. Spesifikasi operasional LSTM adalah `sequence_features` (11numerik+11indikator), `feature_scaling`, `target_scaling`, `internal_split` dan `imputer_fit_scope`. Lingkup setiapfield ditandai pada workbook dan `klarifikasi_lingkup_konfigurasi.json`; konfigurasi sebelumfit tidak ditulis ulang dan tidak ada parameter pelaksanaan diubah.','',
    table(['Fold','Inner train awal','Inner train akhir','Validasi internal awal','Validasi internal akhir','Evaluasi awal','Evaluasi akhir'],e.plan),'',
    'Batas internal= floor(0,85×jumlahslot outertrain); 15%akhir kalender untuk early stopping. Seluruh timestamp outerfold sama persis dengan tahap6. Kalibrasi90hari awal innertrain, prefix update30hari, buffer48slot tetap; tidak disesuaikan menurut skor. Train model berakhir pada origin yang targetnya masih berada dalam innertrain. Origin yang targetnya menyeberang batas internal dipindahkan ke bagian internalvalidation, bukan dipakai melatih. Aturan sama pada batas outer. Konteks historis sebelum batas validasi digunakan karena sudah tersedia saat origin.','',
    table(['Fold','Sampel train bersama','Sampel early stopping bersama','Sampel evaluasi bersama','Lulus','LSTM fit count sebelum gate'],gates),'',
    'Gate ≥64sampel train, ≥1sampel internalvalidation dan evaluasi hanya pemeriksaan kelayakan komputasional, bukan jaminan generalisasi. Fold1 tersisa429sampel train; distribusi sempit akibat kalibrasi dan holdback internal tetap dilaporkan.','',
    '## Fitur sekuens dan kausalitas','',
    'Setiap sampel memuat **48langkah berurutan dari origin−47 sampai origin**, termasuk origin, interval30menit. Ini48slot dengan bentang timestamp23,5jam. Horizon target=origin+30menit. Tidak memampatkan waktu dengan menghapus slot kosong.','',
    table(['Urutan','Fitur'],[{'Urutan':i,'Fitur':c} for i,c in enumerate(e.cfg['sequence_features'])]),'',
    'Sebelas kanal numerik adalah SO₂(t) dan sepuluh kovariat(t−1slot); sebelas kanal lain adalah missingasli sebelum imputasi pada setiap langkah. CurahHujan dan seluruh turunannya tidak dipakai. Lag tabular Ridge0/1/2/6/12/24/48 tidak diduplikasi ke tiap langkah. SO₂ terakhir diasumsikan sudah tersedia saat origin, kovariat dengan keterlambatan30menit; latensi aktual belum dikonfirmasi. Tidak menggunakan arah angin linear atau timestamp sebagai angka mentah.','',
    'P0 median prefix innertrain; P1 pengamatanasli sebelumnya maksimum4slot, tanpa fallback, dan hasil imputasi tidak mereset umur donor; P2 KNNk10, distanceweighting, nan-euclidean, scaler prefixtrain, donor ber-SO₂ asli dan minimaldua kovariat asli. Imputer dibuat baru per fold. Setelah innertrain, parameter median/scalerKNN dan bankdonorKNN dibekukan untuk internalvalidation serta evaluasiouter. Ffill boleh memakai original terdahulu yang baru tersedia, sesuai aturan online sebelumnya. Tidak memakai SO₂ masa depan, backwardfill, interpolasi dua arah atau rollingberpusat.','',
    'Scaler input LSTM di-fit pada **slot inputunik** yang tercakup sekuens traininternal bersama: sebelas kanal numerik StandardScaler, indikator tetap0/1. Target train asli distandardisasi memakai mean danSDpopulasi target traininternal bersama, aturan tetap sebelumfit demi skala optimisasi antar-fold. MSE training di satuan targetstandar; semua prediksi di-inverse sebelum MAE/RMSE/bias pada satuan asli. Tidak ada target kosong yang diisi. Histori, nilai asli dan penanda missing tersimpan untuk pelacakan; nol/lonjakan/konstan/ekstrem tidak dibuang.','',
    '## Model dan early stopping','',
    'Satu LSTM32unit satu layer, unidirectional, hiddenstate awalnol tiap sekuens, tanpa dropout; Dense linear1output, total7.201parameter. Adam learningrate0.001, betas0.9/0.999, eps1e−8, weightdecay0; batch64, maksimal50epoch. Earlystopping patience8, min_delta0, checkpoint best internal MSE dipulihkan. Tidak ada scheduler, gradientclipping atau clippingprediksi. Seed11/23/47; bobot awal dan urutan minibatch identik antar-pipeline untuk seed sama. Tidak menggunakan evaluasiouter untuk pemilihan epoch.','',
    table(['Fold','Pipeline','Seed','Epoch dijalankan','Epoch terbaik','Internal MSE standar terbaik','Fit dan early stopping detik','Predict detik'],e.training),'',
    'Riwayat loss setiapepoch dan checkpoint bobot terbaik tersimpan per fold/pipeline/seed. Parameter dibangun ulang dari nol setiapfit, tanpa fine-tune/reuse antar-fold.','',
    f"Sebanyak {sum(r['Epoch dijalankan']==50 for r in e.training)} dari36 kombinasi mencapai batas50 epoch; {sum(r['Epoch terbaik']==50 for r in e.training)} memiliki bobot terbaik pada epoch50. Mencapai batas anggaran ini bukan bukti konvergensi. Jumlah epoch dan patience tidak ditambah untuk mengejar skor evaluasi.", '',
    '## Sampel bersama dan distribusi target','',
    'Seluruh target asli pada bagian kalender, sebelum pengurangan kalibrasi/buffer atau irisan:','',
    table(['Fold','Bagian','Awal target kalender','Akhir target kalender','Slot kalender','N','Minimum','Median','Maksimum'],e.calendar_distribution),'',
    'Target asli yang benar-benar dipakai pada cohort bersama:','',
    table(['Fold','Bagian','Awal target','Akhir target','N','Minimum','Median','Maksimum','SO2 lebih2.14'],full_target),'',
    'Angka2,14 hanya penelusur hasil lama, bukan ambang kualitas/bahaya. Workbook melaporkan targetkandidat sebelum irisan dan distribusi bulanan.','',
    table(['Fold','Bagian','Pipeline','Kandidat origin setelah buffer purge','Target asli teramati','Sekuens finite mandiri','Sampel bersama','Keluar karena irisan'],[r for f in e.folds for r in f.flow]),'',
    '### Asal429sampel train dan668validasi internal fold1','',
    'Trainouter fold1 memuat5.760slot (120hari). Batas85% adalah4.896slot pertama (102hari) untuk innertrain, dan864slot terakhir (18hari) untuk validasiinternal. Dari innertrain, kalibrasi4.320slot dan buffer48slot tetap dipertahankan; origin terakhir yang targetnya masuk validasiinternal dipurge. Dengan demikian hanya527kandidat target tersisa untuk innertrain sebelum pemeriksaan targetasli dan kelengkapan48langkah. Validasiinternal memiliki864kandidat target dan boleh memakai konteks sebelum batas, sehingga tidak mengulang warmup90hari.','',
    table(['Bagian','Pipeline','Slot target kalender sebelum seleksi','Target asli teramati kalender','Kandidat setelah kalibrasi buffer purge','Target asli sebelum pembentukan sekuens','Setelah sekuens48 finite mandiri','Setelah irisan pipeline'],
        [r for r in e.accounting if r['Fold']=='Fold_01' and r['Bagian'] in ['inner_train','inner_validation']]),'',
    'Jumlah429 dan668 adalah irisan timestamp sekuens yang benar-benar layak pada semua3pipeline, bukan pembagian ulang data untuk mengejar skor. Targetkosong tidak menjadi groundtruth; sekuens dengan inputtak-terisi setelah metode terkait tidak dipakai oleh metode itu. Kehilangan karena syaratsekuens dan kehilangan tambahan karena irisan dipisahkan agar tidak dihitung ganda. Persentase85/15 tidak berlaku terhadap sampel akhir setelah kalibrasi dan syaratsekuens. Rentang, median, mean, SD dan jumlah targetasli masing-masing cohort tercantum pada tabeldistribusi di atas dan workbook.','',
    'Secara rinci: train527kandidat−2targetkosong=525targetasli; P0 danP1 memiliki525sekuens finite, P2 hanya429. Validasiinternal864kandidat−4targetkosong=860targetasli; P0/P1 memiliki860sekuens finite, P2 hanya668. Pembatas irisan fold1 adalah P2. Dua langkahinput train danempat langkahinput validasiinternal memiliki10kovariatlag1 tetapkosong; guard minimal2kovariat asli KNN mencegah pengisian tanpa dukungan. Setiaplangkah ini berada dalam48jendela targetasli yang tidak salingtumpangtindih pada kasus ini:96train dan192validasiinternal. Tidak ditambahkan fallback dan tidak diubah guardnya.','',
    table(['Bagian','Timestamp langkah input','Timestamp sumber kovariat lag1','SO2 asli pada langkah','Kovariat asli tersedia','Kovariat tetap kosong','Target asli kandidat terdampak'],e.fold1_sequence_breaks),'',
    'Distribusi targetasli cohortbersama train danvalidasiinternal fold1 (SDsampel ddof1; tidak ada targetimputasi):','',
    table(['Bagian','N','Minimum','P1','P25','Median','P75','P99','Maksimum','Mean','SD'],[r for r in e.target_quantiles if r['Fold']=='Fold_01']),'',
    'Distribusi target sampel yang layak mandiri tetapi tersisih karena irisan:','',
    table(['Fold','Bagian','Pipeline','N','Minimum','Median','Maksimum','SO2 lebih2.14'],[r for f in e.folds for r in f.losses]),'',
    'Tanggal sampel yang tersisih disimpan pada `Fold_*/timestamp_keluar_irisan.csv`; distribusi bulanan dalam workbook. Satu inputtak-terisi memutus kelayakan banyak sekuens48slot. Karena itu cakupan bisa lebih rendah daripada representasi lagtabular tahap6. Waktu tetap utuh; tidak ada medianfallback tersembunyi pada ffill.','',
    '## Hasil pada timestamp bersama LSTM dan persistence','',
    table(['Fold','Pipeline','MAE mean seed','MAE SD seed','RMSE mean seed','RMSE SD seed','Bias mean seed','Cakupan persen mean seed','N mean seed'],summaries),'',
    'Hasil tiapfold dan seed lengkap:','',
    table(['Fold','Pipeline','Seed','N','MAE','RMSE','Bias','Prediksi negatif'],common),'',
    'Metrik gabungan dihitung ulang dari semuaerror pada targetOOF unik, **terpisah tiap seed**:','',
    table(['Pipeline','Seed','N','Denominator target asli','Cakupan persen','MAE','RMSE','Bias','Prediksi negatif'],pooled),'',
    'MAE=Σ|e|/N, RMSE=√(Σe²/N), bias=Σe/N dengan e=prediksi−targetasli. RMSEgabungan bukan rata-rataRMSEfold. SD antar-seed menggunakan ddof1 dan mengukur variasi inisialisasi/urutanbatch, bukan confidenceinterval atau signifikansi statistik. Tidak menggabungkan tiga seed sebagai tiga kali jumlah target waktu independen.','',
    'Cakupan (%)=jumlahprediksi finite / seluruh targetasli teramati pada jendela terkait ×100. Domainbersama memakai denominator targetasli yang sama, sehingga cakupan memperlihatkan kehilangan akibat irisan. Domainmandiri tersedia di workbook untuk setiapfold danseed. Persentase bukan jumlah nilaivalid sensor; “asli teramati” mempertahankan status kualitas semula.','',
    '![MAE dan cakupan](figures/lstm_preprocessing/mae_cakupan_fold.png)','',
    '## Perbandingan Ridge pada timestamp identik','',
    table(['Fold','Pipeline','N timestamp sama','MAE mean seed','RMSE mean seed','Bias mean seed'],fold_comparison),'',
    table(['Pipeline','Seed','N','Cakupan persen','MAE','RMSE','Bias','Prediksi negatif'],comp),'',
    'Workbook juga memuat perbandingan tiapfold. Semua3LSTM+3Ridge+persistence diberi skor pada irisan timestampidentik; targetasli dan digesttimestamp diverifikasi. PrediksiRidge diambil dari tahap6, tidak di-fit ulang. Ridge menggunakan seluruh trainfold setelahkalibrasi/buffer, sedangkan LSTM menyisihkan15%untuk earlystopping. Representasi input juga berbeda: 48langkah×22kanal versus lagtabular34fitur; scalerindikator, targetscaling dan optimisasi ikut berbeda. Karena itu selisih hasil tidak boleh diklaim efek arsitektur murni.','',
    'Pada timestamp identik, kegagalan fold1 lebih berat untuk konfigurasi LSTM ini daripada Ridge. Fold2 juga tidak menunjukkan keunggulan LSTM terhadap persistence atau Ridge. Pada fold3 dan4 terdapat hasil LSTM yang lebih baik, tetapi konsistensi terhadap Ridge bergantung pipeline, seed dan metrik; misalnya fold4 median LSTM memiliki rerataerror lebih rendah, sedangkan ffill/KNN LSTM tidak mengungguli padanannya Ridge pada rerataMAE/RMSE. Tidak ada bukti bahwa LSTM pasti lebih baik. Metrikgabungan didominasi kegagalan fold1; persistence tetap jauh lebih baik secara gabungan.','',
    '## Fold pertama dan transisi22Juni2025','',
    'Transisi2025-06-22 17:30 ditetapkan diagnosis sebelumnya sebagai penurunan30menit terbesar SO₂ asli pada fold1 (45,25→1,98). Grafik tetap±72jam, band min–max3seed, bukan confidenceinterval.','',
    '![Transisi fold1](figures/lstm_preprocessing/transisi_fold1.png)','',
    table(['Pipeline','Seed','Bagian','N','Target median','Origin asli median','Prediksi median','MAE','RMSE','Bias'],e.transition),'',
    'Originasli≤2,14 hanya penelusur konsentrasi rendah. Prediksi di atas target setelahtransisi dan sesudah7hari harus dibaca sebagai kegagalan mengikuti tingkatbaru bila tampak pada tabel; tidak dibuang atau dituning ulang. Tidak menyimpulkan transisi disebabkan gunung atau kerusakan sensor. Jumlah prediksinegatif ada pada semua tabelskor, tanpa clipping.','',
    '## Kestabilan peringkat dan kandidat','',
    table(['Fold','Pipeline','Rank MAE seed11/23/47','Rank RMSE seed11/23/47','Seed unggul MAE dan RMSE vs persistence'],wins),'',
    'Selisih MAE antar-pipeline pada seed sama dibanding variasi antar-seed (deskriptif, tanpa uji signifikansi):','',
    table(['Fold','Seed','MAE maksimum minus minimum pipeline','Median SD MAE antar seed pipeline'],e.sensitivity),'',
    '![Variasi seed](figures/lstm_preprocessing/mae_antar_seed.png)','']
    for method in METHODS:
        g=delta[delta.Pipeline==method];both=((g['Delta MAE vs persistence']<0)&(g['Delta RMSE vs persistence']<0)).sum()
        r=rank[rank.Pipeline==method]
        lines.append(f'- {method}: rankMAE1 pada {int((r["Rank MAE"]==1).sum())}/12kombinasi fold–seed; unggul terhadap persistence pada MAE danRMSE sekaligus di {int(both)}/12kombinasi.')
    low=[r for r in e.transition if r['Bagian']=='origin asli rendah <=2.14' and r['Pipeline'] in METHODS]
    lines += ['',f"Pada originasli rendah fold1, rentang median prediksi LSTM antar-pipeline/seed {min(r['Prediksi median'] for r in low):.3f}–{max(r['Prediksi median'] for r in low):.3f} µg/m³; median target sekitar1,56. Hasil ini harus dinilai bersama MAE danbias sesudahtransisi, bukan keunggulan rata-rata seluruhperiode saja.", '',
    'Peringkat perperiode danseed pada tabel tidak boleh diganti satu peringkat rata-rata. Kecukupan bukti untuk memilih kandidat harus mempertimbangkan konsistensiMAE/RMSE/bias, cakupan, biayaKNN, ketidakpastian latensi/kualitas dan kegagalanrezimfold1. Hasil ini belum cukup untuk menetapkan satu preprocessingfinal sebelum perbandingan algoritma: median tetap kontrol sederhana, ffillL4 danKNNk10 tetap kandidat yang perlu ditinjau dengan batasan cakupan serta sensitivitasseed yang terukur. Tidak mencari arsitektur lain atau membuka test untuk memutuskan pemenang.','',
    'Secara spesifik, KNN memiliki rankMAE1 di seluruh3seed fold2, tetapi masih tertinggal dari persistence di periode itu. Pada fold3 dan4, pipeline dengan rank1 berubah menurutseed. Pada fold1 perbedaanMAE antar-pipeline hanya sekitar0,0002–0,0004 sementaraSDantar-seed sekitar0,1125; urutan numeriknya tidak cukup untuk klaim perbedaan kuat. Median memiliki cakupanmandiri penuh; irisan dengan ffill/KNN mengurangi cakupan bersama. Karena itu tidak cukup kuat memilih satu kandidat tunggal atau menghapus median sebagai pembanding.','',
    '## Waktu dan integritas','',
    table(['Fold','Histori imputasi seluruh pipeline detik','Prefix fit imputer scaler detik','Transform KNN detik'],e.timeprep),'',
    'Waktu pelatihan termasuk evaluasiinternal setiapepoch; predict adalah inferensiouter. Histori dipakai bersama3seed sehingga waktu preprocessing tidak dikalikan ulang sebagai tigaestimasi independen. Ekspor/render tidak termasuk waktu model.','',
    'Catatan interupsi proses: proses awal tidak lagi aktif saat sesi dilanjutkan;28checkpoint lengkap dipertahankan dan8kombinasi tersisa dilanjutkan dengan konfigurasi serta aturan training yang sama. Untuk kombinasi yang belum memiliki checkpointlengkap, proses dimulai dari seedtetap yang sama karena stateoptimizer parsial tidak tersimpan; bukan melanjutkan epochparsial atau memilih hasil berdasarkan skor. Tidak memilih seedterbaik atau melatih ulang modelselesai. Waktu preprocessing4fold dan waktu fit1checkpoint P0seed11fold4 belum tersimpan ke disk saat proses berhenti; dilaporkan kosong/tidaktersedia, bukan0 atau estimasi. Cutoff/count prefix imputasi direkonstruksi dari konfigurasi dan inputtersimpan tanpa fitulang. `resume_log.json` mencatat hash seluruh artefak sebelumresume tetapidentik.','',
    f"{sum(r['Identik'] for r in e.integrity)} dari{len(e.integrity)} sumber/artefaklama memiliki hashSHA256 identik dengan awal eksperimen, termasuk sumberExcel, gridinput, notebooktahap6, diagnosis dan konfigurasi environmentlama. Satu pengecualian sudah terdeteksi sebelumresume: `notebooks/02_audit_kualitas_data.ipynb` berbeda byte dari hashawal; penyebabnya belum diketahui. File itu tidak menjadi inputnumerik model dan tidak ditimpa. Hashnya stabil sepanjang kelanjutan ini, diverifikasi terhadap baseline sebelumresume. Konfigurasi eksperimen terkunci dan seluruh28checkpoint/artefak sebelumresume juga tetapidentik. Tidak boleh menyatakan semua artefak lama identik sejak awal. Hashingfile hanya membaca byte untuk integritas, bukan mengakses nilai test.", '',
    'Keluaran: `hasil_lstm_preprocessing.xlsx`; `lstm_preprocessing/config_locked.json`; `lstm_preprocessing/prediksi_oof_seluruh_seed.csv` dan versiwide; `Fold_*/timestamp_*.csv`, `history_*.json`, checkpoint`.pt`, histori imputasi serta inputsekuens`.npz`. Parameter scaler dan tanggal fitting disimpan untuk verifikasi. Grafik di `figures/lstm_preprocessing/`.','']
    from polish_lstm_report import polish
    (OUT/'ringkasan_lstm_preprocessing.md').write_text(polish('\n'.join(lines)),encoding='utf-8')
