"""Indonesian stage-six report generated from computed results."""
import pandas as pd
from write_pilot_report import table
from backtesting_stage6 import OUT, METHODS

def write_report(p):
    common=[r for r in p.metrics if r['Domain']=='irisan semua metode']
    pooled=[r for r in p.pooled if r['Domain']=='irisan semua metode']
    sections=['# Tahap 6 — expanding-window preprocessing Way Urang','',
    'Ini **validasi pengembangan**: development set adalah train dan validasi lama yang telah digunakan pada eksperimen sebelumnya. Test akhir mulai **2026-06-19 06:30** tetap dikunci; nilainya tidak dimuat atau dievaluasi. Way Urang masih pilot sementara.','',
    '## Rancangan yang dikunci sebelum model','',
    'Train awal 120 hari kalender; sisa development dibagi empat jendela validasi tidak tumpang tindih, berbeda paling banyak satu slot 30 menit. Konfigurasi tersimpan di `backtesting_way_urang/fold_config_locked.json`; SHA256 '+p.config_hash+'. Tidak ada penyesuaian berdasarkan skor.','',
    table(['Fold','Train mulai','Train akhir','Validasi mulai','Validasi akhir','Slot train','Slot validasi'],p.plan),'',
    'Kalibrasi tetap 90 hari pertama train terkait, pembaruan prefix 30 hari dan buffer histori 48 slot. Kecukupan ditetapkan sebelum Ridge: minimal 340 sampel bersama (10 × 34 fitur) dan dua nilai target berbeda. Ini pemeriksaan kecukupan komputasional, bukan jaminan daya statistik. Seluruh gate diperiksa saat jumlah fit Ridge masih nol; seluruhnya lulus tanpa mengubah batas fold.','',
    table(['Fold','Jumlah train bersama','Target unik train bersama','Cukup sampel','Ridge fit count global sebelum gate'],[f.gate for f in p.folds]),'',
    '## Parameter dan kausalitas','',
    'Sama dengan 5B tanpa Curah Hujan: horizon satu slot/30 menit; SO₂ lag 0, 1, 2, 6, 12, 24, 48; sepuluh kovariat lag satu slot: '+', '.join(p.cfg['covariates'])+'. Sebanyak 17 fitur numerik ditambah 17 indikator missing asli. Ridge alpha=1, intercept aktif, solver SVD; scaler fitur diestimasi hanya pada train bersama fold tersebut.','',
    'P0 mengisi dengan median prefix train; P1 memakai pengamatan asli sebelumnya maksimum empat slot, tanpa fallback; P2 KNN k=10, distance weighting, nan-euclidean, scaling prefix train, donor SO₂ asli dan minimal dua kovariat asli. Semua estimator dibuat ulang dari nol pada setiap fold. Kalibrasi dan prefix train hanya menggunakan waktu sebelumnya; parameter imputasi validasi dibekukan pada akhir train. Histori direkonstruksi sebelum pembentukan lag. Tidak ada nilai target masa depan, backward fill, interpolasi dua arah atau clipping. Pengamatan validasi fold terdahulu boleh menjadi data train fold berikut setelah waktunya tersedia.','',
    'Asumsi ketersediaan tetap: SO₂ pada origin sudah tersedia saat prediksi diterbitkan, kovariat sudah tersedia dengan keterlambatan satu slot. Ini belum dikonfirmasi penyedia data. Tidak ada waktu mentah atau arah angin linear sebagai fitur. Nol, lonjakan dan rentetan konstan dipertahankan; penanda missing asli ikut fitur. Off-grid Way Urang yang diaudit sebelumnya tidak memiliki SO₂; tidak dibulatkan.','',
    'Origin train terakhir yang targetnya memasuki validasi dipurge. Target selalu pengamatan SO₂ asli; target kosong tidak diisi. Persistence memakai SO₂ asli terakhir pada atau sebelum origin, termasuk donor lebih lama ketika origin kosong. Usia donor dalam slot disimpan untuk setiap prediksi.','',
    '## Distribusi dan seleksi sampel','',
    table(['Fold','Bagian','Target asli teramati','Minimum SO2','Median SO2','Maksimum SO2','Target lebih 2.14'],p.distributions),'',
    'Angka 2,14 µg/m³ hanya penelusur hasil tahap kelima, bukan batas kualitas atau bahaya. Distribusi bulanan lengkap tersedia dalam workbook. Distribusi train yang benar-benar digunakan setelah kalibrasi, buffer, purge dan irisan:','',
    table(['Fold','Jumlah slot atau sampel','Minimum SO2','Median SO2','Maksimum SO2','Target lebih 2.14'],[f.train_distribution[0] for f in p.folds]),'',
    table(['Fold','Pipeline','Seluruh origin train','Setelah kalibrasi buffer purge','Target train asli','Setelah fitur finite mandiri','Final train bersama'],[r for f in p.folds for r in f.flow]),'',
    'Sampel layak suatu pipeline yang dikeluarkan hanya untuk menyamakan train:','',
    table(['Fold','Pipeline','Layak mandiri','Aktual bersama','Keluar karena irisan','Minimum SO2','Median SO2','Maksimum SO2','Target lebih 2.14'],[r for f in p.folds for r in f.lost]),'',
    'Workbook memisahkan kehilangan karena fitur nonfinite dari kehilangan tambahan akibat irisan, beserta distribusi target dan bulan. Penanda SO₂ historis/kovariat kosong dapat tumpang tindih, sehingga hitungannya tidak boleh dijumlahkan sebagai kategori eksklusif. Irisan dapat menyeleksi periode; kesamaan jumlah train tidak menjamin representasi distribusi penuh.','',
    '## Skor pada timestamp bersama','',
    table(['Fold','Pipeline','Jumlah prediksi','Denominator target asli','Cakupan persen','MAE','RMSE','Bias'],common),'',
    'Bias = prediksi − pengamatan. MAE, RMSE dan bias memakai N prediksi finite pada domain bersangkutan. Cakupan (%) = N / seluruh target asli teramati pada jendela validasi × 100. Domain bersama memakai satu himpunan timestamp identik untuk ketiga pipeline dan persistence. Cakupan mandiri tetap dilaporkan terpisah terhadap denominator target asli yang sama.','',
    table(['Fold','Pipeline','Jumlah prediksi','Cakupan persen','MAE','RMSE'],[r for r in p.metrics if r['Domain']=='seluruh berhasil']),'',
    '### Perubahan terhadap persistence pada timestamp identik','',
    table(['Fold','Pipeline','Delta MAE vs persistence','Delta RMSE vs persistence','Perbaikan MAE persen','Perbaikan RMSE persen'],p.deltas),'',
    'Delta negatif berarti error lebih rendah. Persentase perbaikan = (error persistence − error pipeline) / error persistence × 100; negatif berarti memburuk.','',
    '### Gabungan prediksi validasi unik','',
    table(['Pipeline','Jumlah prediksi','Denominator target asli','Cakupan persen','MAE','RMSE','Bias'],pooled),'',
    'Metrik gabungan dihitung ulang dari semua error OOF pada timestamp unik: MAE=Σ|e|/N, RMSE=√(Σe²/N), bias=Σe/N. RMSE gabungan bukan rata-rata RMSE fold. Rata-rata dan simpangan baku sampel antar empat fold (ddof=1) adalah ringkasan deskriptif terpisah:','',
    table(['Pipeline','MAE rata-rata fold','MAE SD fold','RMSE rata-rata fold','RMSE SD fold','Bias rata-rata fold','Bias SD fold'],[r for r in p.summary if r['Domain']=='irisan semua metode']),'',
    '### Peringkat deskriptif','',
    table(['Fold','Pipeline','Peringkat MAE','Peringkat RMSE'],p.ranks),'',
    '### Per bulan','',
    table(['Bulan target','Pipeline','Jumlah prediksi','Cakupan persen','MAE','RMSE','Bias'],[r for r in p.pooled_month if r['Domain']=='irisan semua metode']),'',
    'Metrik per bulan per fold serta distribusi target bulanan disimpan dalam workbook. Denominator bulanan adalah target asli pada bulan/jendela terkait; bulan yang melintasi batas fold memiliki model train berbeda dan juga ditampilkan secara terpisah.','',
    '## Waktu komputasi','',table(['Fold','Pembentukan histori semua pipeline detik','Fit imputer median scaler KNN detik','Transform KNN detik'],p.preprocess_time),'',
    table(['Fold','Pipeline','Scaler fitur fit detik','Ridge fit detik','Predict detik'],p.timing),'',
    'Waktu dinding diukur pada mesin ini, satu thread; histori gabungan mencakup ketiga pipeline dan penyusunan fitur, bukan waktu tiap pipeline yang dapat dijumlahkan ulang secara bebas. Fit median/scaler/KNN dan transform KNN memiliki pengukuran prefix tersendiri; waktu ekspor/render tidak termasuk waktu model.','',
    '## Kesimpulan terbatas','']
    for method in METHODS:
        ds=[r for r in p.deltas if r['Pipeline']==method]
        wins=sum(r['Delta MAE vs persistence']<0 and r['Delta RMSE vs persistence']<0 for r in ds)
        ranks=[r for r in p.ranks if r['Pipeline']==method]
        sections.append(f'- {method}: mengungguli persistence pada MAE dan RMSE sekaligus di {wins}/4 fold; peringkat MAE antar-fold '+', '.join(str(int(r['Peringkat MAE'])) for r in ranks)+'.')
    first=p.folds[0].train_distribution[0]
    negative={r['Pipeline']:r['Prediksi negatif'] for r in pooled}
    ages=p.pred['Usia donor persistence slot']
    sections += ['',
    f"Fold pertama penting: setelah kalibrasi dan buffer, hanya {first['Jumlah slot atau sampel']} target train bersama tersisa pada rentang {first['Minimum SO2']:.2f}–{first['Maksimum SO2']:.2f} µg/m³. Ini cukup untuk gate komputasional, tetapi tidak mencakup rezim konsentrasi rendah yang kemudian muncul di validasi. Ketiga Ridge mengalami bias positif besar (sekitar 16 µg/m³) dan MAE sekitar 16,5, sedangkan persistence sekitar 0,429. Ini kegagalan generalisasi yang tampak pada konfigurasi tetap, bukan alasan menggeser fold atau menghapus pengamatan.", '',
    'KNN memiliki MAE dan RMSE terendah di antara preprocessing pada fold 1, 2 dan 4; pada fold 3 median memiliki MAE terendah sementara ffill memiliki RMSE terendah. Dengan demikian peringkat tidak sepenuhnya stabil dan kesetaraan hasil pada satu validasi lama tidak berlaku umum. Selisih ffill–KNN pada periode rendah sangat kecil dibanding perubahan error antar-rezim. Ffill dan KNN mengungguli persistence pada kedua metrik di fold 2–4, tetapi gagal pada fold 1; median tidak konsisten pada RMSE. Metrik gabungan didominasi kegagalan fold pertama dan seluruh Ridge tertinggal dari persistence.','',
    'Prediksi negatif pada irisan OOF (tanpa clipping): '+', '.join(f'{k}={v}' for k,v in negative.items())+'. Nilai ini perlu diperiksa sebagai kegagalan keluaran model, bukan diubah diam-diam. Tabel strata SO₂ >2,14 dan ≤2,14 memisahkan skor pada konsentrasi berbeda; tiga fold terakhir tidak memiliki target >2,14 sehingga tidak mendukung klaim evaluasi konsentrasi tinggi.','',
    f"Persistence: donor tersedia pada origin untuk {int((ages==0).sum())}/{len(ages)} target asli validasi; usia donor median {ages.median():.0f} slot, P95 {ages.quantile(.95):.0f}, maksimum {ages.max():.0f} slot (satu slot=30 menit). Donor lebih lama tetap hanya pengamatan asli sebelum origin; aturan sama seperti tahap 5B.", '',
    'Urutan preprocessing harus dibaca per fold dan per metrik; selisih rata-rata tidak membuktikan signifikansi statistik. Distribusi train dan validasi berubah kuat: konsentrasi lebih tinggi tersedia pada validasi pertama, sedangkan tiga jendela berikutnya sempit/rendah. Perubahan skor bertepatan dengan pergeseran distribusi dan seleksi sampel, tetapi bukan bukti penyebab fisik ataupun efek preprocessing yang terisolasi. Tidak boleh menggeneralisasi kemampuan konsentrasi tinggi dari jendela rendah.','',
    'Ffill L4 dan KNN k10 layak diprioritaskan sebagai kandidat untuk verifikasi terkontrol dengan **satu** model deep learning berikutnya; median tetap sebagai kontrol sederhana dan persistence sebagai acuan wajib. Bawa hasil per periode, cakupan dan biaya KNN ke verifikasi tersebut; belum memilih preprocessing final atau algoritma deep learning terbaik. Parameter dan fold tidak ditata ulang untuk mengejar skor. Rancangan keterlambatan pengukuran, arti penanda kualitas dan dokumentasi sensor masih memerlukan konfirmasi penyedia data.','',
    '## Artefak dan integritas','',
    f'{len(p.integrity)} sumber/artefak lama diverifikasi SHA256 identik sebelum–sesudah. Hash konfigurasi fold juga tetap identik. Hash seluruh berkas hanya verifikasi byte, bukan pembacaan nilai test. Daftar lengkap terdapat pada sheet Integritas_SHA256.','',
    '- `backtesting_preprocessing.xlsx`: tabel skor, distribusi, gate, alur sampel, koefisien, prediksi dan integritas.',
    '- `backtesting_way_urang/fold_config_locked.json`: parameter dan kalender terkunci.',
    '- `backtesting_way_urang/Fold_*/`: histori, fitur, timestamp train/evaluasi, prediksi serta gate tiap fold.',
    '- `backtesting_way_urang/prediksi_oof_development.csv`: seluruh target asli validasi dan prediksi; kosong berarti pipeline tidak dapat memberi prediksi.',
    '- `figures/backtesting_way_urang/`: metrik/cakupan fold, distribusi dan ukuran train, MAE bulanan.',
    '', '![Metrik dan cakupan](figures/backtesting_way_urang/metrik_dan_cakupan_fold.png)', '',
    '![Distribusi](figures/backtesting_way_urang/distribusi_dan_train_fold.png)', '',
    '![MAE bulanan](figures/backtesting_way_urang/mae_bulanan_oof.png)','']
    (OUT/'ringkasan_backtesting_preprocessing.md').write_text('\n'.join(sections),encoding='utf-8')
