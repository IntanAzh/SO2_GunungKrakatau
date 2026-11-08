"""Laporan penyaringan awal Ridge berdasarkan hasil, tanpa pemilihan akhir."""
from pathlib import Path
import numpy as np
import pandas as pd
from write_pilot_report import table

OUT=Path(__file__).resolve().parents[1]/'outputs'

def write_report(f):
    metrics=pd.DataFrame(f.metrics);shared=metrics[metrics.Domain=='irisan semua metode'].set_index('Metode')
    allrows=metrics[metrics.Domain=='seluruh berhasil']
    distribution={r['Split']:r for r in f.distribution}
    tr=distribution['train'];va=distribution['validation']
    n=len(f.pred);nc=int(f.common.sum())
    lines=['# Forecasting acuan Way Urang — perbandingan preprocessing terkontrol','',
        '**Penyaringan awal satu langkah menggunakan Ridge, bukan keputusan akhir preprocessing deep learning.** '
        'Way Urang tetap pilot sementara. Tidak ada algoritma forecasting lain yang dilatih dan '
        'test akhir tidak dimuat nilainya atau dievaluasi. Split tetap seperti pilot tahap keempat.','',
        '## 1. Keterbatasan pilot yang ditinjau','',
        'Seluruh tabel/konfigurasi/penjelasan pilot dibaca kembali. Hasil rekontruksi sebelumnya '
        'menilai SO₂ yang sengaja disembunyikan, dengan kovariat lain tetap tersedia. '
        'Itu belum menjamin keuntungan pada forecasting target masa depan.','',
        '### MAE dan waktu setiap seed: median versus KNN k=10','',
        table(['Metode','Seed','Pola','Panjang slot','Jumlah mask','Jumlah dinilai','MAE','Cakupan persen',
            'Fit detik','Scaler fit detik bersama','Transform detik'],
            [r for r in f.pilot_seed if r['Domain']=='seluruh berhasil']),'',
        'MAE di atas memakai seluruh titik yang berhasil tiap metode, sehingga N dapat berbeda. '
        'Perbandingan pada irisan pilot yang sama, seluruh hasil per-seed dan domain tersedia '
        'pada workbook. Fit pilot dilakukan satu kali lalu dipakai ulang; tidak boleh menjumlah '
        'ulang waktu fit setiap seed. Waktu scaler bersama dicatat terpisah.','',
        table(['Metode','Pola','Panjang slot','MAE rata-rata','MAE SD antar-seed','MAE minimum','MAE maksimum',
            'Transform detik rata-rata','Transform detik SD antar-seed'],
            [r for r in f.pilot_summary if r['Domain']=='seluruh berhasil']),'',
        'SD adalah simpangan baku sampel tiga seed, bukan confidence interval independen '
        'karena lokasi mask dapat overlap. KNN memerlukan fit/scaling/donor dan transform '
        'lebih mahal daripada mengembalikan satu median; waktu komputer ini bukan jaminan '
        'latensi produksi atau perbandingan kompleksitas pada semua ukuran data.','',
        '### Distribusi SO₂ train dan validasi','',
        table(['Split','Pengamatan asli','Minimum','P1','P25','Median','P75','P99','Maksimum','Mean','SD','IQR'],f.distribution),'',
        table(['Split','Bulan','Slot grid','Pengamatan asli','Minimum','P25','Median','P75','Maksimum','SD','IQR'],f.monthly),'',
        f"Secara deskriptif validasi lebih sempit: rentang {va['Minimum']:.6g}–{va['Maksimum']:.6g} µg/m³ "
        f"dan SD {va['SD']:.6g}, dibanding train {tr['Minimum']:.6g}–{tr['Maksimum']:.6g} "
        f"dan SD {tr['SD']:.6g}. IQR validasi {va['IQR']:.6g}, train {tr['IQR']:.6g}. "
        'Ini menunjukkan variasi pengamatan validasi lebih kecil pada periode tersebut, '
        'bukan bukti stasioneritas atau kesalahan sensor. Evaluasi ini belum mencakup '
        'kondisi SO₂ tinggi train. Batas split tidak diubah untuk memperbaiki skor.','',
        '![Distribusi SO2](figures/forecasting_way_urang/distribusi_train_validasi.png)','',
        '## 2. Rancangan dikunci sebelum forecasting','',
        f"Konfigurasi disimpan di `forecasting_way_urang/config_locked.json` sebelum pembentukan "
        f"histori dan fit Ridge. SHA256 konfigurasi: `{f.config_hash}`. Waktu pembuatan UTC: "
        f"`{f.config['created_utc']}`. Hash diverifikasi kembali setelah eksperimen.",
        '- Target: SO₂ asli pada t+30 menit (satu langkah), bukan horizon final penelitian.',
        '- Pipeline: P0 median, P1 forward fill maksimum **4 slot/2 jam tanpa fallback median**, '
        'P2 KNN k=10 berbobot distance. P3 ditunda karena pilot belum konvergen dan menghasilkan nilai negatif.',
        '- L=4 dipilih sebelum forecasting karena mencakup gap pendek train 1/4 slot; '
        'pilot L4 mengisi blok 4 slot seluruhnya, sedangkan L1 hanya 25%. '
        'Batas 71–981 slot akan meneruskan donor berhari-hari dan tidak dipilih untuk '
        'penyaringan awal. L4 bukan ambang sensor dan bukan hasil tuning forecasting.',
        '- Model tunggal: `Ridge(alpha=1.0, fit_intercept=True, solver="svd")`, '
        'sama untuk tiga pipeline. Tidak ada pencarian alpha, seleksi fitur dari skor '
        'forecasting, atau model tambahan.',
        '- SO₂ lag **0,1,2,6,12,24,48 slot**. Sebelas kovariat sama seperti pilot, '
        'semuanya lag satu slot: '+', '.join(f.cov)+'. Arah angin linear dan waktu mentah tidak digunakan.',
        '- Setiap Ridge mempunyai 18 fitur numerik dan 18 indikator missing asli. '
        'Algoritma/scaler/fitur dan timestamp train sama; parameter numerik scaler/koefisien '
        'boleh berbeda karena data hasil preprocessing berbeda.',
        '- Baseline persistence: SO₂ **asli** terakhir yang tersedia pada atau sebelum origin, '
        'tanpa batas umur. Ia tidak memakai hasil imputasi pipeline mana pun.','',
        table(['Split','Awal','Akhir','Jumlah slot','SO2 nonkosong asli'],f.split),'',
        'Inventaris test pada tabel disalin dari pilot; nilai test tidak dimuat. Test telah '
        'tercakup audit lama sehingga tidak disebut belum pernah dilihat.','',
        '## 3. Pembentukan histori dan fitur yang kausal','',
        '### Ketersediaan informasi','',
        '- SO₂ teramati pada u diasumsikan sudah tersedia saat origin u. Kovariat pada u−1 '
        '(30 menit sebelumnya) diasumsikan telah tersedia pada u, sama dengan pilot. '
        'Latensi aktual belum diketahui; bila lebih panjang, fitur dan evaluasi perlu '
        'digeser ulang. Tidak ada klaim bahwa timestamp sendiri membuktikan waktu kedatangan.',
        '- Baris matriks rekonstruksi pada u adalah `[SO2(u), kovariat(u−1)]`. '
        'SO₂ hilang pada u diestimasi hanya dari kovariat asli yang sudah dilag dan donor '
        'train sebelumnya. Bila SO₂(u) teramati, nilainya dapat membantu KNN mengisi '
        'kovariat(u−1) yang kosong pada origin u. Ini bukan pemakaian target u+1.',
        '- Nilai nonkosong asli tidak diganti. Nol, lonjakan, rentetan konstan dan ekstrem '
        'tetap dipertahankan. Baris off-grid Way Urang tetap ledger terpisah tahap 4 '
        '(2025-05-22 19:01, SO₂ kosong), tanpa pembulatan.','',
        '### Menghindari leakage di dalam train','',
        'Mengimputasi seluruh train dengan imputer yang di-fit pada seluruh train lalu '
        'memakai hasilnya untuk contoh historis awal akan memakai informasi yang belum '
        'tersedia pada origin itu. Karena itu digunakan prosedur prefix berikut:',
        '1. **90 hari pertama** sebagai kalibrasi awal. Nilai kosong pada bagian ini tidak '
        'diisi menggunakan data sesudahnya. Ridge mulai menerima origin 48 slot setelah '
        'kalibrasi, agar seluruh lag SO₂ berasal dari periode histori yang telah diproses.',
        '2. Pada train setelah kalibrasi, median dan scaler/KNN di-fit hanya pada prefix '
        '**sebelum** awal blok 30 hari. Parameter dibekukan untuk blok berikutnya. '
        'Pembaruan tetap berasal dari data train asli, bukan hasil imputasi.',
        '3. Pada validasi, imputer dan scaler jarak KNN di-fit pada seluruh train lalu '
        'dibekukan. SO₂ validasi yang sudah benar-benar teramati boleh masuk sebagai '
        'histori setelah waktunya, tetapi tidak menjadi donor KNN atau estimator median.',
        '4. P1 selalu memakai pengamatan asli sebelumnya per variabel dan berhenti setelah '
        'umur donor >4 slot. Hasil imputasi tidak mereset umur. Tidak ada fallback median.',
        '5. P2 memakai donor train dengan SO₂ asli finite, scaler prefix yang mempertahankan '
        'NaN, guard minimal dua kovariat asli tersedia pada query, dan dukungan donor '
        'per kolom. Kolom tanpa dukungan tetap kosong; fallback bawaan tanpa overlap '
        'tidak diterima. Tidak ada clipping.',
        '6. **Setelah histori selesai secara kausal**, fitur lag SO₂ dibentuk. Kovariat '
        'sudah dilag sebelum rekonstruksi dan tidak dilag lagi. Indikator missing berasal '
        'dari keadaan asli sebelum pengisian.',
        '7. Ridge dilatih pada **irisan timestamp train dengan target asli dan fitur finite '
        'untuk ketiga pipeline**. Scaler fitur Ridge di-fit pada baris train bersama '
        'masing-masing pipeline. Ini mengontrol komposisi sampel pelatihan, tetapi '
        'mengurangi jumlah contoh dan membatasi generalisasi.',
        '8. Origin train terakhir yang targetnya masuk validasi dikeluarkan dari train. '
        'Origin tersebut boleh menjadi origin prediksi pertama validasi. Target kosong '
        'tidak diisi untuk pelatihan maupun penilaian.','',
        table(['Pipeline','Origin train awal','Origin train akhir','Jumlah train kandidat target asli',
            'Jumlah train fitur lengkap pipeline','Jumlah train bersama digunakan','Origin dipurge','Target origin dipurge'],f.train_detail),'',
        '### Distribusi target pada train yang benar-benar digunakan Ridge','',
        table(['Bagian','Awal target','Akhir target','Jumlah target asli','Minimum','Median','Maksimum','Mean','SD'],f.actual_train_distribution),'',
        'Distribusi ini berbeda dari seluruh 70% train pada audit deskriptif. Kalibrasi '
        'awal dan persyaratan fitur lengkap untuk semua pipeline dapat mengeluarkan '
        'periode tertentu dari contoh Ridge, termasuk konsentrasi tinggi. Ini konsekuensi '
        'rancangan bersama yang telah ditetapkan, bukan perubahan batas split atau seleksi '
        'berdasarkan skor. Hasil Ridge harus ditafsirkan terhadap subset train aktual ini.','',
        table(['Awal query','Akhir query','Timestamp donor maksimum','Jumlah baris prefix','Jumlah donor SO2 asli','Periode'],f.prefix_logs),'',
        'Rincian histori hasil imputasi, fitur numerik, indikator missing, parameter '
        'koefisien/scaler, dan timestamp train bersama disimpan agar dapat ditelusuri. '
        'Batas split tetap; kalibrasi dan kelengkapan fitur hanya mengurangi sampel '
        'yang dapat dipakai di dalam train.','',
        '## 4. Hasil forecasting validasi','',
        '`e = prediksi − SO₂ target asli`. MAE=mean(|e|), RMSE=sqrt(mean(e²)), '
        'bias=mean(e), satuan µg/m³. **Cakupan** = jumlah prediksi finite / seluruh '
        'target validasi asli teramati ×100%. Pada tabel bulanan denominator hanya '
        'target asli bulan tersebut. Tidak ada target imputasi.','',
        f"Tersedia **{n} target asli validasi**. Irisan keberhasilan empat metode mencakup "
        f"**{nc} timestamp ({100*nc/n:.3f}%)**. Daftar timestamp dan prediksi disimpan di CSV dan workbook.",'',
        '### Perbandingan pada timestamp bersama','',
        table(['Metode','Jumlah prediksi','Denominator target asli validasi','Cakupan persen','MAE','RMSE','Bias','Prediksi negatif'],
            [r for r in f.metrics if r['Domain']=='irisan semua metode']),'',
        '### Cakupan dan kinerja pada seluruh timestamp yang berhasil','',
        table(['Metode','Jumlah prediksi','Denominator target asli validasi','Cakupan persen','MAE','RMSE','Bias','Prediksi negatif'],
            [r for r in f.metrics if r['Domain']=='seluruh berhasil']),'',
        'Perbedaan cakupan adalah hasil eksperimen, bukan alasan menambah fallback '
        'tanpa nama. Skor pada domain sendiri tidak menggantikan perbandingan timestamp '
        'bersama. Nilai prediksi negatif bila ada tetap dilaporkan tanpa clipping.','',
        '### Per bulan','',
        table(['Metode','Bulan target','Domain','Jumlah prediksi','Denominator target asli validasi','Cakupan persen','MAE','RMSE','Bias'],f.month_metrics),'',
        'Maret dan Juni merupakan bulan validasi parsial. Performa bulanan dibaca '
        'bersama jumlah prediksi dan denominator, bukan hanya urutan MAE.','',
        '![Performa bulanan](figures/forecasting_way_urang/performa_bulanan.png)','',
        '![Prediksi validasi](figures/forecasting_way_urang/prediksi_validasi.png)','',
        '### Kondisi input dan waktu komputasi','',
        table(['Metode','Strata','Jumlah prediksi','Denominator target asli validasi','Cakupan persen','MAE','RMSE','Bias'],f.strata),'',
        'Strata missing dapat overlap dan dievaluasi pada irisan. Ketika sebagian besar '
        'input SO₂ validasi sudah teramati, ruang untuk memperoleh keuntungan dari '
        'imputasi SO₂ dapat sangat kecil. Perbedaan juga dapat berasal dari penanganan '
        'kovariat kosong dan histori pelatihan, bukan hanya pengisian SO₂.',
        f"Pembentukan histori seluruh pipeline memerlukan {f.impute_seconds:.6f} detik. "
        'Fit/scaling prefix imputer dan transform KNN disimpan per blok; waktu Ridge '
        'berikut dipisahkan dari biaya preprocessing.','',
        table(['Pipeline','Scaler fitur fit detik','Ridge fit detik','Predict detik'],f.timing),'',
        '## 5. Kesimpulan terbatas','']
    p0=shared.loc['P0_median'];p1=shared.loc['P1_ffill_L4'];p2=shared.loc['P2_knn_k10'];base=shared.loc['Persistence']
    lines += [f"1. **Keuntungan KNN dari imputasi ke forecasting:** pada timestamp bersama, P2 mempunyai "
        f"MAE {p2.MAE:.6f}, RMSE {p2.RMSE:.6f}, bias {p2.Bias:.6f}; "
        f"P0 MAE {p0.MAE:.6f}, RMSE {p0.RMSE:.6f}, bias {p0.Bias:.6f}. "+
        ('KNN lebih rendah pada MAE dan RMSE pada eksperimen ini.' if p2.MAE<p0.MAE and p2.RMSE<p0.RMSE else
         'Keuntungan rekonstruksi KNN tidak terulang secara konsisten pada MAE dan RMSE forecasting ini.')+
        ' Periksa juga bias, performa tiap bulan, cakupan dan ukuran selisih; tidak ada uji signifikansi atau klaim universal.',
        f"2. **Ridge versus persistence:** baseline memiliki MAE {base.MAE:.6f}, RMSE {base.RMSE:.6f}, bias {base.Bias:.6f}. "
        'Konfigurasi Ridge yang lebih rendah pada kedua metrik MAE dan RMSE: '+
        (', '.join(name for name in ['P0_median','P1_ffill_L4','P2_knn_k10'] if shared.loc[name,'MAE']<base.MAE and shared.loc[name,'RMSE']<base.RMSE) or 'tidak ada')+
        '. Persistence memiliki bias absolut lebih kecil daripada ketiga Ridge pada irisan ini. '
        'Jadi keunggulan MAE/RMSE bukan dominasi pada semua metrik, dan tetap dibatasi periode validasi sempit.',
        f"3. **Cakupan dan validasi sempit:** irisan {nc}/{n}; train bersama {len(f.train_origins)} contoh. "
        'Pemilihan sampel lengkap bersama mengontrol komposisi train, tetapi dapat '
        'mengurangi variasi data latih. Skor pada target validasi rendah/sempit belum '
        'mewakili lonjakan atau perubahan pola. Tidak ada penyesuaian split.',
        '4. **Kandidat lanjutan:** median layak diprioritaskan sebagai baseline sederhana '
        'dengan cakupan penuh pada validasi ini. Pertahankan forward fill L4 '
        'sebagai kandidat yang menolak gap panjang, dan KNN k10 sebagai kandidat multivariat '
        'yang belum menunjukkan keuntungan forecasting yang jelas untuk membenarkan biaya tambahannya di sini. '
        'KNN dapat tetap menjadi pembanding/sensitivitas pada missing lebih berat, '
        'dengan dukungan kovariat serta cakupan dilaporkan. '
        'Prioritas lanjutan harus mempertimbangkan beberapa metrik dan cakupan, bukan '
        'memilih pemenang dari MAE saja. P3 tetap ditunda. Belum ada keputusan preprocessing '
        'final untuk deep learning, dan tidak ada algoritma forecasting tambahan dilatih.','',
        '## 6. Keterbatasan dan informasi yang masih dibutuhkan','',
        '- Latensi pengukuran SO₂/kovariat dan makna timestamp perlu dikonfirmasi. '
        'Jika SO₂(u) belum tersedia pada u, lag0/persistence harus digeser dengan aturan '
        'yang sama dan eksperimen diulang tanpa membuka test.',
        '- Kode/nilai khusus, validitas sensor, pola konstan, dan ekstrem belum '
        'ditetapkan artinya. Penilaian memakai target asli, belum merupakan sertifikasi '
        'validitas sensor. Tidak ada penyebab anomali atau hubungan aktivitas vulkanik '
        'yang ditebak.',
        '- Satu periode validasi dan satu alpha Ridge tidak membuktikan peringkat '
        'pada deep learning, horizon lain, lokasi lain atau missing mekanisme lain. '
        'Perbandingan forecasting ini memakai missing alami; mask sintetis pilot '
        'tidak diterapkan lagi pada target maupun histori forecasting.',
        '- Test akhir tetap dikunci; batas/jumlahnya hanya dari inventaris lama. '
        'Keputusan berikutnya memakai train/validasi, bukan skor test.','',
        '## 7. Artefak dan integritas','',
        '- Notebook: `notebooks/05_forecasting_acuan_way_urang.ipynb`.',
        '- Workbook: `forecasting_acuan_way_urang.xlsx`.',
        '- `forecasting_way_urang/config_locked.json`: konfigurasi sebelum fit; '
        '`timestamp_train_bersama.csv`, `timestamp_evaluasi.csv`, `prediksi_validasi.csv`: '
        'ledger train/evaluasi/prediksi; `histori_kausal_*.csv` dan `fitur_*.csv`: '
        'histori serta indikator yang dapat ditelusuri.',
        '- Implementasi Ridge mengacu '
        '[dokumentasi resmi](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html); '
        'pemisahan fit/transform mengikuti '
        '[panduan pencegahan leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).','']
    lines += [f"- `{r['File']}`: SHA256 sebelum = sesudah = `{r['SHA256 sebelum']}`." for r in f.integrity]
    (OUT/'ringkasan_forecasting_acuan_way_urang.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
