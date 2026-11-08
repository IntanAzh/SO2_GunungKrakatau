# Penelusuran pembentukan sampel tahap kelima — Way Urang

**Pemeriksaan baca-saja terhadap catatan eksperimen lama. Tidak ada model, imputer, atau scaler yang dilatih/dijalankan kembali. Split tidak berubah, test tidak dimuat nilainya atau dievaluasi, dan hasil tahap kelima tidak ditimpa.**

## 1. Ringkasan temuan

Train asli memiliki **19197 pengamatan SO₂** dengan rentang **0.92–80.71 µg/m³**. Train bersama lama memiliki **12.654 target, 1,04–2.14 µg/m³**. Penyempitan bukan karena tidak adanya konsentrasi tinggi pada sumber dan bukan filter berdasarkan besar SO₂.

- Terdapat **5.289 pengamatan train asli di atas 2,14 µg/m³**. Kalibrasi 90 hari dan buffer lag mengeluarkan 1.448 di antaranya; 3.841 masih tersedia setelah pembatasan tersebut.
- **Seluruh 3.841 target lebih tinggi yang tersisa tidak memiliki fitur Curah Hujan lag1 finite pada P1.** Curah hujan asli kosong 24 Januari 2025 08:30 sampai 22 Juni 2025 22:30: **7.181 slot = 3.590,5 jam**. Ffill empat slot tidak dapat mengisi rentetan sepanjang itu.
- Median masih mempunyai 3.841 target tinggi yang layak mandiri; KNN mempunyai 3.786. Irisan train yang mensyaratkan semua pipeline lengkap ikut mengeluarkan semuanya dari median dan KNN. Model bersama baru mulai memakai origin 22 Juni 2025 23:30 (target 23 Juni 00:00).

## 2. Semesta timestamp dan urutan penelusuran

Inventaris train murni mempunyai **21.119 slot dan 19.197 SO₂ teramati** pada timestamp train. Untuk mengikuti kode supervised, waterfall memakai **21.119 origin train**, masing-masing dengan target t+30 menit. Sebelum purge terdapat **19.198 target teramati**: slot pertama train yang kosong tidak mempunyai origin dalam train, dan target pertama validasi yang teramati muncul dari origin train terakhir. Satu target lintas batas ini ditandai dan dipurge.
Urutan tabel adalah waterfall masker diagnostik untuk memisahkan dampak aturan, bukan perubahan urutan fungsi kode. **Lag struktural** hanya memeriksa apakah posisi waktu t−48 ada; nilai lag yang kosong belum dikeluarkan. Pada kode lama, imputasi seri dilakukan sebelum pembentukan nilai fitur lag. Kode juga menetapkan batas origin purge sebelum filter finite; operasi masker komutatif dan hasil akhir penelusuran dicocokkan persis dengan ledger train lama.
Statistik minimum/median/maksimum selalu memakai target asli yang teramati. Pada tahap pertama, jumlah sampel masih mencakup target kosong. Tanggal bulanan mengacu timestamp target. Kolom `Target melintasi validasi` mencegah pasangan batas waktu disalahartikan sebagai label train.

### Inventaris train asli per bulan

| Bulan | Slot train | SO2 asli teramati | Minimum | Median | Maksimum | Di atas maksimum final |
|---|---|---|---|---|---|---|
| seluruh train | 21119 | 19197 | 0.92 | 1.61 | 80.71 | 5289 |
| 2025-01 | 1488 | 385 | 0.92 | 1.63 | 2.15 | 1 |
| 2025-02 | 1344 | 730 | 1.24 | 1.39 | 79.57 | 23 |
| 2025-03 | 1488 | 1375 | 43.31 | 45.54 | 80.71 | 1375 |
| 2025-04 | 1440 | 1434 | 43.53 | 45.40 | 47.43 | 1434 |
| 2025-05 | 1488 | 1413 | 43.57 | 45.39 | 47.81 | 1413 |
| 2025-06 | 1440 | 1440 | 1.07 | 45.09 | 47.78 | 1043 |
| 2025-07 | 1488 | 1483 | 1.07 | 1.56 | 2.10 | 0 |
| 2025-08 | 1488 | 1486 | 1.04 | 1.56 | 2.05 | 0 |
| 2025-09 | 1440 | 1438 | 1.05 | 1.56 | 2.08 | 0 |
| 2025-10 | 1488 | 1488 | 1.04 | 1.56 | 2.14 | 0 |
| 2025-11 | 1440 | 1440 | 1.12 | 1.56 | 2.07 | 0 |
| 2025-12 | 1488 | 1488 | 1.08 | 1.57 | 2.03 | 0 |
| 2026-01 | 1488 | 1488 | 1.12 | 1.57 | 2.04 | 0 |
| 2026-02 | 1344 | 1342 | 1.08 | 1.55 | 2.02 | 0 |
| 2026-03 | 767 | 767 | 1.16 | 1.56 | 1.97 | 0 |

### P0_median

| Tahap | Jumlah sampel | Target asli teramati | Origin awal | Origin akhir | SO2 target minimum | SO2 target median | SO2 target maksimum | Keluar dari tahap sebelumnya | Target melintasi validasi |
|---|---|---|---|---|---|---|---|---|---|
| 01 Seluruh slot origin train | 21119 | 19198 | 2025-01-01 00:00:00 | 2026-03-16 23:00:00 | 0.92 | 1.61 | 80.71 | 0 | 1 |
| 02 Target t+1 asli teramati | 19198 | 19198 | 2025-01-02 11:00:00 | 2026-03-16 23:00:00 | 0.92 | 1.61 | 80.71 | 1921 | 1 |
| 03 Setelah lag struktural tersedia | 19198 | 19198 | 2025-01-02 11:00:00 | 2026-03-16 23:00:00 | 0.92 | 1.61 | 80.71 | 0 | 1 |
| 04 Setelah kalibrasi 90 hari | 16707 | 16707 | 2025-04-01 00:00:00 | 2026-03-16 23:00:00 | 1.04 | 1.60 | 47.81 | 2491 | 1 |
| 05 Setelah buffer lag 48 slot | 16659 | 16659 | 2025-04-02 00:00:00 | 2026-03-16 23:00:00 | 1.04 | 1.60 | 47.81 | 48 | 1 |
| 06 Setelah imputasi tanpa menyaring NaN | 16659 | 16659 | 2025-04-02 00:00:00 | 2026-03-16 23:00:00 | 1.04 | 1.60 | 47.81 | 0 | 1 |
| 07 Setelah fitur masih kosong diperiksa | 16659 | 16659 | 2025-04-02 00:00:00 | 2026-03-16 23:00:00 | 1.04 | 1.60 | 47.81 | 0 | 1 |
| 08 Setelah purge target validasi | 16658 | 16658 | 2025-04-02 00:00:00 | 2026-03-16 22:30:00 | 1.04 | 1.60 | 47.81 | 1 | 0 |
| 09 Layak final mandiri belum irisan | 16658 | 16658 | 2025-04-02 00:00:00 | 2026-03-16 22:30:00 | 1.04 | 1.60 | 47.81 | 0 | 0 |
| 10 Aktual dilatih setelah irisan bersama | 12654 | 12654 | 2025-06-22 23:30:00 | 2026-03-16 22:30:00 | 1.04 | 1.56 | 2.14 | 4004 | 0 |

### P1_ffill_L4

| Tahap | Jumlah sampel | Target asli teramati | Origin awal | Origin akhir | SO2 target minimum | SO2 target median | SO2 target maksimum | Keluar dari tahap sebelumnya | Target melintasi validasi |
|---|---|---|---|---|---|---|---|---|---|
| 01 Seluruh slot origin train | 21119 | 19198 | 2025-01-01 00:00:00 | 2026-03-16 23:00:00 | 0.92 | 1.61 | 80.71 | 0 | 1 |
| 02 Target t+1 asli teramati | 19198 | 19198 | 2025-01-02 11:00:00 | 2026-03-16 23:00:00 | 0.92 | 1.61 | 80.71 | 1921 | 1 |
| 03 Setelah lag struktural tersedia | 19198 | 19198 | 2025-01-02 11:00:00 | 2026-03-16 23:00:00 | 0.92 | 1.61 | 80.71 | 0 | 1 |
| 04 Setelah kalibrasi 90 hari | 16707 | 16707 | 2025-04-01 00:00:00 | 2026-03-16 23:00:00 | 1.04 | 1.60 | 47.81 | 2491 | 1 |
| 05 Setelah buffer lag 48 slot | 16659 | 16659 | 2025-04-02 00:00:00 | 2026-03-16 23:00:00 | 1.04 | 1.60 | 47.81 | 48 | 1 |
| 06 Setelah imputasi tanpa menyaring NaN | 16659 | 16659 | 2025-04-02 00:00:00 | 2026-03-16 23:00:00 | 1.04 | 1.60 | 47.81 | 0 | 1 |
| 07 Setelah fitur masih kosong diperiksa | 12665 | 12665 | 2025-06-22 23:30:00 | 2026-03-16 23:00:00 | 1.04 | 1.56 | 2.14 | 3994 | 1 |
| 08 Setelah purge target validasi | 12664 | 12664 | 2025-06-22 23:30:00 | 2026-03-16 22:30:00 | 1.04 | 1.56 | 2.14 | 1 | 0 |
| 09 Layak final mandiri belum irisan | 12664 | 12664 | 2025-06-22 23:30:00 | 2026-03-16 22:30:00 | 1.04 | 1.56 | 2.14 | 0 | 0 |
| 10 Aktual dilatih setelah irisan bersama | 12654 | 12654 | 2025-06-22 23:30:00 | 2026-03-16 22:30:00 | 1.04 | 1.56 | 2.14 | 10 | 0 |

### P2_knn_k10

| Tahap | Jumlah sampel | Target asli teramati | Origin awal | Origin akhir | SO2 target minimum | SO2 target median | SO2 target maksimum | Keluar dari tahap sebelumnya | Target melintasi validasi |
|---|---|---|---|---|---|---|---|---|---|
| 01 Seluruh slot origin train | 21119 | 19198 | 2025-01-01 00:00:00 | 2026-03-16 23:00:00 | 0.92 | 1.61 | 80.71 | 0 | 1 |
| 02 Target t+1 asli teramati | 19198 | 19198 | 2025-01-02 11:00:00 | 2026-03-16 23:00:00 | 0.92 | 1.61 | 80.71 | 1921 | 1 |
| 03 Setelah lag struktural tersedia | 19198 | 19198 | 2025-01-02 11:00:00 | 2026-03-16 23:00:00 | 0.92 | 1.61 | 80.71 | 0 | 1 |
| 04 Setelah kalibrasi 90 hari | 16707 | 16707 | 2025-04-01 00:00:00 | 2026-03-16 23:00:00 | 1.04 | 1.60 | 47.81 | 2491 | 1 |
| 05 Setelah buffer lag 48 slot | 16659 | 16659 | 2025-04-02 00:00:00 | 2026-03-16 23:00:00 | 1.04 | 1.60 | 47.81 | 48 | 1 |
| 06 Setelah imputasi tanpa menyaring NaN | 16659 | 16659 | 2025-04-02 00:00:00 | 2026-03-16 23:00:00 | 1.04 | 1.60 | 47.81 | 0 | 1 |
| 07 Setelah fitur masih kosong diperiksa | 16582 | 16582 | 2025-04-02 00:00:00 | 2026-03-16 23:00:00 | 1.04 | 1.60 | 47.81 | 77 | 1 |
| 08 Setelah purge target validasi | 16581 | 16581 | 2025-04-02 00:00:00 | 2026-03-16 22:30:00 | 1.04 | 1.60 | 47.81 | 1 | 0 |
| 09 Layak final mandiri belum irisan | 16581 | 16581 | 2025-04-02 00:00:00 | 2026-03-16 22:30:00 | 1.04 | 1.60 | 47.81 | 0 | 0 |
| 10 Aktual dilatih setelah irisan bersama | 12654 | 12654 | 2025-06-22 23:30:00 | 2026-03-16 22:30:00 | 1.04 | 1.56 | 2.14 | 3927 | 0 |

Tabel **Tahap_per_bulan** dalam workbook menyajikan jumlah dan rentang SO₂ per bulan untuk seluruh tahap dan pipeline. **Tahap 09 layak mandiri bukan model yang pernah dilatih**; model lama ketiganya memakai irisan tahap 10.

## 3. Alasan sampel keluar dan penanda tumpang tindih

- `Penanda_semua_timestamp` memuat satu baris untuk setiap origin train, target asli, alasan dasar, penanda missing asal, alasan per pipeline dan status aktual bersama. Penanda boolean lebih rinci tersedia pada `penelusuran_tahap5/penanda_lengkap_per_timestamp.csv`.
- **Target kosong:** tidak boleh diisi untuk melatih/menilai. **Histori struktural belum tersedia:** posisi lag di luar cakupan waktu. **Kalibrasi/buffer:** pembatasan rancangan, bukan nilai rusak.
- **SO₂ historis/kovariat asli kosong** hanya penanda asal. Nilai tersebut masih berkesempatan diisi. Baris baru ditolak bila fitur hasil preprocessing tetap nonfinite atau aturan dasar lain tidak terpenuhi.
- **P1:** `P1_BELUM_ADA_DONOR` dan `P1_UMUR_DONOR_LEBIH_4` adalah kebijakan forward fill; hasil imputasi tidak mereset umur donor. Bukan exception algoritma.
- **P2:** `P2_GUARD_KOVARIAT_KURANG_2` berarti query tidak ditransformasi karena tidak memenuhi minimum dukungan. Ini pembatasan rancangan, bukan kegagalan komputasi KNN. Sisa NaN tak terjelaskan, jika ada, ditampilkan terpisah dan tidak diberi penyebab rekaan.
- **Irisan:** `LAYAK_MANDIRI_KELUAR_IRISAN` berarti pipeline itu sebenarnya memiliki fitur dan target yang layak, tetapi pipeline lain tidak. Ini seleksi sampel bersama, bukan kegagalan imputasi pipeline tersebut.
- Alasan dapat overlap pada timestamp yang sama. `Alasan_overlap` tidak boleh dijumlah sebagai jumlah unik yang keluar. `Kombinasi_alasan` memakai kombinasi tepat, sedangkan kolom keluar waterfall menghitung kehilangan berurutan yang tidak dihitung ulang pada tahap berikutnya.

| Pemeriksaan | Hasil |
|---|---|
| dropna terlalu luas | Tidak ditemukan pada penyusunan sampel. Satu g.dropna() hanya statistik SO2 per bulan/split; tidak mengubah data/fitur. |
| Sampel dikeluarkan sebelum imputasi | Tidak ada filter complete-case sebelum imputasi. Kalibrasi 90 hari/buffer 48 slot membatasi kelayakan model; missing warmup tidak diisi dari masa depan. Guard minimum dua kovariat KNN memang mencegah transform query tanpa dukungan, lalu sisa NaN ditolak sebagai fitur Ridge. |
| Pemeriksaan fitur | np.isfinite(X[train_orig]).all(axis=1) tepat 36 fitur (18 numerik,18 indikator). Kolom metadata dan kolom bukan fitur tidak ikut. |
| Urutan kode purge | Kode menetapkan rentang origin sampai a-2 sekaligus sebelum menghitung finite; trace menampilkan finite lalu purge untuk memisahkan efek. Operator mask komutatif, himpunan akhir diverifikasi sama. |
| Imputasi komputasional gagal | Notebook tersimpan tidak memiliki error fit/transform. Sisa NaN berasal dari pembatasan/guard; label tidak terjelaskan diperiksa terpisah. |
| Train pipeline mandiri | Tahap 5 benar-benar melatih semua Ridge pada irisan, bukan pada himpunan layak mandiri. Jumlah mandiri pada trace adalah eligibility tanpa fit ulang. |
| Konsentrasi dipakai sebagai filter | Tidak ada. Ambang lebih tinggi hanya memakai maksimum target train final untuk penelusuran ini, bukan batas validitas sensor. |

### Sampel yang layak sendiri tetapi hilang karena irisan

| Pipeline | Layak sendiri total | Aktual dipakai bersama | Jumlah sampel | SO2 target minimum | SO2 target median | SO2 target maksimum | Target di atas maksimum final |
|---|---|---|---|---|---|---|---|
| P0_median | 16658 | 12654 | 4004 | 1.16 | 45.36 | 47.81 | 3841 |
| P1_ffill_L4 | 12664 | 12654 | 10 | 1.38 | 1.58 | 1.88 | 0 |
| P2_knn_k10 | 16581 | 12654 | 3927 | 1.16 | 45.36 | 47.81 | 3786 |

### Fitur yang menolak target tinggi setelah kalibrasi/buffer

| Pipeline | Fitur | Sisa NaN dalam scope | Sisa NaN target tinggi dalam scope | Denominator target tinggi scope |
|---|---|---|---|---|
| P1_ffill_L4 | SO2_lag0 | 1 | 1 | 3841 |
| P1_ffill_L4 | SO2_lag1 | 2 | 2 | 3841 |
| P1_ffill_L4 | SO2_lag2 | 3 | 3 | 3841 |
| P1_ffill_L4 | SO2_lag6 | 7 | 7 | 3841 |
| P1_ffill_L4 | SO2_lag12 | 13 | 13 | 3841 |
| P1_ffill_L4 | SO2_lag24 | 25 | 25 | 3841 |
| P1_ffill_L4 | SO2_lag48 | 49 | 49 | 3841 |
| P1_ffill_L4 | PM10_lag1 | 6 | 2 | 3841 |
| P1_ffill_L4 | PM2.5_lag1 | 10 | 2 | 3841 |
| P1_ffill_L4 | CO_lag1 | 14 | 2 | 3841 |
| P1_ffill_L4 | O3_lag1 | 2 | 2 | 3841 |
| P1_ffill_L4 | NO2_lag1 | 2 | 2 | 3841 |
| P1_ffill_L4 | Kec.Angin_lag1 | 2 | 2 | 3841 |
| P1_ffill_L4 | Kelembaban_lag1 | 2 | 2 | 3841 |
| P1_ffill_L4 | Suhu_lag1 | 2 | 2 | 3841 |
| P1_ffill_L4 | Tek.Udara_lag1 | 2 | 2 | 3841 |
| P1_ffill_L4 | Sol.Rad_lag1 | 33 | 2 | 3841 |
| P1_ffill_L4 | Curah Hujan_lag1 | 3943 | 3841 | 3841 |
| P2_knn_k10 | SO2_lag0 | 2 | 1 | 3841 |
| P2_knn_k10 | SO2_lag1 | 4 | 2 | 3841 |
| P2_knn_k10 | SO2_lag2 | 6 | 3 | 3841 |
| P2_knn_k10 | SO2_lag6 | 10 | 7 | 3841 |
| P2_knn_k10 | SO2_lag12 | 16 | 13 | 3841 |
| P2_knn_k10 | SO2_lag24 | 28 | 25 | 3841 |
| P2_knn_k10 | SO2_lag48 | 52 | 49 | 3841 |
| P2_knn_k10 | PM10_lag1 | 17 | 8 | 3841 |
| P2_knn_k10 | PM2.5_lag1 | 17 | 8 | 3841 |
| P2_knn_k10 | CO_lag1 | 17 | 8 | 3841 |
| P2_knn_k10 | O3_lag1 | 17 | 8 | 3841 |
| P2_knn_k10 | NO2_lag1 | 17 | 8 | 3841 |
| P2_knn_k10 | Kec.Angin_lag1 | 17 | 8 | 3841 |
| P2_knn_k10 | Kelembaban_lag1 | 17 | 8 | 3841 |
| P2_knn_k10 | Suhu_lag1 | 17 | 8 | 3841 |
| P2_knn_k10 | Tek.Udara_lag1 | 17 | 8 | 3841 |
| P2_knn_k10 | Sol.Rad_lag1 | 17 | 8 | 3841 |
| P2_knn_k10 | Curah Hujan_lag1 | 17 | 8 | 3841 |

Scope adalah target train asli teramati dengan origin>=4.368. Hitungan fitur tumpang tindih: satu sampel dapat kehilangan beberapa lag SO₂ dan beberapa kovariat sekaligus. Pada P1, Curah Hujan saja sudah mencakup seluruh 3.841 target tinggi dalam scope.

## 4. Periode SO₂ lebih tinggi pada train asli

Lebih tinggi didefinisikan **SO₂ > 2.14 µg/m³**, yakni di atas maksimum target train bersama lama. Ini ambang deskriptif untuk pertanyaan penelusuran, bukan batas sensor, batas outlier atau alasan menghapus nilai. Observasi tinggi muncul dari **18 Januari 2025 23:30 sampai 22 Juni 2025 17:00**. Rentetan berikut terputus ketika ada slot kosong atau nilai tidak melebihi ambang.

| Periode | Awal target tinggi | Akhir target tinggi | Jumlah observasi | Minimum SO2 | Median SO2 | Maksimum SO2 |
|---|---|---|---|---|---|---|
| 1 | 2025-01-18 23:30:00 | 2025-01-18 23:30:00 | 1 | 2.15 | 2.15 | 2.15 |
| 2 | 2025-02-28 12:30:00 | 2025-03-15 03:30:00 | 703 | 43.69 | 45.78 | 80.71 |
| 3 | 2025-03-17 11:30:00 | 2025-03-27 10:30:00 | 479 | 43.31 | 45.42 | 47.42 |
| 4 | 2025-03-27 11:30:00 | 2025-03-31 10:00:00 | 190 | 43.80 | 45.36 | 47.03 |
| 5 | 2025-03-31 11:00:00 | 2025-04-04 13:30:00 | 198 | 43.96 | 45.42 | 46.91 |
| 6 | 2025-04-04 14:30:00 | 2025-04-06 01:30:00 | 71 | 44.12 | 45.30 | 46.75 |
| 7 | 2025-04-06 02:30:00 | 2025-04-15 08:30:00 | 445 | 43.53 | 45.33 | 47.43 |
| 8 | 2025-04-15 09:30:00 | 2025-04-21 15:00:00 | 300 | 43.72 | 45.43 | 47.02 |
| 9 | 2025-04-21 16:00:00 | 2025-04-24 00:00:00 | 113 | 43.72 | 45.55 | 46.99 |
| 10 | 2025-04-24 01:00:00 | 2025-04-28 03:00:00 | 197 | 44.04 | 45.45 | 47.10 |
| 11 | 2025-04-28 04:00:00 | 2025-05-17 21:00:00 | 947 | 43.85 | 45.40 | 47.81 |
| 12 | 2025-05-19 11:00:00 | 2025-06-22 17:00:00 | 1645 | 43.14 | 45.38 | 47.78 |

### Target tinggi yang bertahan: P0_median

| Tahap | Jumlah sampel | SO2 target minimum | SO2 target median | SO2 target maksimum |
|---|---|---|---|---|
| 01 Seluruh slot origin train | 5289 | 2.15 | 45.43 | 80.71 |
| 02 Target t+1 asli teramati | 5289 | 2.15 | 45.43 | 80.71 |
| 03 Setelah lag struktural tersedia | 5289 | 2.15 | 45.43 | 80.71 |
| 04 Setelah kalibrasi 90 hari | 3889 | 43.14 | 45.39 | 47.81 |
| 05 Setelah buffer lag 48 slot | 3841 | 43.14 | 45.39 | 47.81 |
| 06 Setelah imputasi tanpa menyaring NaN | 3841 | 43.14 | 45.39 | 47.81 |
| 07 Setelah fitur masih kosong diperiksa | 3841 | 43.14 | 45.39 | 47.81 |
| 08 Setelah purge target validasi | 3841 | 43.14 | 45.39 | 47.81 |
| 09 Layak final mandiri belum irisan | 3841 | 43.14 | 45.39 | 47.81 |
| 10 Aktual dilatih setelah irisan bersama | 0 | — | — | — |

### Target tinggi yang bertahan: P1_ffill_L4

| Tahap | Jumlah sampel | SO2 target minimum | SO2 target median | SO2 target maksimum |
|---|---|---|---|---|
| 01 Seluruh slot origin train | 5289 | 2.15 | 45.43 | 80.71 |
| 02 Target t+1 asli teramati | 5289 | 2.15 | 45.43 | 80.71 |
| 03 Setelah lag struktural tersedia | 5289 | 2.15 | 45.43 | 80.71 |
| 04 Setelah kalibrasi 90 hari | 3889 | 43.14 | 45.39 | 47.81 |
| 05 Setelah buffer lag 48 slot | 3841 | 43.14 | 45.39 | 47.81 |
| 06 Setelah imputasi tanpa menyaring NaN | 3841 | 43.14 | 45.39 | 47.81 |
| 07 Setelah fitur masih kosong diperiksa | 0 | — | — | — |
| 08 Setelah purge target validasi | 0 | — | — | — |
| 09 Layak final mandiri belum irisan | 0 | — | — | — |
| 10 Aktual dilatih setelah irisan bersama | 0 | — | — | — |

### Target tinggi yang bertahan: P2_knn_k10

| Tahap | Jumlah sampel | SO2 target minimum | SO2 target median | SO2 target maksimum |
|---|---|---|---|---|
| 01 Seluruh slot origin train | 5289 | 2.15 | 45.43 | 80.71 |
| 02 Target t+1 asli teramati | 5289 | 2.15 | 45.43 | 80.71 |
| 03 Setelah lag struktural tersedia | 5289 | 2.15 | 45.43 | 80.71 |
| 04 Setelah kalibrasi 90 hari | 3889 | 43.14 | 45.39 | 47.81 |
| 05 Setelah buffer lag 48 slot | 3841 | 43.14 | 45.39 | 47.81 |
| 06 Setelah imputasi tanpa menyaring NaN | 3841 | 43.14 | 45.39 | 47.81 |
| 07 Setelah fitur masih kosong diperiksa | 3786 | 43.14 | 45.39 | 47.81 |
| 08 Setelah purge target validasi | 3786 | 43.14 | 45.39 | 47.81 |
| 09 Layak final mandiri belum irisan | 3786 | 43.14 | 45.39 | 47.81 |
| 10 Aktual dilatih setelah irisan bersama | 0 | — | — | — |

Workbook juga memuat **Periode_tinggi_per_tahap** untuk jumlah bertahan dan keluar pada setiap rentetan, serta **Tahap_per_bulan** untuk distribusi bulanan. Tidak ada konsentrasi tinggi yang dihapus dari sumber atau dinyatakan salah.

| Awal pengukuran asli kosong | Akhir pengukuran asli kosong | Jumlah slot | Durasi slot jam |
|---|---|---|---|
| 2025-01-01 00:00:00 | 2025-01-06 11:30:00 | 264 | 132.00 |
| 2025-01-06 21:30:00 | 2025-01-11 18:00:00 | 234 | 117.00 |
| 2025-01-12 03:00:00 | 2025-01-14 09:30:00 | 110 | 55.00 |
| 2025-01-19 13:00:00 | 2025-01-23 18:00:00 | 203 | 101.50 |
| 2025-01-24 08:30:00 | 2025-06-22 22:30:00 | 7181 | 3590.50 |
| 2025-08-05 02:30:00 | 2025-08-06 05:30:00 | 55 | 27.50 |

## 5. Mengapa cakupan KNN 99,96% dan median 100%

KNN menghasilkan **4.522/4.524 = 99,955791%** prediksi, median **4.524/4.524 = 100%**. Dua origin berikut tidak memenuhi guard minimum dua kovariat asli untuk mengisi fitur KNN. Median memiliki statistik train untuk mengisi kovariat tersebut, sehingga tetap dapat membuat prediksi.
Pada origin 8 April hanya satu kovariat lag tersedia, sedangkan pada 21 April tidak ada. SO₂ origin sendiri teramati pada keduanya. Karena guard menghitung **kovariat asli**, SO₂ origin tidak menggantikan syarat dua kovariat. Keputusan ini sudah tercantum pada rancangan lama.

| Origin | Target | SO2 target asli | SO2 origin asli | Timestamp kovariat yang dicari | Jumlah kovariat asli tersedia | Kovariat asli tersedia | Fitur Ridge masih kosong | Penyebab |
|---|---|---|---|---|---|---|---|---|
| 2026-04-08 11:30:00 | 2026-04-08 12:00:00 | 1.56 | 1.71 | 2026-04-08 11:00:00 | 1 | Curah Hujan | PM10_lag1, PM2.5_lag1, CO_lag1, O3_lag1, NO2_lag1, Kec.Angin_lag1, Kelembaban_lag1, Suhu_lag1, Tek.Udara_lag1, Sol.Rad_lag1 | Guard KNN minimum dua kovariat asli tidak lolos; kovariat tidak diisi; pemeriksaan fitur menolak origin. |
| 2026-04-21 13:00:00 | 2026-04-21 13:30:00 | 1.73 | 1.66 | 2026-04-21 12:30:00 | 0 |  | PM10_lag1, PM2.5_lag1, CO_lag1, O3_lag1, NO2_lag1, Kec.Angin_lag1, Kelembaban_lag1, Suhu_lag1, Tek.Udara_lag1, Sol.Rad_lag1, Curah Hujan_lag1 | Guard KNN minimum dua kovariat asli tidak lolos; kovariat tidak diisi; pemeriksaan fitur menolak origin. |

Fitur yang masih kosong dan nilai pengisi median ditampilkan di **KNN_fitur_kosong**. Tidak ada fallback median tersembunyi pada KNN atau forward fill, dan tidak ada prediksi baru dihitung.

## 6. Persistence dan kesamaan timestamp evaluasi

Persistence memakai SO₂ asli pada origin jika tersedia. Bila SO₂ origin kosong, ia memakai pengamatan asli terakhir yang lebih lama. Distribusi usia berikut dihitung ulang dari sumber prefix train-validasi, bukan sekadar mempercayai kolom usia yang disimpan. Usia satu slot=30 menit.

| Domain | Usia slot | Usia menit | Jumlah prediksi | Denominator prediksi domain | Proporsi persen |
|---|---|---|---|---|---|
| seluruh target asli validasi | 0 | 0 | 4522 | 4524 | 99.956 |
| seluruh target asli validasi | 1 | 30 | 2 | 4524 | 0.044 |
| irisan semua metode | 0 | 0 | 4485 | 4487 | 99.955 |
| irisan semua metode | 1 | 30 | 2 | 4487 | 0.045 |

| Origin | Target | Waktu donor asli | SO2 donor asli | Usia slot | Usia menit | Dalam irisan |
|---|---|---|---|---|---|---|
| 2026-04-08 11:00:00 | 2026-04-08 11:30:00 | 2026-04-08 10:30:00 | 1.50 | 1 | 30 | True |
| 2026-04-21 12:30:00 | 2026-04-21 13:00:00 | 2026-04-21 12:00:00 | 1.70 | 1 | 30 | True |

Pada seluruh target validasi, **4.522 prediksi memakai SO₂ tepat pada origin**, dan **2 memakai donor 30 menit lebih lama**. Pada irisan 4.487 target, pembagiannya 4.485 dan 2. Asumsi bahwa pengukuran pada timestamp itu sudah tiba tetap bergantung pada latensi yang belum dikonfirmasi.
Seluruh delapan baris skor agregat tahap kelima dihitung ulang dari target asli dan prediksi tersimpan; hasilnya cocok. Keempat metode pada domain **irisan semua metode** mempunyai SHA256 daftar timestamp target yang identik. Skor domain **seluruh berhasil** memang memakai himpunan berbeda dan harus dibaca bersama cakupan, bukan dianggap perbandingan timestamp identik.

| Metode | Domain | Jumlah target asli | MAE hitung ulang | RMSE hitung ulang | Bias hitung ulang | Sesuai skor lama |
|---|---|---|---|---|---|---|
| P0_median | seluruh berhasil | 4524 | 0.081652 | 0.105556 | 0.000543 | True |
| P0_median | irisan semua metode | 4487 | 0.081446 | 0.105314 | 0.000343 | True |
| P1_ffill_L4 | seluruh berhasil | 4489 | 0.081451 | 0.105283 | 0.000209 | True |
| P1_ffill_L4 | irisan semua metode | 4487 | 0.081452 | 0.105289 | 0.000230 | True |
| P2_knn_k10 | seluruh berhasil | 4522 | 0.081665 | 0.105547 | 0.000501 | True |
| P2_knn_k10 | irisan semua metode | 4487 | 0.081457 | 0.105296 | 0.000287 | True |
| Persistence | seluruh berhasil | 4524 | 0.115975 | 0.148921 | 0.000029 | True |
| Persistence | irisan semua metode | 4487 | 0.115830 | 0.148758 | -0.000172 | True |

## 7. Kesimpulan dan usulan koreksi minimum

1. **Penyebab rentang sempit:** kalibrasi/buffer mengeluarkan 1.448 pengamatan tinggi awal, kemudian persyaratan Curah Hujan pada P1 mengeluarkan seluruh 3.841 pengamatan tinggi yang tersisa. Irisan bersama memindahkan pembatasan P1 itu ke median dan KNN. Purge hanya mengeluarkan satu target lintas batas dan bukan penyebab hilangnya konsentrasi tinggi.
2. **Implementasi versus rancangan:** tidak ditemukan complete-case dropna pada kolom yang tidak dipakai atau pembuangan nilai lag kosong sebelum kesempatan imputasi. Himpunan akhir cocok dengan ledger model lama. Ini terutama **konsekuensi rancangan train bersama dan kelengkapan kovariat**, dengan bias seleksi terhadap periode konsentrasi lebih tinggi; bukan bukti konsentrasi tinggi salah. Label “kegagalan imputasi” perlu dibedakan dari penolakan guard/batas donor yang disengaja.
3. **Koreksi minimum sekarang:** lengkapi interpretasi catatan tahap 5 dengan hasil penelusuran ini. Skor dan artefak lama dipertahankan sebagai hasil eksperimen pada kohort terbatas; jangan menyebutnya perbandingan yang sudah mewakili rentang 0,92–80,71 µg/m³.
4. **Usulan eksperimen berikut, belum dijalankan:** pertimbangkan ablasi Curah Hujan secara konsisten pada seluruh pipeline karena ketersediaannya membatasi cakupan. Tetapkan ulang daftar fitur dan peran curah hujan sebagai fitur model/fitur bantu imputer sebelum fit. Pertahankan split, target asli, kausalitas dan evaluasi bersama; simpan sebagai eksperimen baru. Jangan memberi median fallback diam-diam pada P1 untuk menyelamatkan sampel.
5. **Simulasi eligibility saja:** mengabaikan dua kolom Ridge terkait Curah Hujan pada pemeriksaan finite atas histori lama akan menyisakan **16.530 sampel bersama**, termasuk **3.786 target >2,14**, dengan rentang **1,04–47,81 µg/m³**. Ini bukan hasil pipeline/model baru dan tidak memprediksi skornya. Rentang 80,71 tetap tidak kembali karena pembatasan kalibrasi; bila periode awal penting untuk tujuan penelitian, desain kalibrasi awal yang lebih sesuai perlu dipraregistrasi sebagai sensitivitas berbasis dukungan data, tanpa memilih dari skor test.
6. **Alternatif, bukan perbaikan diam-diam:** melatih tiap pipeline pada sampel layaknya sendiri akan memisahkan pertanyaan cakupan, tetapi membuat komposisi train berbeda. Jika kelak dilakukan, laporkan sebagai estimand/perbandingan terpisah, tetap tampilkan train bersama dan evaluasi pada target bersama. Tidak ada pelatihan tersebut pada pemeriksaan ini.

| Skenario | Jumlah sampel | Target di atas maksimum final | SO2 target minimum | SO2 target median | SO2 target maksimum | Catatan |
|---|---|---|---|---|---|---|
| Hanya abaikan Curah Hujan pada pemeriksaan fitur | 16530 | 3786 | 1.04 | 1.60 | 47.81 | Tidak melatih/imputasi ulang, tidak mengubah fitur/hasil lama, bukan skor baru. |

## 8. Integritas dan artefak

- Workbook: `penelusuran_sampel_tahap5.xlsx`; penanda lengkap tambahan: `penelusuran_tahap5/penanda_lengkap_per_timestamp.csv`.
- Semua sumber dan artefak tahap kelima yang dibaca dicatat hash sebelum dan sesudah. Tidak ada model/scaler/imputer dijalankan dan tidak ada hasil sebelumnya ditimpa.

- `outputs\forecasting_way_urang\checks.json`: SHA256 sebelum = sesudah = `4560b95dd3c624e18ecbd9dab45a9826ded60bb5306e3b2cd3bc76448ab266c8`.
- `outputs\forecasting_way_urang\config_locked.json`: SHA256 sebelum = sesudah = `f6b7aacb5e9a10f6532321a79acd780246439072871e74311a39b31803e511c1`.
- `outputs\forecasting_way_urang\fitur_P0_median.csv`: SHA256 sebelum = sesudah = `1c11c70be747805c3550670a5cc8d049dfde5de9479a4ee68a2ca8baa12eaf7b`.
- `outputs\forecasting_way_urang\fitur_P1_ffill_L4.csv`: SHA256 sebelum = sesudah = `816765eb832c80573d6ec9298233d158991ae68bd988910208ed77f1c033742c`.
- `outputs\forecasting_way_urang\fitur_P2_knn_k10.csv`: SHA256 sebelum = sesudah = `c660f102f7ddcfbb5acf06d5a41cfec40633676cdec6e86b39d05c376a7f6e56`.
- `outputs\forecasting_way_urang\histori_kausal_P0_median.csv`: SHA256 sebelum = sesudah = `812fc28df88b00574b727e30a56f8160b72d085ea03ebf0cb977b3d0ed363779`.
- `outputs\forecasting_way_urang\histori_kausal_P1_ffill_L4.csv`: SHA256 sebelum = sesudah = `6cd91ae0093a286f243b0d4bc3c5c329bc4245558dcbb0d530d8de13bb68424d`.
- `outputs\forecasting_way_urang\histori_kausal_P2_knn_k10.csv`: SHA256 sebelum = sesudah = `e9f36afc68129a3b0ba43f051d5b15cd025a5940468f3d1be96863761f46cfb6`.
- `outputs\forecasting_way_urang\prediksi_validasi.csv`: SHA256 sebelum = sesudah = `293a8ae383dce1bfb0d8be2b4169e2aaf5235661beb15061aa5e90ae7d930d28`.
- `outputs\forecasting_way_urang\timestamp_evaluasi.csv`: SHA256 sebelum = sesudah = `1853716dff9a0f330c4e73e9315a066b75eaa799a864e54fe15826f9fa57f341`.
- `outputs\forecasting_way_urang\timestamp_train_bersama.csv`: SHA256 sebelum = sesudah = `8309295a37b5364032317b8b83b47fcce39b980f41e130c23ca5f0dbe1f3b774`.
- `outputs\forecasting_acuan_way_urang.xlsx`: SHA256 sebelum = sesudah = `1bfd0761f299d27faf7455448520ac11d8231feb7fc8d5ca930ef579a21a3a2c`.
- `outputs\forecasting_way_urang_tables.json`: SHA256 sebelum = sesudah = `ecb4e92a9b51f28349a4b2b3db5aabdba89742616e10b38fe34cb90726fbecc3`.
- `outputs\ringkasan_forecasting_acuan_way_urang.md`: SHA256 sebelum = sesudah = `5c5c0cdd38eb2b831461af085619dd61dc98e82c6923777b42ad7b71803bca9f`.
- `notebooks\05_forecasting_acuan_way_urang.ipynb`: SHA256 sebelum = sesudah = `630eab31b3eada0aa7cb01bf889aba89062c3dfbb14c83dd8c2cfb5add3fd4b0`.
- `outputs\pilot_way_urang\grid_dan_penanda_asli.csv`: SHA256 sebelum = sesudah = `1d547e9b320bc547ef5a3ec8a1603df87cf4248f596288b86fae2e2cc4c40df8`.
- `scripts\forecast_ridge_way_urang.py`: SHA256 sebelum = sesudah = `439d171cf3292a3a49f503900e7ae4ddd23e1a667e792f2ced7110e6e8019f04`.
- `data_raw\Telkom University Purwokerto - Intan Azizah.xlsx`: SHA256 sebelum = sesudah = `bd00734b954e8df024c048575fad07f1eb1cbe1083d4c8929ae362aeb45a7406`.
