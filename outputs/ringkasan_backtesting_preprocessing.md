# Tahap 6 — expanding-window preprocessing Way Urang

Ini **validasi pengembangan**: development set adalah train dan validasi lama yang telah digunakan pada eksperimen sebelumnya. Test akhir mulai **2026-06-19 06:30** tetap dikunci; nilainya tidak dimuat atau dievaluasi. Way Urang masih pilot sementara.

## Rancangan yang dikunci sebelum model

Train awal 120 hari kalender; sisa development dibagi empat jendela validasi tidak tumpang tindih, berbeda paling banyak satu slot 30 menit. Konfigurasi tersimpan di `backtesting_way_urang/fold_config_locked.json`; SHA256 37d0fc2f8ef09b7a44f6b2f666f598160f78a9fe5f955ef19c3d8969497c13b5. Tidak ada penyesuaian berdasarkan skor.

| Fold | Train mulai | Train akhir | Validasi mulai | Validasi akhir | Slot train | Slot validasi |
|---|---|---|---|---|---|---|
| Fold_01 | 2025-01-01 00:00:00 | 2025-04-30 23:30:00 | 2025-05-01 00:00:00 | 2025-08-12 13:30:00 | 5760 | 4972 |
| Fold_02 | 2025-01-01 00:00:00 | 2025-08-12 13:30:00 | 2025-08-12 14:00:00 | 2025-11-24 03:00:00 | 10732 | 4971 |
| Fold_03 | 2025-01-01 00:00:00 | 2025-11-24 03:00:00 | 2025-11-24 03:30:00 | 2026-03-07 16:30:00 | 15703 | 4971 |
| Fold_04 | 2025-01-01 00:00:00 | 2026-03-07 16:30:00 | 2026-03-07 17:00:00 | 2026-06-19 06:00:00 | 20674 | 4971 |

Kalibrasi tetap 90 hari pertama train terkait, pembaruan prefix 30 hari dan buffer histori 48 slot. Kecukupan ditetapkan sebelum Ridge: minimal 340 sampel bersama (10 × 34 fitur) dan dua nilai target berbeda. Ini pemeriksaan kecukupan komputasional, bukan jaminan daya statistik. Seluruh gate diperiksa saat jumlah fit Ridge masih nol; seluruhnya lulus tanpa mengubah batas fold.

| Fold | Jumlah train bersama | Target unik train bersama | Cukup sampel | Ridge fit count global sebelum gate |
|---|---|---|---|---|
| Fold_01 | 1379 | 284 | True | 0 |
| Fold_02 | 6190 | 436 | True | 0 |
| Fold_03 | 11154 | 449 | True | 0 |
| Fold_04 | 16086 | 449 | True | 0 |

## Parameter dan kausalitas

Sama dengan 5B tanpa Curah Hujan: horizon satu slot/30 menit; SO₂ lag 0, 1, 2, 6, 12, 24, 48; sepuluh kovariat lag satu slot: PM10, PM2.5, CO, O3, NO2, Kec.Angin, Kelembaban, Suhu, Tek.Udara, Sol.Rad. Sebanyak 17 fitur numerik ditambah 17 indikator missing asli. Ridge alpha=1, intercept aktif, solver SVD; scaler fitur diestimasi hanya pada train bersama fold tersebut.

P0 mengisi dengan median prefix train; P1 memakai pengamatan asli sebelumnya maksimum empat slot, tanpa fallback; P2 KNN k=10, distance weighting, nan-euclidean, scaling prefix train, donor SO₂ asli dan minimal dua kovariat asli. Semua estimator dibuat ulang dari nol pada setiap fold. Kalibrasi dan prefix train hanya menggunakan waktu sebelumnya; parameter imputasi validasi dibekukan pada akhir train. Histori direkonstruksi sebelum pembentukan lag. Tidak ada nilai target masa depan, backward fill, interpolasi dua arah atau clipping. Pengamatan validasi fold terdahulu boleh menjadi data train fold berikut setelah waktunya tersedia.

Asumsi ketersediaan tetap: SO₂ pada origin sudah tersedia saat prediksi diterbitkan, kovariat sudah tersedia dengan keterlambatan satu slot. Ini belum dikonfirmasi penyedia data. Tidak ada waktu mentah atau arah angin linear sebagai fitur. Nol, lonjakan dan rentetan konstan dipertahankan; penanda missing asli ikut fitur. Off-grid Way Urang yang diaudit sebelumnya tidak memiliki SO₂; tidak dibulatkan.

Origin train terakhir yang targetnya memasuki validasi dipurge. Target selalu pengamatan SO₂ asli; target kosong tidak diisi. Persistence memakai SO₂ asli terakhir pada atau sebelum origin, termasuk donor lebih lama ketika origin kosong. Usia donor dalam slot disimpan untuk setiap prediksi.

## Distribusi dan seleksi sampel

| Fold | Bagian | Target asli teramati | Minimum SO2 | Median SO2 | Maksimum SO2 | Target lebih 2.14 |
|---|---|---|---|---|---|---|
| Fold_01 | train | 3924 | 0.92 | 45.12 | 80.71 | 2833 |
| Fold_01 | validation | 4890 | 1.07 | 43.85 | 47.81 | 2456 |
| Fold_02 | train | 8814 | 0.92 | 44.82 | 80.71 | 5289 |
| Fold_02 | validation | 4969 | 1.04 | 1.56 | 2.14 | 0 |
| Fold_03 | train | 13783 | 0.92 | 1.66 | 80.71 | 5289 |
| Fold_03 | validation | 4969 | 1.08 | 1.56 | 2.04 | 0 |
| Fold_04 | train | 18752 | 0.92 | 1.61 | 80.71 | 5289 |
| Fold_04 | validation | 4969 | 1.09 | 1.56 | 2.06 | 0 |

Angka 2,14 µg/m³ hanya penelusur hasil tahap kelima, bukan batas kualitas atau bahaya. Distribusi bulanan lengkap tersedia dalam workbook. Distribusi train yang benar-benar digunakan setelah kalibrasi, buffer, purge dan irisan:

| Fold | Jumlah slot atau sampel | Minimum SO2 | Median SO2 | Maksimum SO2 | Target lebih 2.14 |
|---|---|---|---|---|---|
| Fold_01 | 1379 | 43.53 | 45.40 | 47.43 | 1379 |
| Fold_02 | 6190 | 1.07 | 44.85 | 47.81 | 3786 |
| Fold_03 | 11154 | 1.04 | 1.64 | 47.81 | 3786 |
| Fold_04 | 16086 | 1.04 | 1.61 | 47.81 | 3786 |

| Fold | Pipeline | Seluruh origin train | Setelah kalibrasi buffer purge | Target train asli | Setelah fitur finite mandiri | Final train bersama |
|---|---|---|---|---|---|---|
| Fold_01 | P0_median | 5760 | 1391 | 1385 | 1385 | 1379 |
| Fold_01 | P1_ffill_L4 | 5760 | 1391 | 1385 | 1385 | 1379 |
| Fold_01 | P2_knn_k10 | 5760 | 1391 | 1385 | 1379 | 1379 |
| Fold_02 | P0_median | 10732 | 6363 | 6275 | 6275 | 6190 |
| Fold_02 | P1_ffill_L4 | 10732 | 6363 | 6275 | 6214 | 6190 |
| Fold_02 | P2_knn_k10 | 10732 | 6363 | 6275 | 6202 | 6190 |
| Fold_03 | P0_median | 15703 | 11334 | 11244 | 11244 | 11154 |
| Fold_03 | P1_ffill_L4 | 15703 | 11334 | 11244 | 11180 | 11154 |
| Fold_03 | P2_knn_k10 | 15703 | 11334 | 11244 | 11169 | 11154 |
| Fold_04 | P0_median | 20674 | 16305 | 16213 | 16213 | 16086 |
| Fold_04 | P1_ffill_L4 | 20674 | 16305 | 16213 | 16114 | 16086 |
| Fold_04 | P2_knn_k10 | 20674 | 16305 | 16213 | 16136 | 16086 |

Sampel layak suatu pipeline yang dikeluarkan hanya untuk menyamakan train:

| Fold | Pipeline | Layak mandiri | Aktual bersama | Keluar karena irisan | Minimum SO2 | Median SO2 | Maksimum SO2 | Target lebih 2.14 |
|---|---|---|---|---|---|---|---|---|
| Fold_01 | P0_median | 1385 | 1379 | 6 | 44.55 | 45.20 | 46.28 | 6 |
| Fold_01 | P1_ffill_L4 | 1385 | 1379 | 6 | 44.55 | 45.20 | 46.28 | 6 |
| Fold_01 | P2_knn_k10 | 1379 | 1379 | 0 | — | — | — | 0 |
| Fold_02 | P0_median | 6275 | 6190 | 85 | 1.16 | 44.87 | 46.68 | 55 |
| Fold_02 | P1_ffill_L4 | 6214 | 6190 | 24 | 1.28 | 1.62 | 46.28 | 6 |
| Fold_02 | P2_knn_k10 | 6202 | 6190 | 12 | 1.16 | 1.69 | 1.69 | 0 |
| Fold_03 | P0_median | 11244 | 11154 | 90 | 1.16 | 44.68 | 46.68 | 55 |
| Fold_03 | P1_ffill_L4 | 11180 | 11154 | 26 | 1.28 | 1.60 | 46.28 | 6 |
| Fold_03 | P2_knn_k10 | 11169 | 11154 | 15 | 1.16 | 1.69 | 1.70 | 0 |
| Fold_04 | P0_median | 16213 | 16086 | 127 | 1.16 | 1.73 | 46.68 | 55 |
| Fold_04 | P1_ffill_L4 | 16114 | 16086 | 28 | 1.28 | 1.60 | 46.28 | 6 |
| Fold_04 | P2_knn_k10 | 16136 | 16086 | 50 | 1.16 | 1.58 | 1.88 | 0 |

Workbook memisahkan kehilangan karena fitur nonfinite dari kehilangan tambahan akibat irisan, beserta distribusi target dan bulan. Penanda SO₂ historis/kovariat kosong dapat tumpang tindih, sehingga hitungannya tidak boleh dijumlahkan sebagai kategori eksklusif. Irisan dapat menyeleksi periode; kesamaan jumlah train tidak menjamin representasi distribusi penuh.

## Skor pada timestamp bersama

| Fold | Pipeline | Jumlah prediksi | Denominator target asli | Cakupan persen | MAE | RMSE | Bias |
|---|---|---|---|---|---|---|---|
| Fold_01 | P0_median | 4811 | 4890 | 98.384 | 16.491830 | 24.456627 | 15.916489 |
| Fold_01 | P1_ffill_L4 | 4811 | 4890 | 98.384 | 16.599086 | 24.601243 | 16.058483 |
| Fold_01 | P2_knn_k10 | 4811 | 4890 | 98.384 | 16.479999 | 24.436651 | 15.907770 |
| Fold_01 | Persistence | 4811 | 4890 | 98.384 | 0.429472 | 0.887766 | 0.009588 |
| Fold_02 | P0_median | 4964 | 4969 | 99.899 | 0.143361 | 0.288234 | -0.027250 |
| Fold_02 | P1_ffill_L4 | 4964 | 4969 | 99.899 | 0.127656 | 0.166156 | 0.008759 |
| Fold_02 | P2_knn_k10 | 4964 | 4969 | 99.899 | 0.127065 | 0.164926 | 0.009504 |
| Fold_02 | Persistence | 4964 | 4969 | 99.899 | 0.142975 | 0.183708 | 0.000006 |
| Fold_03 | P0_median | 4932 | 4969 | 99.255 | 0.119055 | 0.196441 | -0.008869 |
| Fold_03 | P1_ffill_L4 | 4932 | 4969 | 99.255 | 0.119231 | 0.153836 | -0.003029 |
| Fold_03 | P2_knn_k10 | 4932 | 4969 | 99.255 | 0.119314 | 0.153905 | -0.002855 |
| Fold_03 | Persistence | 4932 | 4969 | 99.255 | 0.139511 | 0.178511 | 0.000189 |
| Fold_04 | P0_median | 4931 | 4969 | 99.235 | 0.101352 | 0.163822 | 0.003558 |
| Fold_04 | P1_ffill_L4 | 4931 | 4969 | 99.235 | 0.099893 | 0.128587 | 0.000113 |
| Fold_04 | P2_knn_k10 | 4931 | 4969 | 99.235 | 0.099866 | 0.128568 | 0.000090 |
| Fold_04 | Persistence | 4931 | 4969 | 99.235 | 0.115818 | 0.148959 | -0.000166 |

Bias = prediksi − pengamatan. MAE, RMSE dan bias memakai N prediksi finite pada domain bersangkutan. Cakupan (%) = N / seluruh target asli teramati pada jendela validasi × 100. Domain bersama memakai satu himpunan timestamp identik untuk ketiga pipeline dan persistence. Cakupan mandiri tetap dilaporkan terpisah terhadap denominator target asli yang sama.

| Fold | Pipeline | Jumlah prediksi | Cakupan persen | MAE | RMSE |
|---|---|---|---|---|---|
| Fold_01 | P0_median | 4890 | 100.000 | 16.458916 | 24.448670 |
| Fold_01 | P1_ffill_L4 | 4829 | 98.753 | 16.656634 | 24.648912 |
| Fold_01 | P2_knn_k10 | 4823 | 98.630 | 16.543760 | 24.496439 |
| Fold_01 | Persistence | 4890 | 100.000 | 0.430562 | 0.885256 |
| Fold_02 | P0_median | 4969 | 100.000 | 0.151868 | 0.513461 |
| Fold_02 | P1_ffill_L4 | 4966 | 99.940 | 0.127615 | 0.166124 |
| Fold_02 | P2_knn_k10 | 4967 | 99.960 | 0.127099 | 0.164965 |
| Fold_02 | Persistence | 4969 | 100.000 | 0.142954 | 0.183682 |
| Fold_03 | P0_median | 4969 | 100.000 | 0.119728 | 0.197273 |
| Fold_03 | P1_ffill_L4 | 4934 | 99.296 | 0.119257 | 0.153849 |
| Fold_03 | P2_knn_k10 | 4967 | 99.960 | 0.119543 | 0.154215 |
| Fold_03 | Persistence | 4969 | 100.000 | 0.139453 | 0.178418 |
| Fold_04 | P0_median | 4969 | 100.000 | 0.101526 | 0.163840 |
| Fold_04 | P1_ffill_L4 | 4933 | 99.276 | 0.099915 | 0.128604 |
| Fold_04 | P2_knn_k10 | 4967 | 99.960 | 0.100020 | 0.128809 |
| Fold_04 | Persistence | 4969 | 100.000 | 0.115935 | 0.149092 |

### Perubahan terhadap persistence pada timestamp identik

| Fold | Pipeline | Delta MAE vs persistence | Delta RMSE vs persistence | Perbaikan MAE persen | Perbaikan RMSE persen |
|---|---|---|---|---|---|
| Fold_01 | P0_median | 16.062358 | 23.568860 | -3740.024014 | -2654.849130 |
| Fold_01 | P1_ffill_L4 | 16.169614 | 23.713477 | -3764.997968 | -2671.139063 |
| Fold_01 | P2_knn_k10 | 16.050527 | 23.548884 | -3737.269392 | -2652.598946 |
| Fold_02 | P0_median | 0.000386 | 0.104526 | -0.269853 | -56.897562 |
| Fold_02 | P1_ffill_L4 | -0.015319 | -0.017552 | 10.714546 | 9.554215 |
| Fold_02 | P2_knn_k10 | -0.015911 | -0.018782 | 11.128371 | 10.223790 |
| Fold_03 | P0_median | -0.020456 | 0.017930 | 14.662544 | -10.044224 |
| Fold_03 | P1_ffill_L4 | -0.020281 | -0.024675 | 14.536849 | 13.822813 |
| Fold_03 | P2_knn_k10 | -0.020197 | -0.024606 | 14.477049 | 13.784178 |
| Fold_04 | P0_median | -0.014466 | 0.014863 | 12.490493 | -9.978110 |
| Fold_04 | P1_ffill_L4 | -0.015925 | -0.020372 | 13.749866 | 13.676151 |
| Fold_04 | P2_knn_k10 | -0.015952 | -0.020390 | 13.773395 | 13.688700 |

Delta negatif berarti error lebih rendah. Persentase perbaikan = (error persistence − error pipeline) / error persistence × 100; negatif berarti memburuk.

### Gabungan prediksi validasi unik

| Pipeline | Jumlah prediksi | Denominator target asli | Cakupan persen | MAE | RMSE | Bias |
|---|---|---|---|---|---|---|
| P0_median | 19638 | 19797 | 99.197 | 4.131825 | 12.106569 | 3.891067 |
| P1_ffill_L4 | 19638 | 19797 | 99.197 | 4.153809 | 12.177303 | 3.935557 |
| P2_knn_k10 | 19638 | 19797 | 99.197 | 4.124500 | 12.095837 | 3.898860 |
| Persistence | 19638 | 19797 | 99.197 | 0.205474 | 0.463880 | 0.002356 |

Metrik gabungan dihitung ulang dari semua error OOF pada timestamp unik: MAE=Σ|e|/N, RMSE=√(Σe²/N), bias=Σe/N. RMSE gabungan bukan rata-rata RMSE fold. Rata-rata dan simpangan baku sampel antar empat fold (ddof=1) adalah ringkasan deskriptif terpisah:

| Pipeline | MAE rata-rata fold | MAE SD fold | RMSE rata-rata fold | RMSE SD fold | Bias rata-rata fold | Bias SD fold |
|---|---|---|---|---|---|---|
| P0_median | 4.213900 | 8.185305 | 6.276281 | 12.120345 | 3.970982 | 7.963681 |
| P1_ffill_L4 | 4.236467 | 8.241754 | 6.262456 | 12.225869 | 4.016082 | 8.028269 |
| P2_knn_k10 | 4.206561 | 8.182300 | 6.221012 | 12.143768 | 3.978627 | 7.952764 |
| Persistence | 0.206944 | 0.148842 | 0.349736 | 0.359013 | 0.002404 | 0.004792 |

### Peringkat deskriptif

| Fold | Pipeline | Peringkat MAE | Peringkat RMSE |
|---|---|---|---|
| Fold_01 | P0_median | 2.000000 | 2.000000 |
| Fold_01 | P1_ffill_L4 | 3.000000 | 3.000000 |
| Fold_01 | P2_knn_k10 | 1.000000 | 1.000000 |
| Fold_02 | P0_median | 3.000000 | 3.000000 |
| Fold_02 | P1_ffill_L4 | 2.000000 | 2.000000 |
| Fold_02 | P2_knn_k10 | 1.000000 | 1.000000 |
| Fold_03 | P0_median | 1.000000 | 3.000000 |
| Fold_03 | P1_ffill_L4 | 2.000000 | 1.000000 |
| Fold_03 | P2_knn_k10 | 3.000000 | 2.000000 |
| Fold_04 | P0_median | 3.000000 | 3.000000 |
| Fold_04 | P1_ffill_L4 | 2.000000 | 2.000000 |
| Fold_04 | P2_knn_k10 | 1.000000 | 1.000000 |

### Per bulan

| Bulan target | Pipeline | Jumlah prediksi | Cakupan persen | MAE | RMSE | Bias |
|---|---|---|---|---|---|---|
| 2025-05 | P0_median | 1364 | 96.532 | 0.496319 | 0.612869 | -0.012192 |
| 2025-06 | P0_median | 1440 | 100.000 | 9.834487 | 18.808382 | 9.405593 |
| 2025-07 | P0_median | 1455 | 98.112 | 32.315470 | 34.449619 | 31.709672 |
| 2025-08 | P0_median | 1484 | 99.865 | 11.850038 | 20.781985 | 11.391968 |
| 2025-09 | P0_median | 1436 | 99.861 | 0.136841 | 0.433777 | 0.002007 |
| 2025-10 | P0_median | 1486 | 99.866 | 0.153061 | 0.196123 | -0.032788 |
| 2025-11 | P0_median | 1439 | 99.931 | 0.155747 | 0.226871 | -0.059568 |
| 2025-12 | P0_median | 1484 | 99.731 | 0.120126 | 0.159410 | 0.001043 |
| 2026-01 | P0_median | 1482 | 99.597 | 0.113235 | 0.148004 | -0.009478 |
| 2026-02 | P0_median | 1317 | 98.137 | 0.123365 | 0.280483 | -0.022256 |
| 2026-03 | P0_median | 1479 | 99.395 | 0.099367 | 0.130623 | 0.001083 |
| 2026-04 | P0_median | 1417 | 98.540 | 0.107164 | 0.229477 | -0.004738 |
| 2026-05 | P0_median | 1484 | 99.731 | 0.102637 | 0.134829 | 0.015319 |
| 2026-06 | P0_median | 871 | 99.316 | 0.103244 | 0.130051 | -0.002487 |
| 2025-05 | P1_ffill_L4 | 1364 | 96.532 | 0.496314 | 0.612933 | -0.012252 |
| 2025-06 | P1_ffill_L4 | 1440 | 100.000 | 9.896901 | 18.919501 | 9.477615 |
| 2025-07 | P1_ffill_L4 | 1455 | 98.112 | 32.536764 | 34.658911 | 31.998630 |
| 2025-08 | P1_ffill_L4 | 1484 | 99.865 | 11.920342 | 20.895974 | 11.511558 |
| 2025-09 | P1_ffill_L4 | 1436 | 99.861 | 0.112673 | 0.147897 | 0.017794 |
| 2025-10 | P1_ffill_L4 | 1486 | 99.866 | 0.144766 | 0.182754 | 0.016083 |
| 2025-11 | P1_ffill_L4 | 1439 | 99.931 | 0.136405 | 0.178062 | -0.013847 |
| 2025-12 | P1_ffill_L4 | 1484 | 99.731 | 0.124889 | 0.162179 | 0.002515 |
| 2026-01 | P1_ffill_L4 | 1482 | 99.597 | 0.114609 | 0.147399 | -0.004568 |
| 2026-02 | P1_ffill_L4 | 1317 | 98.137 | 0.112568 | 0.144985 | -0.008102 |
| 2026-03 | P1_ffill_L4 | 1479 | 99.395 | 0.103410 | 0.133369 | -0.002985 |
| 2026-04 | P1_ffill_L4 | 1417 | 98.540 | 0.097743 | 0.126029 | -0.000138 |
| 2026-05 | P1_ffill_L4 | 1484 | 99.731 | 0.104431 | 0.136779 | 0.004199 |
| 2026-06 | P1_ffill_L4 | 871 | 99.316 | 0.103505 | 0.129809 | -0.003665 |
| 2025-05 | P2_knn_k10 | 1364 | 96.532 | 0.496322 | 0.612798 | -0.012175 |
| 2025-06 | P2_knn_k10 | 1440 | 100.000 | 9.825047 | 18.788477 | 9.397263 |
| 2025-07 | P2_knn_k10 | 1455 | 98.112 | 32.295598 | 34.425411 | 31.696282 |
| 2025-08 | P2_knn_k10 | 1484 | 99.865 | 11.840416 | 20.762600 | 11.397174 |
| 2025-09 | P2_knn_k10 | 1436 | 99.861 | 0.112778 | 0.148043 | 0.017698 |
| 2025-10 | P2_knn_k10 | 1486 | 99.866 | 0.144712 | 0.182658 | 0.015550 |
| 2025-11 | P2_knn_k10 | 1439 | 99.931 | 0.134341 | 0.174080 | -0.010498 |
| 2025-12 | P2_knn_k10 | 1484 | 99.731 | 0.124888 | 0.162178 | 0.002551 |
| 2026-01 | P2_knn_k10 | 1482 | 99.597 | 0.114609 | 0.147400 | -0.004481 |
| 2026-02 | P2_knn_k10 | 1317 | 98.137 | 0.112882 | 0.145258 | -0.007625 |
| 2026-03 | P2_knn_k10 | 1479 | 99.395 | 0.103415 | 0.133372 | -0.002981 |
| 2026-04 | P2_knn_k10 | 1417 | 98.540 | 0.097659 | 0.125966 | -0.000121 |
| 2026-05 | P2_knn_k10 | 1484 | 99.731 | 0.104420 | 0.136775 | 0.004139 |
| 2026-06 | P2_knn_k10 | 871 | 99.316 | 0.103501 | 0.129806 | -0.003684 |
| 2025-05 | Persistence | 1364 | 96.532 | 0.691100 | 0.861007 | 0.001085 |
| 2025-06 | Persistence | 1440 | 100.000 | 0.588215 | 1.372660 | 0.030632 |
| 2025-07 | Persistence | 1455 | 98.112 | 0.151333 | 0.198839 | 0.000254 |
| 2025-08 | Persistence | 1484 | 99.865 | 0.115991 | 0.150994 | 0.000088 |
| 2025-09 | Persistence | 1436 | 99.861 | 0.127834 | 0.167140 | 0.000146 |
| 2025-10 | Persistence | 1486 | 99.866 | 0.166252 | 0.208016 | 0.000007 |
| 2025-11 | Persistence | 1439 | 99.931 | 0.147054 | 0.187359 | -0.000035 |
| 2025-12 | Persistence | 1484 | 99.731 | 0.148127 | 0.189132 | 0.000094 |
| 2026-01 | Persistence | 1482 | 99.597 | 0.132665 | 0.169950 | 0.000223 |
| 2026-02 | Persistence | 1317 | 98.137 | 0.129590 | 0.166103 | 0.000554 |
| 2026-03 | Persistence | 1479 | 99.395 | 0.122096 | 0.156909 | -0.000487 |
| 2026-04 | Persistence | 1417 | 98.540 | 0.111186 | 0.143258 | -0.000247 |
| 2026-05 | Persistence | 1484 | 99.731 | 0.122554 | 0.159373 | -0.000371 |
| 2026-06 | Persistence | 871 | 99.316 | 0.119116 | 0.150296 | 0.000494 |

Metrik per bulan per fold serta distribusi target bulanan disimpan dalam workbook. Denominator bulanan adalah target asli pada bulan/jendela terkait; bulan yang melintasi batas fold memiliki model train berbeda dan juga ditampilkan secara terpisah.

## Waktu komputasi

| Fold | Pembentukan histori semua pipeline detik | Fit imputer median scaler KNN detik | Transform KNN detik |
|---|---|---|---|
| Fold_01 | 0.271963 | 0.030529 | 0.026163 |
| Fold_02 | 1.024618 | 0.048865 | 0.810194 |
| Fold_03 | 2.654292 | 0.095599 | 2.344183 |
| Fold_04 | 6.590626 | 0.267075 | 5.995754 |

| Fold | Pipeline | Scaler fitur fit detik | Ridge fit detik | Predict detik |
|---|---|---|---|---|
| Fold_01 | P0_median | 0.001493 | 0.003114 | 0.002313 |
| Fold_01 | P1_ffill_L4 | 0.001675 | 0.003886 | 0.002680 |
| Fold_01 | P2_knn_k10 | 0.002430 | 0.006268 | 0.003291 |
| Fold_02 | P0_median | 0.011041 | 0.019928 | 0.004597 |
| Fold_02 | P1_ffill_L4 | 0.014609 | 0.025017 | 0.004330 |
| Fold_02 | P2_knn_k10 | 0.011453 | 0.019500 | 0.003509 |
| Fold_03 | P0_median | 0.025442 | 0.040021 | 0.004550 |
| Fold_03 | P1_ffill_L4 | 0.025762 | 0.040466 | 0.005399 |
| Fold_03 | P2_knn_k10 | 0.020657 | 0.029900 | 0.006866 |
| Fold_04 | P0_median | 0.021280 | 0.050143 | 0.004079 |
| Fold_04 | P1_ffill_L4 | 0.023910 | 0.040742 | 0.003004 |
| Fold_04 | P2_knn_k10 | 0.017319 | 0.043292 | 0.003179 |

Waktu dinding diukur pada mesin ini, satu thread; histori gabungan mencakup ketiga pipeline dan penyusunan fitur, bukan waktu tiap pipeline yang dapat dijumlahkan ulang secara bebas. Fit median/scaler/KNN dan transform KNN memiliki pengukuran prefix tersendiri; waktu ekspor/render tidak termasuk waktu model.

## Kesimpulan terbatas

- P0_median: mengungguli persistence pada MAE dan RMSE sekaligus di 0/4 fold; peringkat MAE antar-fold 2, 3, 1, 3.
- P1_ffill_L4: mengungguli persistence pada MAE dan RMSE sekaligus di 3/4 fold; peringkat MAE antar-fold 3, 2, 2, 2.
- P2_knn_k10: mengungguli persistence pada MAE dan RMSE sekaligus di 3/4 fold; peringkat MAE antar-fold 1, 1, 3, 1.

Fold pertama penting: setelah kalibrasi dan buffer, hanya 1379 target train bersama tersisa pada rentang 43.53–47.43 µg/m³. Ini cukup untuk gate komputasional, tetapi tidak mencakup rezim konsentrasi rendah yang kemudian muncul di validasi. Ketiga Ridge mengalami bias positif besar (sekitar 16 µg/m³) dan MAE sekitar 16,5, sedangkan persistence sekitar 0,429. Ini kegagalan generalisasi yang tampak pada konfigurasi tetap, bukan alasan menggeser fold atau menghapus pengamatan.

KNN memiliki MAE dan RMSE terendah di antara preprocessing pada fold 1, 2 dan 4; pada fold 3 median memiliki MAE terendah sementara ffill memiliki RMSE terendah. Dengan demikian peringkat tidak sepenuhnya stabil dan kesetaraan hasil pada satu validasi lama tidak berlaku umum. Selisih ffill–KNN pada periode rendah sangat kecil dibanding perubahan error antar-rezim. Ffill dan KNN mengungguli persistence pada kedua metrik di fold 2–4, tetapi gagal pada fold 1; median tidak konsisten pada RMSE. Metrik gabungan didominasi kegagalan fold pertama dan seluruh Ridge tertinggal dari persistence.

Prediksi negatif pada irisan OOF (tanpa clipping): P0_median=112, P1_ffill_L4=101, P2_knn_k10=106, Persistence=0. Nilai ini perlu diperiksa sebagai kegagalan keluaran model, bukan diubah diam-diam. Tabel strata SO₂ >2,14 dan ≤2,14 memisahkan skor pada konsentrasi berbeda; tiga fold terakhir tidak memiliki target >2,14 sehingga tidak mendukung klaim evaluasi konsentrasi tinggi.

Persistence: donor tersedia pada origin untuk 19786/19797 target asli validasi; usia donor median 0 slot, P95 0, maksimum 75 slot (satu slot=30 menit). Donor lebih lama tetap hanya pengamatan asli sebelum origin; aturan sama seperti tahap 5B.

Urutan preprocessing harus dibaca per fold dan per metrik; selisih rata-rata tidak membuktikan signifikansi statistik. Distribusi train dan validasi berubah kuat: konsentrasi lebih tinggi tersedia pada validasi pertama, sedangkan tiga jendela berikutnya sempit/rendah. Perubahan skor bertepatan dengan pergeseran distribusi dan seleksi sampel, tetapi bukan bukti penyebab fisik ataupun efek preprocessing yang terisolasi. Tidak boleh menggeneralisasi kemampuan konsentrasi tinggi dari jendela rendah.

Ffill L4 dan KNN k10 layak diprioritaskan sebagai kandidat untuk verifikasi terkontrol dengan **satu** model deep learning berikutnya; median tetap sebagai kontrol sederhana dan persistence sebagai acuan wajib. Bawa hasil per periode, cakupan dan biaya KNN ke verifikasi tersebut; belum memilih preprocessing final atau algoritma deep learning terbaik. Parameter dan fold tidak ditata ulang untuk mengejar skor. Rancangan keterlambatan pengukuran, arti penanda kualitas dan dokumentasi sensor masih memerlukan konfirmasi penyedia data.

## Artefak dan integritas

95 sumber/artefak lama diverifikasi SHA256 identik sebelum–sesudah. Hash konfigurasi fold juga tetap identik. Hash seluruh berkas hanya verifikasi byte, bukan pembacaan nilai test. Daftar lengkap terdapat pada sheet Integritas_SHA256.

- `backtesting_preprocessing.xlsx`: tabel skor, distribusi, gate, alur sampel, koefisien, prediksi dan integritas.
- `backtesting_way_urang/fold_config_locked.json`: parameter dan kalender terkunci.
- `backtesting_way_urang/Fold_*/`: histori, fitur, timestamp train/evaluasi, prediksi serta gate tiap fold.
- `backtesting_way_urang/prediksi_oof_development.csv`: seluruh target asli validasi dan prediksi; kosong berarti pipeline tidak dapat memberi prediksi.
- `figures/backtesting_way_urang/`: metrik/cakupan fold, distribusi dan ukuran train, MAE bulanan.

![Metrik dan cakupan](figures/backtesting_way_urang/metrik_dan_cakupan_fold.png)

![Distribusi](figures/backtesting_way_urang/distribusi_dan_train_fold.png)

![MAE bulanan](figures/backtesting_way_urang/mae_bulanan_oof.png)

![Detail fold 2–4](figures/backtesting_way_urang/metrik_fold_2_4.png)
