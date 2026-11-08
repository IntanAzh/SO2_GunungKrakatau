# Pilot imputasi SO₂ — Lampung Selatan Way Urang

**Pilot sementara; bukan pemilihan stasiun final dan bukan eksperimen forecasting.** P0–P3 dibandingkan pada validasi saja. Pengujian akhir tidak ditransformasi, tidak diberi skor, dan tidak digunakan untuk memilih parameter. Data tersebut telah tercakup audit sebelumnya, sehingga tidak disebut sama sekali belum pernah dilihat.

## 1. Data, kualitas dan keputusan yang belum ditetapkan

- Sumber dibaca tanpa perubahan. Salinan seluruh baris, grid audit dan penanda kualitas ada di `pilot_way_urang/`. Nol, lonjakan, rentetan konstan dan ekstrem tidak dihapus. Penanda audit tetap disimpan; tidak digunakan sebagai label sensor rusak.
- Terdapat **satu baris off-grid: 2025-05-22 19:01:00, SO₂ kosong**. Baris dipertahankan pada `offgrid_asli.csv` dan tidak dibulatkan, diagregasi atau menjadi truth.
- Makna kode/nilai 0.01, validitas sensor, perubahan pola, dan latensi belum dikonfirmasi. Pengamatan asli teramati dipakai sebagai referensi rekonstruksi, bukan sertifikasi nilai valid.
- **Asumsi ketersediaan yang memengaruhi hasil:** kovariat bertimestamp paling lambat t−30 menit diasumsikan telah tiba ketika SO₂ pada t direkonstruksi. Timestamp asli tidak membuktikan latensi. Jika latensi aktual lebih panjang, fitur perlu digeser lagi dan hasil ini harus diuji ulang.
- Skenario pilot menyembunyikan SO₂ saja dan mempertahankan kovariat lag yang asli. Ini rekonstruksi kondisional dengan kovariat tersedia, belum meniru seluruh sensor/baris hilang bersamaan. Tidak ada nilai gap alami dijadikan truth atau diisi sebagai produk akhir.

## 2. Pemisahan waktu 70/15/15

N slot grid reguler dibagi `floor(0.70*N)` untuk train, hingga `floor(0.85*N)` untuk validasi, dan sisanya test. Ini pembagian durasi slot waktu, bukan pengacakan jumlah pengamatan nonkosong. Selisih pembulatan hanya satuan slot.

| Split | Awal | Akhir | Jumlah slot | Jumlah baris sumber | SO2 nonkosong asli | Status |
|---|---|---|---|---|---|---|
| train | 2025-01-01 00:00:00 | 2026-03-16 23:00:00 | 21119 | 20915 | 19197 | Digunakan sesuai kontrak |
| validation | 2026-03-16 23:30:00 | 2026-06-19 06:00:00 | 4526 | 4525 | 4524 | Digunakan sesuai kontrak |
| test | 2026-06-19 06:30:00 | 2026-09-21 13:00:00 | 4526 | 4525 | 4525 | Hanya inventaris, tidak dievaluasi |

Semua median, mean/skala, daftar fitur, donor dan model imputasi berasal dari train. Nilai train lebih awal daripada seluruh query validasi. Bagian test hanya digunakan untuk inventaris pada tabel di atas; tidak ada skor test atau transform test.

## 3. Distribusi gap train dan masking

Distribusi panjang gap dihitung ulang pada **train saja**, termasuk gap awal. Kategori blok bukan batas aman imputasi. Panjang blok yang diuji berasal dari gap 4 slot dan kuantil empiris nearest `{0,0.5,0.9,1}` gap panjang >4 slot.

| Jumlah slot | Durasi jam | Jumlah kejadian |
|---|---|---|
| 1 | 0.50 | 15 |
| 4 | 2.00 | 1 |
| 71 | 35.50 | 2 |
| 75 | 37.50 | 1 |
| 94 | 47.00 | 1 |
| 97 | 48.50 | 1 |
| 99 | 49.50 | 1 |
| 111 | 55.50 | 1 |
| 120 | 60.00 | 1 |
| 184 | 92.00 | 1 |
| 981 | 490.50 | 1 |

Panjang blok: **[4, 71, 97, 184, 981] slot**, atau **[2.0, 35.5, 48.5, 92.0, 490.5] jam**. Tiga seed tetap: **11, 23, 47**. Titik terpisah memakai 10% SO₂ teramati validasi (dibulatkan ke bawah), tanpa pasangan mask bertetangga. Blok memakai `max(1,min(30,floor(0.10*N_teramati/L)))` kejadian, tersedia truth penuh dan buffer satu slot teramati pada kedua sisi, tanpa overlap dalam skenario. Untuk blok 981 slot, satu blok saja melebihi 10%; cakupannya dilaporkan apa adanya.

| Seed | Pola | Panjang slot | Durasi blok jam | Jumlah kejadian | Jumlah nilai disembunyikan | Persen dari SO2 teramati validasi |
|---|---|---|---|---|---|---|
| 11 | titik | 1 | 0.50 | 452 | 452 | 9.991 |
| 11 | blok | 4 | 2.00 | 30 | 120 | 2.653 |
| 11 | blok | 71 | 35.50 | 6 | 426 | 9.416 |
| 11 | blok | 97 | 48.50 | 4 | 388 | 8.576 |
| 11 | blok | 184 | 92.00 | 2 | 368 | 8.134 |
| 11 | blok | 981 | 490.50 | 1 | 981 | 21.684 |
| 23 | titik | 1 | 0.50 | 452 | 452 | 9.991 |
| 23 | blok | 4 | 2.00 | 30 | 120 | 2.653 |
| 23 | blok | 71 | 35.50 | 6 | 426 | 9.416 |
| 23 | blok | 97 | 48.50 | 4 | 388 | 8.576 |
| 23 | blok | 184 | 92.00 | 2 | 368 | 8.134 |
| 23 | blok | 981 | 490.50 | 1 | 981 | 21.684 |
| 47 | titik | 1 | 0.50 | 452 | 452 | 9.991 |
| 47 | blok | 4 | 2.00 | 30 | 120 | 2.653 |
| 47 | blok | 71 | 35.50 | 6 | 426 | 9.416 |
| 47 | blok | 97 | 48.50 | 4 | 388 | 8.576 |
| 47 | blok | 184 | 92.00 | 2 | 368 | 8.134 |
| 47 | blok | 981 | 490.50 | 1 | 981 | 21.684 |

- Lokasi mask sama untuk seluruh konfigurasi pada suatu skenario. Mask antar-seed atau antar-skenario boleh overlap; SD tiga seed bukan interval kepercayaan independen.
- `truth_penilaian_saja.csv` menyimpan nilai asli terpisah. Query P2/P3 selalu SO₂ NaN; donor hanya train. State P1 menerima SO₂ validasi yang tidak dimask hanya setelah waktunya terlewati, dan tidak pernah memakai nilai mask atau nilai imputasi untuk mereset donor.
- Blok besar diletakkan hanya pada segmen yang truth-nya lengkap. Keterwakilan terhadap missing alami panjang/serentak masih terbatas; buffer teramati juga merupakan seleksi skenario dan tidak membuktikan kondisi gap awal tanpa donor.

## 4. Implementasi metode

- **P0:** median SO₂ asli train = **1.61 µg/m³**, tetap untuk semua query.
- **P1:** umur donor maksimum **[1, 4, 71, 97, 184, 981] slot**. Donor adalah pengamatan asli terakhir sebelum slot rekonstruksi. Jika melampaui batas, hasil tetap tidak terisi; panjang gap akhir tidak dilihat untuk menentukan perlakuan slot awal.
- **P2:** `KNNImputer`, k=3/5/10, `weights=distance`, `metric=nan_euclidean`. Donor memiliki SO₂ asli finite dan hanya berasal dari train. SO₂ query seluruhnya NaN sehingga jarak dibentuk hanya dari kovariat lag yang overlap, tanpa waktu mentah atau SO₂ validasi. Minimal dua kovariat asli harus tersedia pada query. Jarak bawaan memakai jumlah overlap yang bisa berbeda antar-pasangan. Query tanpa jarak terdefinisi ditolak, bukan diberi fallback mean tersembunyi.
- **P3:** `IterativeImputer(estimator=BayesianRidge(), max_iter=20, tol=0.001, initial_strategy=median, imputation_order=ascending, random_state=42, sample_posterior=False, skip_complete=False)`. Fit hanya train. Kovariat kosong diperkirakan sementara dalam iterasi berdasarkan model train pada baris query itu, tanpa memakai baris waktu berikutnya. Satu keluaran ini adalah imputasi tunggal deterministik, bukan multiple imputation lengkap.
- **Fitur P2 dan P3 sama:** PM10, PM2.5, CO, O3, NO2, Kec.Angin, Kelembaban, Suhu, Tek.Udara, Sol.Rad, Curah Hujan. Semuanya dilag satu slot; tidak ada SO₂ masa depan, backward fill, interpolasi dua arah, rolling terpusat atau fitur waktu numerik mentah. Arah Angin dikeluarkan dari keduanya agar tidak diperlakukan sebagai variabel linear. Pemilihan fitur memakai dukungan dan variasi train saja, tanpa seleksi berdasarkan korelasi/skor validasi.
- **Scaling:** satu `StandardScaler` fit pada matriks train dengan NaN tetap NaN. Transform memakai mean/skala train, dan SO₂ hasil dikembalikan ke µg/m³. Standardisasi membuat variabel besar seperti CO/tekanan tidak otomatis mendominasi jarak KNN; ekstrem train tetap dapat memengaruhi skala. Tidak ada pembandingan scaler lain pada pilot ini, sehingga efek scaling belum diisolasi secara eksperimen.
- Kolom kosong/konstan train tidak dipakai sebagai kovariat informatif. NaN kovariat tidak diisi dengan nol. P2 memakai overlap; P3 memakai model train. Keduanya ditolak bila kurang dari dua kovariat asli tersedia. Tidak ada clipping hasil.
Implementasi mengacu dokumentasi resmi [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html), [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html), dan [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

| Fitur | Lag slot | Nonkosong train | Train teramati persen | Nilai unik train | Dipakai P2 dan P3 |
|---|---|---|---|---|---|
| PM10 | 1 | 19187 | 90.852 | 3286 | True |
| PM2.5 | 1 | 19183 | 90.833 | 2671 | True |
| CO | 1 | 19163 | 90.738 | 6332 | True |
| O3 | 1 | 19195 | 90.890 | 766 | True |
| NO2 | 1 | 19196 | 90.894 | 843 | True |
| Kec.Angin | 1 | 19196 | 90.894 | 1104 | True |
| Kelembaban | 1 | 19196 | 90.894 | 3427 | True |
| Suhu | 1 | 19196 | 90.894 | 1068 | True |
| Tek.Udara | 1 | 19196 | 90.894 | 4319 | True |
| Sol.Rad | 1 | 17291 | 81.874 | 6668 | True |
| Curah Hujan | 1 | 12689 | 60.083 | 50 | True |

## 5. Hasil validasi

`e = prediksi − pengamatan asli`. MAE=mean(|e|), RMSE=sqrt(mean(e²)), bias=mean(e), dalam µg/m³. Cakupan (%)=100×prediksi finite/jumlah nilai sengaja disembunyikan. N=0 menghasilkan skor kosong, bukan nol. Tidak ada missing alami atau nilai test masuk denominator skor.
Workbook `Hasil_per_seed` memuat hasil setiap seed, pola, panjang blok dan domain. Tabel berikut berisi mean dan SD sampel antar-seed; bukan skor gabungan seluruh seed.

### Seluruh titik yang dapat diisi masing-masing metode

| Metode | Pola | Panjang slot | MAE rata-rata | MAE SD antar-seed | RMSE rata-rata | Bias rata-rata | Cakupan persen rata-rata | Cakupan persen SD antar-seed |
|---|---|---|---|---|---|---|---|---|
| P0 | titik | 1 | 0.093053 | 0.009054 | 0.117482 | 0.052404 | 100.000 | 0.000 |
| P1_L1 | titik | 1 | 0.118461 | 0.007688 | 0.149752 | 0.003059 | 99.926 | 0.128 |
| P1_L4 | titik | 1 | 0.118532 | 0.007611 | 0.149812 | 0.002898 | 100.000 | 0.000 |
| P1_L71 | titik | 1 | 0.118532 | 0.007611 | 0.149812 | 0.002898 | 100.000 | 0.000 |
| P1_L97 | titik | 1 | 0.118532 | 0.007611 | 0.149812 | 0.002898 | 100.000 | 0.000 |
| P1_L184 | titik | 1 | 0.118532 | 0.007611 | 0.149812 | 0.002898 | 100.000 | 0.000 |
| P1_L981 | titik | 1 | 0.118532 | 0.007611 | 0.149812 | 0.002898 | 100.000 | 0.000 |
| P2_k3 | titik | 1 | 0.100811 | 0.007545 | 0.129685 | 0.001646 | 99.926 | 0.128 |
| P2_k5 | titik | 1 | 0.093959 | 0.007862 | 0.121854 | 0.002098 | 99.926 | 0.128 |
| P2_k10 | titik | 1 | 0.089505 | 0.008491 | 0.116131 | 0.003407 | 99.926 | 0.128 |
| P3 | titik | 1 | 1.443547 | 0.114360 | 2.320072 | 0.560138 | 99.926 | 0.128 |
| P0 | blok | 4 | 0.097639 | 0.004568 | 0.123392 | 0.054417 | 100.000 | 0.000 |
| P1_L1 | blok | 4 | 0.115667 | 0.013013 | 0.150649 | -0.010778 | 25.000 | 0.000 |
| P1_L4 | blok | 4 | 0.120750 | 0.007419 | 0.152666 | -0.002139 | 100.000 | 0.000 |
| P1_L71 | blok | 4 | 0.120750 | 0.007419 | 0.152666 | -0.002139 | 100.000 | 0.000 |
| P1_L97 | blok | 4 | 0.120750 | 0.007419 | 0.152666 | -0.002139 | 100.000 | 0.000 |
| P1_L184 | blok | 4 | 0.120750 | 0.007419 | 0.152666 | -0.002139 | 100.000 | 0.000 |
| P1_L981 | blok | 4 | 0.120750 | 0.007419 | 0.152666 | -0.002139 | 100.000 | 0.000 |
| P2_k3 | blok | 4 | 0.099920 | 0.002875 | 0.129550 | 0.010146 | 100.000 | 0.000 |
| P2_k5 | blok | 4 | 0.095950 | 0.002194 | 0.123954 | 0.008364 | 100.000 | 0.000 |
| P2_k10 | blok | 4 | 0.090790 | 0.004581 | 0.117924 | 0.008744 | 100.000 | 0.000 |
| P3 | blok | 4 | 1.365839 | 0.134963 | 1.921252 | 0.519366 | 100.000 | 0.000 |
| P0 | blok | 71 | 0.092660 | 0.004562 | 0.118431 | 0.052551 | 100.000 | 0.000 |
| P1_L1 | blok | 71 | 0.145000 | 0.076884 | 0.179780 | -0.089444 | 1.408 | 0.000 |
| P1_L4 | blok | 71 | 0.130694 | 0.003544 | 0.163126 | -0.055139 | 5.634 | 0.000 |
| P1_L71 | blok | 71 | 0.122543 | 0.014085 | 0.154477 | -0.044671 | 100.000 | 0.000 |
| P1_L97 | blok | 71 | 0.122543 | 0.014085 | 0.154477 | -0.044671 | 100.000 | 0.000 |
| P1_L184 | blok | 71 | 0.122543 | 0.014085 | 0.154477 | -0.044671 | 100.000 | 0.000 |
| P1_L981 | blok | 71 | 0.122543 | 0.014085 | 0.154477 | -0.044671 | 100.000 | 0.000 |
| P2_k3 | blok | 71 | 0.099600 | 0.003909 | 0.130860 | -0.000525 | 100.000 | 0.000 |
| P2_k5 | blok | 71 | 0.092172 | 0.005682 | 0.119848 | 0.000447 | 100.000 | 0.000 |
| P2_k10 | blok | 71 | 0.087201 | 0.004863 | 0.113328 | 0.002513 | 100.000 | 0.000 |
| P3 | blok | 71 | 1.531788 | 0.401604 | 2.263263 | 0.732706 | 100.000 | 0.000 |
| P0 | blok | 97 | 0.089175 | 0.002818 | 0.114216 | 0.051649 | 100.000 | 0.000 |
| P1_L1 | blok | 97 | 0.075833 | 0.070460 | 0.098491 | -0.030833 | 1.031 | 0.000 |
| P1_L4 | blok | 97 | 0.103750 | 0.026272 | 0.130961 | 0.017083 | 4.124 | 0.000 |
| P1_L71 | blok | 97 | 0.094472 | 0.008018 | 0.120861 | 0.020153 | 73.196 | 0.000 |
| P1_L97 | blok | 97 | 0.093875 | 0.008047 | 0.120136 | 0.020816 | 100.000 | 0.000 |
| P1_L184 | blok | 97 | 0.093875 | 0.008047 | 0.120136 | 0.020816 | 100.000 | 0.000 |
| P1_L981 | blok | 97 | 0.093875 | 0.008047 | 0.120136 | 0.020816 | 100.000 | 0.000 |
| P2_k3 | blok | 97 | 0.099414 | 0.001870 | 0.127060 | -0.003225 | 100.000 | 0.000 |
| P2_k5 | blok | 97 | 0.091161 | 0.002843 | 0.117864 | -0.000701 | 100.000 | 0.000 |
| P2_k10 | blok | 97 | 0.085512 | 0.002501 | 0.110952 | -0.000916 | 100.000 | 0.000 |
| P3 | blok | 97 | 1.583851 | 0.167586 | 2.742110 | 0.790615 | 100.000 | 0.000 |
| P0 | blok | 184 | 0.091957 | 0.004451 | 0.118502 | 0.052428 | 100.000 | 0.000 |
| P1_L1 | blok | 184 | 0.073333 | 0.014434 | 0.083535 | -0.050000 | 0.543 | 0.000 |
| P1_L4 | blok | 184 | 0.101667 | 0.007324 | 0.127699 | -0.054167 | 2.174 | 0.000 |
| P1_L71 | blok | 184 | 0.125164 | 0.012219 | 0.154637 | -0.071549 | 38.587 | 0.000 |
| P1_L97 | blok | 184 | 0.124450 | 0.010811 | 0.152104 | -0.073316 | 52.717 | 0.000 |
| P1_L184 | blok | 184 | 0.123225 | 0.013460 | 0.153059 | -0.070906 | 100.000 | 0.000 |
| P1_L981 | blok | 184 | 0.123225 | 0.013460 | 0.153059 | -0.070906 | 100.000 | 0.000 |
| P2_k3 | blok | 184 | 0.097182 | 0.002404 | 0.127165 | 0.000590 | 100.000 | 0.000 |
| P2_k5 | blok | 184 | 0.090745 | 0.002579 | 0.119281 | 0.000945 | 100.000 | 0.000 |
| P2_k10 | blok | 184 | 0.088259 | 0.002453 | 0.115314 | 0.003122 | 100.000 | 0.000 |
| P3 | blok | 184 | 1.287914 | 0.134073 | 2.338125 | 0.613497 | 100.000 | 0.000 |
| P0 | blok | 981 | 0.095739 | 0.002228 | 0.121333 | 0.051417 | 100.000 | 0.000 |
| P1_L1 | blok | 981 | 0.163333 | 0.063509 | 0.163333 | 0.163333 | 0.102 | 0.000 |
| P1_L4 | blok | 981 | 0.125000 | 0.074288 | 0.141507 | 0.105000 | 0.408 | 0.000 |
| P1_L71 | blok | 981 | 0.107465 | 0.052811 | 0.131446 | 0.064836 | 7.238 | 0.000 |
| P1_L97 | blok | 981 | 0.111581 | 0.058973 | 0.136228 | 0.068144 | 9.888 | 0.000 |
| P1_L184 | blok | 981 | 0.116830 | 0.053522 | 0.142943 | 0.071322 | 18.756 | 0.000 |
| P1_L981 | blok | 981 | 0.119371 | 0.052073 | 0.144386 | 0.074750 | 100.000 | 0.000 |
| P2_k3 | blok | 981 | 0.106134 | 0.000201 | 0.135274 | -0.002500 | 100.000 | 0.000 |
| P2_k5 | blok | 981 | 0.098342 | 0.001677 | 0.125617 | -0.001685 | 100.000 | 0.000 |
| P2_k10 | blok | 981 | 0.091746 | 0.003189 | 0.118056 | 0.000558 | 100.000 | 0.000 |
| P3 | blok | 981 | 2.012339 | 0.586688 | 2.968745 | 1.196539 | 100.000 | 0.000 |

Skor tersebut memakai domain keberhasilan yang dapat berbeda. Cakupan tinggi atau MAE rendah pada sedikit titik tidak cukup untuk memberi peringkat tunggal.

### Irisan titik yang berhasil diisi SEMUA konfigurasi

| Seed | Pola | Panjang slot | Jumlah mask | Irisan semua metode | Irisan persen |
|---|---|---|---|---|---|
| 11 | titik | 1 | 452 | 452 | 100.000 |
| 11 | blok | 4 | 120 | 30 | 25.000 |
| 11 | blok | 71 | 426 | 6 | 1.408 |
| 11 | blok | 97 | 388 | 4 | 1.031 |
| 11 | blok | 184 | 368 | 2 | 0.543 |
| 11 | blok | 981 | 981 | 1 | 0.102 |
| 23 | titik | 1 | 452 | 452 | 100.000 |
| 23 | blok | 4 | 120 | 30 | 25.000 |
| 23 | blok | 71 | 426 | 6 | 1.408 |
| 23 | blok | 97 | 388 | 4 | 1.031 |
| 23 | blok | 184 | 368 | 2 | 0.543 |
| 23 | blok | 981 | 981 | 1 | 0.102 |
| 47 | titik | 1 | 452 | 451 | 99.779 |
| 47 | blok | 4 | 120 | 30 | 25.000 |
| 47 | blok | 71 | 426 | 6 | 1.408 |
| 47 | blok | 97 | 388 | 4 | 1.031 |
| 47 | blok | 184 | 368 | 2 | 0.543 |
| 47 | blok | 981 | 981 | 1 | 0.102 |

| Metode | Pola | Panjang slot | Jumlah dinilai rata-rata | MAE rata-rata | MAE SD antar-seed | RMSE rata-rata | Bias rata-rata |
|---|---|---|---|---|---|---|---|
| P0 | titik | 1 | 451.67 | 0.093044 | 0.009063 | 0.117488 | 0.052518 |
| P1_L1 | titik | 1 | 451.67 | 0.118461 | 0.007688 | 0.149752 | 0.003059 |
| P1_L4 | titik | 1 | 451.67 | 0.118461 | 0.007688 | 0.149752 | 0.003059 |
| P1_L71 | titik | 1 | 451.67 | 0.118461 | 0.007688 | 0.149752 | 0.003059 |
| P1_L97 | titik | 1 | 451.67 | 0.118461 | 0.007688 | 0.149752 | 0.003059 |
| P1_L184 | titik | 1 | 451.67 | 0.118461 | 0.007688 | 0.149752 | 0.003059 |
| P1_L981 | titik | 1 | 451.67 | 0.118461 | 0.007688 | 0.149752 | 0.003059 |
| P2_k3 | titik | 1 | 451.67 | 0.100811 | 0.007545 | 0.129685 | 0.001646 |
| P2_k5 | titik | 1 | 451.67 | 0.093959 | 0.007862 | 0.121854 | 0.002098 |
| P2_k10 | titik | 1 | 451.67 | 0.089505 | 0.008491 | 0.116131 | 0.003407 |
| P3 | titik | 1 | 451.67 | 1.443547 | 0.114360 | 2.320072 | 0.560138 |
| P0 | blok | 4 | 30.00 | 0.092889 | 0.003151 | 0.117444 | 0.045778 |
| P1_L1 | blok | 4 | 30.00 | 0.115667 | 0.013013 | 0.150649 | -0.010778 |
| P1_L4 | blok | 4 | 30.00 | 0.115667 | 0.013013 | 0.150649 | -0.010778 |
| P1_L71 | blok | 4 | 30.00 | 0.115667 | 0.013013 | 0.150649 | -0.010778 |
| P1_L97 | blok | 4 | 30.00 | 0.115667 | 0.013013 | 0.150649 | -0.010778 |
| P1_L184 | blok | 4 | 30.00 | 0.115667 | 0.013013 | 0.150649 | -0.010778 |
| P1_L981 | blok | 4 | 30.00 | 0.115667 | 0.013013 | 0.150649 | -0.010778 |
| P2_k3 | blok | 4 | 30.00 | 0.089282 | 0.004346 | 0.114053 | 0.008610 |
| P2_k5 | blok | 4 | 30.00 | 0.090016 | 0.003024 | 0.114490 | 0.008132 |
| P2_k10 | blok | 4 | 30.00 | 0.088469 | 0.002727 | 0.111005 | 0.005694 |
| P3 | blok | 4 | 30.00 | 1.435000 | 0.154115 | 1.862328 | 0.506778 |
| P0 | blok | 71 | 6.00 | 0.078889 | 0.074802 | 0.095584 | 0.007778 |
| P1_L1 | blok | 71 | 6.00 | 0.145000 | 0.076884 | 0.179780 | -0.089444 |
| P1_L4 | blok | 71 | 6.00 | 0.145000 | 0.076884 | 0.179780 | -0.089444 |
| P1_L71 | blok | 71 | 6.00 | 0.145000 | 0.076884 | 0.179780 | -0.089444 |
| P1_L97 | blok | 71 | 6.00 | 0.145000 | 0.076884 | 0.179780 | -0.089444 |
| P1_L184 | blok | 71 | 6.00 | 0.145000 | 0.076884 | 0.179780 | -0.089444 |
| P1_L981 | blok | 71 | 6.00 | 0.145000 | 0.076884 | 0.179780 | -0.089444 |
| P2_k3 | blok | 71 | 6.00 | 0.120271 | 0.072739 | 0.144572 | -0.068847 |
| P2_k5 | blok | 71 | 6.00 | 0.113688 | 0.065121 | 0.132499 | -0.064794 |
| P2_k10 | blok | 71 | 6.00 | 0.099096 | 0.074553 | 0.116889 | -0.053556 |
| P3 | blok | 71 | 6.00 | 1.581687 | 0.618972 | 1.940218 | 0.569834 |
| P0 | blok | 97 | 4.00 | 0.085000 | 0.036827 | 0.099710 | 0.000000 |
| P1_L1 | blok | 97 | 4.00 | 0.075833 | 0.070460 | 0.098491 | -0.030833 |
| P1_L4 | blok | 97 | 4.00 | 0.075833 | 0.070460 | 0.098491 | -0.030833 |
| P1_L71 | blok | 97 | 4.00 | 0.075833 | 0.070460 | 0.098491 | -0.030833 |
| P1_L97 | blok | 97 | 4.00 | 0.075833 | 0.070460 | 0.098491 | -0.030833 |
| P1_L184 | blok | 97 | 4.00 | 0.075833 | 0.070460 | 0.098491 | -0.030833 |
| P1_L981 | blok | 97 | 4.00 | 0.075833 | 0.070460 | 0.098491 | -0.030833 |
| P2_k3 | blok | 97 | 4.00 | 0.096396 | 0.071174 | 0.127172 | -0.011081 |
| P2_k5 | blok | 97 | 4.00 | 0.093930 | 0.064021 | 0.119014 | -0.018097 |
| P2_k10 | blok | 97 | 4.00 | 0.090721 | 0.070391 | 0.113182 | -0.043281 |
| P3 | blok | 97 | 4.00 | 1.230749 | 0.375337 | 1.385337 | 0.554921 |
| P0 | blok | 184 | 2.00 | 0.100000 | 0.017321 | 0.123000 | 0.073333 |
| P1_L1 | blok | 184 | 2.00 | 0.073333 | 0.014434 | 0.083535 | -0.050000 |
| P1_L4 | blok | 184 | 2.00 | 0.073333 | 0.014434 | 0.083535 | -0.050000 |
| P1_L71 | blok | 184 | 2.00 | 0.073333 | 0.014434 | 0.083535 | -0.050000 |
| P1_L97 | blok | 184 | 2.00 | 0.073333 | 0.014434 | 0.083535 | -0.050000 |
| P1_L184 | blok | 184 | 2.00 | 0.073333 | 0.014434 | 0.083535 | -0.050000 |
| P1_L981 | blok | 184 | 2.00 | 0.073333 | 0.014434 | 0.083535 | -0.050000 |
| P2_k3 | blok | 184 | 2.00 | 0.078724 | 0.036568 | 0.090049 | 0.029124 |
| P2_k5 | blok | 184 | 2.00 | 0.084987 | 0.015487 | 0.099261 | 0.037407 |
| P2_k10 | blok | 184 | 2.00 | 0.099231 | 0.022067 | 0.117432 | 0.046281 |
| P3 | blok | 184 | 2.00 | 1.463293 | 0.563951 | 1.724727 | 0.697981 |
| P0 | blok | 981 | 1.00 | 0.140000 | 0.065574 | 0.140000 | 0.140000 |
| P1_L1 | blok | 981 | 1.00 | 0.163333 | 0.063509 | 0.163333 | 0.163333 |
| P1_L4 | blok | 981 | 1.00 | 0.163333 | 0.063509 | 0.163333 | 0.163333 |
| P1_L71 | blok | 981 | 1.00 | 0.163333 | 0.063509 | 0.163333 | 0.163333 |
| P1_L97 | blok | 981 | 1.00 | 0.163333 | 0.063509 | 0.163333 | 0.163333 |
| P1_L184 | blok | 981 | 1.00 | 0.163333 | 0.063509 | 0.163333 | 0.163333 |
| P1_L981 | blok | 981 | 1.00 | 0.163333 | 0.063509 | 0.163333 | 0.163333 |
| P2_k3 | blok | 981 | 1.00 | 0.135471 | 0.073994 | 0.135471 | 0.135471 |
| P2_k5 | blok | 981 | 1.00 | 0.139486 | 0.041937 | 0.139486 | 0.139486 |
| P2_k10 | blok | 981 | 1.00 | 0.126772 | 0.062217 | 0.126772 | 0.126772 |
| P3 | blok | 981 | 1.00 | 0.371087 | 0.181699 | 0.371087 | 0.012591 |

Irisan mencakup P1 dengan L=1, sehingga pada blok panjang umumnya hanya slot awal setelah donor yang tersedia. Hasil irisan **tidak mewakili bagian dalam blok panjang**. Persentase irisan wajib dibaca bersama MAE; jangan menyimpulkan kemampuan semua metode pada seluruh blok dari irisan kecil ini.

![Error dan cakupan blok](figures/pilot_way_urang/error_dan_cakupan_blok.png)

![Error pada irisan](figures/pilot_way_urang/error_irisan_semua_metode.png)

### Waktu komputasi

| Metode | Fit detik | Scaler fit detik bersama |
|---|---|---|
| P0 | 0.000529 | 0.000000 |
| P1_L1 | 0.000000 | 0.000000 |
| P1_L4 | 0.000000 | 0.000000 |
| P1_L71 | 0.000000 | 0.000000 |
| P1_L97 | 0.000000 | 0.000000 |
| P1_L184 | 0.000000 | 0.000000 |
| P1_L981 | 0.000000 | 0.000000 |
| P2_k3 | 0.001638 | 0.026290 |
| P2_k5 | 0.001556 | 0.026290 |
| P2_k10 | 0.001626 | 0.026290 |
| P3 | 3.227640 | 0.026290 |

| Metode | Transform semua skenario detik | Jumlah query diproses | Timestamp query unik |
|---|---|---|---|
| P2_k3 | 12.877115 | 8205 | 3888 |
| P2_k5 | 12.755161 | 8205 | 3888 |
| P2_k10 | 13.755619 | 8205 | 3888 |
| P3 | 0.965385 | 8205 | 3888 |

Fit dilakukan satu kali dan dipakai ulang, bukan diulang untuk setiap seed. Waktu transform P2/P3 diukur dengan menjalankan masing-masing skenario secara terpisah, meskipun beberapa query antar-seed identik. Durasi batch aktual dijumlah per skenario pada `Hasil_per_seed`; scaling query adalah tahap bersama di luar timer model. P1 mencakup pembangunan state kausal seluruh histori train+validasi per skenario, sehingga biaya transform tidak sepenuhnya setara dengan query batch P2/P3. Hardware, versi, thread=1 dan parameter tersimpan pada config; waktu dapat berubah antar-eksekusi.

## 6. Hasil yang perlu diperiksa dan kegagalan

| Metode | Prediksi negatif | Di luar rentang train | Gagal atau ditolak |
|---|---|---|---|
| P0 | 0 | 0 | 0 |
| P1_L1 | 0 | 0 | 6721 |
| P1_L4 | 0 | 0 | 6333 |
| P1_L71 | 0 | 0 | 3720 |
| P1_L97 | 0 | 0 | 3174 |
| P1_L184 | 0 | 0 | 2391 |
| P1_L981 | 0 | 0 | 0 |
| P2_k3 | 0 | 0 | 1 |
| P2_k5 | 0 | 0 | 1 |
| P2_k10 | 0 | 0 | 1 |
| P3 | 592 | 1629 | 1 |

Hitungan di atas adalah sel-skenario-metode, sehingga timestamp yang sama dapat muncul pada beberapa seed. Di luar min/max train adalah kandidat pemeriksaan, bukan otomatis salah atau batas sensor. Tidak ada clipping; semua prediksi/status disimpan di `prediksi_penilaian.csv`. Peringatan fit/transform tercatat terpisah.

| Metode | Tahap | Kategori | Pesan |
|---|---|---|---|
| P3 | fit | ConvergenceWarning | [IterativeImputer] Early stopping criterion not reached. |

| Pesan |
|---|
| Tidak ada exception fit/transform; penolakan kovariat/batas donor tercatat per prediksi. |

## 7. Kesimpulan terbatas dan kandidat forecasting

- Pada irisan titik terpisah (rata-rata 451.67 dari 452 sel per seed), P2 k=10 memiliki MAE 0.089505 µg/m³; P0 0.093044; P1 L=1 0.118461. KNN k=10 menjanjikan untuk rekonstruksi kondisional ini, dengan perbedaan terhadap baseline yang perlu diuji lagi.
- Rentang SO₂ teramati validasi adalah 1.09–2.06 µg/m³; train 0.92–80.71 µg/m³. Mask ini tidak menguji kemampuan pada rentang SO₂ tinggi seperti yang ada di train.
- P3 belum layak dianggap kandidat siap pakai dalam konfigurasi ini: kriteria konvergensi tidak tercapai pada 20 iterasi dan terdapat prediksi negatif. Cakupan finite yang tinggi tidak menjadikan hasil tersebut dapat diterima. Perlu diagnosis fitur/regularisasi/konvergensi pada train dan validasi sebelum dipertimbangkan lagi; tidak ada clipping atau retuning test.
- Pada titik 1 slot, di antara konfigurasi yang mengisi seluruh mask, P0 mencatat MAE rata-rata terendah 0.0930531 µg/m³ (SD 0.00905427). Ini temuan pada domain/pola tersebut, bukan peringkat keseluruhan.
- Pada blok 4 slot, di antara konfigurasi yang mengisi seluruh mask, P2_k10 mencatat MAE rata-rata terendah 0.0907896 µg/m³ (SD 0.00458087). Ini temuan pada domain/pola tersebut, bukan peringkat keseluruhan.
- Pada blok 71 slot, di antara konfigurasi yang mengisi seluruh mask, P2_k10 mencatat MAE rata-rata terendah 0.0872005 µg/m³ (SD 0.00486348). Ini temuan pada domain/pola tersebut, bukan peringkat keseluruhan.
- Pada blok 97 slot, di antara konfigurasi yang mengisi seluruh mask, P2_k10 mencatat MAE rata-rata terendah 0.0855124 µg/m³ (SD 0.00250115). Ini temuan pada domain/pola tersebut, bukan peringkat keseluruhan.
- Pada blok 184 slot, di antara konfigurasi yang mengisi seluruh mask, P2_k10 mencatat MAE rata-rata terendah 0.088259 µg/m³ (SD 0.00245334). Ini temuan pada domain/pola tersebut, bukan peringkat keseluruhan.
- Pada blok 981 slot, di antara konfigurasi yang mengisi seluruh mask, P2_k10 mencatat MAE rata-rata terendah 0.0917458 µg/m³ (SD 0.00318855). Ini temuan pada domain/pola tersebut, bukan peringkat keseluruhan.
- P1 tetap kandidat sederhana untuk gap pendek dengan umur donor yang sesuai; ketika umur melebihi L, ia sengaja tidak mengisi. P0 menjadi baseline tetap. P2/P3 perlu dibaca bersama dukungan kovariat, sebaran seed, kondisi ekstrem dan peringatan. Kandidat dengan hasil baik pada validasi dapat dibawa ke eksperimen forecasting yang memakai model, origin, target asli dan cakupan yang sama.
- Skor imputasi belum menentukan metode forecasting terbaik. Kemampuan pada missing alami, hilang semua kovariat, gap awal tanpa donor, perubahan distribusi, dan latensi aktual belum dibuktikan. Konfigurasi menjanjikan perlu diuji ulang dalam skenario tersebut, lalu pada forecasting terkontrol; test tetap dikunci.
- Penyedia data masih perlu mengonfirmasi latensi, makna timestamp dan nilai khusus, validitas sensor serta penyebab pola konstan/lonjakan. Tidak ada penyebab yang disimpulkan dari pilot. Way Urang tetap pilot sementara.

## 8. Reproduksibilitas dan integritas

- Notebook: `notebooks/04_pilot_imputasi_way_urang.ipynb`.
- Konfigurasi: `pilot_way_urang/config.json`; parameter train: `parameter_train.json`.
- Lokasi mask: `lokasi_masking.csv`; kejadian: `kejadian_masking.csv`; truth hanya untuk penilaian: `truth_penilaian_saja.csv`; prediksi: `prediksi_penilaian.csv`.
- Audit sumber, semua penanda, ledger off-grid dan pembatasan test tersimpan terpisah; data sumber dan hasil audit sebelumnya tidak diubah.

- `data_raw\Telkom University Purwokerto - Intan Azizah.xlsx`: SHA256 sebelum = sesudah = `bd00734b954e8df024c048575fad07f1eb1cbe1083d4c8929ae362aeb45a7406`.
- `outputs\rancangan_eksperimen_preprocessing.md`: SHA256 sebelum = sesudah = `d256294ac4568c82cbfeb9e06006b435dce293f52533bbf8f8bb3b865be1168d`.
- `outputs\audit_kualitas_tables.json`: SHA256 sebelum = sesudah = `89bf7a39781e9ea3368a8c7a4731cf7c8c36523eb117a2e034113c056e5e0b05`.
- `outputs\pemeriksaan_lanjutan_tables.json`: SHA256 sebelum = sesudah = `4cb3f7885064fd943ef5efdc1f0e4fdd7448e2587467f9bceb5304b987527030`.
- `outputs\audit_grids\02_grid_audit.csv`: SHA256 sebelum = sesudah = `b438ed9dd3308256071016b37d5df59208df0f961fc91621826f1f8afbb7bd9c`.
