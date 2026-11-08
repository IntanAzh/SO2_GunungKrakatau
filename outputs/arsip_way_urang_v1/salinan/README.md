# Audit awal data SO₂

Tahap pertama Tugas Akhir: inventaris Excel, metadata laporan, struktur header,
satuan, posisi awal data, dan lima observasi awal per sheet. File di `data_raw`
hanya dibaca. Tidak ada pembersihan data, scaling, atau pelatihan model.

## Menjalankan di Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_audit_notebook.py
```

Buka `notebooks/01_audit_data.ipynb` dengan editor notebook, lalu pilih interpreter
`.venv\Scripts\python.exe`. Aktivasi PowerShell tidak diperlukan.
Kernel bernama `Python (.venv SO2)` juga disiapkan di dalam environment.
`ipykernel` menyediakan kernel, bukan aplikasi antarmuka JupyterLab.

## Isi proyek

- `data_raw/`: file Excel asli. Jangan simpan perubahan ke file ini.
- `notebooks/01_audit_data.ipynb`: penjelasan, kode baca-saja, dan hasil eksekusi.
- `outputs/audit_struktur_data.json`: inventaris, ringkasan setiap sheet, metadata,
  kolom/satuan, lima baris awal, posisi awal pengukuran/SO2, serta SHA256 sumber.
- `requirements.txt`: versi dependensi utama yang dipakai.
- `outputs/environment_freeze.txt`: seluruh versi paket untuk reproduksibilitas.
- `scripts/run_audit_notebook.py`: menjalankan notebook melalui IPython dan menyimpan hasilnya.
- `scripts/build_audit_notebook.py`: membuat ulang kerangka notebook; menjalankannya
  akan mengganti notebook dan mengosongkan hasil eksekusi sebelumnya.

Nomor baris pada hasil mengikuti Excel, mulai 1. Header dan satuan dipertahankan
terpisah. Baris bertimestamp bukan jaminan pengukuran tersedia atau valid.
Pola header setiap sheet diperiksa sebelum penetapan awal tabel. Layout berbeda
atau format `.xls`/`.xlsb` dilaporkan untuk pemeriksaan/engine tambahan.

Tahap berikutnya memeriksa kelengkapan SO₂, tipe nilai, konsistensi waktu dan
satuan, duplikat, celah waktu, nilai nol/konstan, serta potensi anomali. Penanganan
nilai dan penentuan desain forecasting dilakukan setelah audit kualitas data.

## Tahap 2 — audit kualitas per stasiun

```powershell
.\.venv\Scripts\python.exe scripts\run_audit_notebook.py 02_audit_kualitas_data.ipynb
```

Notebook `02_audit_kualitas_data.ipynb` memuat hasil waktu, kelengkapan, gap,
statistik, kandidat pemeriksaan dan grafik untuk masing-masing stasiun.
Tabel lengkap ada di `outputs/audit_kualitas_data.xlsx`, ringkasan dan definisi
denominator di `outputs/ringkasan_audit_kualitas.md`, grafik di `outputs/figures/`.
Grid di `outputs/audit_grids/` hanya turunan audit; bukan dataset mentah atau data latih.

Analisis memakai `.venv`; ekspor Excel memakai runtime Node artifact-tool bawaan
Codex melalui `scripts/export_quality.mjs`. Lokasi runtime dapat diatur dengan
`SO2_AUDIT_NODE`; modul `@oai/artifact-tool` harus tersedia bagi skrip eksportir.
Dependensi ekspor ini tidak tersedia hanya dengan `pip install`.
Kode audit berada di `scripts/audit_quality.py`.
Uji kasus celah/duplikat dijalankan dengan `scripts/check_quality_logic.py`.

## Tahap 3 — pemeriksaan lanjutan dan usulan preprocessing

`outputs/pemeriksaan_lanjutan_kualitas.xlsx` berisi rekonsiliasi 13 timestamp
off-grid, skenario pemetaan hipotetis tanpa perubahan waktu, 10 rentetan SO₂
konstan terpanjang per stasiun, konteks variabel lain, dan kelayakan kualitas.
`outputs/rancangan_eksperimen_preprocessing.md` memuat temuan dan usulan P0–P3,
protokol kausal, masking, metrik serta cakupan. Grafik konteks ada di
`outputs/figures/tahap3/` dan ditautkan dari laporan.

Analisis ulang: `.\.venv\Scripts\python.exe scripts\followup_quality.py`.
Ekspor tabel JSON tahap 3 menggunakan `scripts/export_quality.mjs` dengan argumen
`pemeriksaan_lanjutan_tables.json pemeriksaan_lanjutan_kualitas.xlsx .audit_preview_stage3`.
Verifikasi: `.\.venv\Scripts\python.exe scripts\verify_followup_outputs.py`.
Tidak ada pipeline imputasi atau forecasting yang dijalankan pada tahap ini.

## Tahap 4 — pilot imputasi Way Urang

```powershell
.\.venv\Scripts\python.exe scripts\run_audit_notebook.py 04_pilot_imputasi_way_urang.ipynb
.\.venv\Scripts\python.exe scripts\verify_pilot_outputs.py
```

Pilot sementara memakai split waktu 70/15/15. P0 median, P1 forward fill terbatas,
P2 KNN k=3/5/10, dan P3 IterativeImputer–BayesianRidge di-fit pada train;
masking dan evaluasi hanya pada validasi. Pengujian akhir hanya diinventarisasi.
Tidak ada model forecasting dilatih. Dependensi scikit-learn dicatat dalam requirements.

Hasil: `outputs/pilot_imputasi_way_urang.xlsx`,
`outputs/ringkasan_pilot_imputasi_way_urang.md`, dan grafik pada
`outputs/figures/pilot_way_urang/`. Folder `outputs/pilot_way_urang/`
menyimpan konfigurasi, parameter train, lokasi mask, truth khusus penilaian,
prediksi/status dan salinan nilai/penanda asli. Ini artefak eksperimen;
bukan pengganti sumber atau data forecasting final.

## Tahap 5 — penyaringan preprocessing melalui Ridge

```powershell
.\.venv\Scripts\python.exe scripts\run_audit_notebook.py 05_forecasting_acuan_way_urang.ipynb
.\.venv\Scripts\python.exe scripts\verify_forecast_outputs.py
```

Ridge satu langkah 30 menit membandingkan median, ffill L4 tanpa fallback, dan
KNN k10 dengan fitur/timestamp train bersama. Konfigurasi disimpan sebelum fit
di `outputs/forecasting_way_urang/config_locked.json`. Histori train diimputasi
dengan prefix kausal setelah kalibrasi 90 hari, lalu fitur lag dibentuk.
Split tetap 70/15/15; nilai test tidak dimuat atau dievaluasi. P3 ditunda.

Hasil: `outputs/forecasting_acuan_way_urang.xlsx`,
`outputs/ringkasan_forecasting_acuan_way_urang.md`, grafik di
`outputs/figures/forecasting_way_urang/`, serta ledger train/evaluasi/prediksi,
histori dan indikator di `outputs/forecasting_way_urang/`.
Ini penyaringan awal dengan Ridge, bukan keputusan preprocessing final
untuk deep learning atau pemilihan stasiun akhir.

## Penelusuran pembentukan sampel tahap 5

`outputs/penelusuran_sampel_tahap5.xlsx` dan
`outputs/penelusuran_sampel_tahap5.md` menelusuri eligibility per timestamp,
waterfall per tahap/pipeline/bulan, target berkonsentrasi lebih tinggi,
penolakan cakupan KNN, serta usia donor persistence. Artefak tahap 5 lama
tetap dipertahankan dan dicek SHA256 sebelum/sesudah.

`scripts/trace_stage5_samples.py` hanya membaca artefak tersimpan dan menghitung
diagnostik; tidak menjalankan fit/predict/imputasi. Ledger boolean lengkap ada
di `outputs/penelusuran_tahap5/penanda_lengkap_per_timestamp.csv`.
Simulasi mengabaikan fitur curah hujan hanya menghitung eligibility hipotetis,
bukan menerapkan koreksi atau melatih model baru.

## Tahap 5B — ablasi penuh Curah Hujan

```powershell
.\.venv\Scripts\python.exe scripts\run_audit_notebook.py 05B_ablasi_curah_hujan.ipynb
.\.venv\Scripts\python.exe scripts\verify_ablation_5b.py
```

Curah Hujan dikeluarkan dari imputer/scaler, fitur Ridge dan indikatornya;
parameter lain tetap. Histori kausal dihitung ulang pada train yang diizinkan.
Gate sebelum Ridge memastikan kohort bersama meluas dan observasi >2,14 kembali
masuk. Angka ini hanya untuk penelusuran, bukan ambang kualitas/bahaya.

Dua lengan tanpa hujan: train bersama baru dan timestamp train bersama lama.
Perbandingan memakai timestamp validasi identik dengan hasil tahap 5 yang
dipertahankan. Tidak ada nilai test dimuat/dievaluasi.
Hasil ada di `outputs/hasil_5B_ablasi_curah_hujan.xlsx`,
`outputs/ringkasan_5B_ablasi_curah_hujan.md`, folder
`outputs/ablasi_curah_hujan_5B/`, dan grafik `outputs/figures/ablasi_curah_hujan_5B/`.

## Tahap 6: expanding-window backtesting

Notebook `notebooks/06_backtesting_preprocessing.ipynb` memakai development
train+validasi lama, yang sudah digunakan dalam eksperimen sebelumnya.
Test akhir mulai 19 Juni 2026 pukul 06:30 tetap dikunci.
Empat fold kalender: train awal 120 hari, sisa development dibagi empat
validasi tidak tumpang tindih. Semua parameter tetap 5B tanpa Curah Hujan.
Gate kecukupan seluruh fold disimpan sebelum 12 fit Ridge; seluruhnya lolos.

```powershell
.\.venv\Scripts\python.exe scripts\run_audit_notebook.py 06_backtesting_preprocessing.ipynb
.\.venv\Scripts\python.exe scripts\finalize_backtesting_stage6.py
.\.venv\Scripts\python.exe scripts\verify_backtesting_stage6.py
```

Hasil: `outputs/backtesting_preprocessing.xlsx`,
`outputs/ringkasan_backtesting_preprocessing.md`, konfigurasi/fitur/histori/
timestamp/prediksi di `outputs/backtesting_way_urang/`, grafik di
`outputs/figures/backtesting_way_urang/`. Verifikasi tidak melatih model baru.

## Tahap 7: satu LSTM untuk verifikasi preprocessing

Gunakan `.venv-lstm` Python 3.12 dan PyTorch CPU; `.venv` lama dipertahankan.
Dependensi lengkap dicatat pada `requirements-lstm-lock.txt`; paket torch
menggunakan index CPU resmi `https://download.pytorch.org/whl/cpu`.
Notebook `07_lstm_preprocessing.ipynb` memakai kernel `so2-lstm`.

```powershell
.\.venv-lstm\Scripts\python.exe scripts\run_audit_notebook.py 07_lstm_preprocessing.ipynb
.\.venv-lstm\Scripts\python.exe scripts\verify_lstm_stage7.py
```

Satu LSTM32 + Dense linear, seed 11/23/47, semua empat fold tahap6,
48 langkah berurutan × 22 kanal, tanpa Curah Hujan. Tail15%train untuk
early stopping; imputer dan scaler hanya innertrain. Target training asli
distandardisasi dengan parameter innertrain, lalu prediksi di-inverse ke
satuan asli. Parameter dan konfigurasi disimpan sebelum training.
Test akhir tetap dikunci. Ridge tidak dilatih ulang; prediksi tersimpan
dibandingkan pada timestamp evaluasi identik.

Hasil di `outputs/hasil_lstm_preprocessing.xlsx`,
`outputs/ringkasan_lstm_preprocessing.md`, folder
`outputs/lstm_preprocessing/`, dan `outputs/figures/lstm_preprocessing/`.
Verifikasi hanya inferensi checkpoint serta hitung ulang audit/skor.
