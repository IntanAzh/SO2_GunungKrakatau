# Status proyek SO₂

## Tahap selesai

Tahap 1–3 audit struktur, kualitas dan rancangan preprocessing; tahap 4 pilot imputasi; tahap 5 Ridge, penelusuran sampel dan 5B ablasi Curah Hujan; tahap 6 expanding-window backtesting dan diagnosis fold pertama; tahap 7 verifikasi preprocessing LSTM; tahap 8 sensitivitas kalibrasi; arsip Way Urang v1; tahap 9 kontribusi antarstasiun Ridge; tahap 10 verifikasi Pulomerak dengan24 LSTM. Seluruh tahap di atas selesai, dengan keterbatasan masing-masing laporan tetap berlaku.

## Konfigurasi kerja

Way Urang masih pilot. Median kausal warmup 30 hari tanpa Curah Hujan adalah pilihan operasional sementara, bukan metode terbukti paling akurat. Horizon 30 menit, empat fold kalender tetap dari tahap 6, development hanya sampai2026-06-19 06:00. Tahap 10 memakai S0 Way Urang dan S1 tambahan SO₂ Pulomerak lag1 tiap langkah,48 slot berurutan,22/24 kanal termasuk missing asli. LSTM 32 satu layer output linear, Adam 0.001, MSE, batch 64, maks 50 epoch, early stopping8, seed 11/23/47. Validasi internal 15% akhir kalender train. Median/scaler train internal saja, target asli distandardisasi train dan inverse sebelum skor.

## Lokasi hasil

- Arsip historis: outputs/arsip_way_urang_v1/ dan outputs/arsip_way_urang_v1.zip.
- Tahap 7: outputs/lstm_preprocessing/ dan outputs/ringkasan_lstm_preprocessing.md.
- Tahap 8: outputs/sensitivitas_kalibrasi/ dan outputs/ringkasan_sensitivitas_kalibrasi.md.
- Tahap 9: outputs/kontribusi_antarstasiun/, outputs/hasil_kontribusi_antarstasiun.xlsx dan outputs/ringkasan_kontribusi_antarstasiun.md.
- Tahap 10: outputs/lstm_antarstasiun_tahap10/hasil_lstm_antarstasiun.xlsx, ringkasan_lstm_antarstasiun.md, config_locked.json, daftar_fitur.json, dependensi_freeze.txt, tabel_hasil.json.
- Notebook tahap 10: notebooks/10_lstm_antarstasiun.ipynb.
- Tiap model tahap 10: Fold_XX/S0_seedXX atau S1_seedXX, complete.json berisi hash checkpoint/riwayat/prediksi; model lengkap tidak dilatih ulang saat resume.
- Environment kerja: .venv-lstm Python 3.12/PyTorch CPU; .venv lama tetap dipertahankan.

## Temuan utama

Tahap 9 Ridge tidak menunjukkan manfaat Pulomerak yang stabil: perbaikan kecil kedua error hanya pada fold 2, sedangkan donor Talang/kombinasi tidak memberi manfaat stabil. Pada tahap 10 Pulomerak memperbaiki kedua error pada 4/12 pasangan fold–seed dan memperburuk keduanya pada 8/12. Lihat tabel per fold/seed, transisi dan cakupan; tidak memilih seed terbaik.

Keputusan sementara tahap 10: S0 tetap acuan sederhana dan S1 disimpan sebagai sensitivitas, karena manfaat Pulomerak tidak konsisten di seluruh periode dan seed.

| Skenario | MAE mean | MAE SD seed | RMSE mean | RMSE SD seed | Bias mean | Cakupan persen mean |
|---|---|---|---|---|---|---|
| S0 | 4.313477 | 0.666361 | 11.592217 | 1.871904 | 4.105196 | 100.000 |
| S1 | 9.424562 | 1.314924 | 16.220401 | 1.820755 | 9.193938 | 100.000 |
| Persistence | 0.206335 | 0.000000 | 0.464340 | 0.000000 | 0.002240 | 100.000 |

## Keterbatasan

Development telah digunakan berulang; ini validasi pengembangan. Validasi banyak mencakup konsentrasi rendah, dengan transisi fold 1 yang harus dipertahankan. Zona waktu, sinkronisasi dan latensi antarstasiun belum dikonfirmasi; delay30menit merupakan asumsi. Nol, ekstrem/konstan dan off-grid dipertahankan sebagai penanda, bukan otomatis kesalahan. Cakupan model tidak sama dengan ketersediaan donor asli. Tidak ada bukti penyebab vulkanik atau kerusakan sensor.

Ada perubahan hash notebook02 sebelum resume tahap 7 dan sebagian waktu komputasi lama tidak tersimpan. Kedua hal itu tercatat dalam laporan/arsip dan tidak diperbaiki dengan angka yang diarang. Seluruh hash lama dibandingkan terhadap baseline tahap 10 saat ini, bukan klaim identik sepanjang sejarah. Percobaan preflight tahap 10 gagal karena pip tidak tersedia, sebelum training; versi paket dicatat lewat metadata dan percobaan konfigurasi itu disimpan.

## Pekerjaan berikutnya

Tinjau kestabilan S0–S1 dan besarnya manfaat terhadap tambahan kompleksitas sebelum menetapkan kandidat. Klarifikasi waktu/latensi, dokumentasi sensor, periode konstan dan transisi dengan penyedia data. Jika tidak konsisten, simpan Pulomerak sebagai sensitivitas. Tidak melakukan tuning atau membuka test untuk mengejar skor. Rancangan perbandingan algoritma atau eksperimen berikutnya perlu ditetapkan terlebih dahulu.

Untuk melanjutkan sesi, baca file ini, ringkasan tahap 10 dan config_locked.json, lalu periksa complete.json:24 model lengkap. Jangan menimpa hasil lama. Script scripts/lstm_interstation_stage10.py mendukung resume model yang belum lengkap, bukan melatih ulang model selesai. Scripts/summarize_interstation_stage10.py menyusun ulang laporan dari hasil tersimpan tanpa training.

## Status test akhir

**Test akhir mulai2026-06-19 06:30 belum digunakan untuk evaluasi forecasting dan tidak dimuat nilainya pada tahap 10.** Data pernah diaudit menyeluruh pada tahap awal, sehingga jangan menyebutnya belum pernah dilihat sama sekali. Hashing sumber untuk integritas membaca byte, bukan evaluasi nilai test. Test tidak dipakai untuk seleksi preprocessing, arsitektur, seed atau warmup.

## Tahap 11 selesai — perbandingan algoritma S0

GRU32 dan TCN empat blok kausal masing-masing dilatih pada empat fold × tiga seed; total24 fit baru. LSTM S0 tahap10 digunakan dari checkpoint tersimpan, tanpa fit ulang. Tensor S0 median kausal warmup30 hari, 48 langkah, 22 kanal termasuk indikator missing, horizon30 menit, target asli dan batas internal/evaluasi sama. Konfigurasi dan uji kausalitas TCN dibekukan sebelum fit.

Lokasi: outputs/perbandingan_algoritma_tahap11/config_locked.json, daftar_fitur.json, hasil_perbandingan_algoritma.xlsx, ringkasan_perbandingan_algoritma.md, tabel_hasil.json, prediksi_oof_seluruh_seed.csv, timestamp_semua_bagian.csv, Fold_XX/GRU_seedXX dan TCN_seedXX (checkpoint, histori, prediksi, waktu), figures/, dan manifest_tahap11.json. Notebook: notebooks/11_perbandingan_algoritma.ipynb. Environment tetap .venv-lstm PyTorch CPU; .venv lama tidak diubah.

Temuan: rerata MAE terendah di antara deep learning adalah GRU; rerata RMSE terendah adalah LSTM. Tidak ada satu pemenang DL yang sama pada kedua metrik. Tidak ada DL yang mengungguli persistence pada kedua rerata gabungan. Lihat laporan untuk angka per fold, seed, bulan, bias, cakupan, transisi dan waktu.

Batas: development telah dipakai berulang; fold pertama dan transisi22 Juni 2025 tidak dihapus; biaya per model dan optimisasi berbeda walaupun anggaran epoch sama; zona waktu, latensi, sensor dan rentang konsentrasi belum tuntas. Arsip tahap1–10 tetap sama menurut SHA256. File STATUS ini memang diperbarui; hash sebelum/sesudah dicatat. Jangan memilih seed terbaik atau menyebut model siap untuk peringatan dini operasional.

Pekerjaan berikutnya: tinjau apakah ada kandidat yang layak dirancang untuk verifikasi terpisah, bandingkan dengan persistence pada timestamp yang sama, minta klarifikasi metadata waktu/latensi/kualitas dari penyedia data. Bekukan rancangan sebelum test akhir. **Test akhir mulai2026-06-19 06:30 belum digunakan untuk evaluasi forecasting tahap11 dan tidak dimuat nilainya.** Data pernah diaudit menyeluruh pada tahap awal sehingga bukan data yang belum pernah dilihat sama sekali.
