# Ringkasan eksperimen dan keputusan sementara

Arsip v1 menutup rangkaian eksperimen Way Urang tahap 1–8 sebelum eksperimen antarstasiun. Dokumen ini adalah metadata arsip baru yang merangkum berkas historis; bukan hasil eksperimen atau checkpoint baru.

**Keputusan operasional sementara: imputasi median dan warmup 30 hari. Pilihan ini bukan metode yang terbukti paling akurat.** Jangan mengubahnya menjadi klaim pemenang universal atau stasiun final.

| Tahap | Cakupan dan batas kesimpulan | Laporan utama |
|---|---|---|
| 1–2 | Struktur Excel diperiksa sebelum header; kualitas waktu, missing, gap, nilai konstan dan metadata diaudit. Audit bersama semua stasiun disalin utuh. | [Audit kualitas](salinan/outputs/ringkasan_audit_kualitas.md) |
| 3 | Off-grid dan rentetan konstan ditelusuri; rancangan P0–P3 serta evaluasi kausal dibuat. Tidak semua anomali adalah kesalahan sensor. | [Rancangan](salinan/outputs/rancangan_eksperimen_preprocessing.md) |
| 4 | Pilot masking menggunakan seed11/23/47; median, ffill, KNN dan iterative diperiksa. Akurasi rekonstruksi bukan akurasi forecasting. P3 belum konvergen dan ditunda. | [Pilot](salinan/outputs/ringkasan_pilot_imputasi_way_urang.md) |
| 5 dan penelusuran | Ridge satu langkah30menit dibandingkan pada timestamp sama. Irisan sampel/kovariat CurahHujan menyempitkan train; hasil lama dipertahankan. | [Forecasting](salinan/outputs/ringkasan_forecasting_acuan_way_urang.md), [penelusuran](salinan/outputs/penelusuran_sampel_tahap5.md) |
| 5B | Ablasi tanpa CurahHujan memperluas train; lengan cohort tetap memisahkan sebagian efek fitur dari penambahan sampel. Belum memilih preprocessing final. | [Ablasi](salinan/outputs/ringkasan_5B_ablasi_curah_hujan.md) |
| 6 | Empat fold kalender, expanding train awal120hari. Peringkat preprocessing dan keunggulan terhadap persistence tidak konsisten; fold1 gagal saat perubahan rezim. | [Backtesting](salinan/outputs/ringkasan_backtesting_preprocessing.md) |
| Diagnosis6 | Tidak ditemukan kesalahan alignment/scaler/target pada pemeriksaan yang dijalankan. Cohort sempit, respons Ridge lemah dan ekstrapolasi kovariat mendukung keterbatasan model; penyebab fisik data belum diketahui. | [Diagnosis](salinan/outputs/diagnosis_fold1.md) |
| 7 | Satu LSTM32, tiga seed,48langkah×22kanal, innerholdback15%,36model selesai. Fold1 tetap dekat tingkat tinggi setelah SO2 turun. Variasi seed dan cakupan menghalangi klaim pemenang tunggal. | [LSTM](salinan/outputs/ringkasan_lstm_preprocessing.md) |
| 8 | Median/Ridge dengan warmup30/60/90hari. C60 error gabungan terendah, tetapi C90 unggul fold2 dan C30 fold3/4. Tidak ada durasi unggul atas persistence pada MAE danRMSE sekaligus dalam satu fold. Fitur evaluasi sebelum scaling identik; komposisi train ikut berubah. | [Sensitivitas](salinan/outputs/ringkasan_sensitivitas_kalibrasi.md) |

## Keputusan yang belum dapat dibuat

- Belum menetapkan stasiun penelitian final, preprocessing paling akurat, horizon final, atau algoritma forecasting terbaik.
- Median/warmup30 dipakai sebagai pengaturan operasional sementara untuk kerja berikutnya, dengan baseline persistence, pemeriksaan cakupan dan validasi kronologis tetap diperlukan.
- Hasil menunjukkan perlunya memeriksa keterwakilan periode/konsentrasi, ekstrapolasi, kehilangan kovariat dan sensitivitas seed; penambahan data tinggi tidak membuktikan kemampuan meramalkan konsentrasi tinggi.
- Tidak memakai test akhir untuk memilih pipeline. Tidak menghubungkan lonjakan dengan gunung/sensor tanpa bukti.

Rincian asumsi, hash lama yang berubah, waktu hilang dan berkas yang tidak tersedia ada pada [catatan keterbatasan](catatan_keterbatasan.md).
