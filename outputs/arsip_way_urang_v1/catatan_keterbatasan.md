# Keterbatasan arsip Way Urang v1

1. **Median dan warmup 30 hari adalah pilihan operasional sementara**, bukan metode atau durasi yang terbukti paling akurat. Keputusan ini dicatat pada saat arsip atas instruksi pengguna; tidak mengubah protokol maupun hasil historis.
2. Way Urang adalah pilot sementara, bukan stasiun final. Peringkat preprocessing berubah antar-periode dan seed. Development sudah digunakan berulang; hasil bukan holdout independen.
3. **Test akhir belum digunakan untuk evaluasi forecasting/imputasi.** Batasnya 2026-06-19 06:30 menurut kalender grid sebelumnya, tanpa asumsi zona waktu baru. Audit awal sudah mencakup keseluruhan sumber; jangan menyebut test sama sekali belum pernah dilihat. Arsip menyalin berkas audit/grid historis sebagai byte, termasuk jika rentangnya meliputi test, tanpa memuat seri test untuk analisis/evaluasi.
4. **Ada perubahan hash notebook lama sebelum resume LSTM.** Salinan saat ini tetap dipertahankan, bukan direkonstruksi menjadi versi lama. Penyebab perubahan tidak diketahui. Rincian hash historis:

```json
{
  "notebooks\\02_audit_kualitas_data.ipynb": {
    "SHA256 awal eksperimen": "031948f8d28d754b8fe0bf6a04e4341d01a7af32ff1054d18b4489370e623886",
    "SHA256 sebelum resume": "56d04e444794dd4e7f2505bbd0f1a3d4f05fffb99b95c6fb63fedea8995ead56"
  }
}
```

5. **Sebagian waktu komputasi tidak tersimpan:** preprocessing empat fold LSTM serta waktu fit checkpoint Fold_04/P0_median/seed11. Nilai yang hilang tetap kosong, tidak diisi nol atau estimasi. Dua puluh delapan checkpoint selesai dipertahankan saat resume; delapan kombinasi tersisa diselesaikan dengan konfigurasi sama. Kombinasi tanpa checkpoint lengkap dimulai kembali dari seed tetap karena state optimizer parsial tidak tersedia.
6. Notebook tahap 03 tidak ditemukan. Hasil tahap 3 tersedia berupa pemeriksaan_lanjutan_kualitas.xlsx, rancangan_eksperimen_preprocessing.md dan skrip. Tidak dibuat notebook pengganti atau hasil baru.
7. Snapshot notebook02 yang persis cocok hash awal LSTM tidak tersedia sebagai file terpisah yang diketahui. Arsip berisi notebook terkini serta catatan kedua hash; tidak mengarang salinan lama.
8. Checkpoint LSTM yang tersedia adalah bobot terbaik, bukan state optimizer/per-epoch lengkap. Seluruh checkpoint/history yang benar-benar ada disalin; tidak ada checkpoint baru dihasilkan.
9. Excel sumber sengaja tidak disalin; lokasi, ukuran dan SHA256 ada pada referensi_excel_sumber.json. Reproduksi dari sumber membutuhkan berkas eksternal tersebut. Audit awal lintas stasiun disalin utuh karena merupakan konteks bersama, bukan diklaim seluruhnya data Way Urang.
10. Environment biner .venv/.venv-lstm, node_modules dan runtime aplikasi tidak digandakan. Requirements/lock/freeze serta metadata konfigurasi dan pyvenv disalin. Versi JS exporter tidak memiliki package-lock proyek; skrip dan hasil XLSX final tetap tersedia. Bukan jaminan instalasi ulang otomatis di semua sistem.
11. Zona waktu pengukuran, arti kode sensor/kualitas, latensi pelaporan dan penyebab transisi SO2 belum dikonfirmasi penyedia. Tidak menyimpulkan aktivitas vulkanik atau kerusakan sensor. Nilai nol/tinggi/konstan dan prediksi negatif tidak dihapus dalam arsip.
12. Cache runtime dan log internal *.inspect.ndjson tidak disalin; log tersebut diinventarisasi dengan ukuran/hash pada berkas_tidak_disalin.csv. Pratinjau QA dan grafik ilmiah yang ada tetap disalin.

Berkas utama yang benar-benar hilang dari daftar wajib: 0. Checkpoint tersedia: 36/36; history permodel tersedia: 36/36.
