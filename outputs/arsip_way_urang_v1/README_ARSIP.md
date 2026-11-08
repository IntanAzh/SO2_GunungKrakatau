# Arsip Way Urang v1

Dibuat UTC: 2026-11-08T13:56:44.888592+00:00

Snapshot byte berkas yang sudah tersedia, dengan tambahan dokumen inventaris/arsip yang ditandai jelas. Tidak ada pemindahan, penghapusan, training ulang, evaluasi test, atau penciptaan checkpoint/hasil eksperimen yang tidak ada.

- [Ringkasan dan keputusan sementara](ringkasan_eksperimen.md)
- [Keterbatasan dan berkas tidak tersedia](catatan_keterbatasan.md)
- `inventaris_tahap.json`: kelengkapan per tahap dan daftar checkpoint/history.
- `konfigurasi_dependensi_seed.json`: indeks konfigurasi, dependensi dan seed; config penuh berada di salinan/outputs/.
- `referensi_excel_sumber.json`: lokasi dan hash Excel asli yang tidak digandakan.
- `manifest.json`: daftar otoritatif lokasi, ukuran byte, SHA256, asal, peran dan tahap semua payload, termasuk manifest.csv.
- `manifest.csv`: inventaris payload sebelum berkas kontrol manifest ditambahkan.
- `manifest.sha256`: hash manifest.json. Dua berkas kontrol tidak meng-hash dirinya sendiri; keduanya diverifikasi melalui hash manifest dan kecocokan isi kontrol. Tidak ada siklus self-hash.
- `berkas_tidak_disalin.csv`: pengecualian cache/log internal yang eksplisit.
- `salinan/`: struktur relatif proyek dipertahankan; hasil/konfigurasi/notebook/skrip/checkpoint/grafik adalah salinan byte, bukan hasil ulang.

ZIP memiliki satu folder akar arsip_way_urang_v1. Tidak ada file tambahan di luar set manifest+dua kontrol; semua isi ZIP diperiksa ukuran/SHA256 atau isi kontrolnya, serta CRC. Bukti verifikasi dan SHA256 ZIP disimpan di luar ZIP sebagai `../arsip_way_urang_v1.verifikasi.json` agar tidak menimbulkan hash melingkar.

Environment biner tidak disertakan. Notebook/skrip adalah catatan historis, bukan perintah yang dijalankan oleh proses pengarsipan. Reproduksi memerlukan penyiapan dependensi dan sumber eksternal yang tercatat.
