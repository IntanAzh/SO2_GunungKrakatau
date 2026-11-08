"""Menulis temuan penelusuran dari tabel, tanpa fitting atau koreksi hasil lama."""
from pathlib import Path
import pandas as pd
from write_pilot_report import table

OUT=Path(__file__).resolve().parents[1]/'outputs'

def write_report(t,threshold):
    df=pd.DataFrame(t['Ringkasan']);high=pd.DataFrame(t['SO2_tinggi_per_tahap'])
    methods=list(df.Pipeline.unique());scope=df[(df.Pipeline==methods[0])&df.Tahap.str.startswith('08')].iloc[0]
    inventory=t['Inventaris_train_asli'][0]
    lines=['# Penelusuran pembentukan sampel tahap kelima — Way Urang','',
        '**Pemeriksaan baca-saja terhadap catatan eksperimen lama. Tidak ada model, imputer, '
        'atau scaler yang dilatih/dijalankan kembali. Split tidak berubah, test tidak dimuat '
        'nilainya atau dievaluasi, dan hasil tahap kelima tidak ditimpa.**','',
        '## 1. Ringkasan temuan','',
        f"Train asli memiliki **{inventory['SO2 asli teramati']} pengamatan SO₂** dengan rentang "
        f"**{inventory['Minimum']:g}–{inventory['Maksimum']:g} µg/m³**. Train bersama lama memiliki "
        f"**12.654 target, 1,04–{threshold:g} µg/m³**. Penyempitan bukan karena tidak adanya "
        'konsentrasi tinggi pada sumber dan bukan filter berdasarkan besar SO₂.','',
        '- Terdapat **5.289 pengamatan train asli di atas 2,14 µg/m³**. Kalibrasi 90 hari '
        'dan buffer lag mengeluarkan 1.448 di antaranya; 3.841 masih tersedia setelah '
        'pembatasan tersebut.',
        '- **Seluruh 3.841 target lebih tinggi yang tersisa tidak memiliki fitur Curah Hujan '
        'lag1 finite pada P1.** Curah hujan asli kosong 24 Januari 2025 08:30 sampai '
        '22 Juni 2025 22:30: **7.181 slot = 3.590,5 jam**. Ffill empat slot tidak dapat '
        'mengisi rentetan sepanjang itu.',
        '- Median masih mempunyai 3.841 target tinggi yang layak mandiri; KNN mempunyai '
        '3.786. Irisan train yang mensyaratkan semua pipeline lengkap ikut mengeluarkan '
        'semuanya dari median dan KNN. Model bersama baru mulai memakai origin '
        '22 Juni 2025 23:30 (target 23 Juni 00:00).','',
        '## 2. Semesta timestamp dan urutan penelusuran','',
        'Inventaris train murni mempunyai **21.119 slot dan 19.197 SO₂ teramati** pada '
        'timestamp train. Untuk mengikuti kode supervised, waterfall memakai **21.119 '
        'origin train**, masing-masing dengan target t+30 menit. Sebelum purge terdapat '
        '**19.198 target teramati**: slot pertama train yang kosong tidak mempunyai origin '
        'dalam train, dan target pertama validasi yang teramati muncul dari origin train '
        'terakhir. Satu target lintas batas ini ditandai dan dipurge.',
        'Urutan tabel adalah waterfall masker diagnostik untuk memisahkan dampak aturan, '
        'bukan perubahan urutan fungsi kode. **Lag struktural** hanya memeriksa apakah '
        'posisi waktu t−48 ada; nilai lag yang kosong belum dikeluarkan. Pada kode lama, '
        'imputasi seri dilakukan sebelum pembentukan nilai fitur lag. Kode juga menetapkan '
        'batas origin purge sebelum filter finite; operasi masker komutatif dan hasil '
        'akhir penelusuran dicocokkan persis dengan ledger train lama.',
        'Statistik minimum/median/maksimum selalu memakai target asli yang teramati. '
        'Pada tahap pertama, jumlah sampel masih mencakup target kosong. Tanggal bulanan '
        'mengacu timestamp target. Kolom `Target melintasi validasi` mencegah pasangan '
        'batas waktu disalahartikan sebagai label train.','',
        '### Inventaris train asli per bulan','',
        table(['Bulan','Slot train','SO2 asli teramati','Minimum','Median','Maksimum','Di atas maksimum final'],t['Inventaris_train_asli']),'']
    for method in methods:
        rows=[r for r in t['Ringkasan'] if r['Pipeline']==method]
        lines += [f'### {method}','',table(['Tahap','Jumlah sampel','Target asli teramati','Origin awal','Origin akhir',
            'SO2 target minimum','SO2 target median','SO2 target maksimum','Keluar dari tahap sebelumnya','Target melintasi validasi'],rows),'']
    lines += ['Tabel **Tahap_per_bulan** dalam workbook menyajikan jumlah dan rentang '
        'SO₂ per bulan untuk seluruh tahap dan pipeline. **Tahap 09 layak mandiri '
        'bukan model yang pernah dilatih**; model lama ketiganya memakai irisan tahap 10.','',
        '## 3. Alasan sampel keluar dan penanda tumpang tindih','',
        '- `Penanda_semua_timestamp` memuat satu baris untuk setiap origin train, '
        'target asli, alasan dasar, penanda missing asal, alasan per pipeline dan '
        'status aktual bersama. Penanda boolean lebih rinci tersedia pada '
        '`penelusuran_tahap5/penanda_lengkap_per_timestamp.csv`.',
        '- **Target kosong:** tidak boleh diisi untuk melatih/menilai. '
        '**Histori struktural belum tersedia:** posisi lag di luar cakupan waktu. '
        '**Kalibrasi/buffer:** pembatasan rancangan, bukan nilai rusak.',
        '- **SO₂ historis/kovariat asli kosong** hanya penanda asal. Nilai tersebut '
        'masih berkesempatan diisi. Baris baru ditolak bila fitur hasil preprocessing '
        'tetap nonfinite atau aturan dasar lain tidak terpenuhi.',
        '- **P1:** `P1_BELUM_ADA_DONOR` dan `P1_UMUR_DONOR_LEBIH_4` adalah kebijakan '
        'forward fill; hasil imputasi tidak mereset umur donor. Bukan exception algoritma.',
        '- **P2:** `P2_GUARD_KOVARIAT_KURANG_2` berarti query tidak ditransformasi '
        'karena tidak memenuhi minimum dukungan. Ini pembatasan rancangan, bukan '
        'kegagalan komputasi KNN. Sisa NaN tak terjelaskan, jika ada, ditampilkan '
        'terpisah dan tidak diberi penyebab rekaan.',
        '- **Irisan:** `LAYAK_MANDIRI_KELUAR_IRISAN` berarti pipeline itu sebenarnya '
        'memiliki fitur dan target yang layak, tetapi pipeline lain tidak. Ini '
        'seleksi sampel bersama, bukan kegagalan imputasi pipeline tersebut.',
        '- Alasan dapat overlap pada timestamp yang sama. `Alasan_overlap` tidak '
        'boleh dijumlah sebagai jumlah unik yang keluar. `Kombinasi_alasan` memakai '
        'kombinasi tepat, sedangkan kolom keluar waterfall menghitung kehilangan '
        'berurutan yang tidak dihitung ulang pada tahap berikutnya.','',
        table(['Pemeriksaan','Hasil'],t['Pemeriksaan_implementasi']),'',
        '### Sampel yang layak sendiri tetapi hilang karena irisan','',
        table(['Pipeline','Layak sendiri total','Aktual dipakai bersama','Jumlah sampel',
            'SO2 target minimum','SO2 target median','SO2 target maksimum','Target di atas maksimum final'],t['Kehilangan_karena_irisan']),'',
        '### Fitur yang menolak target tinggi setelah kalibrasi/buffer','',
        table(['Pipeline','Fitur','Sisa NaN dalam scope','Sisa NaN target tinggi dalam scope','Denominator target tinggi scope'],
            [r for r in t['Fitur_penyebab_keluar'] if r['Sisa NaN target tinggi dalam scope']>0]),'',
        'Scope adalah target train asli teramati dengan origin>=4.368. '
        'Hitungan fitur tumpang tindih: satu sampel dapat kehilangan beberapa lag '
        'SO₂ dan beberapa kovariat sekaligus. Pada P1, Curah Hujan saja sudah '
        'mencakup seluruh 3.841 target tinggi dalam scope.','',
        '## 4. Periode SO₂ lebih tinggi pada train asli','',
        f"Lebih tinggi didefinisikan **SO₂ > {threshold:g} µg/m³**, yakni di atas maksimum "
        'target train bersama lama. Ini ambang deskriptif untuk pertanyaan penelusuran, '
        'bukan batas sensor, batas outlier atau alasan menghapus nilai. Observasi tinggi '
        'muncul dari **18 Januari 2025 23:30 sampai 22 Juni 2025 17:00**. Rentetan '
        'berikut terputus ketika ada slot kosong atau nilai tidak melebihi ambang.','',
        table(['Periode','Awal target tinggi','Akhir target tinggi','Jumlah observasi','Minimum SO2','Median SO2','Maksimum SO2'],t['Periode_SO2_lebih_tinggi']),'']
    for method in methods:
        rows=[r for r in t['SO2_tinggi_per_tahap'] if r['Pipeline']==method]
        lines += [f'### Target tinggi yang bertahan: {method}','',table(['Tahap','Jumlah sampel','SO2 target minimum',
            'SO2 target median','SO2 target maksimum'],rows),'']
    lines += ['Workbook juga memuat **Periode_tinggi_per_tahap** untuk jumlah bertahan '
        'dan keluar pada setiap rentetan, serta **Tahap_per_bulan** untuk distribusi '
        'bulanan. Tidak ada konsentrasi tinggi yang dihapus dari sumber atau dinyatakan salah.','',
        table(['Awal pengukuran asli kosong','Akhir pengukuran asli kosong','Jumlah slot','Durasi slot jam'],t['Gap_Curah_Hujan_train']),'',
        '## 5. Mengapa cakupan KNN 99,96% dan median 100%','',
        'KNN menghasilkan **4.522/4.524 = 99,955791%** prediksi, median '
        '**4.524/4.524 = 100%**. Dua origin berikut tidak memenuhi guard minimum '
        'dua kovariat asli untuk mengisi fitur KNN. Median memiliki statistik train '
        'untuk mengisi kovariat tersebut, sehingga tetap dapat membuat prediksi.',
        'Pada origin 8 April hanya satu kovariat lag tersedia, sedangkan pada '
        '21 April tidak ada. SO₂ origin sendiri teramati pada keduanya. Karena '
        'guard menghitung **kovariat asli**, SO₂ origin tidak menggantikan '
        'syarat dua kovariat. Keputusan ini sudah tercantum pada rancangan lama.','',
        table(['Origin','Target','SO2 target asli','SO2 origin asli','Timestamp kovariat yang dicari',
            'Jumlah kovariat asli tersedia','Kovariat asli tersedia','Fitur Ridge masih kosong','Penyebab'],t['KNN_tidak_ada_prediksi']),'',
        'Fitur yang masih kosong dan nilai pengisi median ditampilkan di '
        '**KNN_fitur_kosong**. Tidak ada fallback median tersembunyi pada KNN '
        'atau forward fill, dan tidak ada prediksi baru dihitung.','',
        '## 6. Persistence dan kesamaan timestamp evaluasi','',
        'Persistence memakai SO₂ asli pada origin jika tersedia. Bila SO₂ origin '
        'kosong, ia memakai pengamatan asli terakhir yang lebih lama. Distribusi '
        'usia berikut dihitung ulang dari sumber prefix train-validasi, bukan '
        'sekadar mempercayai kolom usia yang disimpan. Usia satu slot=30 menit.','',
        table(['Domain','Usia slot','Usia menit','Jumlah prediksi','Denominator prediksi domain','Proporsi persen'],t['Usia_persistence']),'',
        table(['Origin','Target','Waktu donor asli','SO2 donor asli','Usia slot','Usia menit','Dalam irisan'],t['Persistence_donor_lama']),'',
        'Pada seluruh target validasi, **4.522 prediksi memakai SO₂ tepat pada '
        'origin**, dan **2 memakai donor 30 menit lebih lama**. Pada irisan '
        '4.487 target, pembagiannya 4.485 dan 2. Asumsi bahwa pengukuran pada '
        'timestamp itu sudah tiba tetap bergantung pada latensi yang belum dikonfirmasi.',
        'Seluruh delapan baris skor agregat tahap kelima dihitung ulang dari '
        'target asli dan prediksi tersimpan; hasilnya cocok. Keempat metode pada '
        'domain **irisan semua metode** mempunyai SHA256 daftar timestamp '
        'target yang identik. Skor domain **seluruh berhasil** memang memakai '
        'himpunan berbeda dan harus dibaca bersama cakupan, bukan dianggap '
        'perbandingan timestamp identik.','',
        table(['Metode','Domain','Jumlah target asli','MAE hitung ulang','RMSE hitung ulang','Bias hitung ulang','Sesuai skor lama'],t['Verifikasi_skor_timestamp']),'',
        '## 7. Kesimpulan dan usulan koreksi minimum','',
        '1. **Penyebab rentang sempit:** kalibrasi/buffer mengeluarkan 1.448 '
        'pengamatan tinggi awal, kemudian persyaratan Curah Hujan pada P1 '
        'mengeluarkan seluruh 3.841 pengamatan tinggi yang tersisa. Irisan '
        'bersama memindahkan pembatasan P1 itu ke median dan KNN. Purge '
        'hanya mengeluarkan satu target lintas batas dan bukan penyebab hilangnya '
        'konsentrasi tinggi.',
        '2. **Implementasi versus rancangan:** tidak ditemukan complete-case '
        'dropna pada kolom yang tidak dipakai atau pembuangan nilai lag kosong '
        'sebelum kesempatan imputasi. Himpunan akhir cocok dengan ledger model '
        'lama. Ini terutama **konsekuensi rancangan train bersama dan kelengkapan '
        'kovariat**, dengan bias seleksi terhadap periode konsentrasi lebih '
        'tinggi; bukan bukti konsentrasi tinggi salah. Label “kegagalan imputasi” '
        'perlu dibedakan dari penolakan guard/batas donor yang disengaja.',
        '3. **Koreksi minimum sekarang:** lengkapi interpretasi catatan tahap 5 '
        'dengan hasil penelusuran ini. Skor dan artefak lama dipertahankan '
        'sebagai hasil eksperimen pada kohort terbatas; jangan menyebutnya '
        'perbandingan yang sudah mewakili rentang 0,92–80,71 µg/m³.',
        '4. **Usulan eksperimen berikut, belum dijalankan:** pertimbangkan '
        'ablasi Curah Hujan secara konsisten pada seluruh pipeline karena '
        'ketersediaannya membatasi cakupan. Tetapkan ulang daftar fitur dan '
        'peran curah hujan sebagai fitur model/fitur bantu imputer sebelum '
        'fit. Pertahankan split, target asli, kausalitas dan evaluasi bersama; '
        'simpan sebagai eksperimen baru. Jangan memberi median fallback '
        'diam-diam pada P1 untuk menyelamatkan sampel.',
        '5. **Simulasi eligibility saja:** mengabaikan dua kolom Ridge terkait '
        'Curah Hujan pada pemeriksaan finite atas histori lama akan menyisakan '
        '**16.530 sampel bersama**, termasuk **3.786 target >2,14**, dengan '
        'rentang **1,04–47,81 µg/m³**. Ini bukan hasil pipeline/model baru '
        'dan tidak memprediksi skornya. Rentang 80,71 tetap tidak kembali '
        'karena pembatasan kalibrasi; bila periode awal penting untuk tujuan '
        'penelitian, desain kalibrasi awal yang lebih sesuai perlu dipraregistrasi '
        'sebagai sensitivitas berbasis dukungan data, tanpa memilih dari skor test.',
        '6. **Alternatif, bukan perbaikan diam-diam:** melatih tiap pipeline '
        'pada sampel layaknya sendiri akan memisahkan pertanyaan cakupan, '
        'tetapi membuat komposisi train berbeda. Jika kelak dilakukan, laporkan '
        'sebagai estimand/perbandingan terpisah, tetap tampilkan train bersama '
        'dan evaluasi pada target bersama. Tidak ada pelatihan tersebut pada '
        'pemeriksaan ini.','',
        table(['Skenario','Jumlah sampel','Target di atas maksimum final','SO2 target minimum','SO2 target median','SO2 target maksimum','Catatan'],t['Simulasi_eligibility_tanpa_rain']),'',
        '## 8. Integritas dan artefak','',
        '- Workbook: `penelusuran_sampel_tahap5.xlsx`; penanda lengkap tambahan: '
        '`penelusuran_tahap5/penanda_lengkap_per_timestamp.csv`.',
        '- Semua sumber dan artefak tahap kelima yang dibaca dicatat hash sebelum '
        'dan sesudah. Tidak ada model/scaler/imputer dijalankan dan tidak ada '
        'hasil sebelumnya ditimpa.','']
    lines += [f"- `{r['File']}`: SHA256 sebelum = sesudah = `{r['SHA256 sebelum']}`." for r in t['Integritas_SHA256']]
    (OUT/'penelusuran_sampel_tahap5.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
