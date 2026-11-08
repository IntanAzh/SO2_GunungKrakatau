"""Menulis rancangan usulan; tidak menjalankan algoritma preprocessing/model."""
from pathlib import Path
from collections import Counter

OUT=Path(__file__).resolve().parents[1]/'outputs'

def table(columns,rows):
    def fmt(v):
        if v is None:return '—'
        if isinstance(v,float):return f'{v:.2f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+
        ['| '+' | '.join(fmt(r.get(c)) for c in columns)+' |' for r in rows])

def write_design(t):
    off=t['Offgrid_13']; rec=t['Rekonsiliasi_grid']
    off_so2=sum(r['SO2 nonkosong off-grid terpisah'] for r in rec)
    lines=['# Pemeriksaan lanjutan dan rancangan eksperimen preprocessing SO₂','',
        '**Status: usulan eksperimen, belum dijalankan.** Pemeriksaan menggunakan hasil audit tahap kedua, '
        'grid audit yang sudah tersedia, serta pembacaan ulang sumber hanya untuk verifikasi. '
        'Tidak ada imputasi, pembulatan waktu, agregasi, penghapusan nilai, scaling, pemilihan stasiun utama atau pelatihan model.','',
        '## 1. Definisi dan sumber','',
        '- Acuan: `notebooks/01_audit_data.ipynb`, `notebooks/02_audit_kualitas_data.ipynb`, '
        '`audit_struktur_data.json`, `audit_kualitas_tables.json`, `audit_kualitas_data.xlsx`, dan CSV grid tahap kedua.',
        '- Workbook `pemeriksaan_lanjutan_kualitas.xlsx` memuat rekonsiliasi, semua 13 off-grid, 39 skenario pemetaan hipotetis, '
        '80 rentetan SO₂, 960 pemeriksaan variabel lain, kelayakan setiap stasiun dan SHA256.',
        '- SO₂ teramati grid (%) = 100 × slot dengan sedikitnya satu nilai SO₂ nonkosong / seluruh slot grid 30 menit stasiun. '
        'Nol, nilai berulang dan nilai tinggi tetap teramati, belum otomatis valid.',
        '- Kehilangan awal/tengah/akhir dilaporkan dengan dua denominator: seluruh slot grid, dan seluruh slot SO₂ hilang. '
        'Tiga posisi tersebut mencakup seluruh kehilangan pada delapan stasiun ini.',
        '- Kelengkapan meteorologi grid (%) memakai slot nonkosong variabel / seluruh slot grid. '
        'Tabel rinci juga menyajikan persentase nonkosong per seluruh baris sumber; denominator keduanya tidak dipertukarkan.',
        '- Durasi slot rentetan konstan = jumlah slot × 30 menit. Rentang waktu = akhir − awal, lebih pendek satu slot. '
        'Pengelompokan gap tahap kedua tetap deskriptif dan belum menjadi batas imputasi.',
        '- Tidak ada asumsi zona waktu, kode sensor, penyebab anomali, batas sensor atau kaitan dengan aktivitas Anak Krakatau.','',
        '## 2. Verifikasi timestamp di luar pola 00/30','',
        f'Ditemukan kembali **{len(off)} baris off-grid**, dengan **{off_so2} nilai SO₂ nonkosong**. '
        'Timestamp sebelum/sesudah berikut mengikuti urutan baris asli. Selisih dalam menit.','',
        table(['Stasiun','Baris Excel','Timestamp asli','SO2 asli','Waktu sebelum','Selisih dari sebelum menit','Waktu sesudah','Selisih ke sesudah menit'],off),'',
        '### Rekonsiliasi terhadap tahap kedua','',
        table(['Stasiun','Seluruh baris sumber','Baris pada grid','Baris off-grid terpisah',
               'SO2 nonkosong seluruh baris','SO2 nonkosong pada grid','SO2 nonkosong off-grid terpisah'],rec),'',
        'Untuk setiap stasiun, jumlah baris pada grid + off-grid = seluruh baris sumber. '
        'Jumlah SO₂ nonkosong pada grid + off-grid = seluruh SO₂ nonkosong sumber. Tidak ada timestamp gagal atau duplikat pada data ini. '
        'Numerator kelengkapan tahap kedua sama dengan SO₂ nonkosong **pada grid**. '
        'Off-grid sudah dihitung dalam statistik per baris sumber, tetapi dikeluarkan dari numerator kelengkapan grid. '
        'Jadi reindex grid saja tidak merepresentasikan seluruh pengamatan. Baris off-grid dipertahankan sebagai ledger terpisah; '
        'hasil grid tidak boleh digunakan sebagai pengganti sumber atau satu-satunya inventaris.','',
        '### Konsekuensi kebijakan waktu — belum diterapkan','',
        '- **Pertahankan timestamp asli:** menyimpan semua pengamatan dan kronologi. Model reguler perlu ledger off-grid, '
        'fitur selisih waktu/mask, atau kebijakan pembentukan fitur tersendiri; timestamp tidak boleh diam-diam dibuang.',
        '- **Petakan ke slot floor/ceil/terdekat:** mengubah label waktu dan berpotensi menempatkan beberapa baris pada satu slot. '
        'Simpan timestamp asli, slot kandidat, selisih dan identitas baris. Putuskan kebijakan benturan dengan penyedia data; '
        'jangan mengambil baris pertama, terakhir atau rata-rata tanpa alasan. Pemetaan terdekat/floor dapat menempatkan '
        'pengamatan lebih awal daripada waktu aslinya, sehingga dapat membocorkan informasi bila dianggap tersedia di slot lebih awal.',
        '- **Agregasi interval:** misalnya interval berlabel akhir `(t−30 menit, t]`, hanya menggunakan pengamatan yang sudah '
        'tersedia sampai t. Mean/median mengubah nilai, bobot dan makna observasi; jumlah sampel, konflik dan cakupan interval '
        'harus disimpan. Jangan mengagregasi sebelum diketahui apakah timestamp menyatakan waktu sesaat, awal/akhir interval '
        'atau rata-rata pengukuran. Label interval tidak menentukan waktu kedatangan data.',
        '- **Ketersediaan aktual:** gunakan waktu kedatangan/latensi bila tersedia. Selama metadata itu belum ada, '
        'jangan menganggap nilai tersedia sebelum timestamp aslinya; eksperimen harus menyatakan asumsi latensi yang dipakai.',
        '- Tabel `Pemetaan_hipotetis` hanya menghitung kandidat dan kemungkinan benturan, tidak mengubah timestamp atau nilai.','']
    for method in ['floor','ceil','terdekat']:
        candidates=[r for r in t['Pemetaan_hipotetis'] if r['Skenario saja']==method]
        collisions=[r for r in candidates if r['Benturan']]
        lines.append(f"- Skenario {method}: {len(collisions)} dari {len(candidates)} baris off-grid akan berbenturan dengan baris lain pada slot kandidat. Jumlah ini menghitung baris off-grid terdampak, bukan jumlah slot unik.")
    lines+=['','## 3. Pemetaan rentetan SO₂ konstan','',
        'Sepuluh rentetan terpanjang tiap stasiun disalin dari hasil tahap kedua. '
        'Pemeriksaan variabel lain memakai seluruh baris asli pada interval awal–akhir, termasuk off-grid. '
        'Konstan pada nilai tersedia tidak berarti lengkap di seluruh interval. '
        'Nilai kosong dan jumlah unik tersedia ditampilkan pada `Konteks_variabel`; tidak ada label kerusakan sensor.','']
    for s in t['Ringkasan']:
        station=s['Stasiun']; runs=[r for r in t['Top10_konstan_SO2'] if r['Stasiun']==station]
        figure=next(r for r in t['Grafik_konstan'] if r['Stasiun']==station)
        lines+= [f'### {station}','',table(['Peringkat','Nilai SO2','Awal','Akhir','Jumlah slot','Durasi slot jam','Rentang waktu jam','Penanda deskriptif'],runs),'',
            f"![Konteks rentetan utama {station}]({figure['Grafik relatif laporan']})",'',
            'Grafik menampilkan maksimum tujuh hari sebelum, seluruh rentetan, dan maksimum tujuh hari sesudah. '
            'Celah tetap terlihat; titik off-grid ditandai terpisah. Konteks dibatasi rentang sumber.','']
    lines+=['## 4. Kelayakan kualitas setiap stasiun','',
        'Tabel ini **bukan peringkat dan bukan pemilihan stasiun utama**. '
        'Kesesuaian lokasi memerlukan pertanyaan penelitian, koordinat, lingkungan stasiun dan representativitas spasial. '
        'Persentase grid antarlokasi memiliki rentang tanggal berbeda.','',
        table(['Stasiun','Awal pengamatan','Akhir pengamatan','Denominator slot grid','SO2 teramati grid persen',
               'Hilang awal persen grid','Hilang tengah persen grid','Hilang akhir persen grid',
               'Gap tengah terpanjang jam','Konstan SO2 terpanjang slot'],t['Ringkasan']),'',
        'Proporsi kehilangan di atas memakai seluruh slot grid sebagai denominator. '
        'Proporsi terhadap seluruh slot hilang dan jumlah slot tersedia pada workbook.','',
        '### Kelengkapan meteorologi (% slot grid teramati)','',
        table(['Stasiun']+[f'{v} teramati grid persen' for v in
            ['Kec.Angin','Arah Angin','Kelembaban','Suhu','Tek.Udara','Sol.Rad','Curah Hujan']],t['Ringkasan']),'',
        table(['Stasiun','Masalah untuk diperiksa','Implikasi eksperimen'],t['Ringkasan']),'',
        '## 5. Kontrak eksperimen yang diusulkan','',
        'Semua pilihan berikut adalah **parameter rancangan untuk diuji**, bukan hasil eksperimen atau batas sensor. '
        'Tidak ada metode terbaik yang ditentukan. Tidak ada paket imputation yang dipasang atau algoritma yang dijalankan pada tahap ini.','',
        '### Informasi yang boleh digunakan','',
        '- Nyatakan origin prediksi sebagai `t` dan horizon sebagai `h`. Input mencakup informasi yang telah tersedia sampai t; '
        'target adalah SO₂ asli pada `t+h`. Variabel polutan/meteorologi pada `t+h` tidak boleh menjadi input kecuali '
        'berupa prakiraan eksternal yang memang sudah tersedia pada t dan dievaluasi sebagai skenario berbeda.',
        '- Pisahkan kolom SO₂ historis untuk masukan dari kolom target. Jangan memasukkan target, lead target, hasil imputasi '
        'target atau variabel masa depan ke matriks imputasi. Target evaluasi harus nilai pengamatan sumber yang tersedia '
        'pada timestamp tepat dan telah melewati pemeriksaan yang disepakati. Target yang hilang tetap tidak memiliki label.',
        '- Uji keluarga fitur F0 = SO₂ historis dan mask/umur pengamatan; F1 = F0 + kovariat polutan dan meteorologi yang '
        'telah tersedia pada origin. Gunakan daftar fitur yang sama antar P0–P3 dalam satu perbandingan. '
        'KNN/iterative memerlukan informasi prediktor; F0 tanpa fitur lain/lag dapat menghasilkan fallback median atau model '
        'yang tidak informatif, sehingga harus dicatat. Jangan menambah stasiun donor lain secara implisit.',
        '- Variabel awal kandidat: SO₂ historis, PM10, PM2.5, CO, O3, NO2, Kec.Angin, Arah Angin, Kelembaban, Suhu, '
        'Tek.Udara, Sol.Rad, Curah Hujan. Timestamp dan ID tidak diimputasi. Metadata yang ambigu/kolom seluruhnya '
        'kosong pada train tidak memiliki dukungan estimasi; tandai tidak tersedia dan gunakan skenario fitur bersama '
        'yang ditetapkan dari train, tanpa mengisinya nol secara otomatis.',
        '- Nol dan pengulangan tetap dipertahankan pada skenario dasar. Rentetan deskriptif dipakai untuk pelaporan strata '
        'dan analisis sensitivitas setelah bukti tersedia; jangan otomatis memask seluruhnya sebagai salah. '
        'Nilai di luar rentang pemeriksaan tetap dilacak; kebijakan perbaikannya menunggu dokumentasi.',
        '- Untuk eksperimen imputasi online, origin lokal adalah waktu slot yang dimask `u`: gunakan informasi tersedia '
        'sampai u, tanpa nilai kanan u. Saat forecasting di t, gunakan jejak hasil imputasi kausal yang dibentuk saat '
        'masing-masing slot historis tersedia; jangan memperbaiki ulang histori dengan informasi masa depan.',
        '- Setiap hasil turunan menyimpan mask asli, mask sintetis, status teramati/diimputasi/tidak terselesaikan, '
        'metode, parameter, sumber donor, origin, versi split dan umur pengamatan terakhir. Semua artefak terpisah dari sumber.','',
        '- Untuk pembentukan contoh forecasting di dalam train secara kausal, gunakan prefix kalibrasi awal '
        'atau expanding out-of-fold fit yang berakhir sebelum origin masing-masing. Jangan menerapkan '
        'median/scaler/donor yang dipelajari dari seluruh train untuk meniru prediksi pada origin awal train. '
        'Train beku penuh hanya dapat digunakan untuk origin validasi/test yang datang setelah akhir train.','',
        '### P0 — median data pelatihan','',
        '- Proses sel input yang hilang pada daftar fitur bersama, per variabel dan per stasiun. Median hanya dari '
        'nilai teramati pada train; gunakan kembali parameter beku pada validasi/test. SO₂ target tidak diisi.',
        '- Baseline sederhana untuk gap titik/blok, bukan dugaan bahwa median mewakili blok panjang atau pola musiman. '
        'Bandingkan fitur F0/F1 dan kebijakan guard gap yang sama dengan metode lain. Median global train sebagai P0 '
        'tetap dibedakan dari median musiman yang akan menjadi varian tambahan, bukan diam-diam mengubah baseline.',
        '- Tidak memerlukan scaling untuk perhitungan median. Jika model forecasting memerlukan scaling, scaler '
        'dipelajari dari train saja. Kolom tanpa median train tetap tidak terselesaikan.',
        '- Untuk pembandingan diagnostik, laporkan hasil baseline pada seluruh sel yang memiliki median serta hasil '
        'dengan guard bersama. Nilai median pada gap sangat panjang tetap berstatus sintetis dan belum dinyatakan andal.','',
        '### P1 — forward fill terbatas','',
        '- Proses hanya input historis, per variabel/per stasiun. Donor adalah pengamatan **asli** terakhir yang sudah '
        'tersedia; bukan backfill, interpolasi dua sisi, atau nilai diimputasi yang mereset umur donor.',
        '- Uji batas umur donor `L = {1, 2, 6, 12, 48}` slot = `{30 menit, 1, 3, 6, 24 jam}`. '
        'Pilihan ini mengikuti kategori deskriptif audit untuk perbandingan, bukan batas aman final. '
        'Kandidat >24 jam hanya stress test terpisah jika diputuskan setelah memeriksa dukungan train.',
        '- Isi slot pertama hingga L secara sekuensial; sesudah umur donor melebihi L, sisanya tetap hilang. '
        'Jangan melihat total panjang gap yang baru diketahui setelah seri lengkap untuk menentukan boleh/tidaknya '
        'mengisi slot awal gap. Gap sebelum pengamatan pertama tidak mempunyai donor.',
        '- Tidak membutuhkan scaling. Cocok diuji pada gap pendek; blok panjang dan periode tanpa donor awal '
        'akan membatasi cakupan. Carry state dari train ke validasi boleh hanya jika nilai terakhir memang '
        'tersedia sebelum origin; data validasi tidak digunakan untuk estimasi parameter.','',
        '### P2 — KNN imputation','',
        '- Proses input historis memakai fitur numerik/lag kausal yang sama. Uji `n_neighbors={3,5,10}`, '
        '`weights={uniform,distance}`, dan donor train seluruhnya versus maksimum 90 hari terakhir dalam train. '
        'Ini ruang parameter usulan, dapat diperkecil berdasarkan ukuran dan variasi train.',
        '- KNN berbasis jarak memerlukan skala fitur yang sebanding: bandingkan standardisasi dan skala robust '
        'dengan parameter hanya dari nilai train teramati, sambil mempertahankan NaN sampai imputasi. '
        'Kolom tanpa variasi/train kosong ditangani eksplisit, bukan dibagi nol.',
        '- Fit donor hanya pada train dengan timestamp **sebelum origin evaluasi**. Validasi/test ditransformasi '
        'satu baris asal waktu secara kausal, tidak digabung ke donor dan tidak memakai tetangga dari seluruh seri. '
        'Untuk masking pada train gunakan prefix/fold train yang berakhir sebelum slot mask; tidak boleh '
        'fit_transform seluruh train lalu menganggap estimasi untuk baris awal bersifat kausal.',
        '- Laporkan jumlah prediktor yang tumpang tindih dan donor yang benar-benar mendukung variabel. '
        'Uji syarat overlap minimal `{2,4}` fitur jika tersedia dan jumlah donor efektif minimal `{1,3}`. '
        'Jika dukungan tidak cukup, jangan menyembunyikan fallback; tandai tidak terselesaikan pada skenario ketat. '
        'Fallback mean bawaan bila jarak tidak dapat ditentukan harus ditangkap dan dilaporkan, atau dicegah '
        'dengan pemeriksaan donor sebelum transform.',
        '- Gap panjang/multivariat serentak dapat tidak memiliki overlap yang informatif; tidak ada jaminan '
        'KNN mengisi secara andal. Batas umur gap bersama dan status tidak terselesaikan tetap berlaku. '
        'Dokumentasi menjelaskan jarak nan-euclidean dan pilihan bobot pada '
        '[KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) '
        'dan [panduan nearest-neighbors imputation](https://scikit-learn.org/stable/modules/impute.html#nearest-neighbors-imputation).','',
        '### P3 — iterative imputation dengan Bayesian Ridge','',
        '- Proses matriks input historis dengan estimator `BayesianRidge`, bukan target masa depan. '
        'Uji `max_iter={10,20}`, `tol={1e-3,1e-4}`, urutan `{ascending,roman}`, dan inisialisasi median train. '
        'Gunakan `sample_posterior=False` pada perbandingan deterministik utama dan seed tetap, misalnya 42.',
        '- Gunakan skala fitur yang sama dengan skenario P2, dipelajari dari train; regresi berregularisasi '
        'perlu skala yang konsisten. Nilai meteorologi/polutan yang ambigu tetap memerlukan kebijakan terpisah. '
        'Jangan menetapkan clipping atau batas sensor rekaan pada prediksi hasil imputasi.',
        '- Fit rangkaian regresi hanya pada train sebelum origin, lalu transform masukan yang tersedia hingga '
        'origin. Iterasi terjadi antarfitur pada satu baris, bukan izin menggunakan baris waktu berikutnya. '
        'Pastikan desain lag tidak mengandung lead atau statistik jendela terpusat.',
        '- Relevan diuji bila hubungan kovariat train tersedia; kolom sangat kosong, gap serentak dan perubahan '
        'pola dapat mengurangi dukungan. Catat konvergensi, iterasi nyata, peringatan, dan prediksi di luar '
        'rentang yang perlu ditinjau. Bila tidak konvergen atau dukungan prediktor tidak cukup, '
        'tandai kegagalan/tidak terselesaikan, jangan diam-diam memakai metode lain.',
        '- Kolom tanpa observasi train tidak boleh memperoleh arti hanya karena perilaku default pustaka. '
        'Kelas ini masih berstatus experimental; versi dan pengaktifan fitur experimental harus dicatat '
        'saat kelak diimplementasikan. Lihat '
        '[IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html).','',
        '### Penanganan tambahan bersama dan gap yang belum andal','',
        '- Arah angin bersifat melingkar: nilai 0° dan 360° berdekatan. Jangan langsung memakai jarak '
        'linear/median derajat tanpa pertimbangan. Usulkan varian tanpa arah angin dan varian sin/cos dari '
        'arah yang telah dikonfirmasi; mask keduanya bersama, normalkan pasangan hasil estimasi bila '
        'bermakna, dan jangan menyimpulkan arah ketika norma mendekati nol. Belum ada transformasi dijalankan.',
        '- Gunakan mask dan umur donor untuk semua pipeline. Uji guard umur kehilangan yang sama `{1,2,6,12,48}` '
        'slot untuk P0/P2/P3 sebagai analisis sensitivitas; umur dihitung secara online dari pengamatan asli '
        'terakhir, bukan total panjang gap retrospektif. Laporkan juga cakupan tanpa guard sebagai diagnostik '
        'terpisah, bukan kelayakan otomatis.',
        '- Gap awal tanpa dukungan, gap panjang setelah guard, fitur tanpa estimasi train, konflik timestamp '
        'yang belum terselesaikan, atau dukungan kovariat tidak cukup tetap berstatus tidak terselesaikan. '
        'Jangan membuat rangkaian panjang sintetis untuk memperbesar sampel. Pada tahap forecasting kelak, '
        'laporkan origin yang tidak menghasilkan prediksi karena input belum memenuhi kontrak; '
        'skenario model yang mendukung mask harus ditetapkan terpisah dan sama antar-pipeline.','',
        '## 6. Rancangan evaluasi kausal','',
        '### Split dan estimasi parameter','',
        '1. Usulan awal: pisahkan berdasarkan rentang waktu, sekitar 60% train awal, 20% validasi berikutnya, '
        '20% test paling akhir; tanpa shuffle. Tanggal batas konkret ditetapkan setelah tujuan lokasi, '
        'horizon dan dukungan observasi diperiksa. Jangan memindahkan batas hanya untuk meningkatkan skor.',
        '2. Di bagian train, gunakan expanding-window validation dengan beberapa origin dan durasi validasi '
        'yang sebanding. Kelengkapan/gap pada **train saja** menentukan distribusi masking, daftar fitur, '
        'skala, median, donor dan parameter model imputasi. Audit seluruh seri hanya menjadi konteks, '
        'bukan sumber estimasi parameter eksperimen.',
        '3. Validasi dipakai memilih konfigurasi; test dikunci dan tidak dipakai memilih metode. '
        'Setelah konfigurasi tetap, refit pada train+validasi dapat menjadi train final yang dinyatakan '
        'jelas, lalu uji sekali pada test. Jangan memakai nilai test sebagai donor atau fit scaler.',
        '4. Origin dan target jendela yang dipakai melatih model harus selesai di dalam train: '
        'untuk horizon h, purging origin train yang targetnya mencapai validasi/test. Tetapkan embargo '
        'sesuai horizon dan latensi; histori train yang sudah tersedia boleh menjadi konteks validasi. '
        'Lookback boleh mengakses masa lalu, bukan target masa depan.',
        '5. Timestamp evaluasi reguler memakai slot asli tepat pada grid. Off-grid tidak dibulatkan '
        'menjadi target evaluasi. Jika kebijakan pemetaan kelak disepakati, kunci kebijakan itu sama '
        'untuk semua pipeline dan laporkan hasil sebagai skenario terpisah.',
        'Prinsip fit hanya pada train didukung panduan '
        '[pencegahan data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) '
        'dan pemisahan berurutan oleh '
        '[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). '
        'Grid tetap dipakai agar durasi fold sebanding, tanpa menganggap slot kosong sebagai label.','',
        '### Masking nilai teramati','',
        '- Bentuk ledger kebenaran dari nilai **asli teramati** yang lolos kriteria pemeriksaan yang disepakati. '
        'Jangan memakai nilai imputasi sebagai truth. Nol/konstan tidak otomatis dikeluarkan; laporkan '
        'strata rentetan konstan versus berubah dan analisis sensitivitas bila status belum jelas.',
        '- Masking titik terpisah: usulkan proporsi `{5%,10%,20%}` dari sel teramati yang eligible '
        'pada segmen evaluasi, menggunakan seed `{11,23,47,71,101}` yang sama antar-pipeline. '
        'Denominator proporsi masking adalah jumlah sel asli teramati yang eligible pada segmen itu, '
        'bukan seluruh grid dan bukan sel yang memang sudah hilang.',
        '- Masking blok: sampel panjang empiris gap **train** per stasiun, terstratifikasi kategori '
        '30 menit, >30 menit–1 jam, >1–3 jam, >3–6 jam, >6–24 jam dan >24 jam. '
        'Usulkan titik stress test 48 jam/7 hari hanya jika blok truth kontinu cukup panjang. '
        'Laporkan jumlah blok yang dapat ditempatkan; jangan memendekkan blok sulit tanpa mencatatnya.',
        '- Truth seluruh sel blok harus tersedia. Hapus seluruh blok sintetis dari fitur dan donor '
        'sebelum eksperimen; evaluasi sekuensial pada setiap slot di dalam blok tanpa mengintip '
        'nilai di ujung kanan. Distribusi gap alami yang melintasi missing asli tidak mempunyai '
        'truth lengkap sehingga tidak dapat langsung diberi skor rekonstruksi.',
        '- Uji SO₂ saja dan kehilangan serentak SO₂+kovariat untuk meniru baris tanpa pengamatan. '
        'Bedakan gap awal (cold start), tengah, dan akhir (truncation) dengan hanya data masa lalu. '
        'Simulasi gap awal tidak boleh memakai observasi sesudahnya untuk fit donor.',
        '- Gunakan mask yang sama antar P0–P3. Saat tuning pada bagian train, prefix fit berhenti '
        'sebelum origin mask; mask dalam prefix juga tidak boleh muncul dalam data fit, scaler '
        'atau donor. Pada evaluasi validasi/test, fit beku dari train menghindari donor masa depan.',
        '- Masking mengukur kemampuan pada observasi yang tersedia; belum membuktikan akurasi '
        'pada missing alami yang mekanismenya tidak diketahui. Laporkan batas generalisasi ini.','',
        '### Metrik dan cakupan imputasi','',
        'Definisikan error `e = estimasi − pengamatan asli`. Untuk N sel sintetis yang mempunyai '
        'truth dan berhasil diestimasi: `MAE = Σ|e|/N`, `RMSE = sqrt(Σe²/N)`, '
        '`bias = Σe/N`. Bias positif berarti estimasi terlalu tinggi. Metrik memakai satuan asli '
        '(SO₂ µg/m³), setelah inverse scaling sementara. N=0 dilaporkan tidak tersedia, bukan nol.',
        'Laporkan per stasiun, variabel, posisi/panjang gap, seed dan status konstan. '
        'Laporkan rata-rata dan sebaran antarulangan, bukan hanya penggabungan semua slot. '
        'Waktu komputasi dipisahkan menjadi fit, transform total, dan waktu per slot/baris, '
        'bersama perangkat, versi paket, ukuran donor dan jumlah fitur.',
        '- **Cakupan imputasi (%)** = 100 × sel mask eligible yang berhasil diestimasi / seluruh '
        'sel mask eligible dengan truth. Sel gagal/tidak terselesaikan tetap pada denominator.',
        '- **Cakupan input historis (%)** = 100 × slot input hilang alami yang dapat diproses '
        'sesuai kontrak / seluruh slot input hilang alami pada split dan fitur tersebut. '
        'Pisahkan keluaran finite dari keluaran yang memenuhi guard/dukungan.',
        '- Bandingkan metrik pada irisan sel yang berhasil untuk seluruh pipeline dan tampilkan '
        'metrik/cakupan tiap pipeline pada domainnya sendiri. Jangan menyatakan metode unggul '
        'hanya karena menolak kasus sulit. Daftar sel eligible/mask dan ketidakberhasilan disimpan.',
        '- Belum ada angka MAE/RMSE/bias/waktu/cakupan pipeline karena eksperimen belum dijalankan. '
        'Persentase teramati audit tidak dipakai sebagai pengganti cakupan pipeline.','',
        '### Evaluasi forecasting berikutnya','',
        '- Gunakan origin dan timestamp target yang sama pada semua pipeline, dengan horizon '
        'yang ditetapkan sebelum eksperimen. Usulan awal horizon 1 slot (30 menit), '
        'dengan 6/12 slot sebagai sensitivitas bila sesuai pertanyaan penelitian; belum final.',
        '- Model acuan pertama: persistence/last observation yang sudah tersedia pada origin, '
        'ditambah satu model sederhana yang arsitektur, lookback, tuning dan seed-nya sama '
        'antar P0–P3 (misalnya regresi Ridge berbasis lag). Deep learning dibahas setelah '
        'protokol ini terkunci; belum ada model dilatih.',
        '- Target evaluasi harus SO₂ asli, bukan hasil imputasi, agregasi hipotetis atau '
        'pemetaan off-grid otomatis. Label latih juga perlu observasi asli; label hilang '
        'tidak diisi untuk menambah pasangan latih.',
        '- Laporkan MAE/RMSE/bias forecasting pada irisan origin yang menghasilkan prediksi '
        'semua pipeline, serta kinerja dan cakupan pada daftar target asli bersama yang lebih '
        'luas. **Cakupan forecasting (%)** = 100 × origin eligible yang menghasilkan prediksi / '
        'seluruh origin dengan target asli dan horizon tersedia pada daftar evaluasi bersama. '
        'Laporkan denominator tambahan seluruh origin grid test untuk menunjukkan keterbatasan '
        'ketersediaan target, bukan hanya keberhasilan model.',
        '- Semua fallback, guard, waktu inferensi, jumlah window latih dan origin gagal '
        'dilaporkan. Jangan membandingkan skor dari timestamp berbeda tanpa menyajikan irisan bersama.','',
        '## 7. Keputusan yang didukung data dan informasi yang masih dibutuhkan','',
        f'- Seluruh {len(off)} baris off-grid tetap ada dalam sumber; {off_so2} nilai SO₂ nonkosong '
        'di antaranya tidak masuk numerator kelengkapan grid tahap kedua dan kini direkonsiliasi terpisah.',
        '- Kualitas data perlu dinilai dari posisi/panjang gap, dukungan kovariat, pola konstan '
        'dan timestamp, bukan persentase kelengkapan saja. Tidak ada stasiun utama/metode terbaik yang dipilih.',
        '- Usulan P0–P3 dibatasi kausalitas, parameter dari train, target asli dan pelaporan cakupan. '
        'Nilai yang belum andal dapat tetap tidak terselesaikan; tidak wajib mengisi seluruh grid.',
        '- **Penyedia data:** konfirmasi zona waktu dan latensi; makna timestamp/rata-rata interval; '
        'alasan pencatatan menit 01/03; kebijakan koreksi dan konflik; dokumentasi kode/status '
        'sensor, satuan, batas deteksi dan pemeliharaan; apakah rentetan konstan dan nilai nol '
        'merupakan hasil pengukuran atau representasi sistem. Tidak ada penyebab yang ditebak.',
        '- **Kesesuaian penelitian:** konfirmasi koordinat/lokasi, lingkungan stasiun, tujuan '
        'spasial dan horizon penelitian. Pilihan kualitas tidak menggantikan kecocokan lokasi.',
        '- Semua pemeriksaan yang diminta berhasil dijalankan; interpretasi sensor, pilihan '
        'kebijakan waktu dan penetapan eksperimen final menunggu informasi tersebut.','',
        '## 8. Integritas file sumber dan acuan','']
    lines += [f"- `{r['File']}`: SHA256 sebelum = sesudah = `{r['SHA256 sebelum']}`." for r in t['Integritas_SHA256']]
    (OUT/'rancangan_eksperimen_preprocessing.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
