import pandas as pd
from write_pilot_report import table
from interstation_stage9 import OUT,SCENARIOS,ALL

def report(e):
    common=[r for r in e.metrics if r['Domain']=='bersama'];pooled=[r for r in e.pooled if r['Domain']=='bersama']
    select=[r for f in e.folds for r in f.dist if r['Domain']=='irisan digunakan']
    lines=['# Tahap 9 — kontribusi antarstasiun untuk SO₂ Way Urang','',
    '**Target seluruh skenario adalah SO₂ Way Urang asli, horizon30menit.** Development set train+validasi lama sudah dipakai dalam eksperimen sebelumnya; ini validasi pengembangan, bukan holdout independen. Test akhir mulai2026-06-19 06:30 tidak dimuat nilainya atau dievaluasi. Tidak melatih deep learning atau melakukan tuning.','',
    '## Rancangan dikunci sebelum training','',
    'Konfigurasi disimpan sebelum estimasi median/scaler dan16fitRidge; SHA256: '+e.config_hash+'. Keempat gate dihitung saat fitcount=0. Median danwarmup30hari adalah pilihan operasional sementara, bukan metode terbukti paling akurat.','',
    table(['Skenario','Masukan donor','Jumlah fitur'],[{'Skenario':s,'Masukan donor':', '.join(d) if d else 'tidak ada; Way Urang saja','Jumlah fitur':e.cfg['feature_counts'][s]} for s,d in SCENARIOS.items()]),'',
    table(['Fold','Train mulai','Train akhir','Validasi mulai','Validasi akhir'],e.cfg['folds']),'',
    'S0 memakai persis34fitur C30 tanpa CurahHujan: SO₂lag0/1/2/6/12/24/48,10kovariat WayUrang lag1 dan17indikator missingasli. Tidak ada meteorologi donor. Ridgealpha1, fit_intercept=True, solverSVD, StandardScaler seluruhkolom fit di timestamptrain bersama, CPU1thread. Target tidak diskalakan atau diisi. S0 dilatih ulang, bukan mengambil skor lama.','',
    'Donor terbaru adalah origin−1slot; lag0/1/2/6/12/24/48 relatif waktu itu menghasilkan offset **1/2/3/7/13/25/49 slot sebelum origin**. Tiapdonor menyumbang7nilaiSO₂ dan7indikator missingasli. Tidak ada informasi donor sesudah origin−30menit. Usia donor hanya diagnostik, tidak menjadi fitur. Daftar kolom perskenario ada pada `Fold_*/nama_fitur.json`.','',
    '## Penyelarasan, sumber dan asumsi','',
    table(['Stasiun','Sheet','Satuan literal','Slot development','Baris sumber sebelum test','Nilai numerik grid cocok sumber','Offgrid baris','Offgrid SO2 numerik nonkosong'],e.source_checks),'',
    'Tiga stasiun tetap menjadi seri terpisah; penyelarasan menggunakan timestamp persis00/30 pada grid WayUrang. Clocktertulis dianggap sebanding, tanpa konversizona atau penetapan zona waktu. Penundaan30menit adalah asumsi eksperimen, **bukan latensi operasional terverifikasi**. Jika jam/latensi sebenarnya berbeda, manfaat donor dapat berubah.','',
    'Grid audit historis dibaca hanya25.645slot development. Untuk verifikasi sumber, timestampExcel dibaca sendiri hingga batas test; kolomSO₂ dibaca hanya sampai baris sebelumnya. Tidak membaca nilaiSO₂ test. Nilai grid dibandingkan langsung dengan sumber; fileExcel dibuka read_only tanpa save.','',
    table(['Stasiun','Baris Excel','Timestamp','SO2 literal','SO2 numerik','On grid'],e.offgrid),'',
    'Ledger off-grid dipertahankan tanpa pembulatan. Pengamatan off-grid tidak masuk fitur grid atau median, tetapi jumlah/nilainya dilaporkan dan dapat masuk diagnostik usia “semuaoriginal” jika sudah tersedia sebelum cutoff. Tidak mengarang kode sensor atau sebab anomali.','',
    table(['Stasiun','Slot development','N','Minimum','Median','Maksimum','Slot SO2 kosong/ambigu/nonnumerik','Nol','Negatif'],e.quality),'',
    'N berarti original numerik teramati, bukan otomatis valid menurut dokumentasi sensor. Baris hilang, selkosong, ambigu/nonnumerik tetap ditandai. Nol, ekstrem dan konstanta tidak otomatis dihapus.','',
    '## Median, scaling dan batas kausal','',
    'Warmup1.440slot dan pembaruan prefix1.440slot tetap. Histori WayUrang diimputasi sebelum lag, menggunakan median rawprefix eksklusif awalquery; statistik evaluasi beku rawWay[:a], a=indeks awalvalidasi. Ini identikC30 sebelumnya. Donor diimputasi **masing-masing dari serinya sendiri**; tidak memakai donor untuk mengisi target WayUrang.','',
    'Untuk donor, cutoff beku adalah rawdonor[:a−1], sehingga timestamp sumber maksimuma−2 sudah tersedia pada origin evaluasi pertamaa−1 menurut delay1. Prefix train hanya mengandung masa lalu; querymasa depan tidak menjadi donor statistik. Median yang belum tersedia tetapNaN, tanpa fallback. Tidak backwardfill atau interpolasi dua arah. Observasi asli baru yang sudah tersedia pada origin tetap boleh menjadi fitur historis.','',
    'S0 mempertahankan firstorigin1.488; S1–S3 memakai1.489 untuk tambahan lagdonor49. Commontrain memakai irisan seluruhscenario dengan targetasli, lalu target yang melintasi validasi dipurge. StandardScaler danRidge tiapscenario diestimasi dari **timestamptrain yang sama** dan hanya trainfoldterkait; tidak menggunakan parameter yang fit pada seluruhdevelopment.','',
    '## Kesetaraan sampel sebelum fit','',
    table(['Fold','Train bersama','Evaluasi bersama','S0 train layak mandiri','S0 hilang akibat irisan','Lulus','Fit count sebelum gate'],[f.gate for f in e.folds]),'',
    table(['Fold','Bagian','Skenario','Kandidat setelah warmup buffer dasar purge','Target asli teramati','Buffer tambahan donor49 pada target teramati','Fitur nonfinite pada target teramati','Layak mandiri','Irisan digunakan','Keluar karena irisan'],[r for f in e.folds for r in f.flow]),'',
    table(['Fold','Bagian','Awal target','Akhir target','N','Minimum','P25','Median','P75','Maksimum','Mean','SD'],select),'',
    'Target yang tersisih semata karena irisan setelah layak mandiri:','',
    table(['Fold','Bagian','Skenario','N','Minimum','Median','Maksimum'],[r for f in e.folds for r in f.lost]),'',
    'Workbook memuat distribusi bulanan sebelum/sesudahirisan dan sampelkeluar. LedgerCSV menyimpan setiapkandidat, kelayakan tiapscenario dan flagirisan tanpa menghapus grid. Hitungan buffer/nonfinite dapat tumpangtindih; jumlah layak dihitung dari maskgabungan, bukan penjumlahan penyebab. Apabila kehilanganbesar atau komposisi berubah, manfaat hanya berlaku pada cohortterpilih dan tidak boleh digeneralisasi ke seluruhperiode.','',
    table(['Fold','S0 fitur identik C30 lama','S0 dilatih ulang','Train lama','Train bersama baru','Cohort identik','Maks selisih prediksi jika cohort sama'],e.s0_audit),'',
    '## Skor pada timestamp identik','',
    table(['Fold','Skenario','N','Denominator target asli','Cakupan persen','MAE','RMSE','Bias','Prediksi negatif'],common),'',
    'e=prediksi−targetasli, MAE=Σ|e|/N, RMSE=√(Σe²/N), bias=Σe/N dalamµg/m³. Target evaluasi adalah originalWayUrang, tidak pernah hasil imputasi. Cakupan (%)=Nprediksi finite / seluruh targetWayUrang asli teramati dalam jendela×100. Domainbersama memakai irisanS0–S3+persistence dengan denominatorasli yang sama. “100%” tidak berarti seluruhslot grid memiliki label atau nilai sensor valid.','',
    'Cakupan dan skor mandiri setiap scenario:','',
    table(['Fold','Skenario','N','Cakupan persen','MAE','RMSE','Bias'],[r for r in e.metrics if r['Domain']=='mandiri']),'',
    table(['Fold','Skenario','N identik','Delta MAE vs S0','Perbaikan MAE persen','Delta RMSE vs S0','Perbaikan RMSE persen','Delta MAE vs persistence','Delta RMSE vs persistence'],e.deltas),'',
    'Persentase perbaikan=100×(errorS0−errorskenario)/errorS0, positif berarti error lebih kecil. Delta=skenario−baseline; negatif berarti error lebih kecil. Bias dibaca kedekatannya terhadapnol, bukan hanya tandanya. Semua perbandingan menggunakan digesttimestamp sama. Persistence memakai SO₂WayUrang asli terakhir pada/sebelum origin, tanpa imputasi dan tanpa batasumur; usia donor tersimpan.','',
    '## Gabungan, bulan dan kestabilan','',
    table(['Skenario','N','Cakupan persen','MAE','RMSE','Bias','Prediksi negatif'],pooled),'',
    'Gabungan dihitung ulang dari errorOOF unik, bukan rata-rataRMSEfold. Rata-rata danSDsampel antar4fold (ddof1) deskriptif terpisah:','',
    table(['Skenario','MAE mean fold','MAE SD fold','RMSE mean fold','RMSE SD fold','Bias mean fold'],[r for r in e.variation if r['Domain']=='bersama']),'',
    'Skor perbulan danfold ada di workbook. Fold terkait karena trainexpanding; selisih mean tidak membuktikan signifikansi statistik.','',
    '![Skor per fold](figures/kontribusi_antarstasiun/skor_per_fold.png)','',
    '![Perubahan terhadap S0](figures/kontribusi_antarstasiun/perubahan_vs_S0.png)','',
    '## Ketersediaan dan rentetan konstan donor','',
    table(['Fold','Donor','Bagian sampel','N origin','Minimum menit grid','Median menit grid','P95 menit grid','Maksimum menit grid','Maksimum menit semua','Tanpa pengamatan asli grid'],e.age_stats),'',
    'Usia=origin−timestamp asli numerik terakhir≤origin−30menit. Bila pengamatan pada cutoff tersedia, usia minimum30menit; tambahanusia setelahcutoff=usiagrid−30. Usia “grid” sesuai pengamatan yang dapat menjadi fitur; usia “semua” juga mempertimbangkan original off-grid, hanya diagnostik. Data original tidak otomatis berkualitasvalid. Median dapat menghasilkan fiturfinite saat original sudah lama tidak tersedia; cakupan tidak sama dengan kebaruan informasi.','',
    table(['Fold','Donor','Bagian sampel model','Jumlah rentetan terkait','Run terpanjang slot','Durasi run terpanjang menit','Timestamp sumber konstan unik digunakan'],e.constant_summary),'',
    'Semua rentetan sumber yang masukfitur dicatat dalam workbook. Contoh10run terpanjang tiapdonor pada sampel model (dipilih secara deskriptif, bukan label kerusakan):','']
    for donor in ['Pulomerak','Talang']:
        frame=pd.DataFrame([r for r in e.constant if r['Donor']==donor]);top=frame.sort_values(['Jumlah slot run','Timestamp sumber unik masuk fitur'],ascending=False).drop_duplicates('Run ID').head(10)
        lines += [donor+':','',table(['Run ID','Nilai SO2 asli','Run mulai','Run akhir','Jumlah slot run','Durasi run menit','Terpotong batas development'],top.to_dict('records')),'']
    lines += ['Run adalah nilai original persis sama pada≥2slot30menit berturut-turut; missing memutus. Durasi=Nslot×30menit, bukanakhir−awal. Penggunaan dihitung pada timestamp sumberunik dari semua7lag yang masuk cohortbersama, bukan jumlahnilai imputed. Run yang mencapaiakhirdevelopment ditandai terpotong; tidak membuka test untuk mengetahui kelanjutannya. Run dapat muncul di train dan evaluasi atau beberapafold, sehingga bukan kejadian independen yang boleh dijumlahkan lintasfold. Tidak menyatakan seluruh data konstan salah.','',
    '## Transisi22Juni2025','',
    'Tanggal2025-06-22 17:30 dipilih pada diagnosis terdahulu dari penurunan30menit original terbesar, bukan dari skor stage9. Jendela±72jam sama, semua skenario ditampilkan tanpa clipping atau penghapusan fold1.','',
    '![Transisi](figures/kontribusi_antarstasiun/transisi_22_juni.png)','',
    '## Kesimpulan terbatas','']
    for scenario in ['S1','S2','S3']:
        rows=[r for r in e.deltas if r['Skenario']==scenario and r['Fold']!='Gabungan_OOF'];wins=[r['Fold'] for r in rows if r['Delta MAE vs S0']<0 and r['Delta RMSE vs S0']<0]
        lose=[r['Fold'] for r in rows if r['Delta MAE vs S0']>0 and r['Delta RMSE vs S0']>0]
        lines.append(f'- {scenario} ({", ".join(SCENARIOS[scenario])}): MAE danRMSE sekaligus membaik terhadapS0 pada {len(wins)}/4fold ({", ".join(wins) if wins else "tidak ada"}); keduanya memburuk pada {", ".join(lose) if lose else "tidak ada"}. Hasilmetrikcampuran dibaca pada tabel, bukan dipaksa menjadi satu peringkat.')
    lines += ['',
    'Pulomerak hanya memperbaiki kedua metrik pada fold kedua (12 Agustus–24 November 2025): MAE turun 0,031% dan RMSE 0,099%; manfaat ini kecil dan tidak bertahan pada fold lain. Talang tidak memperbaiki kedua metrik pada satu pun fold. Kombinasi donor pada fold kedua menurunkan RMSE 0,127% tetapi menaikkan MAE 2,176%; pada tiga fold lain keduanya memburuk. Pada fold keempat penurunan kinerja paling jelas secara relatif: MAE S3 naik 31,006% dan RMSE 29,353%. Tidak ada bukti manfaat donor yang stabil untuk spesifikasi ini.', '',
    'Gabungan OOF tetap didominasi kesalahan fold pertama, sehingga tabel per fold dan bulan harus dibaca bersama. Seluruh skenario memiliki cakupan 100% pada 19.797 target teramati; irisan tidak mengurangi sampel S0. Perubahan kinerja di sini tidak disebabkan perubahan cohort train/evaluasi antar-skenario. Cakupan tinggi tidak menjamin donor baru: usia maksimum Pulomerak saat evaluasi fold ketiga mencapai 30.450 menit dan fold keempat 18.300 menit. Indikator missing mempertahankan status input yang diimputasi, tetapi tidak menjamin median mewakili perubahan antarperiode.', '',
    'S0 dan tambahan donor tidak mengalahkan persistence secara konsisten pada MAE dan RMSE sekaligus. S0 pada fold ketiga dan keempat memiliki MAE lebih rendah daripada persistence tetapi RMSE lebih tinggi. Untuk verifikasi dengan satu model deep learning berikutnya, S0 tetap menjadi acuan wajib; S1 Pulomerak dapat dipertahankan sebagai satu hipotesis sensitivitas terbatas karena ada perbaikan kecil fold kedua, bukan sebagai kandidat unggul yang sudah terbukti. Hasil Ridge ini belum memberikan dukungan kuat untuk memprioritaskan S2 atau S3. Tidak memilih preprocessing final dan tidak melatih model tambahan dalam tahap ini.', '',
    'Manfaat donor merupakan informasi prediktif tambahan kondisional pada fitur, clock, delay, median, cohort danRidge ini. Tidak membuktikan hubungan sebab-akibat, transport SO₂ atau sumber vulkanik. Tidak memilih konfigurasi dari rerataerror saja; perhatikan kestabilan antarperiode, bias, usia/konstanta donor dan cakupan. Kandidat yang membaik cukup konsisten dapat diverifikasi kemudian dengan **satu** model deep learning yang tetap, tetapi belum ada training DL baru atau pemilihan metode/arsitektur terbaik. Jika manfaat tidak konsisten, hasil negatif tetap dipertahankan tanpa mengubah split.','',
    '## Waktu, artefak dan integritas','',
    table(['Fold','Skenario','Train N','Fitur N','Scaler fit detik','Ridge fit detik','Predict detik'],[r for f in e.folds for r in f.timing]),'',
    table(['Fold','Detik seluruh skenario'],[{'Fold':f.label,'Detik seluruh skenario':f.preprocess_seconds} for f in e.folds]),'',
    f'{len(e.integrity)} file sumber/artefak lama terdaftarGit cocok SHA256 sebelum–sesudah, termasuk arsipWayUrang, notebook, checkpoint danhasil stage1–8. Tidak ada hasil lama ditimpa. Hashingbytes tidak mengakses seri test untuk evaluasi.','',
    'Notebook: `notebooks/09_kontribusi_antarstasiun.ipynb`. Configterkunci, grid/sourceprefix/offgrid, usia original, ledger sampel train/evaluasi, fitur dan parameter perFold, prediksiOOF tersimpan pada `outputs/kontribusi_antarstasiun/`. Workbook: `hasil_kontribusi_antarstasiun.xlsx`; grafik: `figures/kontribusi_antarstasiun/`.','']
    from polish_lstm_report import polish
    text=polish('\n'.join(lines))
    for old,new in {'testakhir':'test akhir','trainfoldterkait':'train fold terkait','terhadapS0':'terhadap S0','semua7':'semua 7','sebelumorigin':'sebelum origin',
        'WayUrang':'Way Urang','rawprefix':'raw prefix','awalquery':'awal query','sourceprefix':'source prefix','clocktertulis':'clock tertulis',
        'trainexpanding':'train expanding','timestamptrain':'timestamp train','Configterkunci':'Config terkunci','Ridgealpha1':'Ridge alpha 1','CPU1thread':'CPU 1 thread',
        'interceptaktif':'intercept aktif','solverSVD':'solver SVD','rawWay':'raw Way','rawdonor':'raw donor','originpertama':'origin pertama',
        'hashingbytes':'hashing bytes','hasilmetrikcampuran':'hasil metrik campuran','Hasilmetrikcampuran':'Hasil metrik campuran'}.items():text=text.replace(old,new)
    # Formatting only; these replacements never touch configuration or numeric results.
    pairs={'horizon30menit':'horizon 30 menit','dan16fitRidge':'dan 16 fit Ridge','fitcount':'fit count',
      'danwarmup':'dan warmup','CurahHujan':'Curah Hujan','seluruhkolom':'seluruh kolom','persis34':'persis 34',
      'dan17':'dan 17','SO₂lag':'SO₂ lag','Tiapdonor':'Tiap donor','perskenario':'per skenario','tiaplag':'tiap lag',
      'timestampExcel':'timestamp Excel','kolomSO₂':'kolom SO₂','Clocktertulis':'Clock tertulis','konversizona':'konversi zona',
      'persis00/30':'persis 00/30','hanya25.645':'hanya 25.645','semuaoriginal':'semua original','selkosong':'sel kosong',
      'Warmup1.440slot':'Warmup 1.440 slot','pembaruan prefix1.440slot':'pembaruan prefix 1.440 slot',
      'sebelumquery':'sebelum query','identikC30':'identik C30','maksimuma−2':'maksimum a−2','pertamaa−1':'pertama a−1',
      'delay1':'delay 1','querymasa':'query masa','seluruhdevelopment':'seluruh development','tetapNaN':'tetap NaN',
      'firstorigin1.488':'origin pertama 1.488','memakai1.489':'memakai 1.489','lagdonor49':'lag donor 49',
      'Commontrain':'Train bersama','seluruhscenario':'seluruh skenario','targetasli':'target asli','danRidge':'dan Ridge',
      'sesudahirisan':'sesudah irisan','sampelkeluar':'sampel keluar','LedgerCSV':'Ledger CSV','setiapkandidat':'setiap kandidat',
      'tiapscenario':'tiap skenario','flagirisan':'flag irisan','tumpangtindih':'tumpang tindih','maskgabungan':'mask gabungan',
      'kehilanganbesar':'kehilangan besar','cohortterpilih':'cohort terpilih','seluruhperiode':'seluruh periode',
      'dalamµg':'dalam µg','originalWay':'original Way','Nprediksi':'N prediksi','targetWay':'target Way',
      'Domainbersama':'Domain bersama','irisanS0':'irisan S0','denominatorasli':'denominator asli','errorS0':'error S0',
      'errorskenario':'error skenario','terhadapnol':'terhadap nol','digesttimestamp':'digest timestamp','SO₂Way':'SO₂ Way',
      'batasumur':'batas umur','errorOOF':'error OOF','rata-rataRMSEfold':'rata-rata RMSE fold','danSDsampel':'dan SD sampel',
      'antar4fold':'antar 4 fold','perbulan':'per bulan','timestamp sumberunik':'timestamp sumber unik',
      'pengamatan asli valid otomatis':'pengamatan asli valid secara otomatis','cutoff tersedia':'cutoff tersedia',
      'tambahanusia':'tambahan usia','setelahcutoff':'setelah cutoff','usiagrid':'usia grid','berkualitasvalid':'berkualitas valid',
      'masukfitur':'masuk fitur','Contoh10run':'Contoh 10 run','tiapdonor':'tiap donor','pada≥2slot30menit':'pada ≥2 slot 30 menit',
      'bukanakhir':'bukan akhir','cohortbersama':'cohort bersama','jumlahnilai imputed':'jumlah nilai yang diimputasi',
      'mencapaiakhirdevelopment':'mencapai akhir development','beberapafold':'beberapa fold','lintasfold':'lintas fold',
      'Transisi22Juni2025':'Transisi 22 Juni 2025','Tanggal2025':'Tanggal 2025','penurunan30menit':'penurunan 30 menit',
      'Jendela±72jam':'Jendela ±72 jam','fold1':'fold 1','danRMSE':'dan RMSE','/4fold':'/4 fold',
      'rerataerror':'rerata error','antarperiode':'antarperiode','arsipWay':'arsip Way','danhasil':'dan hasil',
      'terdaftarGit':'terdaftar Git','Hashingbytes':'Hashing bytes','perFold':'per fold','prediksiOOF':'prediksi OOF'}
    for old,new in pairs.items():text=text.replace(old,new)
    (OUT/'ringkasan_kontribusi_antarstasiun.md').write_text(text,encoding='utf-8')
