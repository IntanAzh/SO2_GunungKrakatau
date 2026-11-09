"""Indonesian interpretation and project continuation record from completed results."""
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
from write_pilot_report import table
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'lstm_antarstasiun_tahap10'

def tidy(text):
    lines=[]
    for line in text.splitlines():
        if not line.startswith('|'):
            line=re.sub(r'\b(tahap|Tahap|fold|seed|horizon|Horizon|warmup|LSTM|Adam|batch|patience|minimal|maksimal|maks|sekitar|mencakup|hanya|mencapai|masing-masing|dan|untuk|tersedia|model|epoch|internal|Python)(?=\d)',r'\1 ',line)
            line=re.sub(r'\b(\d+)(hari|menit|jam|slot|epoch|model)\b',r'\1 \2',line)
            line=line.replace('PyTorchCPU','PyTorch CPU')
        lines.append(line)
    return '\n'.join(lines)+'\n'

def write_report(t,c):
    common=[r for r in t['Skor_fold_seed'] if r['Domain']=='bersama'];pooled=[r for r in t['Gabungan_per_seed'] if r['Domain']=='bersama']
    variation=[r for r in t['Ringkasan'] if r['Fold']=='Gabungan_OOF'];changes=pd.DataFrame(t['Perubahan_S1_vs_S0']);foldchanges=changes[changes.Fold.ne('Gabungan_OOF')]
    wins=(foldchanges['Perbaikan MAE persen']>0)&(foldchanges['Perbaikan RMSE persen']>0)
    losses=(foldchanges['Perbaikan MAE persen']<0)&(foldchanges['Perbaikan RMSE persen']<0)
    predictions=pd.read_csv(RUN/'prediksi_oof_seluruh_seed.csv')
    extremes=[]
    for fold in ['Fold_03','Fold_04']:
        for seed in c['seeds']:
            q=predictions[predictions.Fold.eq(fold)&predictions.Seed.eq(seed)]
            worst=q.loc[(q.S1-q['SO2 target asli']).abs().idxmax()]
            outside=q.S1>q['SO2 target asli'].max()
            extremes.append({'Fold':fold,'Seed':seed,'Evaluasi N':len(q),'Maksimum target asli evaluasi':q['SO2 target asli'].max(),
                'S1 di atas maksimum target N':int(outside.sum()),
                'Di antaranya donor asli lengkap 48 posisi N':int((outside&q['Pulomerak posisi missing asli dari48'].eq(0)).sum()),
                'Waktu error absolut terbesar':worst['Timestamp target'],'Target pada error terbesar':worst['SO2 target asli'],
                'Prediksi S1 pada error terbesar':worst.S1,'Posisi donor missing pada error terbesar':int(worst['Pulomerak posisi missing asli dari48'])})
    evidence=[]
    for fold,g in foldchanges.groupby('Fold',sort=False):
        w=((g['Perbaikan MAE persen']>0)&(g['Perbaikan RMSE persen']>0)).sum();l=((g['Perbaikan MAE persen']<0)&(g['Perbaikan RMSE persen']<0)).sum()
        evidence.append({'Fold':fold,'Seed MAE dan RMSE membaik':int(w),'Seed keduanya memburuk':int(l),'Seed hasil campuran':int(3-w-l),
            'Perbaikan MAE min persen':g['Perbaikan MAE persen'].min(),'Perbaikan MAE max persen':g['Perbaikan MAE persen'].max(),
            'Perbaikan RMSE min persen':g['Perbaikan RMSE persen'].min(),'Perbaikan RMSE max persen':g['Perbaikan RMSE persen'].max()})
    cols=['Fold','Seed','Skenario','N','Cakupan persen','MAE','RMSE','Bias','Prediksi negatif']
    lines=['# Tahap 10 — verifikasi kontribusi Pulomerak dengan LSTM','',
      'Eksperimen selesai dengan 24 model: empat fold kalender × S0/S1 × seed 11/23/47. Target seluruh model adalah SO₂ Way Urang asli pada 30 menit berikutnya. Development set telah digunakan dalam tahap sebelumnya, sehingga ini validasi pengembangan. Test akhir mulai '+c['test_start']+' tidak dimuat nilainya atau dievaluasi. Tidak memilih seed terbaik, mengubah split, menghapus fold pertama, atau melakukan tuning.','',
      '## Rancangan dan lingkungan','',
      'S0 memakai kanal dasar Way Urang tahap 7 tanpa Curah Hujan. S1 menambahkan hanya SO₂ Pulomerak lag1 dan indikator missing aslinya. Tidak memasukkan meteorologi Pulomerak atau Talang. Satu langkah sekuens mempunyai 11 kanal numerik + 11 indikator pada S0 dan 12 + 12 pada S1. Sekuens berisi 48 slot berurutan, dari origin−47 sampai origin, tanpa kompresi waktu atau duplikasi lag tabular.','',
      table(['Skenario','Daftar kanal berurutan'],[{'Skenario':s,'Daftar kanal berurutan':', '.join(names)} for s,names in c['features'].items()]),'',
      'SO₂ Way Urang pada langkah u diasumsikan tersedia pada u. Sepuluh kovariat Way Urang memakai u−1. SO₂ Pulomerak pada setiap langkah u berasal dari u−1, sehingga slot donor terbaru dalam sekuens adalah origin−30 menit dan tertua origin−48 slot. Keterlambatan 30 menit merupakan asumsi eksperimen, bukan latensi operasional terverifikasi. Clock tertulis dianggap sebanding tanpa menebak zona waktu. Ketidakpastian sinkronisasi dan latensi membatasi interpretasi.','',
      'Model tetap: LSTM satu layer 32 unit, output linear satu unit, Adam learning rate 0,001, loss MSE target standar, batch64, maksimal50 epoch, early stopping patience8 dan min_delta0 pada validasi internal. Bobot terbaik internal dipulihkan. Tidak ada dropout, gradient clipping, clipping prediksi, backward fill, atau informasi masa depan. Jumlah parameter bertambah dari 7.201 pada S0 menjadi 7.457 pada S1 karena dua kanal tambahan. Seed sama tidak membuat bobot awal arsitektur dengan dimensi input berbeda identik. Urutan shuffle minibatch tetap sama antar-skenario untuk seed dan epoch yang sama.','',
      'Environment lama dipertahankan: `.venv-lstm`, Python '+c['environment']['python'].split()[0]+', PyTorch '+c['environment']['versions']['torch']+', CPU, satu thread, deterministic algorithms aktif, GPU tidak digunakan. Versi lengkap dan perangkat ada pada konfigurasi dan `dependensi_freeze.txt`.','',
      'Percobaan persiapan pertama berhenti sebelum gate dan training karena modul pip tidak tersedia. Versi paket kemudian dicatat melalui importlib.metadata. Konfigurasi percobaan persiapan itu tetap disimpan sebagai `config_prefit_attempt_01.json`; rancangan dan parameter model tidak berubah. Konfigurasi operasional terkunci sebelum seluruh gate dan fit pada `config_locked.json`.','',
      '## Kalibrasi, scaling dan pembagian internal','',
      'Batas validasi internal adalah floor(0,85 × jumlah slot kalender train fold), bukan pembagian berdasarkan jumlah pengamatan. Sebanyak 15% bagian akhir kalender train digunakan untuk early stopping. Median kausal memakai warmup statistik30 hari (1.440slot), diperbarui tiap30 hari dari raw prefix eksklusif awal query selama train internal, lalu dibekukan pada batas train internal untuk validasi internal dan evaluasi fold. Histori yang sudah diproses tidak dihitung ulang memakai statistik masa depan.','',
      'Kanal tertunda disusun sebelum imputasi. Untuk Pulomerak(u−1), prefix langkah [:start] hanya memuat timestamp sumber paling baru start−2. Karena itu median donor tidak menggunakan pengukuran yang belum tersedia pada origin. Ini mengikuti prosedur kanal sekuens tahap7 dengan C30, bukan menggandakan fitur lag tabular tahap9. Pengamatan asli baru yang sudah tersedia tetap menjadi input. Tanpa statistik median yang dapat diestimasi, fitur tetap kosong dan sekuens tidak layak; tidak menambah fallback. Target tidak pernah diimputasi atau diisi oleh stasiun lain.','',
      'Origin pertama tetap warmup+48=1.488 sesuai buffer sebelumnya. Jendela48 mencakup origin−47 sampai origin. Sampel train dengan target memasuki validasi internal dipurge. Sampel validasi internal berhenti sebelum target memasuki evaluasi fold. Konteks sebelumnya yang tersedia boleh masuk jendela validasi.','',
      'StandardScaler hanya di-fit pada slot langkah unik yang benar-benar masuk sekuens train internal bersama. Kanal numerik distandardisasi; indikator missing asli tetap0/1. Mean dan SD populasi target dihitung hanya dari target asli train internal bersama, lalu prediksi di-inverse ke µg/m³ sebelum semua metrik. SD nol akan memakai1 secara eksplisit; parameter yang digunakan tersimpan.','',
      table(['Fold','Train internal akhir kalender','Validasi internal awal kalender','Validasi internal akhir kalender','Evaluasi awal kalender','Evaluasi akhir kalender','Train bersama','Validasi internal bersama','Evaluasi bersama','Fit count sebelum gate','Lulus'],t['Gate_sebelum_training']),'',
      '## Jumlah sampel dan distribusi target','',
      table(['Fold','Bagian','Skenario','Slot kalender target','Target asli seluruh bagian','Kandidat origin setelah warmup buffer purge','Target asli sebelum sekuens','Origin finite setelah imputasi','Sekuens lengkap 48 slot','Sesudah irisan','Keluar karena sekuens','Keluar karena irisan'],t['Alur_sampel']),'',
      '“Target asli sebelum sekuens” memakai kandidat setelah warmup, buffer dan purge. Kelengkapan satu origin setelah imputasi tidak menjamin seluruh48 langkah lengkap. Ledger setiap timestamp menyimpan alasan yang dapat tumpang tindih: target kosong, input origin masih kosong, sekuens tidak lengkap, serta hasil irisan. Jumlah per tahap bukan penjumlahan alasan yang tumpang tindih. Tidak menghapus slot dari grid untuk merapatkan sekuens.','',
      table(['Fold','Bagian','Skenario','N','Minimum','Median','Maksimum'],t['Keluar_irisan']),'',
      table(['Fold','Bagian','Alasan','Slot tersisih','N','Minimum','Median','Maksimum'],t['Sampel_tersisih']),'',
      'Tabel sampel tersisih membedakan target kosong (N numerik=0), histori sebelum grid, warmup, buffer, dan sampel sesudah warmup yang tidak masuk irisan. Distribusi bulanan dan timestamp batas purge tersedia di workbook. Sampel yang targetnya melintasi batas dilarang masuk bagian sebelumnya, tetapi tetap dapat dievaluasi pada bagian sesudahnya. Dengan demikian purge bukan penghapusan label global.','',
      table(['Fold','Bagian','Domain','Target awal','Target akhir','N','Minimum','P25','Median','P75','Maksimum','Mean','SD'],[r for r in t['Distribusi_target'] if r['Domain']=='bersama digunakan']),'',
      'Distribusi sebelum/sesudah pembentukan sekuens dan perbulan ada dalam workbook. Median C30 adalah pilihan operasional sementara, bukan metode yang telah terbukti paling akurat. Kecukupan komputasional minimal64 sampel tidak menjamin keterwakilan distribusi.','',
      'Kolom hitungan SO₂>2,14 pada tabel distribusi hanya penelusuran konsentrasi yang digunakan pada tahap terdahulu, bukan ambang kualitas, bahaya, atau aturan seleksi.','',
      'Pada fold pertama, target train internal mencakup1,24–80,71 µg/m³, tetapi validasi internal hanya43,72–47,29 µg/m³. Dengan demikian early stopping memantau periode berkonsentrasi tinggi yang sempit, sedangkan evaluasi melintasi penurunan ke tingkat rendah. Pemeriksaan ini menunjukkan keterbatasan keterwakilan validasi internal, bukan bukti penyebab tunggal kegagalan atau alasan untuk mengganti split. Pada fold2–4, jendela evaluasi jauh lebih rendah dan sempit, sekitar1,04–2,14 µg/m³.','',
      '## Hasil per fold dan seed pada timestamp identik','',table(cols,common),'',
      'Error e=prediksi−target asli. MAE=mean(|e|), RMSE=sqrt(mean(e²)), bias=mean(e), seluruhnya dalam µg/m³. Cakupan(%)=100×N prediksi finite / seluruh target Way Urang asli numerik teramati dalam jendela tersebut. Cakupan bersama memakai irisan S0/S1/persistence. Target kosong tidak diberi nilai untuk penilaian. Pengamatan numerik teramati tidak otomatis dinyatakan valid berdasarkan dokumentasi sensor.','',
      'Hasil mandiri dan cakupan terhadap denominator seluruh target asli:','',table(cols,[r for r in t['Skor_fold_seed'] if r['Domain']=='mandiri']),'',
      'Persistence menggunakan SO₂ Way Urang asli terakhir pada/sebelum origin, tanpa imputasi dan tanpa batas usia. Usia donor persistence disimpan per prediksi. Persistence identik untuk ketiga seed dan bukan tiga replikasi stokastik independen.','',
      table(['Fold','Skenario','MAE mean','MAE SD seed','RMSE mean','RMSE SD seed','Bias mean','Cakupan persen mean','Prediksi negatif mean'],[r for r in t['Ringkasan'] if r['Fold']!='Gabungan_OOF']),'',
      'SD antar-seed adalah SD sampel ddof1 dari tiga seed, bukan interval keyakinan atau uji signifikansi. Perbandingan utama mempertahankan semua seed.','',
      table(['Fold','Seed','N identik','Perbaikan MAE persen','Perbaikan RMSE persen','Delta MAE S1 minus S0','Delta RMSE S1 minus S0','Delta MAE S0 minus persistence','Delta MAE S1 minus persistence','Delta RMSE S0 minus persistence','Delta RMSE S1 minus persistence'],t['Perubahan_S1_vs_S0']),'',
      'Perbaikan (%)=100×(error S0−error S1)/error S0, sehingga positif berarti S1 lebih baik. Delta adalah error skenario dikurangi baseline, sehingga negatif berarti error lebih rendah. Bias dinilai kedekatannya dengan nol, bukan hanya tandanya.','',
      '## Gabungan dan kestabilan antarperiode','',table(cols,pooled),'',
      'Metrik gabungan dihitung ulang dari error seluruh fold evaluasi yang tidak tumpang tindih, terpisah untuk setiap seed. RMSE gabungan bukan rata-rata RMSE fold.','',
      table(['Skenario','MAE mean','MAE SD seed','RMSE mean','RMSE SD seed','Bias mean','Cakupan persen mean'],variation),'',
      table(['Fold','Seed MAE dan RMSE membaik','Seed keduanya memburuk','Seed hasil campuran','Perbaikan MAE min persen','Perbaikan MAE max persen','Perbaikan RMSE min persen','Perbaikan RMSE max persen'],evidence),'',
      f'Pulomerak memperbaiki MAE dan RMSE sekaligus pada {int(wins.sum())}/12 pasangan fold–seed dan memperburuk keduanya pada {int(losses.sum())}/12. Hasil lainnya campuran. Selisih ini deskriptif; fold expanding berbagi histori dan tiga seed tidak membuktikan signifikansi statistik. Skor dan persentase perubahan perbulan disimpan terpisah menurut fold dan seed agar periode yang berbeda tidak tersamarkan oleh gabungan.','',
      '![Skor](figures/skor_per_fold.png)','', '![Perubahan](figures/perubahan_pulomerak.png)','',
      '## Transisi konsentrasi 22 Juni 2025','',
      'Pusat transisi2025-06-22 17:30 dan jendela±72 jam mengikuti diagnosis terdahulu, bukan dipilih dari skor LSTM tahap10. Fold pertama, prediksi negatif dan semua seed dipertahankan tanpa clipping. Tabel berikut memisahkan sebelum dan sesudah pusat transisi dengan timestamp yang sama.','',
      table(['Seed','Periode','Skenario','N','MAE','RMSE','Bias','Prediksi negatif'],t['Transisi_per_seed']),'',
      table(['Seed','Periode','Skenario','N','Target mean','Target median','Prediksi mean','Prediksi median','Prediksi minimum','Prediksi maksimum'],t['Tingkat_transisi']),'',
      '![Transisi](figures/transisi_22_juni.png)','',
      '## Ketersediaan informasi donor dan kualitas','',
      table(['Fold','Bagian','N sekuens bersama','Posisi fitur donor','Posisi donor missing asli diimputasi','Proporsi posisi donor diimputasi persen','Sekuens dengan donor diimputasi persen','Usia donor origin median menit','Usia donor origin P95 menit','Usia donor origin maksimum menit'],t['Ketersediaan_donor']),'',
      'Usia donor=origin−timestamp Pulomerak asli numerik terakhir≤origin−30 menit. Minimum30 menit mengikuti delay asumsi. Proporsi posisi donor diimputasi memakai denominator N sekuens×48; slot yang masuk beberapa jendela dihitung pada setiap posisi fitur, sehingga ini proporsi paparan fitur, bukan persentase missing slot unik. Proporsi sekuens dengan imputasi memakai N sekuens sebagai denominator. Distribusi usia dan ringkasan bulanan tersimpan. Cakupan prediksi tinggi dapat terjadi meskipun informasi donor asli tidak tersedia dan kanal diisi median lama.','',
      'Pada evaluasi fold3 dan4, masing-masing27,601% dan24,834% posisi fitur donor diimputasi. Usia original maksimum mencapai30.450 dan18.300 menit. Cakupan model100% pada kondisi ini tidak dapat dibaca sebagai Pulomerak original tersedia100%. Perubahan manfaat donor mungkin berkaitan dengan beberapa perubahan distribusi sekaligus; desain ini tidak mengisolasi pengaruh missing, usia, atau nilai konstan sebagai penyebab tunggal.','',
      table(['Fold','Seed','Evaluasi N','Maksimum target asli evaluasi','S1 di atas maksimum target N','Di antaranya donor asli lengkap 48 posisi N','Waktu error absolut terbesar','Target pada error terbesar','Prediksi S1 pada error terbesar','Posisi donor missing pada error terbesar'],extremes),'',
      '“Di atas maksimum target asli evaluasi” adalah diagnostik rentang prediksi pada jendela validasi, bukan batas sensor, kualitas, atau bahaya. Kejadian dengan semua48 posisi donor original teramati menunjukkan bahwa kesalahan S1 pada fold3–4 tidak dapat dijelaskan hanya oleh imputasi donor. Belum ada pemeriksaan yang membuktikan penyebab tunggal keluaran model tersebut. Seluruh prediksi, termasuk nilai negatif dan ekstrem, tetap masuk skor.','',
      table(['Stasiun','Baris Excel','Timestamp','SO2 literal','SO2 numerik','On grid'],t['Offgrid_development']),'',
      'Ledger off-grid dipertahankan tanpa pembulatan dan tidak dipakai untuk fitur atau median. Tiga timestamp development terkait Way Urang/Pulomerak pada ledger ini tidak memiliki SO₂ numerik. Nol, ekstrem dan konstanta donor tidak dihapus. Seluruh rentetan konstan Pulomerak yang masuk train, validasi internal dan evaluasi ada dalam workbook. Run memakai original yang tepat sama pada≥2 slot berurutan, missing memutus, durasi=Nslot×30 menit. Run yang sama dapat berkontribusi pada beberapa fold; tidak boleh dijumlahkan sebagai kejadian independen atau dilabel kerusakan sensor.','',
      '## Perbandingan tambahan dengan Ridge tahap9','',
      table(['Fold','Seed','Skenario','N','Cakupan persen','MAE','RMSE','Bias'],[r for r in t['Ridge_timestamp_identik'] if r['Fold']=='Gabungan_OOF']),'',
      'Ridge tidak dilatih ulang. Skor dihitung ulang dari prediksi tahap9 yang tersimpan pada irisan timestamp S0/S1 LSTM dan persistence yang identik, tersedia per fold dan seed. Perbedaan mencakup representasi input (lag tabular vs48 langkah), holdback15% kalender untuk early stopping, standardisasi target dan optimisasi, bukan pengaruh arsitektur saja.','',
      '## Waktu komputasi dan reproduksibilitas','',
      table(['Fold','Skenario','Seed','Epoch dijalankan','Epoch terbaik','N train bersama','N validasi internal bersama','Jumlah parameter','Fit early stopping detik','Predict detik'],t['Training_dan_waktu']),'',
      'Waktu fit merupakan jumlah durasi epoch train+validasi internal yang diukur, tidak mencakup penulisan checkpoint, persiapan tensor, atau jeda resume. Waktu predict mengukur inferensi dan inverse scaling. Waktu preprocessing disimpan sekali per fold pada preprocessing.json. Waktu ini deskriptif untuk lingkungan lokal yang juga menjalankan penulisan dan verifikasi, bukan benchmark perangkat terisolasi. Tidak mengarang waktu yang tidak tersimpan.','',
      'Setiap epoch menyimpan model, optimizer, RNG, best state, histori dan akumulasi waktu. Setelah tiap model selesai, checkpoint best, prediksi, histori, waktu dan SHA256 disimpan segera; complete.json menjadi penanda terakhir. Model lengkap dilewati pada resume. Early stopping memilih epoch berdasarkan validasi internal, bukan seed atau skor evaluasi fold.','',
      '![Histori](figures/riwayat_early_stopping.png)','',
      '## Kesimpulan terbatas','']
    if wins.all():
        lines+=['Pada spesifikasi ini, tambahan Pulomerak menurunkan kedua error pada semua fold dan seed. Ini mendukung kelanjutan sebagai kandidat, dengan tetap meninjau besaran manfaat, bias, donor yang diimputasi dan transisi. Belum membuktikan keunggulan di test akhir atau kondisi operasional.','']
    else:
        lines+=['Tambahan Pulomerak belum memperbaiki MAE dan RMSE secara konsisten pada seluruh periode dan seed. Hasil ini mendukung penyimpanan S1 sebagai hasil sensitivitas, bukan pengganti otomatis S0. S0 tetap acuan sederhana. Keputusan didasarkan pada kestabilan dan besaran manfaat, bukan pemilihan seed terbaik atau error rata-rata terkecil.','']
    tr=pd.DataFrame(t['Transisi_per_seed']);wtrans=0
    for seed in c['seeds']:
        g=tr[tr.Seed.eq(seed)&tr.Periode.eq('sesudah transisi')].set_index('Skenario')
        if g.loc['S1','MAE']<g.loc['S0','MAE'] and g.loc['S1','RMSE']<g.loc['S0','RMSE']:wtrans+=1
    level=pd.DataFrame(t['Tingkat_transisi']);post=level[level.Periode.eq('sesudah transisi')]
    target_post=post['Target mean'].iloc[0]
    ranges={s:(post.loc[post.Skenario.eq(s),'Prediksi mean'].min(),post.loc[post.Skenario.eq(s),'Prediksi mean'].max()) for s in ['S0','S1']}
    lines += [f'Pada jendela setelah transisi, mean target asli adalah {target_post:.6f} µg/m³. Mean prediksi S0 antar-seed berada pada {ranges["S0"][0]:.6f}–{ranges["S0"][1]:.6f} µg/m³ dan S1 pada {ranges["S1"][0]:.6f}–{ranges["S1"][1]:.6f} µg/m³. Selisih terhadap tingkat target dan bias pada tabel adalah bukti langsung keterbatasan mengikuti transisi, bukan penjelasan penyebab perubahan data.','']
    if min(ranges['S0'][0],ranges['S1'][0])>target_post:
        lines += ['Mean prediksi kedua skenario pada seluruh seed masih lebih tinggi daripada mean target sesudah transisi. Dengan selisih yang ditampilkan di atas, tambahan Pulomerak belum menghilangkan kecenderungan mempertahankan prediksi tinggi setelah penurunan konsentrasi.','']
    timing=pd.DataFrame(t['Training_dan_waktu']);total_fit=timing.groupby('Skenario')['Fit early stopping detik'].sum();total_predict=timing.groupby('Skenario')['Predict detik'].sum()
    lines += [f'Dalam72 jam sesudah transisi, S1 memperbaiki kedua metrik terhadap S0 pada {wtrans}/3 seed. Grafik, bias dan prediksi negatif menunjukkan apakah manfaat tersebut juga berarti mengikuti tingkat konsentrasi rendah; perbaikan relatif saja tidak cukup untuk menyatakan transisi teratasi. Tidak mengaitkan perubahan dengan gunung atau kerusakan alat tanpa bukti tambahan.','',
      f'Total waktu epoch terukur adalah {total_fit["S0"]:.3f} detik untuk12 model S0 dan {total_fit["S1"]:.3f} detik untuk12 model S1. Inferensi terukur masing-masing {total_predict["S0"]:.3f} dan {total_predict["S1"]:.3f} detik. Ini biaya keseluruhan eksperimen dengan jumlah epoch early stopping yang dapat berbeda, bukan perbandingan FLOP murni.','',
      'Kompleksitas S1 bertambah dua kanal dan256 parameter, serta memerlukan sinkronisasi/latensi, ledger kualitas dan pemeliharaan sumber donor. Pada eksperimen ini, manfaat fold kedua tidak sebanding dengan kemerosotan besar fold ketiga dan keempat. Biaya waktu terukur dipengaruhi jumlah epoch yang berbeda, sehingga tidak cukup untuk mengukur biaya operasional murni. Tidak menyimpulkan sebab-akibat, sumber vulkanik, preprocessing final, atau arsitektur terbaik. Eksperimen ini tidak membuka test akhir untuk memilih kandidat.','',
      '## Integritas dan lokasi hasil','',
      f'{len(t["Integritas_SHA256"])} sumber/artefak lama cocok SHA256. Baseline tahap10 mencakup {len(c["initial_hashes"])} berkas; pemeriksaan dilengkapi baseline tahap9 yang sudah diverifikasi pada akhir tahap9. Kolom Baseline menunjukkan asal hash, termasuk seluruh hasil tahap7–9 dan arsip. Perubahan hash notebook02 sebelum resume tahap7 adalah catatan historis sebelumnya dan tetap dipertahankan; kecocokan saat ini tidak menghapus kejadian itu. Sebagian waktu komputasi tahap7 tidak tersedia, tidak direkonstruksi sebagai angka baru.','',
      'Seluruh keluaran tahap10 berada pada outputs/lstm_antarstasiun_tahap10/: konfigurasi terkunci, daftar kanal, dependensi, gate, prefix median, ledger sampel, scaler, input sekuens, timestamp,24 direktori model dengan checkpoint/riwayat/prediksi/waktu, tabel hasil dan grafik. Notebook: notebooks/10_lstm_antarstasiun.ipynb. Catatan kelanjutan: outputs/STATUS_PROYEK.md.','']
    text='\n'.join(lines)
    for old,new in {'statistik30':'statistik 30','1.440slot':'1.440 slot','tiap30':'tiap 30','tahap7':'tahap 7','tahap9':'tahap 9','tahap10':'tahap 10',
       'batch64':'batch 64','maksimal50':'maksimal 50','patience8':'patience 8','min_delta0':'min_delta 0','tetap0/1':'tetap 0/1','memakai1':'memakai 1',
       'seluruh48':'seluruh 48','transisi2025':'transisi 2025','jendela±72':'jendela ±72','Minimum30':'Minimum 30','pada≥2':'pada ≥2',
       'vs48':'vs 48','holdback15%':'holdback 15%','Dalam72':'Dalam 72','dan256':'dan 256','timestamp,24':'timestamp, 24','notebook02':'notebook 02'}.items():text=text.replace(old,new)
    # A word-boundary replacement avoids altering folder identifiers such as _tahap10.
    text=text.replace('lstm_antarstasiun_tahap 10','lstm_antarstasiun_tahap10')
    (RUN/'ringkasan_lstm_antarstasiun.md').write_text(tidy(text),encoding='utf-8')
    status=['# Status proyek SO₂','',
      '## Tahap selesai','',
      'Tahap1–3 audit struktur, kualitas dan rancangan preprocessing; tahap4 pilot imputasi; tahap5 Ridge, penelusuran sampel dan5B ablasi Curah Hujan; tahap6 expanding-window backtesting dan diagnosis fold pertama; tahap7 verifikasi preprocessing LSTM; tahap8 sensitivitas kalibrasi; arsip Way Urang v1; tahap9 kontribusi antarstasiun Ridge; tahap10 verifikasi Pulomerak dengan24 LSTM. Seluruh tahap di atas selesai, dengan keterbatasan masing-masing laporan tetap berlaku.','',
      '## Konfigurasi kerja','',
      'Way Urang masih pilot. Median kausal warmup30 hari tanpa Curah Hujan adalah pilihan operasional sementara, bukan metode terbukti paling akurat. Horizon30 menit, empat fold kalender tetap dari tahap6, development hanya sampai2026-06-19 06:00. Tahap10 memakai S0 Way Urang dan S1 tambahan SO₂ Pulomerak lag1 tiap langkah,48 slot berurutan,22/24 kanal termasuk missing asli. LSTM32 satu layer output linear, Adam0.001, MSE, batch64, maks50epoch, early stopping8, seed11/23/47. Validasi internal15% akhir kalender train. Median/scaler train internal saja, target asli distandardisasi train dan inverse sebelum skor.','',
      '## Lokasi hasil','',
      '- Arsip historis: outputs/arsip_way_urang_v1/ dan outputs/arsip_way_urang_v1.zip.','- Tahap7: outputs/lstm_preprocessing/ dan outputs/ringkasan_lstm_preprocessing.md.','- Tahap8: outputs/sensitivitas_kalibrasi/ dan outputs/ringkasan_sensitivitas_kalibrasi.md.','- Tahap9: outputs/kontribusi_antarstasiun/, outputs/hasil_kontribusi_antarstasiun.xlsx dan outputs/ringkasan_kontribusi_antarstasiun.md.','- Tahap10: outputs/lstm_antarstasiun_tahap10/hasil_lstm_antarstasiun.xlsx, ringkasan_lstm_antarstasiun.md, config_locked.json, daftar_fitur.json, dependensi_freeze.txt, tabel_hasil.json.','- Notebook tahap10: notebooks/10_lstm_antarstasiun.ipynb.','- Tiap model tahap10: Fold_XX/S0_seedXX atau S1_seedXX, complete.json berisi hash checkpoint/riwayat/prediksi; model lengkap tidak dilatih ulang saat resume.','- Environment kerja: .venv-lstm Python3.12/PyTorchCPU; .venv lama tetap dipertahankan.','',
      '## Temuan utama','',
      'Tahap9 Ridge tidak menunjukkan manfaat Pulomerak yang stabil: perbaikan kecil kedua error hanya pada fold2, sedangkan donor Talang/kombinasi tidak memberi manfaat stabil. Pada tahap10 Pulomerak memperbaiki kedua error pada '+str(int(wins.sum()))+'/12 pasangan fold–seed dan memperburuk keduanya pada '+str(int(losses.sum()))+'/12. Lihat tabel per fold/seed, transisi dan cakupan; tidak memilih seed terbaik.','',
      'Keputusan sementara tahap10: '+('S1 layak sebagai kandidat lanjutan dengan keterbatasan validasi pengembangan.' if wins.all() else 'S0 tetap acuan sederhana dan S1 disimpan sebagai sensitivitas, karena manfaat Pulomerak tidak konsisten di seluruh periode dan seed.'),'',
      table(['Skenario','MAE mean','MAE SD seed','RMSE mean','RMSE SD seed','Bias mean','Cakupan persen mean'],variation),'',
      '## Keterbatasan','',
      'Development telah digunakan berulang; ini validasi pengembangan. Validasi banyak mencakup konsentrasi rendah, dengan transisi fold1 yang harus dipertahankan. Zona waktu, sinkronisasi dan latensi antarstasiun belum dikonfirmasi; delay30menit merupakan asumsi. Nol, ekstrem/konstan dan off-grid dipertahankan sebagai penanda, bukan otomatis kesalahan. Cakupan model tidak sama dengan ketersediaan donor asli. Tidak ada bukti penyebab vulkanik atau kerusakan sensor.','',
      'Ada perubahan hash notebook02 sebelum resume tahap7 dan sebagian waktu komputasi lama tidak tersimpan. Kedua hal itu tercatat dalam laporan/arsip dan tidak diperbaiki dengan angka yang diarang. Seluruh hash lama dibandingkan terhadap baseline tahap10 saat ini, bukan klaim identik sepanjang sejarah. Percobaan preflight tahap10 gagal karena pip tidak tersedia, sebelum training; versi paket dicatat lewat metadata dan percobaan konfigurasi itu disimpan.','',
      '## Pekerjaan berikutnya','',
      'Tinjau kestabilan S0–S1 dan besarnya manfaat terhadap tambahan kompleksitas sebelum menetapkan kandidat. Klarifikasi waktu/latensi, dokumentasi sensor, periode konstan dan transisi dengan penyedia data. Jika tidak konsisten, simpan Pulomerak sebagai sensitivitas. Tidak melakukan tuning atau membuka test untuk mengejar skor. Rancangan perbandingan algoritma atau eksperimen berikutnya perlu ditetapkan terlebih dahulu.','',
      'Untuk melanjutkan sesi, baca file ini, ringkasan tahap10 dan config_locked.json, lalu periksa complete.json:24 model lengkap. Jangan menimpa hasil lama. Script scripts/lstm_interstation_stage10.py mendukung resume model yang belum lengkap, bukan melatih ulang model selesai. Scripts/summarize_interstation_stage10.py menyusun ulang laporan dari hasil tersimpan tanpa training.','',
      '## Status test akhir','',
      '**Test akhir mulai2026-06-19 06:30 belum digunakan untuk evaluasi forecasting dan tidak dimuat nilainya pada tahap10.** Data pernah diaudit menyeluruh pada tahap awal, sehingga jangan menyebutnya belum pernah dilihat sama sekali. Hashing sumber untuk integritas membaca byte, bukan evaluasi nilai test. Test tidak dipakai untuk seleksi preprocessing, arsitektur, seed atau warmup.','']
    (OUT/'STATUS_PROYEK.md').write_text(tidy('\n'.join(status)),encoding='utf-8')
