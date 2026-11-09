"""Indonesian research report and project status update for stage11."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from write_pilot_report import table
from audit_quality import hashfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'perbandingan_algoritma_tahap11'

def write_report(t,c):
    common=[r for r in t['Skor_fold_seed'] if r['Domain']=='bersama'];pooled=[r for r in t['Gabungan_per_seed'] if r['Domain']=='bersama']
    mean=[r for r in t['Ringkasan'] if r['Domain']=='bersama'];overall=[r for r in mean if r['Fold']=='Gabungan_OOF']
    oldstatus=OUT/'STATUS_PROYEK.md';previous=oldstatus.read_text(encoding='utf-8')
    assert hashfile(oldstatus)==c['status_before_sha256'] or '## Tahap 11 selesai — perbandingan algoritma S0' in previous
    if '## Tahap 11 selesai — perbandingan algoritma S0' in previous:
        previous=previous.split('\n## Tahap 11 selesai — perbandingan algoritma S0')[0]
    p=pd.DataFrame(pooled);v=pd.DataFrame(overall).set_index('Algoritma')
    dl=['LSTM','GRU','TCN'];mae_best=min(dl,key=lambda x:v.loc[x,'MAE mean']);rmse_best=min(dl,key=lambda x:v.loc[x,'RMSE mean'])
    candidates=[mae_best] if mae_best==rmse_best else [mae_best,rmse_best]
    cmp=[];foldmatrix=pd.DataFrame([r for r in common if r['Fold']!='Gabungan_OOF'])
    for alg in dl:
        both=0;mae=0;rmse=0;beatsdl=0
        for (fold,seed),g in foldmatrix.groupby(['Fold','Seed'],sort=False):
            z=g.set_index('Algoritma');r=z.loc[alg];base=z.loc['Persistence']
            mae+=int(r.MAE<base.MAE);rmse+=int(r.RMSE<base.RMSE);both+=int(r.MAE<base.MAE and r.RMSE<base.RMSE)
            beatsdl+=int(all(r.MAE<=z.loc[o,'MAE'] and r.RMSE<=z.loc[o,'RMSE'] for o in dl if o!=alg))
        cmp.append({'Algoritma':alg,'MAE lebih rendah dari persistence N/12':mae,'RMSE lebih rendah dari persistence N/12':rmse,
            'Kedua metrik lebih rendah dari persistence N/12':both,'Mengungguli dua DL lain pada kedua metrik N/12':beatsdl})
    times=pd.DataFrame(t['Ringkas_waktu']).set_index('Algoritma')
    post=pd.DataFrame(t['Transisi_22_juni']);post=post[post.Periode.eq('sesudah')]
    trans=[]
    for alg,g in post.groupby('Algoritma',sort=False):
        trans.append({'Algoritma':alg,'Target mean':g['Target mean'].iloc[0],
            'Prediksi mean minimum antar-seed':g['Prediksi mean'].min(),'Prediksi mean maksimum antar-seed':g['Prediksi mean'].max(),
            'MAE mean antar-seed':g.MAE.mean(),'RMSE mean antar-seed':g.RMSE.mean()})
    allcoverage=all(np.isclose(v.loc[x,'Cakupan persen mean'],100) for x in ['LSTM','GRU','TCN'])
    haspooledwin=any(all(v.loc[alg,k+' mean']<v.loc['Persistence',k+' mean'] for k in ['MAE','RMSE']) for alg in dl)
    lines=['# Tahap 11 — perbandingan algoritma SO₂ Way Urang','',
      'Eksperimen selesai dengan 24 model baru: GRU dan TCN masing-masing empat fold × tiga seed. Dua belas LSTM S0 tahap10 dipakai sebagai hasil acuan tersimpan, tanpa fit ulang. Target seluruh model adalah SO₂ Way Urang asli 30 menit setelah origin. Empat fold kalender, 15% akhir train sebagai validasi internal, warmup median30 hari, kanal Way Urang tanpa Curah Hujan, indikator missing dan sekuens48 slot identik. Development set telah digunakan sebelumnya; ini validasi pengembangan. Test akhir mulai '+c['test_start']+' tidak dimuat nilainya atau dievaluasi.','',
      '## Protokol terkunci sebelum training','',
      'Konfigurasi dikunci sebelum fit baru dengan SHA256 '+hashfile(RUN/'config_locked.json')+'. Gate semua fold dan uji kausalitas TCN selesai saat jumlah fit baru=0. Input S0 langsung memakai tensor `.npz`, timestamp train/internal/evaluasi dan parameter scaler/target tahap10 yang diverifikasi. Tidak ada estimator imputasi atau scaler baru yang di-fit. Median kausal statistik hanya berasal dari prefix train internal tahap10; target asli tidak diimputasi.','',
      table(['Fold','Train N','Internal N','Evaluasi N','Fitur kanal','Fitur terverifikasi','Scaler train internal','Fit baru sebelum gate','Lulus'],t['Gate_sebelum_fit']),'',
      table(['Fold','Bagian','Awal target','Akhir target','N','Minimum','P25','Median','P75','Maksimum','Mean','SD'],t['Distribusi_target']),'',
      'Setiap sampel memiliki48 slot30 menit origin−47 hingga origin, tanpa kompresi waktu. Kanal numerik11 distandardisasi dari slot train internal tahap10; 11 indikator missing asli tetap0/1. Target distandardisasi memakai mean/SD populasi train internal yang sama untuk semua arsitektur, lalu prediksi di-inverse sebelum penilaian. Sampel yang targetnya melintasi batas antarbagian sudah dipurge pada tahap10 dan indeksnya cocok dengan tensor tersimpan. Target evaluasi kosong tidak dinilai. Nol, ekstrem dan rentetan konstan tidak dihapus berdasarkan besarnya.','',
      'GRU: satu layer unidirectional32 unit, Dense linear satu output. TCN: empat residual block16 kanal, tiapblock dua Conv1d kernel3 dengan dilation1/2/4/8, ReLU, dropout0 dan output linear dari langkah terakhir. Blok pertama memakai proyeksi skip1×1 dari22 ke16 kanal agar penjumlahan residual terdefinisi. Tiapconv melakukan padding2×dilation lalu memangkas bagian kanan sehingga keluaran waktu t hanya bergantung pada masukan≤t. Receptive field teoritis61 slot dibatasi jendela48 slot. Uji pemotongan prefix dan gangguan masa depan pada enam panjang prefix menunjukkan keluaran terdahulu tidak berubah.','',
      table(['Algoritma','Model N','Jumlah parameter','Epoch total','Epoch median','Fit total detik','Fit median detik','Predict total detik'],t['Ringkas_waktu']),'',
      'Semua model baru memakai Adam learning rate0,001, MSE target standar, batch64, maksimal50 epoch, early stopping patience8, bobot terbaik internal dipulihkan. Seed11/23/47 ditetapkan sebelum fit. Anggaran ini sama, tetapi jumlah parameter, biaya per-epoch, epoch terbaik dan tingkat optimisasi tidak otomatis sama. TCN memiliki receptive field lebih panjang dari jendela input tanpa membaca setelah origin. Tidak ada clipping atau pemilihan seed terbaik.','',
      '## Hasil pada timestamp yang identik','',
      table(['Fold','Seed','Algoritma','N','Denominator target asli','Cakupan persen','MAE','RMSE','Bias','Prediksi negatif'],common),'',
      'Error e=prediksi−target asli; MAE=mean(|e|), RMSE=sqrt(mean(e²)), bias=mean(e), seluruhnya µg/m³. Cakupan=100×N prediksi finite / seluruh target Way Urang numerik asli teramati pada jendela fold. Skor bersama memakai irisan timestamp LSTM/GRU/TCN/persistence; digest timestamp disimpan dalam workbook. Workbook juga memuat domain mandiri sehingga kemungkinan kegagalan prediksi tidak tersembunyi oleh irisan. Persistence memakai pengamatan SO₂ Way Urang asli terakhir pada/sebelum origin, tanpa imputasi atau batas usia.','',
      table(['Fold','Algoritma','MAE mean','MAE SD seed','RMSE mean','RMSE SD seed','Bias mean','Bias SD seed','Cakupan persen mean','Prediksi negatif mean'],[r for r in mean if r['Fold']!='Gabungan_OOF']),'',
      'Rata-rata dan SD sampel (ddof1) memakai tiga seed tetap. Persistence deterministik diulang dalam tabel seed hanya agar semua pembandingan memakai timestamp sama; pengulangannya bukan tiga replikasi independen. Tidak menarik kesimpulan signifikansi statistik dari SD atau selisih rerata.','',
      '## Gabungan fold dan perubahan terhadap persistence','',
      table(['Seed','Algoritma','N','Cakupan persen','MAE','RMSE','Bias','Prediksi negatif'],pooled),'',
      'Gabungan dihitung dari seluruh error evaluasi fold yang tidak tumpang tindih per seed; RMSE gabungan tidak dirata-ratakan dari RMSE fold.','',
      table(['Algoritma','MAE mean','MAE SD seed','RMSE mean','RMSE SD seed','Bias mean','Cakupan persen mean','Prediksi negatif mean'],overall),'',
      table(['Algoritma','MAE lebih rendah dari persistence N/12','RMSE lebih rendah dari persistence N/12','Kedua metrik lebih rendah dari persistence N/12','Mengungguli dua DL lain pada kedua metrik N/12'],cmp),'',
      table(['Fold','Seed','Algoritma','N identik','Delta MAE vs persistence','Delta RMSE vs persistence','Perbaikan MAE persen','Perbaikan RMSE persen'],t['Vs_persistence']),'',
      'Delta=error model−error persistence; negatif berarti model lebih baik. Perbaikan persen=100×(error persistence−error model)/error persistence, sehingga positif berarti error lebih rendah. Semua selisih dihitung pada target asli dan timestamp yang sama. Hasil bulanan, perubahan bulanan, dan distribusi target tiap bulan ada dalam workbook; bulan yang memotong batas fold dipisahkan menurut fold.','',
      '![Skor per fold](figures/skor_per_fold.png)','',
      '## Transisi 22 Juni 2025','',
      'Tanggal pusat 2025-06-22 17:30 mengikuti diagnosis sebelumnya; jendela ±72 jam dan fold pertama dipertahankan tanpa memilihnya berdasarkan skor tahap11. Grafik menampilkan target asli, ketiga arsitektur dan persistence untuk semua seed tanpa clipping.','',
      table(['Algoritma','Target mean','Prediksi mean minimum antar-seed','Prediksi mean maksimum antar-seed','MAE mean antar-seed','RMSE mean antar-seed'],trans),'',
      table(['Periode','Fold','Seed','Algoritma','N','MAE','RMSE','Bias','Prediksi negatif'],t['Transisi_22_juni']),'',
      '![Transisi](figures/transisi_22_juni.png)','',
      '## Biaya dan reproduksibilitas','',
      table(['Fold','Algoritma','Seed','Sumber','Jumlah parameter','Epoch dijalankan','Epoch terbaik','Fit early stopping detik','Predict detik'],t['Waktu_dan_parameter']),'',
      'Waktu fit adalah jumlah durasi epoch train+validasi internal, tidak termasuk persiapan tensor, penulisan checkpoint, dan jeda resume. Waktu predict mencakup inferensi serta inverse target scaling. Angka berasal dari pengukuran tiap model, bukan estimasi waktu hilang. LSTM berasal dari run tahap10 di lingkungan yang sama namun pada waktu berbeda; total waktu ini deskriptif, bukan benchmark perangkat terisolasi.','',
      'Setiap epoch model baru menyimpan model, optimizer, RNG, bobot terbaik internal, histori dan waktu kumulatif secara atomik. Setelah model selesai, checkpoint terbaik, histori, prediksi dan waktu masing-masing langsung ditulis lalu ditandai `complete.json` beserta SHA256. Model yang sudah lengkap dilewati pada resume. Tidak melatih ulang LSTM tahap10. Grafik validasi internal tersedia tanpa memakai fold evaluasi untuk early stopping.','',
      '![Riwayat internal](figures/riwayat_internal.png)','',
      '![Waktu training](figures/waktu_training.png)','',
      '## Kesimpulan terbatas','']
    if mae_best==rmse_best:
        candidate=mae_best
        lines += [f'Di antara tiga arsitektur deep learning yang diuji, **{candidate}** memiliki rerata MAE dan RMSE gabungan terendah antar-seed. Ini kandidat relatif dalam rancangan tetap ini, dengan sebaran seed/fold, cakupan dan waktu pada tabel sebagai batas interpretasi. Peringkat satu rerata tidak membuktikan kestabilan, signifikansi, atau kemampuan pada rezim tinggi yang tidak tersedia di validasi berikutnya.','']
    else:
        lines += [f'Rerata MAE gabungan paling rendah dimiliki {mae_best}, sedangkan RMSE paling rendah dimiliki {rmse_best}. Keduanya adalah kandidat berbasis metrik yang berbeda; tabel per fold/seed, cakupan dan waktu belum mendukung satu pemenang deep learning tanpa pilihan kriteria tambahan.','']
    if mae_best=='GRU' and rmse_best=='LSTM':
        lines += [f'Untuk verifikasi penelitian berikutnya, **LSTM S0** adalah kandidat relatif yang lebih hemat waktu fit terukur ({times.loc["LSTM","Fit total detik"]:.1f} vs {times.loc["GRU","Fit total detik"]:.1f} detik untuk 12 model), memiliki RMSE gabungan lebih rendah ({v.loc["LSTM","RMSE mean"]:.3f} vs {v.loc["GRU","RMSE mean"]:.3f}) dan SD antar-seed gabungan lebih kecil pada MAE/RMSE. **GRU S0** tetap kandidat sensitivitas karena MAE gabungannya sedikit lebih rendah ({v.loc["GRU","MAE mean"]:.3f} vs {v.loc["LSTM","MAE mean"]:.3f}). Cakupan keduanya 100% pada target asli teramati; kelebihan kecil ini tidak konsisten untuk semua fold dan keduanya tertinggal jauh dari persistence. Perbedaan waktu mencerminkan juga jumlah epoch dan kondisi run, bukan hanya arsitektur.','']
    q1=pd.DataFrame([r for r in common if r['Fold']=='Fold_01']).groupby('Algoritma')[['MAE','RMSE']].mean()
    tr=pd.DataFrame(trans).set_index('Algoritma')
    lines += [f'Fold pertama memperlihatkan kegagalan TCN yang besar: rerata MAE {q1.loc["TCN","MAE"]:.3f} dan RMSE {q1.loc["TCN","RMSE"]:.3f} µg/m³, dibanding persistence {q1.loc["Persistence","MAE"]:.3f} dan {q1.loc["Persistence","RMSE"]:.3f}. Pada 72 jam sesudah pusat transisi, target mean {tr.loc["Persistence","Target mean"]:.3f}, sedangkan prediksi mean LSTM {tr.loc["LSTM","Prediksi mean minimum antar-seed"]:.3f}–{tr.loc["LSTM","Prediksi mean maksimum antar-seed"]:.3f}, GRU {tr.loc["GRU","Prediksi mean minimum antar-seed"]:.3f}–{tr.loc["GRU","Prediksi mean maksimum antar-seed"]:.3f}, dan TCN {tr.loc["TCN","Prediksi mean minimum antar-seed"]:.3f}–{tr.loc["TCN","Prediksi mean maksimum antar-seed"]:.3f}. Ketiganya lambat mengikuti konsentrasi rendah; ini deskripsi prediksi dan tidak menetapkan penyebab perubahan data.','']
    if haspooledwin:
        lines += ['Setidaknya satu arsitektur memiliki rerata gabungan MAE dan RMSE lebih rendah daripada persistence pada timestamp yang sama. Keunggulan tersebut tetap perlu dibaca per fold/seed serta diuji di luar development sebelum klaim operasional.','']
    else:
        lines += ['**Tidak ada arsitektur deep learning yang mengungguli persistence pada rerata gabungan MAE dan RMSE sekaligus.** Sebutan kandidat relatif di antara deep learning tidak berarti lebih baik daripada persistence. Hasil ini dipertahankan tanpa mengubah fold, seed, konfigurasi, atau mengevaluasi test untuk mengejar kemenangan.','']
    if not allcoverage:lines += ['Cakupan beberapa model berbeda; peringkat berdasarkan domain bersama tidak boleh digeneralisasi ke seluruh target tanpa membaca cakupan mandiri.','']
    else:lines += ['Ketiga arsitektur memiliki cakupan bersama dan mandiri penuh pada target asli teramati. Kestabilan error, terutama fold pertama dan transisi, tetap menjadi batas utama; cakupan penuh tidak berarti semua slot grid atau pengamatan sensor valid.','']
    lines += ['Model yang relatif paling baik di antara tiga deep learning tetap hanya **kandidat penelitian**. Hasil ini belum menjadikan model siap dipakai sebagai peringatan dini operasional. Zona waktu, latensi, kualitas sensor dan variasi konsentrasi memerlukan klarifikasi lanjutan. Tidak menyimpulkan penyebab transisi atau kaitan dengan sumber vulkanik.','',
      '## Integritas dan lokasi','',
      f'{len(t["Integritas_SHA256"])} berkas sumber/artefak lama cocok SHA256 terhadap baseline sebelum tahap11. `STATUS_PROYEK.md` adalah satu-satunya artefak lama yang diperbarui sesuai permintaan, dengan hash sebelum tercatat di konfigurasi. Workbook, notebook, konfigurasi terkunci, gate, daftar fitur, timestamp, prediksi tiap seed, checkpoint, histori, tabel dan grafik ada di outputs/perbandingan_algoritma_tahap11/. LSTM lama tetap di outputs/lstm_antarstasiun_tahap10/. Test akhir tidak dibuka.','']
    (RUN/'ringkasan_perbandingan_algoritma.md').write_text('\n'.join(lines),encoding='utf-8')
    addition=['','## Tahap 11 selesai — perbandingan algoritma S0','',
      'GRU32 dan TCN empat blok kausal masing-masing dilatih pada empat fold × tiga seed; total24 fit baru. LSTM S0 tahap10 digunakan dari checkpoint tersimpan, tanpa fit ulang. Tensor S0 median kausal warmup30 hari, 48 langkah, 22 kanal termasuk indikator missing, horizon30 menit, target asli dan batas internal/evaluasi sama. Konfigurasi dan uji kausalitas TCN dibekukan sebelum fit.','',
      'Lokasi: outputs/perbandingan_algoritma_tahap11/config_locked.json, daftar_fitur.json, hasil_perbandingan_algoritma.xlsx, ringkasan_perbandingan_algoritma.md, tabel_hasil.json, prediksi_oof_seluruh_seed.csv, timestamp_semua_bagian.csv, Fold_XX/GRU_seedXX dan TCN_seedXX (checkpoint, histori, prediksi, waktu), figures/, dan manifest_tahap11.json. Notebook: notebooks/11_perbandingan_algoritma.ipynb. Environment tetap .venv-lstm PyTorch CPU; .venv lama tidak diubah.','',
      f'Temuan: rerata MAE terendah di antara deep learning adalah {mae_best}; rerata RMSE terendah adalah {rmse_best}. '+('Tidak ada satu pemenang DL yang sama pada kedua metrik.' if mae_best!=rmse_best else 'Ini hanya kandidat relatif di antara DL.')+(' Tidak ada DL yang mengungguli persistence pada kedua rerata gabungan.' if not haspooledwin else ' Keunggulan terhadap persistence perlu dibaca per fold dan seed.')+' Lihat laporan untuk angka per fold, seed, bulan, bias, cakupan, transisi dan waktu.','',
      'Batas: development telah dipakai berulang; fold pertama dan transisi22 Juni 2025 tidak dihapus; biaya per model dan optimisasi berbeda walaupun anggaran epoch sama; zona waktu, latensi, sensor dan rentang konsentrasi belum tuntas. Arsip tahap1–10 tetap sama menurut SHA256. File STATUS ini memang diperbarui; hash sebelum/sesudah dicatat. Jangan memilih seed terbaik atau menyebut model siap untuk peringatan dini operasional.','',
      'Pekerjaan berikutnya: tinjau apakah ada kandidat yang layak dirancang untuk verifikasi terpisah, bandingkan dengan persistence pada timestamp yang sama, minta klarifikasi metadata waktu/latensi/kualitas dari penyedia data. Bekukan rancangan sebelum test akhir. **Test akhir mulai2026-06-19 06:30 belum digunakan untuk evaluasi forecasting tahap11 dan tidak dimuat nilainya.** Data pernah diaudit menyeluruh pada tahap awal sehingga bukan data yang belum pernah dilihat sama sekali.','']
    oldstatus.write_text(previous.rstrip()+'\n'+'\n'.join(addition),encoding='utf-8')

