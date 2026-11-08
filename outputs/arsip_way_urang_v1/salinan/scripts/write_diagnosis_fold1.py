"""Report stored-model diagnosis, no training."""
import pandas as pd
from write_pilot_report import table
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'outputs'

def report(t,cfg,train_min,train_max,transition):
    model=t['Model_Ridge'];low=t['Origin_rendah'];checks=t['Pemeriksaan'];co=t['Koefisien_scaler']
    cols=['Pipeline','Periode','N','MAE','RMSE','Bias']
    overall=[r for r in t['Skor'] if r['Domain']=='bersama']
    features=[r for r in t['Variasi_fitur'] if r['Fitur']=='SO2_lag0']
    contrib=pd.DataFrame(t['Kontribusi_per_prediksi']);groupcols=['SO2 historis','Polutan lain','Meteorologi','Indikator SO2 missing','Indikator kovariat missing','Kontribusi SO2 lag0']
    avg=[]
    solar=next(r for r in t['Variasi_fitur'] if r['Pipeline']=='P0_median' and r['Fitur']=='Sol.Rad_lag1')
    solarcoef=next(r for r in co if r['Pipeline']=='P0_median' and r['Fitur']=='Sol.Rad_lag1')
    solar_z=(solar['Validasi maksimum']-solarcoef['Mean fitur train'])/solarcoef['Scale fitur train']
    for method,g in contrib[contrib['Irisan bersama']&(contrib['SO2 origin asli']<=2.14)].groupby('Pipeline',sort=False):
        avg.append({'Pipeline':method,'N':len(g),'Intercept':g.Intercept.iloc[0],**{k:float(g[k].mean()) for k in groupcols},'Prediksi mean':float(g.Prediksi.mean())})
    lines=['# Diagnosis kegagalan fold pertama tahap 6','',
    '**Tanpa fit model atau estimator baru, tuning, perubahan split maupun akses test akhir.** Seluruh prediksi, koefisien, mean dan scale berasal dari artefak fold pertama yang sudah tersimpan. Rekonstruksi hanya operasi aljabar atas parameter tersebut. Hasil lama dipertahankan.','',
    '## Penyelarasan prediksi dan sumber','',
    f"Fold pertama: train kalender {cfg['folds'][0]['Train mulai']} sampai {cfg['folds'][0]['Train akhir']}; target validasi {cfg['folds'][0]['Validasi mulai']} sampai {cfg['folds'][0]['Validasi akhir']}. Train model setelah kalibrasi90hari dan buffer48slot hanya mencakup April, dengan {model[0]['Train N']} target bersama pada rentang {train_min:.2f}–{train_max:.2f} µg/m³.", '',
    f"Seluruh {len(checks)} pemeriksaan tercantum pada workbook. Indeks target tepat origin+1, selisih timestamp tepat30menit, target finite sama dengan target asli. CSV fitur memiliki seluruh {cfg['fold_cuts'][1]-1} origin grid berurutan, termasuk slot berfitur kosong; seleksi finite memakai mask setelah penyusunan fitur. Tidak ditemukan perpindahan posisi akibat dropna/reindex. Lag0 sama dengan histori SO₂ pada origin; jika asli teramati nilainya harus sama persis. Untuk origin asli kosong, lag0 adalah hasil imputasi yang ditandai, bukan klaim pengamatan asli.", '',
    'Ketujuh lag SO₂ dibandingkan pada seluruh origin terhadap shift histori tersimpan; seluruh kovariat asli dibandingkan dengan origin−30menit. Urutan34kolom saat rekonstruksi sama dengan urutan parameter tersimpan dan jalur kode fit/predict.','',
    'Excel sumber dibaca read_only sampai sekitar akhir validasi fold pertama (Agustus2025), sebelum batas test akhir. Seluruh nilai pada slot grid dibandingkan langsung dengan Excel; baris yang tidak tersedia dipastikan memiliki nilai grid kosong. Tidak ada Excel disimpan ulang.','',
    table(['Variabel','Slot grid','Slot ada baris Excel','Slot tanpa baris','Ketidaksesuaian nilai','Satuan literal Excel'],t['Cocok_sumber_variabel']),'',
    'Contoh pemeriksaan sel sumber (daftar lengkap beserta nomor sel ada di workbook):','',
    table(['Peran','Timestamp grid','Sel SO2','SO2 literal Excel','SO2 grid audit'],[r for r in t['Sampel_Excel'] if abs((pd.Timestamp(r['Timestamp grid'])-transition).total_seconds())<=3600]),'',
    'Timestamp off-grid sumber dilaporkan terpisah pada workbook, tidak dibulatkan dan tidak menggeser grid. SO₂ pada baris off-grid ini kosong menurut audit sebelumnya.','',
    '## Scaling dan rekonstruksi keluaran','',
    'Mean dan scale seluruh34fitur dihitung ulang secara deskriptif dari timestamp train bersama tersimpan, lalu dibandingkan dengan parameter scaler. Semuanya cocok; kode tahap6 menempatkan `StandardScaler().fit(X[ids])` sebelum transform validasi. Tidak ada scaling target: Ridge dilatih terhadap SO₂ asli dalam satuan file. Karena itu tidak diperlukan inverse transform target. Intercept cocok dengan mean target train.','',
    'Persamaan setiap prediksi: ŷ = intercept + Σ βⱼ (xⱼ−mean_trainⱼ)/scale_trainⱼ. Rekonstruksi cocok dengan prediksi tersimpan pada semua timestamp pipeline yang finite; galat maksimum numerik:','',
    table(['Pipeline','Alpha','Intercept','Train N','Train R2 deskriptif','Jumlah slope SO2 satuan asli','Maks galat prediksi rekonstruksi'],model),'',
    'Tidak ditemukan indikasi penggunaan scaler validasi, kolom tertukar, target salah skala atau inverse transform yang hilang. Statistik minimum, persentil, mean dan SD target asli/prediksi per metode, bulan, serta domain mandiri/bersama ada pada sheet Statistik_target_prediksi.','',
    table(['Pipeline','Origin rendah asli N','Origin rendah median','Target median','Prediksi median','Prediksi mean','Prediksi minimum','Prediksi maksimum','MAE'],low),'',
    '“Origin rendah” di sini berarti SO₂ asli ≤2,14 µg/m³, penelusur hasil lama dan bukan ambang sensor/kualitas/bahaya. Statistik ini memakai origin yang asli teramati dan timestamp evaluasi bersama. Prediksi Ridge tetap jauh di atas konsentrasi rendah meskipun lag0 sudah rendah; tidak semua prediksi harus selalu tinggi, sehingga rentang lengkap dan prediksi negatif juga ditampilkan.','',
    '## Ridge, variasi fitur dan penalti','',
    'Konfigurasi tetap alpha=1, fit_intercept=True, solver=svd. Koefisien di workbook menyertakan urutan fitur, mean, scale, koefisien standar dan koefisien pada satuan asli. Fungsi objektif SSE+alpha×Σβ² memberi penalti L2 pada **semua34koefisien**, termasuk17indikator missing, polutan dan meteorologi; intercept tidak dipenalti. Tidak ada pengecualian khusus untuk meteorologi atau indikator. Kondisi stasioner Xstandarᵀ(ŷ−y)+alphaβ≈0 dan Σ(ŷ−y)≈0 diverifikasi dari parameter lama, tanpa menyelesaikan fit baru.','',
    table(['Pipeline','Fitur','Koefisien standar','Koefisien satuan asli','Mean fitur train','Scale fitur train'],[r for r in co if r['Fitur'].startswith('SO2_lag')]),'',
    'Variasi lag0 dan pergeseran pada validasi bersama:','',
    table(['Pipeline','Train minimum','Train median','Train maksimum','Train SD','Train nilai unik','Validasi minimum','Validasi median','Validasi maksimum','Di luar minmax train persen','Abs z lebih5'],features),'',
    'Lag0 tidak konstan persis, tetapi variasinya kecil dibanding perpindahan ke konsentrasi sekitar1–2: mean≈45,4 dan SD≈0,62 pada train. Saat origin≈1,56, z lag0 sekitar−70; slope pada satuan asli hanya sekitar0,04, sehingga kontribusi lag0 sekitar−1,6, tidak cukup membawa intercept≈45,4 menuju1,56. Jumlah slope ketujuh SO₂ historis juga jauh dari1. Jumlah slope adalah sensitivitas aljabar jika seluruh lag bergerak satu unit bersamaan, bukan efek kausal atau atribusi independen antar lag berkorelasi.','',
    'Rata-rata kontribusi saat origin asli rendah, pada timestamp bersama:','',
    table(['Pipeline','N','Intercept','SO2 historis','Polutan lain','Meteorologi','Indikator SO2 missing','Indikator kovariat missing','Prediksi mean'],avg),'',
    'Jumlah kontribusi+intercept persis merekonstruksi prediksi. Kontribusi rinci setiap fitur pada sampel terpilih dan kontribusi kelompok pada seluruh prediksi ada di workbook. Korelasi Pearson tiap fitur numerik dengan target train juga dicatat. R² train yang rendah dan respons SO₂ historis yang kecil mendukung bahwa model terutama mempelajari tingkat rata-rata dalam cohort sempit. Ini bukan bukti alpha=1 sebagai penyebab tunggal: tanpa eksperimen pembanding yang diizinkan, pengaruh penalti tidak dapat dipisahkan dari hubungan train, korelasi fitur dan keterbatasan representasi.','',
    'Variasi seluruh fitur:','',
    table(['Fitur','Train minimum','Train maksimum','Train SD','Konstan persis train','Validasi minimum','Validasi maksimum','Di luar minmax train persen','Abs z lebih5'],[r for r in t['Variasi_fitur'] if r['Pipeline']=='P0_median' and not r['Fitur'].startswith('missing_')]),'',
    'Rentang train, jumlah nilai unik, SD dan CV disajikan tanpa menetapkan ambang sensor. |z|>5 dan di luar min–maks train hanya deskripsi perpindahan fitur, bukan alasan menghapus data. Indikator yang selalu nol pada train tidak memiliki variasi untuk mempelajari efek missing; scale konstannya1 dan koefisiennya secara numerik mendekati nol. Sebagian fitur meteorologi memiliki SD kecil, sehingga pergeseran yang moderat pada satuan asli dapat menghasilkan z besar.','',
    f"Contoh konkret ekstrapolasi meteorologi P0: Sol.Rad lag1 memiliki rentang train {solar['Train minimum']:.2f}–{solar['Train maksimum']:.2f}, sementara maksimum validasi bersama {solar['Validasi maksimum']:.2f}, pada satuan literal sumber. Dengan scale train {solarcoef['Scale fitur train']:.6f}, maksimum tersebut menjadi z≈{solar_z:.1f}; kontribusi fitur ini sendiri ≈{solar_z*solarcoef['Koefisien standar']:.2f} µg/m³. Karena kontribusi dijumlahkan dengan intercept dan fitur lain, ekstrapolasi ini juga menjelaskan kemungkinan prediksi sangat rendah/negatif, selain kegagalan mengikuti tingkat rendah. Tidak menyatakan nilai meteorologi itu salah: kecocokan dengan Excel diverifikasi, tetapi validitas fisik dan dokumentasinya belum dipastikan.", '',
    '## Transisi dan kelanjutan error','',
    t['Aturan_transisi'][0]['Aturan'],'',
    table(['Timestamp sebelum','SO2 sebelum','Timestamp transisi','SO2 sesudah','Selisih SO2'],t['Aturan_transisi']),'',
    'Jendela grafik lokal dipatok ±72jam, dan grafik seluruh validasi disertakan agar error jauh setelah transisi tetap terlihat. SO₂ pada origin diplot pada timestamp target pasangannya; label ini adalah nilai tersedia30menit sebelumnya. Garis memiliki celah ketika baris evaluasi atau nilainya kosong.','',
    '![Sekitar transisi](figures/diagnosis_fold1/sekitar_transisi.png)','',
    '![Seluruh validasi](figures/diagnosis_fold1/seluruh_validasi.png)','',
    table(cols,overall),'',
    'Skor berdasarkan jarak dari transisi, seluruhnya pada timestamp bersama:','',
    table(['Pipeline','Periode','N','Target median','Prediksi median','MAE','RMSE','Bias','Abs residual lebih lebar train'],t['Sebelum_sesudah']),'',
    'Error tidak terbatas pada satu slot pergantian: tabel >7hari dan skor Juli/Agustus memperlihatkan error menetap sesudah transisi. “Residual besar” untuk penelusuran ini berarti |residual| lebih besar daripada lebar target train bersama, yakni '+f'{train_max-train_min:.2f} µg/m³; batas deskriptif dari train, bukan ambang kualitas.','',
    table(['Pipeline','Periode','N','Minimum','P1','P5','Median','P95','P99','Maksimum','Mean','SD'],[r for r in t['Residual'] if r['Domain']=='bersama']),'',
    '![Residual dan skor bulanan](figures/diagnosis_fold1/residual_dan_bulanan.png)','',
    '## Persistence dan cakupan','',
    'Donor persistence diverifikasi sebagai pengamatan SO₂ asli terakhir pada atau sebelum origin, bukan hasil imputasi dan bukan pengamatan masa depan. Usia donor=indeks origin−indeks donor, satu slot30menit.','',
    table(['Domain','Usia slot','Usia jam','Jumlah','Denominator target asli domain','Persen'],t['Persistence_usia']),'',
    table(['Pipeline','N','Denominator target asli','Cakupan persen','MAE','RMSE','Bias','SHA256 timestamp'],[r for r in t['Skor'] if r['Domain']=='bersama' and r['Periode']=='seluruh fold']),'',
    'Cakupan (%)=prediksi finite/seluruh target asli teramati jendela×100. Skor bersama memakai digest timestamp identik. Workbook juga menyimpan skor mandiri dengan denominator target asli yang sama; skor mandiri tidak digunakan untuk mengklaim perbandingan terkontrol. MAE=Σ|e|/N; RMSE=√(Σe²/N); bias=Σe/N, e=prediksi−pengamatan. Distribusi residual memakai N pada domain yang dinyatakan.','',
    '## Klasifikasi temuan','',
    '**A. Kesalahan implementasi yang terbukti:** tidak ditemukan pada pemeriksaan ini. Alignment, grid, lag, urutan fitur, scaler train-only, target tanpa scaling, rekonstruksi prediksi dan baseline cocok. Ini bukan jaminan bahwa semua kemungkinan bug telah dibuktikan tidak ada. Tidak ada perbaikan atau penimpaan hasil lama.','',
    '**B. Keterbatasan model yang didukung pemeriksaan:** kalibrasi dan buffer meninggalkan cohort train tinggi dan sempit; Ridge memiliki intercept sekitar45,4 dan respons historis yang lemah, sehingga tidak mengikuti perpindahan ke tingkat rendah. Fitur validasi keluar jauh dari rentang train; error berlanjut lama setelah transisi. Gate minimal340sampel menguji kecukupan komputasional, bukan keterwakilan distribusi atau daya generalisasi. Kegagalan ini konsisten dengan konsekuensi rancangan dan keterbatasan model pada distribusi berbeda. Tidak ada tuning atau model tandingan untuk mengisolasi penyebab tunggal.','',
    '**C. Ketidakpastian data yang belum dapat dijelaskan:** penurunan SO₂ dan perubahan kovariat tercantum pada sumber, tetapi penyebab, keabsahan pengamatan, riwayat alat/kalibrasi, satuan atau kode yang perlu klarifikasi, keterlambatan pelaporan, serta zona waktu memerlukan penyedia data. Tidak menyimpulkan aktivitas vulkanik atau kerusakan sensor. Nilai tinggi/rendah dan transisi tidak dihapus.','',
    f"Integritas: {len(t['Integritas_SHA256'])} berkas sumber/artefak lama memiliki SHA256 sebelum–sesudah identik. Tidak ada estimator di-fit dan test akhir tidak dibuka; hashing byte dilakukan hanya untuk integritas.", '',
    'Artefak: `diagnosis_fold1.xlsx`, `diagnosis_fold1/aturan_diagnosis.json`, `diagnosis_fold1/prediksi_dan_origin.csv`, dan grafik `figures/diagnosis_fold1/`. Parameter lengkap, pemeriksaan, nilai sumber, kontribusi dan hash tersimpan dalam workbook.','']
    (OUT/'diagnosis_fold1.md').write_text('\n'.join(lines),encoding='utf-8')
