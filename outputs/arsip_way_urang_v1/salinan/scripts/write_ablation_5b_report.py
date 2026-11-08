"""Laporan 5B membedakan ablasi pada sampel tetap dan perluasan train."""
from pathlib import Path
import numpy as np
import pandas as pd
from write_pilot_report import table as base_table

OUT=Path(__file__).resolve().parents[1]/'outputs'

def table(columns,rows):
    formatted=[]
    for r in rows:
        item=dict(r)
        for key,value in item.items():
            if isinstance(value,(float,np.floating)) and np.isfinite(value) and 'delta' in key.lower():
                item[key]=f'{value:.8f}'
        formatted.append(item)
    return base_table(columns,formatted)

def write_report(f):
    from ablation_5b import MAIN,FIXED,LEGACY
    comp=pd.DataFrame(f.compare);old=comp[comp.Lengan==LEGACY].set_index('Pipeline')
    fixed=comp[comp.Lengan==FIXED].set_index('Pipeline');new=comp[comp.Lengan==MAIN].set_index('Pipeline')
    g=f.gate;count=int(f.global_common.sum());den=len(f.target_ids)
    spread={label:float(frame.loc[f.methods,'MAE'].max()-frame.loc[f.methods,'MAE'].min()) for label,frame in [(LEGACY,old),(FIXED,fixed),(MAIN,new)]}
    lines=['# Tahap 5B — ablasi Curah Hujan pada pilot Way Urang','',
        '**Eksperimen terpisah; seluruh sumber dan artefak lama dipertahankan.** '
        'Test akhir tidak dimuat nilainya atau dievaluasi. Hanya Ridge dengan konfigurasi '
        'tahap kelima digunakan. P3 tetap ditunda. Ini bukan pemilihan preprocessing final.','',
        '## 1. Rancangan dan perubahan yang diizinkan','',
        '- Curah Hujan dikeluarkan dari **matriks imputasi**, scaler jarak KNN, '
        'fitur Ridge, dan indikator missing terkait. Sepuluh kovariat tersisa sama '
        'untuk ketiga pipeline. Imputer/scaler prefix diestimasi ulang, bukan '
        'sekadar menghapus kolom dari hasil imputasi lama.',
        '- P0 median, P1 forward fill L4 tanpa fallback, P2 KNN k10 berbobot '
        'distance/nan_euclidean. Guard minimum dua kovariat asli tetap **dua**, '
        'meskipun jumlah kovariat berkurang. Tidak ada parameter yang dilonggarkan.',
        '- Split70/15/15, horizon30menit, lagSO₂0/1/2/6/12/24/48, lagkovariat1, '
        'kalibrasi90hari, buffer48slot, pembaruanprefix30hari, aturan kualitas '
        'dan `Ridge(alpha=1, fit_intercept=True, solver=svd)` tetap seperti tahap5.',
        '- Histori train dibentuk dari prefix yang berakhir sebelum query. '
        'Saat validasi, imputer/scaler jarak dibekukan dari seluruh train, '
        'histori lama tidak dihitung ulang. Tidak ada target masa depan menjadi '
        'fitur, donor atau nilai pengisi. Tidak ada clipping/fallback tambahan.',
        '- Fitur Ridge menjadi **34 kolom**: tujuh lag SO₂, sepuluh kovariat lag1 '
        'dan tujuh belas indikator missing asli. Nol, lonjakan, pola konstan '
        'dan ledger off-grid dipertahankan sesuai aturan lama.',
        '- Asumsi latensi tidak berubah: SO₂(u) dianggap tersedia saat origin u; '
        'kovariat(u−1) dianggap sudah tiba pada u. Waktu kedatangan aktual '
        'masih perlu konfirmasi penyedia data.',
        '- Dua lengan memakai **histori tanpa hujan yang sama**: '
        '`5B_train_diperluas` memakai train bersama baru; `5B_train_lama_tetap` '
        'memakai timestamp Ridge train lama persis. Imputer prefix memakai '
        'bagian train yang diizinkan, bukan validasi, pada kedua lengan.',
        f"- Konfigurasi disimpan sebelum fit di `ablasi_curah_hujan_5B/config_locked.json`, "
        f"SHA256 `{f.config_hash}`. Gate kelayakan disimpan ketika hitungan Ridge fit masih nol.",'',
        table(['Parameter','Sama dengan tahap 5','Nilai'],f.invariants),'',
        '## 2. Verifikasi sampel sebelum pelatihan Ridge','',
        f"Gate **lulus sebelum Ridge**: train bersama bertambah dari **{g['Train bersama lama']}** "
        f"menjadi **{g['Train bersama baru']}**; tambahan **{g['Tambahan timestamp']}** origin. "
        f"Seluruh timestamp train lama tetap layak ({g['Timestamp lama tidak lagi layak']} hilang). "
        f"Pengamatan target **>2,14 µg/m³** bertambah dari **{g['Target tinggi lama']}** "
        f"menjadi **{g['Target tinggi baru']}**, seluruh tambahannya kembali masuk. "
        f"Rentang target baru **{g['Minimum target baru']:g}–{g['Maksimum target baru']:g} µg/m³**, "
        f"median **{g['Median target baru']:g}**.",'',
        '**2,14 hanya ambang penelusuran kohort lama**, bukan batas kualitas, '
        'bahaya atau outlier. Nilai tidak dihapus berdasarkan konsentrasi. '
        'Gate ini dilakukan setelah rekonstruksi histori agar kelengkapan '
        'fitur dapat diketahui, tetapi sebelum ada model Ridge dilatih.','',
        '### Alur per pipeline','',
        'Seperti penelusuran sebelumnya, tahap awal memakai origin pada train '
        'dengan target t+1. Satu target awal validasi muncul sebelum purge. '
        'Inventaris train murni tetap 19.197 SO₂ teramati; pasangan origin '
        'mempunyai19.198 target teramati sebelum purge. Lag struktural hanya '
        'memeriksa indeks waktu, bukan membuang missing sebelum imputasi.','',
        table(['Pipeline','Tahap','Jumlah sampel','Target asli teramati','Minimum SO2','Median SO2','Maksimum SO2','Target lebih 2.14'],f.flow),'',
        '### Distribusi dan perubahan train per bulan','',
        table(['Lengan','Bulan','Jumlah sampel','Target awal','Target akhir','Minimum SO2','Median SO2','Maksimum SO2','Target lebih 2.14'],f.train_distribution),'',
        table(['Bulan target','Train lama','Train baru','Perubahan jumlah','Target lebih 2.14 lama','Target lebih 2.14 baru'],f.monthly_added),'',
        '![Perluasan train](figures/ablasi_curah_hujan_5B/perluasan_train.png)','',
        '## 3. Hasil utama dan cakupan','',
        'Target evaluasi selalu SO₂ asli teramati. `e=prediksi−pengamatan`, '
        'MAE=mean(abs(e)), RMSE=sqrt(mean(e²)), bias=mean(e), satuanµg/m³. '
        'Cakupan(%)=prediksi finite/seluruh target asli validasi×100%. '
        'Pada tabel bulanan denominator hanya target asli bulan itu. '
        'Prediksi negatif, bila ada, dilaporkan tanpa clipping.',
        'Persistence tidak dilatih. Donornya SO₂ asli terakhir pada/sebelum '
        'origin dengan umur yang sama persis seperti tahap kelima; kesamaan '
        'nilai dan usia donor diverifikasi.','',
        table(['Lengan','Pipeline','Domain','Jumlah prediksi','Denominator target asli','Cakupan persen','MAE','RMSE','Bias','Prediksi negatif'],f.metrics),'',
        'Tabel `Bulanan_5B_cakupan` pada workbook memuat domain seluruh berhasil '
        'dan irisan internal5B setiap bulan untuk kedua lengan. Skor domain '
        'masing-masing tidak digunakan sebagai pengganti perbandingan timestamp identik.','',
        '## 4. Perbandingan lama dan baru pada target identik','',
        f"Irisan global mencakup **{count}/{den} target asli validasi "
        f"({100*count/den:.3f}%)**: ketiga pipeline dan persistence harus finite "
        'pada kondisi lama, 5B trainlama dan 5B traindiperluas. Semua baris '
        'perbandingan memakai hash timestamp target yang sama. Ini menjaga '
        'perubahan eligibility evaluasi tidak tercampur dengan perubahan skor.','',
        table(['Lengan','Pipeline','Jumlah prediksi','Cakupan persen','MAE','RMSE','Bias','Prediksi negatif'],f.compare),'',
        '![Perbandingan identik](figures/ablasi_curah_hujan_5B/perbandingan_identik.png)','',
        '### Per bulan pada timestamp identik','',
        table(['Lengan','Pipeline','Bulan target','Jumlah prediksi','Denominator target asli','Cakupan persen','MAE','RMSE','Bias'],f.compare_month),'',
        'Maret dan Juni adalah bulan parsial. Perbedaan bulanan harus dibaca '
        'bersama jumlah target dan bias, bukan hanya MAE agregat.','',
        '![Perbandingan bulanan](figures/ablasi_curah_hujan_5B/bulanan_identik.png)','',
        '## 5. Analisis sensitivitas','',
        '- **Lama→5B trainlama:** timestamp Ridge train tetap. Ini menguji '
        'ablasi Curah Hujan pada sampel yang sama, termasuk perubahan representasi '
        'akibat refit imputer tanpa hujan. Untuk KNN, jarak dan rekonstruksi '
        'kovariat dapat berubah. Bukan sekadar menghapus satu koefisien Ridge.',
        '- **5B trainlama→5B diperluas:** kedua lengan memakai matriks fitur dan '
        'histori tanpa hujan yang sama; perubahan berasal dari sampel fit Ridge '
        'dan scaler fiturnya yang bertambah. Tidak ada tuning parameter tambahan.',
        '- **Lama→5B diperluas:** gabungan ablasi dan komposisi train baru. '
        'Eksperimen utama ini tidak mengisolasi pengaruh murni Curah Hujan. '
        'Sensitivitas membantu pemisahan operasional, bukan bukti kausal universal.',
        '- Delta di bawah adalah **sesudah−sebelum** pada timestamp identik. '
        'Delta negatif MAE/RMSE berarti lebih rendah. Delta bias bertanda '
        'tidak otomatis berarti lebih baik; nilai absolut bias juga perlu dibaca.','',
        table(['Perbandingan','Pipeline','Jumlah target identik','Delta MAE','Delta RMSE','Delta bias'],f.deltas),'',
        '## 6. Waktu komputasi','',
        f"Histori/imputer tanpa hujan dibentuk sekali dalam **{f.impute_seconds:.6f} detik**, "
        'kemudian dipakai bersama oleh dua lengan. Jangan menggandakan biaya '
        'ini saat menjumlah waktu. Prefix log mencatat biaya estimasi median, '
        'scaler dan KNN serta transform tiap blok; waktunya terikat mesin dan run ini.','',
        table(['Lengan','Pipeline','Jumlah train','Scaler fitur fit detik','Ridge fit detik','Predict detik'],f.timing),'',
        'Tidak ada refit model lama. Nilai lama disalin dari prediksi tersimpan '
        'dan diberi skor ulang hanya pada subset validasi identik.','',
        '## 7. Kesimpulan terbatas','',
        f"1. **Kesetaraan preprocessing:** rentang MAE ketiga pipeline pada "
        f"timestamp identik adalah {spread[LEGACY]:.6g} pada tahap5, "
        f"{spread[FIXED]:.6g} setelah ablasi dengan train tetap, dan "
        f"{spread[MAIN]:.6g} pada train diperluas. Baca selisih ini bersama "
        'RMSE, bias, cakupan dan bulan. Kedekatan skor bukan uji formal ekuivalensi.',
        f"- Pada train lama tetap, tiga metode tetap sangat dekat setelah Curah Hujan "
        f"dikeluarkan. Pada train diperluas, kesetaraan deskriptif itu **tidak sepenuhnya "
        f"bertahan**: RMSE P0 {new.loc['P0_median','RMSE']:.6f} dibanding "
        f"P1 {new.loc['P1_ffill_L4','RMSE']:.6f} dan P2 {new.loc['P2_knn_k10','RMSE']:.6f}. "
        'P1 dan P2 masih dekat; perbedaan MAE juga perlu dibaca bersama RMSE/bias dan cakupan.']
    for method in f.methods:
        lines.append(f"2. **{method}:** MAE lama/trainlama tanpa hujan/traindiperluas = "
            f"{old.loc[method,'MAE']:.6f}/{fixed.loc[method,'MAE']:.6f}/{new.loc[method,'MAE']:.6f}; "
            f"RMSE = {old.loc[method,'RMSE']:.6f}/{fixed.loc[method,'RMSE']:.6f}/{new.loc[method,'RMSE']:.6f}. "
            f"Penambahan train mengubah MAE sebesar {new.loc[method,'MAE']-fixed.loc[method,'MAE']:+.6f} "
            'pada validasi berkonsentrasi rendah, bukan pada target tinggi.')
    lines += ['3. **Periode tinggi dalam train:** pengamatan tinggi kembali menjadi '
        'contoh pelatihan, tetapi target validasi masih1,09–2,06µg/m³. '
        'Hasil ini menilai generalisasi dari komposisi train baru ke validasi '
        'rendah, bukan kemampuan memprediksi konsentrasi tinggi. Pada run ini '
        'MAE dan RMSE seluruh pipeline meningkat ketika train tanpa hujan '
        'diperluas, sedangkan perubahan pada train tetap sangat kecil. Jadi '
        'perubahan besar terutama terjadi pada perbandingan komposisi train '
        'di dalam rancangan ini, bukan bukti bahwa Curah Hujan sendiri '
        'menjadi penyebab murni. Perluasan cakupan train tidak menjamin skor '
        'validasi rendah membaik.',
        '4. **Kandidat pengeluaran Curah Hujan:** layak dipertahankan sebagai '
        'kandidat karena mengurangi hambatan kelengkapan dan memperluas cakupan '
        'periode train. Kelayakan kualitas ini terpisah dari apakah skor rendah '
        'di validasi membaik. Sensitivitas train tetap dan hasil perluasan '
        'harus dibaca bersama; jangan menyimpulkan manfaat universal atau '
        'memilih preprocessing final.',
        '5. **Batas interpretasi:** satu split, Ridge tunggal alpha tetap, '
        'rentang validasi sempit, latensi yang diasumsikan dan validitas sensor '
        'yang belum terkonfirmasi membatasi kesimpulan. P3 tetap ditunda. '
        'Test akhir tetap dikunci dan tidak digunakan untuk memilih fitur/parameter.','',
        '## 8. Artefak, reproduksibilitas dan integritas','',
        '- Notebook: `notebooks/05B_ablasi_curah_hujan.ipynb`.',
        '- Workbook: `hasil_5B_ablasi_curah_hujan.xlsx`.',
        '- Folder `ablasi_curah_hujan_5B/`: konfigurasi terkunci, gate sebelum '
        'Ridge, daftar train dua lengan, timestamp evaluasi, prediksi kedua '
        'lengan, histori kausal dan fitur/indikator tanpa hujan.',
        '- Seluruh artefak lama yang dibaca/hash tidak diubah. File sumber '
        'juga diverifikasi SHA256 sebelum/sesudah. Log/cache inspect internal '
        'tidak diperlakukan sebagai hasil eksperimen.','']
    lines += [f"- `{r['File']}`: SHA256 sebelum=sesudah=`{r['SHA256 sebelum']}`." for r in f.integrity]
    (OUT/'ringkasan_5B_ablasi_curah_hujan.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
