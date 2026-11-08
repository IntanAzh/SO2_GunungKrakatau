"""Laporan pilot dari hasil yang telah dihitung; tanpa klaim pemilihan forecasting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from write_preprocessing_design import table as base_table

OUT=Path(__file__).resolve().parents[1]/'outputs'

def table(columns,rows):
    formatted=[]
    for row in rows:
        item=dict(row)
        for key,value in item.items():
            if isinstance(value,(float,np.floating)) and np.isfinite(value):
                if any(term in key for term in ['MAE','RMSE','Bias','detik']):item[key]=f'{value:.6f}'
                elif 'persen' in key.lower():item[key]=f'{value:.3f}'
        formatted.append(item)
    return base_table(columns,formatted)

def write_report(p):
    metrics=pd.DataFrame(p.metrics);summary=pd.DataFrame(p.summary)
    masks=pd.DataFrame(p.mask_ledger);events=pd.DataFrame(p.events)
    mask_summary=[]
    for (sid,seed,pattern,length),g in masks.groupby(['Skenario','Seed','Pola','Panjang slot'],sort=False):
        mask_summary.append({'Skenario':sid,'Seed':seed,'Pola':pattern,'Panjang slot':length,'Durasi blok jam':length/2,
            'Jumlah kejadian':int(g['Kejadian'].nunique()),'Jumlah nilai disembunyikan':len(g),
            'Persen dari SO2 teramati validasi':100*len(g)/np.isfinite(p.y[p.a:]).sum()})
    p.tables['Ringkas_masking']=mask_summary
    # write_report dipanggil setelah payload pertama: perbarui payload tabel secara deterministik.
    from pilot_way_urang import dump
    dump(OUT/'pilot_way_urang_tables.json',p.tables)
    blocks=summary[(summary.Domain=='seluruh berhasil')&(summary.Pola=='blok')]
    points=summary[(summary.Domain=='seluruh berhasil')&(summary.Pola=='titik')]
    pred=pd.DataFrame(p.predictions)
    findings=[]
    for method in pred.Metode.unique():
        g=pred[pred.Metode==method]
        findings.append({'Metode':method,'Prediksi negatif':int(g['Prediksi negatif'].sum()),
            'Di luar rentang train':int(g['Di luar rentang train'].sum()),'Gagal atau ditolak':int((~g['Berhasil finite']).sum())})
    lines=['# Pilot imputasi SO₂ — Lampung Selatan Way Urang','',
        '**Pilot sementara; bukan pemilihan stasiun final dan bukan eksperimen forecasting.** '
        'P0–P3 dibandingkan pada validasi saja. Pengujian akhir tidak ditransformasi, tidak diberi skor, '
        'dan tidak digunakan untuk memilih parameter. Data tersebut telah tercakup audit sebelumnya, '
        'sehingga tidak disebut sama sekali belum pernah dilihat.','',
        '## 1. Data, kualitas dan keputusan yang belum ditetapkan','',
        '- Sumber dibaca tanpa perubahan. Salinan seluruh baris, grid audit dan penanda kualitas '
        'ada di `pilot_way_urang/`. Nol, lonjakan, rentetan konstan dan ekstrem tidak dihapus. '
        'Penanda audit tetap disimpan; tidak digunakan sebagai label sensor rusak.',
        '- Terdapat **satu baris off-grid: 2025-05-22 19:01:00, SO₂ kosong**. '
        'Baris dipertahankan pada `offgrid_asli.csv` dan tidak dibulatkan, diagregasi atau menjadi truth.',
        '- Makna kode/nilai 0.01, validitas sensor, perubahan pola, dan latensi belum dikonfirmasi. '
        'Pengamatan asli teramati dipakai sebagai referensi rekonstruksi, bukan sertifikasi nilai valid.',
        '- **Asumsi ketersediaan yang memengaruhi hasil:** kovariat bertimestamp paling lambat t−30 menit '
        'diasumsikan telah tiba ketika SO₂ pada t direkonstruksi. Timestamp asli tidak membuktikan latensi. '
        'Jika latensi aktual lebih panjang, fitur perlu digeser lagi dan hasil ini harus diuji ulang.',
        '- Skenario pilot menyembunyikan SO₂ saja dan mempertahankan kovariat lag yang asli. '
        'Ini rekonstruksi kondisional dengan kovariat tersedia, belum meniru seluruh sensor/baris hilang '
        'bersamaan. Tidak ada nilai gap alami dijadikan truth atau diisi sebagai produk akhir.','',
        '## 2. Pemisahan waktu 70/15/15','',
        'N slot grid reguler dibagi `floor(0.70*N)` untuk train, hingga `floor(0.85*N)` '
        'untuk validasi, dan sisanya test. Ini pembagian durasi slot waktu, bukan pengacakan '
        'jumlah pengamatan nonkosong. Selisih pembulatan hanya satuan slot.','',
        table(['Split','Awal','Akhir','Jumlah slot','Jumlah baris sumber','SO2 nonkosong asli','Status'],p.split),'',
        'Semua median, mean/skala, daftar fitur, donor dan model imputasi berasal dari train. '
        'Nilai train lebih awal daripada seluruh query validasi. Bagian test hanya digunakan '
        'untuk inventaris pada tabel di atas; tidak ada skor test atau transform test.','',
        '## 3. Distribusi gap train dan masking','',
        'Distribusi panjang gap dihitung ulang pada **train saja**, termasuk gap awal. '
        'Kategori blok bukan batas aman imputasi. Panjang blok yang diuji berasal dari gap '
        '4 slot dan kuantil empiris nearest `{0,0.5,0.9,1}` gap panjang >4 slot.','',
        table(['Jumlah slot','Durasi jam','Jumlah kejadian'],pd.DataFrame(p.train_gaps).groupby(['Jumlah slot','Durasi jam']).size().rename('Jumlah kejadian').reset_index().to_dict('records')),'',
        f"Panjang blok: **{p.block_lengths} slot**, atau **{[v/2 for v in p.block_lengths]} jam**. "
        'Tiga seed tetap: **11, 23, 47**. Titik terpisah memakai 10% SO₂ teramati validasi '
        '(dibulatkan ke bawah), tanpa pasangan mask bertetangga. Blok memakai '
        '`max(1,min(30,floor(0.10*N_teramati/L)))` kejadian, tersedia truth penuh dan '
        'buffer satu slot teramati pada kedua sisi, tanpa overlap dalam skenario. '
        'Untuk blok 981 slot, satu blok saja melebihi 10%; cakupannya dilaporkan apa adanya.','',
        table(['Seed','Pola','Panjang slot','Durasi blok jam','Jumlah kejadian','Jumlah nilai disembunyikan','Persen dari SO2 teramati validasi'],mask_summary),'',
        '- Lokasi mask sama untuk seluruh konfigurasi pada suatu skenario. Mask antar-seed atau '
        'antar-skenario boleh overlap; SD tiga seed bukan interval kepercayaan independen.',
        '- `truth_penilaian_saja.csv` menyimpan nilai asli terpisah. Query P2/P3 selalu SO₂ NaN; '
        'donor hanya train. State P1 menerima SO₂ validasi yang tidak dimask hanya setelah waktunya '
        'terlewati, dan tidak pernah memakai nilai mask atau nilai imputasi untuk mereset donor.',
        '- Blok besar diletakkan hanya pada segmen yang truth-nya lengkap. Keterwakilan terhadap '
        'missing alami panjang/serentak masih terbatas; buffer teramati juga merupakan seleksi '
        'skenario dan tidak membuktikan kondisi gap awal tanpa donor.','',
        '## 4. Implementasi metode','',
        f'- **P0:** median SO₂ asli train = **{p.median:.8g} µg/m³**, tetap untuk semua query.',
        f'- **P1:** umur donor maksimum **{p.limits} slot**. Donor adalah pengamatan asli terakhir '
        'sebelum slot rekonstruksi. Jika melampaui batas, hasil tetap tidak terisi; panjang gap '
        'akhir tidak dilihat untuk menentukan perlakuan slot awal.',
        '- **P2:** `KNNImputer`, k=3/5/10, `weights=distance`, `metric=nan_euclidean`. '
        'Donor memiliki SO₂ asli finite dan hanya berasal dari train. SO₂ query seluruhnya '
        'NaN sehingga jarak dibentuk hanya dari kovariat lag yang overlap, tanpa waktu mentah '
        'atau SO₂ validasi. Minimal dua kovariat asli harus tersedia pada query. '
        'Jarak bawaan memakai jumlah overlap yang bisa berbeda antar-pasangan. '
        'Query tanpa jarak terdefinisi ditolak, bukan diberi fallback mean tersembunyi.',
        '- **P3:** `IterativeImputer(estimator=BayesianRidge(), max_iter=20, tol=0.001, '
        'initial_strategy=median, imputation_order=ascending, random_state=42, '
        'sample_posterior=False, skip_complete=False)`. Fit hanya train. Kovariat kosong '
        'diperkirakan sementara dalam iterasi berdasarkan model train pada baris query itu, '
        'tanpa memakai baris waktu berikutnya. Satu keluaran ini adalah imputasi tunggal '
        'deterministik, bukan multiple imputation lengkap.',
        '- **Fitur P2 dan P3 sama:** '+', '.join(p.features)+'. Semuanya dilag satu slot; '
        'tidak ada SO₂ masa depan, backward fill, interpolasi dua arah, rolling terpusat '
        'atau fitur waktu numerik mentah. Arah Angin dikeluarkan dari keduanya agar tidak '
        'diperlakukan sebagai variabel linear. Pemilihan fitur memakai dukungan dan variasi '
        'train saja, tanpa seleksi berdasarkan korelasi/skor validasi.',
        '- **Scaling:** satu `StandardScaler` fit pada matriks train dengan NaN tetap NaN. '
        'Transform memakai mean/skala train, dan SO₂ hasil dikembalikan ke µg/m³. '
        'Standardisasi membuat variabel besar seperti CO/tekanan tidak otomatis mendominasi '
        'jarak KNN; ekstrem train tetap dapat memengaruhi skala. Tidak ada pembandingan '
        'scaler lain pada pilot ini, sehingga efek scaling belum diisolasi secara eksperimen.',
        '- Kolom kosong/konstan train tidak dipakai sebagai kovariat informatif. NaN kovariat '
        'tidak diisi dengan nol. P2 memakai overlap; P3 memakai model train. Keduanya ditolak '
        'bila kurang dari dua kovariat asli tersedia. Tidak ada clipping hasil.',
        'Implementasi mengacu dokumentasi resmi '
        '[KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html), '
        '[IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html), '
        'dan [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).','',
        table(['Fitur','Lag slot','Nonkosong train','Train teramati persen','Nilai unik train','Dipakai P2 dan P3'],p.feature_info),'',
        '## 5. Hasil validasi','',
        '`e = prediksi − pengamatan asli`. MAE=mean(|e|), RMSE=sqrt(mean(e²)), '
        'bias=mean(e), dalam µg/m³. Cakupan (%)=100×prediksi finite/jumlah nilai sengaja '
        'disembunyikan. N=0 menghasilkan skor kosong, bukan nol. Tidak ada missing alami '
        'atau nilai test masuk denominator skor.',
        'Workbook `Hasil_per_seed` memuat hasil setiap seed, pola, panjang blok dan domain. '
        'Tabel berikut berisi mean dan SD sampel antar-seed; bukan skor gabungan seluruh seed.','',
        '### Seluruh titik yang dapat diisi masing-masing metode','',
        table(['Metode','Pola','Panjang slot','MAE rata-rata','MAE SD antar-seed','RMSE rata-rata',
               'Bias rata-rata','Cakupan persen rata-rata','Cakupan persen SD antar-seed'],
              summary[summary.Domain=='seluruh berhasil'].to_dict('records')),'',
        'Skor tersebut memakai domain keberhasilan yang dapat berbeda. Cakupan tinggi atau '
        'MAE rendah pada sedikit titik tidak cukup untuk memberi peringkat tunggal.','',
        '### Irisan titik yang berhasil diisi SEMUA konfigurasi','',
        table(['Seed','Pola','Panjang slot','Jumlah mask','Irisan semua metode','Irisan persen'],p.intersections),'',
        table(['Metode','Pola','Panjang slot','Jumlah dinilai rata-rata','MAE rata-rata','MAE SD antar-seed','RMSE rata-rata','Bias rata-rata'],
              summary[summary.Domain=='irisan semua metode'].to_dict('records')),'',
        'Irisan mencakup P1 dengan L=1, sehingga pada blok panjang umumnya hanya slot '
        'awal setelah donor yang tersedia. Hasil irisan **tidak mewakili bagian dalam blok '
        'panjang**. Persentase irisan wajib dibaca bersama MAE; jangan menyimpulkan '
        'kemampuan semua metode pada seluruh blok dari irisan kecil ini.','',
        '![Error dan cakupan blok](figures/pilot_way_urang/error_dan_cakupan_blok.png)','',
        '![Error pada irisan](figures/pilot_way_urang/error_irisan_semua_metode.png)','',
        '### Waktu komputasi','',
        table(['Metode','Fit detik','Scaler fit detik bersama'],p.tables['Waktu_fit']),'',
        table(['Metode','Transform semua skenario detik','Jumlah query diproses','Timestamp query unik'],p.tables['Waktu_transform_total']),'',
        'Fit dilakukan satu kali dan dipakai ulang, bukan diulang untuk setiap seed. '
        'Waktu transform P2/P3 diukur dengan menjalankan masing-masing skenario secara '
        'terpisah, meskipun beberapa query antar-seed identik. Durasi batch aktual '
        'dijumlah per skenario pada `Hasil_per_seed`; scaling query adalah tahap bersama '
        'di luar timer model. P1 mencakup pembangunan state '
        'kausal seluruh histori train+validasi per skenario, sehingga biaya transform '
        'tidak sepenuhnya setara dengan query batch P2/P3. Hardware, versi, thread=1 dan '
        'parameter tersimpan pada config; waktu dapat berubah antar-eksekusi.','',
        '## 6. Hasil yang perlu diperiksa dan kegagalan','',
        table(['Metode','Prediksi negatif','Di luar rentang train','Gagal atau ditolak'],findings),'',
        'Hitungan di atas adalah sel-skenario-metode, sehingga timestamp yang sama dapat '
        'muncul pada beberapa seed. Di luar min/max train adalah kandidat pemeriksaan, '
        'bukan otomatis salah atau batas sensor. Tidak ada clipping; semua prediksi/status '
        'disimpan di `prediksi_penilaian.csv`. Peringatan fit/transform tercatat terpisah.','',
        table(list(p.tables['Peringatan'][0]),p.tables['Peringatan']),'',
        table(list(p.tables['Kegagalan'][0]),p.tables['Kegagalan']),'',
        '## 7. Kesimpulan terbatas dan kandidat forecasting','']
    point_common=summary[(summary.Domain=='irisan semua metode')&(summary.Pola=='titik')].set_index('Metode')
    lines += [f"- Pada irisan titik terpisah (rata-rata {point_common.loc['P0','Jumlah dinilai rata-rata']:.2f} dari 452 sel per seed), "
        f"P2 k=10 memiliki MAE {point_common.loc['P2_k10','MAE rata-rata']:.6f} µg/m³; "
        f"P0 {point_common.loc['P0','MAE rata-rata']:.6f}; "
        f"P1 L=1 {point_common.loc['P1_L1','MAE rata-rata']:.6f}. "
        'KNN k=10 menjanjikan untuk rekonstruksi kondisional ini, dengan perbedaan terhadap baseline yang perlu diuji lagi.',
        f"- Rentang SO₂ teramati validasi adalah {np.nanmin(p.y[p.a:]):g}–{np.nanmax(p.y[p.a:]):g} µg/m³; "
        f"train {p.train_min:g}–{p.train_max:g} µg/m³. Mask ini tidak menguji kemampuan pada rentang SO₂ tinggi seperti yang ada di train.",
        '- P3 belum layak dianggap kandidat siap pakai dalam konfigurasi ini: kriteria konvergensi '
        'tidak tercapai pada 20 iterasi dan terdapat prediksi negatif. Cakupan finite yang tinggi '
        'tidak menjadikan hasil tersebut dapat diterima. Perlu diagnosis fitur/regularisasi/konvergensi '
        'pada train dan validasi sebelum dipertimbangkan lagi; tidak ada clipping atau retuning test.']
    # Kesimpulan numerik domain spesifik, tanpa memilih pemenang global.
    for pattern,sub in [('titik',points),('blok',blocks)]:
        for length,g in sub.groupby('Panjang slot'):
            full=g[g['Cakupan persen rata-rata']>=99.999]
            if len(full):
                best=full.loc[full['MAE rata-rata'].idxmin()]
                lines.append(f"- Pada {pattern} {int(length)} slot, di antara konfigurasi yang mengisi seluruh mask, "
                    f"{best['Metode']} mencatat MAE rata-rata terendah {best['MAE rata-rata']:.6g} µg/m³ "
                    f"(SD {best['MAE SD antar-seed']:.6g}). Ini temuan pada domain/pola tersebut, bukan peringkat keseluruhan.")
    lines += ['- P1 tetap kandidat sederhana untuk gap pendek dengan umur donor yang sesuai; '
        'ketika umur melebihi L, ia sengaja tidak mengisi. P0 menjadi baseline tetap. '
        'P2/P3 perlu dibaca bersama dukungan kovariat, sebaran seed, kondisi ekstrem dan '
        'peringatan. Kandidat dengan hasil baik pada validasi dapat dibawa ke eksperimen '
        'forecasting yang memakai model, origin, target asli dan cakupan yang sama.',
        '- Skor imputasi belum menentukan metode forecasting terbaik. Kemampuan pada '
        'missing alami, hilang semua kovariat, gap awal tanpa donor, perubahan distribusi, '
        'dan latensi aktual belum dibuktikan. Konfigurasi menjanjikan perlu diuji ulang '
        'dalam skenario tersebut, lalu pada forecasting terkontrol; test tetap dikunci.',
        '- Penyedia data masih perlu mengonfirmasi latensi, makna timestamp dan nilai '
        'khusus, validitas sensor serta penyebab pola konstan/lonjakan. Tidak ada penyebab '
        'yang disimpulkan dari pilot. Way Urang tetap pilot sementara.','',
        '## 8. Reproduksibilitas dan integritas','',
        '- Notebook: `notebooks/04_pilot_imputasi_way_urang.ipynb`.',
        '- Konfigurasi: `pilot_way_urang/config.json`; parameter train: `parameter_train.json`.',
        '- Lokasi mask: `lokasi_masking.csv`; kejadian: `kejadian_masking.csv`; truth hanya '
        'untuk penilaian: `truth_penilaian_saja.csv`; prediksi: `prediksi_penilaian.csv`.',
        '- Audit sumber, semua penanda, ledger off-grid dan pembatasan test tersimpan '
        'terpisah; data sumber dan hasil audit sebelumnya tidak diubah.','']
    lines += [f"- `{r['File']}`: SHA256 sebelum = sesudah = `{r['SHA256 sebelum']}`." for r in p.integrity]
    (OUT/'ringkasan_pilot_imputasi_way_urang.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
