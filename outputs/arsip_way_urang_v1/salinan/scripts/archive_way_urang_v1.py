"""Byte-preserving scientific archive. Standard library only; never trains/evaluates."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from hashlib import sha256
import csv,json,os,shutil,zipfile,sys

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';ARCHIVE=OUT/'arsip_way_urang_v1'
ZIP=OUT/'arsip_way_urang_v1.zip';RECEIPT=OUT/'arsip_way_urang_v1.verifikasi.json'
CACHE_DIRS={'.uvcache','.mplconfig','.ipython','.jupyter'}
CONFIGS={
    '4_pilot_imputasi':'outputs/pilot_way_urang/config.json',
    '5_forecasting_Ridge':'outputs/forecasting_way_urang/config_locked.json',
    '5B_ablasi_hujan':'outputs/ablasi_curah_hujan_5B/config_locked.json',
    '6_backtesting':'outputs/backtesting_way_urang/fold_config_locked.json',
    '7_LSTM':'outputs/lstm_preprocessing/config_locked.json',
    '8_sensitivitas_warmup':'outputs/sensitivitas_kalibrasi/config_locked.json',
    '6_diagnosis':'outputs/diagnosis_fold1/aturan_diagnosis.json'}
STAGES=[
    ('1','Audit struktur Excel',['notebooks/01_audit_data.ipynb','outputs/audit_struktur_data.json']),
    ('2','Audit kualitas',['notebooks/02_audit_kualitas_data.ipynb','outputs/audit_kualitas_data.xlsx','outputs/ringkasan_audit_kualitas.md']),
    ('3','Pemeriksaan lanjutan dan rancangan',['outputs/pemeriksaan_lanjutan_kualitas.xlsx','outputs/rancangan_eksperimen_preprocessing.md']),
    ('4','Pilot imputasi',['notebooks/04_pilot_imputasi_way_urang.ipynb','outputs/pilot_imputasi_way_urang.xlsx','outputs/ringkasan_pilot_imputasi_way_urang.md']),
    ('5','Forecasting Ridge',['notebooks/05_forecasting_acuan_way_urang.ipynb','outputs/forecasting_acuan_way_urang.xlsx','outputs/ringkasan_forecasting_acuan_way_urang.md']),
    ('5-penelusuran','Penelusuran sampel',['outputs/penelusuran_sampel_tahap5.xlsx','outputs/penelusuran_sampel_tahap5.md']),
    ('5B','Ablasi Curah Hujan',['notebooks/05B_ablasi_curah_hujan.ipynb','outputs/hasil_5B_ablasi_curah_hujan.xlsx','outputs/ringkasan_5B_ablasi_curah_hujan.md']),
    ('6','Expanding-window backtesting',['notebooks/06_backtesting_preprocessing.ipynb','outputs/backtesting_preprocessing.xlsx','outputs/ringkasan_backtesting_preprocessing.md']),
    ('6-diagnosis','Diagnosis fold pertama',['outputs/diagnosis_fold1.xlsx','outputs/diagnosis_fold1.md']),
    ('7','LSTM32 dengan tiga seed',['notebooks/07_lstm_preprocessing.ipynb','outputs/hasil_lstm_preprocessing.xlsx','outputs/ringkasan_lstm_preprocessing.md']),
    ('8','Sensitivitas warmup median/Ridge',['notebooks/08_sensitivitas_kalibrasi_ridge.ipynb','outputs/hasil_sensitivitas_kalibrasi.xlsx','outputs/ringkasan_sensitivitas_kalibrasi.md'])]

def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def stage(path):
    s=path.as_posix().lower()
    if 'arsip_way_urang' in s or 'archive_way_urang' in s:return 'administrasi arsip'
    if 'ablas' in s or 'ablation' in s or '5b' in s:return '5B'
    if 'sensitivitas' in s or 'kalibrasi' in s or 'calibration' in s or '/08_' in s:return '8'
    if 'diagnos' in s:return '6-diagnosis'
    if 'lstm' in s or '/07_' in s:return '7'
    if 'backtest' in s or '/06_' in s:return '6'
    if 'penelusuran' in s or 'trace' in s:return '5-penelusuran'
    if 'forecast' in s or '/05_' in s:return '5'
    if 'pilot' in s or '/04_' in s:return '4'
    if 'tahap3' in s or 'stage3' in s or 'lanjutan' in s or 'preprocessing_design' in s or 'followup' in s or 'rancangan_' in s:return '3'
    if '/01_' in s and s.startswith('notebooks/') or 'struktur_data' in s or 'build_audit_notebook' in s:return '1'
    if 'audit' in s or 'quality' in s or s.startswith('outputs/figures/'):return '2/konteks bersama'
    return 'pendukung lintas tahap'

def role(path):
    s=path.as_posix().lower()
    if '/.audit_preview' in s:return 'pratinjau/verifikasi visual historis'
    if path.suffix=='.ipynb':return 'notebook historis'
    if path.suffix=='.pt':return 'checkpoint historis (salinan byte)'
    if path.suffix=='.npz':return 'input/fitur dan cohort tersimpan'
    if path.suffix in ['.png','.svg','.jpg']:return 'grafik'
    if 'history' in s or 'training' in s:return 'riwayat/ringkasan training historis'
    if 'config' in s or 'konfigurasi' in s or 'scaler' in s or 'parameter' in s:return 'konfigurasi/parameter'
    if 'requirements' in s or 'freeze' in s or 'pyvenv' in s or 'kernel.json' in s:return 'dependensi/lingkungan'
    if s.startswith('scripts/'):return 'skrip reproduksi/verifikasi; tidak dijalankan saat arsip'
    if 'prediksi' in s or 'timestamp' in s or 'masking' in s:return 'prediksi/timestamp/masking historis'
    if path.suffix=='.xlsx':return 'tabel hasil'
    if path.suffix=='.md':return 'laporan/rancangan'
    return 'data audit/tabel/checks pendukung'

def csv_write(path,rows,columns):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def inventory():
    selected=[];excluded=[]
    for base in ['notebooks','scripts']:
        selected.extend(p for p in (ROOT/base).iterdir() if p.is_file() and p.suffix in ['.ipynb','.py','.mjs'])
    for parent,dirs,files in os.walk(OUT,topdown=True,followlinks=False):
        folder=Path(parent)
        kept=[]
        for name in sorted(dirs):
            p=folder/name
            if name in CACHE_DIRS or name.startswith('arsip_way_urang_'):
                excluded.append({'Lokasi':str(p.relative_to(ROOT)).replace('\\','/'),'Tipe':'direktori','Ukuran byte':'','SHA256':'',
                    'Alasan':'Cache/runtime atau arsip lain; bukan hasil eksperimen yang diduplikasi'})
            else:kept.append(name)
        dirs[:]=kept
        for name in sorted(files):
            p=folder/name
            if name.startswith('arsip_way_urang_v1'):continue
            if name.endswith('.inspect.ndjson'):
                excluded.append({'Lokasi':p.relative_to(ROOT).as_posix(),'Tipe':'file','Ukuran byte':p.stat().st_size,'SHA256':digest(p),
                    'Alasan':'Log inspeksi internal exporter; tidak memuat hasil ilmiah tambahan. XLSX/JSON/PNG hasil utama tetap disalin.'})
            else:selected.append(p)
    selected.extend(p for p in ROOT.iterdir() if p.is_file() and (p.name in ['README.md','.gitignore'] or p.name.startswith('requirements') and p.suffix=='.txt'))
    for rel in ['.venv/pyvenv.cfg','.venv-lstm/pyvenv.cfg','.venv-lstm/share/jupyter/kernels/so2-lstm/kernel.json']:
        p=ROOT/rel
        if p.exists():selected.append(p)
    return sorted(set(selected),key=lambda p:p.relative_to(ROOT).as_posix()),excluded

def configuration_summary():
    items=[];loaded={}
    keys=['station','stasiun','status','seeds','seed','p3','p2_k','p1_limits_slots','covariates','common_covariates',
        'covariate_lag_slots','so2_lags','horizon_minutes','warmup_slots','prefix_update_slots','ffill_limit_slots','knn','ridge',
        'fold_cuts','folds','architecture','optimization','sequence_features','sequence_length','internal_split','imputer_fit_scope',
        'feature_scaling','target_scaling','availability','quality','device','threads','torch_threads','deterministic_algorithms',
        'environment','versions','python','calibration_days','buffer_slots','test_start','test_loaded','test_evaluated','deep_learning_trained']
    for label,rel in CONFIGS.items():
        p=ROOT/rel
        if not p.exists():items.append({'Tahap':label,'File':rel,'Tersedia':False});continue
        value=json.loads(p.read_text(encoding='utf-8'));loaded[label]=value
        items.append({'Tahap':label,'File sumber':rel,'Salinan':'salinan/'+rel,'SHA256':digest(p),'Tersedia':True,
            'Parameter tercatat':{k:value[k] for k in keys if k in value}})
    return {'Catatan':'Ekstraksi metadata konfigurasi lama; tidak mengestimasi parameter atau membuat hasil baru. File penuh disalin utuh.',
        'Pilihan operasional saat arsip':{'imputasi':'median','warmup_hari':30,'status':'sementara; bukan metode atau durasi terbukti paling akurat'},
        'Seed':{'pilot_masking':[11,23,47],'LSTM':[11,23,47],'Ridge':'Konfigurasi lama mencatat seed42; solverSVD deterministik, bukan pemilihan seed terbaik.',
            'IterativeImputer_pilot':'random_state42 sesuai config p3; bukan multiple imputation lengkap'},
        'Dependensi disalin':['salinan/requirements.txt','salinan/requirements-lstm-lock.txt','salinan/outputs/environment_freeze.txt'],
        'Lingkungan':'Python3.15.2 untuk audit/Ridge; environment LSTM terpisah Python3.12.14 dengan PyTorch2.15.0+cpu. Detail aktual perstage pada config. Environment biner/node_modules tidak disalin.',
        'Konfigurasi':items},loaded

def verify_archive(folder,zip_path):
    manifest_bytes=(folder/'manifest.json').read_bytes();manifest=json.loads(manifest_bytes);mh=sha256(manifest_bytes).hexdigest()
    control=f'{mh}  manifest.json\n'.encode('ascii');assert (folder/'manifest.sha256').read_bytes()==control
    entries=manifest['entries'];expected={r['archive_path'] for r in entries}|{'manifest.json','manifest.sha256'}
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()};assert actual==expected,(actual-expected,expected-actual)
    for r in entries:
        p=folder/r['archive_path'];assert p.stat().st_size==r['size_bytes'] and digest(p)==r['sha256'],r['archive_path']
    with zipfile.ZipFile(zip_path) as z:
        names=z.namelist();assert len(names)==len(set(names)) and all(not n.endswith('/') for n in names)
        prefix=folder.name+'/'
        assert set(names)=={prefix+s for s in expected}
        assert z.read(prefix+'manifest.json')==manifest_bytes and z.read(prefix+'manifest.sha256')==control
        for r in entries:
            name=prefix+r['archive_path'];info=z.getinfo(name);assert info.file_size==r['size_bytes']
            h=sha256()
            with z.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            assert h.hexdigest()==r['sha256'],name
        assert z.testzip() is None
    return {'manifest_sha256':mh,'manifest_entries':len(entries),'control_files_verified':2,'total_files_folder_and_zip':len(expected),
        'folder_exact_file_set':True,'zip_exact_file_set':True,'zip_payload_sizes_sha256_match':True,'zip_crc_pass':True}

def main():
    if any(p.exists() for p in [ARCHIVE,ZIP,RECEIPT]):raise FileExistsError('Tujuan v1 sudah ada; tidak ditimpa. Gunakan versi baru hanya bila diminta.')
    files,excluded=inventory();before={p.relative_to(ROOT).as_posix():{'size':p.stat().st_size,'sha256':digest(p)} for p in files}
    external=[{'project_path':p.relative_to(ROOT).as_posix(),'absolute_path':str(p.resolve()),'size_bytes':p.stat().st_size,'sha256':digest(p),
        'included_in_archive':False,'reason':'Excel sumber tidak digandakan sesuai instruksi; hanya referensi lokasi/hash.'} for p in sorted((ROOT/'data_raw').glob('*.xlsx'))]
    assert external,'Excel sumber tidak ditemukan; inventaris harus dilengkapi sebelum arsip.'
    configs,loaded=configuration_summary();stage_rows=[];missing=[]
    for number,label,required in STAGES:
        absent=[p for p in required if not (ROOT/p).is_file()]
        stage_rows.append({'Tahap':number,'Eksperimen':label,'Berkas utama':required,'Berkas utama tidak tersedia':absent,'Status':'berkas utama tersedia' if not absent else 'tidak lengkap'})
        missing.extend({'Tahap':number,'Berkas':p,'Catatan':'Tidak ditemukan; tidak dibuat ulang.'} for p in absent)
    lstm=loaded['7_LSTM'];models=[];histories=[]
    for fold in [f'Fold_{i:02d}' for i in range(1,len(lstm['fold_cuts']))]:
        for method in lstm['pipelines']:
            for seed in lstm['seeds']:
                for kind,name,collection in [('checkpoint',f'LSTM_{method}_seed{seed}.pt',models),('history',f'history_{method}_seed{seed}.json',histories)]:
                    rel=f'outputs/lstm_preprocessing/{fold}/{name}';collection.append({'File':rel,'Tersedia':(ROOT/rel).is_file()})
                    if not (ROOT/rel).is_file():missing.append({'Tahap':'7','Berkas':rel,'Catatan':f'{kind} tidak tersedia; tidak dibuat pengganti.'})
    print(f'Inventaris: {len(files)} file untuk disalin; {sum(p["Tersedia"] for p in models)}/36 checkpoint; {sum(p["Tersedia"] for p in histories)}/36 history.',flush=True)
    ARCHIVE.mkdir();records=[]
    for i,p in enumerate(files,1):
        rel=p.relative_to(ROOT).as_posix();dest=ARCHIVE/'salinan'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        assert digest(dest)==before[rel]['sha256'],rel
        records.append({'archive_path':'salinan/'+rel,'project_source_path':rel,'original_absolute_path':str(p.resolve()),
            'size_bytes':before[rel]['size'],'sha256':before[rel]['sha256'],'stage':stage(Path(rel)),'role':role(Path(rel)),
            'origin':'copy_existing','source_mtime_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()})
        if i%100==0:print('Disalin dan cocok hash:',i,flush=True)
    dump(ARCHIVE/'konfigurasi_dependensi_seed.json',configs)
    dump(ARCHIVE/'inventaris_tahap.json',{'Tahap':stage_rows,'Checkpoint LSTM':models,'Riwayat permodel':histories,'Berkas wajib hilang':missing})
    dump(ARCHIVE/'referensi_excel_sumber.json',external)
    csv_write(ARCHIVE/'berkas_tidak_disalin.csv',excluded,['Lokasi','Tipe','Ukuran byte','SHA256','Alasan'])
    discrepancies=json.loads((ROOT/'outputs/lstm_preprocessing/resume_existing_discrepancies.json').read_text(encoding='utf-8'))
    limits=['# Keterbatasan arsip Way Urang v1','',
        '1. **Median dan warmup 30 hari adalah pilihan operasional sementara**, bukan metode atau durasi yang terbukti paling akurat. Keputusan ini dicatat pada saat arsip atas instruksi pengguna; tidak mengubah protokol maupun hasil historis.',
        '2. Way Urang adalah pilot sementara, bukan stasiun final. Peringkat preprocessing berubah antar-periode dan seed. Development sudah digunakan berulang; hasil bukan holdout independen.',
        '3. **Test akhir belum digunakan untuk evaluasi forecasting/imputasi.** Batasnya 2026-06-19 06:30 menurut kalender grid sebelumnya, tanpa asumsi zona waktu baru. Audit awal sudah mencakup keseluruhan sumber; jangan menyebut test sama sekali belum pernah dilihat. Arsip menyalin berkas audit/grid historis sebagai byte, termasuk jika rentangnya meliputi test, tanpa memuat seri test untuk analisis/evaluasi.',
        '4. **Ada perubahan hash notebook lama sebelum resume LSTM.** Salinan saat ini tetap dipertahankan, bukan direkonstruksi menjadi versi lama. Penyebab perubahan tidak diketahui. Rincian hash historis:',
        '','```json',json.dumps(discrepancies,ensure_ascii=False,indent=2),'```','',
        '5. **Sebagian waktu komputasi tidak tersimpan:** preprocessing empat fold LSTM serta waktu fit checkpoint Fold_04/P0_median/seed11. Nilai yang hilang tetap kosong, tidak diisi nol atau estimasi. Dua puluh delapan checkpoint selesai dipertahankan saat resume; delapan kombinasi tersisa diselesaikan dengan konfigurasi sama. Kombinasi tanpa checkpoint lengkap dimulai kembali dari seed tetap karena state optimizer parsial tidak tersedia.',
        '6. Notebook tahap 03 tidak ditemukan. Hasil tahap 3 tersedia berupa pemeriksaan_lanjutan_kualitas.xlsx, rancangan_eksperimen_preprocessing.md dan skrip. Tidak dibuat notebook pengganti atau hasil baru.',
        '7. Snapshot notebook02 yang persis cocok hash awal LSTM tidak tersedia sebagai file terpisah yang diketahui. Arsip berisi notebook terkini serta catatan kedua hash; tidak mengarang salinan lama.',
        '8. Checkpoint LSTM yang tersedia adalah bobot terbaik, bukan state optimizer/per-epoch lengkap. Seluruh checkpoint/history yang benar-benar ada disalin; tidak ada checkpoint baru dihasilkan.',
        '9. Excel sumber sengaja tidak disalin; lokasi, ukuran dan SHA256 ada pada referensi_excel_sumber.json. Reproduksi dari sumber membutuhkan berkas eksternal tersebut. Audit awal lintas stasiun disalin utuh karena merupakan konteks bersama, bukan diklaim seluruhnya data Way Urang.',
        '10. Environment biner .venv/.venv-lstm, node_modules dan runtime aplikasi tidak digandakan. Requirements/lock/freeze serta metadata konfigurasi dan pyvenv disalin. Versi JS exporter tidak memiliki package-lock proyek; skrip dan hasil XLSX final tetap tersedia. Bukan jaminan instalasi ulang otomatis di semua sistem.',
        '11. Zona waktu pengukuran, arti kode sensor/kualitas, latensi pelaporan dan penyebab transisi SO2 belum dikonfirmasi penyedia. Tidak menyimpulkan aktivitas vulkanik atau kerusakan sensor. Nilai nol/tinggi/konstan dan prediksi negatif tidak dihapus dalam arsip.',
        '12. Cache runtime dan log internal *.inspect.ndjson tidak disalin; log tersebut diinventarisasi dengan ukuran/hash pada berkas_tidak_disalin.csv. Pratinjau QA dan grafik ilmiah yang ada tetap disalin.',
        '',f'Berkas utama yang benar-benar hilang dari daftar wajib: {len(missing)}. Checkpoint tersedia: {sum(x["Tersedia"] for x in models)}/36; history permodel tersedia: {sum(x["Tersedia"] for x in histories)}/36.','']
    (ARCHIVE/'catatan_keterbatasan.md').write_text('\n'.join(limits),encoding='utf-8')
    summary=['# Ringkasan eksperimen dan keputusan sementara','',
        'Arsip v1 menutup rangkaian eksperimen Way Urang tahap 1–8 sebelum eksperimen antarstasiun. Dokumen ini adalah metadata arsip baru yang merangkum berkas historis; bukan hasil eksperimen atau checkpoint baru.','',
        '**Keputusan operasional sementara: imputasi median dan warmup 30 hari. Pilihan ini bukan metode yang terbukti paling akurat.** Jangan mengubahnya menjadi klaim pemenang universal atau stasiun final.','',
        '| Tahap | Cakupan dan batas kesimpulan | Laporan utama |','|---|---|---|',
        '| 1–2 | Struktur Excel diperiksa sebelum header; kualitas waktu, missing, gap, nilai konstan dan metadata diaudit. Audit bersama semua stasiun disalin utuh. | [Audit kualitas](salinan/outputs/ringkasan_audit_kualitas.md) |',
        '| 3 | Off-grid dan rentetan konstan ditelusuri; rancangan P0–P3 serta evaluasi kausal dibuat. Tidak semua anomali adalah kesalahan sensor. | [Rancangan](salinan/outputs/rancangan_eksperimen_preprocessing.md) |',
        '| 4 | Pilot masking menggunakan seed11/23/47; median, ffill, KNN dan iterative diperiksa. Akurasi rekonstruksi bukan akurasi forecasting. P3 belum konvergen dan ditunda. | [Pilot](salinan/outputs/ringkasan_pilot_imputasi_way_urang.md) |',
        '| 5 dan penelusuran | Ridge satu langkah30menit dibandingkan pada timestamp sama. Irisan sampel/kovariat CurahHujan menyempitkan train; hasil lama dipertahankan. | [Forecasting](salinan/outputs/ringkasan_forecasting_acuan_way_urang.md), [penelusuran](salinan/outputs/penelusuran_sampel_tahap5.md) |',
        '| 5B | Ablasi tanpa CurahHujan memperluas train; lengan cohort tetap memisahkan sebagian efek fitur dari penambahan sampel. Belum memilih preprocessing final. | [Ablasi](salinan/outputs/ringkasan_5B_ablasi_curah_hujan.md) |',
        '| 6 | Empat fold kalender, expanding train awal120hari. Peringkat preprocessing dan keunggulan terhadap persistence tidak konsisten; fold1 gagal saat perubahan rezim. | [Backtesting](salinan/outputs/ringkasan_backtesting_preprocessing.md) |',
        '| Diagnosis6 | Tidak ditemukan kesalahan alignment/scaler/target pada pemeriksaan yang dijalankan. Cohort sempit, respons Ridge lemah dan ekstrapolasi kovariat mendukung keterbatasan model; penyebab fisik data belum diketahui. | [Diagnosis](salinan/outputs/diagnosis_fold1.md) |',
        '| 7 | Satu LSTM32, tiga seed,48langkah×22kanal, innerholdback15%,36model selesai. Fold1 tetap dekat tingkat tinggi setelah SO2 turun. Variasi seed dan cakupan menghalangi klaim pemenang tunggal. | [LSTM](salinan/outputs/ringkasan_lstm_preprocessing.md) |',
        '| 8 | Median/Ridge dengan warmup30/60/90hari. C60 error gabungan terendah, tetapi C90 unggul fold2 dan C30 fold3/4. Tidak ada durasi unggul atas persistence pada MAE danRMSE sekaligus dalam satu fold. Fitur evaluasi sebelum scaling identik; komposisi train ikut berubah. | [Sensitivitas](salinan/outputs/ringkasan_sensitivitas_kalibrasi.md) |','',
        '## Keputusan yang belum dapat dibuat','',
        '- Belum menetapkan stasiun penelitian final, preprocessing paling akurat, horizon final, atau algoritma forecasting terbaik.',
        '- Median/warmup30 dipakai sebagai pengaturan operasional sementara untuk kerja berikutnya, dengan baseline persistence, pemeriksaan cakupan dan validasi kronologis tetap diperlukan.',
        '- Hasil menunjukkan perlunya memeriksa keterwakilan periode/konsentrasi, ekstrapolasi, kehilangan kovariat dan sensitivitas seed; penambahan data tinggi tidak membuktikan kemampuan meramalkan konsentrasi tinggi.',
        '- Tidak memakai test akhir untuk memilih pipeline. Tidak menghubungkan lonjakan dengan gunung/sensor tanpa bukti.',
        '', 'Rincian asumsi, hash lama yang berubah, waktu hilang dan berkas yang tidak tersedia ada pada [catatan keterbatasan](catatan_keterbatasan.md).','']
    (ARCHIVE/'ringkasan_eksperimen.md').write_text('\n'.join(summary),encoding='utf-8')
    readme=['# Arsip Way Urang v1','',f'Dibuat UTC: {datetime.now(timezone.utc).isoformat()}',
        '', 'Snapshot byte berkas yang sudah tersedia, dengan tambahan dokumen inventaris/arsip yang ditandai jelas. Tidak ada pemindahan, penghapusan, training ulang, evaluasi test, atau penciptaan checkpoint/hasil eksperimen yang tidak ada.',
        '', '- [Ringkasan dan keputusan sementara](ringkasan_eksperimen.md)',
        '- [Keterbatasan dan berkas tidak tersedia](catatan_keterbatasan.md)',
        '- `inventaris_tahap.json`: kelengkapan per tahap dan daftar checkpoint/history.',
        '- `konfigurasi_dependensi_seed.json`: indeks konfigurasi, dependensi dan seed; config penuh berada di salinan/outputs/.',
        '- `referensi_excel_sumber.json`: lokasi dan hash Excel asli yang tidak digandakan.',
        '- `manifest.json`: daftar otoritatif lokasi, ukuran byte, SHA256, asal, peran dan tahap semua payload, termasuk manifest.csv.',
        '- `manifest.csv`: inventaris payload sebelum berkas kontrol manifest ditambahkan.',
        '- `manifest.sha256`: hash manifest.json. Dua berkas kontrol tidak meng-hash dirinya sendiri; keduanya diverifikasi melalui hash manifest dan kecocokan isi kontrol. Tidak ada siklus self-hash.',
        '- `berkas_tidak_disalin.csv`: pengecualian cache/log internal yang eksplisit.',
        '- `salinan/`: struktur relatif proyek dipertahankan; hasil/konfigurasi/notebook/skrip/checkpoint/grafik adalah salinan byte, bukan hasil ulang.',
        '', 'ZIP memiliki satu folder akar arsip_way_urang_v1. Tidak ada file tambahan di luar set manifest+dua kontrol; semua isi ZIP diperiksa ukuran/SHA256 atau isi kontrolnya, serta CRC. Bukti verifikasi dan SHA256 ZIP disimpan di luar ZIP sebagai `../arsip_way_urang_v1.verifikasi.json` agar tidak menimbulkan hash melingkar.',
        '', 'Environment biner tidak disertakan. Notebook/skrip adalah catatan historis, bukan perintah yang dijalankan oleh proses pengarsipan. Reproduksi memerlukan penyiapan dependensi dan sumber eksternal yang tercatat.','']
    (ARCHIVE/'README_ARSIP.md').write_text('\n'.join(readme),encoding='utf-8')
    for p in sorted(ARCHIVE.iterdir()):
        if p.is_file():records.append({'archive_path':p.relative_to(ARCHIVE).as_posix(),'project_source_path':None,'original_absolute_path':None,
            'size_bytes':p.stat().st_size,'sha256':digest(p),'stage':'administrasi arsip','role':'metadata arsip baru, bukan hasil eksperimen','origin':'archive_metadata','source_mtime_utc':None})
    records.sort(key=lambda r:r['archive_path'])
    columns=['archive_path','project_source_path','original_absolute_path','size_bytes','sha256','stage','role','origin','source_mtime_utc']
    csv_write(ARCHIVE/'manifest.csv',records,columns)
    p=ARCHIVE/'manifest.csv';records.append({'archive_path':'manifest.csv','project_source_path':None,'original_absolute_path':None,
        'size_bytes':p.stat().st_size,'sha256':digest(p),'stage':'administrasi arsip','role':'inventaris payload','origin':'archive_metadata','source_mtime_utc':None})
    manifest={'archive_version':'way_urang_v1','created_utc':datetime.now(timezone.utc).isoformat(),'project_root':str(ROOT),'entries':records,
        'external_source_references':external,'control_files':['manifest.json','manifest.sha256'],'payload_copied_count':len(files),
        'generated_metadata_count_in_entries':sum(r['origin']=='archive_metadata' for r in records),'source_excel_copied':False,
        'verification_rule':'Exact file set, all payload sizes and SHA256; manifest.json checked against manifest.sha256; control content and both control bytes checked in ZIP; CRC checked.'}
    dump(ARCHIVE/'manifest.json',manifest);mh=digest(ARCHIVE/'manifest.json');(ARCHIVE/'manifest.sha256').write_text(f'{mh}  manifest.json\n',encoding='ascii')
    print('Membuat ZIP; jumlah file folder:',len(records)+2,flush=True)
    with zipfile.ZipFile(ZIP,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,strict_timestamps=False) as z:
        for p in sorted(ARCHIVE.rglob('*')):
            if p.is_file():z.write(p,arcname=ARCHIVE.name+'/'+p.relative_to(ARCHIVE).as_posix())
    result=verify_archive(ARCHIVE,ZIP)
    # Recheck originals AFTER copying and compression, including referenced source and excluded internal logs.
    for p in files:
        rel=p.relative_to(ROOT).as_posix();assert p.stat().st_size==before[rel]['size'] and digest(p)==before[rel]['sha256'],rel
    for ref in external:
        p=ROOT/ref['project_path'];assert p.stat().st_size==ref['size_bytes'] and digest(p)==ref['sha256']
    for row in excluded:
        if row['Tipe']=='file':assert digest(ROOT/row['Lokasi'])==row['SHA256']
    result.update({'archive_path':str(ARCHIVE),'zip_path':str(ZIP),'zip_size_bytes':ZIP.stat().st_size,'zip_sha256':digest(ZIP),
        'copied_existing_files':len(files),'originals_unchanged':True,'external_sources_unchanged':True,'source_excel_copied':False,
        'checkpoint_files_existing_copied':sum(x['Tersedia'] for x in models),'training_histories_existing_copied':sum(x['Tersedia'] for x in histories),
        'stage_required_missing':missing,'notebook03_available':any(p.name.startswith('03_') and p.suffix=='.ipynb' for p in files),
        'new_model_training':False,'new_experiment_results':False,'test_evaluation':False,'verified_utc':datetime.now(timezone.utc).isoformat()})
    dump(RECEIPT,result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    if '--verify-only' in sys.argv:
        result=verify_archive(ARCHIVE,ZIP);receipt=json.loads(RECEIPT.read_text(encoding='utf-8'))
        assert digest(ZIP)==receipt['zip_sha256'] and ZIP.stat().st_size==receipt['zip_size_bytes']
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:main()
