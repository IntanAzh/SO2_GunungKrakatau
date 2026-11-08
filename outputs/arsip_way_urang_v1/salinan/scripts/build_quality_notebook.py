"""Membuat notebook audit tahap 2 dengan kode analisis di scripts/audit_quality.py."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s})
def code(s): cells.append({'cell_type':'code','metadata':{},'source':s,'execution_count':None,'outputs':[]})
md('''# 02 — Audit kualitas data SO₂ per stasiun

Acuan: hasil eksekusi `01_audit_data.ipynb` dan `outputs/audit_struktur_data.json`.
Header baris 8, satuan baris 9, awal tabel baris 10 diperiksa ulang untuk tiap sheet.
Kode audit yang dapat ditinjau berada di `scripts/audit_quality.py`.

File sumber hanya dibaca. Grid 30 menit adalah turunan audit, bukan pengganti data mentah.
Tidak ada imputasi, penghapusan outlier, smoothing, scaling, pemilihan stasiun final,
pelatihan model, dugaan zona waktu/kode sensor/penyebab anomali, atau atribusi ke Anak Krakatau.
''')
code('''from pathlib import Path
import sys, os
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
assert (ROOT/'data_raw').is_dir()
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs'/'.mplconfig'))
sys.path.insert(0, str(ROOT/'scripts'))
import pandas as pd
from IPython.display import display, Markdown, Image
from audit_quality import run_audit, export_excel, DEFINITIONS
print('Python:',sys.version)
print('Interpreter:',sys.executable)
display(pd.DataFrame(DEFINITIONS,columns=['Istilah','Definisi']))''')
md('''## Menjalankan seluruh pemeriksaan

Timestamp dibaca dengan format eksplisit; nilai numerik Excel yang tidak bertipe tanggal
tidak ditebak sebagai timestamp. Semua baris sumber tetap dipertahankan pada analisis per baris.
Duplikat dianalisis sebelum grid dibuat dan tidak dihapus dari data sumber.
Kelengkapan grid menghitung slot dengan sedikitnya satu nilai nonkosong; konflik disajikan terpisah.
Nilai nonnumerik/nonfinite tidak diperlakukan sebagai pengukuran numerik pada statistik/grafik.
''')
code('''tables, results = run_audit()
display(pd.DataFrame(tables['Sumber_SHA256']))
display(pd.DataFrame(tables['Kendala']))''')
md('''## Hasil terpisah untuk setiap stasiun

Tabel rentang pemeriksaan berisi hitungan seluruh kejadian; contoh dibatasi 10 per aturan.
Tabel kandidat ekstrem menampilkan minimum dan maksimum, tanpa menyatakan batas sensor.
Semua gap, interval bermasalah, duplikat dan rentetan SO2 tersedia pada workbook keluaran.
''')
code('''for result in results:
    station = result['summary']['Stasiun']
    display(Markdown(f'## {station}'))
    for title in ['Ringkasan','Metadata','Kelengkapan_variabel','SO2_bulanan',
                  'Kelompok_gap','Statistik_nilai','Ringkas_konstan',
                  'Aturan_pemeriksaan','Contoh_pemeriksaan','Kandidat_ekstrem']:
        display(Markdown(f'### {title.replace("_"," ")}'))
        display(pd.DataFrame([r for r in tables.get(title,[]) if r.get('Stasiun')==station]))
    display(Markdown('### Gap SO2 — 10 terpanjang; daftar lengkap pada workbook'))
    display(pd.DataFrame(sorted(result['gaps'],key=lambda g:-g['Slot hilang'])[:10]))
    display(Markdown('### 10 rentetan SO2 konstan terpanjang'))
    display(pd.DataFrame(result['top_runs']))
    for title in ['Duplikat','Masalah_waktu','Nilai_nonnumerik']:
        subset=[r for r in tables.get(title,[]) if r.get('Stasiun')==station]
        display(Markdown(f'### {title}: {len(subset)} kejadian (maksimal 10 ditampilkan)'))
        display(pd.DataFrame(subset[:10]))
    display(Image(filename=str(result['figure'])))''')
md('''## Ekspor tabel audit

Workbook ditulis oleh runtime artifact-tool; analisis Python disampaikan sebagai JSON
perantara. Timestamp pada workbook berupa tanggal Excel tanpa asumsi zona waktu.
Tabel seluruh stasiun memiliki kolom `Stasiun` agar dapat difilter, tanpa penggabungan
seri antarstasiun. File CSV grid hanya untuk audit di `outputs/audit_grids/`.
''')
code('''export_excel()
print('Workbook:', ROOT/'outputs'/'audit_kualitas_data.xlsx')
print('Grafik:', ROOT/'outputs'/'figures')
print('Ringkasan:', ROOT/'outputs'/'ringkasan_audit_kualitas.md')''')
md('''## Temuan, keterbatasan, dan implikasi eksperimen

Persentase teramati belum menyatakan validitas. Kelompok panjang gap hanya deskriptif.
Keputusan preprocessing, interpretasi kode, dan validasi sensor memerlukan bukti tambahan.
''')
code("display(Markdown((ROOT/'outputs'/'ringkasan_audit_kualitas.md').read_text(encoding='utf-8').replace('(figures/', '(../outputs/figures/')))")
for i,c in enumerate(cells): c['id']=f'quality-{i:02d}'
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
    'kernelspec':{'display_name':'Python (.venv SO2)','language':'python','name':'so2-audit'},
    'language_info':{'name':'python'}}}
(ROOT/'notebooks'/'02_audit_kualitas_data.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
