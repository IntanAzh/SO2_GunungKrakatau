"""Membuat notebook pilot tahap 4; tidak menjalankan forecasting."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s})
def code(s):cells.append({'cell_type':'code','metadata':{},'source':s,'execution_count':None,'outputs':[]})
md('''# 04 — Pilot imputasi SO₂ Way Urang

Pilot sementara, bukan pemilihan stasiun final. Acuan: rancangan tahap ketiga dan hasil audit.
Eksperimen hanya menilai rekonstruksi SO₂ yang sengaja disembunyikan pada validasi.
Split kronologis 70/15/15 berdasarkan slot waktu; test hanya diinventarisasi, tidak dievaluasi.
Data test telah tercakup audit sebelumnya. Tidak ada forecasting dilatih.

Implementasi dapat ditinjau di `scripts/pilot_way_urang.py`. Sumber dan seluruh penanda
kualitas tetap dipertahankan; off-grid dipisahkan tanpa pembulatan.
''')
code('''from pathlib import Path
import sys, os
ROOT=Path.cwd().resolve()
if ROOT.name=='notebooks':ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.mplconfig'))
from IPython.display import display,Markdown,Image
import pandas as pd
from pilot_way_urang import Pilot
print('Interpreter:',sys.executable)
pilot=Pilot().prepare()
display(pd.DataFrame(pilot.split))
display(pd.DataFrame(pilot.quality))
display(pilot.offgrid)
display(pd.DataFrame(pilot.feature_info))''')
md('''## Masking yang dapat direproduksi

Seed 11, 23, 47. Titik terpisah dan blok dipilih dari nilai SO₂ asli teramati validasi.
Panjang blok berasal dari distribusi gap train, bukan test. Lokasi sama untuk seluruh metode.
Truth disimpan hanya di ledger penilaian, tidak dimasukkan kembali sebagai fitur/donor.
Kovariat P2/P3 dilag satu slot. Ketersediaan aktual masih bersyarat latensi <=30 menit.
''')
code('''pilot.make_masks()
display(pd.DataFrame(pilot.train_gaps))
display(pd.DataFrame(pilot.events).groupby(['Seed','Pola','Panjang slot']).agg(Jumlah_kejadian=('Kejadian','count'),Jumlah_mask=('Jumlah slot','sum')).reset_index())
display(pd.DataFrame(pilot.mask_ledger).head(10))
display(pd.DataFrame([{'Parameter':k,'Nilai':str(v)} for k,v in pilot.config.items()]))''')
md('''## Fit pada train saja

P0 median; P1 donor asli sebelumnya dengan batas umur; P2 KNN k=3/5/10;
P3 IterativeImputer–BayesianRidge satu keluaran deterministik.
P2/P3 memakai kovariat lag yang sama dan scaler yang dipelajari dari train.
Waktu mentah dan arah angin linear tidak menjadi fitur. Query SO₂ P2/P3 selalu NaN.
''')
code('''pilot.fit()
display(pd.DataFrame(pilot.params))
display(pd.DataFrame(pilot.scaling))
display(pd.DataFrame([{'Metode':k,'Fit detik':v} for k,v in pilot.fit_times.items()]))''')
md('''## Evaluasi validasi

Bias = prediksi − pengamatan asli; MAE/RMSE dalam µg/m³.
Cakupan = prediksi finite / jumlah nilai sengaja disembunyikan × 100%.
Nilai hilang alami tidak menjadi truth. Semua metode memakai ledger mask yang identik.
Irisan seluruh konfigurasi dapat sangat kecil pada blok panjang karena P1 L=1.
Transform setiap skenario dijalankan dan diukur terpisah; tidak memakai cache prediksi.
''')
code('''pilot.evaluate()
display(pd.DataFrame(pilot.metrics))
display(pd.DataFrame(pilot.summary))
display(pd.DataFrame(pilot.intersections))
display(pd.DataFrame(pilot.warn))
display(pd.DataFrame(pilot.failures))
for name in ['error_dan_cakupan_blok.png','error_irisan_semua_metode.png']:
    display(Image(filename=str(ROOT/'outputs/figures/pilot_way_urang'/name)))
display(pd.DataFrame(pilot.integrity))''')
md('''## Ekspor dan kesimpulan terbatas

Workbook memuat skor per seed/pola/panjang/domain, ringkasan variasi, cakupan,
lokasi mask, konfigurasi parameter train, penanda kualitas dan hash sumber.
Prediksi lengkap dan truth terpisah tersedia pada `outputs/pilot_way_urang/`.
Keunggulan rekonstruksi pada pilot ini belum menjadi keunggulan forecasting.
''')
code('''pilot.export_excel()
report=(ROOT/'outputs/ringkasan_pilot_imputasi_way_urang.md').read_text(encoding='utf-8')
display(Markdown(report.replace('(figures/', '(../outputs/figures/')))
print('Selesai. Test akhir tidak dievaluasi. Tidak ada model forecasting dilatih.')''')
for i,c in enumerate(cells):c['id']=f'pilot-{i:02d}'
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
    'kernelspec':{'display_name':'Python (.venv SO2)','language':'python','name':'so2-audit'},
    'language_info':{'name':'python'}}}
(ROOT/'notebooks/04_pilot_imputasi_way_urang.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
