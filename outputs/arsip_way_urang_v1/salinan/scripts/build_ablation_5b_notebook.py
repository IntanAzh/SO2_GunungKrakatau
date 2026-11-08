"""Notebook ablasi 5B utama dan sensitivitas train tetap."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s})
def code(s):cells.append({'cell_type':'code','metadata':{},'source':s,'execution_count':None,'outputs':[]})
md('''# 05B — Ablasi penuh Curah Hujan

Konfigurasi tahap kelima tetap kecuali pengeluaran Curah Hujan dan seluruh turunannya.
Imputer/scaler di-fit ulang tanpa hujan. Test akhir tidak dimuat nilainya atau dievaluasi.
Lengan utama: train bersama baru. Sensitivitas: timestamp train bersama lama persis.
Tidak ada preprocessing final dipilih. Implementasi: `scripts/ablation_5b.py`.
''')
code('''from pathlib import Path
import sys,os
ROOT=Path.cwd().resolve()
if ROOT.name=='notebooks':ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.mplconfig'))
from IPython.display import display,Markdown,Image
import pandas as pd
from ablation_5b import Ablation
experiment=Ablation().prepare()
display(pd.DataFrame(experiment.invariants))
print('Kovariat tersisa:',experiment.cov)
print('Konfigurasi dikunci:',experiment.config_hash)''')
md('''## Gate sebelum melatih Ridge

Imputasi histori memakai prefix train yang sama, tanpa Curah Hujan pada matriks imputer.
Verifikasi jumlah/komposisi train bersama dan pengamatan >2,14 yang kembali masuk.
2,14 adalah alat penelusuran kohort lama, bukan ambang kualitas/bahaya.
Sel berikut wajib lolos sebelum enam Ridge (tiga per lengan) dilatih.
''')
code('''experiment.impute_and_gate()
display(pd.DataFrame(experiment.flow))
display(pd.DataFrame([experiment.gate]))
display(pd.DataFrame(experiment.train_distribution))
display(pd.DataFrame(experiment.monthly_added))
print('Ridge yang telah dilatih:',experiment.ridge_fit_count)''')
md('''## Utama dan sensitivitas

Semua konfigurasi Ridge identik dengan tahap kelima; scaler fitur di-fit pada kohort train
masing-masing. Hasil lama hanya dibaca. Perbandingan memakai irisan timestamp identik
untuk hasil lama dan dua lengan baru. Persistence memakai donor asli dan umur yang sama.
''')
code('''experiment.run_models()
display(pd.DataFrame(experiment.metrics))
display(pd.DataFrame(experiment.compare))
display(pd.DataFrame(experiment.compare_month))
display(pd.DataFrame(experiment.deltas))
display(pd.DataFrame(experiment.timing))
for name in ['perluasan_train.png','perbandingan_identik.png','bulanan_identik.png']:
    display(Image(filename=str(ROOT/'outputs/figures/ablasi_curah_hujan_5B'/name)))
display(pd.DataFrame(experiment.integrity))''')
md('''## Ekspor dan kesimpulan

Efek utama menggabungkan ablasi dan perubahan komposisi train. Sensitivitas train lama
menguji ablasi pada sampel tetap; perbandingan dua lengan tanpa hujan menilai penambahan
sampel. Validasi tetap berkonsentrasi rendah, sehingga kemampuan pada konsentrasi tinggi
tidak dapat disimpulkan hanya karena pengamatan tinggi kembali masuk train.
''')
code('''experiment.export_excel()
report=(ROOT/'outputs/ringkasan_5B_ablasi_curah_hujan.md').read_text(encoding='utf-8')
display(Markdown(report.replace('(figures/', '(../outputs/figures/')))
print('Selesai. Sumber dan hasil lama identik; test tidak dimuat/dievaluasi.')''')
for i,c in enumerate(cells):c['id']=f'ablation-5b-{i:02d}'
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{
    'display_name':'Python (.venv SO2)','language':'python','name':'so2-audit'},'language_info':{'name':'python'}}}
(ROOT/'notebooks/05B_ablasi_curah_hujan.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
