"""Notebook penyaringan preprocessing dengan Ridge satu langkah."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s})
def code(s):cells.append({'cell_type':'code','metadata':{},'source':s,'execution_count':None,'outputs':[]})
md('''# 05 — Forecasting acuan Way Urang

Penyaringan awal preprocessing memakai Ridge satu langkah t+30 menit, bukan keputusan final
untuk deep learning. P0 median, P1 ffill L4 dan P2 KNN k10; P3 ditunda.
Split 70/15/15 dipertahankan. Test tidak dimuat nilainya atau dievaluasi.
Seluruh algoritma dijalankan dari `scripts/forecast_ridge_way_urang.py`.

Histori train diimputasi dari prefix masa lalu, lalu fitur lag dibentuk.
Seluruh Ridge memakai timestamp pelatihan bersama dan fitur/model yang sama.
''')
code('''from pathlib import Path
import sys,os
ROOT=Path.cwd().resolve()
if ROOT.name=='notebooks':ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'scripts'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.mplconfig'))
from IPython.display import display,Markdown,Image
import pandas as pd
from forecast_ridge_way_urang import Forecast
experiment=Forecast().prepare()
display(pd.DataFrame(experiment.pilot_seed))
display(pd.DataFrame(experiment.pilot_summary))
display(pd.DataFrame(experiment.distribution))
display(pd.DataFrame(experiment.monthly))
display(pd.DataFrame(experiment.locked_protocol))
print('Konfigurasi sudah dikunci sebelum fit:',experiment.config_hash)''')
md('''## Histori kausal sebelum lag

Kalibrasi awal 90 hari, pembaruan prefix setiap 30 hari di dalam train, lalu imputer
train dibekukan pada validasi. Histori lama tidak diperbaiki ulang.
Lag SO₂: 0,1,2,6,12,24,48. Kovariat pilot menggunakan satu slot keterlambatan.
Nilai SO₂ pada origin diasumsikan tersedia; latensi aktual masih perlu konfirmasi.
P1 memakai donor asli sebelumnya dan tidak diberi fallback median.
''')
code('''experiment.impute_history()
display(pd.DataFrame(experiment.prefix_logs))
display(pd.DataFrame(experiment.input_counts))
display(pd.DataFrame({'Fitur':experiment.feature_names}))''')
md('''## Ridge dan persistence pada target asli validasi

Ridge alpha=1, intercept=True, solver=svd untuk tiga pipeline; tanpa tuning.
Origin train yang targetnya masuk validasi dipurge. Timestamp train sama antar Ridge.
Persistence menggunakan SO₂ asli terakhir yang sudah tersedia pada origin.
Cakupan memakai seluruh target validasi asli teramati sebagai denominator.
Skor irisan dinilai pada timestamp yang berhasil untuk seluruh pipeline dan persistence.
''')
code('''experiment.fit_predict()
display(pd.DataFrame(experiment.train_detail))
display(pd.DataFrame(experiment.metrics))
display(pd.DataFrame(experiment.month_metrics))
display(pd.DataFrame(experiment.strata))
display(pd.DataFrame(experiment.timing))
for name in ['distribusi_train_validasi.png','performa_bulanan.png','prediksi_validasi.png']:
    display(Image(filename=str(ROOT/'outputs/figures/forecasting_way_urang'/name)))
display(pd.DataFrame(experiment.integrity))''')
md('''## Ekspor dan kesimpulan terbatas

Prediksi, timestamp evaluasi, histori imputasi dan indikator disimpan terpisah dari sumber.
Perbandingan ini menyaring kandidat dengan Ridge; tidak menentukan preprocessing terbaik
untuk deep learning dan tidak menambahkan algoritma forecasting lain.
''')
code('''experiment.export_excel()
report=(ROOT/'outputs/ringkasan_forecasting_acuan_way_urang.md').read_text(encoding='utf-8')
display(Markdown(report.replace('(figures/', '(../outputs/figures/')))
print('Selesai. Tidak ada nilai atau skor test akhir dimuat/dievaluasi.')''')
for i,c in enumerate(cells):c['id']=f'forecast-{i:02d}'
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{
    'kernelspec':{'display_name':'Python (.venv SO2)','language':'python','name':'so2-audit'},
    'language_info':{'name':'python'}}}
(ROOT/'notebooks/05_forecasting_acuan_way_urang.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
