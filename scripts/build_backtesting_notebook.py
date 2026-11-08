from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def add(kind,source):
    c={'cell_type':kind,'id':f'backtest-{len(cells):02d}','metadata':{},'source':source}
    if kind=='code':c.update(execution_count=None,outputs=[])
    cells.append(c)
add('markdown','# Tahap 6: expanding-window preprocessing Way Urang\n\nValidasi pengembangan pada train+validasi lama; **test akhir tidak dimuat atau dievaluasi**. Semua hasil sebelumnya dipertahankan. Empat fold kalender disimpan sebelum model. Konfigurasi tetap tahap 5B tanpa Curah Hujan.')
add('code',"from pathlib import Path\nimport sys\nROOT=Path.cwd()\nif ROOT.name=='notebooks': ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT/'scripts'))\nfrom backtesting_stage6 import Backtest\nfrom IPython.display import display, Markdown, Image\nimport pandas as pd\nb=Backtest().prepare()\ndisplay(pd.DataFrame(b.plan))\ndisplay(pd.DataFrame(b.distributions))\ndisplay(pd.DataFrame(b.invariants))\nprint('Konfigurasi pra-model SHA256:',b.config_hash)\n")
add('markdown','## Pemeriksaan kecukupan sebelum Ridge\n\nKalibrasi 90 hari, buffer 48 slot, minimal 340 target train bersama dan dua nilai berbeda. Seluruh fold diperiksa sebelum satu pun Ridge dilatih. Distribusi target dan kehilangan karena irisan diperiksa tanpa mengubah kalender berdasarkan skor.')
add('code',"b.preprocess_and_check_all()\ndisplay(pd.DataFrame([f.gate for f in b.folds]))\ndisplay(pd.DataFrame([r for f in b.folds for r in f.flow]))\ndisplay(pd.DataFrame([r for f in b.folds for r in f.lost]))\ndisplay(pd.DataFrame([f.train_distribution[0] for f in b.folds]))\nprint('Jumlah Ridge sebelum seluruh gate:',b.fit_count)\n")
add('markdown','## Fit dari nol per fold dan evaluasi OOF\n\nParameter imputasi hanya prefix train; validasi dibekukan. Fitur historis dibentuk kausal sebelum lag. Train bersama, target asli, purge pada batas. Persistence memakai pengamatan asli terakhir pada/sebelum origin. Semua metrik bersama memakai timestamp identik.')
add('code',"b.run_models()\ndisplay(pd.DataFrame(b.metrics))\ndisplay(pd.DataFrame(b.pooled))\ndisplay(pd.DataFrame(b.summary))\ndisplay(pd.DataFrame(b.ranks))\ndisplay(pd.DataFrame(b.deltas))\ndisplay(pd.DataFrame(b.preprocess_time))\ndisplay(pd.DataFrame(b.timing))\ndisplay(pd.DataFrame(b.integrity))\nfor name in ['metrik_dan_cakupan_fold.png','distribusi_dan_train_fold.png','mae_bulanan_oof.png']:\n    display(Image(filename=str(ROOT/'outputs/figures/backtesting_way_urang'/name)))\n")
add('code',"b.export_excel()\ndisplay(Markdown((ROOT/'outputs/ringkasan_backtesting_preprocessing.md').read_text(encoding='utf-8')))\n")
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python (.venv SO2 audit)','language':'python','name':'so2-audit'},'language_info':{'name':'python','version':sys.version.split()[0]}},'cells':cells}
(ROOT/'notebooks/06_backtesting_preprocessing.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
