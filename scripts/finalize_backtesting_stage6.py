"""Refresh presentation and explicit audit ledgers, without fitting models."""
from types import SimpleNamespace
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from backtesting_stage6 import ROOT,OUT,RUN,FIG,METHODS,ALL,plt
from write_backtesting_report import write_report
t=json.loads((OUT/'tabel_backtesting_preprocessing.json').read_text(encoding='utf-8'))
cfg=json.loads((RUN/'fold_config_locked.json').read_text(encoding='utf-8'))
checks=json.loads((RUN/'checks.json').read_text(encoding='utf-8'))
p=SimpleNamespace(cfg=cfg,config_hash=checks['config_sha256'],plan=t['Rencana_fold'],distributions=t['Distribusi_train_validasi'],
    metrics=t['Ringkasan_per_fold'],pooled=t['Ringkasan'],summary=t['Variasi_antar_fold'],ranks=t['Peringkat_deskriptif'],
    deltas=t['Vs_persistence_identik'],pooled_month=t['Skor_bulanan_gabungan'],timing=t['Waktu_Ridge'],
    preprocess_time=t['Waktu_preprocessing'],integrity=t['Integritas_SHA256'],
    pred=pd.read_csv(RUN/'prediksi_oof_development.csv'),folds=[])
for fold in ['Fold_01','Fold_02','Fold_03','Fold_04']:
    f=SimpleNamespace(gate=next(r for r in t['Gate_sebelum_Ridge'] if r['Fold']==fold),
        flow=[r for r in t['Alur_sampel'] if r['Fold']==fold],lost=[r for r in t['Keluar_irisan_train'] if r['Fold']==fold],
        train_distribution=[r for r in t['Distribusi_train_aktual'] if r['Fold']==fold])
    p.folds.append(f)
    a=next(r['Index awal validasi'] for r in p.plan if r['Fold']==fold)
    train=pd.read_csv(RUN/fold/'timestamp_train_bersama.csv');ids=train['Index origin'].to_numpy()
    n=next(r['Index akhir validasi eksklusif'] for r in p.plan if r['Fold']==fold)
    raw=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=n,usecols=['Timestamp','SO2 numerik'])
    y=raw['SO2 numerik'].to_numpy();cand=np.arange(cfg['warmup_slots']+max(cfg['so2_lags']),a-1)
    ledger=[]
    for method in METHODS:
        X=pd.read_csv(RUN/fold/f'fitur_{method}.csv').iloc[:,1:].to_numpy()
        own=cand[np.isfinite(y[cand+1])&np.isfinite(X[cand]).all(axis=1)]
        for i in np.setdiff1d(own,ids):
            ledger.append({'Pipeline':method,'Index origin':int(i),'Origin':raw.Timestamp.iloc[i],
                'Timestamp target':raw.Timestamp.iloc[i+1],'SO2 target asli':float(y[i+1]),'Alasan':'Layak mandiri tetapi keluar karena irisan train'})
    pd.DataFrame(ledger).to_csv(RUN/fold/'timestamp_keluar_karena_irisan.csv',index=False)
write_report(p)
# A second view keeps later low-concentration folds legible without hiding fold 1.
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for method in ALL:
    g=pd.DataFrame([r for r in p.metrics if r['Pipeline']==method and r['Domain']=='irisan semua metode' and r['Fold']!='Fold_01'])
    for ax,col in zip(axes,['MAE','RMSE']):ax.plot(g.Fold,g[col],marker='o',label=method);ax.set(title=col+' — fold 2–4',ylabel='µg/m³');ax.grid(alpha=.2)
axes[1].legend(fontsize=8);fig.savefig(FIG/'metrik_fold_2_4.png',dpi=150);plt.close(fig)
report=OUT/'ringkasan_backtesting_preprocessing.md'
report.write_text(report.read_text(encoding='utf-8')+'\n![Detail fold 2–4](figures/backtesting_way_urang/metrik_fold_2_4.png)\n',encoding='utf-8')
nbpath=ROOT/'notebooks/06_backtesting_preprocessing.ipynb';nb=json.loads(nbpath.read_text(encoding='utf-8'))
# Refresh only the displayed report; execution and saved model results remain intact.
last=[c for c in nb['cells'] if c['cell_type']=='code'][-1]
nb['metadata']['language_info']['version']=sys.version.split()[0]
for output in last['outputs']:
    if output.get('output_type')=='display_data' and 'text/markdown' in output.get('data',{}):
        output['data']['text/markdown']=report.read_text(encoding='utf-8')
nbpath.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
from PIL import Image,ImageOps,ImageDraw
files=[f for f in (OUT/'.audit_preview_backtesting').glob('*.png') if f.name!='contact_all.png']
width=660;height=250;canvas=Image.new('RGB',(width*3,height*((len(files)+2)//3)),'#dddddd');draw=ImageDraw.Draw(canvas)
for i,file in enumerate(files):
    img=Image.open(file).convert('RGB');img.thumbnail((width-10,height-30))
    x=(i%3)*width;y=(i//3)*height;canvas.paste(img,(x,y+25));draw.text((x+5,y+5),file.stem,fill='black')
canvas.save(OUT/'.audit_preview_backtesting/contact_all.png')
print('Laporan, grafik detail dan ledger irisan diperbarui tanpa fit/akses test.')
