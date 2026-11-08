"""Refresh report/visual QA from persisted results, without fitting."""
from types import SimpleNamespace
import json
from lstm_stage7 import ROOT,OUT,RUN,FIG,plt,METHODS
from write_lstm_stage7 import report
import pandas as pd
t=json.loads((OUT/'tabel_lstm_preprocessing.json').read_text(encoding='utf-8'))
cfg=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'))
checks=json.loads((RUN/'checks.json').read_text(encoding='utf-8'))
e=SimpleNamespace(cfg=cfg,config_hash=checks['config_sha256'],plan=t['Rencana_fold'],metrics=t['Metrik_fold_seed'],pooled=t['Ringkasan'],
    variation=t['Variasi_seed'],ranks=t['Peringkat'],deltas=t['Vs_persistence'],transition=t['Diagnosis_transisi'],
    ridge_comparison=t['Ridge_timestamp_identik'],training=t['Training_ringkas'],timeprep=t['Waktu_preprocessing'],
    integrity=t['Integritas_SHA256'],sensitivity=t['Sensitivitas_seed_metode'],calendar_distribution=t['Target_kalender'],accounting=t['Alur_rinci'],fold1_sequence_breaks=t['Fold1_sekuens_terputus'],target_quantiles=t['Target_kuantil'],folds=[])
for label in [f'Fold_{i:02d}' for i in range(1,5)]:
    e.folds.append(SimpleNamespace(gate=next(r for r in t['Gate_pra_training'] if r['Fold']==label),
        distributions=[r for r in t['Target_distribusi'] if r['Fold']==label],flow=[r for r in t['Alur_sampel'] if r['Fold']==label],
        losses=[r for r in t['Keluar_irisan'] if r['Fold']==label]))
report(e)
h=pd.DataFrame(t['Training_epoch']);fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
for ax,label in zip(axes,[f'Fold_{i:02d}' for i in range(1,5)]):
    for method in METHODS:
        for seed in cfg['seeds']:
            g=h[(h.Fold==label)&(h.Pipeline==method)&(h.Seed==seed)]
            ax.plot(g.Epoch,g['Internal MSE standar'],label=method+f' seed{seed}',alpha=.7,lw=1)
    ax.set(title=label,xlabel='Epoch',ylabel='MSE internal target standar',yscale='log');ax.grid(alpha=.2)
axes[-1].legend(fontsize=5);fig.savefig(FIG/'riwayat_internal.png',dpi=150);plt.close(fig)
path=OUT/'ringkasan_lstm_preprocessing.md';path.write_text(path.read_text(encoding='utf-8')+'\n![Riwayat validasi internal](figures/lstm_preprocessing/riwayat_internal.png)\n',encoding='utf-8')
nbpath=ROOT/'notebooks/07_lstm_preprocessing.ipynb';nb=json.loads(nbpath.read_text(encoding='utf-8'))
for output in [c for c in nb['cells'] if c['cell_type']=='code'][-1]['outputs']:
    if output.get('output_type')=='display_data' and 'text/markdown' in output.get('data',{}):output['data']['text/markdown']=path.read_text(encoding='utf-8')
nbpath.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
from PIL import Image,ImageDraw
files=[f for f in (OUT/'.audit_preview_lstm').glob('*.png') if f.name!='contact_all.png']
canvas=Image.new('RGB',(1980,250*((len(files)+2)//3)),'#dddddd');draw=ImageDraw.Draw(canvas)
for i,file in enumerate(files):
    im=Image.open(file).convert('RGB');im.thumbnail((650,220));x=(i%3)*660;y=(i//3)*250;canvas.paste(im,(x,y+25));draw.text((x+5,y+5),file.stem,fill='black')
canvas.save(OUT/'.audit_preview_lstm/contact_all.png')
print('Report dan QA diperbarui tanpa fit atau akses test.')
