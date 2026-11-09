"""Refresh report/notebook display from saved results, without fitting or scoring test."""
import json
from types import SimpleNamespace as NS
from pathlib import Path
from write_interstation_stage9 import report
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'kontribusi_antarstasiun'
t=json.loads((OUT/'tabel_kontribusi_antarstasiun.json').read_text(encoding='utf-8'))
cfg=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8')); ck=json.loads((RUN/'checks.json').read_text())
e=NS(cfg=cfg,config_hash=ck['config_sha256'],folds=[])
for attr,key in {'metrics':'Skor_per_fold','pooled':'Ringkasan','variation':'Variasi_antar_fold','deltas':'Vs_S0_persistence','source_checks':'Sumber_penyelarasan','offgrid':'Offgrid_development','quality':'Kualitas_SO2_dev','s0_audit':'Audit_S0_C30','age_stats':'Usia_donor','constant':'Rentetan_konstan','constant_summary':'Ringkas_konstan','integrity':'Integritas_SHA256'}.items():setattr(e,attr,t[key])
for g in t['Gate_pra_fit']:
    label=g['Fold'];f=NS(label=label,gate=g)
    for attr,key in {'flow':'Alur_sampel','dist':'Target_distribusi','lost':'Keluar_irisan','timing':'Waktu_Ridge'}.items():setattr(f,attr,[x for x in t[key] if x['Fold']==label])
    f.preprocess_seconds=next(x['Detik seluruh skenario'] for x in t['Waktu_preprocessing'] if x['Fold']==label);e.folds.append(f)
report(e)
nbp=ROOT/'notebooks/09_kontribusi_antarstasiun.ipynb'; nb=json.loads(nbp.read_text(encoding='utf-8'))
for cell in nb['cells']:
    for o in cell.get('outputs',[]):
        if 'data' in o and 'text/markdown' in o['data']:o['data']['text/markdown']=(OUT/'ringkasan_kontribusi_antarstasiun.md').read_text(encoding='utf-8').splitlines(True)
nbp.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print('Report and notebook display refreshed from saved results; zero model fits.')
