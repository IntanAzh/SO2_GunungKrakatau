"""Verifikasi hasil penelusuran tanpa fit/transform/predict."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import ast,json,math
import numpy as np
import pandas as pd
import openpyxl
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OLD=OUT/'forecasting_way_urang'
t=json.loads((OUT/'penelusuran_tahap5_tables.json').read_text(encoding='utf-8'))
cfg=json.loads((OLD/'config_locked.json').read_text(encoding='utf-8'))
a=cfg['split'][0]['Jumlah slot'];b=a+cfg['split'][1]['Jumlah slot']
d=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=b,parse_dates=['Timestamp'])
y=d['SO2 numerik'].to_numpy(dtype=float);times=pd.DatetimeIndex(d.Timestamp);o=np.arange(a);target=y[o+1]
ledger=t['Penanda_semua_timestamp'];assert len(ledger)==a==21119
assert np.allclose([np.nan if r['SO2 target asli'] is None else r['SO2 target asli'] for r in ledger],target,equal_nan=True)
common=np.array([r['Aktual train bersama'] for r in ledger]);saved=pd.read_csv(OLD/'timestamp_train_bersama.csv')['Index origin'].to_numpy()
assert np.array_equal(np.flatnonzero(common),saved)
threshold=target[common].max();high=(target>threshold)&(o+1<a)
stages={}
for name in cfg['pipelines']:
    X=pd.read_csv(OLD/f'fitur_{name}.csv',nrows=a).iloc[:,1:].to_numpy(dtype=float)
    finite=np.isfinite(X).all(axis=1);obs=np.isfinite(target);after=(o>=4368);inside=o+1<a
    masks=[np.ones(a,bool),obs,obs&(o>=48),obs&(o>=4320),obs&after,obs&after,
        obs&after&finite,obs&after&finite&inside,obs&after&finite&inside,common]
    rows=[r for r in t['Ringkasan'] if r['Pipeline']==name]
    assert len(rows)==10
    for row,mask in zip(rows,masks):
        vals=target[mask];v=vals[np.isfinite(vals)]
        assert len(vals)==row['Jumlah sampel'] and len(v)==row['Target asli teramati']
        assert np.allclose([v.min(),np.median(v),v.max()],[row['SO2 target minimum'],row['SO2 target median'],row['SO2 target maksimum']])
        monthly=[r for r in t['Tahap_per_bulan'] if r['Pipeline']==name and r['Tahap']==row['Tahap']]
        assert sum(r['Jumlah sampel'] for r in monthly)==len(vals)
        assert sum(r['Target asli teramati'] for r in monthly)==len(v)
        assert row['Target di atas maksimum final']==int((mask&high).sum())
    eligible=masks[-2];lost=eligible&~common
    loss=next(r for r in t['Kehilangan_karena_irisan'] if r['Pipeline']==name)
    assert lost.sum()==loss['Jumlah sampel']
    if name=='P1_ffill_L4':assert np.isnan(X[(o>=4368)&high,17]).all()
assert int(high.sum())==5289 and int((high&(o<4368)).sum())==1448
assert sum(r['Jumlah observasi'] for r in t['Periode_SO2_lebih_tinggi'])==5289
assert not any(r['Penanda']=='SISA_NAN_PERLU_TELUSUR_LANJUT' for r in t['Alasan_overlap'])
assert t['Inventaris_train_asli'][0]['SO2 asli teramati']==19197
pred=pd.read_csv(OLD/'prediksi_validasi.csv',parse_dates=['Origin','Timestamp target'])
bad=pred[pred.P2_knn_k10.isna()]
assert len(bad)==len(t['KNN_tidak_ada_prediksi'])==2
for _,r in bad.iterrows():
    u=int(r['Index origin']);assert np.isfinite(y[u])
    count=np.isfinite([d[f'{c} numerik'].iloc[u-1] for c in cfg['covariates']]).sum()
    assert count<2
for r in t['Verifikasi_skor_timestamp']:
    keep=pred['Irisan semua metode'].to_numpy() if r['Domain']=='irisan semua metode' else np.isfinite(pred[r['Metode']]).to_numpy()
    digest=sha256('\n'.join(pred.loc[keep,'Timestamp target'].astype(str)).encode()).hexdigest()
    assert digest==r['SHA256 daftar timestamp target']
    e=pred.loc[keep,r['Metode']]-pred.loc[keep,'SO2 target asli']
    assert np.allclose([e.abs().mean(),np.sqrt(np.mean(e**2)),e.mean()],[r['MAE hitung ulang'],r['RMSE hitung ulang'],r['Bias hitung ulang']])
    assert np.allclose(pred.loc[keep,'SO2 target asli'],y[pred.loc[keep,'Index target'].to_numpy(dtype=int)])
for r in t['Integritas_SHA256']:
    assert sha256((ROOT/r['File']).read_bytes()).hexdigest()==r['SHA256 sebelum']==r['SHA256 sesudah']
for source in ['trace_stage5_samples.py','write_stage5_trace_report.py']:
    tree=ast.parse((ROOT/'scripts'/source).read_text(encoding='utf-8'))
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {'fit','fit_transform','transform','predict'} for node in ast.walk(tree))
w=openpyxl.load_workbook(OUT/'penelusuran_sampel_tahap5.xlsx',read_only=True,data_only=False)
for name,records in t.items():
    rows=list(w[name].iter_rows(values_only=True));assert len(rows)==len(records)+1,(name,len(rows),len(records))
    for actual,expected in zip(rows[1:],records):
        for key,val in zip(rows[0],actual):
            exp=expected.get(key)
            if exp=='' and val is None:continue
            if isinstance(val,datetime):assert val.isoformat(sep=' ')[:19]==str(exp).replace('T',' ')[:19]
            elif isinstance(val,(int,float)) and not isinstance(val,bool) and isinstance(exp,(int,float)):
                assert math.isclose(val,exp,rel_tol=1e-11,abs_tol=1e-10)
            else:assert val==exp,(name,key,val,exp)
w.close()
print(f'Lolos: {len(t)} sheet, 21.119 penanda origin, waterfall/bulan, 5.289 target tinggi, 2 penolakan KNN, timestamp skor sama, hash artefak lama identik; tanpa pemanggilan model.')
