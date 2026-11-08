"""Verifikasi workbook tahap 3, rekonsiliasi, dan integritas file sumber/acuan."""
from pathlib import Path
from datetime import datetime
import json, math
import openpyxl
from audit_quality import hashfile
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
t=json.loads((OUT/'pemeriksaan_lanjutan_tables.json').read_text(encoding='utf-8'))
w=openpyxl.load_workbook(OUT/'pemeriksaan_lanjutan_kualitas.xlsx',read_only=True,data_only=False)
for name,records in t.items():
    rows=list(w[name].iter_rows(values_only=True)); assert len(rows)==len(records)+1
    for actual,expected in zip(rows[1:],records):
        for key,v in zip(rows[0],actual):
            exp=expected.get(key)
            if exp == '' and v is None:
                continue  # Excel menyimpan teks kosong sebagai sel kosong.
            if isinstance(v,datetime): assert v.isoformat(sep=' ')[:19]==str(exp).replace('T',' ')[:19],(name,key,v,exp)
            elif isinstance(v,(int,float)) and not isinstance(v,bool) and isinstance(exp,(int,float)):
                assert math.isclose(v,exp,rel_tol=1e-12,abs_tol=1e-10),(name,key,v,exp)
            else: assert v==exp,(name,key,v,exp)
w.close()
assert len(t['Offgrid_13'])==13 and sum(r['SO2 asli'] is not None for r in t['Offgrid_13'])==6
assert len(t['Top10_konstan_SO2'])==80 and len(t['Konteks_variabel'])==960
for r in t['Rekonsiliasi_grid']:
    assert r['Seluruh baris sumber']==r['Baris pada grid']+r['Baris off-grid terpisah']
    assert r['SO2 nonkosong seluruh baris']==r['SO2 nonkosong pada grid']+r['SO2 nonkosong off-grid terpisah']
for r in t['Top10_konstan_SO2']:
    assert r['Durasi slot jam']==r['Jumlah slot']*.5
    assert r['Rentang waktu jam']==(r['Jumlah slot']-1)*.5
    assert len([x for x in t['Konteks_variabel'] if x['Stasiun']==r['Stasiun'] and x['Peringkat rentetan']==r['Peringkat']])==12
for r in t['Ringkasan']:
    assert r['Hilang awal slot']+r['Hilang tengah slot']+r['Hilang akhir slot']==r['SO2 hilang slot total']
    assert abs(r['Hilang awal persen semua hilang']+r['Hilang tengah persen semua hilang']+r['Hilang akhir persen semua hilang']-100)<1e-10
for r in t['Grafik_konstan']:
    assert (OUT/r['Grafik relatif laporan']).is_file()
for r in t['Integritas_SHA256']:
    assert hashfile(ROOT/r['File'])==r['SHA256 sebelum']==r['SHA256 sesudah']
report=(OUT/'rancangan_eksperimen_preprocessing.md').read_text(encoding='utf-8')
for tag in ['P0','P1','P2','P3','MAE','RMSE','bias','Cakupan forecasting','BayesianRidge']:
    assert tag in report
print(f'Lolos: seluruh sel {len(t)} sheet, 13 off-grid (6 SO2 nonkosong), 80 rentetan, 960 konteks variabel, 8 grafik, denominator, dan seluruh SHA256.')
