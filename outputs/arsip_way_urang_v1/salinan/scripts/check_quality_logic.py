"""Uji kasus audit: celah, duplikat, waktu gagal/off-grid, dan rentetan konstan."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'outputs'/'.mplconfig'))
import pandas as pd
import openpyxl
import audit_quality as aq

headers=['Waktu','PM10','PM2.5','SO2','CO','O3','NO2','Kec.Angin','Arah Angin',
         'Kelembaban','Suhu','Tek.Udara','Sol.Rad','Curah Hujan']
wb=openpyxl.Workbook(); ws=wb.active; ws.title='Fixture'
ws.cell(4,5,': 01 Januari 2025 - 01 Januari 2025')
for i,c in enumerate(headers,2): ws.cell(8,i,c)
for i in range(3,16): ws.cell(9,i,'µg/m³' if i<=8 else 'unit fixture')
items=[('2025-01-01 00:00:00',None),('2025-01-01 00:30:00',5),
       ('2025-01-01 01:00:00',5),('2025-01-01 02:00:00',5),
       ('2025-01-01 02:30:00',5),('2025-01-01 02:30:00',5),
       ('2025-01-01 03:00:00',6),('2025-01-01 03:00:00',7),
       ('2025-01-01 03:30:00','kode-x'),('gagal',-1),
       ('2025-01-01 03:45:00',9),('2025-01-01 04:00:00',None)]
for r,(t,v) in enumerate(items,10): ws.cell(r,2,t); ws.cell(r,5,v)
with TemporaryDirectory(prefix='so2_audit_check_') as d:
    aq.OUT=Path(d); aq.FIG=Path(d)/'figures'; aq.FIG.mkdir()
    tables={'Ringkasan':[]}
    result=aq.audit_station(Path('fixture.xlsx'),ws,{'Header':8,'Satuan':9,'Awal timestamp':10},tables,1)
    s=result['summary']
    assert s['Timestamp gagal']==1
    assert s['Di luar pola 00 atau 30']==1
    assert s['Grup duplikat identik']==1 and s['Grup duplikat konflik']==1
    assert s['Denominator slot grid']==9
    assert s['Tanpa baris slot']==1 and s['Baris ada SO2 kosong slot']==2
    assert s['SO2 teramati slot']==6
    assert s['Slot SO2 ambigu']==1
    assert len(result['gaps'])==3
    assert all(g['Slot hilang']==1 and g['Durasi menit']==30 for g in result['gaps'])
    assert {g['Posisi'] for g in result['gaps']}=={'sebelum pengamatan pertama','tengah seri','setelah pengamatan terakhir'}
    assert [r['Jumlah slot'] for r in result['top_runs']]==[2,2]
    assert tables['Nilai_nonnumerik'][0]['Nilai asli']=='kode-x'
    print('Lolos: denominator grid, gap satu slot, posisi gap, duplikat identik/konflik, nonnumerik, waktu gagal/off-grid, dan rentetan terputus.')
