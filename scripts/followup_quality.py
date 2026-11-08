"""Tahap 3: rekonsiliasi off-grid, konteks konstan, dan kelayakan deskriptif."""
from pathlib import Path
from collections import Counter
from datetime import datetime,date
import json
import os
import numpy as np
import pandas as pd
import openpyxl
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_quality import hashfile, empty, stamp

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
FIG=OUT/'figures'/'tahap3'; FIG.mkdir(exist_ok=True)
METEO=['Kec.Angin','Arah Angin','Kelembaban','Suhu','Tek.Udara','Sol.Rad','Curah Hujan']
T2=json.loads((OUT/'audit_kualitas_tables.json').read_text(encoding='utf-8'))
REF=json.loads((OUT/'audit_struktur_data.json').read_text(encoding='utf-8'))
tables={'Ringkasan':[], 'Definisi':[
 {'Istilah':'Status','Definisi':'Pemeriksaan deskriptif, belum menjalankan preprocessing atau pemilihan stasiun final.'},
 {'Istilah':'SO2 teramati grid persen','Definisi':'100 × slot grid dengan SO2 nonkosong / seluruh slot 30 menit dalam [minimum,maksimum] per stasiun. Bukan validitas.'},
 {'Istilah':'Proporsi kehilangan posisi','Definisi':'Awal/tengah/akhir masing-masing dilaporkan sebagai persen dari seluruh slot grid dan persen dari seluruh slot SO2 hilang; jumlah ketiganya = seluruh kehilangan pada data ini.'},
 {'Istilah':'Durasi konstan slot jam','Definisi':'Jumlah slot × 0.5 jam, sedangkan rentang waktu jam = akhir − awal. Keduanya dilaporkan, tidak disamakan.'},
 {'Istilah':'Variabel lain pada rentetan','Definisi':'Dinilai dari baris sumber dalam rentang awal–akhir, termasuk off-grid. Numerik finite; unik=1 dan >=2 nilai disebut konstan pada nilai tersedia, unik>1 berubah. Sel kosong tetap dihitung; tidak menyatakan seluruh rentang lengkap.'},
 {'Istilah':'Kelengkapan meteorologi grid','Definisi':'100 × slot dengan nilai variabel nonkosong / seluruh slot grid stasiun. Tabel tambahan juga mencantumkan nonkosong per baris sumber dan SO2 off-grid.'},
 {'Istilah':'Off-grid','Definisi':'Timestamp yang bukan menit 00/30 dengan detik nol. Tetap berada pada sumber dan statistik per baris tahap 2, tidak menjadi numerator kelengkapan grid.'},
 {'Istilah':'Skenario pemetaan','Definisi':'Hanya audit kemungkinan floor/ceil/terdekat. Tidak menerapkan pemetaan atau agregasi; benturan dihitung terhadap seluruh timestamp sumber pada skenario yang sama.'},
 {'Istilah':'Benturan','Definisi':'Jumlah baris yang akan jatuh pada slot kandidat >1; beda nilai SO2 dihitung dari nilai nonkosong, tanpa memilih donor/nilai.'},
 {'Istilah':'Kelayakan','Definisi':'Pertimbangan kualitas dan cakupan data; lokasi, relevansi spasial dan tujuan penelitian harus diputuskan terpisah.'},
 {'Istilah':'Konteks grafik','Definisi':'Rentetan utama per stasiun ditampilkan sebelum 7 hari, seluruh rentetan, dan sesudah 7 hari; konteks dipotong cakupan sumber. Titik off-grid ditampilkan terpisah tanpa pembulatan.'},
]}
tracked=[ROOT/r['File'] for r in REF['ringkasan']]
tracked=list(dict.fromkeys(tracked))+[OUT/'audit_kualitas_tables.json',OUT/'audit_struktur_data.json',
    OUT/'audit_kualitas_data.xlsx',ROOT/'notebooks/01_audit_data.ipynb',ROOT/'notebooks/02_audit_kualitas_data.ipynb']
before={str(p.relative_to(ROOT)):hashfile(p) for p in tracked}

def add(name,station,**row): tables.setdefault(name,[]).append({'Stasiun':station,**row})
def pc(n,d): return 100*n/d if d else None
def t2rows(name,station): return [r for r in T2[name] if r.get('Stasiun')==station]

def process(path,ws,number):
    station=ws.title
    print('Pemeriksaan lanjutan:',station,flush=True)
    st=next(r for r in REF['ringkasan'] if ROOT/r['File']==path and r['Sheet']==station)
    preview=list(ws.iter_rows(min_row=1,max_row=9,values_only=True))
    header=list(preview[st['Header']-1][1:15]); units=list(preview[st['Satuan']-1][1:15])
    raw=pd.DataFrame([r[1:15] for r in ws.iter_rows(min_row=st['Awal timestamp'],max_col=15,values_only=True)],columns=header,dtype=object)
    time=pd.Series([stamp(v) for v in raw['Waktu']],dtype='datetime64[ns]')
    assert time.notna().all() and time.is_unique and time.is_monotonic_increasing
    aligned=time.eq(time.dt.floor('30min'))
    numeric={c:pd.to_numeric(raw[c],errors='coerce') for c in header[1:]}
    grid=pd.read_csv(OUT/'audit_grids'/f'{number:02d}_grid_audit.csv',parse_dates=['Timestamp'])
    assert int(grid['SO2 nonkosong teramati'].sum())==t2rows('Ringkasan',station)[0]['SO2 teramati slot']
    observed=~raw['SO2'].map(empty)
    g_times=pd.DatetimeIndex(grid['Timestamp'])
    off_ids=raw.index[~aligned]
    for i in off_ids:
        prev=i-1 if i>0 else None; nxt=i+1 if i+1<len(raw) else None
        add('Offgrid_13',station,**{'Baris Excel':int(i+st['Awal timestamp']),'Timestamp asli':raw.at[i,'Waktu'],
            'SO2 asli':raw.at[i,'SO2'],'Waktu sebelum':time[prev] if prev is not None else None,
            'SO2 sebelum':raw.at[prev,'SO2'] if prev is not None else None,
            'Selisih dari sebelum menit':(time[i]-time[prev]).total_seconds()/60 if prev is not None else None,
            'Waktu sesudah':time[nxt] if nxt is not None else None,
            'SO2 sesudah':raw.at[nxt,'SO2'] if nxt is not None else None,
            'Selisih ke sesudah menit':(time[nxt]-time[i]).total_seconds()/60 if nxt is not None else None,
            'Dalam numerator grid tahap 2':False,'Dalam statistik per baris tahap 2':True,
            'Slot floor tersedia baris asli':bool(time.eq(time[i].floor('30min')).any()),
            'Slot ceil tersedia baris asli':bool(time.eq(time[i].ceil('30min')).any())})
    for method in ['floor','ceil','terdekat']:
        candidates=time.dt.floor('30min') if method=='floor' else time.dt.ceil('30min') if method=='ceil' else time.dt.round('30min')
        counts=candidates.value_counts()
        for i in off_ids:
            slot=candidates[i]; ids=raw.index[candidates.eq(slot)]
            vals=raw.loc[ids,'SO2']; nonempty=vals[~vals.map(empty)]
            add('Pemetaan_hipotetis',station,**{'Baris Excel':int(i+10),'Timestamp asli':time[i],
                'Skenario saja':method,'Slot kandidat':slot,'Pergeseran menit':(slot-time[i]).total_seconds()/60,
                'Jumlah baris akan satu slot':int(counts[slot]),'Benturan':int(counts[slot])>1,
                'Jumlah nilai SO2 nonkosong':len(nonempty),'Jumlah nilai SO2 berbeda':len(set(map(str,nonempty))),
                'Baris Excel dalam slot':', '.join(str(j+10) for j in ids)})
    s=t2rows('Ringkasan',station)[0]
    raw_nonempty=int(observed.sum()); off_nonempty=int((observed & ~aligned).sum()); on_nonempty=int((observed & aligned).sum())
    assert on_nonempty==s['SO2 teramati slot'] and raw_nonempty==on_nonempty+off_nonempty
    assert len(raw)==int(aligned.sum())+len(off_ids)
    assert off_nonempty==s['SO2 nonkosong baris di luar grid']
    add('Rekonsiliasi_grid',station,**{'Seluruh baris sumber':len(raw),'Baris pada grid':int(aligned.sum()),
        'Baris off-grid terpisah':len(off_ids),'SO2 nonkosong seluruh baris':raw_nonempty,
        'SO2 nonkosong pada grid':on_nonempty,'SO2 nonkosong off-grid terpisah':off_nonempty,
        'Numerator grid tahap 2':s['SO2 teramati slot'],'Persen grid tahap 2':s['SO2 teramati persen'],
        'Denominator grid':len(grid),'Rekonsiliasi baris lolos':True,'Rekonsiliasi SO2 lolos':True,
        'Status off-grid':'Tidak dihitung pada grid; tetap tersimpan dan dihitung dalam statistik sumber.'})
    tops=sorted(t2rows('Rentetan_SO2',station),key=lambda r:r['Peringkat'])[:10]
    descriptions=[]
    for run in tops:
        a=pd.Timestamp(run['Awal']); b=pd.Timestamp(run['Akhir']); mask=time.between(a,b)
        subset=raw.loc[mask]; change=constant=unavailable=0; variable_states={}
        for col in header[1:]:
            if col=='SO2': continue
            vals=numeric[col][mask]; finite=vals[np.isfinite(vals)]; unique=int(finite.nunique())
            state='berubah pada nilai tersedia' if unique>1 else 'konstan pada nilai tersedia' if unique==1 and len(finite)>=2 else 'bukti tidak cukup'
            change+=unique>1; constant+=state=='konstan pada nilai tersedia'; unavailable+=state=='bukti tidak cukup'
            variable_states[col]=state
            add('Konteks_variabel',station,**{'Peringkat rentetan':run['Peringkat'],'Awal SO2 konstan':a,'Akhir SO2 konstan':b,
                'Variabel':col,'Satuan':units[header.index(col)],'Jumlah baris dalam rentang':len(subset),
                'Baris off-grid dalam rentang':int((mask & ~aligned).sum()),'Nonkosong baris':int((~subset[col].map(empty)).sum()),
                'Numerik finite':len(finite),'Jumlah nilai numerik berbeda':unique,
                'Minimum':finite.min(),'Maksimum':finite.max(),'Penanda deskriptif':state})
        label=f'SO2 konstan; {change} variabel lain berubah, {constant} konstan tersedia, {unavailable} bukti tidak cukup'
        add('Top10_konstan_SO2',station,**{'Peringkat':run['Peringkat'],'Nilai SO2':run['Nilai'],'Satuan':'µg/m³',
            'Awal':a,'Akhir':b,'Jumlah slot':run['Jumlah slot'],'Durasi slot jam':run['Jumlah slot']/2,
            'Rentang waktu jam':(b-a).total_seconds()/3600,'Penanda deskriptif':label,
            'Variabel lain berubah':', '.join(c for c,v in variable_states.items() if v.startswith('berubah')),
            'Variabel lain konstan tersedia':', '.join(c for c,v in variable_states.items() if v.startswith('konstan')),
            'Interpretasi':'Tidak menyatakan kerusakan sensor atau seluruh nilai salah.'})
        descriptions.append(label)
    main=tops[0]; a=pd.Timestamp(main['Awal']); b=pd.Timestamp(main['Akhir'])
    fig,axes=plt.subplots(1,3,figsize=(17,4.8),sharey=True,layout='constrained')
    windows=[(a-pd.Timedelta(days=7),a,'Sebelum (maks. 7 hari)'),(a,b,'Selama rentetan'),(b,b+pd.Timedelta(days=7),'Sesudah (maks. 7 hari)')]
    for ax,(lo,hi,label) in zip(axes,windows):
        part=grid.loc[grid['Timestamp'].between(lo,hi)]
        ax.plot(part['Timestamp'],part['SO2 numerik untuk grafik'],lw=.7,marker='.',markersize=2,color='#24577e',label='SO2 grid (celah dipertahankan)')
        off=(~aligned)&time.between(lo,hi)
        ax.scatter(time[off],numeric['SO2'][off],marker='x',s=30,color='#bf5733',label='SO2 off-grid asli')
        ax.set(title=label,xlabel='Timestamp asli (zona waktu belum diketahui)')
        ax.tick_params(axis='x',rotation=25,labelsize=8); ax.grid(alpha=.15)
        if label.startswith('Selama'): ax.set_facecolor('#fff6d9')
        if part.empty: ax.text(.15,.5,'Tidak ada cakupan data',transform=ax.transAxes)
    axes[0].set_ylabel('SO2 (µg/m³)'); axes[0].legend(fontsize=7)
    fig.suptitle(f"{station} — rentetan deskriptif {main['Nilai']:g} µg/m³, {main['Jumlah slot']} slot",fontsize=12)
    figure=FIG/f'{number:02d}_konteks_konstan_utama.png'; fig.savefig(figure,dpi=150); plt.close(fig)
    add('Grafik_konstan',station,**{'Grafik relatif laporan':f'figures/tahap3/{figure.name}',
        'Awal rentetan':a,'Akhir rentetan':b,'Konteks sebelum tersedia dari':max(time.min(),a-pd.Timedelta(days=7)),
        'Konteks sesudah tersedia sampai':min(time.max(),b+pd.Timedelta(days=7))})
    meteo_coverage={}
    for col in METEO:
        nonempty=~raw[col].map(empty)
        on=nonempty & aligned; oncount=int(on.sum())
        assert time[on].isin(g_times).all()
        meteo_coverage[col]=pc(oncount,len(grid))
        add('Kelengkapan_meteo',station,**{'Variabel':col,'Satuan':units[header.index(col)],
            'Nonkosong grid slot':oncount,'Denominator slot grid':len(grid),'Teramati grid persen':pc(oncount,len(grid)),
            'Nonkosong baris sumber':int(nonempty.sum()),'Denominator baris sumber':len(raw),
            'Teramati baris persen':pc(int(nonempty.sum()),len(raw)),
            'Nonkosong off-grid terpisah':int((nonempty & ~aligned).sum()),'Makna':'Nonkosong, belum validasi sensor.'})
    gaps=t2rows('Gap_SO2',station)
    positions={p:sum(g['Slot hilang'] for g in gaps if g['Posisi']==p) for p in
        ['sebelum pengamatan pertama','tengah seri','setelah pengamatan terakhir']}
    assert sum(positions.values())==s['SO2 hilang slot']
    middle=max((g['Durasi jam'] for g in gaps if g['Posisi']=='tengah seri'),default=0)
    flags=[r for r in t2rows('Aturan_pemeriksaan',station) if r['Jumlah baris']>0]
    issues=[f'{len(off_ids)} timestamp off-grid',f"{s['SO2 hilang slot']} slot SO2 hilang",
        f"rentetan SO2 konstan {main['Jumlah slot']} slot"]
    issues += [f"{r['Variabel']} {r['Aturan']}: {r['Jumlah baris']} baris" for r in flags]
    lowest=min(meteo_coverage,key=meteo_coverage.get)
    implication=(f'Uji gap tengah sampai {middle:g} jam secara terstratifikasi; '
        f"periksa rentetan utama ({descriptions[0]}). Kovariat {lowest} memiliki kelengkapan grid {meteo_coverage[lowest]:.2f}%; "
        'bandingkan fitur SO2 historis saja dan fitur tambahan yang benar-benar tersedia; laporkan cakupan, tanpa memilih stasiun final.')
    row={'Stasiun':station,'Awal pengamatan':time.min(),'Akhir pengamatan':time.max(),
        'Denominator slot grid':len(grid),'SO2 teramati grid persen':s['SO2 teramati persen'],
        'SO2 hilang slot total':s['SO2 hilang slot'],'Gap tengah terpanjang jam':middle,
        'Konstan SO2 terpanjang nilai':main['Nilai'],'Konstan SO2 terpanjang slot':main['Jumlah slot'],
        'Konstan SO2 durasi slot jam':main['Jumlah slot']/2,'Konstan SO2 rentang waktu jam':main['Rentang jam'],
        'Masalah untuk diperiksa':'; '.join(issues),'Implikasi eksperimen':implication,
        'Kesesuaian lokasi penelitian':'Belum dinilai: perlu tujuan spasial, koordinat dan kriteria lokasi.'}
    for key,pos in [('awal','sebelum pengamatan pertama'),('tengah','tengah seri'),('akhir','setelah pengamatan terakhir')]:
        row[f'Hilang {key} slot']=positions[pos]
        row[f'Hilang {key} persen grid']=pc(positions[pos],len(grid))
        row[f'Hilang {key} persen semua hilang']=pc(positions[pos],s['SO2 hilang slot'])
    row.update({f'{col} teramati grid persen':val for col,val in meteo_coverage.items()})
    tables['Ringkasan'].append(row)
    for g in t2rows('Kelompok_gap',station):
        tables.setdefault('Distribusi_gap_acuan',[]).append(g)

for path in dict.fromkeys(ROOT/r['File'] for r in REF['ringkasan']):
    wb=openpyxl.load_workbook(path,read_only=True,data_only=False)
    try:
        for ws in wb.worksheets: process(path,ws,len(tables['Ringkasan'])+1)
    finally: wb.close()
assert len(tables['Offgrid_13'])==13
assert len(tables['Top10_konstan_SO2'])==80
assert len(tables['Konteks_variabel'])==80*12
for path in tracked:
    key=str(path.relative_to(ROOT)); after=hashfile(path); assert before[key]==after
    tables.setdefault('Integritas_SHA256',[]).append({'File':key,'SHA256 sebelum':before[key],'SHA256 sesudah':after,'Identik':True})
tables['Kendala']=[{'Kendala':'Tidak ada pemeriksaan gagal. Interpretasi sensor/lokasi dan kebijakan timestamp tetap membutuhkan penyedia data.'}]
def clean(v):
    if isinstance(v,(datetime,date,pd.Timestamp)): return v.isoformat(sep=' ') if isinstance(v,(datetime,pd.Timestamp)) else v.isoformat()
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,np.bool_): return bool(v)
    if isinstance(v,(float,np.floating)) and not np.isfinite(v): return None
    return v
serial={name:[{k:clean(v) for k,v in r.items()} for r in rows] for name,rows in tables.items()}
(OUT/'pemeriksaan_lanjutan_tables.json').write_text(json.dumps(serial,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
from write_preprocessing_design import write_design
write_design(serial)
print('Selesai: 13 off-grid, 80 rentetan, 960 pemeriksaan variabel lain, 8 grafik, rekonsiliasi dan hash lolos.')
