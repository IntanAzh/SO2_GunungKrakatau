"""Audit kualitas baca-saja. Semua grid/tabel adalah turunan audit, bukan data latih."""
from pathlib import Path
from datetime import datetime, date
from hashlib import sha256
from collections import Counter
import json
import re
import subprocess
import os
import numpy as np
import pandas as pd
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
FIG = OUT / 'figures'
BUCKETS = ['30 menit', '>30 menit–1 jam', '>1–3 jam', '>3–6 jam', '>6–24 jam', '>24 jam']
MONTHS = {v: i+1 for i, v in enumerate(['januari','februari','maret','april','mei','juni',
    'juli','agustus','september','oktober','november','desember'])}
DEFINITIONS = [
 ('Sumber', 'Excel asli pada data_raw; struktur mengacu notebook 01 dan audit_struktur_data.json. Sumber dibaca read_only, tidak disimpan ulang.'),
 ('Persen kosong variabel', '100 × jumlah sel kosong / seluruh baris tabel stasiun (termasuk timestamp gagal dan duplikat). Kosong = None, NaN, atau teks hanya whitespace; nol bukan kosong.'),
 ('Grid audit', 'Slot menit 00/30 dengan detik dan mikrodetik nol dalam [timestamp minimum, maksimum]. Tidak membulatkan timestamp di luar pola; slot di luar pola dilaporkan terpisah.'),
 ('Persen SO2 teramati', '100 × slot grid dengan sedikitnya satu nilai SO2 nonkosong / seluruh slot grid. Nonkosong tidak otomatis valid; teks, nol, negatif, ekstrem dan duplikat bertentangan tetap dapat dihitung teramati.'),
 ('Persen tanpa baris', '100 × slot grid tanpa baris sumber / seluruh slot grid.'),
 ('Persen baris ada SO2 kosong', '100 × slot grid dengan baris tetapi semua SO2 kosong / seluruh slot grid.'),
 ('Persen hilang SO2', '100 × (slot tanpa baris + slot baris ada semua SO2 kosong) / seluruh slot grid.'),
 ('Persen bulanan', 'Definisi kelengkapan SO2 yang sama, denominator slot grid dalam bulan tersebut; bulan awal/akhir dapat parsial.'),
 ('Duplikat', 'Grup berdasarkan timestamp berhasil dibaca; identik jika seluruh 13 nilai pengukuran sama termasuk pola kosong dan tipe Python. Konflik bila ada perbedaan. Baris tambahan = jumlah baris grup − 1; tidak ada baris dihapus.'),
 ('Plot dan konstan', 'Slot numerik hanya jika SO2 semua baris pada timestamp setuju dan tidak kosong/nonnumerik/nonfinite. Konflik SO2, sel kosong, atau slot hilang memutus garis dan rentetan. Duplikat SO2 identik ditampilkan sekali hanya pada grid audit.'),
 ('Rentetan konstan', 'Kesamaan numerik tepat (tanpa toleransi), minimal 2 slot 30 menit berurutan. Jumlah slot dan rentang akhir−awal dilaporkan terpisah. Per variabel, kosong/konflik/nonnumerik memutus rentetan.'),
 ('Gap SO2', 'Rentetan slot grid tanpa SO2 nonkosong. Durasi = jumlah slot × 30 menit; bukan akhir−awal. Satu slot berdurasi 30 menit.'),
 ('Posisi gap', 'Sebelum slot SO2 teramati pertama; setelah slot teramati terakhir; tengah seri; atau seluruh seri jika tidak ada SO2 teramati.'),
 ('Kelompok gap', '30 menit; >30 menit–1 jam; >1–3 jam; >3–6 jam; >6–24 jam; >24 jam. Kelompok deskriptif, bukan batas imputasi.'),
 ('Statistik nilai', 'Dihitung per baris sumber dari nilai yang dapat direpresentasikan numerik dan finite; tidak mengganti kolom sumber. Teks numerik dihitung setelah parsing sementara dan dilaporkan tipenya. Jumlah numerik bukan jumlah valid.'),
 ('Nol dan negatif', 'Jumlah di antara nilai numerik finite per baris sumber. Tidak dihapus dan tidak diasumsikan salah.'),
 ('Kandidat ekstrem', 'Contoh nilai minimum/maksimum setiap variabel, bukan keputusan outlier atau batas sensor.'),
 ('Metadata', 'Tanggal laporan tanpa jam dibandingkan secara tanggal kalender; akhir hari inklusif. Zona waktu, kode sensor, ambang sensor dan penyebab anomali tidak disimpulkan.'),
 ('Interval sumber', 'Selisih dua baris bertetangga pada urutan asli hanya jika keduanya berhasil dibaca. Timestamp gagal memutus pasangan. Interval unik terurut dihitung terpisah, tanpa mengubah sumber.'),
 ('Persentase lain', 'Tidak ada persentase lain selain denominator eksplisit pada tabel dan definisi di atas. Angka 0–100 adalah persen, bukan fraksi Excel.'),
]

def empty(v):
    return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, float) and np.isnan(v))

def signature(v):
    return ('kosong', '') if empty(v) else (type(v).__name__, str(v))

def stamp(v):
    if isinstance(v, datetime):
        return pd.Timestamp(v) if v.tzinfo is None else pd.NaT
    if isinstance(v, str):
        try:
            return pd.Timestamp(datetime.strptime(v, '%Y-%m-%d %H:%M:%S'))
        except ValueError:
            return pd.NaT
    return pd.NaT

def metadata_period(text):
    hits = re.findall(r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', str(text))
    if len(hits) != 2:
        return None, None
    try:
        return tuple(date(int(y), MONTHS[m.lower()], int(d)) for d,m,y in hits)
    except (KeyError, ValueError):
        return None, None

def pct(n, d):
    return 100*n/d if d else None

def hashfile(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def bucket(slots):
    return BUCKETS[0 if slots==1 else 1 if slots<=2 else 2 if slots<=6 else 3 if slots<=12 else 4 if slots<=48 else 5]

def runs(values, grid):
    """Rentetan numerik tepat; NaN memutus, grid wajib 30 menit berurutan."""
    result = []
    start = 0
    for i in range(1, len(values)+1):
        if i < len(values) and np.isfinite(values[i]) and np.isfinite(values[i-1]) and values[i] == values[i-1] and grid[i]-grid[i-1] == pd.Timedelta(minutes=30):
            continue
        if i-start >= 2 and np.isfinite(values[start]):
            result.append({'Awal':grid[start], 'Akhir':grid[i-1], 'Jumlah slot':i-start,
                           'Rentang jam':(i-start-1)/2, 'Nilai':float(values[start])})
        start = i
    return sorted(result, key=lambda r: (-r['Jumlah slot'], r['Awal']))

def audit_station(path, ws, structure, tables, number):
    station = ws.title
    common = {'File':path.name, 'Stasiun':station}
    def add(table, **row):
        tables.setdefault(table, []).append({**common, **row})
    h = structure['Header']; u = structure['Satuan']; start = structure['Awal timestamp']
    preview = list(ws.iter_rows(min_row=1, max_row=start-1, values_only=True))
    header = list(preview[h-1][1:15]); units = list(preview[u-1][1:15])
    assert header[0] == 'Waktu' and header[3] == 'SO2', f'Layout berubah: {station}'
    rows = list(ws.iter_rows(min_row=start, max_col=15, values_only=True))
    raw = pd.DataFrame([r[1:15] for r in rows], columns=header, dtype=object)
    excel_rows = np.arange(start, start+len(raw))
    times = pd.Series([stamp(v) for v in raw['Waktu']], dtype='datetime64[ns]')
    ok = times.notna(); valid = times[ok]
    if valid.empty:
        raise ValueError(f'{station}: tidak ada timestamp terbaca; grid tidak dapat dibuat.')
    low, high = valid.min(), valid.max()
    aligned = ok & times.map(lambda t: False if pd.isna(t) else t.minute in (0,30) and t.second==0 and t.microsecond==0 and t.nanosecond==0)
    grid = pd.date_range(low.ceil('30min'), high.floor('30min'), freq='30min')
    groups = {t:list(ix) for t,ix in times[ok].groupby(times[ok]).groups.items()}
    numeric = {}; masks = {}
    for j, col in enumerate(header[1:], start=1):
        values = raw[col]
        missing = values.map(empty)
        # Konversi hanya salinan audit; nilai sumber tidak diubah.
        nums = pd.to_numeric(values.where(~missing), errors='coerce').astype(float)
        finite = np.isfinite(nums)
        numeric[col] = nums
        masks[col] = missing
        bad = ~missing & ~finite
        types = Counter(type(v).__name__ for v in values if not empty(v))
        num = nums[finite]
        q = num.quantile([.01,.25,.5,.75,.99])
        add('Kelengkapan_variabel', Variabel=col, Satuan=units[j],
            **{'Denominator baris':len(raw),'Kosong':int(missing.sum()),'Kosong persen':pct(int(missing.sum()),len(raw)),
               'Nonkosong teramati':int((~missing).sum()),'Numerik finite':len(num),
               'Nonnumerik atau nonfinite':int(bad.sum()),'Tipe asli nonkosong':json.dumps(types,ensure_ascii=False)})
        add('Statistik_nilai', Variabel=col, Satuan=units[j], Jumlah=len(num), Minimum=num.min(),
            P1=q.loc[.01], P25=q.loc[.25], P50=q.loc[.5], P75=q.loc[.75], P99=q.loc[.99], Maksimum=num.max(),
            Nol=int((num==0).sum()), Negatif=int((num<0).sum()))
        for idx in raw.index[bad]:
            add('Nilai_nonnumerik', Variabel=col, **{'Baris Excel':int(excel_rows[idx]),'Timestamp asli':raw.at[idx,'Waktu'],
                'Nilai asli':str(values[idx]),'Tipe asli':type(values[idx]).__name__,
                'Kategori':'nonnumerik' if pd.isna(nums[idx]) else 'nonfinite'})
        for label, val in [('minimum',num.min()),('maksimum',num.max())]:
            if pd.notna(val):
                idx = num[num==val].index[0]
                add('Kandidat_ekstrem', Variabel=col, Satuan=units[j], Kandidat=label, Nilai=val,
                    **{'Baris Excel contoh':int(excel_rows[idx]),'Timestamp':raw.at[idx,'Waktu'],
                       'Jumlah kemunculan':int((num==val).sum())})
    for idx in raw.index[~ok]:
        add('Masalah_waktu', Jenis='timestamp gagal', **{'Baris Excel':int(excel_rows[idx]),'Nilai asli':str(raw.at[idx,'Waktu'])})
    for idx in raw.index[ok & ~aligned]:
        add('Masalah_waktu', Jenis='di luar menit 00/30 atau detik bukan nol', **{'Baris Excel':int(excel_rows[idx]),'Nilai asli':str(raw.at[idx,'Waktu'])})
    dup_identical = dup_conflict = dup_rows = dup_extra = 0
    for t, ix in groups.items():
        if len(ix) < 2:
            continue
        different = [col for col in header[1:] if len({signature(raw.at[i,col]) for i in ix}) > 1]
        kind = 'bertentangan' if different else 'identik'
        dup_conflict += bool(different); dup_identical += not bool(different)
        dup_rows += len(ix); dup_extra += len(ix)-1
        add('Duplikat', Timestamp=t, Kategori=kind, **{'Jumlah baris':len(ix),'Baris tambahan':len(ix)-1,
            'Kolom berbeda':', '.join(different),'Baris Excel':', '.join(str(excel_rows[i]) for i in ix)})
        for i in ix:
            add('Detail_duplikat', Timestamp=t, Kategori=kind, **{'Baris Excel':int(excel_rows[i])},
                **{c:raw.at[i,c] for c in header[1:]})
    delta = times.diff()
    pairs = ok & ok.shift(1, fill_value=False)
    irregular = pairs & delta.ne(pd.Timedelta(minutes=30))
    for i in raw.index[irregular]:
        add('Interval_tidak_30m', Jenis='urutan sumber', **{'Baris Excel akhir':int(excel_rows[i]),
            'Waktu awal':times[i-1], 'Waktu akhir':times[i], 'Interval menit':delta[i].total_seconds()/60})
    unique = pd.DatetimeIndex(sorted(groups))
    unique_delta = unique[1:]-unique[:-1]
    for i in np.flatnonzero(unique_delta != pd.Timedelta(minutes=30)):
        add('Interval_tidak_30m', Jenis='timestamp unik terurut', **{'Waktu awal':unique[i],
            'Waktu akhir':unique[i+1], 'Interval menit':unique_delta[i].total_seconds()/60})
    present = np.array([t in groups for t in grid])
    observed = np.array([any(not empty(raw.at[i,'SO2']) for i in groups.get(t,[])) for t in grid])
    blank = present & ~observed
    missing_so2 = ~observed
    grid_numeric = {}
    ambiguous = np.zeros(len(grid), dtype=bool)
    for col in header[1:]:
        arr = np.full(len(grid), np.nan)
        for k,t in enumerate(grid):
            ix = groups.get(t, [])
            if ix and len({signature(raw.at[i,col]) for i in ix}) == 1 and all(not empty(raw.at[i,col]) for i in ix):
                v = numeric[col].loc[ix[0]]
                if np.isfinite(v): arr[k] = v
            elif col == 'SO2' and ix and len({signature(raw.at[i,col]) for i in ix}) > 1:
                ambiguous[k] = True
        grid_numeric[col] = arr
        const = runs(arr, grid)
        add('Ringkas_konstan', Variabel=col, **{'Jumlah rentetan minimal 2 slot':len(const),
            'Rentetan terpanjang slot':const[0]['Jumlah slot'] if const else 0})
        if col=='SO2':
            for rank, run in enumerate(const,1):
                add('Rentetan_SO2', Peringkat=rank, **run)
            top_runs = const[:10]
    gaps=[]; k=0
    observed_ix = np.flatnonzero(observed)
    while k < len(grid):
        if not missing_so2[k]: k+=1; continue
        first=k
        while k < len(grid) and missing_so2[k]: k+=1
        length=k-first
        position = ('seluruh seri' if not len(observed_ix) else 'sebelum pengamatan pertama' if k-1 < observed_ix[0]
                    else 'setelah pengamatan terakhir' if first > observed_ix[-1] else 'tengah seri')
        run={'Awal':grid[first],'Akhir':grid[k-1],'Slot hilang':length,'Durasi menit':length*30,
             'Durasi jam':length/2,'Posisi':position,'Kelompok':bucket(length),
             'Slot tanpa baris':int((~present[first:k]).sum()), 'Slot baris ada SO2 kosong':int(blank[first:k].sum())}
        gaps.append(run); add('Gap_SO2', **run)
    for label in BUCKETS:
        subset=[g for g in gaps if g['Kelompok']==label]
        add('Kelompok_gap', Kelompok=label, **{'Jumlah kejadian':len(subset),
            'Jumlah slot hilang':sum(g['Slot hilang'] for g in subset),
            'Total durasi jam':sum(g['Durasi jam'] for g in subset)})
    gd = pd.DataFrame({'Timestamp':grid,'Ada baris':present,'SO2 nonkosong teramati':observed,
        'Baris ada SO2 kosong':blank,'SO2 ambigu duplikat':ambiguous,'SO2 numerik untuk grafik':grid_numeric['SO2']})
    grid_dir=OUT/'audit_grids'; grid_dir.mkdir(exist_ok=True)
    gd.to_csv(grid_dir/f'{number:02d}_grid_audit.csv',index=False)
    monthly=[]
    for month, frame in gd.groupby(gd['Timestamp'].dt.to_period('M')):
        total=len(frame); obs=int(frame['SO2 nonkosong teramati'].sum())
        m={'Bulan':str(month),'Awal cakupan grid':frame['Timestamp'].min(),'Akhir cakupan grid':frame['Timestamp'].max(),
           'Denominator slot grid':total,'SO2 teramati slot':obs,'SO2 teramati persen':pct(obs,total),
           'Tanpa baris slot':int((~frame['Ada baris']).sum()),
           'Baris ada SO2 kosong slot':int(frame['Baris ada SO2 kosong'].sum())}
        monthly.append(m); add('SO2_bulanan', **m)
    meta_text=preview[3][4]
    meta_low, meta_high = metadata_period(meta_text)
    outside = sum(t.date()<meta_low or t.date()>meta_high for t in valid) if meta_low else None
    add('Metadata', **{'Periode teks asli':meta_text,'Awal metadata':meta_low,'Akhir metadata':meta_high,
        'Tanggal aktual minimum':low.date(),'Tanggal aktual maksimum':high.date(),
        'Awal tanggal sesuai':low.date()==meta_low if meta_low else None,
        'Akhir tanggal sesuai':high.date()==meta_high if meta_high else None,
        'Baris di luar tanggal metadata':outside,'Keterbatasan':'Tanggal tanpa jam; zona waktu dan kode sensor tidak dijelaskan.'})
    checks=[('Kelembaban','di luar 0–100%',lambda v:(v<0)|(v>100)),
            ('Arah Angin','di luar 0–360 derajat',lambda v:(v<0)|(v>360)),
            ('Kec.Angin','negatif',lambda v:v<0)]
    for col,rule,fn in checks:
        mask=fn(numeric[col]) & np.isfinite(numeric[col]); ids=raw.index[mask]
        add('Aturan_pemeriksaan', Variabel=col, Aturan=rule, **{'Jumlah baris':len(ids),
            'Denominator numerik finite':int(np.isfinite(numeric[col]).sum()),'Contoh dibatasi':10})
        for i in ids[:10]:
            add('Contoh_pemeriksaan', Variabel=col, Aturan=rule, Nilai=numeric[col][i],
                **{'Baris Excel':int(excel_rows[i]),'Timestamp':raw.at[i,'Waktu']})
    s={**common,'Jumlah baris sumber':len(raw),'Timestamp minimum':low,'Timestamp maksimum':high,
       'Timestamp gagal':int((~ok).sum()),'Waktu nonmenurun':bool(valid.is_monotonic_increasing),
       'Waktu meningkat ketat':bool(valid.is_monotonic_increasing and valid.is_unique),
       'Pasangan waktu mundur':int((delta[pairs]<pd.Timedelta(0)).sum()),
       'Grup duplikat identik':int(dup_identical),'Grup duplikat konflik':int(dup_conflict),
       'Baris dalam grup duplikat':dup_rows,'Baris duplikat tambahan':dup_extra,
       'Di luar pola 00 atau 30':int((ok & ~aligned).sum()),
       'Pasangan sumber terbaca':int(pairs.sum()),'Pasangan sumber bukan 30 menit':int(irregular.sum()),
       'Interval unik terurut bukan 30 menit':int((unique_delta!=pd.Timedelta(minutes=30)).sum()),
       'Denominator slot grid':len(grid),'SO2 teramati slot':int(observed.sum()),
       'SO2 teramati persen':pct(int(observed.sum()),len(grid)),
       'Tanpa baris slot':int((~present).sum()),'Tanpa baris persen':pct(int((~present).sum()),len(grid)),
       'Baris ada SO2 kosong slot':int(blank.sum()),'Baris ada SO2 kosong persen':pct(int(blank.sum()),len(grid)),
       'SO2 hilang slot':int(missing_so2.sum()),'SO2 hilang persen':pct(int(missing_so2.sum()),len(grid)),
       'Slot SO2 ambigu':int(ambiguous.sum()),'Jumlah gap SO2':len(gaps),
       'Gap terpanjang jam':max((g['Durasi jam'] for g in gaps), default=0),
       'SO2 nonkosong baris di luar grid':int((~masks['SO2'] & ok & ~aligned).sum())}
    tables['Ringkasan'].append(s)
    assert int(observed.sum())+int(blank.sum())+int((~present).sum())==len(grid)
    assert sum(g['Slot hilang'] for g in gaps)==int(missing_so2.sum())
    assert sum(m['Denominator slot grid'] for m in monthly)==len(grid)
    assert sum(m['SO2 teramati slot'] for m in monthly)==int(observed.sum())
    fig,axs=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    axs[0,0].plot(grid,grid_numeric['SO2'],lw=.6,marker='.',markersize=1,color='#235789')
    axs[0,0].set(title='SO2 terhadap waktu — celah dipertahankan',ylabel='SO2 (µg/m³)',xlabel='Waktu (zona tidak dinyatakan)')
    axs[0,0].tick_params(axis='x',labelrotation=25)
    axs[0,1].bar([m['Bulan'] for m in monthly],[m['SO2 teramati persen'] for m in monthly],color='#3b8c6e')
    axs[0,1].set(title='Kelengkapan SO2 bulanan',ylabel='% slot nonkosong / grid bulan',ylim=(0,105))
    axs[0,1].tick_params(axis='x',labelrotation=90,labelsize=8)
    axs[1,0].bar(range(6),[sum(g['Kelompok']==b for g in gaps) for b in BUCKETS],color='#b37838')
    axs[1,0].set(xticks=range(6),xticklabels=BUCKETS,title='Distribusi panjang gap SO2',ylabel='Jumlah kejadian')
    axs[1,0].tick_params(axis='x',labelrotation=25,labelsize=8)
    if top_runs:
        labels=[f"{r['Awal']:%Y-%m-%d %H:%M} | {r['Nilai']:g}" for r in top_runs]
        axs[1,1].barh(range(len(top_runs)),[r['Jumlah slot'] for r in top_runs],color='#72528c')
        axs[1,1].set(yticks=range(len(top_runs)),yticklabels=labels,xlabel='Jumlah slot berurutan',title='10 rentetan SO2 konstan terpanjang')
        axs[1,1].invert_yaxis(); axs[1,1].tick_params(axis='y',labelsize=8)
    else:
        axs[1,1].text(.1,.5,'Tidak ada rentetan minimal 2 slot',transform=axs[1,1].transAxes)
    fig.suptitle(station,fontsize=15)
    figure=FIG/f'{number:02d}_audit_so2.png'; fig.savefig(figure,dpi=140); plt.close(fig)
    return {'summary':s,'monthly':monthly,'gaps':gaps,'top_runs':top_runs,'figure':figure,'grid':gd}

def run_audit():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    reference=json.loads((OUT/'audit_struktur_data.json').read_text(encoding='utf-8'))
    nb01=json.loads((ROOT/'notebooks'/'01_audit_data.ipynb').read_text(encoding='utf-8'))
    assert all(not any(o['output_type']=='error' for o in c.get('outputs',[])) for c in nb01['cells'])
    tables={'Ringkasan':[], 'Definisi':[{'Istilah':k,'Definisi':v} for k,v in DEFINITIONS]}
    results=[]; source_records=[]; issues=[]
    paths=sorted({ROOT/r['File'] for r in reference['ringkasan']})
    before={str(p.relative_to(ROOT)):hashfile(p) for p in paths}
    for path in paths:
        wb=openpyxl.load_workbook(path,read_only=True,data_only=False)
        try:
            for ws in wb.worksheets:
                match=[r for r in reference['ringkasan'] if ROOT/r['File']==path and r['Sheet']==ws.title]
                if len(match)!=1:
                    issues.append(f'{path.name}/{ws.title}: struktur tahap 1 tidak tersedia.'); continue
                print(f'Audit: {ws.title}',flush=True)
                results.append(audit_station(path,ws,match[0],tables,len(results)+1))
        finally:
            wb.close()
    for path in paths:
        key=str(path.relative_to(ROOT)); after=hashfile(path)
        source_records.append({'File sumber':key,'SHA256 sebelum':before[key],'SHA256 sesudah':after,
            'Identik selama audit':before[key]==after,
            'Identik dengan tahap 1':before[key]==reference['sha256_sesudah'].get(key)})
        assert before[key]==after,'Sumber berubah selama audit!'
    tables['Sumber_SHA256']=source_records
    tables['Kendala']=[{'Kendala':x} for x in issues] or [{'Kendala':'Tidak ada pemeriksaan yang gagal dijalankan.'}]
    # Nama/tabel kosong tetap tersedia, tanpa mengarang observasi.
    for name in ['Duplikat','Detail_duplikat','Masalah_waktu','Nilai_nonnumerik','Contoh_pemeriksaan','Rentetan_SO2','Gap_SO2']:
        tables.setdefault(name,[])
    def clean(v):
        if isinstance(v,(datetime,date,pd.Timestamp)): return v.isoformat(sep=' ') if isinstance(v,(datetime,pd.Timestamp)) else v.isoformat()
        if isinstance(v,(np.integer,)): return int(v)
        if isinstance(v,(np.bool_,)): return bool(v)
        if isinstance(v,(float,np.floating)) and not np.isfinite(v): return None
        return v
    serial={name:[{k:clean(v) for k,v in r.items()} for r in rows] for name,rows in tables.items()}
    (OUT/'audit_kualitas_tables.json').write_text(json.dumps(serial,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_report(serial,results)
    return tables,results

def write_report(tables, results):
    lines=['# Ringkasan audit kualitas data SO₂','',
        'Audit terpisah untuk setiap stasiun, mengacu struktur notebook 01. File sumber dibaca tanpa modifikasi. '
        'Tidak dilakukan imputasi, penghapusan outlier, smoothing, scaling, pemilihan stasiun final atau pelatihan model.','',
        '## Definisi dan denominator','']
    lines += [f'- **{k}:** {v}' for k,v in DEFINITIONS]
    lines += ['', '## Ringkasan per stasiun','',
        '| Stasiun | Baris sumber | Slot grid | SO2 teramati slot | Teramati (%) | Tanpa baris | Baris ada, SO2 kosong | Gap terpanjang (jam) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in tables['Ringkasan']:
        lines.append(f"| {s['Stasiun']} | {s['Jumlah baris sumber']} | {s['Denominator slot grid']} | {s['SO2 teramati slot']} | {s['SO2 teramati persen']:.2f} | {s['Tanpa baris slot']} | {s['Baris ada SO2 kosong slot']} | {s['Gap terpanjang jam']:g} |")
    for s in tables['Ringkasan']:
        station=s['Stasiun']; result=next(r for r in results if r['summary']['Stasiun']==station)
        lines+=['',f'### {station}','',
            f"- Waktu aktual: {s['Timestamp minimum']} sampai {s['Timestamp maksimum']}. Urutan nonmenurun: {s['Waktu nonmenurun']}; meningkat ketat: {s['Waktu meningkat ketat']}.",
            f"- Timestamp gagal: {s['Timestamp gagal']}; grup duplikat identik: {s['Grup duplikat identik']}; konflik: {s['Grup duplikat konflik']}; baris tambahan duplikat: {s['Baris duplikat tambahan']}.",
            f"- Di luar pola 00/30: {s['Di luar pola 00 atau 30']}; pasangan sumber bukan 30 menit: {s['Pasangan sumber bukan 30 menit']} dari {s['Pasangan sumber terbaca']} pasangan terbaca.",
            f"- SO2 hilang: {s['SO2 hilang slot']} slot ({s['SO2 hilang persen']:.2f}% dari {s['Denominator slot grid']} slot); {s['Jumlah gap SO2']} kejadian. SO2 ambigu pada duplikat: {s['Slot SO2 ambigu']} slot."]
        meta=next(r for r in tables['Metadata'] if r['Stasiun']==station)
        lines.append(f"- Metadata periode: `{meta['Periode teks asli']}`. Awal tanggal sesuai: {meta['Awal tanggal sesuai']}; akhir tanggal sesuai: {meta['Akhir tanggal sesuai']}; baris di luar tanggal metadata: {meta['Baris di luar tanggal metadata']}.")
        positions=Counter(g['Posisi'] for g in result['gaps'])
        lines.append('- Posisi gap (jumlah kejadian): '+(', '.join(f'{k}: {v}' for k,v in positions.items()) or 'tidak ada gap')+'.')
        if result['gaps']:
            g=max(result['gaps'],key=lambda x:x['Slot hilang'])
            lines.append(f"- Gap terpanjang: {g['Awal']} sampai {g['Akhir']}; {g['Slot hilang']} slot × 30 menit = {g['Durasi jam']:g} jam ({g['Posisi']}).")
        if result['top_runs']:
            c=result['top_runs'][0]
            lines.append(f"- Rentetan SO2 konstan terpanjang: {c['Nilai']:g} µg/m³; {c['Jumlah slot']} slot, {c['Awal']} sampai {c['Akhir']}. Belum dinyatakan sebagai kesalahan.")
        flags=[r for r in tables['Aturan_pemeriksaan'] if r['Stasiun']==station]
        lines.append('- Pemeriksaan rentang: '+ '; '.join(f"{r['Variabel']} {r['Aturan']}: {r['Jumlah baris']} baris" for r in flags)+'.')
        vals=[r for r in tables['Kelengkapan_variabel'] if r['Stasiun']==station]
        lines.append(f"- Nilai nonnumerik/nonfinite pada kolom pengukuran: {sum(r['Nonnumerik atau nonfinite'] for r in vals)} sel. Tipe asli, statistik, nol dan negatif tersedia pada tabel Excel.")
        lines+=['',f"![Audit {station}](figures/{result['figure'].name})"]
    lines+=['','## Keterbatasan dan implikasi rancangan eksperimen','',
        '- Tanggal metadata tidak menjelaskan jam, zona waktu, status sensor, kode khusus, batas deteksi atau batas sensor. Kesesuaian tanggal tidak menjamin seri lengkap.',
        '- Grid mencakup minimum sampai maksimum timestamp tiap stasiun, bukan sebelum/di luar cakupan tersebut. Kelengkapan antarstasiun memiliki denominator periode yang berbeda; bulan tepi dapat parsial.',
        '- Timestamp di luar pola 00/30 tidak dipindahkan ke grid. Jika ada, bandingkan hasil grid dengan observasi di luar grid sebelum merancang preprocessing.',
        '- Sel nonkosong adalah teramati, belum otomatis valid. Nol, pengulangan dan SO2 tinggi belum membuktikan kesalahan. Nilai ekstrem minimum/maksimum ditampilkan sebagai kandidat pemeriksaan tanpa batas sensor rekaan.',
        '- Statistik per baris mempertahankan duplikat, sedangkan kelengkapan grid menghitung satu slot per timestamp. Konflik duplikat harus diselesaikan berdasarkan bukti sebelum eksperimen; audit tidak memilih nilai sumber.',
        '- Bandingkan eksperimen berdasarkan panjang dan posisi gap; kelompok gap hanya deskriptif. Gap panjang/pada tepi perlu dipertimbangkan terpisah dari gap pendek di tengah, tanpa menetapkan metode imputasi sekarang.',
        '- Pemisahan train/validasi/test kelak harus kronologis. Rancang estimasi parameter preprocessing hanya dari train dan evaluasi kehilangan sintetis hanya pada observasi yang telah diverifikasi, agar tidak terjadi kebocoran.',
        '- Rentetan konstan, lonjakan, dan rentang meteorologi perlu ditelusuri dengan dokumentasi/penyedia data. Tidak ada dugaan zona waktu, penyebab anomali, atau kaitan dengan aktivitas Anak Krakatau.',
        '- Grafik tidak menghubungkan titik melalui nilai kosong, baris hilang atau SO2 ambigu/nonnumerik. Grid CSV disimpan di outputs/audit_grids khusus audit dan tidak menggantikan file mentah.',
        '', '## Kendala','']
    lines += ['- '+r['Kendala'] for r in tables['Kendala']]
    lines += ['', '## Integritas sumber','']
    for r in tables['Sumber_SHA256']:
        lines += [f"- File: `{r['File sumber']}`",f"- SHA256 sebelum: `{r['SHA256 sebelum']}`",
            f"- SHA256 sesudah: `{r['SHA256 sesudah']}`",f"- Identik selama audit: {r['Identik selama audit']}; identik dengan tahap 1: {r['Identik dengan tahap 1']}."]
    (OUT/'ringkasan_audit_kualitas.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def export_excel():
    node=Path(os.environ.get('SO2_AUDIT_NODE',r'C:\Users\INTAN\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'))
    if not node.exists():
        raise FileNotFoundError('Node runtime untuk ekspor Excel tidak tersedia. Atur SO2_AUDIT_NODE ke runtime yang memiliki @oai/artifact-tool.')
    completed=subprocess.run([str(node),str(ROOT/'scripts'/'export_quality.mjs')],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    print(completed.stdout)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
