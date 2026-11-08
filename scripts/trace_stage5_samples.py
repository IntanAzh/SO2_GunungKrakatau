"""Penelusuran baca-saja dari artefak tahap 5; tidak memanggil fit/predict/model."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OLD=OUT/'forecasting_way_urang'
SUPPORT=OUT/'penelusuran_tahap5';SUPPORT.mkdir(exist_ok=True)
def digest(p):
    h=sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def dump(path,value):
    def convert(v):
        if isinstance(v,dict):return {k:convert(x) for k,x in v.items()}
        if isinstance(v,list):return [convert(x) for x in v]
        if isinstance(v,pd.Timestamp):return v.isoformat(sep=' ')
        if isinstance(v,np.integer):return int(v)
        if isinstance(v,np.bool_):return bool(v)
        if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
        return v
    path.write_text(json.dumps(convert(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

cfg=json.loads((OLD/'config_locked.json').read_text(encoding='utf-8'))
prior=json.loads((OUT/'forecasting_way_urang_tables.json').read_text(encoding='utf-8'))
methods=cfg['pipelines'];a=cfg['split'][0]['Jumlah slot'];b=a+cfg['split'][1]['Jumlah slot']
refs=list(OLD.glob('*'))+[OUT/'forecasting_acuan_way_urang.xlsx',OUT/'forecasting_way_urang_tables.json',
    OUT/'ringkasan_forecasting_acuan_way_urang.md',ROOT/'notebooks/05_forecasting_acuan_way_urang.ipynb',
    OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',ROOT/'scripts/forecast_ridge_way_urang.py']+list((ROOT/'data_raw').glob('*.xlsx'))
refs=[p for p in refs if p.is_file()];before={str(p.relative_to(ROOT)):digest(p) for p in refs}
d=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=b,parse_dates=['Timestamp'])
times=pd.DatetimeIndex(d.Timestamp);y=d['SO2 numerik'].to_numpy(dtype=float)
assert times[-1]<pd.Timestamp(cfg['split'][2]['Awal'])
raw=np.column_stack([y]+[d[f'{col} numerik'].shift(1).to_numpy(dtype=float) for col in cfg['covariates']])
hist={};features={};feature_names=None
for name in methods:
    f=pd.read_csv(OLD/f'fitur_{name}.csv',nrows=a)
    feature_names=list(f.columns[1:]);features[name]=f.iloc[:,1:].to_numpy(dtype=float)
    assert features[name].shape==(a,36) and np.isfinite(features[name][:,18:]).all()
    h=pd.read_csv(OLD/f'histori_kausal_{name}.csv',nrows=a)
    hist[name]=h.iloc[:,1:13].to_numpy(dtype=float)
assert len(feature_names)==36
origins=np.arange(a);targets=origins+1;target_y=y[targets]
observed=np.isfinite(target_y);inside=targets<a
structural=origins>=max(cfg['so2_lags']);after_warmup=origins>=cfg['warmup_slots']
after_buffer=origins>=cfg['warmup_slots']+max(cfg['so2_lags'])
raw_features=np.column_stack([pd.Series(raw[:a,0]).shift(lag).to_numpy() for lag in cfg['so2_lags']]+[raw[:a,j] for j in range(1,12)])
missing_original=~np.isfinite(raw_features)
finite={name:np.isfinite(X).all(axis=1) for name,X in features.items()}
eligible={name:observed&after_buffer&inside&finite[name] for name in methods}
common=np.logical_and.reduce(list(eligible.values()))
saved=pd.read_csv(OLD/'timestamp_train_bersama.csv')['Index origin'].to_numpy(dtype=int)
assert np.array_equal(np.flatnonzero(common),saved)
threshold=float(np.max(target_y[common]));high=observed&inside&(target_y>threshold)
tables={'Ringkasan':[]};stages={};monthly=[];high_stage=[]

def stats(mask,**context):
    ids=np.flatnonzero(mask);values=target_y[ids];valid=values[np.isfinite(values)]
    return {**context,'Jumlah sampel':len(ids),'Target asli teramati':len(valid),
        'Origin awal':times[origins[ids[0]]] if len(ids) else None,'Origin akhir':times[origins[ids[-1]]] if len(ids) else None,
        'Target awal':times[targets[ids[0]]] if len(ids) else None,'Target akhir':times[targets[ids[-1]]] if len(ids) else None,
        'SO2 target minimum':float(valid.min()) if len(valid) else None,'SO2 target median':float(np.median(valid)) if len(valid) else None,
        'SO2 target maksimum':float(valid.max()) if len(valid) else None,
        'Target melintasi validasi':int((mask&~inside).sum()),'Target di atas maksimum final':int((mask&high).sum())}

for method in methods:
    sequence=[('01 Seluruh slot origin train',np.ones(a,bool)),
        ('02 Target t+1 asli teramati',observed),
        ('03 Setelah lag struktural tersedia',observed&structural),
        ('04 Setelah kalibrasi 90 hari',observed&after_warmup),
        ('05 Setelah buffer lag 48 slot',observed&after_buffer),
        ('06 Setelah imputasi tanpa menyaring NaN',observed&after_buffer),
        ('07 Setelah fitur masih kosong diperiksa',observed&after_buffer&finite[method]),
        ('08 Setelah purge target validasi',eligible[method]),
        ('09 Layak final mandiri belum irisan',eligible[method]),
        ('10 Aktual dilatih setelah irisan bersama',common)]
    prev=None
    for stage,mask in sequence:
        stages[(method,stage)]=mask
        row=stats(mask,Pipeline=method,Tahap=stage)
        row['Keluar dari tahap sebelumnya']=int((prev&~mask).sum()) if prev is not None else 0
        tables['Ringkasan'].append(row)
        for month in times[targets].to_period('M').unique():
            selected=mask & (times[targets].to_period('M')==month)
            monthly.append(stats(selected,Pipeline=method,Tahap=stage,Bulan_target=str(month)))
        high_stage.append(stats(mask&high,Pipeline=method,Tahap=stage,Ambang_deskriptif=threshold))
        prev=mask
tables['Tahap_per_bulan']=monthly;tables['SO2_tinggi_per_tahap']=high_stage
# Inventaris murni target pada waktu train, sebelum pasangan origin dibentuk.
tables['Inventaris_train_asli']=[]
for month,mask in [('seluruh train',np.ones(a,bool))]+[(str(m),times[:a].to_period('M')==m) for m in times[:a].to_period('M').unique()]:
    values=y[:a][mask];v=values[np.isfinite(values)]
    tables['Inventaris_train_asli'].append({'Bulan':month,'Slot train':int(mask.sum()),'SO2 asli teramati':len(v),
        'Minimum':v.min() if len(v) else None,'Median':np.median(v) if len(v) else None,'Maksimum':v.max() if len(v) else None,
        'Di atas maksimum final':int((v>threshold).sum())})

# Penyebab sisa NaN pada matriks rekontruksi, tanpa menjalankan imputer kembali.
last=np.maximum.accumulate(np.where(np.isfinite(raw[:a]),origins[:,None],-1),axis=0)
age=origins[:,None]-last
remaining={name:~np.isfinite(matrix) for name,matrix in hist.items()}
cov_known=np.isfinite(raw[:a,1:]).sum(axis=1)
cause={name:np.full((a,12),'',object) for name in methods}
for name in methods:
    r=remaining[name]
    cause[name][r & (~after_warmup[:,None])]='KALIBRASI_BELUM_IMPUTASI'
    active=r&after_warmup[:,None]
    if name=='P1_ffill_L4':
        cause[name][active&(last<0)]='P1_BELUM_ADA_DONOR'
        cause[name][active&(last>=0)&(age>4)]='P1_UMUR_DONOR_LEBIH_4'
    elif name=='P2_knn_k10':
        cause[name][active&(cov_known[:,None]<2)]='P2_GUARD_KOVARIAT_KURANG_2'
    unknown=active&(cause[name]=='')
    # Jika ada kasus ini, laporkan sebagai tidak terjelaskan; tidak menebak penyebab.
    cause[name][unknown]='SISA_NAN_PERLU_TELUSUR_LANJUT'

def feature_causes(method,t):
    result=set()
    for j in np.flatnonzero(~np.isfinite(features[method][t,:18])):
        if j<7:
            u=t-cfg['so2_lags'][j];col=0
        else:u=t;col=j-6
        if u<0:result.add('HISTORI_DI_LUAR_CAKUPAN')
        else:result.add(cause[method][u,col] or 'SISA_NAN_PERLU_TELUSUR_LANJUT')
    return sorted(result)

all_flags=[];ledger=[];overlap=[];combinations=[];var_drops=[]
base_flags={'TARGET_KOSONG':~observed,'LAG_STRUKTURAL_BELUM_ADA':~structural,
    'KALIBRASI_90_HARI':~after_warmup,'BUFFER_LAG_SETELAH_KALIBRASI':after_warmup&~after_buffer,
    'TARGET_MASUK_VALIDASI':~inside}
for t in origins:
    original=[]
    if missing_original[t,:7].any():original.append('SO2_HISTORIS_ASLI_KOSONG')
    if missing_original[t,7:].any():original.append('KOVARIAT_ASLI_KOSONG')
    base=[key for key,value in base_flags.items() if value[t]]
    row={'Origin':times[t],'Timestamp target':times[t+1],'SO2 target asli':target_y[t],
        'Alasan dasar':'; '.join(base),'Penanda missing asli':'; '.join(original)}
    full={'Index origin':int(t),**row,'Target di atas maksimum final':bool(high[t]),'Aktual train bersama':bool(common[t])}
    for name in methods:
        why=feature_causes(name,t)
        fatal=list(base)
        if not finite[name][t]:fatal += ['FITUR_MASIH_KOSONG']+why
        if eligible[name][t] and not common[t]:fatal.append('LAYAK_MANDIRI_KELUAR_IRISAN')
        row[name+' alasan']='; '.join(sorted(set(fatal))) if fatal else 'DIPAKAI_TRAIN_BERSAMA'
        full[name+' layak mandiri']=bool(eligible[name][t]);full[name+' fitur finite']=bool(finite[name][t])
        full[name+' SO2 post masih kosong']=bool((~np.isfinite(features[name][t,:7])).any())
        full[name+' kovariat post masih kosong']=bool((~np.isfinite(features[name][t,7:18])).any())
        full[name+' sebab sisa NaN']='; '.join(why)
    row['Aktual train bersama']=bool(common[t]);ledger.append(row);all_flags.append(full)
tables['Penanda_semua_timestamp']=ledger
pd.DataFrame(all_flags).to_csv(SUPPORT/'penanda_lengkap_per_timestamp.csv',index=False)

scope=observed&after_buffer&inside
for method in methods:
    flagmap={**base_flags,'SO2_HISTORIS_ASLI_KOSONG':missing_original[:,:7].any(axis=1),
        'KOVARIAT_ASLI_KOSONG':missing_original[:,7:].any(axis=1),
        'SO2_MASIH_KOSONG_SETELAH_IMPUTASI':(~np.isfinite(features[method][:,:7])).any(axis=1),
        'KOVARIAT_MASIH_KOSONG_SETELAH_IMPUTASI':(~np.isfinite(features[method][:,7:18])).any(axis=1),
        'FITUR_MASIH_KOSONG':~finite[method],
        'LAYAK_MANDIRI_KELUAR_IRISAN':eligible[method]&~common}
    for key,mask in flagmap.items():
        overlap.append({'Pipeline':method,'Penanda':key,'Jumlah semua origin train':int(mask.sum()),
            'Jumlah dalam scope pasca buffer target train asli':int((mask&scope).sum()),'Denominator scope':int(scope.sum()),
            'Jumlah target tinggi dalam scope':int((mask&scope&high).sum()),
            'Catatan':'Penanda boleh tumpang tindih; missing asli bukan otomatis alasan keluar.'})
    causes_list=[feature_causes(method,int(t)) for t in origins]
    for key in sorted({v for row in causes_list for v in row}):
        mask=np.array([key in row for row in causes_list])
        overlap.append({'Pipeline':method,'Penanda':key,'Jumlah semua origin train':int(mask.sum()),
            'Jumlah dalam scope pasca buffer target train asli':int((mask&scope).sum()),'Denominator scope':int(scope.sum()),
            'Jumlah target tinggi dalam scope':int((mask&scope&high).sum()),'Catatan':'Penyebab sisa NaN pada setidaknya satu fitur; bisa overlap.'})
    combo=pd.Series([('; '.join(causes_list[i]) or 'tidak ada sisa NaN') for i in origins[scope]])
    for text,count in combo.value_counts().items():combinations.append({'Pipeline':method,'Kombinasi sebab sisa NaN':text,'Jumlah sampel':count,'Denominator scope':int(scope.sum())})
    for j,col in enumerate(feature_names[:18]):
        mask=~np.isfinite(features[method][:,j])
        var_drops.append({'Pipeline':method,'Fitur':col,'Sisa NaN dalam scope':int((mask&scope).sum()),
            'Sisa NaN target tinggi dalam scope':int((mask&scope&high).sum()),
            'Denominator target tinggi scope':int((scope&high).sum()),'Denominator scope':int(scope.sum())})
tables['Alasan_overlap']=overlap;tables['Kombinasi_alasan']=combinations;tables['Fitur_penyebab_keluar']=var_drops
tables['Kehilangan_karena_irisan']=[]
for name in methods:
    lost=eligible[name]&~common
    tables['Kehilangan_karena_irisan'].append({**stats(lost,Pipeline=name),
        'Layak sendiri total':int(eligible[name].sum()),'Aktual dipakai bersama':int(common.sum()),
        'Keterangan':'Kelayakan kontrafaktual berdasarkan fitur tersimpan; tidak melatih model mandiri baru.'})
    for mo in times[targets].to_period('M').unique():
        tables.setdefault('Irisan_hilang_per_bulan',[]).append(stats(lost&(times[targets].to_period('M')==mo),Pipeline=name,Bulan_target=str(mo)))

# Rentetan waktu observasi tinggi asli train; ambang hanya maksimum kohort final.
source_high=np.isfinite(y[:a])&(y[:a]>threshold);periods=[];i=0
while i<a:
    if not source_high[i]:i+=1;continue
    first=i
    while i<a and source_high[i]:i+=1
    periods.append({'Periode':len(periods)+1,'Awal target tinggi':times[first],'Akhir target tinggi':times[i-1],
        'Jumlah observasi':i-first,'Minimum SO2':float(y[first:i].min()),'Median SO2':float(np.median(y[first:i])),
        'Maksimum SO2':float(y[first:i].max()),'Ambang deskriptif maksimum train final':threshold})
tables['Periode_SO2_lebih_tinggi']=periods
for period in periods:
    inperiod=(times[targets]>=period['Awal target tinggi'])&(times[targets]<=period['Akhir target tinggi'])
    for (method,stage),mask in stages.items():
        tables.setdefault('Periode_tinggi_per_tahap',[]).append({'Periode':period['Periode'],'Pipeline':method,'Tahap':stage,
            'Observasi asli periode':period['Jumlah observasi'],'Bertahan':int((mask&inperiod&high).sum()),
            'Keluar atau belum masuk':period['Jumlah observasi']-int((mask&inperiod&high).sum())})
# Rentetan missing curah hujan asli, tanpa pengisian.
rain_bad=~np.isfinite(d['Curah Hujan numerik'].to_numpy()[:a]);i=0
while i<a:
    if not rain_bad[i]:i+=1;continue
    first=i
    while i<a and rain_bad[i]:i+=1
    if i-first>=48:
        tables.setdefault('Gap_Curah_Hujan_train',[]).append({'Awal pengukuran asli kosong':times[first],
            'Akhir pengukuran asli kosong':times[i-1],'Jumlah slot':i-first,'Durasi slot jam':(i-first)*.5})

# Dua kegagalan cakupan KNN; tidak menjalankan transform atau prediksi.
pred=pd.read_csv(OLD/'prediksi_validasi.csv',parse_dates=['Origin','Timestamp target'])
val_features={name:pd.read_csv(OLD/f'fitur_{name}.csv').iloc[:,1:] for name in methods}
tables['KNN_tidak_ada_prediksi']=[];tables['KNN_fitur_kosong']=[]
for _,r in pred[pred.P2_knn_k10.isna()].iterrows():
    u=int(r['Index origin']);original=raw[u];known=np.flatnonzero(np.isfinite(original[1:]));cols=val_features['P2_knn_k10'].columns[:18]
    empty_cols=[col for col in cols if not np.isfinite(val_features['P2_knn_k10'].iloc[u][col])]
    tables['KNN_tidak_ada_prediksi'].append({'Origin':r.Origin,'Target':r['Timestamp target'],'SO2 target asli':r['SO2 target asli'],
        'SO2 origin asli':y[u],'Timestamp kovariat yang dicari':times[u-1],
        'Jumlah kovariat asli tersedia':len(known),'Kovariat asli tersedia':', '.join(cfg['covariates'][j] for j in known),
        'Fitur Ridge masih kosong':', '.join(empty_cols),'Prediksi median':r.P0_median,
        'Penyebab':'Guard KNN minimum dua kovariat asli tidak lolos; kovariat tidak diisi; pemeriksaan fitur menolak origin.'})
    assert len(known)<2
    for col in empty_cols:
        tables['KNN_fitur_kosong'].append({'Origin':r.Origin,'Target':r['Timestamp target'],'Fitur':col,
            'Nilai fitur median':val_features['P0_median'].iloc[u][col],
            'Nilai fitur KNN':None,'Keterangan':'Tidak ada fallback ke median pada KNN.'})
last=np.maximum.accumulate(np.where(np.isfinite(y[:b-1]),np.arange(b-1),-1))
oi=pred['Index origin'].to_numpy(dtype=int);donor=last[oi];computed_age=oi-donor
assert np.array_equal(computed_age,pred['Persistence umur donor slot'].to_numpy())
assert np.allclose(y[donor],pred.Persistence)
tables['Usia_persistence']=[];tables['Persistence_donor_lama']=[]
for label,mask in [('seluruh target asli validasi',np.ones(len(pred),bool)),('irisan semua metode',pred['Irisan semua metode'].to_numpy())]:
    for age,count in pd.Series(computed_age[mask]).value_counts().sort_index().items():
        tables['Usia_persistence'].append({'Domain':label,'Usia slot':age,'Usia menit':age*30,'Jumlah prediksi':count,
            'Denominator prediksi domain':int(mask.sum()),'Proporsi persen':100*count/int(mask.sum())})
for j in np.flatnonzero(computed_age>0):
    r=pred.iloc[j];tables['Persistence_donor_lama'].append({'Origin':r.Origin,'Target':r['Timestamp target'],
        'Waktu donor asli':times[donor[j]],'SO2 donor asli':y[donor[j]],'Usia slot':computed_age[j],
        'Usia menit':computed_age[j]*30,'Dalam irisan':bool(r['Irisan semua metode'])})
tables['Verifikasi_skor_timestamp']=[]
shared=pred['Irisan semua metode'].to_numpy()
for r in prior['Ringkasan']:
    keep=shared if r['Domain']=='irisan semua metode' else np.isfinite(pred[r['Metode']]).to_numpy()
    errors=pred.loc[keep,r['Metode']]-pred.loc[keep,'SO2 target asli']
    for col,value in [('MAE',errors.abs().mean()),('RMSE',np.sqrt(np.mean(errors**2))),('Bias',errors.mean())]:
        assert np.isclose(value,r[col],rtol=1e-10,atol=1e-10)
    target_hash=sha256('\n'.join(pred.loc[keep,'Timestamp target'].astype(str)).encode()).hexdigest()
    tables['Verifikasi_skor_timestamp'].append({'Metode':r['Metode'],'Domain':r['Domain'],'Jumlah target asli':int(keep.sum()),
        'SHA256 daftar timestamp target':target_hash,'MAE hitung ulang':errors.abs().mean(),
        'RMSE hitung ulang':np.sqrt(np.mean(errors**2)),'Bias hitung ulang':errors.mean(),'Sesuai skor lama':True})
assert len({r['SHA256 daftar timestamp target'] for r in tables['Verifikasi_skor_timestamp'] if r['Domain']=='irisan semua metode'})==1
assert np.isclose(threshold,2.14) and int(common.sum())==12654
assert int(source_high.sum())==5289 and int((high&after_buffer).sum())==3841

# Pemeriksaan statis: dropna hanya statistik deskriptif, bukan penyusunan fitur.
code=(ROOT/'scripts/forecast_ridge_way_urang.py').read_text(encoding='utf-8')
tables['Pemeriksaan_implementasi']=[
 {'Pemeriksaan':'dropna terlalu luas','Hasil':'Tidak ditemukan pada penyusunan sampel. Satu g.dropna() hanya statistik SO2 per bulan/split; tidak mengubah data/fitur.'},
 {'Pemeriksaan':'Sampel dikeluarkan sebelum imputasi','Hasil':'Tidak ada filter complete-case sebelum imputasi. Kalibrasi 90 hari/buffer 48 slot membatasi kelayakan model; missing warmup tidak diisi dari masa depan. Guard minimum dua kovariat KNN memang mencegah transform query tanpa dukungan, lalu sisa NaN ditolak sebagai fitur Ridge.'},
 {'Pemeriksaan':'Pemeriksaan fitur','Hasil':'np.isfinite(X[train_orig]).all(axis=1) tepat 36 fitur (18 numerik,18 indikator). Kolom metadata dan kolom bukan fitur tidak ikut.'},
 {'Pemeriksaan':'Urutan kode purge','Hasil':'Kode menetapkan rentang origin sampai a-2 sekaligus sebelum menghitung finite; trace menampilkan finite lalu purge untuk memisahkan efek. Operator mask komutatif, himpunan akhir diverifikasi sama.'},
 {'Pemeriksaan':'Imputasi komputasional gagal','Hasil':'Notebook tersimpan tidak memiliki error fit/transform. Sisa NaN berasal dari pembatasan/guard; label tidak terjelaskan diperiksa terpisah.'},
 {'Pemeriksaan':'Train pipeline mandiri','Hasil':'Tahap 5 benar-benar melatih semua Ridge pada irisan, bukan pada himpunan layak mandiri. Jumlah mandiri pada trace adalah eligibility tanpa fit ulang.'},
 {'Pemeriksaan':'Konsentrasi dipakai sebagai filter','Hasil':'Tidak ada. Ambang lebih tinggi hanya memakai maksimum target train final untuk penelusuran ini, bukan batas validitas sensor.'}]
assert code.count('.dropna(')==1
nb=json.loads((ROOT/'notebooks/05_forecasting_acuan_way_urang.ipynb').read_text(encoding='utf-8'))
assert not any(o['output_type']=='error' for c in nb['cells'] for o in c.get('outputs',[]))
# Simulasi eligibility mengabaikan kolom curah hujan, bukan perubahan pipeline/model.
remove={'Curah Hujan_lag1','missing_asli_Curah Hujan_lag1'}
kept=[j for j,col in enumerate(feature_names) if col not in remove]
hypo=scope.copy()
for name in methods:hypo &= np.isfinite(features[name][:,kept]).all(axis=1)
assert int(hypo.sum())==16530 and int((hypo&high).sum())==3786
tables['Simulasi_eligibility_tanpa_rain']=[stats(hypo,Skenario='Hanya abaikan Curah Hujan pada pemeriksaan fitur',
    Catatan='Tidak melatih/imputasi ulang, tidak mengubah fitur/hasil lama, bukan skor baru.')]
tables['Definisi']=[{'Istilah':k,'Definisi':v} for k,v in [
    ('Semesta trace','Satu baris per origin pada train, target=t+30 menit. Sebelum purge, satu target (origin train terakhir) masuk awal validasi. Inventaris_train_asli memakai timestamp SO2 train murni.'),
    ('19.197 versus 19.198','Train murni mempunyai 19.197 nilai SO2. Pasangan origin train mencakup 19.198 target teramati: target pertama train yang kosong tidak memiliki origin train, dan target pertama validasi yang teramati muncul sebelum purge.'),
    ('Jumlah sampel dan statistik','Jumlah baris pada himpunan tahap, termasuk target kosong pada tahap 01. Minimum/median/maksimum hanya target asli teramati. Tanggal bulanan dikelompokkan berdasarkan timestamp target.'),
    ('Setelah lag struktural','Origin>=48 agar semua posisi lag ada dalam cakupan. Missing nilai SO2 pada posisi lag tidak dibuang di tahap ini; berkesempatan diisi.'),
    ('Setelah imputasi','Tahap ini tidak membuang baris, bahkan bila masih NaN. Baris baru ditolak pada pemeriksaan fitur finite, setelah metode memperoleh kesempatan sesuai kontraknya.'),
    ('Kalibrasi dan buffer','Origin<4320 tidak dapat menjadi contoh Ridge; 4320–4367 menunggu lag maksimum berasal dari histori yang sudah diproses. Ini aturan rancangan, bukan complete-case otomatis.'),
    ('Layak mandiri','Memenuhi target, warmup/buffer, 36 fitur finite dan purge untuk pipeline itu. Bukan model baru yang dilatih.'),
    ('Aktual bersama','Irisan ketiga himpunan layak mandiri, sama dengan ledger train tahap 5. Semua Ridge lama memakai himpunan ini.'),
    ('Alasan overlap','Penanda target, histori, kovariat, guard, dan irisan dapat tumpang tindih. Jangan menjumlah tabel Alasan_overlap untuk mendapatkan total keluar; waterfall Keluar dari tahap sebelumnya bersifat terurut.'),
    ('Scope alasan','Origin>=4368, target asli finite dan timestamp target masih train. Mengisolasi sisa missing dari pengecualian awal dan purge.'),
    ('Rentang lebih tinggi','SO2 asli train > maksimum target pada train bersama lama (2.14 µg/m³). Deskriptif, bukan batas sensor/outlier atau alasan menghapus nilai.'),
    ('Kegagalan versus kebijakan','P1 melebihi umur donor/tidak ada donor, serta guard minimum dua kovariat KNN adalah penolakan sesuai kebijakan. Tidak disamakan dengan exception algoritma.'),
    ('Penanda asli','SO2_HISTORIS_ASLI_KOSONG/KOVARIAT_ASLI_KOSONG hanya informasi missing asal. Nilai dapat sudah diisi dan tetap dipakai; bukan alasan fatal sendiri.'),
    ('Cakupan validasi','Jumlah prediksi finite / 4524 target validasi asli teramati. Tidak ada imputasi target. Perbandingan bersama memakai timestamp identik untuk keempat metode.'),
    ('Usia persistence','Origin dikurangi timestamp donor SO2 asli, satu slot=30 menit. Proporsi = jumlah usia tertentu / prediksi pada domain yang disebutkan ×100%.'),
    ('Test','Tidak dimuat nilainya, tidak dievaluasi. Artefak tahap 5 hanya dibaca dan diverifikasi hash sebelum/sesudah.'),
    ('Simulasi tanpa rain','Hanya menghitung kelayakan memakai histori tersimpan dengan dua kolom terkait Curah Hujan diabaikan pada pengecekan. Bukan refit imputer/Ridge dan bukan koreksi skor lama.')]]
tables['Integritas_SHA256']=[]
for p in refs:
    key=str(p.relative_to(ROOT));after=digest(p);assert before[key]==after
    tables['Integritas_SHA256'].append({'File':key,'SHA256 sebelum':before[key],'SHA256 sesudah':after,'Identik':True})
dump(OUT/'penelusuran_tahap5_tables.json',tables)
from write_stage5_trace_report import write_report
write_report(tables,threshold)
print('Selesai: trace 21.119 origin; ledger lama cocok; seluruh artefak tahap 5 dan sumber identik. Tidak ada model dijalankan.')
