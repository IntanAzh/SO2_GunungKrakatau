"""Read-only diagnosis of stored Fold_01. No estimator fit, no test access."""
from pathlib import Path
from datetime import datetime
from hashlib import sha256
import json,os,subprocess
import numpy as np
import pandas as pd
import openpyxl
from pilot_way_urang import dump
from audit_quality import hashfile,stamp
from write_pilot_report import table
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OLD=OUT/'backtesting_way_urang';F=OLD/'Fold_01'
RUN=OUT/'diagnosis_fold1';FIG=OUT/'figures/diagnosis_fold1';RUN.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
METHODS=['P0_median','P1_ffill_L4','P2_knn_k10'];ALL=METHODS+['Persistence']

def stats(v):
    v=np.asarray(v,dtype=float);v=v[np.isfinite(v)]
    if not len(v):return {'N':0}
    q=np.quantile(v,[0,.01,.05,.25,.5,.75,.95,.99,1])
    return dict(zip(['Minimum','P1','P5','P25','Median','P75','P95','P99','Maksimum'],q),N=len(v),Mean=float(v.mean()),SD=float(v.std(ddof=1)) if len(v)>1 else None)

def metrics(frame,method,mask,label,domain,den):
    e=frame.loc[mask,method].to_numpy()-frame.loc[mask,'SO2 target asli'].to_numpy()
    return {'Periode':label,'Pipeline':method,'Domain':domain,'N':len(e),'Denominator target asli':int(den),
        'Cakupan persen':100*len(e)/den if den else None,'MAE':float(np.mean(np.abs(e))) if len(e) else None,
        'RMSE':float(np.sqrt(np.mean(e**2))) if len(e) else None,'Bias':float(np.mean(e)) if len(e) else None,
        'SHA256 timestamp':sha256('\n'.join(frame.loc[mask,'Timestamp target'].astype(str)).encode()).hexdigest()}

def main():
    refs=[p for p in OUT.rglob('*') if p.is_file() and not any(s.startswith('.') for s in p.relative_to(OUT).parts)
        and 'diagnosis_fold1' not in p.relative_to(OUT).parts and not p.name.startswith(('diagnosis_fold1','tabel_diagnosis_fold1')) and not p.name.endswith('.inspect.ndjson')]
    refs+=list((ROOT/'data_raw').glob('*.xlsx'))+list((ROOT/'notebooks').glob('*.ipynb'))
    refs+= [ROOT/'scripts/backtesting_stage6.py',ROOT/'scripts/forecast_ridge_way_urang.py']
    before={str(p.relative_to(ROOT)):hashfile(p) for p in refs}
    cfg=json.loads((OLD/'fold_config_locked.json').read_text(encoding='utf-8'));a,b=cfg['fold_cuts'][:2]
    saved=json.loads((OUT/'tabel_backtesting_preprocessing.json').read_text(encoding='utf-8'))
    d=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=b,parse_dates=['Timestamp'])
    times=pd.DatetimeIndex(d.Timestamp);y=d['SO2 numerik'].to_numpy(dtype=float)
    assert times.max()<pd.Timestamp(cfg['split'][2]['Awal'])
    p=pd.read_csv(F/'prediksi_validasi.csv',parse_dates=['Origin','Timestamp target'])
    ids=pd.read_csv(F/'timestamp_train_bersama.csv')['Index origin'].to_numpy(dtype=int)
    assert np.array_equal(p['Index target'],np.flatnonzero(np.isfinite(y[a:b]))+a)
    origins=p['Index origin'].to_numpy();targets=p['Index target'].to_numpy()
    assert np.array_equal(origins+1,targets) and np.array_equal(times[origins].values,p.Origin.values)
    assert np.array_equal(times[targets].values,p['Timestamp target'].values) and np.allclose(y[targets],p['SO2 target asli'])
    assert (p['Timestamp target']-p.Origin).eq(pd.Timedelta(minutes=30)).all() and (ids+1<a).all()
    common=p['Irisan semua metode'].to_numpy();assert np.array_equal(common,np.isfinite(p[ALL]).all(axis=1))
    p['SO2 asli origin']=y[origins]
    # Pick the most negative one-slot change of original observations, no model-score criterion.
    changes=np.diff(y);eligible=np.isfinite(y[1:])&np.isfinite(y[:-1])&(np.arange(1,b)>=a)
    transition_idx=int(np.argmin(np.where(eligible,changes,np.inf)))+1;transition=times[transition_idx]
    rule={'Aturan':'Penurunan SO2 asli terbesar antara dua slot berurutan30menit pada validasi fold1; keduanya finite; ikatan dipilih waktu paling awal. Tidak memakai residual/prediksi.',
        'Timestamp sebelum':times[transition_idx-1],'SO2 sebelum':y[transition_idx-1],
        'Timestamp transisi':transition,'SO2 sesudah':y[transition_idx],'Selisih SO2':y[transition_idx]-y[transition_idx-1],
        'Jendela grafik jam sebelum sesudah':72,'Origin rendah penelusuran':'SO2 asli origin <=2.14, hanya penelusur tahap5, bukan ambang kualitas/bahaya.'}
    dump(RUN/'aturan_diagnosis.json',rule)
    checks=[]
    def check(name,value,detail):
        assert bool(value),name
        checks.append({'Pemeriksaan':name,'Lulus':True,'Rincian':str(detail)})
    check('Origin dan target tepat30menit',True,f'{len(p)} target asli; indeks grid, timestamp dan nilai cocok')
    source=next((ROOT/'data_raw').glob('*.xlsx'));wb=openpyxl.load_workbook(source,read_only=True,data_only=True)
    ws=wb['Kab. Lampung Selatan Way Urang'];headers=list(next(ws.iter_rows(min_row=8,max_row=8,min_col=2,max_col=15,values_only=True)))
    units=list(next(ws.iter_rows(min_row=9,max_row=9,min_col=2,max_col=15,values_only=True)))
    source_records=[];source_lookup={};offgrid=[]
    # Source is chronological as audited. Stop in August2025, well before final test.
    for rownum,row in enumerate(ws.iter_rows(min_row=10,max_col=15,values_only=True),10):
        ts=stamp(row[1])
        if pd.isna(ts):continue
        if ts>times[-1]:break
        rec=dict(zip(headers,row[1:15]));rec['Baris Excel']=rownum
        if ts not in times:offgrid.append({'Timestamp':ts,'SO2 asli':rec['SO2'],'Baris Excel':rownum});continue
        source_lookup.setdefault(ts,[]).append(rec)
    wb.close()
    compare=[]
    for variable in ['SO2']+cfg['covariates']:
        count=0;missingrow=0;mismatch=0;duplicates=0
        for i,ts in enumerate(times):
            rr=source_lookup.get(ts,[])
            if not rr:missingrow+=1;assert not np.isfinite(d[f'{variable} numerik'].iloc[i]);continue
            if len(rr)>1:duplicates+=len(rr)-1
            vv=pd.to_numeric(pd.Series([r[variable] for r in rr]),errors='coerce').to_numpy(dtype=float)
            expected=d[f'{variable} numerik'].iloc[i]
            good=np.allclose(vv,expected,equal_nan=True)
            mismatch+=not good;count+=1
        compare.append({'Variabel':variable,'Slot grid':b,'Slot ada baris Excel':count,'Slot tanpa baris':missingrow,
            'Baris tambahan duplikat':duplicates,'Ketidaksesuaian nilai':mismatch,'Satuan literal Excel':units[headers.index(variable)]})
        check('Sumber langsung '+variable,mismatch==0,f'{count} slot ada baris, {missingrow} tanpa baris; tidak ada nilai berbeda')
    selected=set([ids[0],ids[len(ids)//2],ids[-1],a-1,a,a+1,transition_idx-1,transition_idx,transition_idx+1,transition_idx+48,b-2])
    for mo,frame in p.groupby(p['Timestamp target'].dt.to_period('M')):selected.add(int(frame.iloc[len(frame)//2]['Index origin']))
    for origin in sorted(selected):
        if origin+1>=b:continue
        for role,index in [('origin',origin),('target',origin+1)]:
            for rec in source_lookup.get(times[index],[]):
                source_records.append({'Index origin sampel':int(origin),'Peran':role,'Timestamp grid':times[index],
                    'Baris Excel':rec['Baris Excel'],'Sel waktu':f'B{rec["Baris Excel"]}','Waktu literal Excel':str(rec['Waktu']),
                    'Sel SO2':f'E{rec["Baris Excel"]}','SO2 literal Excel':rec['SO2'],'SO2 grid audit':y[index]})
    coefs=[];feature_stats=[];model_stats=[];decomposition=[];corrs=[];Xall={};zs={};contribs={};groups={
        'SO2 historis':slice(0,7),'Polutan lain':slice(7,12),'Meteorologi':slice(12,17),
        'Indikator SO2 missing':slice(17,24),'Indikator kovariat missing':slice(24,34)}
    reconstructed=[]
    for method in METHODS:
        feature=pd.read_csv(F/f'fitur_{method}.csv',parse_dates=['Origin']);names=feature.columns[1:].tolist();X=feature.iloc[:,1:].to_numpy(dtype=float);Xall[method]=X
        h=pd.read_csv(F/f'histori_kausal_{method}.csv',parse_dates=['Origin'])
        c=pd.DataFrame([r for r in saved['Koefisien_Ridge'] if r['Fold']=='Fold_01' and r['Pipeline']==method])
        check('Urutan fitur fit-predict '+method,names==c.Fitur.tolist(),'; '.join(names))
        check('Indeks fitur utuh '+method,np.array_equal(feature.Origin.values,times[:b-1].values),f'{b-1} origin berurutan, tidak dropna/reindex posisi')
        check('lag0 tepat origin '+method,np.allclose(X[:,0],h.SO2,equal_nan=True),'lag0=histori tersimpan pada timestamp origin; nilai asli tetap jika teramati')
        seen=np.isfinite(y[:b-1]);check('lag0 asli tidak bergeser '+method,np.allclose(X[seen,0],y[:b-1][seen]),f'{seen.sum()} origin teramati diperiksa')
        for j,lag in enumerate(cfg['so2_lags']):
            check(f'lag{lag} posisi '+method,np.allclose(X[:,j],h.SO2.shift(lag),equal_nan=True),'Semua origin dibandingkan terhadap shift histori berbasis grid utuh')
        for j,covar in enumerate(cfg['covariates'],7):
            original=d[f'{covar} numerik'].shift(1).to_numpy()[:b-1];obs=np.isfinite(original)
            check('Kovariat lag1 '+method+' '+covar,np.allclose(X[obs,j],original[obs]),'Nilai asli waktu origin−30menit cocok')
        xt=X[ids];mean=xt.mean(axis=0);sd=xt.std(axis=0);sd[sd<1e-12]=1
        check('Scaler hanya train bersama '+method,np.allclose(mean,c['Mean fitur train']) and np.allclose(sd,c['Skala fitur train']),f'N={len(ids)}, batas target train <validasi')
        z=(X[origins]-c['Mean fitur train'].to_numpy())/c['Skala fitur train'].to_numpy();beta=c['Koefisien fitur standar'].to_numpy();intercept=float(c.Intercept.iloc[0])
        con=z*beta;good=np.isfinite(X[origins]).all(axis=1);pred=intercept+con.sum(axis=1)
        maxerr=float(np.max(np.abs(pred[good]-p.loc[good,method])))
        check('Rekonstruksi prediksi '+method,np.allclose(pred[good],p.loc[good,method],atol=1e-8,rtol=1e-10),f'Maksimum galat numerik={maxerr:.3g}; tanpa transform target/inverse')
        zt=(xt-mean)/sd;target=y[ids+1];res=intercept+zt@beta-target
        gradient=zt.T@res+cfg['ridge']['alpha']*beta
        check('Persamaan penalti Ridge '+method,np.max(np.abs(gradient))<1e-6 and abs(res.sum())<1e-6,'Xstandarᵀ(pred−y)+alpha×beta≈0, sum(pred−y)≈0; tanpa fit model baru')
        zs[method]=z;contribs[method]=con
        row={'Pipeline':method,'Alpha':cfg['ridge']['alpha'],'Intercept':intercept,'Target scaling':False,'Train N':len(ids),
            'Train target mean':float(target.mean()),'Train MAE tersimpan direkonstruksi':float(np.abs(res).mean()),
            'Train R2 deskriptif':float(1-np.sum(res**2)/np.sum((target-target.mean())**2)),
            'Jumlah slope SO2 satuan asli':float(np.sum(beta[:7]/sd[:7])),
            'L2 penalty alpha sum beta2':float(cfg['ridge']['alpha']*np.sum(beta**2)),
            'Maks galat prediksi rekonstruksi':maxerr,'Maks residual persamaan penalti':float(np.max(np.abs(gradient)))}
        model_stats.append(row)
        for j,name in enumerate(names):
            traincol=xt[:,j];valcol=X[origins[common],j];v=valcol[np.isfinite(valcol)]
            coefs.append({'Pipeline':method,'Urutan fitur':j,'Fitur':name,'Alpha L2':cfg['ridge']['alpha'],
                'Koefisien standar':float(beta[j]),'Koefisien satuan asli':float(beta[j]/sd[j]),
                'Mean fitur train':float(mean[j]),'Scale fitur train':float(sd[j]),'Intercept':intercept,
                'Dikenai penalti':True,'Kelompok':next(k for k,s in groups.items() if j in range(*s.indices(34)))})
            feature_stats.append({'Pipeline':method,'Fitur':name,'Train N':len(ids),'Train minimum':float(traincol.min()),
                'Train median':float(np.median(traincol)),'Train maksimum':float(traincol.max()),'Train SD':float(traincol.std()),
                'Train nilai unik':len(np.unique(traincol)),'Konstan persis train':bool(np.ptp(traincol)==0),
                'Train SD dibagi abs mean persen':100*float(traincol.std())/abs(float(traincol.mean())) if traincol.mean()!=0 else None,
                'Validasi bersama N':len(v),'Validasi minimum':float(v.min()),'Validasi median':float(np.median(v)),
                'Validasi maksimum':float(v.max()),'Di luar minmax train':int(((v<traincol.min())|(v>traincol.max())).sum()),
                'Di luar minmax train persen':100*int(((v<traincol.min())|(v>traincol.max())).sum())/len(v),
                'Abs z lebih5':int((np.abs(z[common,j])>5).sum())})
            if j<17:
                corrs.append({'Pipeline':method,'Fitur':name,'Korelasi Pearson train vs target':float(np.corrcoef(traincol,target)[0,1]) if np.std(traincol)>0 else None})
        for k,origin in enumerate(origins):
            if not good[k]:continue
            decomposition.append({'Pipeline':method,'Origin':p.Origin.iloc[k],'Timestamp target':p['Timestamp target'].iloc[k],
                'Irisan bersama':bool(common[k]),'SO2 origin asli':y[origin],'SO2 lag0 setelah imputasi':X[origin,0],
                'SO2 target asli':y[origin+1],'Prediksi':float(pred[k]),'Intercept':intercept,
                **{name:float(con[k,sl].sum()) for name,sl in groups.items()},
                'Kontribusi SO2 lag0':float(con[k,0]),'Residual':float(pred[k]-y[origin+1])})
        for i in selected:
            match=np.flatnonzero(origins==i)
            if not len(match):continue
            k=match[0]
            if not good[k]:continue
            for j,name in enumerate(names):
                reconstructed.append({'Pipeline':method,'Origin':times[i],'Fitur':name,'Nilai input':float(X[i,j]),
                    'Mean train':float(mean[j]),'Scale train':float(sd[j]),'z':float(z[k,j]),
                    'Koefisien standar':float(beta[j]),'Kontribusi':float(con[k,j])})
    train_min=float(y[ids+1].min());train_max=float(y[ids+1].max());train_width=train_max-train_min
    residual_stats=[];scoring=[];value_stats=[];low_stats=[];parts=[]
    monthly=p['Timestamp target'].dt.to_period('M').astype(str)
    for method in ALL:
        good=np.isfinite(p[method]).to_numpy()
        for label,within in [('seluruh fold',np.ones(len(p),bool))]+[(mo,monthly.eq(mo).to_numpy()) for mo in monthly.unique()]:
            for domain,valid in [('mandiri',good),('bersama',common)]:
                mask=within&valid;scoring.append(metrics(p,method,mask,label,domain,int(within.sum())))
                err=p.loc[mask,method]-p.loc[mask,'SO2 target asli']
                residual_stats.append({'Periode':label,'Pipeline':method,'Domain':domain,**stats(err),
                    'Abs residual lebih lebar train':int((np.abs(err)>train_width).sum()),'Lebar train acuan':train_width,
                    'Residual positif':int((err>0).sum()),'Residual negatif':int((err<0).sum())})
                value_stats += [{'Periode':label,'Pipeline':method,'Domain':domain,'Variabel':col,**stats(p.loc[mask,col])} for col in ['SO2 target asli',method]]
        low=common&np.isfinite(y[origins])&(y[origins]<=2.14)
        low_stats.append({'Pipeline':method,'Origin rendah asli N':int(low.sum()),'Origin rendah median':float(np.median(y[origins[low]])),
            'Target median':float(p.loc[low,'SO2 target asli'].median()),'Prediksi median':float(p.loc[low,method].median()),
            'Prediksi mean':float(p.loc[low,method].mean()),'Prediksi minimum':float(p.loc[low,method].min()),
            'Prediksi maksimum':float(p.loc[low,method].max()),'Prediksi di atas min target train':int((p.loc[low,method]>=train_min).sum()),
            'Prediksi di atas midpoint train rendah':int((p.loc[low,method]>(train_min+2.14)/2).sum()),
            'MAE':float(np.abs(p.loc[low,method]-p.loc[low,'SO2 target asli']).mean())})
        offsets=(p['Timestamp target']-transition).dt.total_seconds()/3600
        periods=[('sebelum transisi',offsets<0),('0–48 jam',offsets.between(0,48,inclusive='left')),
            ('48 jam–7 hari',offsets.between(48,168,inclusive='left')),('lebih atau sama7hari',offsets>=168)]
        for label,within in periods:
            mask=common&within.to_numpy();err=p.loc[mask,method]-p.loc[mask,'SO2 target asli']
            parts.append({**metrics(p,method,mask,label,'bersama',int(within.sum())),
                'Target median':float(p.loc[mask,'SO2 target asli'].median()),'Prediksi median':float(p.loc[mask,method].median()),
                'Abs residual lebih lebar train':int((np.abs(err)>train_width).sum())})
    last=np.maximum.accumulate(np.where(np.isfinite(y[:b-1]),np.arange(b-1),-1));donors=last[origins]
    check('Persistence donor asli sebelum origin',np.all(donors>=0) and np.all(donors<=origins) and np.allclose(p.Persistence,y[donors]),'Tidak memakai nilai imputasi atau masa depan')
    check('Usia donor persistence',np.array_equal(origins-donors,p['Usia donor persistence slot']),'Beda slot origin−indeks donor')
    check('Timestamp skor bersama',len({r['SHA256 timestamp'] for r in scoring if r['Periode']=='seluruh fold' and r['Domain']=='bersama'})==1,'Keempat metode memiliki digest timestamp yang sama')
    for row in saved['Ringkasan_per_fold']:
        if row['Fold']!='Fold_01':continue
        new=next(r for r in scoring if r['Pipeline']==row['Pipeline'] and r['Periode']=='seluruh fold' and r['Domain']==('bersama' if row['Domain']=='irisan semua metode' else 'mandiri'))
        check('Skor identik hasil lama '+row['Pipeline']+' '+new['Domain'],np.allclose([new[k] for k in ['MAE','RMSE','Bias']],[row[k] for k in ['MAE','RMSE','Bias']]),'Hanya hitung ulang error prediksi tersimpan')
    donor_table=[]
    for label,mask in [('semua target asli',np.ones(len(p),bool)),('irisan bersama',common)]:
        age=origins[mask]-donors[mask]
        for slots,count in zip(*np.unique(age,return_counts=True)):
            donor_table.append({'Domain':label,'Usia slot':int(slots),'Usia jam':float(slots/2),'Jumlah':int(count),
                'Denominator target asli domain':int(mask.sum()),'Persen':100*int(count)/mask.sum()})
    summary=[{'Kategori':'A. Implementasi','Temuan':'Tidak ditemukan kesalahan implementasi pada pemeriksaan yang dijalankan. Alignment, fitur, scaler train-only, urutan kolom, persamaan Ridge, target tanpa scaling, persistence dan timestamp skor cocok.'},
        {'Kategori':'B. Model','Temuan':'Train aktual sempit dan tinggi; intercept dekat mean train, respons efektif lag SO2 lemah dan banyak fitur validasi jauh di luar rentang train. Error menetap sesudah transisi; bukan hanya keterlambatan satu langkah.'},
        {'Kategori':'C. Data','Temuan':'Transisi konsentrasi dan perubahan kovariat ada pada Excel asli; penyebab, status validitas, kode kualitas, latensi dan zona waktu belum dapat dijelaskan tanpa dokumentasi penyedia.'}]
    # Graphs use original gaps; do not connect predictions across missing grid slots.
    graphframe=p.set_index('Timestamp target')[['SO2 target asli','SO2 asli origin']+ALL].reindex(times[a:b])
    for name,lo,hi in [('sekitar_transisi',transition-pd.Timedelta(hours=72),transition+pd.Timedelta(hours=72)),('seluruh_validasi',times[a],times[b-1])]:
        fig,ax=plt.subplots(figsize=(15,5),layout='constrained');g=graphframe.loc[lo:hi]
        for col in g:ax.plot(g.index,g[col],label=col,alpha=.85,linewidth=1 if col in METHODS else 1.5)
        ax.axvline(transition,color='black',ls='--',label='Transisi terpilih');ax.legend(ncol=3,fontsize=8);ax.set(title='Fold1 — '+name.replace('_',' '),ylabel='SO₂ (µg/m³)');ax.grid(alpha=.2)
        fig.savefig(FIG/(name+'.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,4),layout='constrained')
    for method in ALL:
        e=(p.loc[common,method]-p.loc[common,'SO2 target asli']).to_numpy();x=np.sort(e);axes[0].plot(x,np.arange(1,len(x)+1)/len(x),label=method)
        rows=[r for r in scoring if r['Pipeline']==method and r['Domain']=='bersama' and r['Periode']!='seluruh fold']
        axes[1].plot([r['Periode'] for r in rows],[r['MAE'] for r in rows],marker='o',label=method)
    axes[0].set(title='ECDF residual, timestamp bersama',xlabel='Prediksi − pengamatan (µg/m³)',ylabel='Proporsi kumulatif');axes[1].set(title='MAE bulanan',ylabel='µg/m³')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(FIG/'residual_dan_bulanan.png',dpi=150);plt.close(fig)
    integrity=[]
    for path in refs:
        key=str(path.relative_to(ROOT));after=hashfile(path);assert before[key]==after,key
        integrity.append({'File':key,'SHA256 sebelum':before[key],'SHA256 sesudah':after,'Identik':True})
    tables={'Ringkasan':summary,'Pemeriksaan':checks,'Aturan_transisi':[rule],'Skor':scoring,'Residual':residual_stats,
        'Statistik_target_prediksi':value_stats,'Origin_rendah':low_stats,'Sebelum_sesudah':parts,'Model_Ridge':model_stats,
        'Koefisien_scaler':coefs,'Variasi_fitur':feature_stats,'Korelasi_train':corrs,'Kontribusi_per_prediksi':decomposition,
        'Kontribusi_fitur_sampel':reconstructed,'Cocok_sumber_variabel':compare,'Sampel_Excel':source_records,
        'Persistence_usia':donor_table,'Offgrid_sumber':offgrid,'Integritas_SHA256':integrity,
        'Definisi':[{'Istilah':k,'Definisi':v} for k,v in [
            ('Batas pembacaan','Hanya grid/source sampai akhir fold1 Agustus2025; hashing byte bukan akses nilai test. Tidak ada estimator di-fit.'),
            ('Kontribusi','Prediksi=intercept+Σ beta_j×(x_j−mean_train_j)/scale_train_j. Kontribusi deskriptif aljabar, bukan efek kausal. Tidak ada scaling target.'),
            ('Penalti','Ridge meminimalkan SSE+alphaΣbeta² pada seluruh34fitur standar: SO2, polutan, meteorologi dan missing. Intercept tidak dipenalti. Alpha1 tetap.'),
            ('Origin rendah','SO2 asli origin <=2.14; hanya penelusur lama. Prediksi di atas midpoint: >(minimum target train+2.14)/2, ringkasan kedekatan arah konsentrasi tinggi, bukan ambang validitas.'),
            ('Pergeseran fitur','Di luar minmax train dan |z|>5 bersifat deskriptif. Tidak menjadi batas sensor, alasan penghapusan atau tuning.'),
            ('Lebar train','Maksimum−minimum target train bersama; |residual| melebihi lebar tersebut dipakai hanya untuk melacak persistensi error.'),
            ('Persentase','Cakupan=prediksi finite/target asli teramati jendela×100. Keluar rentang=jumlah fitur validasi bersama di luar minmax train/N validasi bersama×100. CV=SDpopulasi/|mean|×100 bila mean nonnol. Usia donor%=jumlah usia/Ndomain×100.'),
            ('Sumber','Excel dibuka read_only, data_only, tanpa save. Nilai asli dibaca sampai sebelum akhir fold1 berikut1baris terminasi, bukan test. Satuan hanya literal file.'),
            ('Klasifikasi','Tidak ditemukan implementasi salah pada pemeriksaan ini; tidak membuktikan semua kemungkinan bug atau validitas sensor. Rancangan warmup90hari menyebabkan cohort model hanyaApril.'),
        ]]}
    dump(OUT/'tabel_diagnosis_fold1.json',tables)
    p.to_csv(RUN/'prediksi_dan_origin.csv',index=False)
    write_report(tables,cfg,train_min,train_max,transition)
    print('Diagnosis tanpa fit selesai:',len(checks),'checks;',len(integrity),'hash lama identik; transisi',transition,flush=True)

def write_report(t,cfg,train_min,train_max,transition):
    from write_diagnosis_fold1 import report
    report(t,cfg,train_min,train_max,transition)

if __name__=='__main__':main()
