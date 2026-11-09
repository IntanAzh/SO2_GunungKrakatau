"""Aggregate completed stage10 models. No training and no final-test access."""
from pathlib import Path
from hashlib import sha256
import json,os
import numpy as np
import pandas as pd
from audit_quality import hashfile
from pilot_way_urang import dump
from calibration_sensitivity import distribution
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';RUN=OUT/'lstm_antarstasiun_tahap10';FIG=RUN/'figures'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def score(f,col,mask,fold,seed,domain,den):
    q=f.loc[mask];p=q[col].to_numpy(float);e=p-q['SO2 target asli'].to_numpy(float);n=len(e)
    return {'Fold':fold,'Seed':seed,'Skenario':col,'Domain':domain,'N':n,'Denominator target asli':int(den),
        'Cakupan persen':100*n/den if den else None,'MAE':float(np.abs(e).mean()) if n else None,
        'RMSE':float(np.sqrt(np.mean(e**2))) if n else None,'Bias':float(e.mean()) if n else None,
        'Prediksi negatif':int((p<0).sum()),'SHA256 timestamp':sha256('\n'.join(q['Timestamp target'].astype(str)).encode()).hexdigest()}

def summarize():
    c=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'));ch=hashfile(RUN/'config_locked.json')
    files=sorted(RUN.glob('Fold_*/S*_seed*/complete.json'));assert len(files)==24
    records=[];hist=[];frames=[];pre=[];scalers=[];samplets=[]
    for file in files:
        r=json.loads(file.read_text(encoding='utf-8'));assert r['config_sha256']==ch
        for name,h in r['files_sha256'].items():assert hashfile(file.parent/name)==h
        records.append({k:v for k,v in r.items() if k not in ['files_sha256','config_sha256']})
        hist+=json.loads((file.parent/'history.json').read_text(encoding='utf-8'))
    for i in range(1,5):
        folder=RUN/f'Fold_{i:02d}';pre.append(json.loads((folder/'preprocessing.json').read_text(encoding='utf-8')))
        scalers+=json.loads((folder/'parameter_scaler.json').read_text(encoding='utf-8'))
        for part in ['inner_train','inner_validation','evaluation']:
            ts=pd.read_csv(folder/f'timestamp_{part}_bersama.csv');ts.insert(0,'Bagian',part);ts.insert(0,'Fold',folder.name);samplets.append(ts)
        for seed in c['seeds']:
            x=pd.read_csv(folder/f'S0_seed{seed}/prediksi.csv',parse_dates=['Origin','Timestamp target'])
            z=pd.read_csv(folder/f'S1_seed{seed}/prediksi.csv',parse_dates=['Origin','Timestamp target'])
            assert np.array_equal(x['Index target'],z['Index target']) and np.array_equal(x['SO2 target asli'],z['SO2 target asli'])
            x=x.drop(columns=['Skenario']).rename(columns={'Prediksi':'S0'});x['S1']=z.Prediksi
            x['Bersama digunakan']=x['Bersama digunakan']&z['Bersama digunakan']&np.isfinite(x.Persistence)
            assert x['Timestamp target'].max()<pd.Timestamp(c['test_start']);frames.append(x)
    p=pd.concat(frames,ignore_index=True).sort_values(['Seed','Timestamp target']).reset_index(drop=True)
    assert not p[['Seed','Timestamp target']].duplicated().any()
    p.to_csv(RUN/'prediksi_oof_seluruh_seed.csv',index=False)
    pd.concat(samplets,ignore_index=True).to_csv(RUN/'timestamp_seluruh_bagian.csv',index=False)
    metrics=[];pooled=[];monthly=[];deltas=[];ridge=[];transition=[];transition_levels=[]
    old=pd.read_csv(OUT/'kontribusi_antarstasiun/prediksi_oof.csv',usecols=['Timestamp target','S0','S1'],parse_dates=['Timestamp target'])
    for fold in [f'Fold_{i:02d}' for i in range(1,5)]+['Gabungan_OOF']:
        for seed in c['seeds']:
            f=p[p.Seed.eq(seed)&(True if fold=='Gabungan_OOF' else p.Fold.eq(fold))];common=f['Bersama digunakan'].to_numpy(bool)
            for s in ['S0','S1','Persistence']:
                for domain,mask in [('mandiri',np.isfinite(f[s]).to_numpy()),('bersama',common)]:
                    r=score(f,s,mask,fold,seed,domain,len(f));(pooled if fold=='Gabungan_OOF' else metrics).append(r)
                    for month in f['Timestamp target'].dt.to_period('M').unique():
                        inside=(f['Timestamp target'].dt.to_period('M')==month).to_numpy()
                        monthly.append({'Bulan target':str(month),**score(f,s,mask&inside,fold,seed,domain,int(inside.sum()))})
            selected=(pooled if fold=='Gabungan_OOF' else metrics)
            rows={r['Skenario']:r for r in selected if r['Fold']==fold and r['Seed']==seed and r['Domain']=='bersama'}
            a=rows['S0'];b=rows['S1'];per=rows['Persistence']
            deltas.append({'Fold':fold,'Seed':seed,'N identik':a['N'],**{f'Perbaikan {m} persen':100*(a[m]-b[m])/a[m] for m in ['MAE','RMSE']},
                **{f'Delta {m} S1 minus S0':b[m]-a[m] for m in ['MAE','RMSE','Bias']},
                **{f'Delta {m} {s} minus persistence':rows[s][m]-per[m] for m in ['MAE','RMSE'] for s in ['S0','S1']}})
            joined=f[f['Bersama digunakan']].merge(old[['Timestamp target','S0','S1']],on='Timestamp target',suffixes=('','_Ridge'),validate='one_to_one')
            keep=np.isfinite(joined[['S0','S1','S0_Ridge','S1_Ridge','Persistence']]).all(axis=1)
            for s in ['S0','S1','S0_Ridge','S1_Ridge','Persistence']:ridge.append(score(joined,s,keep,fold,seed,'LSTM dan Ridge timestamp identik',len(f)))
            if fold=='Fold_01':
                center=pd.Timestamp('2025-06-22 17:30:00');within=f['Timestamp target'].between(center-pd.Timedelta(hours=72),center+pd.Timedelta(hours=72))
                for period,condition in [('sebelum transisi',f['Timestamp target']<center),('sesudah transisi',f['Timestamp target']>=center)]:
                    mask=within&condition&common
                    for s in ['S0','S1','Persistence']:
                        transition.append({'Periode':period,**score(f,s,mask,'Fold_01',seed,'jendela transisi ±72jam',int((within&condition).sum()))})
                        q=f.loc[mask];transition_levels.append({'Seed':seed,'Periode':period,'Skenario':s,'N':len(q),
                            'Target mean':float(q['SO2 target asli'].mean()),'Target median':float(q['SO2 target asli'].median()),
                            'Prediksi mean':float(q[s].mean()),'Prediksi median':float(q[s].median()),'Prediksi minimum':float(q[s].min()),'Prediksi maksimum':float(q[s].max())})
    variation=[]
    for (fold,s,domain),g in pd.DataFrame(metrics+pooled).groupby(['Fold','Skenario','Domain'],sort=False):
        row={'Fold':fold,'Skenario':s,'Domain':domain,'Seed N':len(g)}
        for k in ['MAE','RMSE','Bias','Cakupan persen','Prediksi negatif']:
            row[k+' mean']=float(g[k].mean());row[k+' SD seed']=float(g[k].std(ddof=1));row[k+' min']=float(g[k].min());row[k+' max']=float(g[k].max())
        variation.append(row)
    monthly_delta=[]
    mdf=pd.DataFrame([r for r in monthly if r['Domain']=='bersama'])
    for (fold,seed,mo),g in mdf.groupby(['Fold','Seed','Bulan target'],sort=False):
        g=g.set_index('Skenario');a=g.loc['S0'];b=g.loc['S1']
        monthly_delta.append({'Fold':fold,'Seed':int(seed),'Bulan target':mo,'N identik':int(a.N),
            **{f'Perbaikan {m} persen':float(100*(a[m]-b[m])/a[m]) for m in ['MAE','RMSE']},
            **{f'Delta {m} S1 minus S0':float(b[m]-a[m]) for m in ['MAE','RMSE']}})
    oldhash=[]
    previous=json.loads((OUT/'kontribusi_antarstasiun/config_locked.json').read_text(encoding='utf-8'))['initial_hashes']
    baseline=dict(previous)
    for rel,before in c['initial_hashes'].items():
        if rel in baseline:assert baseline[rel]==before,rel
        baseline[rel]=before
    for rel,before in baseline.items():
        after=hashfile(ROOT/rel);assert before==after,rel
        oldhash.append({'File':rel,'Baseline':'awal tahap10' if rel in c['initial_hashes'] else 'awal tahap9; identik diverifikasi akhir tahap9',
            'SHA256 sebelum':before,'SHA256 sesudah':after,'Identik':True})
    # Donor availability summarized once per origin, not three times across seeds.
    availability_month=[];age_distribution=[];constants=[];excluded=[];excluded_month=[];purges=[]
    n=c['fold_cuts'][-1];dg=pd.read_csv(OUT/'kontribusi_antarstasiun/grid_dev_Pulomerak.csv',nrows=n);x=dg['SO2 numerik untuk grafik'].to_numpy(float);times=pd.DatetimeIndex(pd.to_datetime(dg.Timestamp))
    runlist=[];begin=0
    for end in range(1,len(x)+1):
        if end<len(x) and np.isfinite(x[end]) and np.isfinite(x[end-1]) and x[end]==x[end-1]:continue
        if end-begin>=2 and np.isfinite(x[begin]):runlist.append((begin,end,x[begin]))
        begin=end
    for i in range(1,5):
        fold=f'Fold_{i:02d}';z=np.load(RUN/fold/'inputs_sekuens.npz')
        a=c['fold_cuts'][i-1];b=c['fold_cuts'][i];inner=int(np.floor(.85*a))
        way_y=pd.read_csv(OUT/'pilot_way_urang/grid_dan_penanda_asli.csv',nrows=n)['SO2 numerik'].to_numpy(float)
        for part,key,lo,hi in [('inner_train','train',0,inner),('inner_validation','internal',inner,a),('evaluation','evaluation',a,b)]:
            target=np.arange(lo,hi);observed=np.isfinite(way_y[target]);used=z[key]+1
            reason_groups={'target kosong':target[~observed],
                'histori sebelum grid':target[observed&(target==0)],
                'warmup statistik':target[observed&(target>=1)&(target<=1440)],
                'buffer setelah warmup':target[observed&(target>1440)&(target<1489)],
                'sesudah warmup tetapi tidak masuk irisan':target[observed&(target>=1489)&~np.isin(target,used)]}
            for reason,ts in reason_groups.items():
                excluded.append({'Fold':fold,'Bagian':part,'Alasan':reason,'Slot tersisih':len(ts),'Target awal':times[ts[0]] if len(ts) else None,
                    'Target akhir':times[ts[-1]] if len(ts) else None,**distribution(way_y[ts])})
                for mo in times[target].to_period('M').unique():
                    sub=ts[times[ts].to_period('M')==mo]
                    excluded_month.append({'Fold':fold,'Bagian':part,'Bulan':str(mo),'Alasan':reason,'Slot tersisih':len(sub),**distribution(way_y[sub])})
        for part,origin in [('inner_train',inner-1),('inner_validation',a-1)]:
            purges.append({'Fold':fold,'Bagian yang dilarang':part,'Origin dipurge':times[origin],'Timestamp target':times[origin+1],
                'SO2 target asli':way_y[origin+1],'Target asli teramati':bool(np.isfinite(way_y[origin+1])),
                'Penjelasan':'Dikeluarkan dari bagian sebelumnya; menjadi kandidat bagian sesudahnya. Bukan penghapusan label global.'})
        for part,key in [('inner_train','train'),('inner_validation','internal'),('evaluation','evaluation')]:
            ids=z[key];last=np.maximum.accumulate(np.where(np.isfinite(x),np.arange(n),-1));age=(ids-last[ids-1])*30
            for agevalue,count in zip(*np.unique(age,return_counts=True)):
                age_distribution.append({'Fold':fold,'Bagian':part,'Usia menit':int(agevalue),'N origin':int(count),'Denominator origin':len(ids),'Persen origin':100*count/len(ids)})
            window=ids[:,None]-np.arange(1,49)[None,:];missing=~np.isfinite(x[window]);months=times[ids+1].to_period('M')
            for month in months.unique():
                selected=months==month;m=missing[selected];ages=age[selected]
                availability_month.append({'Fold':fold,'Bagian':part,'Bulan target':str(month),'N sekuens':int(selected.sum()),'Posisi donor':m.size,
                    'Posisi donor diimputasi':int(m.sum()),'Persen posisi donor diimputasi':100*m.mean(),
                    'Usia origin median menit':float(np.median(ages)),'Usia origin P95 menit':float(np.percentile(ages,95)),'Usia origin maksimum menit':int(ages.max())})
            used=np.unique(window.ravel())
            for runid,(begin,end,value) in enumerate(runlist,1):
                num=int(((used>=begin)&(used<end)).sum())
                if num:constants.append({'Fold':fold,'Bagian':part,'Run ID':runid,'SO2 asli Pulomerak':value,'Run mulai':times[begin],
                    'Run akhir':times[end-1],'Slot run':end-begin,'Durasi menit':30*(end-begin),'Sumber unik masuk sekuens':num,'Terpotong batas development':end==n})
    flat=p[p.Seed.eq(11)].drop(columns=['Seed','S0','S1'])
    for seed in c['seeds']:
        q=p[p.Seed.eq(seed)].set_index('Timestamp target')
        for s in ['S0','S1']:flat[f'{s}_seed{seed}']=flat['Timestamp target'].map(q[s])
    flat.to_csv(RUN/'prediksi_wide_seluruh_seed.csv',index=False)
    defs=[{'Istilah':'Populasi','Definisi':'Development yang sudah dipakai sebelumnya; target SO2 Way Urang asli; fold kalender tetap; test mulai '+c['test_start']+' tidak dimuat/dievaluasi.'},
        {'Istilah':'Error','Definisi':'e=prediksi−original target; MAE=mean(abs(e)); RMSE=sqrt(mean(e²)); bias=mean(e), semua µg/m³ setelah inverse target scaling.'},
        {'Istilah':'Cakupan persen','Definisi':'100×N prediksi finite / seluruh target SO2 Way Urang numerik asli teramati pada jendela terkait. Bersama memakai timestamp S0/S1/persistence identik.'},
        {'Istilah':'Perbaikan persen','Definisi':'100×(error S0−error S1)/error S0; positif berarti error lebih rendah. Tidak menunjukkan signifikansi statistik.'},
        {'Istilah':'Variasi seed','Definisi':'Mean dan SD sampel ddof=1 dari seed 11/23/47, tanpa memilih seed terbaik. Persistence deterministik diulang sebagai pembanding, bukan tiga replikasi independen.'},
        {'Istilah':'Gabungan','Definisi':'Error semua fold evaluasi tidak tumpang tindih digabung per seed, bukan rata-rata RMSE fold.'},
        {'Istilah':'Usia donor','Definisi':'origin−timestamp numerik asli Pulomerak terakhir≤origin−30menit, bukan usia median imputasi; minimum30menit. Clock dan latensi belum terverifikasi.'},
        {'Istilah':'Proporsi posisi donor diimputasi','Definisi':'100×posisi missing asli Pulomerak pada sekuens bersama / (N sekuens×48); posisi waktu dapat berulang pada sekuens yang tumpang tindih.'},
        {'Istilah':'Konstan','Definisi':'Nilai original tepat sama ≥2 slot berturut-turut; missing memutus; durasi=Nslot×30menit; penanda deskriptif bukan kerusakan sensor.'},
        {'Istilah':'Waktu','Definisi':c['timing']},
        {'Istilah':'Sumber','Definisi':'Excel data_raw/Telkom University Purwokerto - Intan Azizah.xlsx, sheet Way Urang dan Pulomerak, header8/satuan9. Grid development terverifikasi tahap9; satuan literal µg/m³. SHA256 di Integritas_SHA256.'}]
    tables={'Ringkasan':[r for r in variation if r['Domain']=='bersama'],'Skor_fold_seed':metrics,'Gabungan_per_seed':pooled,
        'Perubahan_S1_vs_S0':deltas,'Skor_bulanan':monthly,'Perubahan_bulanan':monthly_delta,'Ridge_timestamp_identik':ridge,'Transisi_per_seed':transition,'Tingkat_transisi':transition_levels,
        'Alur_sampel':[r for q in pre for r in q['flow']],'Distribusi_target':[r for q in pre for r in q['distribusi']],
        'Target_bulanan':[r for q in pre for r in q['bulanan']],'Keluar_irisan':[r for q in pre for r in q['irisan_keluar']],
        'Sampel_tersisih':excluded,'Tersisih_bulanan':excluded_month,'Purge_batas':purges,
        'Ketersediaan_donor':[r for q in pre for r in q['ketersediaan']],'Donor_bulanan':availability_month,'Distribusi_usia_donor':age_distribution,
        'Rentetan_konstan_donor':constants,'Training_dan_waktu':records,'Riwayat_training':hist,'Scaler_train_internal':scalers,
        'Prefix_median':[r for q in pre for r in q['prefix']],'Gate_sebelum_training':json.loads((RUN/'gate_semua_sebelum_fit.json').read_text())['gates'],
        'Offgrid_development':pd.read_csv(RUN/'ledger_offgrid_development.csv').to_dict('records'),
        'Prediksi_seluruh_seed':flat.to_dict('records'),
        'Konfigurasi':[{'Parameter':k,'Nilai':('UTC: '+v) if k=='created_utc' else json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else str(v)} for k,v in c.items() if k!='initial_hashes'],
        'Definisi':defs,'Integritas_SHA256':oldhash}
    dump(RUN/'tabel_hasil.json',tables)
    dump(RUN/'checks.json',{'completed_models':24,'config_sha256':ch,'old_files_sha256_identical':len(oldhash),'unique_targets_per_seed':p.groupby('Seed').size().to_dict(),
        'common_targets_per_seed':p[p['Bersama digunakan']].groupby('Seed').size().to_dict(),'test_loaded':False,'test_evaluated':False})
    plots(p,metrics,deltas,records,hist)
    from report_interstation_stage10 import write_report
    write_report(tables,c)
    print('Ringkasan tahap10 selesai, 24 model; tabel dan grafik tersimpan.',flush=True)
    return tables

def plots(p,metrics,deltas,records,hist):
    common=pd.DataFrame([r for r in metrics if r['Domain']=='bersama']);d=pd.DataFrame(deltas);colors={'S0':'#225ea8','S1':'#e6550d','Persistence':'#555555'}
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for row,metric in enumerate(['MAE','RMSE']):
        for col,folds in enumerate([[f'Fold_{i:02d}' for i in range(1,5)],[f'Fold_{i:02d}' for i in range(2,5)]]):
            ax=axes[row,col]
            for s in ['S0','S1','Persistence']:
                q=common[common.Skenario.eq(s)&common.Fold.isin(folds)].groupby('Fold')[metric].agg(['mean','std']).reindex(folds)
                ax.errorbar(np.arange(len(folds)),q['mean'],yerr=q['std'],label=s,marker='o',capsize=3,color=colors[s])
            ax.set_xticks(np.arange(len(folds)),[x[-2:] for x in folds]);ax.set_xlabel('Fold');ax.set_ylabel(metric+' (µg/m³)');ax.grid(alpha=.2)
            ax.set_title('Seluruh fold' if col==0 else 'Detail fold 2–4');ax.legend()
    fig.suptitle('LSTM: mean ± SD antar-seed, timestamp evaluasi identik');fig.tight_layout();fig.savefig(FIG/'skor_per_fold.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,m in zip(axes,['MAE','RMSE']):
        for seed in [11,23,47]:
            q=d[d.Seed.eq(seed)&d.Fold.ne('Gabungan_OOF')];ax.plot(q.Fold.str[-2:],q[f'Perbaikan {m} persen'],marker='o',label=f'Seed {seed}')
        ax.axhline(0,color='black',lw=.8);ax.set_title(m);ax.set_xlabel('Fold');ax.set_ylabel('Perbaikan S1 terhadap S0 (%)');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(FIG/'perubahan_pulomerak.png',dpi=150);plt.close(fig)
    center=pd.Timestamp('2025-06-22 17:30:00');fig,axes=plt.subplots(3,1,figsize=(15,10),sharex=True)
    for ax,seed in zip(axes,[11,23,47]):
        q=p[p.Seed.eq(seed)&p['Timestamp target'].between(center-pd.Timedelta(hours=72),center+pd.Timedelta(hours=72))]
        for s in ['SO2 target asli','S0','S1','Persistence']:
            ax.plot(q['Timestamp target'],q[s],label=s,lw=1,alpha=.9)
        ax.axvline(center,color='black',ls='--');ax.set_title(f'Seed {seed}, tanpa clipping');ax.set_ylabel('µg/m³');ax.legend(ncol=4);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'transisi_22_juni.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    history=pd.DataFrame(hist)
    for ax,i in zip(axes.ravel(),range(1,5)):
        for s in ['S0','S1']:
            for seed in [11,23,47]:
                q=history[history.Fold.eq(f'Fold_{i:02d}')&history.Skenario.eq(s)&history.Seed.eq(seed)];ax.plot(q.Epoch,q['Internal MSE standar'],label=f'{s} seed{seed}')
        ax.set_title(f'Fold {i}');ax.set_xlabel('Epoch');ax.set_ylabel('MSE internal standar');ax.set_yscale('log');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'riwayat_early_stopping.png',dpi=150);plt.close(fig)

if __name__=='__main__':summarize()
