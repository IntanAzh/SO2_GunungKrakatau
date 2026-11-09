"""Read completed stage11 checkpoints and saved stage10 LSTM predictions; no fit."""
from pathlib import Path
from hashlib import sha256
import os,json
import numpy as np
import pandas as pd
from audit_quality import hashfile
from pilot_way_urang import dump
from calibration_sensitivity import distribution

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OLD=OUT/'lstm_antarstasiun_tahap10';RUN=OUT/'perbandingan_algoritma_tahap11';FIG=RUN/'figures'
ALGORITHMS=['LSTM','GRU','TCN','Persistence'];SEEDS=[11,23,47]
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def score(frame,alg,mask,fold,seed,domain,den):
    q=frame.loc[mask];p=q[alg].to_numpy(float);y=q['SO2 target asli'].to_numpy(float);error=p-y;n=len(error)
    return {'Fold':fold,'Seed':seed,'Algoritma':alg,'Domain':domain,'N':n,'Denominator target asli':int(den),
        'Cakupan persen':100*n/den if den else None,'MAE':float(np.abs(error).mean()) if n else None,
        'RMSE':float(np.sqrt(np.mean(error**2))) if n else None,'Bias':float(error.mean()) if n else None,
        'Prediksi negatif':int((p<0).sum()),
        'SHA256 timestamp':sha256('\n'.join(q['Timestamp target'].astype(str)).encode()).hexdigest()}

def summarize():
    cfg=json.loads((RUN/'config_locked.json').read_text(encoding='utf-8'));ch=hashfile(RUN/'config_locked.json')
    gate=json.loads((RUN/'gate_semua_sebelum_fit.json').read_text(encoding='utf-8'))
    assert gate['config_sha256']==ch and gate['fit_count']==0 and len(gate['folds'])==4
    assert all(x['Lulus'] for x in gate['folds'])
    records=[];hist=[];predframes=[];targetdist=[];targetmonth=[];samplefiles=[]
    for i in range(1,5):
        fold=f'Fold_{i:02d}';fdir=RUN/fold;old=OLD/fold
        for part in ['inner_train','inner_validation','evaluation']:
            p=pd.read_csv(fdir/f'timestamp_{part}.csv',parse_dates=['Origin','Timestamp target'])
            p.insert(0,'Bagian',part);p.insert(0,'Fold',fold);samplefiles.append(p)
            values=p['SO2 target asli'].to_numpy(float)
            targetdist.append({'Fold':fold,'Bagian':part,'Awal target':p['Timestamp target'].iloc[0],
                'Akhir target':p['Timestamp target'].iloc[-1],**distribution(values)})
            for month,g in p.groupby(p['Timestamp target'].dt.to_period('M'),sort=False):
                targetmonth.append({'Fold':fold,'Bagian':part,'Bulan target':str(month),**distribution(g['SO2 target asli'])})
        for seed in SEEDS:
            olddir=old/f'S0_seed{seed}';prior=json.loads((olddir/'complete.json').read_text(encoding='utf-8'))
            for name,h in prior['files_sha256'].items():assert hashfile(olddir/name)==h,name
            frame=pd.read_csv(olddir/'prediksi.csv',parse_dates=['Origin','Timestamp target'])
            frame=frame[['Fold','Index origin','Origin','Index target','Timestamp target','SO2 target asli','Persistence','Persistence usia slot','Prediksi']]
            frame=frame.rename(columns={'Prediksi':'LSTM'});frame.insert(1,'Seed',seed)
            records.append({'Fold':fold,'Algoritma':'LSTM','Seed':seed,'Sumber':'checkpoint tahap10 S0',
                'Epoch dijalankan':prior['Epoch dijalankan'],'Epoch terbaik':prior['Epoch terbaik'],
                'Internal MSE standar terbaik':prior['Internal MSE standar terbaik'],
                'Fit early stopping detik':prior['Fit early stopping detik'],'Predict detik':prior['Predict detik'],
                'Jumlah parameter':prior['Jumlah parameter'],'N train':prior['N train bersama'],'N internal':prior['N validasi internal bersama'],
                'Checkpoint SHA256':prior['files_sha256']['best.pt'],'Path checkpoint':(olddir/'best.pt').relative_to(ROOT).as_posix()})
            hist += [{**row,'Algoritma':'LSTM'} for row in json.loads((olddir/'history.json').read_text(encoding='utf-8'))]
            for new in ['GRU','TCN']:
                modeldir=fdir/f'{new}_seed{seed}';done=json.loads((modeldir/'complete.json').read_text(encoding='utf-8'))
                assert done['config_sha256']==ch
                for name,h in done['files_sha256'].items():assert hashfile(modeldir/name)==h,name
                f=pd.read_csv(modeldir/'prediksi.csv',parse_dates=['Origin','Timestamp target'])
                assert np.array_equal(f['Index origin'],frame['Index origin'])
                assert np.array_equal(f['Timestamp target'],frame['Timestamp target'])
                assert np.array_equal(f['SO2 target asli'],frame['SO2 target asli'])
                assert np.array_equal(f.Persistence,frame.Persistence)
                frame[new]=f.Prediksi
                records.append({'Fold':fold,'Algoritma':new,'Seed':seed,'Sumber':'fit baru tahap11',
                    'Epoch dijalankan':done['Epoch dijalankan'],'Epoch terbaik':done['Epoch terbaik'],
                    'Internal MSE standar terbaik':done['Internal MSE standar terbaik'],
                    'Fit early stopping detik':done['Fit early stopping detik'],'Predict detik':done['Predict detik'],
                    'Jumlah parameter':done['Jumlah parameter'],'N train':done['N train'],'N internal':done['N internal'],
                    'Checkpoint SHA256':done['files_sha256']['best.pt'],'Path checkpoint':(modeldir/'best.pt').relative_to(ROOT).as_posix()})
                hist += json.loads((modeldir/'history.json').read_text(encoding='utf-8'))
            frame['Timestamp bersama']=np.isfinite(frame[ALGORITHMS]).all(axis=1)
            predframes.append(frame)
    pred=pd.concat(predframes,ignore_index=True).sort_values(['Seed','Timestamp target']).reset_index(drop=True)
    assert not pred[['Seed','Timestamp target']].duplicated().any()
    assert pred['Timestamp target'].max()<pd.Timestamp(cfg['test_start'])
    pred.to_csv(RUN/'prediksi_oof_seluruh_seed.csv',index=False)
    pd.concat(samplefiles,ignore_index=True).to_csv(RUN/'timestamp_semua_bagian.csv',index=False)
    metrics=[];pooled=[];monthly=[];deltas=[];monthly_deltas=[];transition=[]
    for fold in [f'Fold_{i:02d}' for i in range(1,5)]+['Gabungan_OOF']:
        for seed in SEEDS:
            f=pred[pred.Seed.eq(seed)&(True if fold=='Gabungan_OOF' else pred.Fold.eq(fold))]
            common=f['Timestamp bersama'].to_numpy(bool);rows={}
            for alg in ALGORITHMS:
                for domain,mask in [('mandiri',np.isfinite(f[alg]).to_numpy()),('bersama',common)]:
                    r=score(f,alg,mask,fold,seed,domain,len(f));(pooled if fold=='Gabungan_OOF' else metrics).append(r)
                    if domain=='bersama':rows[alg]=r
                    for mo in f['Timestamp target'].dt.to_period('M').unique():
                        within=f['Timestamp target'].dt.to_period('M').eq(mo).to_numpy()
                        monthly.append({'Bulan target':str(mo),**score(f,alg,mask&within,fold,seed,domain,int(within.sum()))})
            base=rows['Persistence']
            for alg in ALGORITHMS[:-1]:
                r=rows[alg]
                deltas.append({'Fold':fold,'Seed':seed,'Algoritma':alg,'N identik':r['N'],
                    'Delta MAE vs persistence':r['MAE']-base['MAE'],'Delta RMSE vs persistence':r['RMSE']-base['RMSE'],
                    'Perbaikan MAE persen':100*(base['MAE']-r['MAE'])/base['MAE'],
                    'Perbaikan RMSE persen':100*(base['RMSE']-r['RMSE'])/base['RMSE']})
            if fold=='Fold_01':
                center=pd.Timestamp('2025-06-22 17:30');within=f['Timestamp target'].between(center-pd.Timedelta(hours=72),center+pd.Timedelta(hours=72))
                for label,side in [('sebelum',f['Timestamp target']<center),('sesudah',f['Timestamp target']>=center)]:
                    mask=(within&side).to_numpy()&common
                    for alg in ALGORITHMS:
                        q=f.loc[mask];r=score(f,alg,mask,fold,seed,'transisi ±72 jam',int((within&side).sum()))
                        transition.append({'Periode':label,'Target mean':float(q['SO2 target asli'].mean()) if len(q) else None,
                            'Prediksi mean':float(q[alg].mean()) if len(q) else None,'Prediksi median':float(q[alg].median()) if len(q) else None,**r})
    var=[]
    for (fold,alg,domain),g in pd.DataFrame(metrics+pooled).groupby(['Fold','Algoritma','Domain'],sort=False):
        row={'Fold':fold,'Algoritma':alg,'Domain':domain,'Seed N':len(g)}
        for key in ['MAE','RMSE','Bias','Cakupan persen','Prediksi negatif']:
            row[key+' mean']=float(g[key].mean());row[key+' SD seed']=float(g[key].std(ddof=1))
        var.append(row)
    monthly_shared=pd.DataFrame([r for r in monthly if r['Domain']=='bersama'])
    for (fold,seed,mo),g in monthly_shared.groupby(['Fold','Seed','Bulan target'],sort=False):
        g=g.set_index('Algoritma');base=g.loc['Persistence']
        for alg in ALGORITHMS[:-1]:
            r=g.loc[alg]
            monthly_deltas.append({'Fold':fold,'Seed':int(seed),'Bulan target':mo,'Algoritma':alg,'N identik':int(r.N),
                'Delta MAE vs persistence':float(r.MAE-base.MAE),'Delta RMSE vs persistence':float(r.RMSE-base.RMSE)})
    summarytime=[]
    for alg,g in pd.DataFrame(records).groupby('Algoritma',sort=False):
        summarytime.append({'Algoritma':alg,'Model N':len(g),'Jumlah parameter':int(g['Jumlah parameter'].iloc[0]),
            'Epoch total':int(g['Epoch dijalankan'].sum()),'Epoch median':float(g['Epoch dijalankan'].median()),
            'Fit total detik':float(g['Fit early stopping detik'].sum()),'Fit median detik':float(g['Fit early stopping detik'].median()),
            'Predict total detik':float(g['Predict detik'].sum())})
    wide=pred[pred.Seed.eq(11)][['Fold','Index origin','Origin','Index target','Timestamp target','SO2 target asli','Persistence','Persistence usia slot']].copy()
    for seed in SEEDS:
        f=pred[pred.Seed.eq(seed)].set_index('Timestamp target')
        for alg in ALGORITHMS[:-1]:wide[f'{alg}_seed{seed}']=wide['Timestamp target'].map(f[alg])
    wide.to_csv(RUN/'prediksi_wide_seluruh_seed.csv',index=False)
    integrity=[]
    for rel,before in cfg['initial_hashes'].items():
        after=hashfile(ROOT/rel);assert after==before,rel
        integrity.append({'File':rel,'SHA256 awal':before,'SHA256 akhir':after,'Identik':True})
    assert hashfile(OLD/'config_locked.json')==cfg['stage10_config_sha256']
    assert hashfile(OLD/'manifest_tahap10.json')==cfg['stage10_manifest_sha256']
    definitions=[{'Istilah':'Target dan domain','Definisi':'SO2 Way Urang asli pada t+30menit; hanya development sebelum '+cfg['test_start']+'. Test akhir tidak dimuat nilai atau dievaluasi.'},
        {'Istilah':'Sampel bersama','Definisi':'Irisan prediksi finite LSTM, GRU, TCN, persistence pada target dan timestamp evaluasi yang sama untuk seed/fold terkait.'},
        {'Istilah':'Cakupan','Definisi':'100×N prediksi finite/seluruh target Way Urang asli teramati pada jendela fold. Bersama dan mandiri menggunakan denominator target sama.'},
        {'Istilah':'Error','Definisi':'e=prediksi−target asli setelah inverse target scaling; MAE=mean(abs(e)), RMSE=sqrt(mean(e²)), bias=mean(e), µg/m³.'},
        {'Istilah':'Gabungan OOF','Definisi':'Error prediksi seluruh fold yang tidak tumpang tindih digabung per seed; RMSE tidak dihitung dari rata-rata RMSE fold.'},
        {'Istilah':'Variasi seed','Definisi':'Rerata dan SD sampel ddof1 dari seed 11/23/47. Persistence yang diulang per seed deterministik, bukan replikasi stokastik.'},
        {'Istilah':'Perbaikan vs persistence','Definisi':'100×(error persistence−error algoritma)/error persistence; negatif berarti algoritma lebih buruk pada timestamp identik.'},
        {'Istilah':'TCN kausal','Definisi':'Setiap Conv1d kernel3 padding kiri efektif 2×dilation, keluaran kanan dipotong; output t memakai input≤t. Empat blok dua conv d1/2/4/8, RF61 dibatasi sekuens48.'},
        {'Istilah':'Waktu','Definisi':'Fit adalah jumlah detik epoch train+validasi internal; checkpoint I/O/tensor preparation/jeda resume tidak termasuk. Predict mencakup inferensi/inverse. Total waktu beda epoch, tidak menyetarakan optimisasi.'},
        {'Istilah':'Sumber','Definisi':'Input numerik dari grid development terverifikasi tahap10, sumber Excel data_raw/Telkom University Purwokerto - Intan Azizah.xlsx sheet Kab. Lampung Selatan Way Urang, header8/satuan9, SO2 µg/m³ literal.'}]
    tables={'Ringkasan':[r for r in var if r['Domain']=='bersama'],'Skor_fold_seed':metrics,'Gabungan_per_seed':pooled,
        'Vs_persistence':deltas,'Skor_bulanan':monthly,'Bulanan_vs_persistence':monthly_deltas,
        'Transisi_22_juni':transition,'Waktu_dan_parameter':records,'Ringkas_waktu':summarytime,'Riwayat_training':hist,
        'Gate_sebelum_fit':gate['folds'],'Distribusi_target':targetdist,'Target_bulanan':targetmonth,
        'Prediksi_seluruh_seed':wide.to_dict('records'),
        'Konfigurasi':[{'Parameter':k,'Nilai':('UTC: '+str(v)) if k=='created_utc' else ('tanggal: '+str(v)) if k=='test_start' else json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else str(v)} for k,v in cfg.items() if k!='initial_hashes'],
        'Definisi':definitions,'Integritas_SHA256':integrity}
    dump(RUN/'tabel_hasil.json',tables)
    dump(RUN/'checks.json',{'new_completed_models':24,'old_lstm_reference_models':12,'config_sha256':ch,
        'old_files_identical':len(integrity),'prediction_rows_per_seed':pred.groupby('Seed').size().to_dict(),
        'common_rows_per_seed':pred[pred['Timestamp bersama']].groupby('Seed').size().to_dict(),
        'test_loaded':False,'test_evaluated':False})
    plots(pred,metrics,deltas,hist,summarytime)
    from report_algorithms_stage11 import write_report
    write_report(tables,cfg)
    print('Tabel, laporan, grafik selesai; tanpa fit tambahan.',flush=True)
    return tables

def plots(pred,metrics,deltas,hist,timing):
    shared=pd.DataFrame([r for r in metrics if r['Domain']=='bersama']);colors={'LSTM':'#1f77b4','GRU':'#e6550d','TCN':'#5b8c3a','Persistence':'#444444'}
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for row,metric in enumerate(['MAE','RMSE']):
        for col,folds in enumerate([[f'Fold_{i:02d}' for i in range(1,5)],[f'Fold_{i:02d}' for i in range(2,5)]]):
            ax=axes[row,col]
            for alg in ALGORITHMS:
                g=shared[shared.Algoritma.eq(alg)&shared.Fold.isin(folds)].groupby('Fold')[metric].agg(['mean','std']).reindex(folds)
                ax.errorbar(range(len(folds)),g['mean'],yerr=g['std'],marker='o',capsize=3,label=alg,color=colors[alg])
            ax.set_xticks(range(len(folds)),[f[-2:] for f in folds]);ax.set_xlabel('Fold');ax.set_ylabel(metric+' (µg/m³)');ax.set_title('Semua fold' if col==0 else 'Detail fold 2–4');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Mean ± SD antar-seed pada target evaluasi yang sama');fig.tight_layout();fig.savefig(FIG/'skor_per_fold.png',dpi=150);plt.close(fig)
    center=pd.Timestamp('2025-06-22 17:30');fig,axes=plt.subplots(3,1,figsize=(15,10),sharex=True)
    for ax,seed in zip(axes,SEEDS):
        q=pred[pred.Seed.eq(seed)&pred['Timestamp target'].between(center-pd.Timedelta(hours=72),center+pd.Timedelta(hours=72))]
        for alg in ['SO2 target asli']+ALGORITHMS:ax.plot(q['Timestamp target'],q[alg],lw=.9,label=alg)
        ax.axvline(center,color='black',ls='--');ax.set_title(f'Seed {seed}, prediksi tanpa clipping');ax.set_ylabel('SO₂ (µg/m³)');ax.grid(alpha=.2);ax.legend(ncol=5)
    fig.tight_layout();fig.savefig(FIG/'transisi_22_juni.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7));h=pd.DataFrame(hist)
    for ax,i in zip(axes.ravel(),range(1,5)):
        for alg in ALGORITHMS[:-1]:
            for seed in SEEDS:
                q=h[h.Fold.eq(f'Fold_{i:02d}')&h.Seed.eq(seed)&h.Algoritma.eq(alg)]
                ax.plot(q.Epoch,q['Internal MSE standar'],label=f'{alg} {seed}',lw=.9)
        ax.set_title(f'Fold {i}');ax.set_yscale('log');ax.set_xlabel('Epoch');ax.set_ylabel('MSE internal standar');ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=3)
    fig.tight_layout();fig.savefig(FIG/'riwayat_internal.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));algs=['LSTM','GRU','TCN'];d={r['Algoritma']:r for r in timing}
    ax.bar(algs,[d[a]['Fit total detik'] for a in algs],color=[colors[a] for a in algs]);ax.set_ylabel('Total detik fit terukur (12 model)')
    for i,a in enumerate(algs):ax.text(i,d[a]['Fit total detik'],f"{d[a]['Epoch total']} epoch",ha='center',va='bottom')
    ax.set_title('Biaya training dengan epoch early stopping berbeda');fig.tight_layout();fig.savefig(FIG/'waktu_training.png',dpi=150);plt.close(fig)

if __name__=='__main__':summarize()
