"""Space technical prose without altering table data, code, or file links."""
import re
def polish(text):
    replacements={
        'traininternal':'train internal','validasiinternal':'validasi internal','internalvalidation':'validasi internal',
        'innertrain':'inner train','outertrain':'outer train','trainfold':'train fold','targetasli':'target asli',
        'earlystopping':'early stopping','targetstandar':'target standar','targetimputasi':'target imputasi',
        'targetkosong':'target kosong','slotouter':'slot outer','preprocessingfinal':'preprocessing final',
        'seedterbaik':'seed terbaik','seedtetap':'seed tetap','modelfinal':'model final','modelselesai':'model selesai',
        'checkpointlengkap':'checkpoint lengkap','sekuensfinite':'sekuens finite','checkpointbest':'checkpoint best',
        'timestampidentik':'timestamp identik','timestampouter':'timestamp outer','konsentrasitinggi':'konsentrasi tinggi',
        'cohortbersama':'cohort bersama','sekuens48slot':'sekuens 48 slot','sekuens48':'sekuens 48',
        'min_delta0':'min_delta 0','patience8':'patience 8','max50epoch':'max 50 epoch','weightdecay0':'weight decay 0',
        'learningrate0.001':'learning rate 0.001','fit_count=0':'fit_count=0',
        'meanseed':'mean seed','minmax3seed':'min–max 3 seed','semua3':'semua 3',
        'semua4':'semua 4','seluruh3':'seluruh 3','ketiga3':'ketiga 3','fiturstandar':'fitur standar',
        'minmax':'min–max','modelbaru':'model baru','targetOOF':'target OOF','RMSEfold':'RMSE fold',
        'RMSEgabungan':'RMSE gabungan','metrikgabungan':'metrik gabungan','Metrikgabungan':'Metrik gabungan',
        'biasseed':'bias seed','tanpacipping':'tanpa clipping','confidenceinterval':'confidence interval',
        'framewaktu':'frame waktu','hashSHA256':'hash SHA256','tetapidentik':'tetap identik',
        'sumberExcel':'sumber Excel','gridinput':'grid input','notebooktahap6':'notebook tahap 6',
        'environmentlama':'environment lama','artefaklama':'artefak lama','semuafold':'semua fold',
        'donorKNN':'donor KNN','bankdonorKNN':'bank donor KNN','SO2asli':'SO2 asli','originasli':'origin asli',
        'missingasli':'missing asli','indikatorasli':'indikator asli','modelevaluasi':'model evaluasi',
        'seed11/23/47':'seed 11/23/47','SDsampel':'SD sampel','SDantar-seed':'SD antar-seed',
        'sekuens traininternal':'sekuens train internal','slotinputunik':'slot input unik','slot inputunik':'slot input unik',
        'tanpafallback':'tanpa fallback','medianfallback':'median fallback','samplemask':'sample mask',
        'tidaktersedia':'tidak tersedia','tidakdituning':'tidak dituning','diperfold':'di per fold',
        'sebelumresume':'sebelum resume','setelahtransisi':'setelah transisi','foldterkait':'fold terkait',
        'sumberkovariat':'sumber kovariat','perperiode':'per periode','perpipeline':'per pipeline',
        'antar-pipeline':'antar-pipeline','rankMAE1':'rank MAE 1','rank1':'rank 1','epochparsial':'epoch parsial',
        'stateoptimizer':'state optimizer','hasiltraining':'hasil training','waktu modelselesai':'waktu model selesai',
        'sekitar1,56':'sekitar 1,56','danbias':'dan bias','sesudahtransisi':'sesudah transisi','seluruhperiode':'seluruh periode',
        'danseed':'dan seed','konsistensiMAE':'konsistensi MAE','biayaKNN':'biaya KNN','kegagalanrezimfold1':'kegagalan rezim fold 1',
        'ffillL4':'ffill L4','danKNNk10':'dan KNN k10','KNNk10':'KNN k10','sensitivitasseed':'sensitivitas seed',
        'dan4':'dan 4','menurutseed':'menurut seed','perbedaanMAE':'perbedaan MAE','sementaraSD':'sementara SD',
        'cakupanmandiri':'cakupan mandiri','evaluasiinternal':'evaluasi internal','setiapepoch':'setiap epoch',
        'inferensiouter':'inferensi outer','bersama3':'bersama 3','tigaestimasi':'tiga estimasi',';28':'; 28','dan8':'dan 8',
        'preprocessing4fold':'preprocessing 4 fold','fit1':'fit 1','P0seed11fold4':'P0 seed 11 fold 4','bukan0':'bukan 0',
        'inputtersimpan':'input tersimpan','fitulang':'fit ulang','dari167':'dari 167','hashawal':'hash awal',
        'inputnumerik':'input numerik','seluruh28':'seluruh 28','Hashingfile':'Hashing file','versiwide':'versi wide','inputsekuens':'input sekuens'}
    lines=[]
    for line in text.splitlines():
        if line.startswith(('|','![','- `')):lines.append(line);continue
        chunks=line.split('`')
        for i in range(0,len(chunks),2):
            s=chunks[i]
            for old,new in replacements.items():s=s.replace(old,new)
            s=re.sub(r'\bfold([1-6])\b',r'fold \1',s,flags=re.I)
            s=re.sub(r'(\d+(?:[.,]\d+)?)(?=(?:sampel|target|slot|kanal|langkah|hari|epoch|fitur|indikator|parameter|checkpoint|kombinasi|seed|pipeline|kovariat|model)\b)',r'\1 ',s)
            chunks[i]=s
        lines.append('`'.join(chunks))
    return '\n'.join(lines)+'\n'
