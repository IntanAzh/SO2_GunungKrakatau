from pathlib import Path
import json
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'outputs/lstm_preprocessing'
gate=json.loads((RUN/'all_gates_before_training.json').read_text(encoding='utf-8'))
assert sha256((RUN/'config_locked.json').read_bytes()).hexdigest()==gate['config_sha256']
p=ROOT/'notebooks/07_lstm_preprocessing.ipynb';nb=json.loads(p.read_text(encoding='utf-8'));cells=[c for c in nb['cells'] if c['cell_type']=='code']
cells[0]['source']=cells[0]['source'].replace('e=Experiment().prepare()',
    'from resume_lstm_stage7 import restore_experiment, continue_models\ne=restore_experiment()')
cells[1]['source']=cells[1]['source'].replace('e.prepare_all_folds()',
    "if not getattr(e,'restored',False): e.prepare_all_folds()")
cells[1]['source']=cells[1]['source'].replace("print('LSTM fit sebelum seluruh gate:',e.fit_count)",
    "print('Checkpoint selesai saat resume:',e.fit_count,'; gate awal tersimpan dengan fit_count=0')")
cells[2]['source']=cells[2]['source'].replace('e.train()','continue_models(e)')
assert 'e=Experiment().prepare()' not in cells[0]['source'] and 'continue_models(e)' in cells[2]['source']
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print('Notebook resume disiapkan; konfigurasi awal tetap cocok SHA256:',gate['config_sha256'])
