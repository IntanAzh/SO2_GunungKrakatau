"""Jalankan sel notebook dengan IPython dalam .venv, lalu simpan outputnya."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs' / '.mplconfig'))
os.environ.setdefault('IPYTHONDIR', str(ROOT / 'outputs' / '.ipython'))
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

path = ROOT / 'notebooks' / (sys.argv[1] if len(sys.argv) > 1 else '01_audit_data.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
shell = InteractiveShell.instance()
count = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    count += 1
    print(f'Menjalankan sel {count}...', flush=True)
    with capture_output() as captured:
        result = shell.run_cell(cell['source'], store_history=True)
    cell['execution_count'] = count
    cell['outputs'] = []
    for name in ('stdout', 'stderr'):
        content = getattr(captured, name)
        if content:
            cell['outputs'].append({'output_type': 'stream', 'name': name, 'text': content})
    for rich in captured.outputs:
        cell['outputs'].append({'output_type': 'display_data', 'data': rich.data, 'metadata': rich.metadata})
    error = result.error_before_exec or result.error_in_exec
    if error:
        cell['outputs'].append({'output_type': 'error', 'ename': type(error).__name__,
                                'evalue': str(error), 'traceback': []})
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    if error:
        raise RuntimeError(f'Sel {count} gagal: {error}') from error
print(f'Selesai: {count} sel kode, output tersimpan di notebook.')
