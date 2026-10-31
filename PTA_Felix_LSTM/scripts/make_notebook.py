"""Buat notebook panduan tanpa dependency Jupyter untuk proses packaging."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "source": text.splitlines(True), "outputs": [], "execution_count": None})


markdown("""# Penelitian PaySim — Felix Gunawan
Notebook ini mengarahkan tahapan implementasi di `fraud_lstm/`. Jalankan dengan kernel Python dari lingkungan proyek.
Hasil bawaan paket adalah **uji teknis smoke**, bukan hasil final skripsi. Panduan lengkap: `README.md`.
""")
code("""from pathlib import Path
import json, sys, subprocess
import numpy as np
import pandas as pd
ROOT = Path.cwd()
if not (ROOT / 'configs').exists():
    ROOT = ROOT.parent
assert (ROOT / 'configs/full.json').exists(), 'Buka notebook dari folder proyek/notebooks.'
sys.path.insert(0, str(ROOT))
from fraud_lstm.data import load_dataset, audit_dataset, prepare_dataset
print(ROOT)
""")
markdown("""## 1. Data dan pembentukan sequence
Populasi C, urutan akun/waktu/baris asli, window inclusive 3 transaksi. Lima nilai numerik dan lima one-hot type.
Dataset terproses sudah disertakan. Jika belum ada, sel berikut membangunnya dari CSV asli.
""")
code("""data_dir = ROOT / 'data/processed'
if not (data_dir / 'metadata.json').exists():
    prepare_dataset(ROOT / 'data/raw/PS.csv', data_dir)
data = load_dataset(data_dir)
print(json.dumps(data['metadata'], indent=2))
print('X:', data['X'].shape, 'y:', data['y'].shape)
""")
markdown("""## 2. Audit temporal dan target jumlah SMOTE
Kolom planned hanya hasil perhitungan target rasio, bukan bukti bahwa pelatihan full sudah dijalankan.
""")
code("""audit_dataset(data_dir, ROOT / 'results/notebook_audit')
""")
markdown("""## 3. Tuning, pelatihan dan evaluasi
Pilih `full` untuk eksperimen penelitian. Ubah RUN_EXPERIMENT menjadi True untuk mulai.
Full dapat memerlukan waktu lama pada CPU. Output yang sudah ada tidak ditimpa.
Pipeline: Optuna tiap kondisi -> 3 fold -> final holdout -> pooled metrics -> paired bootstrap -> grafik/laporan.
""")
code("""PROFILE = 'smoke'  # ganti 'full' untuk eksperimen seluruh data
RUN_EXPERIMENT = False
OUTPUT = 'results/notebook_run'
if RUN_EXPERIMENT:
    subprocess.run([sys.executable, '-m', 'fraud_lstm', 'run',
                    '--config', f'configs/{PROFILE}.json', '--output', OUTPUT],
                   cwd=ROOT, check=True)
else:
    print('Pelatihan belum dimulai. Ubah RUN_EXPERIMENT=True untuk menjalankan.')
""")
markdown("""## 4. Membaca hasil yang disertakan
Angka smoke di bawah hanya memastikan pipeline berjalan. Gunakan folder hasil full setelah eksperimen penuh selesai.
""")
code("""result_dir = ROOT / 'results/smoke'
manifest = json.loads((result_dir / 'run_manifest.json').read_text())
print('Profil:', manifest['config']['profile'], '| Status:', manifest['status'])
pd.read_csv(result_dir / 'metrics.csv')
""")
code("""pd.read_csv(result_dir / 'pooled_summary.csv')
""")
code("""pd.read_csv(result_dir / 'bootstrap_uncertainty.csv')
""")
markdown("""## 5. Menelusuri prediksi ke transaksi asal
target_row adalah nomor baris data berbasis 0 (header CSV tidak dihitung). Contoh row0 berarti baris data pertama.
""")
code("""with np.load(result_dir / 'predictions/pure_final_holdout.npz', allow_pickle=False) as saved:
    preview = pd.DataFrame({name: saved[name][:10] for name in
                            ['target_row', 'target_step', 'y_true', 'probability', 'account_id']})
preview
""")
markdown("""## Batas interpretasi
- Hanya 29,989% fraud mentah memenuhi syarat sequence.
- Saldo setelah transaksi membuat eksperimen ini klasifikasi setelah pencatatan.
- AP noninterpolasi dan PR-AUC trapezoidal berbeda; keduanya dilaporkan.
- CI bootstrap sequence tidak mengatasi ketergantungan akibat window overlap.
- Jangan gunakan hasil eksperimen lama sebagai hasil versi perbaikan.
""")
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks/01_alur_penelitian.ipynb"
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(destination)
