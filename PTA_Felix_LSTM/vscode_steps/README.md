# Menjalankan penelitian langkah demi langkah di VS Code

## Persiapan

1. Buka folder `PTAFELIX` melalui **File > Open Folder** di VS Code.
2. Pasang ekstensi **Python** dari Microsoft.
3. Pilih interpreter Python melalui `Ctrl+Shift+P` → **Python: Select Interpreter**. Jika sudah menjalankan `PERSIAPAN.bat`, pilih `.venv\Scripts\python.exe`.
4. Buka folder `vscode_steps` pada Explorer VS Code.

## Urutan eksekusi

Jalankan setiap file menggunakan tombol **Run Python File** di kanan atas:

1. `00_verifikasi.py` — memeriksa dependency, checksum dataset, dan 56+ unit test.
2. `01_diagnosis_data.py` — menampilkan jumlah data latih dan validasi tanpa melatih model.
3. `02_optuna_satu_trial.py` — menjalankan satu trial Pure LSTM dan satu trial LSTM+SMOTE, masing-masing maksimal 30 epoch.
4. `03_latih_dan_evaluasi.py` — menggunakan parameter trial untuk tiga fold dan final holdout.
5. `04_lihat_hasil.py` — menampilkan metrik aktual dan lokasi laporan.

Jangan menjalankan Langkah 2 atau 3 secara bersamaan di dua terminal. Sistem lock akan menolak proses kedua. Jika VS Code atau komputer berhenti, jalankan kembali file yang sama; checkpoint yang lengkap akan digunakan kembali.

Satu trial per kondisi menghemat pencarian Optuna, tetapi tidak menghilangkan pelatihan Fold 1, Fold 2, Fold 3, dan final holdout yang diwajibkan rancangan evaluasi temporal.

## Hasil

Semua hasil mode ini disimpan di `results/full_single_trial/`. `progress.json` menunjukkan epoch/tahap terakhir dan `run_manifest.json` menunjukkan status run. Hasil final hanya sah jika status manifest adalah `completed`.
