# Perbandingan kode lama dan kode baru

## Kesimpulan

Kode lama tidak salah seluruhnya: pembentukan sequence, arsitektur LSTM, dan penerapan SMOTE train-only pada beberapa tahap sudah mengarah benar. Namun, kode lama tidak sebaiknya dipakai tanpa revisi karena prosedur validasi evaluasinya dapat menghasilkan penilaian yang bias dan beberapa parameter tuning tidak sama dengan proposal final.

## Perbedaan utama

| Bagian | Kode lama | Kode baru | Dampak |
|---|---|---|---|
| Validasi evaluasi | `validation_split=0.1` setelah SMOTE | Validasi temporal dipisahkan sebelum scaler dan SMOTE | Validasi baru tetap bersih dan tidak didominasi data sintetis |
| Urutan validasi | Ekor array, bukan batas waktu yang eksplisit | 60 step terakhir dari rentang latih | Menjaga urutan waktu |
| Data tuning | Hanya 10% kelas normal (`TUNE_NEG_FRAC=0.10`) | Profil full memakai seluruh data latih tuning | Distribusi tuning sesuai populasi rancangan |
| Dropout | 0,0–0,5 | 0,2–0,5 | Sesuai proposal |
| Batch size | 512 atau 1024 | 128, 256, atau 512 | Sesuai proposal |
| Patience | 2–6 | 3–10 | Sesuai proposal |
| Hasil parameter lama | Dropout 0,0697 dan batch 1024 | Ditolak oleh ruang proposal final | Parameter lama tidak dipakai sebagai “terbaik” versi baru |
| Final holdout | Dievaluasi, tetapi pencatatan artefak terbatas | Prediksi, model, scaler, histori, checksum, dan CI tersimpan | Hasil dapat diaudit dan dilanjutkan |
| Proses terhenti | Harus banyak mengulang | Trial dan model selesai dapat dilanjutkan dengan `--resume` | Mengurangi pemborosan komputasi |
| Metrik lintas fold | Belum lengkap | Precision/Recall/F1 pooled, AP mean dan sample SD | Sesuai rancangan pelaporan |

## Tentang satu trial

`n_trials=1` diperbolehkan secara teknis, tetapi satu trial bukan pencarian hyperparameter. Kode baru memperlakukannya sebagai satu konfigurasi yang ditetapkan sebelum evaluasi:

- learning rate: 0,001;
- dropout: 0,2;
- batch size: 512;
- patience: 5;
- seed: 42.

Nilai tersebut berada dalam batas proposal dan digunakan sama untuk Pure LSTM serta LSTM+SMOTE. Hal ini membuat perbandingan perlakuan SMOTE lebih mudah dibaca, tetapi hasilnya tidak boleh disebut “hyperparameter optimal” atau “hasil optimasi TPE”. Kalimat metode yang tepat:

> Untuk membatasi beban komputasi, penelitian menggunakan satu konfigurasi hyperparameter yang ditetapkan sebelum evaluasi. Optuna digunakan untuk menjalankan dan mencatat satu trial pada masing-masing kondisi, bukan untuk melakukan pencarian hyperparameter.

Jika proposal tetap menyatakan “pencarian hyperparameter optimal menggunakan TPE”, gunakan `configs/full.json` dengan beberapa trial. Jika memilih satu trial, perbarui bagian metodologi dengan kalimat di atas dan gunakan `configs/full_single_trial.json`.

## Menjalankan mode hemat

Klik `JALANKAN_HEMAT_1_TRIAL.bat`, atau jalankan:

```powershell
python -m fraud_lstm run --config configs/full_single_trial.json
```

Jika berhenti di tengah proses:

```powershell
python -m fraud_lstm run --config configs/full_single_trial.json --resume
```

Perintah tersebut memakai seluruh data latih. Ini berbeda dari smoke test yang hanya memakai 30.000 sequence per partisi.
