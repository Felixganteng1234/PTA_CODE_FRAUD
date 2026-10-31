# Tugas Akhir Felix Gunawan — PaySim, LSTM dan SMOTE

Implementasi penelitian **Pengaruh SMOTE terhadap Performa LSTM dalam Deteksi Fraud pada Dataset PaySim**, mengikuti proposal `PTA_FelixGunawan.pdf`, khususnya halaman PDF 59–70 (halaman cetak 52–63).

## Mulai dari sini

1. Ekstrak seluruh ZIP proyek ke satu folder.
2. Buka **HASIL_PENGERJAAN.md** untuk melihat apa yang sudah dijalankan dan apa yang masih perlu dijalankan.
3. Buka **results/smoke/laporan.html** untuk hasil uji teknis yang disertakan. Angka mode smoke **bukan hasil final skripsi**.
4. Untuk menjalankan pada komputer lain, pasang Python 3.10 atau 3.11 lalu klik **PERSIAPAN.bat**. Skrip membuat lingkungan `.venv` dan mengunduh dependency; tidak mengubah Python global.
5. Klik **JALANKAN_PENELITIAN.bat** untuk eksperimen penuh. Hasil baru muncul di `results/full/`.

Untuk menjalankan dan memahami setiap tahap melalui VS Code, buka `vscode_steps/README.md`, lalu jalankan skrip `00` sampai `04` secara berurutan menggunakan tombol **Run Python File**.

Jika beban komputasi harus dibatasi, klik **JALANKAN_HEMAT_1_TRIAL.bat**. Mode ini tetap memakai seluruh data dan prosedur proposal, tetapi menjalankan satu konfigurasi tetap (`lr=0,001`, `dropout=0,2`, `batch=512`, `patience=5`) untuk kedua kondisi. Secara akademik ini bukan optimasi hyperparameter TPE; proposal/laporan harus menyebutnya sebagai parameter yang ditetapkan sebelumnya.

Pada komputer saat penyusunan, dependency sudah tersedia pada Python sistem. TensorFlow mendeteksi **CPU saja**. Eksperimen penuh melibatkan jutaan sequence, 20 trial per kondisi dan beberapa pelatihan ulang; durasinya jauh lebih lama daripada smoke. Tidak ada waktu selesai yang dijamin. GPU TensorFlow tidak tersedia melalui native Windows pada runtime ini; gunakan lingkungan GPU yang kompatibel jika diperlukan.

## Perintah terminal

Jalankan dari folder proyek. Jika memakai virtual environment, ganti `python` dengan `.venv\Scripts\python.exe` (Windows).

```powershell
python -m unittest discover -s tests -v
python -m fraud_lstm verify-source --config configs/full.json
python -m fraud_lstm audit --config configs/full.json --output results/audit
python -m fraud_lstm run --config configs/smoke.json --output results/smoke_baru
python -m fraud_lstm run --config configs/full.json
python -m fraud_lstm run --config configs/full_single_trial.json
```

`run` otomatis membentuk data jika belum ada, menjalankan tuning/pelatihan/evaluasi, lalu membuat laporan. Data terproses sudah disertakan dalam paket, sehingga tidak perlu dibangun ulang. Untuk membentuk ulang, pilih `data_dir` baru pada salinan konfigurasi, kemudian jalankan `prepare` atau `run`. Perintah tidak menimpa folder data yang sudah terisi atau eksperimen yang sudah memiliki manifest.

Tahap tersedia: `prepare`, `audit`, `train`, `run`, `report`, `verify-source`. Contoh regenerasi laporan tanpa melatih ulang:

```powershell
python -m fraud_lstm report --config configs/smoke.json
```

File konfigurasi diletakkan dalam folder `configs/`. Path dataset, data dan hasil di dalam JSON dihitung dari root proyek (induk folder `configs`), sehingga tidak bergantung pada path Downloads pengguna.

## Alur sesuai proposal

| Tahap | Implementasi |
|---|---|
| Data mentah | `data/raw/PS.csv`; validasi kolom, nilai hilang, label biner, jenis transaksi dan waktu |
| Populasi | `nameDest` diawali C; seluruh transaksi M diperiksa tidak memiliki fraud |
| Urutan | `nameDest`, `step`, indeks baris asli sebagai pemecah waktu sama |
| Sequence | `[T1,T2,T3]`, panjang 3, stride 1, target `isFraud(T3)` |
| Fitur | Lima nilai numerik + lima one-hot `type`; total 10 |
| Yang dikecualikan | Label, `isFlaggedFraud`, ID akun dan `step` tidak menjadi fitur model |
| Standardisasi | StandardScaler hanya lima fitur numerik; fit pada latih yang relevan |
| SMOTE | Flatten `(N,3,10)` ke `(N,30)`, ratio 0.5, k=5, reshape; hanya data latih |
| Model | LSTM64 → Dropout → LSTM32 → Dropout → Dense1 sigmoid |
| Loss/optimizer | Binary cross-entropy / Adam, tanpa class weight dan focal loss |
| Tuning | Optuna TPE terpisah per kondisi; latih ≤340, validasi 341–400 |
| Ruang pencarian | LR 0.0001–0.01 (log), dropout 0.2–0.5, batch 128/256/512, patience 3–10 |
| Pengujian | Fold waktu dan holdout asli tanpa resampling |

`step` hanya menentukan urutan dan partisi. Kamus one-hot tetap mengikuti lima jenis PaySim; bukan belajar kategori dari data uji. Karena SMOTE standar menginterpolasi seluruh vektor, one-hot sintetis dapat bernilai pecahan; ini konsekuensi metode dalam proposal. Duplikat baris dipertahankan: tanpa ID transaksi unik tidak dapat dipastikan bahwa baris identik adalah duplikasi pencatatan.

### Pembagian waktu dan jumlah aktual

| Tahap | Latih | Uji | N latih | Fraud latih | N uji | Fraud uji |
|---|---|---|---:|---:|---:|---:|
| Fold 1 | ≤400 | 401–480 | 2.901.650 | 1.349 | 123.186 | 251 |
| Fold 2 | ≤480 | 481–560 | 3.024.836 | 1.600 | 62.086 | 260 |
| Fold 3 | ≤560 | 561–640 | 3.086.922 | 1.860 | 53.119 | 261 |
| Final | ≤640 | 641–743 | 3.140.041 | 2.121 | 40.558 | 342 |

Data mentah: **6.362.620 transaksi**, **8.213 fraud**. Data sequence: **3.180.599**, **2.463 fraud**. Cakupan fraud **29,989%**. Hasil hanya berlaku pada transaksi C yang memiliki dua transaksi sebelumnya dalam kelompok akun; tidak otomatis mewakili seluruh transaksi PaySim.

### Pencegahan kebocoran

Sebelum pemilihan epoch, 60 step terakhir dari rentang latih menjadi validasi internal yang bersih:

- Fold 1: latih ≤340, validasi 341–400.
- Fold 2: latih ≤420, validasi 421–480.
- Fold 3: latih ≤500, validasi 501–560.
- Final: latih ≤580, validasi 581–640.

Scaler dan SMOTE pada tahap ini hanya menggunakan bagian latih awal. Early stopping memilih epoch terbaik melalui `val_loss`. Model baru kemudian dilatih ulang dengan epoch terpilih menggunakan seluruh data latih partisi (atau subset seragam dalam smoke), serta scaler baru yang di-fit pada latih tersebut. Uji luar dan holdout tidak digunakan memilih epoch, hyperparameter atau threshold. Tidak ada `validation_split` setelah SMOTE.

Sequence validasi boleh memakai riwayat dari periode sebelumnya. Ini tidak memakai label historis sebagai fitur dan tidak memasukkan transaksi setelah target. Pemecah seri indeks baris memberi urutan konsisten, bukan bukti urutan waktu aktual dalam jam yang sama.

## Keputusan implementasi untuk detail yang belum ditetapkan

- Profil penuh: **20 trial per kondisi**, maksimum **30 epoch**, seed **42**. Angka ini dapat diedit sebelum eksperimen dan selalu dicatat di manifest.
- Profil hemat: **1 trial per kondisi** dengan satu konfigurasi yang sama dan telah ditetapkan sebelum evaluasi. Ini menghemat tuning, tetapi tidak boleh disebut pencarian hyperparameter terbaik.
- Threshold **0.5** ditetapkan sebelum evaluasi, bukan dipilih dari test/holdout.
- Objective Optuna adalah **Average Precision (AP)** noninterpolasi. Laporan juga menyertakan **PR-AUC trapezoidal** secara terpisah; keduanya bukan metrik identik.
- Precision/recall/F1 pengembangan dihitung dari prediksi tiga fold yang digabung. AP dilaporkan mean dan sample SD (`ddof=1`) antar-fold.
- Bootstrap **1.000 kali**, percentile 95%, indeks resampling sama untuk Pure dan SMOTE. Selisih didefinisikan SMOTE dikurangi Pure. Bootstrap memakai prediksi tersimpan, tidak melatih model baru.
- Bootstrap per sequence mengasumsikan independensi, padahal window bisa tumpang tindih. Modul metrik menyediakan opsi cluster akun untuk analisis tambahan, tetapi pipeline default memakai bootstrap sequence sesuai keputusan eksplisit ini.
- Data memiliki nilai saldo setelah transaksi (`newbalance*`), termasuk transaksi target. Ini klasifikasi setelah transaksi tercatat, bukan sistem pencegahan sebelum transaksi disetujui.

Rujukan API: [SMOTE sampling_strategy](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html), [definisi Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

## Smoke dan full

| | Smoke | Full |
|---|---|---|
| Tujuan | Verifikasi perangkat lunak | Eksperimen penelitian |
| Latih | Subset acak seragam 30.000/partisi; dapat diperbesar jika fraud tidak cukup untuk k=5 | Seluruh data latih |
| Validasi/uji | Seluruh data asli | Seluruh data asli |
| Trial/kondisi | 2 | 20 |
| Max epoch | 2 | 30 |
| Bootstrap | 1.000 | 1.000 |
| Dapat jadi hasil akhir skripsi | Tidak | Setelah selesai dan ditinjau |

Subset latih smoke sama untuk kedua kondisi dan tidak sengaja memperkaya kelas fraud. Detail draw, seed dan ukuran aktual tersimpan. Karena ukuran kecil dan epoch dibatasi, model smoke dapat gagal mendeteksi fraud; ini tidak membuktikan kegagalan atau keberhasilan metode penelitian.

## File keluaran

- `run_manifest.json`: status, konfigurasi, checksum konfigurasi, runtime, metadata dataset dan model yang sudah selesai.
- `metrics.csv`: hasil per kondisi dan per fold, confusion matrix dan ukuran latih/uji.
- `pooled_summary.csv`: pooled Precision/Recall/F1 serta mean/SD AP antar-fold.
- `bootstrap_uncertainty.csv`: CI 95% tiap kondisi dan selisihnya.
- `predictions/*.npz`: probabilitas, label asli, target step, indeks sequence, baris asal dan akun.
- `models/<kondisi>/<fold>/model.keras`: delapan model; scaler, history dan catatan pelatihannya berada di folder sama.
- `tuning/<kondisi>/`: trial, histori dan hyperparameter terbaik kondisi tersebut.
- `selection/` dan `selection_counts.json`: indeks dan jumlah data terpilih.
- `plots/`: kurva PR dan confusion matrix holdout.
- `LAPORAN_HASIL.md` dan `laporan.html`: ringkasan yang dibuat dari hasil aktual.

Tabel `data/processed/dataset_audit.csv` memuat `planned_*_after_smote`: **jumlah target secara matematis**, bukan klaim resampling penuh sudah dieksekusi. Jumlah resampling yang benar-benar dilakukan ada pada catatan pelatihan setiap run.

## Kode lama

Kode dan hasil lama dari ZIP pengguna dipertahankan di `referensi_kode_lama/` dalam paket. Jangan menggunakan `evaluation_results.csv` atau `best_params_*.json` lama sebagai hasil versi ini. Versi lama memakai validasi setelah SMOTE dan ruang pencarian yang berbeda dari proposal final. Hasil baru tidak disalin dari CSV lama.

## Menggunakan model tersimpan

```python
import joblib
import numpy as np
import tensorflow as tf

folder = "results/full/models/pure/final_holdout/"
model = tf.keras.models.load_model(folder + "model.keras")
scaler = joblib.load(folder + "scaler.joblib")  # hanya artefak buatan sendiri
# X_raw: float32 (N,3,10), urutan fitur sama dengan metadata.json.
X = X_raw.copy()
X[:, :, :5] = scaler.transform(X[:, :, :5].reshape(-1, 5)).reshape(-1, 3, 5)
probabilitas = model.predict(X).ravel()
prediksi = (probabilitas >= 0.5).astype(np.int8)
```

`X_raw` harus dibentuk dengan aturan akun/urutan yang sama, tidak boleh diisi satu transaksi lalu dipaksa menjadi tiga. Setiap model harus dipasangkan dengan scaler dari folder model yang sama. Model dan scaler dari profil smoke hanya untuk demonstrasi teknis.
