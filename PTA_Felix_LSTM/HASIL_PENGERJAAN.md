# Hasil pengerjaan coding tugas akhir Felix Gunawan

**Kode dan pengaturan dataset selesai. Uji teknis seluruh alur telah selesai. Eksperimen penelitian penuh belum dijalankan.**

## Yang tersedia

1. Kode Python modular untuk preprocessing, sequence, pembagian waktu, StandardScaler, SMOTE, Stacked-LSTM, Optuna, evaluasi, bootstrap dan laporan.
2. Dataset asli `data/raw/PS.csv` dan data terproses `data/processed/` yang dapat langsung digunakan.
3. Konfigurasi `configs/full.json` sesuai rancangan penelitian serta `configs/smoke.json` untuk uji teknis.
4. Notebook `notebooks/01_alur_penelitian.ipynb` dan panduan `README.md`.
5. Skrip Windows `PERSIAPAN.bat`, `JALANKAN_UJI_CEPAT.bat`, dan `JALANKAN_PENELITIAN.bat`.
6. Delapan model uji teknis beserta scaler, prediksi, grafik, log pelatihan dan hasil bootstrap 1.000 kali di `results/smoke/`.

## Hasil yang benar-benar sudah dijalankan

- Seluruh **6.362.620 transaksi** dibaca dan diperiksa.
- Dibentuk **3.180.599 sequence**, termasuk **2.463 target fraud**. Jumlah setiap fold sama dengan proposal.
- **33 pengujian otomatis lulus**, meliputi kausalitas urutan, batas waktu, scaler, SMOTE, metrik, bootstrap dan konsistensi laporan.
- Profil smoke selesai: 2 trial Optuna per kondisi, maksimal 2 epoch, 30.000 sequence latih per partisi, seluruh validasi dan pengujian tetap memakai data asli.
- Tiga fold dan holdout untuk kedua kondisi selesai; bootstrap berpasangan 1.000 kali selesai.
- Durasi eksperimen smoke yang tercatat pada manifest: **244,103 detik**, sekitar **4,07 menit**, menggunakan CPU. Ini tidak termasuk persiapan data dan penyusunan kode.

Uji smoke sudah menghasilkan laporan di [laporan HTML](results/smoke/laporan.html), [laporan Markdown](results/smoke/LAPORAN_HASIL.md), dan [tabel metrik](results/smoke/metrics.csv).

## Batas hasil saat ini

**Angka smoke bukan hasil final skripsi.** Latihnya memakai subset dan epoch sangat terbatas. Pada holdout smoke, Pure LSTM tidak mendeteksi fraud pada threshold 0,5; LSTM+SMOTE mendeteksi sebagian fraud tetapi menghasilkan banyak false positive. Ini menunjukkan artefak hasil nyata, bukan klaim bahwa salah satu metode sudah terbukti unggul pada eksperimen final.

Eksperimen penuh tersedia melalui **JALANKAN_PENELITIAN.bat** atau:

```powershell
python -m fraud_lstm run --config configs/full.json
```

Alternatif hemat komputasi tersedia melalui **JALANKAN_HEMAT_1_TRIAL.bat**. Alternatif tersebut memakai seluruh data dengan satu konfigurasi parameter yang ditetapkan sebelumnya; hasilnya sah sebagai eksperimen konfigurasi tetap, tetapi tidak boleh ditulis sebagai hasil optimasi hyperparameter TPE.

Profil full memakai seluruh data latih, 20 trial per kondisi, maksimal 30 epoch, dan bootstrap 1.000 kali. Hasilnya akan dibuat di `results/full/` dan dapat dipakai untuk penyusunan hasil penelitian setelah diperiksa. Komputer ini terdeteksi menggunakan TensorFlow CPU; eksperimen penuh membutuhkan waktu jauh lebih lama daripada uji smoke.

## Perbaikan utama dari kode awal

- Memisahkan validasi temporal **sebelum** scaler dan SMOTE. Kode lama melakukan `validation_split` setelah resampling.
- Menyesuaikan ruang pencarian dropout, batch size dan patience dengan proposal.
- Menyimpan prediksi yang sama untuk tabel evaluasi dan bootstrap sehingga tidak ada pelatihan ulang diam-diam saat menghitung CI.
- Menambahkan pooled Precision/Recall/F1, mean dan SD AP antar-fold, model/scaler tersimpan, serta metadata transaksi asal dan checksum dataset.
- Mengganti path Downloads yang dikunci pada satu komputer dengan konfigurasi relatif yang portabel.

Penjelasan semua keputusan implementasi dan cara menjalankan tersedia di [README](README.md). Kode serta hasil lama dalam paket berada di `referensi_kode_lama/` dan hanya menjadi arsip.
