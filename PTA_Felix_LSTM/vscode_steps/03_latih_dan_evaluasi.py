"""LANGKAH 3 - Latih fold 1-3 dan evaluasi final holdout."""
from __future__ import annotations

from _common import CONFIG, MANIFEST, current_status, print_progress, run_module


print("=" * 72)
print("LANGKAH 3 - PELATIHAN FOLD DAN FINAL HOLDOUT")
print("=" * 72)

status = current_status()
if status is None:
    raise SystemExit("Jalankan 02_optuna_satu_trial.py terlebih dahulu.")
if status == "completed":
    print("Seluruh model dan evaluasi sudah completed; tidak dilatih ulang.")
else:
    if status not in {"tuned", "tuning_partial", "failed", "interrupted", "running"}:
        raise SystemExit(f"Status tidak dapat dilanjutkan: {status}")
    run_module(
        "train", "--config", str(CONFIG), "--stage", "all", "--resume"
    )

print_progress()
if current_status() != "completed":
    raise SystemExit("Run belum completed. Jalankan skrip ini lagi untuk melanjutkan.")
print("\nLANGKAH 3 SELESAI: 8 model, prediksi, bootstrap, grafik, dan laporan tersimpan.")
