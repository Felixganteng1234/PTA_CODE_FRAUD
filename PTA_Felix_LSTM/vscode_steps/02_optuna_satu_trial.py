"""LANGKAH 2 - Satu trial maksimal 30 epoch untuk Pure dan SMOTE."""
from __future__ import annotations

from _common import CONFIG, MANIFEST, OUTPUT, current_status, print_progress, read_json, run_module


print("=" * 72)
print("LANGKAH 2 - OPTUNA SATU TRIAL PER KONDISI")
print("=" * 72)
print("Pure LSTM   : 1 trial, maksimal 30 epoch")
print("LSTM+SMOTE  : 1 trial, maksimal 30 epoch")
print("Parameter awal: lr=0.001, dropout=0.2, batch=512, patience=5")
print("Validasi internal bersih: target_step 341-400")

status = current_status()
if status == "completed":
    print("Run sudah completed; tuning tidak diulang.")
elif status == "tuned":
    print("Tuning kedua kondisi sudah selesai; lanjutkan ke Langkah 3.")
else:
    arguments = ["train", "--config", str(CONFIG), "--stage", "tune"]
    if MANIFEST.exists():
        arguments.append("--resume")
        print(f"Melanjutkan run dengan status sebelumnya: {status}")
    run_module(*arguments)

print_progress()
manifest = read_json(MANIFEST)
print("\nStatus manifest:", manifest["status"])
print("Trial selesai:", manifest.get("tuning_completed_trials"))
for condition in ["pure", "smote"]:
    best = read_json(OUTPUT / "tuning" / condition / "best_params.json")
    print(f"\n{condition.upper()}")
    print("  parameter       :", best["parameters"])
    print("  AP validasi     :", best["objective_average_precision"])
    print("  epoch terbaik   :", best["selected_epoch"])
    print("  trial selesai   :", f"{best['completed_trials']}/{best['target_trials']}")

print("\nLANGKAH 2 SELESAI. Belum ada fold test atau final holdout yang dibuka.")
