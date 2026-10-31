"""LANGKAH 1 - Audit jumlah data tanpa melatih atau membuka holdout."""
from __future__ import annotations

from _common import CONFIG, OUTPUT, read_json, run_module


print("=" * 72)
print("LANGKAH 1 - DIAGNOSIS DATA LATIH DAN VALIDASI INTERNAL")
print("=" * 72)

run_module("diagnose", "--config", str(CONFIG), "--output", str(OUTPUT))
diagnostic = read_json(OUTPUT / "training_diagnostics.json")

print("\nRingkasan anggaran:")
print("  Profil                 :", diagnostic["profile"])
print("  Trial per kondisi      :", diagnostic["budget"]["n_trials_per_condition"])
print("  Maksimum epoch/trial   :", diagnostic["budget"]["max_epochs"])
print("  Rentang patience       :", diagnostic["budget"]["patience_range"])

print("\nJumlah data yang benar-benar akan digunakan:")
for name, partition in diagnostic["partitions"].items():
    selected = partition["selected_training"]["original"]
    validation = partition["internal_validation"]
    print(
        f"  {name:12} latih={selected['n']:,} fraud_latih={selected['fraud']:,} "
        f"validasi={validation['n']:,} fraud_validasi={validation['fraud']:,}"
    )

print("\nPeringatan metodologi:")
for warning in diagnostic["budget"]["warnings"]:
    print(" -", warning["message"])
    print("   ", warning["action"])

print("\nLANGKAH 1 SELESAI: belum ada model atau data pengujian yang dievaluasi.")
