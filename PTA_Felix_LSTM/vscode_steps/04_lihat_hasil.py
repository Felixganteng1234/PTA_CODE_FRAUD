"""LANGKAH 4 - Tampilkan hasil CSV yang benar-benar tersimpan."""
from __future__ import annotations

import csv

from _common import OUTPUT, current_status, print_progress


print("=" * 72)
print("LANGKAH 4 - HASIL PENELITIAN")
print("=" * 72)
print_progress()

if current_status() != "completed":
    raise SystemExit("Hasil final belum tersedia. Selesaikan Langkah 2 dan 3.")

with (OUTPUT / "metrics.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))

print("\nMetrik setiap fold:")
header = f"{'Kondisi':<8} {'Fold':<14} {'AP':>9} {'Precision':>10} {'Recall':>9} {'F1':>9} {'TP':>6} {'FP':>7} {'FN':>6}"
print(header)
print("-" * len(header))
for row in rows:
    print(
        f"{row['condition']:<8} {row['fold']:<14} "
        f"{float(row['average_precision']):>9.5f} "
        f"{float(row['precision']):>10.5f} "
        f"{float(row['recall']):>9.5f} "
        f"{float(row['f1']):>9.5f} "
        f"{int(row['tp']):>6} {int(row['fp']):>7} {int(row['fn']):>6}"
    )

print("\nFile hasil:")
for path in [
    OUTPUT / "LAPORAN_HASIL.md",
    OUTPUT / "laporan.html",
    OUTPUT / "metrics.csv",
    OUTPUT / "pooled_summary.csv",
    OUTPUT / "bootstrap_uncertainty.csv",
    OUTPUT / "plots" / "precision_recall.png",
]:
    print(" ", path)

print("\nBaca laporan sebelum menulis kesimpulan. Satu trial adalah konfigurasi tetap, bukan optimasi TPE.")
