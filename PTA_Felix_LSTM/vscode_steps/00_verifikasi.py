"""LANGKAH 0 - Periksa Python, dependency, dataset, dan unit test."""
from __future__ import annotations

import importlib.metadata
import subprocess
import sys

from _common import CONFIG, ROOT, run_module


print("=" * 72)
print("LANGKAH 0 - VERIFIKASI LINGKUNGAN DAN DATASET")
print("=" * 72)
print("Python      :", sys.version.replace("\n", " "))
print("Executable  :", sys.executable)
print("Folder proyek:", ROOT)
print("Konfigurasi :", CONFIG)

for package in [
    "numpy", "pandas", "scikit-learn", "imbalanced-learn",
    "tensorflow", "keras", "optuna", "matplotlib", "joblib",
]:
    print(f"{package:18}: {importlib.metadata.version(package)}")

run_module("verify-source", "--config", str(CONFIG))

print("\nMenjalankan seluruh unit test...")
subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    cwd=ROOT,
    check=True,
)
print("\nLANGKAH 0 SELESAI: dependency, checksum dataset, dan pengujian lulus.")
