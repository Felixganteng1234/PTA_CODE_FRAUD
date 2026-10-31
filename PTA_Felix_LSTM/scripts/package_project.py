"""Paket portabel kode, dataset, artefak smoke yang sudah diverifikasi, dan panduan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/PTA_Felix_LSTM_SiapJalan.zip"


def main():
    manifest = json.loads((ROOT / "results/smoke/run_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((ROOT / "results/smoke/verification.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and verification["status"] == "passed"
    mapping = []
    for folder in ["fraud_lstm", "configs", "tests", "notebooks", "vscode_steps", ".vscode", "data/raw", "data/processed", "results/smoke"]:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                mapping.append((path, path.relative_to(ROOT).as_posix()))
    for name in ["README.md", "HASIL_PENGERJAAN.md", "PERBANDINGAN_KODE_LAMA_DAN_BARU.md", "requirements.txt", ".gitignore", "PERSIAPAN.bat", "JALANKAN_UJI_CEPAT.bat", "JALANKAN_PENELITIAN.bat", "JALANKAN_HEMAT_1_TRIAL.bat"]:
        mapping.append((ROOT / name, name))
    for name in ["make_notebook.py", "verify_results.py", "package_project.py"]:
        mapping.append((ROOT / "scripts" / name, f"scripts/{name}"))
    original_dir = ROOT / "source_original/TUGASAKHIR_FRAUD"
    if not original_dir.exists():
        original_dir = ROOT / "referensi_kode_lama"
    for path in sorted(original_dir.iterdir()):
        if path.is_file() and path.suffix in {".py", ".json", ".md", ".csv"} and path.name != "PS.csv":
            mapping.append((path, f"referensi_kode_lama/{path.name}"))
    log_dir = ROOT / "tmp/review"
    if not (log_dir / "tests.log").exists():
        log_dir = ROOT / "validasi"
    mapping.append((log_dir / "tests.log", "validasi/tests.log"))
    mapping.append((log_dir / "smoke.log", "validasi/smoke.log"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    inventory = []
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path, relative in mapping:
            archive.write(path, "PTA_Felix_LSTM/" + relative)
            inventory.append({"path": relative, "bytes": path.stat().st_size})
        archive.writestr("PTA_Felix_LSTM/DAFTAR_FILE.json", json.dumps(inventory, indent=2))
    with zipfile.ZipFile(OUTPUT) as archive:
        damaged = archive.testzip()
        if damaged:
            raise RuntimeError(f"ZIP CRC gagal: {damaged}")
    digest = hashlib.sha256()
    with OUTPUT.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    (OUTPUT.parent / "PTA_Felix_LSTM_SHA256.txt").write_text(f"{checksum}  {OUTPUT.name}\n", encoding="ascii")
    print(json.dumps({"file": str(OUTPUT), "bytes": OUTPUT.stat().st_size, "files": len(inventory), "sha256": checksum, "zip_crc": "passed"}, indent=2))


if __name__ == "__main__":
    main()
