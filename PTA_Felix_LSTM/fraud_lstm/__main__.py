"""Antarmuka perintah: python -m fraud_lstm --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .data import audit_dataset, prepare_dataset, sha256_file


def read_config(path: Path) -> tuple[dict, Path]:
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("profile") not in {"full", "smoke"}:
        raise ValueError("profile harus full atau smoke.")
    for name in ["n_trials", "max_epochs", "bootstrap_iterations", "cpu_threads"]:
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} harus bilangan bulat positif.")
    if config.get("smote_ratio") != 0.5 or config.get("smote_k") != 5:
        raise ValueError("Rancangan menetapkan SMOTE ratio=0.5 dan k=5.")
    if config.get("threshold") != 0.5:
        raise ValueError("Evaluasi utama dibekukan pada threshold 0.5; jangan menyetel dari data uji.")
    if type(config.get("seed")) is not int:
        raise ValueError("seed harus bilangan bulat.")
    if type(config.get("steps_per_execution", 1)) is not int or config.get("steps_per_execution", 1) < 1:
        raise ValueError("steps_per_execution harus bilangan bulat positif.")
    if config["profile"] == "full" and config["max_epochs"] <= 3:
        raise ValueError("Profil full memerlukan max_epochs > 3 agar melampaui patience minimum proposal. Gunakan smoke untuk tes pendek.")
    return config, path.parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Penelitian PaySim: Pure LSTM vs LSTM+SMOTE.")
    parser.add_argument("command", choices=["prepare", "audit", "train", "run", "report", "verify-source", "diagnose", "status"],
                        help="run=prepare jika belum ada, train, lalu report")
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--output", type=Path, help="Folder hasil baru, relatif root proyek atau absolut")
    parser.add_argument("--resume", action="store_true", help="Lanjutkan run dengan identitas data dan konfigurasi yang sama")
    parser.add_argument("--stage", choices=["tune", "all"], help="Jalankan hanya tuning, atau seluruh rangkaian")
    parser.add_argument("--trial-budget", type=int, help="Batas trial baru per kondisi pada pemanggilan ini; target total trial tidak berubah")
    args = parser.parse_args(argv)
    config, root = read_config(args.config)
    if args.resume:
        config["resume"] = True
    if args.stage:
        config["run_stage"] = args.stage
    if args.trial_budget is not None:
        if args.trial_budget < 1:
            raise ValueError("trial-budget harus positif.")
        config["trial_budget"] = args.trial_budget
    data_dir = root / config["data_dir"]
    dataset = root / config["dataset"]
    output = root / (args.output or Path(config["output_dir"]))
    if args.command == "status":
        for filename in ["run_manifest.json", "progress.json"]:
            path = output / filename
            if path.exists():
                print(path.read_text(encoding="utf-8"))
        return 0
    if args.command == "diagnose":
        from .diagnostics import summarize_training
        print(json.dumps(summarize_training(data_dir, config, output), indent=2, ensure_ascii=False))
        return 0
    if args.command in {"prepare", "run"}:
        if args.command == "prepare" or not (data_dir / "metadata.json").exists():
            prepare_dataset(dataset, data_dir)
    if args.command == "audit":
        print(audit_dataset(data_dir, output).to_string(index=False))
    if args.command in {"verify-source", "train", "run"}:
        expected = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))["source_sha256"]
        actual = sha256_file(dataset)
        if actual != expected:
            raise ValueError("Checksum CSV berbeda dari data yang diproses. Bangun ulang di folder baru.")
        print(f"SHA-256 CSV cocok: {actual}")
    if args.command in {"train", "run"}:
        from .training import run_experiment
        print(f"Profil {config['profile']}; hasil: {output}", flush=True)
        if config["profile"] == "smoke":
            print("UJI TEKNIS DENGAN SUBSET LATIH. BUKAN HASIL FINAL PENELITIAN.", flush=True)
        run_experiment(data_dir, output, config)
        status = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["status"]
        if status != "completed":
            print(f"Tahap selesai dengan status {status}. Model final belum dievaluasi; lanjutkan dengan --resume.", flush=True)
            return 0
    if args.command in {"report", "train", "run"}:
        from .report import build_report
        print(build_report(data_dir, output), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"GAGAL: {exc}", file=sys.stderr)
        sys.exit(1)
