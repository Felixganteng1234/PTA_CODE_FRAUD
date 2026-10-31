"""Fungsi bersama untuk skrip langkah demi langkah di VS Code."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "full_single_trial.json"
OUTPUT = ROOT / "results" / "full_single_trial"
MANIFEST = OUTPUT / "run_manifest.json"
PROGRESS = OUTPUT / "progress.json"


def run_module(*arguments: str) -> None:
    command = [sys.executable, "-m", "fraud_lstm", *arguments]
    print("\nMenjalankan:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def current_status() -> str | None:
    return read_json(MANIFEST).get("status") if MANIFEST.exists() else None


def print_progress() -> None:
    if not PROGRESS.exists():
        print("Belum ada progress.json.")
        return
    progress = read_json(PROGRESS)
    fields = [
        "status", "stage", "event", "condition", "fold", "trial",
        "epoch", "max_epochs", "best_epoch", "best_loss", "updated_at_utc",
    ]
    print("\nStatus proses terakhir:")
    for field in fields:
        if progress.get(field) is not None:
            print(f"  {field:15}: {progress[field]}")
