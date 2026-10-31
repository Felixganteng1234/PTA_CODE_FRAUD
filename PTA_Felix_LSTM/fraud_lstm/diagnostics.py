"""Kesiapan pelatihan dari train/validasi internal; tidak membaca hasil pengujian.

Modul ini tidak mengimpor TensorFlow dan tidak menyimpulkan kualitas model dari
profil smoke. Nama ``final_train`` hanya mencakup target_step <= 640.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .data import load_dataset, write_json
from .training import _counts, _select_train

PATIENCE_RANGE = (3, 10)
BATCH_SIZES = (128, 256, 512)
TRAINING_PARTITIONS = (("fold1", 400), ("fold2", 480), ("fold3", 560), ("final_train", 640))


def assess_training_budget(config: dict) -> dict:
    """Peringatan diagnostik, bukan perubahan atau penolakan konfigurasi."""
    profile = config.get("profile", "full")
    max_epochs = int(config.get("max_epochs", 30))
    n_trials = int(config.get("n_trials", 20))
    warnings = []
    if profile == "smoke":
        warnings.append({
            "code": "smoke_is_software_check",
            "message": "Smoke memakai subset data latih dan hanya memeriksa alur perangkat lunak.",
            "action": "Gunakan profil full untuk eksperimen penelitian; jangan membuat klaim kualitas dari smoke.",
        })
    if max_epochs <= PATIENCE_RANGE[1]:
        warnings.append({
            "code": "epoch_cap_not_above_max_patience",
            "message": f"Batas {max_epochs} epoch tidak melebihi patience maksimum {PATIENCE_RANGE[1]}; sebagian trial tidak dapat menghabiskan patience setelah epoch awal.",
            "action": "Gunakan anggaran profil full (30 epoch) dan biarkan early stopping memakai validasi internal bersih.",
        })
    if n_trials == 1:
        warnings.append({
            "code": "single_prespecified_configuration",
            "message": "Satu trial bukan pencarian hyperparameter TPE; hasil hanya menilai satu konfigurasi yang ditetapkan sebelumnya.",
            "action": "Gunakan istilah konfigurasi tetap dan jangan menyebutnya hyperparameter optimal. Gunakan beberapa trial bila proposal tetap menyatakan optimasi TPE.",
        })
    elif n_trials < 10:
        warnings.append({
            "code": "limited_hyperparameter_search",
            "message": f"Hanya {n_trials} trial per kondisi; cakupan pencarian hyperparameter masih terbatas.",
            "action": "Gunakan konfigurasi full multi-trial bila ingin mempertahankan klaim pencarian TPE; jumlah trial bukan jaminan hasil optimal.",
        })
    return {
        "profile": profile,
        "max_epochs": max_epochs,
        "n_trials_per_condition": n_trials,
        "patience_range": list(PATIENCE_RANGE),
        "epoch_cap_exceeds_max_patience": max_epochs > PATIENCE_RANGE[1],
        "research_budget_warning": max_epochs <= PATIENCE_RANGE[1],
        "smoke_quality_claims_allowed": False,
        "performance_certified": False,
        "warnings": warnings,
    }


def _training_counts(y, config):
    before = _counts(y)
    target_fraud = int(before["nonfraud"] * float(config["smote_ratio"]))
    feasible = before["fraud"] >= int(config["smote_k"]) + 1 and target_fraud > before["fraud"]
    planned_smote_n = before["nonfraud"] + target_fraud if feasible else None
    return {
        "original": before,
        "smote_feasible": feasible,
        "smote_planned": {
            "n": planned_smote_n,
            "fraud": target_fraud if feasible else None,
            "nonfraud": before["nonfraud"],
            "synthetic_added": target_fraud - before["fraud"] if feasible else None,
            "note": "Hitungan rencana, bukan hasil eksekusi SMOTE; jumlah fraud asli tidak bertambah.",
        },
        "steps_per_epoch_by_batch_size": {
            str(batch): {
                "pure": math.ceil(before["n"] / batch),
                "smote": math.ceil(planned_smote_n / batch) if feasible else None,
            } for batch in BATCH_SIZES
        },
    }


def summarize_loss_history(path: Path, max_epochs: int, patience: int | None = None) -> dict:
    """Baca histori loss yang eksplisit diizinkan, tanpa menganggap cap sebagai konvergensi."""
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            return {"status": "empty", "epochs_observed": 0}
        epochs = np.array([int(row["epoch"]) for row in rows])
        losses = np.array([float(row["loss"]) for row in rows])
        if not np.array_equal(epochs, np.arange(1, len(rows) + 1)) or not np.isfinite(losses).all():
            raise ValueError("Urutan epoch tidak valid atau loss non-finite.")
        validation = None
        if "val_loss" in rows[0] and rows[0]["val_loss"] != "":
            validation = np.array([float(row["val_loss"]) for row in rows])
            if not np.isfinite(validation).all():
                raise ValueError("Validation loss non-finite.")
        cap_reached = len(rows) >= max_epochs
        summary = {"status": "observed", "epochs_observed": len(rows), "configured_max_epochs": max_epochs,
                   "epoch_cap_reached": cap_reached, "loss_first": float(losses[0]),
                   "loss_last": float(losses[-1]), "has_validation_loss": validation is not None}
        if validation is None:
            summary["status"] = "training_loss_only"
            return summary
        best_epoch = int(np.argmin(validation)) + 1
        stalled_epochs = len(rows) - best_epoch
        improving_at_end = len(rows) > 1 and best_epoch == len(rows) and validation[-1] < validation[-2]
        summary.update({"validation_loss_first": float(validation[0]), "validation_loss_last": float(validation[-1]),
                        "validation_loss_best": float(validation[best_epoch - 1]), "best_epoch": best_epoch,
                        "epochs_since_best": stalled_epochs, "improving_at_cap": bool(cap_reached and improving_at_end),
                        "patience": patience, "convergence_certified": False})
        if cap_reached:
            summary["status"] = "cap_reached_still_improving" if improving_at_end else "epoch_cap_reached"
        elif patience is not None and stalled_epochs >= patience:
            summary["status"] = "consistent_with_early_stopping"
        else:
            summary["status"] = "stopped_before_cap_reason_unknown"
        return summary
    except (ValueError, KeyError, OSError) as exc:
        return {"status": "invalid_history", "error": str(exc)}


def _validation_predictions(path: Path, data: dict, threshold: float) -> dict:
    """Accept only the complete ordered 341-400 validation population."""
    if not path.exists():
        return {"status": "unavailable", "reason": "Probabilitas validasi tuning belum disimpan; loss saja tidak membuktikan prediction collapse."}
    expected = np.flatnonzero((data["target_step"] >= 341) & (data["target_step"] <= 400))
    try:
        with np.load(path, allow_pickle=False) as saved:
            required = {"partition", "train_end", "val_start", "val_end", "target_step", "sequence_index", "y_true", "probability"}
            if not required.issubset(saved.files):
                raise ValueError("Metadata batas validasi tidak lengkap.")
            if (saved["partition"].item() != "internal_validation" or int(saved["train_end"].item()) != 340
                    or int(saved["val_start"].item()) != 341 or int(saved["val_end"].item()) != 400):
                raise ValueError("Artefak bukan validasi internal tuning 341-400.")
            indices, steps = saved["sequence_index"], saved["target_step"]
            if not np.array_equal(indices, expected) or not np.array_equal(steps, data["target_step"][expected]):
                raise ValueError("Indeks/batas waktu tidak cocok dengan populasi validasi internal.")
            y, probabilities = saved["y_true"], saved["probability"]
            if not np.array_equal(y, data["y"][expected]):
                raise ValueError("Label tidak cocok dengan validasi internal.")
            if (probabilities.shape != (len(expected),) or not len(expected)
                    or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))):
                raise ValueError("Probabilitas validasi tidak valid.")
        positive = int(np.count_nonzero(probabilities >= threshold))
        constant = bool(float(np.ptp(probabilities)) <= 1e-8)
        return {
            "status": "observed", "partition": "internal_validation", "target_step_range": [341, 400],
            "threshold": threshold, "n": len(probabilities), "fraud_labels": int(np.count_nonzero(y == 1)),
            "predicted_fraud": positive, "predicted_nonfraud": len(probabilities) - positive,
            "all_predictions_same_class": positive in (0, len(probabilities)),
            "no_predicted_fraud_at_threshold": positive == 0,
            "constant_probabilities": constant,
            "probability_min": float(probabilities.min()), "probability_max": float(probabilities.max()),
            "probability_std": float(probabilities.std()),
            "interpretation": "Prediksi satu kelas pada threshold tetap berbeda dari probabilitas konstan. Diagnosis ini hanya memakai validasi internal, bukan bukti kualitas hasil penelitian.",
        }
    except (ValueError, KeyError, OSError) as exc:
        return {"status": "rejected", "reason": str(exc)}


def _inspect_training_artifacts(directory: Path | None, data: dict, config: dict) -> dict:
    if directory is None or not directory.exists():
        return {"status": "unavailable", "loss_histories": [], "tuning_validation": {}}
    histories, validations = [], {}
    for condition in ("pure", "smote"):
        tuning = directory / "tuning" / condition
        params_path = tuning / "best_params.json"
        best_trial = patience = None
        if params_path.exists():
            try:
                params = json.loads(params_path.read_text(encoding="utf-8"))
                best_trial = int(params["trial_number"])
                patience = int(params["parameters"]["patience"])
            except (ValueError, KeyError, TypeError, OSError):
                pass
        for path in sorted(tuning.glob("trial_*_history.csv")):
            if not path.stem.removeprefix("trial_").removesuffix("_history").isdigit():
                continue
            trial = int(path.stem.removeprefix("trial_").removesuffix("_history"))
            summary = summarize_loss_history(path, int(config["max_epochs"]), patience if trial == best_trial else None)
            histories.append({"condition": condition, "stage": "optuna", "trial": trial,
                              "file": path.relative_to(directory).as_posix(), **summary})
        # Explicit allowlist: no test metrics, prediction folder, or final_holdout path is read.
        for fold in ("fold1", "fold2", "fold3"):
            for name in ("epoch_selection_history.csv", "refit_history.csv"):
                path = directory / "models" / condition / fold / name
                if path.exists():
                    summary = summarize_loss_history(path, int(config["max_epochs"]), patience)
                    histories.append({"condition": condition, "stage": fold,
                                      "file": path.relative_to(directory).as_posix(), **summary})
        validations[condition] = _validation_predictions(tuning / "best_validation_predictions.npz", data, float(config["threshold"]))
    return {"status": "inspected_training_artifacts", "loss_histories": histories, "tuning_validation": validations,
            "excluded": ["metrics.csv", "predictions/*", "models/*/final_holdout/*", "pooled_summary*", "bootstrap_uncertainty*"]}


def summarize_training(data_dir: Path, config: dict, output_dir: Path | None = None) -> dict:
    """Report exact training exposure and budgets; optional output writes one JSON.

    ``output_dir`` is the run directory containing whitelisted training artifacts.
    If omitted, ``config['output_dir']`` may be inspected read-only. No report is
    written without an explicit output_dir. This function never trains a model.
    """
    config = {"profile": "full", "seed": 42, "max_epochs": 30, "n_trials": 20,
              "smote_k": 5, "smote_ratio": 0.5, "threshold": 0.5, "smoke_max_train": 12000, **config}
    if config["profile"] not in ("full", "smoke"):
        raise ValueError("profile harus full atau smoke.")
    data = load_dataset(Path(data_dir))
    partitions, selection_errors = {}, []
    stages = (("optuna", 340, None, 341, 400),) + tuple(
        (name, end, end - 60, end - 59, end) for name, end in TRAINING_PARTITIONS)
    for name, end, inner_end, val_start, val_end in stages:
        try:
            selected, record = _select_train(data, end, config, inner_end=inner_end)
        except ValueError as exc:
            selection_errors.append({"partition": name, "error": str(exc)})
            continue
        early = selected if inner_end is None else selected[data["target_step"][selected] <= inner_end]
        validation = np.flatnonzero((data["target_step"] >= val_start) & (data["target_step"] <= val_end))
        source = record["source"]
        chosen = record["selected"]
        partitions[name] = {
            "train_end": end, "epoch_selection_train_end": inner_end or end,
            "validation_start": val_start, "validation_end": val_end,
            "source_training": source, "selected_training": _training_counts(data["y"][selected], config),
            "epoch_selection_training": _training_counts(data["y"][early], config),
            "internal_validation": _counts(data["y"][validation]),
            "all_original_training_labels_retained": chosen["n"] == source["n"],
            "fraud_retained_fraction": chosen["fraud"] / source["fraud"] if source["fraud"] else None,
            "selection_method": record["method"], "selection_attempts": record["attempts"],
        }
    artifact_dir = Path(output_dir) if output_dir is not None else (Path(config["output_dir"]) if config.get("output_dir") else None)
    budget = assess_training_budget(config)
    observations = []
    if config["profile"] == "smoke" and partitions:
        fraud_counts = [p["selected_training"]["original"]["fraud"] for p in partitions.values()]
        observations.append({"code": "limited_original_fraud_exposure",
                             "message": f"Subset latih mengandung {min(fraud_counts)}-{max(fraud_counts)} fraud asli per partisi; sampel sintetis tidak menambah kasus fraud asli yang diamati.",
                             "action": "Jalankan profil full tanpa subsampling untuk melatih seluruh label yang memenuhi syarat sequence."})
    if budget["research_budget_warning"]:
        observations.append({"code": "short_training_budget", "message": f"Pelatihan dibatasi {config['max_epochs']} epoch; hasil belum membuktikan model telah konvergen.",
                             "action": "Periksa histori validasi internal dengan anggaran full; jangan memilih perubahan menggunakan hasil holdout."})
    summary = {
        "scope": "training_and_internal_validation_only", "profile": config["profile"],
        "source_sha256": data.get("metadata", {}).get("source_sha256"),
        "max_label_step_read": 640,
        "quality_claim": "Tidak ada sertifikasi performa; profil smoke hanya uji teknis.",
        "budget": budget, "partitions": partitions, "selection_errors": selection_errors,
        "observations": observations, "artifacts": _inspect_training_artifacts(artifact_dir, data, config),
    }
    if output_dir is not None:
        write_json(Path(output_dir) / "training_diagnostics.json", summary)
    return summary
