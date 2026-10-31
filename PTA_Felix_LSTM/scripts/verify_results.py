"""Verifikasi artefak tersimpan terhadap data asal dan model yang dimuat ulang."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraud_lstm.data import FOLDS, load_dataset, write_json
from fraud_lstm.metrics import evaluate_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/smoke")
    args = parser.parse_args()
    output = args.output
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert len(manifest["completed_models"]) == 8
    data = load_dataset(ROOT / manifest["config"]["data_dir"])
    threshold = manifest["config"]["threshold"]
    checks, paired = [], {}
    for condition in ["pure", "smote"]:
        for fold, (end, start, stop) in FOLDS.items():
            file = output / "predictions" / f"{condition}_{fold}.npz"
            with np.load(file, allow_pickle=False) as saved:
                idx = saved["sequence_index"]
                expected = np.flatnonzero((data["target_step"] >= start) & (data["target_step"] <= stop))
                np.testing.assert_array_equal(idx, expected)
                np.testing.assert_array_equal(saved["y_true"], data["y"][idx])
                np.testing.assert_array_equal(saved["target_row"], data["target_row"][idx])
                np.testing.assert_array_equal(saved["account_id"], data["account_id"][idx])
                computed = evaluate_predictions(saved["y_true"], saved["probability"], threshold)
                assert np.isfinite(saved["probability"]).all()
                paired[(condition, fold)] = saved["target_row"].copy()
                checks.append({"condition": condition, "fold": fold, "n_test": len(idx), "fraud_test": int(saved["y_true"].sum()), "prediction_alignment": True, "recomputed_metrics": computed})
            selected = np.load(output / "selection" / f"{fold}_indices.npy", allow_pickle=False)
            assert (data["target_step"][selected] <= end).all()
            assert np.intersect1d(selected, expected).size == 0
    for fold in FOLDS:
        np.testing.assert_array_equal(paired[("pure", fold)], paired[("smote", fold)])
    import tensorflow as tf
    reload_errors = {}
    for condition in ["pure", "smote"]:
        directory = output / "models" / condition / "final_holdout"
        model = tf.keras.models.load_model(directory / "model.keras")
        scaler = joblib.load(directory / "scaler.joblib")
        with np.load(output / "predictions" / f"{condition}_final_holdout.npz", allow_pickle=False) as saved:
            idx = saved["sequence_index"][:128]
            X = np.array(data["X"][idx], copy=True)
            X[:, :, :5] = scaler.transform(X[:, :, :5].reshape(-1, 5)).reshape(-1, 3, 5)
            restored = np.asarray(model(X, training=False)).ravel()
            np.testing.assert_allclose(restored, saved["probability"][:128], atol=2e-6, rtol=2e-5)
            reload_errors[condition] = float(np.max(np.abs(restored - saved["probability"][:128])))
        tf.keras.backend.clear_session()
    write_json(output / "verification.json", {"status": "passed", "prediction_files_checked": 8, "paired_targets_equal": True, "all_clean_tests_complete": True, "holdout_model_reload_max_abs_error": reload_errors, "checks": checks})
    print("PASS: 8 prediksi lengkap, target sama, partisi benar, 2 model holdout berhasil dimuat ulang.")


if __name__ == "__main__":
    main()
