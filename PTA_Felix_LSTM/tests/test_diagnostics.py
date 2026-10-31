"""Diagnostics must use only train and internal validation, with no TF/model work."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fraud_lstm.diagnostics import assess_training_budget, summarize_loss_history, summarize_training
from fraud_lstm.training import _select_train


def prepared_data():
    steps = np.repeat([100, 300, 340, 341, 400, 420, 421, 480, 500, 501, 560, 580, 581, 640, 641, 743], 20)
    labels = np.tile(np.array([1, 1] + [0] * 18, dtype="int8"), 16)
    return {"target_step": steps, "y": labels, "metadata": {"source_sha256": "test-source"}}


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fraud_diagnostics_")
        self.root = Path(self.temp.name)
        self.data = prepared_data()
        self.full = {"profile": "full", "seed": 42, "smote_ratio": 0.5, "smote_k": 5,
                     "max_epochs": 30, "n_trials": 20, "threshold": 0.5}

    def tearDown(self):
        self.temp.cleanup()

    def summary(self, config=None, output_dir=None):
        with patch("fraud_lstm.diagnostics.load_dataset", return_value=self.data):
            return summarize_training(self.root / "unused-data", config or self.full, output_dir)

    def save_validation(self, probabilities, **overrides):
        directory = self.root / "tuning" / "pure"
        directory.mkdir(parents=True, exist_ok=True)
        indices = np.flatnonzero((self.data["target_step"] >= 341) & (self.data["target_step"] <= 400))
        contents = {"partition": "internal_validation", "train_end": 340, "val_start": 341, "val_end": 400,
                    "sequence_index": indices, "target_step": self.data["target_step"][indices],
                    "y_true": self.data["y"][indices], "probability": probabilities}
        np.savez(directory / "best_validation_predictions.npz", **{**contents, **overrides})

    def test_exact_full_training_counts_and_steps_per_epoch(self):
        report = self.summary()
        final = report["partitions"]["final_train"]
        self.assertEqual(final["source_training"]["n"], 280)
        self.assertEqual(final["selected_training"]["original"]["fraud"], 28)
        self.assertTrue(final["all_original_training_labels_retained"])
        self.assertEqual(final["selected_training"]["smote_planned"]["fraud"], 126)
        self.assertEqual(final["selected_training"]["smote_planned"]["n"], 378)
        self.assertEqual(final["selected_training"]["steps_per_epoch_by_batch_size"]["128"], {"pure": 3, "smote": 3})
        self.assertEqual(report["partitions"]["optuna"]["selected_training"]["original"]["fraud"], 6)
        self.assertEqual(report["partitions"]["optuna"]["internal_validation"]["n"], 40)
        self.assertEqual(report["partitions"]["fold1"]["epoch_selection_training"]["original"]["n"], 60)
        self.assertEqual(report["selection_errors"], [])

    def test_smoke_counts_match_actual_training_selector(self):
        config = {**self.full, "profile": "smoke", "smoke_max_train": 100, "max_epochs": 2, "n_trials": 2}
        selected, record = _select_train(self.data, 640, config, inner_end=580)
        report = self.summary(config)
        final = report["partitions"]["final_train"]
        self.assertEqual(final["selected_training"]["original"], record["selected"])
        self.assertEqual(final["selection_attempts"], record["attempts"])
        self.assertEqual(final["selected_training"]["original"]["n"], len(selected))
        self.assertFalse(report["budget"]["smoke_quality_claims_allowed"])
        self.assertFalse(report["budget"]["performance_certified"])
        self.assertIn("limited_original_fraud_exposure", [item["code"] for item in report["observations"]])

    def test_epoch_budget_guard_warns_at_or_below_maximum_patience(self):
        for cap in (1, 2, 10):
            with self.subTest(cap=cap):
                assessment = assess_training_budget({**self.full, "max_epochs": cap})
                self.assertTrue(assessment["research_budget_warning"])
                self.assertFalse(assessment["epoch_cap_exceeds_max_patience"])
                self.assertIn("epoch_cap_not_above_max_patience", [item["code"] for item in assessment["warnings"]])
        assessment = assess_training_budget(self.full)
        self.assertFalse(assessment["research_budget_warning"])
        self.assertTrue(assessment["epoch_cap_exceeds_max_patience"])
        self.assertFalse(assessment["performance_certified"])

    def test_writes_json_only_when_output_directory_explicit(self):
        with patch("fraud_lstm.diagnostics.write_json") as writer:
            self.summary({**self.full, "output_dir": str(self.root)})
            writer.assert_not_called()
        report = self.summary(output_dir=self.root)
        saved = json.loads((self.root / "training_diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, report)

    def test_holdout_labels_and_prediction_files_never_used(self):
        # Changing every holdout label must not change a training diagnostic.
        baseline = self.summary(output_dir=self.root)
        self.data["y"][self.data["target_step"] >= 641] = 99
        forbidden = [self.root / "metrics.csv", self.root / "predictions" / "pure_final_holdout.npz",
                     self.root / "models" / "pure" / "final_holdout" / "epoch_selection_history.csv"]
        for path in forbidden:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("THIS MUST NEVER BE PARSED", encoding="utf-8")
        self.assertEqual(self.summary(output_dir=self.root), baseline)

    def test_threshold_single_class_is_not_automatically_constant_scores(self):
        self.save_validation(np.linspace(0.01, 0.4, 40))
        diagnosis = self.summary(output_dir=self.root)["artifacts"]["tuning_validation"]["pure"]
        self.assertEqual(diagnosis["status"], "observed")
        self.assertEqual(diagnosis["predicted_fraud"], 0)
        self.assertTrue(diagnosis["all_predictions_same_class"])
        self.assertTrue(diagnosis["no_predicted_fraud_at_threshold"])
        self.assertFalse(diagnosis["constant_probabilities"])

    def test_constant_scores_and_threshold_equality(self):
        self.save_validation(np.full(40, 0.5))
        diagnosis = self.summary(output_dir=self.root)["artifacts"]["tuning_validation"]["pure"]
        self.assertEqual(diagnosis["predicted_fraud"], 40)
        self.assertTrue(diagnosis["constant_probabilities"])

    def test_rejects_non_validation_predictions_or_misaligned_labels(self):
        cases = [{"partition": "final_holdout"}, {"val_end": 743},
                 {"target_step": np.full(40, 641)}, {"sequence_index": np.arange(40)},
                 {"y_true": np.zeros(40)}, {"probability": np.full(40, np.nan)}]
        for invalid in cases:
            with self.subTest(invalid=list(invalid)):
                self.save_validation(np.full(40, 0.1), **invalid)
                diagnosis = self.summary(output_dir=self.root)["artifacts"]["tuning_validation"]["pure"]
                self.assertEqual(diagnosis["status"], "rejected")

    def test_missing_validation_probabilities_cannot_claim_collapse(self):
        diagnosis = self.summary(output_dir=self.root)["artifacts"]["tuning_validation"]["pure"]
        self.assertEqual(diagnosis["status"], "unavailable")
        self.assertNotIn("all_predictions_same_class", diagnosis)

    def test_loss_cap_and_early_stop_are_distinguished(self):
        history = self.root / "history.csv"
        history.write_text("epoch,loss,val_loss\n1,0.6,0.5\n2,0.3,0.2\n", encoding="utf-8")
        summary = summarize_loss_history(history, 2, patience=3)
        self.assertEqual(summary["status"], "cap_reached_still_improving")
        self.assertTrue(summary["improving_at_cap"])
        self.assertFalse(summary["convergence_certified"])
        history.write_text("epoch,loss,val_loss\n1,0.6,0.5\n2,0.3,0.6\n3,0.2,0.7\n4,0.1,0.8\n", encoding="utf-8")
        summary = summarize_loss_history(history, 30, patience=3)
        self.assertEqual(summary["status"], "consistent_with_early_stopping")
        history.write_text("epoch,loss,val_loss\n1,nan,0.5\n", encoding="utf-8")
        self.assertEqual(summarize_loss_history(history, 30)["status"], "invalid_history")

    def test_insufficient_training_classes_are_reported_without_sampling_validation(self):
        self.data["y"][self.data["target_step"] <= 340] = 0
        report = self.summary()
        self.assertNotIn("optuna", report["partitions"])
        self.assertIn("optuna", [error["partition"] for error in report["selection_errors"]])


if __name__ == "__main__":
    unittest.main()
