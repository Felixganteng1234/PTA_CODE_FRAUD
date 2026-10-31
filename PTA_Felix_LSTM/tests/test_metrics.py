"""Pengujian metrik, pooling, dan bootstrap berpasangan tanpa training model."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from fraud_lstm.metrics import (
    BOOTSTRAP_METRICS,
    create_plots,
    evaluate_predictions,
    paired_bootstrap,
    pooled_summary,
)


class PredictionMetricsTests(unittest.TestCase):
    def test_known_confusion_and_threshold_equality(self):
        result = evaluate_predictions([0, 0, 1, 1], [0.1, 0.8, 0.5, 0.2])
        self.assertEqual([result[key] for key in ("tn", "fp", "fn", "tp")], [1, 1, 1, 1])
        for metric in ("precision", "recall", "f1"):
            self.assertEqual(result[metric], 0.5)
        self.assertEqual(result["n_samples"], 4)
        self.assertEqual(result["n_fraud"], 2)
        self.assertAlmostEqual(result["average_precision"], 7 / 12)
        self.assertNotEqual(result["average_precision"], result["pr_auc_trapezoid"])
        json.dumps(result, allow_nan=False)

    def test_single_class_and_zero_division(self):
        result = evaluate_predictions([0, 0], [0.1, 0.2])
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["average_precision"])
        self.assertIsNone(result["pr_auc_trapezoid"])
        self.assertEqual(result["f1"], 0)
        positive = evaluate_predictions([1, 1], [0.1, 0.2])
        self.assertIsNone(positive["roc_auc"])
        self.assertEqual(positive["average_precision"], 1)
        self.assertEqual(positive["pr_auc_trapezoid"], 1)
        self.assertEqual(positive["precision"], 0)

    def test_rejects_invalid_inputs(self):
        invalid = [([], []), ([0], [0, 1]), ([0, 2], [0.1, 0.2]),
                   ([0, 1], [np.nan, 0.1]), ([0, 1], [-0.1, 1]),
                   ([[0], [1]], [0.1, 0.2]), ([0, np.inf], [0.1, 0.2])]
        for labels, probabilities in invalid:
            with self.subTest(labels=labels, probabilities=probabilities):
                with self.assertRaises(ValueError):
                    evaluate_predictions(labels, probabilities)
        for threshold in (float("nan"), -1, 2, "bad"):
            with self.assertRaises(ValueError):
                evaluate_predictions([0, 1], [0.1, 0.9], threshold)


class PooledMetricsTests(unittest.TestCase):
    def test_pooled_f1_is_not_mean_fold_f1(self):
        folds = {"small": ([0, 1], [0.1, 0.9]),
                 "large": ([0, 0, 0, 1, 1, 1], [0.9, 0.9, 0.9, 0.1, 0.1, 0.1])}
        summary = pooled_summary(folds, 0.5)
        expected = precision_recall_fscore_support(
            [0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 1, 0, 0, 0],
            average="binary", zero_division=0,
        )
        self.assertEqual(summary["pooled"]["f1"], expected[2])
        self.assertNotEqual(summary["pooled"]["f1"], 0.5)
        ap_values = [average_precision_score(y, p) for y, p in folds.values()]
        self.assertAlmostEqual(summary["average_precision_std"], np.std(ap_values, ddof=1))

    def test_undefined_fold_ap_does_not_corrupt_summary(self):
        result = pooled_summary({"negative": ([0, 0], [0.1, 0.2]), "mixed": ([0, 1], [0.1, 0.9])})
        self.assertEqual(result["n_defined_ap_folds"], 1)
        self.assertEqual(result["average_precision_mean"], 1)
        self.assertIsNone(result["average_precision_std"])
        with self.assertRaises(ValueError):
            pooled_summary({})


class BootstrapTests(unittest.TestCase):
    def test_paired_predictions_produce_exact_zero_delta(self):
        result = paired_bootstrap([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.2],
                                  [0.1, 0.7, 0.8, 0.2], n_bootstrap=40, seed=3)
        for metric in BOOTSTRAP_METRICS:
            delta = result["delta_smote_minus_pure"][metric]
            self.assertEqual(delta["estimate"], 0)
            self.assertEqual(delta["ci_lower"], 0)
            self.assertEqual(delta["ci_upper"], 0)
            self.assertEqual(delta["n_valid"] + delta["n_skipped"], 40)
        json.dumps(result, allow_nan=False)

    def test_reproducibility_and_delta_direction(self):
        args = ([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], [0.1, 0.2, 0.8, 0.9])
        result = paired_bootstrap(*args, n_bootstrap=25, seed=9)
        self.assertEqual(result, paired_bootstrap(*args, n_bootstrap=25, seed=9))
        self.assertEqual(result["delta_smote_minus_pure"]["f1"]["estimate"], 1)

    def test_cluster_bootstrap_keeps_whole_group(self):
        # Satu cluster berarti setiap replikasi persis seluruh data semula.
        result = paired_bootstrap([0, 1, 1], [0.1, 0.9, 0.2], [0.2, 0.8, 0.9],
                                  groups=["C1", "C1", "C1"], n_bootstrap=15)
        self.assertEqual(result["method"], "paired_cluster_percentile")
        self.assertEqual(result["n_groups"], 1)
        for condition in ("pure", "smote", "delta_smote_minus_pure"):
            for metric in BOOTSTRAP_METRICS:
                values = result[condition][metric]
                self.assertEqual(values["estimate"], values["ci_lower"])
                self.assertEqual(values["estimate"], values["ci_upper"])

    def test_no_fraud_bootstrap_reports_all_ap_skipped(self):
        result = paired_bootstrap([0, 0], [0.1, 0.2], [0.2, 0.3], n_bootstrap=10)
        for condition in ("pure", "smote", "delta_smote_minus_pure"):
            self.assertEqual(result[condition]["average_precision"]["n_skipped"], 10)
            self.assertIsNone(result[condition]["average_precision"]["ci_lower"])
            self.assertEqual(result[condition]["precision"]["n_valid"], 10)

    def test_invalid_bootstrap_arguments(self):
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                paired_bootstrap([0, 1], [0.1, 0.9], [0.1, 0.9], n_bootstrap=count)
        for groups in (["C1"], [["C1"], ["C2"]], [None, "C2"]):
            with self.assertRaises(ValueError):
                paired_bootstrap([0, 1], [0.1, 0.9], [0.1, 0.9], groups=groups)


class PlotTests(unittest.TestCase):
    def test_writes_png_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = create_plots(folder, {"Pure LSTM": ([0, 1, 1], [0.1, 0.8, 0.4])})
            self.assertEqual(len(paths), 2)
            for path in paths.values():
                self.assertEqual(Path(path).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
