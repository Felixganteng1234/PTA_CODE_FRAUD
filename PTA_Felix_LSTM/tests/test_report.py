"""Laporan harus mengungkap profil smoke dan menolak artefak yang tidak lengkap."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from fraud_lstm.metrics import evaluate_predictions, paired_bootstrap
from fraud_lstm.report import build_report


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data, self.output = self.root / "data", self.root / "run"
        self.data.mkdir()
        self.output.mkdir()
        self.metadata = {
            "status": "complete", "sequences": 120, "sequence_fraud": 12,
            "raw_transactions": 400, "raw_fraud": 40, "customer_transactions": 300,
            "merchant_transactions": 100, "merchant_fraud": 0, "customer_accounts": 70,
            "eligible_accounts": 30, "fraud_coverage": 0.3, "sequence_length": 3,
            "sort_keys": ["nameDest", "step", "original_row_id"], "label_rule": "isFraud(T3)",
            "min_target_step": 1, "max_target_step": 743,
            "features": ["amount", "newbalanceOrig", "newbalanceDest"],
            "source_file": "PS.csv", "source_sha256": "example_checksum",
        }
        write_json(self.data / "metadata.json", self.metadata)
        self.config = {"profile": "smoke", "threshold": 0.5, "cpu_threads": 2,
                       "smote_ratio": 0.5, "smote_k": 5, "n_trials": 2, "max_epochs": 2,
                       "bootstrap_iterations": 10, "note": "<script>alert(1)</script>"}
        self.manifest = {
            "status": "completed", "config": self.config, "dataset_counts": {"n": 120, "fraud": 12},
            "completed_models": [f"{condition}/{fold}" for condition in ("pure", "smote")
                                 for fold in ("fold1", "fold2", "fold3", "final_holdout")],
            "elapsed_seconds": 34.56, "runtime": {"devices": ["/physical_device:CPU:0"],
                                                  "platform": "test", "packages": {"numpy": "test"}},
        }
        write_json(self.output / "run_manifest.json", self.manifest)
        self.results = []
        for condition in ("pure", "smote"):
            for fold, end, start, stop in (("fold1", 400, 401, 480), ("fold2", 480, 481, 560),
                                          ("fold3", 560, 561, 640), ("final_holdout", 640, 641, 743)):
                values = evaluate_predictions([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.2])
                values.update(condition=condition, fold=fold, profile="smoke", train_end=end,
                              test_start=start, test_end=stop, n_train_original=10,
                              n_train_after_smote=10 if condition == "pure" else 15,
                              n_test=4, fraud_test=2, selected_epoch=2)
                self.results.append(values)
        write_csv(self.output / "metrics.csv", self.results)
        pooled = []
        for condition in ("pure", "smote"):
            values = evaluate_predictions([0, 0, 1, 1] * 3, [0.1, 0.7, 0.8, 0.2] * 3)
            values.update(condition=condition, profile="smoke", pr_auc_fold_mean=values["average_precision"],
                          pr_auc_fold_std_sample=0.0)
            pooled.append(values)
        write_json(self.output / "pooled_summary.json", pooled)
        write_csv(self.output / "pooled_summary.csv", pooled)
        self.bootstrap = paired_bootstrap([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.2],
                                          [0.1, 0.7, 0.8, 0.2], n_bootstrap=10)
        write_json(self.output / "bootstrap_uncertainty.json", self.bootstrap)
        self.bootstrap_rows = []
        for condition in ("pure", "smote", "delta_smote_minus_pure"):
            for metric, values in self.bootstrap[condition].items():
                self.bootstrap_rows.append({"condition": condition, "metric": metric, "profile": "smoke", **values})
        write_csv(self.output / "bootstrap_uncertainty.csv", self.bootstrap_rows)
        write_json(self.output / "selection_counts.json", [
            {"stage": "fold1", "train_end": 400, "source": {"n": 100},
             "selected": {"n": 10, "fraud": 2}, "method": "uniform_without_replacement"}
        ])
        (self.output / "plots").mkdir()
        (self.output / "plots" / "test.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        write_json(self.output / "figures.json", {"precision_recall": str(self.output / "plots" / "test.png")})

    def tearDown(self):
        self.temporary.cleanup()

    def test_smoke_report_discloses_scope_and_uses_recorded_values(self):
        path = build_report(self.data, self.output)
        self.assertEqual(path, (self.output / "laporan.html").resolve())
        text = path.read_text(encoding="utf-8")
        markdown = (self.output / "LAPORAN_HASIL.md").read_text(encoding="utf-8")
        for body in (text, markdown):
            self.assertIn("UJI TEKNIS - BUKAN HASIL FINAL PENELITIAN", body)
            self.assertIn("34.560 detik", body)
            self.assertIn("30.00%", body)
            self.assertIn("post-transaction", body)
            self.assertIn("0.833333", body)
            self.assertIn("Profil full belum dijalankan", body)
            self.assertIn("bootstrap_uncertainty.csv", body)
        self.assertIn('src="plots/test.png"', text)
        self.assertNotIn("<script>", text)
        self.assertIn("&lt;script&gt;", text)

    def test_refuses_incomplete_status_without_writing_report(self):
        self.manifest["status"] = "running"
        write_json(self.output / "run_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "completed"):
            build_report(self.data, self.output)
        self.assertFalse((self.output / "laporan.html").exists())

    def test_single_trial_is_disclosed_as_fixed_configuration_not_tpe_search(self):
        self.config["n_trials"] = 1
        write_json(self.output / "run_manifest.json", self.manifest)
        path = build_report(self.data, self.output)
        text = path.read_text(encoding="utf-8")
        self.assertIn("satu konfigurasi hyperparameter yang ditetapkan sebelum evaluasi", text)
        self.assertIn("bukan pencarian atau optimasi TPE", text)
        self.assertIn("digunakan untuk mencatat performa validasi konfigurasi tunggal", text)

    def test_refuses_missing_model_metric(self):
        write_csv(self.output / "metrics.csv", self.results[:-1])
        with self.assertRaisesRegex(ValueError, "8 kombinasi"):
            build_report(self.data, self.output)

    def test_refuses_conflicting_bootstrap_csv_json(self):
        self.bootstrap_rows[0]["estimate"] = 0.123456
        write_csv(self.output / "bootstrap_uncertainty.csv", self.bootstrap_rows)
        with self.assertRaisesRegex(ValueError, "tidak konsisten"):
            build_report(self.data, self.output)

    def test_refuses_conflicting_dataset(self):
        self.metadata["sequence_fraud"] = 99
        write_json(self.data / "metadata.json", self.metadata)
        with self.assertRaisesRegex(ValueError, "Jumlah fraud"):
            build_report(self.data, self.output)


if __name__ == "__main__":
    unittest.main()
