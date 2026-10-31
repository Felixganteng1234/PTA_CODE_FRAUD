"""CLI dan pembatas metode penelitian; semua training dimock tanpa TensorFlow."""
import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fraud_lstm import __main__ as cli


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        self.path = self.root / "configs" / "full.json"
        self.config = {"profile": "full", "seed": 42, "n_trials": 20,
                       "max_epochs": 30, "bootstrap_iterations": 1000, "cpu_threads": 4,
                       "smote_ratio": 0.5, "smote_k": 5, "threshold": 0.5,
                       "dataset": "data/raw/PS.csv", "data_dir": "data/processed",
                       "output_dir": "results/full"}

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, **changes):
        value = {**self.config, **changes}
        self.path.write_text(json.dumps(value), encoding="utf-8")
        return value

    def test_valid_config_resolves_project_root_from_config_folder(self):
        expected = self.save()
        actual, root = cli.read_config(self.path)
        self.assertEqual(actual, expected)
        self.assertEqual(root, self.root.resolve())

    def test_resource_counts_require_positive_integers(self):
        for field in ("n_trials", "max_epochs", "bootstrap_iterations", "cpu_threads"):
            for value in (0, -1, True, 2.5, "20", None):
                with self.subTest(field=field, value=value):
                    self.save(**{field: value})
                    with self.assertRaisesRegex(ValueError, field):
                        cli.read_config(self.path)

    def test_full_rejects_smoke_epoch_budget(self):
        self.save(max_epochs=3)
        with self.assertRaisesRegex(ValueError, "max_epochs > 3"):
            cli.read_config(self.path)
        self.save(profile="smoke", max_epochs=1)
        self.assertEqual(cli.read_config(self.path)[0]["max_epochs"], 1)

    def test_smote_and_primary_threshold_are_fixed(self):
        for changes in ({"smote_ratio": 1}, {"smote_k": 3}, {"threshold": 0.2},
                        {"threshold": 1}, {"threshold": float("nan")}):
            with self.subTest(changes=changes):
                self.save(**changes)
                with self.assertRaises(ValueError):
                    cli.read_config(self.path)

    def test_seed_and_steps_per_execution_types(self):
        for value in (True, 1.5, "42", None):
            self.save(seed=value)
            with self.assertRaisesRegex(ValueError, "seed"):
                cli.read_config(self.path)
        for value in (0, -1, True, 2.5, "4"):
            self.save(steps_per_execution=value)
            with self.assertRaisesRegex(ValueError, "steps_per_execution"):
                cli.read_config(self.path)

    def test_unknown_profile_rejected(self):
        self.save(profile="fast")
        with self.assertRaisesRegex(ValueError, "profile"):
            cli.read_config(self.path)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        (self.data / "metadata.json").write_text(json.dumps({"source_sha256": "expected"}), encoding="utf-8")
        self.config = {"profile": "full", "data_dir": "data", "dataset": "PS.csv",
                       "output_dir": "results/full", "n_trials": 2, "max_epochs": 4}
        self.output = self.root / self.config["output_dir"]

    def tearDown(self):
        self.temporary.cleanup()

    def fake_modules(self, status="tuning_incomplete"):
        training, report = types.ModuleType("fraud_lstm.training"), types.ModuleType("fraud_lstm.report")
        def fake_run(data, output, config):
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_manifest.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        training.run_experiment = Mock(side_effect=fake_run)
        report.build_report = Mock(return_value=self.output / "laporan.html")
        return training, report

    def test_status_uses_full_config_by_default_without_training(self):
        with patch.object(cli, "read_config", return_value=(dict(self.config), self.root)) as read:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["status"]), 0)
        read.assert_called_once_with(Path("configs/full.json"))

    def test_checksum_mismatch_blocks_training(self):
        training, report = self.fake_modules()
        with patch.object(cli, "read_config", return_value=(dict(self.config), self.root)), \
                patch.object(cli, "sha256_file", return_value="changed"), \
                patch.dict("sys.modules", {"fraud_lstm.training": training, "fraud_lstm.report": report}):
            with self.assertRaisesRegex(ValueError, "Checksum"):
                cli.main(["train"])
        training.run_experiment.assert_not_called()
        report.build_report.assert_not_called()

    def test_resume_and_stage_controls_forward_without_reporting_incomplete_run(self):
        training, report = self.fake_modules()
        with patch.object(cli, "read_config", return_value=(dict(self.config), self.root)), \
                patch.object(cli, "sha256_file", return_value="expected"), \
                patch.dict("sys.modules", {"fraud_lstm.training": training, "fraud_lstm.report": report}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["train", "--resume", "--stage", "tune", "--trial-budget", "1"]), 0)
        received = training.run_experiment.call_args.args[2]
        self.assertEqual(received["n_trials"], 2)
        self.assertEqual(received["trial_budget"], 1)
        self.assertEqual(received["run_stage"], "tune")
        self.assertIs(received["resume"], True)
        report.build_report.assert_not_called()

    def test_completed_training_builds_report(self):
        training, report = self.fake_modules("completed")
        with patch.object(cli, "read_config", return_value=(dict(self.config), self.root)), \
                patch.object(cli, "sha256_file", return_value="expected"), \
                patch.dict("sys.modules", {"fraud_lstm.training": training, "fraud_lstm.report": report}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["train", "--stage", "all"]), 0)
        report.build_report.assert_called_once_with(self.data, self.output)

    def test_trial_budget_zero_rejected_before_training(self):
        training, report = self.fake_modules()
        with patch.object(cli, "read_config", return_value=(dict(self.config), self.root)), \
                patch.dict("sys.modules", {"fraud_lstm.training": training, "fraud_lstm.report": report}):
            with self.assertRaisesRegex(ValueError, "trial-budget"):
                cli.main(["train", "--trial-budget", "0"])
        training.run_experiment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
