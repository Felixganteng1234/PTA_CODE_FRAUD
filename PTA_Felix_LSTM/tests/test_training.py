"""Verifikasi batas waktu, scaler dan SMOTE tanpa mengimpor TensorFlow."""
import unittest

import numpy as np

from fraud_lstm.training import _resample, _scaled, _select_train


class TrainingPreprocessingTests(unittest.TestCase):
    def test_scaler_uses_training_only_and_keeps_one_hot_unchanged(self):
        X = np.zeros((4, 3, 10), dtype="float32")
        X[0, :, :5] = 0
        X[1, :, :5] = 2
        X[2, :, :5] = 1000
        X[3, :, :5] = 2000
        X[:, :, 7] = 1
        original = X.copy()
        data = {"X": X, "y": np.array([0, 1, 0, 1], dtype="int8")}
        train, y_train, validation, y_val, scaler = _scaled(data, np.array([0, 1]), np.array([2, 3]))
        np.testing.assert_array_equal(scaler.mean_, np.ones(5))
        np.testing.assert_array_equal(scaler.scale_, np.ones(5))
        self.assertEqual(int(scaler.n_samples_seen_), 6)
        np.testing.assert_allclose(train[0, :, :5], -1)
        np.testing.assert_allclose(train[1, :, :5], 1)
        np.testing.assert_allclose(validation[0, :, :5], 999)
        np.testing.assert_allclose(validation[1, :, :5], 1999)
        np.testing.assert_array_equal(train[:, :, 5:], original[:2, :, 5:])
        np.testing.assert_array_equal(validation[:, :, 5:], original[2:, :, 5:])
        np.testing.assert_array_equal(y_train, [0, 1])
        np.testing.assert_array_equal(y_val, [0, 1])
        np.testing.assert_array_equal(X, original)

    def test_pure_training_does_not_resample(self):
        X = np.arange(300, dtype="float32").reshape(10, 3, 10)
        y = np.array([0] * 9 + [1], dtype="int8")
        returned, labels, counts = _resample(X, y, "pure", {"seed": 42, "smote_ratio": 0.5, "smote_k": 5})
        np.testing.assert_array_equal(returned, X)
        np.testing.assert_array_equal(labels, y)
        self.assertEqual(counts["synthetic_added"], 0)
        self.assertEqual(counts["before"], counts["after"])

    def test_smote_preserves_original_sequences_and_has_exact_ratio(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(30, 3, 10)).astype("float32")
        y = np.array([0] * 24 + [1] * 6, dtype="int8")
        original = X.copy()
        config = {"seed": 42, "smote_ratio": 0.5, "smote_k": 5}
        sampled, labels, counts = _resample(X, y, "smote", config)
        repeated, repeated_y, _ = _resample(X, y, "smote", config)
        self.assertEqual(sampled.shape, (36, 3, 10))
        self.assertEqual(int(labels.sum()), 12)
        self.assertEqual(counts["synthetic_added"], 6)
        self.assertEqual(sampled.dtype, np.float32)
        np.testing.assert_array_equal(sampled[:30], original)
        np.testing.assert_array_equal(labels[:30], y)
        np.testing.assert_array_equal(sampled, repeated)
        np.testing.assert_array_equal(labels, repeated_y)
        np.testing.assert_array_equal(X, original)

    def test_smote_rejects_too_few_minority_examples(self):
        X = np.zeros((30, 3, 10), dtype="float32")
        y = np.array([0] * 25 + [1] * 5, dtype="int8")
        with self.assertRaisesRegex(ValueError, "requires >= 6 fraud"):
            _resample(X, y, "smote", {"seed": 42, "smote_ratio": 0.5, "smote_k": 5})

    def test_full_selection_keeps_every_sequence_until_inclusive_boundary(self):
        steps = np.array([1, 200, 300, 340, 340, 340, 340, 340, 340, 341, 400, 401, 743])
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype="int8")
        data = {"target_step": steps, "y": labels}
        selected, record = _select_train(data, 400, {"profile": "full", "seed": 42, "smote_k": 5}, inner_end=340)
        np.testing.assert_array_equal(selected, np.arange(11))
        self.assertEqual(record["method"], "all_sequences")
        self.assertEqual(record["attempts"][0]["early_training_counts"]["fraud"], 6)

    def test_smoke_selection_reproducible_and_excludes_future(self):
        steps = np.r_[np.repeat(300, 200), np.repeat(400, 50), np.repeat(641, 50)]
        labels = np.tile(np.array([0, 1], dtype="int8"), 150)
        data = {"target_step": steps, "y": labels}
        config = {"profile": "smoke", "seed": 42, "smote_k": 5, "smoke_max_train": 50}
        selected, record = _select_train(data, 400, config, inner_end=340)
        repeated, _ = _select_train(data, 400, config, inner_end=340)
        np.testing.assert_array_equal(selected, repeated)
        self.assertEqual(len(selected), 50)
        self.assertEqual(len(set(selected)), 50)
        self.assertTrue(np.all(steps[selected] <= 400))
        self.assertEqual(record["source"]["n"], 250)
        self.assertEqual(record["method"], "uniform_without_replacement")

    def test_insufficient_early_fraud_is_not_repaired_using_validation(self):
        # Abundant fraud after the internal cutoff cannot satisfy SMOTE's training requirement.
        data = {"target_step": np.r_[np.repeat(300, 100), np.repeat(400, 100)],
                "y": np.r_[np.zeros(100, dtype="int8"), np.ones(100, dtype="int8")]}
        config = {"profile": "smoke", "seed": 42, "smote_k": 5, "smoke_max_train": 20}
        with self.assertRaisesRegex(ValueError, "Insufficient classes in training period <= 340"):
            _select_train(data, 400, config, inner_end=340)


if __name__ == "__main__":
    unittest.main()
