"""Uji integritas kausal sequence dan validasi input; tidak membutuhkan CSV PaySim besar."""
from __future__ import annotations

import contextlib
import gc
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_lstm.data import FEATURES, FOLDS, audit_dataset, load_dataset, prepare_dataset


def transaction(account, step, amount, fraud=0, kind="TRANSFER"):
    return {
        "step": step, "type": kind, "amount": amount, "nameOrig": "Csource",
        "oldbalanceOrg": amount + 100, "newbalanceOrig": 100,
        "nameDest": account, "oldbalanceDest": 0, "newbalanceDest": amount,
        "isFraud": fraud, "isFlaggedFraud": 1,
    }


def example_frame():
    # Input rows deliberately do not follow either account or time order.
    return pd.DataFrame([
        transaction("Cbeta", 401, 10, 1),
        transaction("Calpha", 400, 20, 0),
        transaction("Calpha", 399, 30, 1),
        transaction("Mmerchant", 400, 999, 0, "PAYMENT"),
        transaction("Calpha", 400, 40, 0),
        transaction("Cbeta", 399, 50, 0),
        transaction("Cbeta", 400, 60, 0),
        transaction("Calpha", 401, 80, 1),
        transaction("Cshort", 399, 90, 0),
        transaction("Cshort", 400, 100, 0),
        transaction("Calpha", 640, 110, 0),
        transaction("Calpha", 641, 120, 1),
    ])


def close_mmaps(data):
    for value in data.values():
        if isinstance(value, np.memmap):
            value._mmap.close()
    gc.collect()


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fraud_data_test_")
        self.root = Path(self.temp.name)

    def tearDown(self):
        gc.collect()
        self.temp.cleanup()

    def prepare(self, frame=None, name="prepared", chunk_size=3):
        frame = example_frame() if frame is None else frame
        csv_path = self.root / f"{name}.csv"
        frame.to_csv(csv_path, index=False)
        output = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            metadata = prepare_dataset(csv_path, output, chunk_size=chunk_size)
        return output, metadata

    def test_inclusive_windows_ties_account_boundaries_and_label(self):
        output, meta = self.prepare()
        data = load_dataset(output)
        try:
            # Calpha has six rows -> four windows; Cbeta three -> one window.
            self.assertEqual(data["X"].shape, (5, 3, 10))
            np.testing.assert_array_equal(data["X"][:, :, 0], [
                [30, 20, 40], [20, 40, 80], [40, 80, 110], [80, 110, 120],
                [50, 60, 10],
            ])
            np.testing.assert_array_equal(data["target_row"], [4, 7, 10, 11, 0])
            np.testing.assert_array_equal(data["target_step"], [400, 401, 640, 641, 401])
            np.testing.assert_array_equal(data["y"], [0, 1, 0, 1, 1])
            self.assertTrue(np.all(data["account_id"][:4] == data["account_id"][0]))
            self.assertNotEqual(data["account_id"][0], data["account_id"][4])
            self.assertNotIn(999, data["X"][:, :, 0])
            # Labels, identity, step and flagged fraud are deliberately absent from inputs.
            self.assertEqual(meta["features"], FEATURES)
            self.assertEqual(meta["raw_transactions"], 12)
            self.assertEqual(meta["customer_transactions"], 11)
            self.assertEqual(meta["merchant_transactions"], 1)
            self.assertEqual(meta["raw_fraud"], 4)
            self.assertEqual(meta["sequence_fraud"], 3)
            self.assertEqual(meta["eligible_accounts"], 2)
            self.assertEqual(meta["fraud_coverage"], 0.75)
            np.testing.assert_array_equal(data["X"][:, :, 5:].sum(axis=2), np.ones((5, 3)))
        finally:
            close_mmaps(data)

    def test_cross_period_context_is_allowed_but_assignment_uses_target(self):
        output, _ = self.prepare()
        table = audit_dataset(output, self.root / "audit").set_index("fold")
        self.assertEqual(table.loc["fold1", "n_train"], 1)
        self.assertEqual(table.loc["fold1", "n_test"], 2)
        self.assertEqual(table.loc["fold1", "fraud_train"], 0)
        self.assertEqual(table.loc["fold1", "fraud_test"], 2)
        self.assertEqual(table.loc["final_holdout", "n_train"], 4)
        self.assertEqual(table.loc["final_holdout", "n_test"], 1)
        data = load_dataset(output)
        source = example_frame()
        try:
            for index, row_id in enumerate(data["target_row"]):
                row = source.iloc[int(row_id)]
                group = source.loc[source["nameDest"] == row["nameDest"]].copy()
                group["original_row"] = group.index
                group = group.sort_values(["step", "original_row"])
                position = list(group.index).index(int(row_id))
                expected = group.iloc[position - 2:position + 1]
                self.assertEqual(len(expected), 3)
                self.assertTrue((expected["step"] <= row["step"]).all())
                np.testing.assert_array_equal(data["X"][index, :, 0], expected["amount"])
            holdout = data["target_step"] >= 641
            np.testing.assert_array_equal(data["X"][holdout][0, :, 0], [80, 110, 120])
        finally:
            close_mmaps(data)

    def test_chunk_size_does_not_change_results(self):
        first, _ = self.prepare(name="first", chunk_size=1)
        second, _ = self.prepare(name="second", chunk_size=100)
        a, b = load_dataset(first), load_dataset(second)
        try:
            for key in ["X", "y", "target_step", "target_row", "account_id"]:
                np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(a["metadata"]["source_sha256"], b["metadata"]["source_sha256"])
        finally:
            close_mmaps(a)
            close_mmaps(b)

    def test_identical_rows_are_preserved_and_short_accounts_excluded(self):
        repeated = transaction("Crepeated", 1, 5)
        frame = pd.DataFrame([repeated.copy(), repeated.copy(), repeated.copy(),
                              transaction("Cshort", 1, 7), transaction("Cshort", 2, 8)])
        output, meta = self.prepare(frame)
        data = load_dataset(output)
        try:
            self.assertEqual(meta["sequences"], 1)
            np.testing.assert_array_equal(data["target_row"], [2])
            np.testing.assert_array_equal(data["X"][0, :, 0], [5, 5, 5])
        finally:
            close_mmaps(data)

    def test_malformed_schema_and_values_fail_before_casting(self):
        cases = [
            ("missing_column", lambda x: x.drop(columns="amount")),
            ("missing_amount", lambda x: x.assign(amount=np.nan)),
            ("infinite_amount", lambda x: x.assign(amount=np.inf)),
            ("float32_positive_overflow", lambda x: x.assign(amount=1e100)),
            ("float32_negative_overflow", lambda x: x.assign(oldbalanceOrg=-1e100)),
            ("unknown_type", lambda x: x.assign(type="UNKNOWN")),
            ("fractional_label", lambda x: x.assign(isFraud=0.5)),
            ("label_out_of_range", lambda x: x.assign(isFraud=2)),
            ("fractional_step", lambda x: x.assign(step=1.5)),
            ("step_zero", lambda x: x.assign(step=0)),
            ("step_after_holdout", lambda x: x.assign(step=744)),
            ("missing_dest", lambda x: x.assign(nameDest=None)),
            ("unknown_dest", lambda x: x.assign(nameDest="Xunknown")),
            ("merchant_fraud", lambda x: x.assign(nameDest="Mmerchant", isFraud=1)),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.prepare(mutate(example_frame()), name=name)
                self.assertFalse((self.root / name / "metadata.json").exists())

    def test_empty_csv_and_no_eligible_account_fail(self):
        for name, frame in [("empty", example_frame().iloc[:0]),
                            ("too_short", pd.DataFrame([transaction("Csingle", 1, 1)]))]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.prepare(frame, name=name)

    def test_existing_output_is_not_overwritten(self):
        output, _ = self.prepare()
        metadata_before = (output / "metadata.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_dataset(self.root / "prepared.csv", output)
        self.assertEqual((output / "metadata.json").read_bytes(), metadata_before)

    def test_loader_requires_complete_metadata_and_correct_shapes(self):
        output, meta = self.prepare()
        metadata_path = output / "metadata.json"
        for change in [{"status": "running"}, {"features": FEATURES[::-1]}, {"sequences": 99}]:
            with self.subTest(change=change):
                metadata_path.write_text(json.dumps({**meta, **change}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_dataset(output)
        metadata_path.write_text(json.dumps(meta), encoding="utf-8")
        np.save(output / "y.npy", np.array([1, 0], dtype="int8"), allow_pickle=False)
        with self.assertRaises(ValueError):
            load_dataset(output)

    def test_pickle_backed_array_is_rejected(self):
        output, _ = self.prepare()
        # Object arrays require pickle and must not be loaded by the dataset reader.
        np.save(output / "y.npy", np.array([{}] * 5, dtype=object), allow_pickle=True)
        with self.assertRaises(ValueError):
            load_dataset(output)

    def test_fold_endpoints_match_proposal(self):
        self.assertEqual(FOLDS, {"fold1": (400, 401, 480), "fold2": (480, 481, 560),
                                 "fold3": (560, 561, 640), "final_holdout": (640, 641, 743)})


if __name__ == "__main__":
    unittest.main()
