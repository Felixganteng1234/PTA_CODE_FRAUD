"""Pembentukan sequence kausal, dapat dilacak, dan hemat memori dari CSV asli."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
FEATURES = NUMERIC + [f"type_{name}" for name in TYPES]
REQUIRED = ["step", "type", *NUMERIC, "nameDest", "isFraud"]
FOLDS = {
    "fold1": (400, 401, 480),
    "fold2": (480, 481, 560),
    "fold3": (560, 561, 640),
    "final_holdout": (640, 641, 743),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def validate_frame(frame: pd.DataFrame) -> None:
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak tersedia: {sorted(missing)}")
    if frame[REQUIRED].isna().any().any():
        raise ValueError("Data kosong ditemukan pada kolom wajib; koreksi sumber sebelum melanjutkan.")
    if not np.isfinite(frame[NUMERIC + ["step", "isFraud"]].to_numpy(dtype=float)).all():
        raise ValueError("Nilai numerik NaN/inf ditemukan.")
    if np.any(np.abs(frame[NUMERIC].to_numpy(dtype="float64")) > np.finfo(np.float32).max):
        raise ValueError("Nilai fitur numerik melampaui kapasitas float32.")
    if not frame["isFraud"].isin([0, 1]).all():
        raise ValueError("Label isFraud harus 0 atau 1.")
    if not frame["type"].isin(TYPES).all():
        raise ValueError("Jenis transaksi di luar kamus PaySim.")
    steps = frame["step"].to_numpy(dtype=float)
    if np.any((steps < 1) | (steps > 743) | (steps != np.floor(steps))):
        raise ValueError("step harus bilangan bulat 1-743 sesuai rancangan.")


def prepare_dataset(csv_path: Path, output_dir: Path, chunk_size: int = 250_000) -> dict:
    """Tidak membuang duplikat: baris identik tetap dapat menjadi transaksi berbeda."""
    csv_path, output_dir = Path(csv_path), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Folder data sudah berisi file: {output_dir}. Gunakan folder baru untuk membangun ulang.")
    output_dir.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(csv_path, nrows=0)
    missing = set(REQUIRED) - set(header.columns)
    if missing:
        raise ValueError(f"Kolom CSV tidak lengkap: {sorted(missing)}")
    print("Membaca seluruh CSV dan memvalidasi schema...", flush=True)
    chunks, total, fraud, merchants, merchant_fraud, other_dest = [], 0, 0, 0, 0, 0
    # Read numeric as float64 first so malformed fractional labels/steps cannot be silently truncated.
    dtypes = {name: "float64" for name in NUMERIC + ["step", "isFraud"]}
    dtypes.update({"nameDest": "string", "type": "string"})
    with pd.read_csv(csv_path, usecols=REQUIRED, dtype=dtypes, chunksize=chunk_size) as reader:
        for chunk in reader:
            validate_frame(chunk)
            chunk["row_id"] = np.arange(total, total + len(chunk), dtype=np.int64)
            total += len(chunk)
            fraud += int(chunk["isFraud"].sum())
            customer = chunk["nameDest"].str.startswith("C")
            merchant = chunk["nameDest"].str.startswith("M")
            merchants += int(merchant.sum())
            merchant_fraud += int(chunk.loc[merchant, "isFraud"].sum())
            other_dest += int((~customer & ~merchant).sum())
            part = chunk.loc[customer].copy()
            part[NUMERIC] = part[NUMERIC].astype("float32")
            part["step"] = part["step"].astype("int16")
            part["isFraud"] = part["isFraud"].astype("int8")
            part["type"] = part["type"].astype(pd.CategoricalDtype(TYPES))
            chunks.append(part)
    if not total:
        raise ValueError("CSV tidak memiliki transaksi.")
    if other_dest:
        raise ValueError(f"Ditemukan {other_dest} akun tujuan di luar C/M.")
    if merchant_fraud:
        raise ValueError("Ditemukan merchant fraud; asumsi populasi proposal tidak terpenuhi.")
    frame = pd.concat(chunks, ignore_index=True)
    del chunks, chunk, part
    gc.collect()
    print(f"Mengurutkan {len(frame):,} transaksi akun C menurut akun, step, baris asli...", flush=True)
    frame.sort_values(["nameDest", "step", "row_id"], kind="stable", inplace=True, ignore_index=True)
    account_ids, accounts = pd.factorize(frame["nameDest"], sort=False)
    ids = account_ids.astype("int32")
    del account_ids
    positions = np.arange(2, len(frame), dtype=np.int64)
    endpoints = positions[ids[2:] == ids[:-2]]
    del positions
    if not len(endpoints):
        raise ValueError("Tidak ada akun C dengan minimal tiga transaksi.")
    steps = frame["step"].to_numpy(copy=True)
    labels = frame["isFraud"].to_numpy(copy=True)
    rows = frame["row_id"].to_numpy(copy=True)
    features = np.empty((len(frame), len(FEATURES)), dtype="float32")
    features[:, :5] = frame[NUMERIC].to_numpy(dtype="float32")
    codes = frame["type"].cat.codes.to_numpy()
    features[:, 5:] = (codes[:, None] == np.arange(len(TYPES))[None, :]).astype("float32")
    customer_count = len(frame)
    del frame, codes
    gc.collect()
    X = np.lib.format.open_memmap(output_dir / "X.npy", mode="w+", dtype="float32", shape=(len(endpoints), 3, 10))
    print(f"Menulis {len(endpoints):,} sequence (3,10)...", flush=True)
    for start in range(0, len(endpoints), 100_000):
        stop = min(start + 100_000, len(endpoints))
        p = endpoints[start:stop]
        for timestep in range(3):
            X[start:stop, timestep] = features[p - 2 + timestep]
    X.flush()
    target_steps, y = steps[endpoints], labels[endpoints]
    arrays = {"y": y, "target_step": target_steps, "target_row": rows[endpoints], "account_id": ids[endpoints]}
    for name, values in arrays.items():
        np.save(output_dir / f"{name}.npy", values, allow_pickle=False)
    np.save(output_dir / "account_names.npy", np.asarray(accounts, dtype=str), allow_pickle=False)
    metadata = {
        "schema_version": 1, "source_file": csv_path.name, "source_sha256": sha256_file(csv_path),
        "raw_transactions": total, "raw_fraud": fraud, "customer_transactions": customer_count,
        "merchant_transactions": merchants, "merchant_fraud": merchant_fraud,
        "customer_accounts": len(accounts), "eligible_accounts": int(np.unique(ids[endpoints]).size),
        "sequences": len(endpoints), "sequence_fraud": int(y.sum()),
        "fraud_coverage": float(y.sum() / fraud) if fraud else 0.0,
        "sequence_length": 3, "features": FEATURES, "numeric_features": NUMERIC,
        "sort_keys": ["nameDest", "step", "original_row_id"], "label_rule": "isFraud(T3), inclusive",
        "min_target_step": int(target_steps.min()), "max_target_step": int(target_steps.max()),
        "duplicate_policy": "preserve_rows", "status": "complete",
    }
    # Metadata is written last; an interrupted preparation cannot masquerade as a complete dataset.
    write_json(output_dir / "metadata.json", metadata)
    audit_dataset(output_dir, output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return metadata


def load_dataset(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("status") != "complete" or meta.get("features") != FEATURES:
        raise ValueError("Metadata data tidak lengkap atau urutan fitur berubah.")
    result = {name: np.load(data_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ["X", "y", "target_step", "target_row", "account_id"]}
    n = meta["sequences"]
    if result["X"].shape != (n, 3, 10) or any(result[k].shape != (n,) for k in result if k != "X"):
        raise ValueError("Ukuran array tidak cocok dengan metadata.")
    result["metadata"] = meta
    return result


def audit_dataset(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    data = load_dataset(data_dir)
    rows = []
    for name, (train_end, test_start, test_end) in FOLDS.items():
        train = data["target_step"] <= train_end
        test = (data["target_step"] >= test_start) & (data["target_step"] <= test_end)
        n_train, n_fraud = int(train.sum()), int(data["y"][train].sum())
        nonfraud = n_train - n_fraud
        rows.append({"fold": name, "train_end": train_end, "test_start": test_start, "test_end": test_end,
                     "n_train": n_train, "fraud_train": n_fraud, "nonfraud_train": nonfraud,
                     "n_test": int(test.sum()), "fraud_test": int(data["y"][test].sum()),
                     "planned_fraud_after_smote": int(nonfraud * 0.5),
                     "planned_total_after_smote": nonfraud + int(nonfraud * 0.5),
                     "planned_synthetic": int(nonfraud * 0.5) - n_fraud})
    table = pd.DataFrame(rows)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(output_dir) / "dataset_audit.csv", index=False)
    write_json(Path(output_dir) / "dataset_summary.json", data["metadata"])
    return table
