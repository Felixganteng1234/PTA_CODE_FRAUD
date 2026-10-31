"""Temporal LSTM experiment with train-only preprocessing and clean validation.

The smoke profile exercises the real pipeline on uniform subsets of training
sequences. All validation and test sequences remain untouched and complete.
It is a software check, not the final research experiment.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from .data import load_dataset
from . import metrics


FOLDS = (
    ("fold1", 400, 401, 480),
    ("fold2", 480, 481, 560),
    ("fold3", 560, 561, 640),
    ("final_holdout", 640, 641, 743),
)
NUMERIC_FEATURES = 5
PROTOCOL_VERSION = 2
RUNTIME_CONTROLS = {"resume", "run_stage", "trial_budget"}
INITIAL_TRIAL = {"lr": 0.001, "dropout": 0.2, "batch_size": 512, "patience": 5}


def _json(path: Path, value) -> None:
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=convert, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in records for k in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str, allow_nan=False).encode()).hexdigest()


def _experiment_identity(data_dir: Path, metadata: dict, config: dict) -> dict:
    """Hash actual processed arrays; identical metadata alone is insufficient."""
    semantic_config = {key: value for key, value in config.items() if key not in RUNTIME_CONTROLS}
    hashes = {f"{name}.npy": _sha256_file(data_dir / f"{name}.npy")
              for name in ("X", "y", "target_step", "target_row", "account_id")}
    dataset = {"metadata": metadata, "array_sha256": hashes}
    identity = {"protocol_version": PROTOCOL_VERSION, "config_sha256": _canonical_hash(semantic_config),
                "data_sha256": _canonical_hash(dataset)}
    identity["experiment_signature"] = _canonical_hash(identity)
    identity["array_sha256"] = hashes
    return identity


def _progress(output: Path, **updates):
    path = Path(output) / "progress.json"
    prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    prior.update(updates)
    prior.update({"updated_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()})
    _json(path, prior)


def _atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


@contextmanager
def _run_lock(output: Path):
    """OS releases this local advisory lock even after a killed process."""
    handle = (output / ".run.lock").open("a+b")
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise FileExistsError(f"Another process is using this run: {output}") from error
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _load_completed_fold(output: Path, condition: str, name: str, signature: str):
    marker = output / "models" / condition / name / "completion.json"
    if not marker.exists():
        return None
    completed = json.loads(marker.read_text(encoding="utf-8"))
    if completed.get("experiment_signature") != signature:
        raise ValueError(f"Fold checkpoint has a different experiment signature: {marker}")
    required = {f"models/{condition}/{name}/{file}" for file in
                ("model.keras", "scaler.joblib", "training_record.json", "epoch_selection_history.csv", "refit_history.csv")}
    required.add(f"predictions/{condition}_{name}.npz")
    if set(completed.get("artifacts", {})) != required:
        raise ValueError(f"Incomplete artifact list in fold checkpoint: {marker}")
    for relative, expected in completed["artifacts"].items():
        path = output / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Corrupt fold checkpoint artifact: {path}; preserve this run and use a new output directory.")
    return completed["result"]


def _commit_fold(output: Path, staging: Path, condition: str, name: str, signature: str, result: dict):
    """Publish artifacts individually; only the final atomic marker commits them."""
    mapping = {file: f"models/{condition}/{name}/{file}" for file in
               ("model.keras", "scaler.joblib", "training_record.json", "epoch_selection_history.csv", "refit_history.csv")}
    mapping["predictions.npz"] = f"predictions/{condition}_{name}.npz"
    hashes = {relative: _sha256_file(staging / file) for file, relative in mapping.items()}
    for file, relative in mapping.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        (staging / file).replace(target)
    _json(output / "models" / condition / name / "completion.json", {
        "experiment_signature": signature, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": hashes, "result": result})


def _counts(y) -> dict:
    count = len(y)
    positive = int(np.count_nonzero(np.asarray(y) == 1))
    return {"n": count, "fraud": positive, "nonfraud": count - positive,
            "fraud_fraction": positive / count if count else None}


def _ensure_classes(y, name: str, min_positive: int = 1) -> None:
    counts = _counts(y)
    if counts["fraud"] < min_positive or counts["nonfraud"] < 1:
        raise ValueError(f"{name}: requires >= {min_positive} fraud and >= 1 nonfraud; {counts}")


def _select_train(data: dict, end: int, config: dict, *, inner_end=None):
    """Select once per period, shared by both treatments; never sample tests."""
    candidates = np.flatnonzero(data["target_step"] <= end)
    if not len(candidates):
        raise ValueError(f"No training sequences at target_step <= {end}.")
    min_positive = int(config.get("smote_k", 5)) + 1
    maximum = len(candidates) if config["profile"] == "full" else min(
        int(config.get("smoke_max_train", 12000)), len(candidates))
    if maximum < 1:
        raise ValueError("smoke_max_train must be positive.")
    attempts = []
    while True:
        if maximum == len(candidates):
            selected = candidates
        else:
            rng = np.random.default_rng(int(config["seed"]) + end)
            selected = np.sort(rng.choice(candidates, size=maximum, replace=False))
        check = selected if inner_end is None else selected[data["target_step"][selected] <= inner_end]
        counts = _counts(data["y"][check])
        attempts.append({"draw_size": maximum, "early_training_counts": counts})
        if counts["fraud"] >= min_positive and counts["nonfraud"] > 0:
            break
        if maximum == len(candidates):
            raise ValueError(f"Insufficient classes in training period <= {inner_end or end}: {counts}")
        maximum = min(len(candidates), max(maximum + 1, maximum * 2))
    return selected, {
        "train_end": end, "internal_train_end": inner_end,
        "method": "all_sequences" if config["profile"] == "full" else "uniform_without_replacement",
        "seed": int(config["seed"]) + end,
        "source": _counts(data["y"][candidates]),
        "selected": _counts(data["y"][selected]),
        "attempts": attempts,
        "note": "Smoke sample size doubles if early-training fraud count is too low for k=5 SMOTE; no class weighting or stratified selection.",
    }


def _scaled(data: dict, train_indices, eval_indices=None):
    """Fit only selected training observations, retaining float32 in RAM."""
    X_train = np.asarray(data["X"][train_indices], dtype=np.float32).copy()
    y_train = np.asarray(data["y"][train_indices], dtype=np.int8).copy()
    scaler = StandardScaler()
    chunk_size = 50000
    for start in range(0, len(X_train), chunk_size):
        scaler.partial_fit(X_train[start:start + chunk_size, :, :NUMERIC_FEATURES].reshape(-1, NUMERIC_FEATURES))
    arrays = [X_train]
    X_eval = y_eval = None
    if eval_indices is not None:
        X_eval = np.asarray(data["X"][eval_indices], dtype=np.float32).copy()
        y_eval = np.asarray(data["y"][eval_indices], dtype=np.int8).copy()
        arrays.append(X_eval)
    for values in arrays:
        for start in range(0, len(values), chunk_size):
            block = values[start:start + chunk_size, :, :NUMERIC_FEATURES]
            block[:] = scaler.transform(block.reshape(-1, NUMERIC_FEATURES)).reshape(block.shape)
    return X_train, y_train, X_eval, y_eval, scaler


def _resample(X, y, condition: str, config: dict):
    before = _counts(y)
    if condition == "smote":
        _ensure_classes(y, "SMOTE train", int(config.get("smote_k", 5)) + 1)
        if before["fraud"] >= int(before["nonfraud"] * float(config["smote_ratio"])):
            raise ValueError("SMOTE target ratio must exceed the observed fraud/nonfraud ratio.")
        n, steps, features = X.shape
        sampler = SMOTE(sampling_strategy=float(config["smote_ratio"]),
                        k_neighbors=int(config.get("smote_k", 5)),
                        random_state=int(config["seed"]))
        flattened, y = sampler.fit_resample(X.reshape(n, steps * features), y)
        X = flattened.reshape(-1, steps, features).astype(np.float32, copy=False)
    return X, y, {"before": before, "after": _counts(y),
                  "synthetic_added": len(y) - before["n"]}


def build_model(input_shape, parameters: dict, seed: int, steps_per_execution: int = 1):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(float(parameters["dropout"])),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(float(parameters["dropout"])),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(parameters["lr"])),
                  loss="binary_crossentropy", steps_per_execution=int(steps_per_execution))
    return model


def _dataset(X, y, batch_size: int, seed: int, shuffle: bool):
    """Batch generator avoids an additional full TensorFlow tensor copy."""
    import tensorflow as tf
    epoch = [0]
    def batches():
        rng = np.random.default_rng(seed + epoch[0])
        epoch[0] += 1
        order = rng.permutation(len(X)) if shuffle else None
        for start in range(0, len(X), batch_size):
            indices = slice(start, start + batch_size) if order is None else order[start:start + batch_size]
            yield np.asarray(X[indices], dtype=np.float32), np.asarray(y[indices], dtype=np.float32)
    dataset = tf.data.Dataset.from_generator(batches, output_signature=(
        tf.TensorSpec(shape=(None, *X.shape[1:]), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ))
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(math.ceil(len(X) / batch_size)))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    options.experimental_deterministic = True
    return dataset.with_options(options).prefetch(1)


def _fit(X, y, params, config, *, validation=None, epochs=None, label="training",
         history_path=None, progress_output=None, progress_fields=None):
    import tensorflow as tf
    model = build_model(X.shape[1:], params, int(config["seed"]),
                        steps_per_execution=int(config.get("steps_per_execution", 1)))
    callbacks = []
    maximum_epochs = int(epochs or config["max_epochs"])
    checkpoint_history = []
    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if progress_output is not None:
                _progress(progress_output, **(progress_fields or {}), status="running", label=label,
                          epoch=epoch + 1, max_epochs=maximum_epochs, event="epoch_started")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            clean_logs = {key: float(value) for key, value in logs.items()}
            if not all(np.isfinite(value) for value in clean_logs.values()):
                raise ValueError(f"Non-finite training loss in {label}, epoch {epoch + 1}: {clean_logs}")
            checkpoint_history.append({"epoch": epoch + 1, **clean_logs})
            if history_path is not None:
                _csv(Path(history_path), checkpoint_history)
            if progress_output is not None:
                monitor = "val_loss" if validation is not None else "loss"
                best = min(checkpoint_history, key=lambda row: row[monitor])
                _progress(progress_output, **(progress_fields or {}), status="running", label=label,
                          epoch=epoch + 1, max_epochs=maximum_epochs, event="epoch_completed",
                          best_epoch=best["epoch"], best_loss=best[monitor], logs=clean_logs)
            detail = " ".join(f"{key}={value:.6f}" for key, value in logs.items())
            print(f"[{label}] epoch {epoch + 1}: {detail}", flush=True)
    callbacks.append(Progress())
    validation_data = None
    if validation is not None:
        validation_data = _dataset(*validation, int(params["batch_size"]), int(config["seed"]), False)
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=int(params["patience"]), restore_best_weights=True))
    train_data = _dataset(X, y, int(params["batch_size"]), int(config["seed"]), True)
    history = model.fit(train_data, validation_data=validation_data,
                        epochs=maximum_epochs, callbacks=callbacks, verbose=0)
    rows = [{"epoch": i + 1, **{key: float(values[i]) for key, values in history.history.items()}}
            for i in range(len(history.history["loss"]))]
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1 if validation is not None else len(rows)
    return model, rows, best_epoch


def _predict(model, X, batch_size: int = 4096):
    result = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), batch_size):
        result[start:start + batch_size] = np.asarray(model(X[start:start + batch_size], training=False)).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("Model returned non-finite probabilities.")
    return result


def _tune(data, output, config, condition, selected, validation_indices, signature):
    import optuna
    from optuna.trial import TrialState
    from sklearn.metrics import average_precision_score
    tune_dir = output / "tuning" / condition
    tune_dir.mkdir(parents=True, exist_ok=True)
    startup_trials = max(1, min(10, int(config["n_trials"]) // 2))
    sampler_path = tune_dir / "sampler.joblib"
    sampler = optuna.samplers.TPESampler(seed=int(config["seed"]), n_startup_trials=startup_trials)
    if sampler_path.exists():
        sampler_state = joblib.load(sampler_path)
        if sampler_state["experiment_signature"] != signature:
            raise ValueError("Sampler checkpoint belongs to a different experiment.")
        sampler = sampler_state["sampler"]
    storage = optuna.storages.RDBStorage(
        url="sqlite:///" + (output / "optuna.sqlite3").resolve().as_posix(),
        engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.create_study(study_name=f"{condition}_{signature[:16]}", direction="maximize",
                                sampler=sampler, storage=storage, load_if_exists=True)
    known_signature = study.user_attrs.get("experiment_signature")
    if known_signature not in (None, signature) or (known_signature is None and study.trials):
        raise ValueError("Optuna study signature is absent or differs from this experiment.")
    study.set_user_attr("experiment_signature", signature)
    study.set_user_attr("initial_trial_reason", "Prespecified generic starting point inside proposal bounds; identical for both conditions; no test or holdout evidence.")
    if not study.trials:
        study.enqueue_trial(INITIAL_TRIAL, user_attrs={"prespecified_initial_trial": True})
    # The output-directory OS lock guarantees no live worker owns RUNNING trials.
    # A fully written result can be committed after a crash; otherwise retry the
    # same hyperparameters from epoch 1 and retain the partial history for audit.
    for frozen in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        marker = tune_dir / f"trial_{frozen.number:03d}_result.json"
        recovered = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
        prediction_path = tune_dir / f"trial_{frozen.number:03d}_validation.npz"
        history_path = tune_dir / f"trial_{frozen.number:03d}_history.csv"
        valid = (recovered is not None and recovered.get("experiment_signature") == signature
                 and prediction_path.is_file() and history_path.is_file()
                 and _sha256_file(prediction_path) == recovered.get("prediction_sha256")
                 and _sha256_file(history_path) == recovered.get("history_sha256")
                 and "selected_epoch" in frozen.user_attrs)
        if valid:
            study.tell(frozen.number, float(recovered["value"]))
        else:
            study.tell(frozen.number, state=TrialState.FAIL)
            study.enqueue_trial(frozen.params, user_attrs={"retry_of_trial": frozen.number})
    completed_before = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    if completed_before > int(config["n_trials"]):
        raise ValueError("Study contains more completed trials than the signed target.")
    remaining = int(config["n_trials"]) - completed_before
    budget = config.get("trial_budget")
    allowance = remaining if budget is None else min(remaining, int(budget))
    _progress(output, stage="tuning", condition=condition, fold=None, trial=None,
              completed_trials=completed_before, target_trials=int(config["n_trials"]),
              invocation_trial_budget=allowance, event="study_opened")
    if allowance:
        print(f"[{condition}] tuning <=340; validation 341-400; completed={completed_before}; new budget={allowance}", flush=True)
        _progress(output, event="preparing_training_data")
        X, y, X_val, y_val, scaler = _scaled(data, selected, validation_indices)
        X, y, counts = _resample(X, y, condition, config)
        _ensure_classes(y_val, "Optuna clean validation")
        _json(tune_dir / "data_counts.json", {"train": counts, "validation": _counts(y_val)})
        def objective(trial):
            params = {"lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                      "dropout": trial.suggest_float("dropout", 0.2, 0.5),
                      "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                      "patience": trial.suggest_int("patience", 3, 10)}
            history_path = tune_dir / f"trial_{trial.number:03d}_history.csv"
            model, history, best_epoch = _fit(
                X, y, params, config, validation=(X_val, y_val), label=f"{condition}/trial{trial.number}",
                history_path=history_path, progress_output=output,
                progress_fields={"stage": "tuning", "condition": condition, "fold": None, "trial": trial.number})
            probability = _predict(model, X_val)
            value = float(average_precision_score(y_val, probability))
            trial.set_user_attr("selected_epoch", best_epoch)
            trial.set_user_attr("validation_fraud", int(y_val.sum()))
            _csv(history_path, history)
            prediction_path = tune_dir / f"trial_{trial.number:03d}_validation.npz"
            _atomic_npz(prediction_path, y_true=y_val, probability=probability,
                        sequence_index=validation_indices, target_step=data["target_step"][validation_indices],
                        target_row=data["target_row"][validation_indices], partition="internal_validation",
                        train_end=340, val_start=341, val_end=400, trial_number=trial.number)
            _json(tune_dir / f"trial_{trial.number:03d}_result.json", {
                "experiment_signature": signature, "value": value, "parameters": params,
                "selected_epoch": best_epoch, "validation_fraud": int(y_val.sum()),
                "prediction_sha256": _sha256_file(prediction_path), "history_sha256": _sha256_file(history_path)})
            del model, probability
            gc.collect()
            return value
        def checkpoint(study, frozen):
            temp = sampler_path.with_suffix(".joblib.tmp")
            joblib.dump({"experiment_signature": signature, "sampler": study.sampler,
                         "last_finished_trial": frozen.number}, temp)
            temp.replace(sampler_path)
            study.trials_dataframe().to_csv(tune_dir / "trials.csv.tmp", index=False)
            (tune_dir / "trials.csv.tmp").replace(tune_dir / "trials.csv")
            _progress(output, event="trial_completed", condition=condition, trial=frozen.number,
                      completed_trials=len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))))
        study.optimize(objective, n_trials=allowance, gc_after_trial=True, callbacks=[checkpoint])
        del X, y, X_val, y_val, scaler
        gc.collect()
    completed = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    study.trials_dataframe().to_csv(tune_dir / "trials.csv.tmp", index=False)
    (tune_dir / "trials.csv.tmp").replace(tune_dir / "trials.csv")
    params = None
    if completed:
        best = study.best_trial
        params = dict(best.params)
        saved = {
            "parameters": params, "objective_average_precision": study.best_value,
            "trial_number": best.number, "selected_epoch": best.user_attrs["selected_epoch"],
            "tpe_startup_trials": startup_trials, "completed_trials": completed,
            "target_trials": int(config["n_trials"]), "target_complete": completed == int(config["n_trials"]),
            "experiment_signature": signature, "initial_trial": INITIAL_TRIAL,
            "initial_trial_reason": study.user_attrs["initial_trial_reason"]}
        _json(tune_dir / "best_params.json", saved)
        source = tune_dir / f"trial_{best.number:03d}_validation.npz"
        result_record = json.loads((tune_dir / f"trial_{best.number:03d}_result.json").read_text(encoding="utf-8"))
        if _sha256_file(source) != result_record["prediction_sha256"]:
            raise ValueError("Best trial validation prediction checkpoint is corrupt.")
        target = tune_dir / "best_validation_predictions.npz"
        shutil.copyfile(source, target.with_suffix(".npz.tmp"))
        target.with_suffix(".npz.tmp").replace(target)
        _json(tune_dir / "best_validation_metadata.json", {
            "partition": "internal_validation", "train_end": 340, "val_start": 341, "val_end": 400,
            "trial_number": best.number, "experiment_signature": signature,
            "prediction_sha256": _sha256_file(target), "selected_by": "average_precision_internal_validation"})
    return {"parameters": params, "completed_trials": completed, "target_complete": completed == int(config["n_trials"])}


def _evaluate_fold(data, output, config, condition, fold, selected, params, signature):
    name, end, test_start, test_end = fold
    completed = _load_completed_fold(output, condition, name, signature)
    if completed is not None:
        print(f"[{condition}/{name}] verified checkpoint reused", flush=True)
        return completed
    internal_end = end - 60
    early_train = selected[data["target_step"][selected] <= internal_end]
    validation = np.flatnonzero((data["target_step"] > internal_end) & (data["target_step"] <= end))
    test = np.flatnonzero((data["target_step"] >= test_start) & (data["target_step"] <= test_end))
    _ensure_classes(data["y"][validation], f"{name} clean validation")
    _ensure_classes(data["y"][test], f"{name} clean test")
    directory = output / ".pending" / condition / name / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    print(f"[{condition}/{name}] selecting epoch using train<={internal_end}; validation {internal_end+1}-{end}", flush=True)
    X, y, X_val, y_val, scaler = _scaled(data, early_train, validation)
    X, y, early_counts = _resample(X, y, condition, config)
    model, history, best_epoch = _fit(X, y, params, config, validation=(X_val, y_val),
                                     label=f"{condition}/{name}/epoch-selection",
                                     history_path=directory / "epoch_selection_history.csv", progress_output=output,
                                     progress_fields={"stage": "epoch_selection", "condition": condition, "fold": name, "trial": None})
    _csv(directory / "epoch_selection_history.csv", history)
    del X, y, X_val, y_val, scaler, model
    gc.collect()
    print(f"[{condition}/{name}] refit on selected train<={end}, epochs={best_epoch}; untouched test {test_start}-{test_end}", flush=True)
    X, y, X_test, y_test, scaler = _scaled(data, selected, test)
    X, y, final_counts = _resample(X, y, condition, config)
    model, history, _ = _fit(X, y, params, config, epochs=best_epoch,
                             label=f"{condition}/{name}/refit", history_path=directory / "refit_history.csv",
                             progress_output=output,
                             progress_fields={"stage": "refit", "condition": condition, "fold": name, "trial": None})
    _csv(directory / "refit_history.csv", history)
    model.save(directory / "model.keras")
    joblib.dump(scaler, directory / "scaler.joblib")
    probability = _predict(model, X_test)
    result = metrics.evaluate_predictions(y_test, probability, threshold=float(config["threshold"]))
    result.update({"condition": condition, "fold": name, "train_end": end,
                   "test_start": test_start, "test_end": test_end,
                   "selected_epoch": best_epoch, "n_train_original": final_counts["before"]["n"],
                   "n_train_after_smote": final_counts["after"]["n"],
                   "n_test": len(y_test), "fraud_test": int(y_test.sum()),
                   "profile": config["profile"]})
    _atomic_npz(directory / "predictions.npz",
                        y_true=y_test, probability=probability, sequence_index=test,
                        target_step=data["target_step"][test],
                        target_row=data["target_row"][test], account_id=data["account_id"][test])
    _json(directory / "training_record.json", {
        "parameters": params, "selected_epoch": best_epoch,
        "epoch_selection": {"train_end": internal_end, "validation_start": internal_end + 1,
                            "validation_end": end, "train_counts": early_counts,
                            "validation_counts": _counts(data["y"][validation])},
        "refit": {"train_end": end, "counts": final_counts},
        "test_counts": _counts(y_test), "scaler_numeric_feature_indices": list(range(NUMERIC_FEATURES)),
        "threshold": float(config["threshold"]), "test_used_for_selection": False})
    _commit_fold(output, directory, condition, name, signature, result)
    print(f"[{condition}/{name}] metrics: {json.dumps(result, default=str)}", flush=True)
    del X, y, X_test, y_test, scaler, model, probability
    gc.collect()
    return result


def run_experiment(data_dir: Path, output_dir: Path, config: dict) -> None:
    """Resume only the same signed experiment; runtime stage/budget may change."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    defaults = {"profile": "full", "seed": 42, "n_trials": 20, "max_epochs": 50,
                "bootstrap_iterations": 1000, "threshold": 0.5, "cpu_threads": 4,
                "smote_ratio": 0.5, "smote_k": 5, "smoke_max_train": 12000,
                "steps_per_execution": 1, "resume": False, "run_stage": "all", "trial_budget": None}
    config = {**defaults, **dict(config)}
    if config["profile"] not in ("full", "smoke") or config["run_stage"] not in ("tune", "all"):
        raise ValueError("profile must be full/smoke and run_stage must be tune/all.")
    for key in ("n_trials", "max_epochs", "steps_per_execution", "cpu_threads", "bootstrap_iterations"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if type(config["resume"]) is not bool:
        raise ValueError("resume must be a boolean.")
    if config["trial_budget"] is not None and (type(config["trial_budget"]) is not int or config["trial_budget"] < 1):
        raise ValueError("trial_budget must be a positive integer or null.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with _run_lock(output_dir):
        _run_locked(data_dir, output_dir, config)


def _run_locked(data_dir, output_dir, config):
    manifest_path = output_dir / "run_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if previous is not None and not config["resume"]:
        raise FileExistsError(f"Run already exists: {output_dir}. Use resume with the unchanged experiment configuration.")
    if previous is None and config["resume"]:
        raise FileNotFoundError(f"No run manifest to resume in {output_dir}.")
    started = time.time()
    data = load_dataset(data_dir)
    identity = _experiment_identity(data_dir, data["metadata"], config)
    signature = identity["experiment_signature"]
    if previous is not None:
        if previous.get("experiment_signature") != signature:
            raise ValueError("Resume rejected: configuration, processed data, or protocol differs. Only resume, run_stage and trial_budget may change. Legacy runs require a new output directory.")
        manifest = previous
        accumulated_seconds = float(manifest.get("elapsed_seconds", 0.0))
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest.pop("error", None)
        manifest.pop("error_type", None)
    else:
        accumulated_seconds = 0.0
        manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                    **identity, "data_dir": str(data_dir.resolve()), "output_dir": str(output_dir.resolve()),
                    "validation_rule": "Last 60 steps of training only; scaler and SMOTE fit earlier training only. Select best val_loss epoch, then refit selected training with a new scaler. No test tuning.",
                    "smoke_warning": "SMOKE: training uniformly subsampled. Not a final research result." if config["profile"] == "smoke" else None,
                    "evaluation_sampling": "All clean validation and test sequences are evaluated in both profiles.",
                    "pr_auc_definition": "Average precision (non-interpolated sklearn average_precision_score)",
                    "completed_models": [], "resume_count": 0,
                    "initial_trial": INITIAL_TRIAL,
                    "initial_trial_reason": "Prespecified generic starting point within proposal bounds, identical for both treatments, not based on test/holdout.",
                    "checkpoint_rule": "Completed Optuna trials and hash-verified committed folds are reused. An interrupted trial/fold restarts from epoch 1; partial histories are retained."}
    manifest.update({"status": "running", "requested_stage": config["run_stage"],
                     "invocation_trial_budget": config["trial_budget"],
                     "dataset_counts": _counts(data["y"]), "dataset_metadata": dict(data["metadata"]),
                     "source_sha256": data["metadata"].get("source_sha256"),
                     "last_started_at_utc": datetime.now(timezone.utc).isoformat()})
    _json(manifest_path, manifest)
    _progress(output_dir, status="running", stage="initializing", event="run_started",
              completed_models=manifest["completed_models"], experiment_signature=signature)
    try:
        import tensorflow as tf
        manifest["runtime"] = {"python": sys.version, "platform": platform.platform(),
                               "packages": {name: importlib.metadata.version(name) for name in
                                            ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "tensorflow", "keras", "optuna")},
                               "devices": [device.name for device in tf.config.list_physical_devices()]}
        try:
            tf.config.threading.set_intra_op_parallelism_threads(int(config["cpu_threads"]))
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:
            manifest["thread_configuration_note"] = "TensorFlow already initialized; existing thread settings retained."
        tf.keras.utils.set_random_seed(int(config["seed"]))
        tf.config.experimental.enable_op_determinism()
        _json(manifest_path, manifest)
        selections = {}
        selection_records = []
        for name, end, _, _ in FOLDS:
            chosen, record = _select_train(data, end, config, inner_end=end - 60)
            selections[name] = chosen
            selection_records.append({"stage": name, **record})
        tune_selected, tune_record = _select_train(data, 340, config)
        selection_records.append({"stage": "optuna_train", **tune_record})
        _json(output_dir / "selection_counts.json", selection_records)
        selection_dir = output_dir / "selection"
        selection_dir.mkdir(exist_ok=True)
        for name, indices in {**selections, "optuna_train": tune_selected}.items():
            target = selection_dir / f"{name}_indices.npy"
            if target.exists():
                if not np.array_equal(np.load(target, allow_pickle=False), indices):
                    raise ValueError(f"Saved training selection differs: {target}")
            else:
                with target.with_suffix(".npy.tmp").open("wb") as stream:
                    np.save(stream, indices, allow_pickle=False)
                target.with_suffix(".npy.tmp").replace(target)
        validation_indices = np.flatnonzero((data["target_step"] >= 341) & (data["target_step"] <= 400))
        tuning = {}
        for condition in ("pure", "smote"):
            tuning[condition] = _tune(data, output_dir, config, condition, tune_selected, validation_indices, signature)
            manifest["tuning_completed_trials"] = {key: value["completed_trials"] for key, value in tuning.items()}
            _json(manifest_path, manifest)
        target_complete = all(item["target_complete"] for item in tuning.values())
        if not target_complete or config["run_stage"] == "tune":
            manifest["status"] = "tuned" if target_complete else "tuning_partial"
            manifest["elapsed_seconds"] = round(accumulated_seconds + time.time() - started, 3)
            _json(manifest_path, manifest)
            _progress(output_dir, status=manifest["status"], stage="tuning", event="invocation_completed",
                      tuning_completed_trials=manifest["tuning_completed_trials"])
            print(f"{manifest['status']}: {output_dir.resolve()}; no test or holdout evaluation performed in this invocation.", flush=True)
            return
        results = []
        # Tuning of both treatments finishes before any evaluation. Fold
        # checkpoints are reused without inspecting their scores for selection.
        for fold in FOLDS:
            for condition in ("pure", "smote"):
                result = _evaluate_fold(data, output_dir, config, condition, fold, selections[fold[0]],
                                        tuning[condition]["parameters"], signature)
                results.append(result)
                _csv(output_dir / "metrics.csv", results)
                model_key = f"{condition}/{fold[0]}"
                if model_key not in manifest["completed_models"]:
                    manifest["completed_models"].append(model_key)
                _json(manifest_path, manifest)
                _progress(output_dir, stage="evaluation", event="model_completed",
                          completed_models=manifest["completed_models"], condition=condition, fold=fold[0])
        _progress(output_dir, stage="summaries", event="bootstrap_started")
        _write_summaries(output_dir, config, results)
        manifest["status"] = "completed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = round(accumulated_seconds + time.time() - started, 3)
        _json(manifest_path, manifest)
        _progress(output_dir, status="completed", stage="completed", event="run_completed")
        print(f"Completed {config['profile']} experiment: {output_dir.resolve()}", flush=True)
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        manifest["elapsed_seconds"] = round(accumulated_seconds + time.time() - started, 3)
        _json(manifest_path, manifest)
        _progress(output_dir, status=manifest["status"], event="run_stopped", error_type=type(error).__name__, error=str(error))
        raise


def _write_summaries(output_dir, config, results):
    """Aggregate the exact persisted predictions; no model is retrained here."""
    pooled = []
    for condition in ("pure", "smote"):
        y_parts, p_parts = [], []
        for fold_name, _, _, _ in FOLDS[:3]:
            with np.load(output_dir / "predictions" / f"{condition}_{fold_name}.npz", allow_pickle=False) as saved:
                y_parts.append(saved["y_true"])
                p_parts.append(saved["probability"])
        summary = metrics.evaluate_predictions(np.concatenate(y_parts), np.concatenate(p_parts),
                                               threshold=float(config["threshold"]))
        summary["condition"] = condition
        fold_metrics = [row for row in results if row["condition"] == condition and row["fold"] != "final_holdout"]
        ap_key = next(key for key in ("pr_auc", "average_precision", "ap") if key in fold_metrics[0])
        values = np.array([row[ap_key] for row in fold_metrics])
        summary["pr_auc_fold_mean"] = float(values.mean())
        summary["pr_auc_fold_std_sample"] = float(values.std(ddof=1))
        summary["profile"] = config["profile"]
        pooled.append(summary)
    _csv(output_dir / "pooled_summary.csv", pooled)
    _json(output_dir / "pooled_summary.json", pooled)
    # The metrics module owns the paired bootstrap API and report formatting.
    with np.load(output_dir / "predictions" / "pure_final_holdout.npz", allow_pickle=False) as saved:
        y_true, pure_prob, target = saved["y_true"], saved["probability"], saved["target_row"]
    with np.load(output_dir / "predictions" / "smote_final_holdout.npz", allow_pickle=False) as saved:
        if not np.array_equal(y_true, saved["y_true"]) or not np.array_equal(target, saved["target_row"]):
            raise ValueError("Holdout predictions do not refer to the same ordered targets.")
        smote_prob = saved["probability"]
    bootstrap = metrics.paired_bootstrap(y_true, pure_prob, smote_prob,
                                         n_bootstrap=int(config["bootstrap_iterations"]),
                                         seed=int(config["seed"]), threshold=float(config["threshold"]))
    _json(output_dir / "bootstrap_uncertainty.json", bootstrap)
    bootstrap_rows = []
    for condition in ("pure", "smote", "delta_smote_minus_pure"):
        for metric_name, values in bootstrap[condition].items():
            bootstrap_rows.append({"condition": condition, "metric": metric_name,
                                   "profile": config["profile"], **values})
    _csv(output_dir / "bootstrap_uncertainty.csv", bootstrap_rows)
    figures = metrics.create_plots(output_dir / "plots", {
        "pure": (y_true, pure_prob), "smote": (y_true, smote_prob)},
        threshold=float(config["threshold"]))
    _json(output_dir / "figures.json", {
        key: Path(path).resolve().relative_to(output_dir.resolve()).as_posix()
        for key, path in figures.items()
    })
