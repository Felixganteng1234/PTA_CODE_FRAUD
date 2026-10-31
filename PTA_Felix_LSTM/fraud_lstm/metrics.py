"""Evaluasi probabilitas fraud, agregasi temporal, dan bootstrap berpasangan.

Average precision (AP) dan luas trapezoidal kurva precision-recall dilaporkan
terpisah. Bootstrap default mengambil ulang sequence; interval tersebut
mengasumsikan sequence independen meskipun sliding window dapat bertumpang tindih.
Berikan ``groups`` berupa identitas akun untuk bootstrap cluster per akun.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

BOOTSTRAP_METRICS = (
    "precision", "recall", "f1", "average_precision", "pr_auc_trapezoid"
)


def _validate(y_true, probabilities, threshold=0.5):
    """Pastikan label biner dan probabilitas satu dimensi yang sejajar."""
    try:
        y = np.asarray(y_true, dtype=np.float64)
        p = np.asarray(probabilities, dtype=np.float64)
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("Label, probabilitas, dan threshold harus numerik.") from exc
    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("Label dan probabilitas harus berupa array satu dimensi.")
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Label dan probabilitas harus sama panjang dan tidak kosong.")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Label hanya boleh berisi 0 atau 1.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilitas harus finite dan berada dalam rentang [0, 1].")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold harus finite dan berada dalam rentang [0, 1].")
    return y.astype(np.int8, copy=False), p, threshold


def _calculate(y, p, threshold, include_roc=True):
    predicted = p >= threshold
    actual = y == 1
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    tn = int(len(y) - tp - fp - fn)
    n_fraud = tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_fraud if n_fraud else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    if n_fraud:
        curve_precision, curve_recall, _ = precision_recall_curve(y, p)
        ap = float(average_precision_score(y, p))
        pr_auc = float(auc(curve_recall, curve_precision))
    else:
        ap = pr_auc = None
    roc_auc = (
        float(roc_auc_score(y, p))
        if include_roc and 0 < n_fraud < len(y)
        else None
    )
    return {
        "average_precision": ap,
        "pr_auc_trapezoid": pr_auc,
        "roc_auc": roc_auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "n_samples": int(len(y)), "n_fraud": int(n_fraud),
    }


def evaluate_predictions(y_true, probabilities, threshold=0.5) -> dict[str, Any]:
    """Hitung metrik; AP/PR-AUC None tanpa fraud, ROC-AUC None jika satu kelas.

    Kelas prediksi adalah 1 bila probabilitas >= threshold. Pembagian nol pada
    precision, recall, dan F1 diberi nilai 0. Semua nilai dapat disimpan ke JSON.
    """
    y, p, threshold = _validate(y_true, probabilities, threshold)
    return _calculate(y, p, threshold)


def pooled_summary(predictions: Mapping, threshold=0.5) -> dict[str, Any]:
    """Gabungkan prediksi fold untuk P/R/F1; ringkas AP dan PR-AUC antar-fold.

    SD menggunakan ddof=1 dan bernilai None jika kurang dari dua fold memiliki
    metrik terdefinisi. Fold tanpa fraud tetap masuk metrik pooled tetapi tidak
    dimasukkan ke rata-rata AP/PR-AUC.
    """
    if not predictions:
        raise ValueError("Minimal satu fold diperlukan untuk agregasi.")
    folds, labels, probabilities = {}, [], []
    for fold, pair in predictions.items():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("Setiap fold harus berisi pasangan (y_true, probabilities).")
        y, p, checked_threshold = _validate(pair[0], pair[1], threshold)
        folds[str(fold)] = _calculate(y, p, checked_threshold)
        labels.append(y)
        probabilities.append(p)
    result = {
        "pooled": evaluate_predictions(
            np.concatenate(labels), np.concatenate(probabilities), threshold
        ),
        "folds": folds,
        "n_folds": len(folds),
    }
    for metric in ("average_precision", "pr_auc_trapezoid"):
        values = [item[metric] for item in folds.values() if item[metric] is not None]
        result[f"{metric}_mean"] = float(np.mean(values)) if values else None
        result[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    result["n_defined_ap_folds"] = sum(
        item["average_precision"] is not None for item in folds.values()
    )
    return result


def paired_bootstrap(
    y, pure, smote, threshold=0.5, n_bootstrap=1000, seed=42, groups=None
) -> dict[str, Any]:
    """CI percentile 95% dengan indeks bootstrap sama untuk kedua model.

    ``groups`` opsional harus sejajar dengan prediksi, umumnya nameDest target.
    Bootstrap cluster mengambil akun dengan pengembalian dan memasukkan seluruh
    sequence setiap akun terpilih, termasuk duplikasi saat akun dipilih berulang.
    Setiap metrik mencatat jumlah replikasi valid dan dilewati. Replikasi tanpa
    fraud hanya dilewati untuk AP/PR-AUC; P/R/F1 tetap mengikuti zero_division=0.
    """
    y, pure, threshold = _validate(y, pure, threshold)
    _, smote, _ = _validate(y, smote, threshold)
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap harus bilangan bulat positif.")
    rng = np.random.default_rng(seed)
    cluster_indices = None
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(y):
            raise ValueError("groups harus satu dimensi dan sejajar dengan label.")
        if any(value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) for value in groups):
            raise ValueError("groups tidak boleh berisi identitas kosong.")
        try:
            _, inverse = np.unique(groups, return_inverse=True)
        except TypeError as exc:
            raise ValueError("Identitas groups harus memiliki tipe yang konsisten.") from exc
        order = np.argsort(inverse, kind="stable")
        boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
        cluster_indices = np.split(order, boundaries)
    estimates = {
        "pure": _calculate(y, pure, threshold, include_roc=False),
        "smote": _calculate(y, smote, threshold, include_roc=False),
    }
    samples = {
        condition: {metric: [] for metric in BOOTSTRAP_METRICS}
        for condition in ("pure", "smote", "delta_smote_minus_pure")
    }
    for _ in range(int(n_bootstrap)):
        if cluster_indices is None:
            indices = rng.integers(0, len(y), size=len(y))
        else:
            chosen = rng.integers(0, len(cluster_indices), size=len(cluster_indices))
            indices = np.concatenate([cluster_indices[index] for index in chosen])
        metrics_pure = _calculate(y[indices], pure[indices], threshold, include_roc=False)
        metrics_smote = _calculate(y[indices], smote[indices], threshold, include_roc=False)
        for metric in BOOTSTRAP_METRICS:
            left, right = metrics_pure[metric], metrics_smote[metric]
            if left is not None:
                samples["pure"][metric].append(left)
            if right is not None:
                samples["smote"][metric].append(right)
            if left is not None and right is not None:
                samples["delta_smote_minus_pure"][metric].append(right - left)
    result = {
        "method": "paired_cluster_percentile" if cluster_indices is not None else "paired_sequence_percentile",
        "confidence_level": 0.95,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed) if seed is not None else None,
        "n_groups": len(cluster_indices) if cluster_indices is not None else None,
        "threshold": threshold,
        "assumption": (
            "Cluster akun diasumsikan independen; ketergantungan lintas akun/waktu belum diatasi."
            if cluster_indices is not None else
            "Sequence diasumsikan independen. Window yang tumpang tindih dapat membuat CI terlalu sempit."
        ),
    }
    for condition, condition_samples in samples.items():
        result[condition] = {}
        for metric, values in condition_samples.items():
            if condition == "delta_smote_minus_pure":
                left, right = estimates["pure"][metric], estimates["smote"][metric]
                estimate = right - left if left is not None and right is not None else None
            else:
                estimate = estimates[condition][metric]
            ci = np.percentile(values, [2.5, 97.5]) if values else [None, None]
            result[condition][metric] = {
                "estimate": estimate,
                "ci_lower": float(ci[0]) if ci[0] is not None else None,
                "ci_upper": float(ci[1]) if ci[1] is not None else None,
                "n_valid": len(values),
                "n_skipped": int(n_bootstrap) - len(values),
            }
    return result


def create_plots(output_dir, predictions: Mapping, threshold=0.5) -> dict[str, str]:
    """Simpan kurva PR gabungan dan confusion matrix tiap kondisi sebagai PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not predictions:
        raise ValueError("Prediksi minimal satu kondisi diperlukan.")
    validated = {
        str(condition): _validate(pair[0], pair[1], threshold)
        for condition, pair in predictions.items()
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {}
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    for condition, (y, p, _) in validated.items():
        if np.count_nonzero(y):
            precision, recall, _ = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            ax.step(recall, precision, where="post", label=f"{condition} (AP={ap:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Kurva Precision-Recall", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.2)
    if ax.lines:
        ax.legend(loc="best")
    path = output_dir / "precision_recall.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    output["precision_recall"] = str(path.resolve())
    for position, (condition, (y, p, checked_threshold)) in enumerate(validated.items(), start=1):
        matrix = confusion_matrix(y, p >= checked_threshold, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5.8, 5), layout="constrained")
        ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center",
                        color="white" if matrix[row, col] > matrix.max() / 2 else "black", fontsize=13)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-fraud", "Fraud"],
               yticklabels=["Non-fraud", "Fraud"], xlabel="Prediksi", ylabel="Label aktual",
               title=f"{condition}\nThreshold = {checked_threshold:g}")
        safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in condition)
        path = output_dir / f"confusion_{position}_{safe_name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        output[f"confusion_{condition}"] = str(path.resolve())
    return output
