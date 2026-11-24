"""Metric helpers shared by every model, so all results are computed the same way."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def message_metrics(y_true, y_pred, labels: list[str], benign: str = "ham") -> dict:
    """Multiclass metrics plus the spam-vs-ham and scam-vs-ham slices the proposal requires."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(labels)
        },
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }
    benign_mask = y_true == benign
    # False-positive rate: benign messages flagged as anything malicious.
    out["fpr_benign_flagged"] = float((y_pred[benign_mask] != benign).mean()) if benign_mask.any() else None

    # Binary slices: each malicious class against ham only.
    out["slices"] = {}
    for cls in labels:
        if cls == benign:
            continue
        m = np.isin(y_true, [benign, cls])
        yt = y_true[m] == cls
        flagged = y_pred[m] != benign  # caught as *any* malicious class
        tp = int((yt & flagged).sum())
        fp = int((~yt & flagged).sum())
        fn = int((yt & ~flagged).sum())
        out["slices"][f"{cls}_vs_{benign}"] = {
            "n": int(m.sum()),
            "recall_flagged": tp / max(1, tp + fn),
            "precision_flagged": tp / max(1, tp + fp),
            "recall_exact_class": float((y_pred[m][yt] == cls).mean()) if yt.any() else None,
        }
    return out


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "threshold": threshold,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "fpr": fp / max(1, fp + tn),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": {"labels": ["benign", "phishing"], "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]},
    }


def plot_confusion(matrix, labels, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(np.asarray(matrix), display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
