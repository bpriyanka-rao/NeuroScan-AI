"""
=============================================================================
 Alzheimer's Disease Detection — Evaluation Module
 File: src/evaluate.py
 Author: [Your Name] | Final Year ML Project
=============================================================================
 Generates:
  - Classification report (Precision, Recall, F1, Accuracy)
  - Confusion matrix (saved as PNG)
  - ROC-AUC curves (per class, saved as PNG)
  - All metrics saved to models/evaluation_results.json
=============================================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")       # headless backend — safe for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import tensorflow as tf
from preprocess import get_data_generators, CLASS_NAMES

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(STATIC_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODELS_DIR, "alzheimer_model.h5")
RESULTS_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")


def evaluate(data_dir: str | None = None):
    if data_dir is None:
        data_dir = os.path.join(BASE_DIR, "data")

    print("[evaluate.py] Loading model …")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("[evaluate.py] Loading validation data …")
    _, val_gen = get_data_generators(data_dir)

    # ── Get predictions ─────────────────────────────────────────────────────
    y_proba = model.predict(val_gen, verbose=1)           # (N, 4)
    y_pred  = np.argmax(y_proba, axis=1)
    y_true  = val_gen.classes

    labels  = list(val_gen.class_indices.keys())

    # ── Classification report ────────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    print("\n" + classification_report(y_true, y_pred, target_names=labels))

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(cm, labels)

    # ── ROC-AUC ──────────────────────────────────────────────────────────────
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=4)
    try:
        auc = roc_auc_score(y_true_onehot, y_proba, multi_class="ovr", average="macro")
        _plot_roc_curves(y_true_onehot, y_proba, labels)
    except Exception:
        auc = None

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "accuracy":  report["accuracy"],
        "macro_f1":  report["macro avg"]["f1-score"],
        "roc_auc":   auc,
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall":    report[cls]["recall"],
                "f1_score":  report[cls]["f1-score"],
                "support":   report[cls]["support"],
            }
            for cls in labels
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Evaluation results saved → {RESULTS_PATH}")
    return results


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray, labels: list[str]):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Alzheimer's Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(STATIC_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"   → Confusion matrix saved: {save_path}")


def _plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, labels: list[str]):
    colors = ["#3B82F6", "#06B6D4", "#8B5CF6", "#F59E0B"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (label, color) in enumerate(zip(labels, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_proba[:, i])
        auc_val = roc_auc_score(y_true[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC-AUC Curves — Multi-class", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(STATIC_DIR, "roc_curves.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"   → ROC curves saved: {save_path}")


if __name__ == "__main__":
    evaluate()
