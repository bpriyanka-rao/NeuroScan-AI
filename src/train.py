"""
=============================================================================
 Alzheimer's Disease Detection — Model Training Script
 File: src/train.py
 Author: [Your Name] | Final Year ML Project
=============================================================================
 Pipeline:
  1. Load data generators from data/ directory
  2. Build MobileNetV2 transfer-learning model
  3. Phase 1 — Train only the custom head (frozen base)
  4. Phase 2 — Fine-tune top 30 layers of MobileNetV2
  5. Save best model to models/alzheimer_model.h5
  6. Save training history for dashboard charts
=============================================================================
"""

import argparse
import os
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from preprocess import get_data_generators, IMG_SIZE, CLASS_NAMES
from model_manager import save_metadata

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODELS_DIR, "alzheimer_model.h5")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")
LABELS_PATH  = os.path.join(MODELS_DIR, "class_labels.json")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# ── Hyperparameters ─────────────────────────────────────────────────────────
EPOCHS_HEAD   = 15      # Phase 1 — head only
EPOCHS_FINETUNE = 20   # Phase 2 — fine-tuning
LR_HEAD       = 1e-3
LR_FINETUNE   = 1e-5
NUM_CLASSES   = 4
DROPOUT_RATE  = 0.40


def _compute_class_weights(labels: np.ndarray) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    total = labels.shape[0]
    weights = {int(cls): float(total / (len(unique) * count)) for cls, count in zip(unique, counts)}
    return weights


def build_model(architecture: str = "mobilenetv2") -> tuple[Model, tf.keras.Model]:
    """
    Construct the classification model using the selected backbone.

    Supported architectures:
      - mobilenetv2
      - efficientnet
    """
    arch = architecture.lower()
    if arch == "mobilenetv2":
        base = MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        model_name = "MobileNetV2"
    elif arch == "efficientnet":
        base = EfficientNetB0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        model_name = "EfficientNetB0"
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    base.trainable = False   # Freeze for Phase 1
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs, name=f"AlzheimerDetector_{model_name}")
    return model, base


def train(architecture: str = "mobilenetv2"):
    print("=" * 60)
    print("  Alzheimer's Disease Detection — Training Pipeline")
    print("=" * 60)
    print(f"  Backbone: {architecture}")

    # ── Step 1: Load data ───────────────────────────────────────────────────
    print("\n[1/5] Loading data generators …")
    train_gen, val_gen = get_data_generators(DATA_DIR, model_name=architecture)
    print(f"      Train samples : {train_gen.samples}")
    print(f"      Val   samples : {val_gen.samples}")
    print(f"      Class indices : {train_gen.class_indices}")

    class_weights = _compute_class_weights(train_gen.classes)
    print(f"      Class weights: {class_weights}")

    # ── Step 2: Build model ─────────────────────────────────────────────────
    print(f"\n[2/5] Building {architecture.title()} model …")
    model, base = build_model(architecture)
    model.summary()

    # ── Step 3: Phase 1 — Train head only ──────────────────────────────────
    print("\n[3/5] Phase 1 — Training classification head …")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_p1 = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss", verbose=1),
        CSVLogger(os.path.join(MODELS_DIR, "phase1_log.csv")),
    ]

    history_p1 = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen,
        callbacks=callbacks_p1,
        class_weight=class_weights,
    )

    # ── Step 4: Phase 2 — Fine-tune top layers ──────────────────────────────
    print(f"\n[4/5] Phase 2 — Fine-tuning top 30 {architecture.title()} layers …")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_p2 = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=7, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.3, patience=4, monitor="val_loss", verbose=1),
        CSVLogger(os.path.join(MODELS_DIR, "phase2_log.csv")),
    ]

    history_p2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=callbacks_p2,
        class_weight=class_weights,
    )

    # ── Step 5: Save artefacts ──────────────────────────────────────────────
    print("\n[5/5] Saving model and training history …")
    model.save(MODEL_PATH)

    # Merge phase histories
    combined_history = {
        "accuracy":     history_p1.history["accuracy"]     + history_p2.history["accuracy"],
        "val_accuracy": history_p1.history["val_accuracy"] + history_p2.history["val_accuracy"],
        "loss":         history_p1.history["loss"]         + history_p2.history["loss"],
        "val_loss":     history_p1.history["val_loss"]     + history_p2.history["val_loss"],
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(combined_history, f)

    # Save class label mapping
    labels = {v: k for k, v in train_gen.class_indices.items()}
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f)

    save_metadata({
        "architecture": architecture.lower(),
        "input_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "model_name": model.name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    print(f"\n✅ Model saved  → {MODEL_PATH}")
    print(f"✅ History saved → {HISTORY_PATH}")
    print(f"✅ Labels saved  → {LABELS_PATH}")
    print(f"✅ Metadata saved → {METADATA_PATH}")
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Alzheimer MRI classification model")
    parser.add_argument(
        "--architecture",
        choices=["mobilenetv2", "efficientnet"],
        default="mobilenetv2",
        help="Backbone architecture for transfer learning",
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Path to the dataset directory containing class subfolders",
    )
    parser.add_argument(
        "--epochs-head",
        type=int,
        default=EPOCHS_HEAD,
        help="Number of epochs to train the model head",
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=EPOCHS_FINETUNE,
        help="Number of epochs to fine-tune the backbone",
    )
    args = parser.parse_args()

    EPOCHS_HEAD = args.epochs_head
    EPOCHS_FINETUNE = args.epochs_finetune
    DATA_DIR = args.data_dir

    train(architecture=args.architecture)
