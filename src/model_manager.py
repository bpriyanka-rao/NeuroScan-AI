"""
=============================================================================
Model Manager — central model metadata, loading, and artifact utilities
File: src/model_manager.py
Author: Project Enhancement
=============================================================================
Provides a single source of truth for the saved model, labels, and training
metadata used throughout the application and CLI training pipeline.
"""

import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")

MODEL_PATH = os.path.join(MODELS_DIR, "alzheimer_model.h5")
LABELS_PATH = os.path.join(MODELS_DIR, "class_labels.json")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")
EVAL_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]

DEFAULT_METADATA = {
    "architecture": "mobilenetv2",
    "input_size": 224,
    "num_classes": len(CLASS_NAMES),
    "model_name": "MobileNetV2",
    "created_at": None,
}

_model = None
_labels = None
_metadata = None


def ensure_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)


def _load_metadata() -> dict:
    global _metadata
    if _metadata is None:
        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    _metadata = json.load(f)
            except Exception:
                _metadata = DEFAULT_METADATA.copy()
        else:
            _metadata = DEFAULT_METADATA.copy()
    return _metadata


def save_metadata(metadata: dict) -> None:
    ensure_directories()
    merged = _load_metadata().copy()
    merged.update(metadata)
    merged["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    global _metadata
    _metadata = merged


def load_labels() -> dict:
    global _labels
    if _labels is None:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            _labels = {int(k): v for k, v in raw.items()}
        else:
            _labels = {i: c for i, c in enumerate(CLASS_NAMES)}
    return _labels


def load_model():
    global _model
    ensure_directories()
    if _model is None:
        if not model_exists():
            _model = "DEMO"
        else:
            try:
                import tensorflow as tf
                _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            except Exception:
                _model = "DEMO"
    load_labels()
    _load_metadata()
    return _model


def get_model_info() -> dict:
    metadata = _load_metadata()
    return {
        "has_model": model_exists(),
        "mode": "real" if model_exists() and _model not in (None, "DEMO") else "demo",
        "model_name": metadata.get("model_name", "MobileNetV2"),
        "architecture": metadata.get("architecture", "mobilenetv2"),
        "input_size": metadata.get("input_size", 224),
        "num_classes": metadata.get("num_classes", len(CLASS_NAMES)),
        "model_path": MODEL_PATH,
        "labels_path": LABELS_PATH,
        "metadata_path": METADATA_PATH,
        "created_at": metadata.get("created_at"),
        "updated_at": metadata.get("updated_at"),
        "classes": CLASS_NAMES,
    }


def get_training_history() -> dict:
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_evaluation_results() -> dict:
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    ensure_directories()
    print("[model_manager] model_exists:", model_exists())
    print("[model_manager] metadata:", _load_metadata())
    print("[model_manager] labels:", load_labels())
