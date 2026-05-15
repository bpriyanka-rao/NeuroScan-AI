"""
=============================================================================
 Alzheimer's Disease Detection — Image Preprocessing Pipeline
 File: src/preprocess.py
 Author: [Your Name] | Final Year ML Project
=============================================================================
 This module handles:
  - Loading images from directory folders
  - Resizing to MobileNetV2 input size (224x224)
  - Normalization (pixel values 0→1)
  - One-hot label encoding
  - Data augmentation for training robustness
=============================================================================
"""

import os
import numpy as np
from PIL import Image
# TF is imported lazily inside functions — app loads fine without TensorFlow installed

# ── Constants ──────────────────────────────────────────────────────────────
IMG_SIZE       = 224          # Standard input size for MobileNetV2 / EfficientNetB0
BATCH_SIZE     = 32
SEED           = 42

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]

MODEL_BACKBONES = ["mobilenetv2", "efficientnet"]


def _get_preprocess_function(model_name: str):
    import tensorflow as tf

    normalized_name = model_name.lower()
    if normalized_name == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if normalized_name == "efficientnet":
        return tf.keras.applications.efficientnet.preprocess_input
    return tf.keras.applications.mobilenet_v2.preprocess_input


# ── Data Generators ─────────────────────────────────────────────────────────

def get_data_generators(data_dir: str, val_split: float = 0.20, model_name: str = "mobilenetv2"):
    """
    Create train & validation ImageDataGenerators with augmentation.

    Args:
        data_dir  : Root folder containing one sub-folder per class.
        val_split : Fraction of data reserved for validation (default 20%).
        model_name: Backbone name used for preprocessing.

    Returns:
        train_gen, val_gen — Keras DirectoryIterator objects ready for fit().
    """

    # Lazy import — only needed during training
    import tensorflow as tf
    try:
        from keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

    preprocess_fn = _get_preprocess_function(model_name)

    # ── Training generator WITH augmentation ───────────────────────────────
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=val_split,
    )

    # ── Validation generator — NO augmentation ─────────────────────────────
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        validation_split=val_split,
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=SEED,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
        shuffle=False,
    )

    return train_gen, val_gen


def preprocess_single_image(image_path: str, model_name: str = "mobilenetv2") -> np.ndarray:
    """
    Load and preprocess a single MRI image for inference.

    Args:
        image_path : Absolute path to the image file.
        model_name : Backbone name used for preprocessing.

    Returns:
        NumPy array of shape (1, 224, 224, 3) ready for model.predict().
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    preprocess_fn = _get_preprocess_function(model_name)
    img_array = preprocess_fn(img_array)
    img_array = np.expand_dims(img_array, axis=0)   # add batch dimension
    return img_array


if __name__ == "__main__":
    print("[preprocess.py] Class mapping:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {i} → {cls}")
