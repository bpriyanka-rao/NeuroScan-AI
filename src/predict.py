"""
=============================================================================
 Alzheimer's Disease Detection — Inference / Prediction Module
 File: src/predict.py
 Author: Pratikshya Gopal   | Final Year ML Project
=============================================================================
 Usage:
    from predict import predict_image
    result = predict_image("path/to/mri_scan.jpg")
    # result = {
    #   "class":      "MildDemented",
    #   "confidence": 0.8723,
    #   "risk_level": "Moderate Risk",
    #   "risk_color": "#F97316",
    #   "all_probs":  {"MildDemented": 0.87, ...}
    # }
=============================================================================
"""

import json
import random
import numpy as np
from preprocess import preprocess_single_image
from model_manager import load_model, load_labels, _load_metadata, get_model_info

# ── Risk level mapping ────────────────────────────────────────────────────────
RISK_MAP = {
    "NonDemented":      ("No Risk Detected",    "#22C55E", "low"),
    "VeryMildDemented": ("Very Low Risk",        "#84CC16", "very_low"),
    "MildDemented":     ("Moderate Risk",        "#F97316", "moderate"),
    "ModerateDemented": ("High Risk Detected",   "#EF4444", "high"),
}


def predict_image(image_path: str) -> dict:
    """
    Run inference on a single MRI image.

    Args:
        image_path : Path to the uploaded MRI image (jpg/png/jpeg).

    Returns:
        dict with keys:
            class       — Predicted class name (str)
            confidence  — Prediction confidence 0–1 (float)
            risk_level  — Human-readable risk label (str)
            risk_color  — Hex color for the badge (str)
            risk_tag    — Short tag for CSS class (str)
            all_probs   — {class_name: probability} for all 4 classes
    """
    model = load_model()
    metadata = _load_metadata()
    labels = load_labels()
    model_info = get_model_info()

    if model == "DEMO":
        result = _demo_prediction()
        result["model_mode"] = "demo"
        result["model_architecture"] = model_info["architecture"]
        result["model_name"] = model_info["model_name"]
        return result

    img_array = preprocess_single_image(image_path, metadata.get("architecture", "mobilenetv2"))
    proba = model.predict(img_array, verbose=0)[0]

    pred_idx = int(np.argmax(proba))
    pred_class = labels.get(pred_idx, "NonDemented")
    confidence = float(proba[pred_idx])

    risk_level, risk_color, risk_tag = RISK_MAP.get(
        pred_class, ("Unknown", "#6B7280", "unknown")
    )
    all_probs = {labels.get(i, f"class_{i}"): float(proba[i]) for i in range(len(proba))}

    return {
        "class":      pred_class,
        "confidence": round(confidence * 100, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_tag":   risk_tag,
        "all_probs":  {k: round(v * 100, 2) for k, v in all_probs.items()},
        "demo_mode":  False,
        "model_mode": "real",
        "model_architecture": metadata.get("architecture", "mobilenetv2"),
        "model_name": metadata.get("model_name", "MobileNetV2"),
    }


def _demo_prediction() -> dict:
    """Return a realistic-looking demo result when model isn't trained yet."""
    classes = list(RISK_MAP.keys())
    weights = [0.40, 0.10, 0.30, 0.20]   # NonDemented most likely in demo
    chosen  = random.choices(classes, weights=weights, k=1)[0]

    base_conf = random.uniform(0.72, 0.95)
    remainder = 1.0 - base_conf
    others    = [random.random() for _ in range(3)]
    s         = sum(others)
    others    = [remainder * o / s for o in others]

    other_classes = [c for c in classes if c != chosen]
    probs = {chosen: round(base_conf * 100, 2)}
    for cls, p in zip(other_classes, others):
        probs[cls] = round(p * 100, 2)

    risk_level, risk_color, risk_tag = RISK_MAP[chosen]
    return {
        "class":      chosen,
        "confidence": round(base_conf * 100, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_tag":   risk_tag,
        "all_probs":  probs,
        "demo_mode":  True,
    }


if __name__ == "__main__":
    # Quick smoke-test
    result = predict_image("dummy_path.jpg")
    print(json.dumps(result, indent=2))
