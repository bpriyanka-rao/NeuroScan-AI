"""
=============================================================================
 Explainable AI-Based Alzheimer's Disease Detection and Clinical Insight System
 File: app/app.py
 Author: Pratikshya Gopal | Healthcare AI System
=============================================================================
 Routes:
   GET  /             → Home / Upload page
   POST /predict      → Run inference, Grad-CAM, clinical insights → JSON
   GET  /result       → Full result display (MRI + heatmap + insights)
   GET  /dashboard    → Model performance dashboard
   GET  /about        → Alzheimer's educational content
   GET  /report       → PDF report download
   GET  /api/metrics  → JSON: live model metrics
   GET  /api/history  → JSON: training history
=============================================================================
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()
import json
import uuid
import time

from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session, send_from_directory,
    send_file, make_response
)
from werkzeug.utils import secure_filename
import io

# ── Add src/ to path so we can import ML modules ─────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict_image                              # noqa: E402
from gradcam import generate_gradcam                          # noqa: E402
from clinical_insights import get_clinical_insights           # noqa: E402
from pdf_report import generate_pdf_report                    # noqa: E402
from chatbot import get_chatbot_response                      # noqa: E402
from email_service import init_mail, send_report_email        # noqa: E402

# ── App config ────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR   = os.path.join(BASE_DIR, "templates")
STATIC_DIR     = os.path.join(BASE_DIR, "static")
UPLOAD_FOLDER  = os.path.join(STATIC_DIR, "uploads")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
ALLOWED_EXT    = {"jpg", "jpeg", "png", "bmp", "tiff"}
MAX_MB         = 16

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
)
app.secret_key = "alzheimer-xai-healthcare-2025"
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Initialize Mail
init_mail(app)


# ════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _load_metrics() -> dict:
    """Load pre-computed evaluation metrics (or return demo values)."""
    path = os.path.join(MODELS_DIR, "evaluation_results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "accuracy": 0.9412,
        "macro_f1": 0.9387,
        "roc_auc":  0.9831,
        "per_class": {
            "MildDemented":     {"precision": 0.93, "recall": 0.91, "f1_score": 0.92, "support": 179},
            "ModerateDemented": {"precision": 0.96, "recall": 0.93, "f1_score": 0.94, "support": 12},
            "NonDemented":      {"precision": 0.97, "recall": 0.98, "f1_score": 0.97, "support": 640},
            "VeryMildDemented": {"precision": 0.92, "recall": 0.93, "f1_score": 0.92, "support": 448},
        },
        "confusion_matrix": [
            [163, 5, 2, 9],
            [1, 11, 0, 0],
            [3, 0, 627, 10],
            [8, 0, 23, 417]
        ],
        "class_order": ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"],
        "demo": True,
    }


def _load_history() -> dict:
    """Load training history (or return demo values)."""
    path = os.path.join(MODELS_DIR, "training_history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    import random
    random.seed(42)
    epochs = 25
    acc      = [min(0.50 + 0.018 * i + random.uniform(-0.01, 0.01), 0.98) for i in range(epochs)]
    val_acc  = [min(0.48 + 0.017 * i + random.uniform(-0.015, 0.015), 0.95) for i in range(epochs)]
    loss     = [max(1.40 - 0.04 * i + random.uniform(-0.02, 0.02), 0.08) for i in range(epochs)]
    val_loss = [max(1.45 - 0.038 * i + random.uniform(-0.025, 0.025), 0.12) for i in range(epochs)]
    return {"accuracy": acc, "val_accuracy": val_acc,
            "loss": loss, "val_loss": val_loss, "demo": True}


# ════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Home page — hero section + MRI upload interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts a multipart/form-data file upload.
    Runs inference, Grad-CAM, and clinical insights.
    Returns JSON with full result.
    """
    if "mri_image" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["mri_image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not _allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use JPG, PNG, or BMP."}), 400

    # ── Save uploaded file ──────────────────────────────────────────────────
    ext      = file.filename.rsplit(".", 1)[1].lower()
    uid      = uuid.uuid4().hex[:10]
    filename = f"scan_{uid}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ── Step 1: Run inference ───────────────────────────────────────────────
    try:
        start  = time.time()
        result = predict_image(filepath)
        elapsed = round((time.time() - start) * 1000, 1)
        result["inference_ms"] = elapsed
        result["image_url"]    = f"/static/uploads/{filename}"
        result["filename"]     = filename
        result["filepath"]     = filepath
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # ── Step 2: Generate Grad-CAM heatmap ───────────────────────────────────
    try:
        # Pass model=None → auto demo mode if no real model
        heatmap_b64 = generate_gradcam(filepath, model=None)
        if heatmap_b64:
            import base64
            heatmap_data = base64.b64decode(heatmap_b64)
            heatmap_filename = filename.replace(".", "_heatmap.")
            heatmap_filepath = os.path.join(app.config["UPLOAD_FOLDER"], heatmap_filename)
            with open(heatmap_filepath, "wb") as f:
                f.write(heatmap_data)
            result["heatmap_url"] = f"/static/uploads/{heatmap_filename}"
            result["heatmap_b64"] = None
    except Exception as e:
        result["heatmap_url"] = None
        result["heatmap_error"] = str(e)

    # ── Step 3: Generate AI clinical insights ───────────────────────────────
    try:
        insights = get_clinical_insights(result["class"], result["confidence"])
        result["insights"] = insights
    except Exception as e:
        result["insights"] = {}
        result["insights_error"] = str(e)

    # ── Store in session for result page and history ────────────────────────
    session["last_result"] = result
    
    # Store in history
    if "history" not in session:
        session["history"] = []
        
    history = session["history"]
    history.insert(0, {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": result["class"],
        "confidence": result["confidence"],
        "filename": filename,
        "image_url": result["image_url"],
        "heatmap_url": result.get("heatmap_url")
    })
    session["history"] = history
    session.modified = True
    
    return jsonify(result)


@app.route("/result")
def result():
    """Result display page — reads prediction + insights from session."""
    pred = session.get("last_result")
    if not pred:
        return redirect(url_for("index"))
    return render_template("result.html", result=pred)


@app.route("/dashboard")
def dashboard():
    """Model performance dashboard with metrics, confusion matrix, training curves."""
    metrics = _load_metrics()
    history = _load_history()
    return render_template("dashboard.html", metrics=metrics, history=history)


@app.route("/about")
def about():
    """Alzheimer's educational information page."""
    return render_template("about.html")


@app.route("/report", methods=["GET", "POST"])
def report():
    """
    GET/POST /report
    Generates and returns a PDF medical report.
    Patient name is passed as a query param or form field.
    """
    pred = session.get("last_result")
    if not pred:
        return redirect(url_for("index"))

    patient_name = (
        request.form.get("patient_name")
        or request.args.get("patient_name")
        or "Anonymous Patient"
    )

    # Get image path
    image_path = pred.get("filepath", None)

    # Get insights (re-generate if not in session)
    insights = pred.get("insights") or get_clinical_insights(
        pred.get("class", "NonDemented"), pred.get("confidence", 0)
    )

    try:
        pdf_bytes = generate_pdf_report(
            result=pred,
            insights=insights,
            image_path=image_path,
            patient_name=patient_name
        )
        response = make_response(pdf_bytes)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = (
            f"attachment; filename=AlzheimerReport_{timestamp}.pdf"
        )
        return response
    except Exception as e:
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500


@app.route("/send_report", methods=["POST"])
def send_report():
    """Send the PDF report via email."""
    pred = session.get("last_result")
    if not pred:
        return jsonify({"success": False, "message": "No active result found."}), 400

    doctor_email = request.form.get("doctor_email")
    patient_name = request.form.get("patient_name") or "Anonymous Patient"

    if not doctor_email:
        return jsonify({"success": False, "message": "Doctor's email is required."}), 400

    # Generate PDF
    image_path = pred.get("filepath", None)
    insights = pred.get("insights") or get_clinical_insights(
        pred.get("class", "NonDemented"), pred.get("confidence", 0)
    )

    try:
        pdf_bytes = generate_pdf_report(
            result=pred,
            insights=insights,
            image_path=image_path,
            patient_name=patient_name
        )
        
        success, msg = send_report_email(
            to_email=doctor_email,
            patient_name=patient_name,
            stage=pred.get("class"),
            confidence=pred.get("confidence"),
            pdf_bytes=pdf_bytes
        )
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": f"Failed: {str(e)}"}), 500


@app.route("/api/metrics")
def api_metrics():
    """JSON endpoint — live model metrics for charts."""
    return jsonify(_load_metrics())


@app.route("/api/history")
def api_history():
    """JSON endpoint — training history for charts."""
    return jsonify(_load_history())


@app.route("/history")
def scan_history():
    """Display session history of all scans."""
    history_data = session.get("history", [])
    return render_template("history.html", history=history_data)


@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear the session history."""
    session.pop("history", None)
    session.pop("last_result", None)
    return redirect(url_for("scan_history"))


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chatbot API endpoint."""
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"]
    
    # Get scan context if available
    scan_context = None
    last_result = session.get("last_result")
    if last_result and "class" in last_result:
        stage = last_result["class"]
        conf = last_result.get("confidence", 0)
        scan_context = f"The current scan result is: {stage} with {conf}% confidence."

    # Call Gemini
    reply = get_chatbot_response(user_message, scan_context)
    return jsonify({"reply": reply})


@app.route("/compare", methods=["GET", "POST"])
def compare():
    """Compare two MRI scans."""
    if request.method == "GET":
        return render_template("compare.html")
        
    if "scan1" not in request.files or "scan2" not in request.files:
        return "Missing files", 400
        
    f1 = request.files["scan1"]
    f2 = request.files["scan2"]
    
    if f1.filename == "" or f2.filename == "":
        return "No selected files", 400
        
    # Save
    uid1 = uuid.uuid4().hex[:6]
    uid2 = uuid.uuid4().hex[:6]
    ext1 = f1.filename.rsplit(".", 1)[-1].lower()
    ext2 = f2.filename.rsplit(".", 1)[-1].lower()
    
    p1 = os.path.join(app.config["UPLOAD_FOLDER"], f"comp1_{uid1}.{ext1}")
    p2 = os.path.join(app.config["UPLOAD_FOLDER"], f"comp2_{uid2}.{ext2}")
    f1.save(p1)
    f2.save(p2)
    
    # Predict
    res1 = predict_image(p1)
    res2 = predict_image(p2)
    
    res1["image_url"] = f"/static/uploads/comp1_{uid1}.{ext1}"
    res2["image_url"] = f"/static/uploads/comp2_{uid2}.{ext2}"
    
    try:
        res1["heatmap_b64"] = generate_gradcam(p1, model=None)
    except:
        res1["heatmap_b64"] = None
        
    try:
        res2["heatmap_b64"] = generate_gradcam(p2, model=None)
    except:
        res2["heatmap_b64"] = None
        
    severity_map = {"NonDemented": 0, "VeryMildDemented": 1, "MildDemented": 2, "ModerateDemented": 3}
    s1 = severity_map.get(res1["class"], 0)
    s2 = severity_map.get(res2["class"], 0)
    
    if s2 > s1:
        progression = "worsened"
    elif s2 < s1:
        progression = "improved"
    else:
        progression = "stable"
        
    return render_template("compare.html", res1=res1, res2=res2, progression_status=progression)


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  Explainable AI-Based Alzheimer's Disease Detection System")
    print("  URL: http://localhost:5000")
    print("=" * 70)
    app.run(debug=True, host="0.0.0.0", port=5000)
