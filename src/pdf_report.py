"""
=============================================================================
 PDF Medical Report Generator
 File: src/pdf_report.py
 Author: Pratikshya Gopal | Healthcare AI System
=============================================================================
 Generates a professional downloadable PDF medical report containing:
   - Patient information & timestamp
   - Uploaded MRI image thumbnail
   - AI prediction result & confidence score
   - Probability breakdown for all 4 classes
   - AI-generated clinical insights
   - Precautions & lifestyle recommendations
   - Disclaimer footer

 Uses fpdf2 (pure-Python, no external dependencies)
 Install: pip install fpdf2
=============================================================================
"""

import io
import os
import base64
from datetime import datetime

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


# ── Label mappings ─────────────────────────────────────────────────────────────
CLASS_LABELS = {
    "NonDemented":      "Non Demented",
    "VeryMildDemented": "Very Mild Demented",
    "MildDemented":     "Mild Demented",
    "ModerateDemented": "Moderate Demented",
}

RISK_LABELS = {
    "low":      "No Risk Detected",
    "moderate": "Very Low Risk",
    "high":     "Moderate Risk",
    "critical": "High Risk Detected",
}


# ==============================================================================
#  PUBLIC API
# ==============================================================================

def generate_pdf_report(result: dict, insights: dict, image_path: str = None,
                        patient_name: str = "Anonymous Patient") -> bytes:
    """
    Generate a professional PDF medical report.

    Args:
        result       : Prediction result dict from predict.py
        insights     : Clinical insights dict from clinical_insights.py
        image_path   : Optional path to uploaded MRI image
        patient_name : Patient name for the report header

    Returns:
        PDF file as bytes (ready for Flask send_file)
    """
    if not FPDF_AVAILABLE:
        return _fallback_pdf(result, patient_name)

    pdf = _MedicalReportPDF()
    pdf.alias_nb_pages()

    # ── Cover / Header Page ────────────────────────────────────────────────────
    pdf.add_page()
    _draw_header(pdf, patient_name, result)
    _draw_mri_and_result(pdf, result, image_path)
    _draw_probability_breakdown(pdf, result)
    _draw_clinical_insights(pdf, insights)
    _draw_precautions(pdf, insights)
    _draw_disclaimer(pdf)

    return bytes(pdf.output())


# ==============================================================================
#  PDF CLASS
# ==============================================================================

class _MedicalReportPDF(FPDF):
    """Custom FPDF subclass with header and footer."""

    # ── Page Header ────────────────────────────────────────────────────────────
    def header(self):
        # Teal top bar
        self.set_fill_color(0, 168, 150)      # #00A896
        self.rect(0, 0, 210, 14, 'F')
        # System name
        self.set_xy(10, 2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "NeuroScan AI  |  Explainable Alzheimer's Detection System", ln=False)
        # Page number on right
        self.set_xy(170, 2)
        self.set_font("Helvetica", "", 8)
        self.cell(30, 10, f"Page {self.page_no()} / {{nb}}", align="R")
        self.set_text_color(20, 20, 20)
        self.ln(12)

    # ── Page Footer ────────────────────────────────────────────────────────────
    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10,
                  "DISCLAIMER: For educational and research purposes only. "
                  "Not a substitute for professional medical diagnosis.",
                  align="C")
        self.set_text_color(20, 20, 20)


# ==============================================================================
#  SECTION DRAWERS
# ==============================================================================

def _draw_header(pdf: _MedicalReportPDF, patient_name: str, result: dict):
    """Draw the report title block and patient metadata."""
    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 80, 80)
    pdf.cell(0, 10, "AI-Assisted MRI Analysis Report", ln=True, align="C")
    pdf.ln(2)

    # Subtitle
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6,
             "Explainable AI-Based Alzheimer's Disease Detection and Clinical Insight System",
             ln=True, align="C")
    pdf.ln(5)

    # Horizontal rule
    pdf.set_draw_color(0, 168, 150)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # Patient info block — 2 columns
    col_w = 90
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.set_fill_color(240, 250, 250)
    pdf.rect(10, pdf.get_y(), 190, 22, 'F')

    pdf.set_xy(14, pdf.get_y() + 3)
    pdf.cell(col_w, 5, f"Patient Name:  {patient_name}", ln=False)
    pdf.cell(col_w, 5, f"Report Date:  {datetime.now().strftime('%d %B %Y')}", ln=True)

    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(col_w, 5, f"Analysis Time: {datetime.now().strftime('%H:%M:%S IST')}", ln=False)
    pdf.cell(col_w, 5, f"Analysis Mode: AI Deep Learning (MobileNetV2)", ln=True)
    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, "REPORT ID: " + datetime.now().strftime("NSA-%Y%m%d-%H%M%S"), ln=True)

    pdf.ln(6)
    pdf.set_text_color(20, 20, 20)


def _draw_mri_and_result(pdf: _MedicalReportPDF, result: dict, image_path: str):
    """Draw MRI thumbnail and prediction result side by side."""
    # Section title
    _section_title(pdf, "[AI]  Prediction Result")

    start_y = pdf.get_y()

    # ── Left: MRI Image ──────────────────────────────────────────────────────
    if image_path and os.path.exists(image_path):
        try:
            pdf.image(image_path, x=10, y=start_y, w=70, h=70)
            pdf.set_xy(10, start_y + 72)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(70, 4, "Uploaded MRI Brain Scan", align="C")
        except Exception:
            _placeholder_box(pdf, 10, start_y, 70, 70, "MRI Image\n(Preview unavailable)")
    else:
        _placeholder_box(pdf, 10, start_y, 70, 70, "MRI Image\n(Not provided)")

    # ── Right: Prediction Result ─────────────────────────────────────────────
    pred_class = CLASS_LABELS.get(result.get("class", ""), result.get("class", "Unknown"))
    confidence = result.get("confidence", 0)
    risk_tag   = result.get("risk_tag", "low")
    risk_label = result.get("risk_level", "Unknown")

    # Risk color
    risk_colors = {
        "low":      (34, 197, 94),
        "very_low": (132, 204, 22),
        "moderate": (249, 115, 22),
        "high":     (239, 68, 68),
        "critical": (220, 38, 38),
    }
    r, g, b = risk_colors.get(risk_tag, (100, 100, 100))

    rx = 88   # Right column X
    pdf.set_xy(rx, start_y)

    # Prediction label
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(0, 80, 80)
    pdf.cell(115, 8, f"Prediction: {pred_class}", ln=True)
    pdf.set_xy(rx, pdf.get_y())

    # Risk badge
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(50, 7, f"  {risk_label}", fill=True, border=0)
    pdf.ln(10)

    # Confidence score
    pdf.set_xy(rx, pdf.get_y())
    pdf.set_text_color(60, 60, 60)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, f"Confidence Score: {confidence}%", ln=True)

    # Progress bar
    bar_x, bar_y = rx, pdf.get_y()
    bar_w = 112
    pdf.set_fill_color(220, 220, 220)
    pdf.rect(bar_x, bar_y, bar_w, 6, 'F')
    pdf.set_fill_color(r, g, b)
    fill_w = min(bar_w * confidence / 100, bar_w)
    pdf.rect(bar_x, bar_y, fill_w, 6, 'F')
    pdf.ln(12)

    # Model info
    pdf.set_xy(rx, pdf.get_y())
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Model: MobileNetV2  |  Input: 224x224px  |  Classes: 4", ln=True)

    # Demo mode notice
    if result.get("demo_mode"):
        pdf.set_xy(rx, pdf.get_y() + 2)
        pdf.set_fill_color(254, 243, 199)
        pdf.set_text_color(146, 64, 14)
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(112, 7, "  [!] Demo Mode - Train model for real predictions", fill=True)

    pdf.set_y(max(start_y + 82, pdf.get_y() + 5))
    pdf.set_text_color(20, 20, 20)
    pdf.ln(3)


def _draw_probability_breakdown(pdf: _MedicalReportPDF, result: dict):
    """Draw probability breakdown for all 4 classes."""
    _section_title(pdf, "[DATA]  Probability Breakdown")

    all_probs = result.get("all_probs", {})
    label_map = {
        "NonDemented": "Non Demented",
        "VeryMildDemented": "Very Mild Demented",
        "MildDemented": "Mild Demented",
        "ModerateDemented": "Moderate Demented",
    }
    bar_colors = {
        "NonDemented":      (34, 197, 94),
        "VeryMildDemented": (132, 204, 22),
        "MildDemented":     (249, 115, 22),
        "ModerateDemented": (239, 68, 68),
    }

    for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        label = label_map.get(cls, cls)
        r, g, b = bar_colors.get(cls, (100, 100, 100))

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(60, 6, label, ln=False)

        bar_x = pdf.get_x()
        bar_y = pdf.get_y() + 1
        bar_w = 100
        pdf.set_fill_color(220, 220, 220)
        pdf.rect(bar_x, bar_y, bar_w, 4, 'F')
        pdf.set_fill_color(r, g, b)
        pdf.rect(bar_x, bar_y, max(bar_w * prob / 100, 0.5), 4, 'F')

        pdf.set_xy(bar_x + bar_w + 3, pdf.get_y())
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(20, 6, f"{prob}%", ln=True)
        pdf.ln(1)

    pdf.ln(4)
    pdf.set_text_color(20, 20, 20)


def _draw_clinical_insights(pdf: _MedicalReportPDF, insights: dict):
    """Draw the AI clinical insights section."""
    _section_title(pdf, "[MEDICAL]  AI Clinical Insights")

    # Disease explanation
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(0, 100, 100)
    pdf.cell(0, 6, "Stage Overview & Clinical Explanation", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    explanation = insights.get("disease_explanation", "")
    pdf.multi_cell(0, 5, _sanitize(explanation))
    pdf.ln(4)

    # Symptoms
    _bullet_list(pdf, "Key Symptoms", insights.get("symptoms", []))
    _bullet_list(pdf, "Cognitive Care Recommendations", insights.get("cognitive_care", []))


def _draw_precautions(pdf: _MedicalReportPDF, insights: dict):
    """Draw precautions, lifestyle and doctor advice."""
    _bullet_list(pdf, "Precautions", insights.get("precautions", []))
    _bullet_list(pdf, "Lifestyle Recommendations", insights.get("lifestyle", []))

    # Doctor advice
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(0, 100, 100)
    pdf.cell(0, 6, "Doctor Consultation Advice", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 5, _sanitize(insights.get("doctor_advice", "")))
    pdf.ln(4)

    # Follow-up
    pdf.set_fill_color(240, 250, 250)
    pdf.set_draw_color(0, 168, 150)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(0, 80, 80)
    y_before = pdf.get_y()
    pdf.rect(10, y_before, 190, 12, 'FD')
    pdf.set_xy(14, y_before + 2)
    pdf.cell(0, 8, _sanitize(f"Recommended Follow-up: {insights.get('follow_up', 'As advised by physician')}"))
    pdf.ln(16)
    pdf.set_text_color(20, 20, 20)


def _draw_disclaimer(pdf: _MedicalReportPDF):
    """Draw the final disclaimer box."""
    pdf.set_fill_color(255, 243, 220)
    pdf.set_draw_color(245, 158, 11)
    y = pdf.get_y()
    pdf.rect(10, y, 190, 22, 'FD')
    pdf.set_xy(14, y + 3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(146, 64, 14)
    pdf.cell(0, 5, "[!] Important Disclaimer", ln=True)
    pdf.set_xy(14, pdf.get_y())
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(182, 4,
        "This report was generated by an AI system for educational and research purposes only. "
        "It does NOT constitute a medical diagnosis. All findings must be reviewed and confirmed "
        "by a qualified medical professional. Never make treatment decisions based solely on this report.")
    pdf.set_text_color(20, 20, 20)
    pdf.ln(6)


# ==============================================================================
#  HELPER UTILITIES
# ==============================================================================

def _sanitize(text: str) -> str:
    """Strip non-latin-1 characters so fpdf2 Helvetica font renders without errors."""
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u00d7": "x",  # multiplication sign
        "\u00e2": "a",
        "\u2022": "-",   # bullet
        "\u26a0": "[!]", # warning sign
        "\u00e9": "e",
        "\u00e8": "e",
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
    # Final fallback: encode to latin-1, replacing remaining unknowns
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _section_title(pdf: _MedicalReportPDF, title: str):
    """Draw a styled section heading."""
    pdf.set_fill_color(0, 168, 150)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, f"  {_sanitize(title)}", fill=True, ln=True)
    pdf.ln(3)
    pdf.set_text_color(20, 20, 20)


def _bullet_list(pdf: _MedicalReportPDF, title: str, items: list):
    """Draw a titled bullet list."""
    if not items:
        return
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(0, 100, 100)
    pdf.cell(0, 6, _sanitize(title), ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    for item in items:
        pdf.set_x(16)
        pdf.cell(4, 5, "-", ln=False)
        pdf.multi_cell(180, 5, _sanitize(str(item)))
    pdf.ln(3)


def _placeholder_box(pdf: _MedicalReportPDF, x, y, w, h, text):
    """Draw a grey placeholder box."""
    pdf.set_fill_color(200, 200, 200)
    pdf.rect(x, y, w, h, 'F')
    pdf.set_xy(x, y + h / 2 - 5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(w, 5, text, align="C")
    pdf.set_text_color(20, 20, 20)


def _fallback_pdf(result: dict, patient_name: str) -> bytes:
    """Return a minimal PDF-like bytes object if fpdf2 is not installed."""
    content = (
        f"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        f"% fpdf2 not installed — install with: pip install fpdf2\n"
        f"% Patient: {patient_name}\n"
        f"% Result: {result.get('class', 'N/A')} ({result.get('confidence', 0)}%)\n"
    )
    return content.encode("utf-8")


# ==============================================================================
#  SMOKE TEST
# ==============================================================================

if __name__ == "__main__":
    if FPDF_AVAILABLE:
        dummy_result = {
            "class": "MildDemented", "confidence": 87.5,
            "risk_level": "Moderate Risk", "risk_tag": "high",
            "risk_color": "#F97316",
            "all_probs": {"NonDemented": 5.2, "VeryMildDemented": 3.1,
                          "MildDemented": 87.5, "ModerateDemented": 4.2},
            "demo_mode": True
        }
        dummy_insights = {
            "disease_explanation": "Test explanation for mild demented stage.",
            "symptoms": ["Memory loss", "Confusion"],
            "precautions": ["See a doctor", "Monitor progress"],
            "cognitive_care": ["Brain exercises", "Social engagement"],
            "lifestyle": ["MIND diet", "Regular exercise"],
            "doctor_advice": "Consult a neurologist immediately.",
            "follow_up": "Every 3-6 months",
        }
        pdf_bytes = generate_pdf_report(dummy_result, dummy_insights, patient_name="Test Patient")
        with open("test_report.pdf", "wb") as f:
            f.write(pdf_bytes)
        print(f"[pdf_report.py] Test PDF generated: {len(pdf_bytes)} bytes")
    else:
        print("[pdf_report.py] fpdf2 not installed. Run: pip install fpdf2")
