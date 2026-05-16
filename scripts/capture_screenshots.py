from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"
UPLOADS = ROOT / "static" / "uploads"
MRI = UPLOADS / "scan_cf5afccdfc.png"
HEATMAP = UPLOADS / "scan_cf5afccdfc_heatmap.png"

BG = "#08111f"
PANEL = "#101b2e"
BORDER = "#21324f"
TEXT = "#eef6ff"
MUTED = "#91a4bd"
TEAL = "#00c9a7"
BLUE = "#3b82f6"
ORANGE = "#f97316"
GREEN = "#22c55e"
RED = "#ef4444"


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


F12 = font(12)
F14 = font(14)
F16 = font(16)
F18 = font(18, True)
F22 = font(22, True)
F30 = font(30, True)
F44 = font(44, True)
F64 = font(64, True)


def rounded(draw, box, fill, outline=BORDER, radius=18, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, value, fill=TEXT, fnt=F16, anchor=None):
    draw.text(xy, value, fill=fill, font=fnt, anchor=anchor)


def multiline(draw, xy, value, width, fill=MUTED, fnt=F16, line_gap=8):
    x, y = xy
    for line in wrap(value, width):
        text(draw, (x, y), line, fill, fnt)
        y += fnt.size + line_gap
    return y


def paste_contain(canvas, image_path, box):
    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = box
    image.thumbnail((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
    x = x1 + (x2 - x1 - image.width) // 2
    y = y1 + (y2 - y1 - image.height) // 2
    canvas.paste(image, (x, y))


def sidebar(draw):
    rounded(draw, (30, 30, 250, 970), "#0d1728", "#182842", 24)
    text(draw, (60, 74), "NeuroScan AI", TEAL, F22)
    text(draw, (60, 112), "Alzheimer MRI XAI", MUTED, F14)
    items = ["MRI Analysis", "Dashboard", "Reports", "History", "About"]
    y = 180
    for index, item in enumerate(items):
        fill = "#13273c" if index == 0 else "#0d1728"
        outline = TEAL if index == 0 else "#172840"
        rounded(draw, (52, y, 228, y + 48), fill, outline, 12)
        text(draw, (76, y + 15), item, TEXT if index == 0 else MUTED, F16)
        y += 64
    text(draw, (60, 900), "Research demo", MUTED, F12)
    text(draw, (60, 922), "For education only", MUTED, F12)


def shell(title):
    img = Image.new("RGB", (1440, 1000), BG)
    draw = ImageDraw.Draw(img)
    sidebar(draw)
    text(draw, (300, 58), title, TEXT, F30)
    text(draw, (1220, 64), "Live Demo", TEAL, F16)
    return img, draw


def homepage():
    img, draw = shell("MRI Analysis & Upload")
    text(draw, (720, 145), "AI-Powered Neurological Analysis", TEAL, F18, "mm")
    text(draw, (720, 230), "Early Alzheimer's Detection", TEXT, F64, "mm")
    text(draw, (720, 300), "Powered by Explainable AI", TEAL, F44, "mm")
    multiline(
        draw,
        (450, 345),
        "Upload a brain MRI scan and receive an instant AI-assisted prediction with Grad-CAM heatmap visualization, clinical insights, and a downloadable PDF report.",
        72,
        MUTED,
        F18,
    )

    stats = [("94.1%", "Model Accuracy"), ("4", "Disease Stages"), ("98.3%", "ROC-AUC"), ("<2s", "Inference")]
    x = 315
    for value, label in stats:
        rounded(draw, (x, 445, x + 245, 560), PANEL, BORDER, 18)
        text(draw, (x + 122, 485), value, TEAL, F30, "mm")
        text(draw, (x + 122, 525), label, MUTED, F14, "mm")
        x += 270

    rounded(draw, (410, 610, 1030, 900), PANEL, BORDER, 24)
    text(draw, (455, 655), "Step 1 - Upload MRI Scan", TEAL, F16)
    text(draw, (455, 702), "Analyze Your Brain MRI", TEXT, F30)
    rounded(draw, (455, 750, 985, 850), "#0b1425", "#284567", 18)
    text(draw, (720, 790), "Drag & Drop MRI scan here", TEXT, F22, "mm")
    text(draw, (720, 825), "or click to browse files", MUTED, F16, "mm")
    rounded(draw, (560, 865, 880, 920), TEAL, TEAL, 16)
    text(draw, (720, 883), "Analyze MRI Scan", "#04111b", F18, "ma")
    img.save(OUT / "app-homepage.png")


def upload_prediction():
    img, draw = shell("Upload & Prediction Flow")
    rounded(draw, (320, 130, 760, 900), PANEL, BORDER, 24)
    text(draw, (360, 175), "Selected MRI Scan", TEAL, F18)
    rounded(draw, (360, 220, 720, 560), "#05080d", "#284567", 18)
    paste_contain(img, MRI, (380, 240, 700, 540))
    rounded(draw, (360, 590, 720, 645), "#0b2a30", TEAL, 14)
    text(draw, (382, 608), "MildDemented_0.png", TEAL, F16)
    rounded(draw, (420, 705, 660, 770), TEAL, TEAL, 18)
    text(draw, (540, 724), "Analyze MRI Scan", "#04111b", F18, "ma")

    rounded(draw, (805, 130, 1320, 900), PANEL, BORDER, 24)
    text(draw, (850, 175), "AI Diagnosis", TEAL, F18)
    text(draw, (850, 230), "Mild Demented", TEXT, F44)
    rounded(draw, (850, 300, 1045, 346), "#331e12", ORANGE, 20)
    text(draw, (947, 313), "Moderate Risk", ORANGE, F16, "ma")
    text(draw, (850, 420), "75.98%", TEAL, F64)
    text(draw, (850, 500), "Confidence Score", MUTED, F16)
    rounded(draw, (850, 535, 1260, 553), "#0b1425", "#0b1425", 9)
    rounded(draw, (850, 535, 1162, 553), ORANGE, ORANGE, 9)
    y = 630
    for label, value, color in [
        ("Mild Demented", "75.98%", ORANGE),
        ("Moderate Demented", "13.77%", RED),
        ("Non Demented", "6.62%", GREEN),
        ("Very Mild Demented", "3.63%", "#eab308"),
    ]:
        text(draw, (850, y), label, TEXT, F16)
        text(draw, (1260, y), value, color, F16, "ra")
        rounded(draw, (850, y + 28, 1260, y + 40), "#0b1425", "#0b1425", 6)
        fill = int(410 * float(value.strip("%")) / 100)
        rounded(draw, (850, y + 28, 850 + fill, y + 40), color, color, 6)
        y += 72
    img.save(OUT / "upload-prediction.png")


def gradcam_report():
    img, draw = shell("Grad-CAM & Medical Report")
    rounded(draw, (305, 120, 835, 555), PANEL, BORDER, 24)
    text(draw, (345, 165), "Grad-CAM Explainability", TEAL, F18)
    text(draw, (405, 215), "Original MRI", MUTED, F14, "ma")
    text(draw, (650, 215), "Grad-CAM Heatmap", TEAL, F14, "ma")
    rounded(draw, (345, 245, 565, 465), "#05080d", "#284567", 18)
    rounded(draw, (575, 245, 795, 465), "#05080d", "#284567", 18)
    paste_contain(img, MRI, (365, 265, 545, 445))
    paste_contain(img, HEATMAP, (595, 265, 775, 445))
    multiline(
        draw,
        (345, 500),
        "Highlighted regions indicate areas most influential in the model prediction. Warmer colors represent higher activation.",
        58,
        MUTED,
        F14,
    )

    rounded(draw, (875, 120, 1320, 555), PANEL, BORDER, 24)
    text(draw, (915, 165), "Medical Report", TEAL, F18)
    multiline(
        draw,
        (915, 220),
        "Generate and download a PDF report, or email it directly to a doctor with patient details and clinical guidance.",
        42,
        MUTED,
        F16,
    )
    rounded(draw, (915, 350, 1265, 405), "#0b1425", BORDER, 14)
    text(draw, (940, 366), "Patient Name (optional)", MUTED, F16)
    rounded(draw, (915, 430, 1105, 490), TEAL, TEAL, 16)
    text(draw, (1010, 449), "Download PDF", "#04111b", F18, "ma")

    rounded(draw, (305, 595, 1320, 910), PANEL, BORDER, 24)
    text(draw, (345, 640), "AI Clinical Insights", TEAL, F18)
    text(draw, (345, 690), "Stage: Mild Demented", TEXT, F30)
    multiline(
        draw,
        (345, 745),
        "The MRI scan shows structural changes consistent with mild Alzheimer's disease. Cognitive decline is noticeable and begins to affect daily activities. Prompt medical consultation and structured support are recommended.",
        112,
        MUTED,
        F18,
    )
    tags = ["Symptoms", "Precautions", "Cognitive Care", "Lifestyle", "Doctor Advice"]
    x = 345
    for i, tag in enumerate(tags):
        color = TEAL if i == 0 else BORDER
        rounded(draw, (x, 850, x + 170, 895), "#0b2330" if i == 0 else "#0b1425", color, 15)
        text(draw, (x + 85, 864), tag, TEXT if i == 0 else MUTED, F14, "ma")
        x += 185
    img.save(OUT / "gradcam-report.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    homepage()
    upload_prediction()
    gradcam_report()
    print("Updated screenshots/app-homepage.png")
    print("Updated screenshots/upload-prediction.png")
    print("Updated screenshots/gradcam-report.png")
