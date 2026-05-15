import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tensorflow.keras.datasets import cifar10

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = ROOT / "screenshots"
DATA_DIR = ROOT / "data" / "demo_cifar10"
CLASS_MAP = {
    "MildDemented": [0, 2, 4, 6],
    "ModerateDemented": [1, 3, 5, 7],
    "NonDemented": [8],
    "VeryMildDemented": [9],
}
SAMPLE_COUNT = 80
IMG_SIZE = (224, 224)

SCREENSHOT_CONFIGS = [
    {
        "filename": "app-homepage.png",
        "title": "NeuroScan AI Dashboard",
        "subtitle": "Upload MRI • Predict Stage • View Insights",
    },
    {
        "filename": "upload-prediction.png",
        "title": "MRI Upload & Prediction",
        "subtitle": "Prediction results, confidence, and next steps.",
    },
    {
        "filename": "gradcam-report.png",
        "title": "Grad-CAM + Downloadable Report",
        "subtitle": "Explainability maps and PDF report delivery.",
    },
]


def draw_placeholder(image: Image.Image, title: str, subtitle: str):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    color = (248, 250, 252)
    accent = (79, 70, 229)
    draw.rectangle([20, 20, w - 20, h - 20], outline=accent, width=4)
    text_x = 40
    draw.text((text_x, 40), title, fill=color, font=ImageFont.load_default())
    draw.text((text_x, 80), subtitle, fill=color, font=ImageFont.load_default())
    draw.rectangle([40, 140, 260, 230], fill=(30, 41, 59), outline=accent)
    draw.rectangle([300, 140, w - 40, 240], fill=(30, 41, 59), outline=accent)
    draw.rectangle([300, 270, w - 40, h - 40], fill=(30, 41, 59), outline=accent)
    draw.text((50, 160), "Upload MRI", fill=color, font=ImageFont.load_default())
    draw.text((310, 160), "AI Prediction", fill=color, font=ImageFont.load_default())
    draw.text((310, 300), "Clinical insights and report", fill=color, font=ImageFont.load_default())


def create_screenshots():
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for config in SCREENSHOT_CONFIGS:
        img = Image.new("RGB", (900, 520), (15, 23, 42))
        draw_placeholder(img, config["title"], config["subtitle"])
        img.save(SCREENSHOT_DIR / config["filename"], format="PNG")


def create_demo_dataset():
    (DATA_DIR / "MildDemented").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "ModerateDemented").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "NonDemented").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "VeryMildDemented").mkdir(parents=True, exist_ok=True)

    (x_train, y_train), _ = cifar10.load_data()
    x_train = x_train.astype("uint8")
    y_train = y_train.flatten()

    class_queues = {name: [] for name in CLASS_MAP}
    for image, label in zip(x_train, y_train):
        for class_name, labels in CLASS_MAP.items():
            if int(label) in labels and len(class_queues[class_name]) < SAMPLE_COUNT:
                class_queues[class_name].append(image)
                break
        if all(len(q) >= SAMPLE_COUNT for q in class_queues.values()):
            break

    for class_name, images in class_queues.items():
        for idx, img in enumerate(images):
            pil_img = Image.fromarray(img).resize(IMG_SIZE)
            pil_img.save(DATA_DIR / class_name / f"{class_name}_{idx}.png", format="PNG")


if __name__ == "__main__":
    create_screenshots()
    create_demo_dataset()
    print("Generated PNG screenshots and demo CIFAR-10 dataset at data/demo_cifar10")
