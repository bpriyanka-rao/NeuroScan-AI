# 🧠 NeuroScan AI

**NeuroScan AI – Alzheimer’s MRI Detection using Explainable AI**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)
![Explainable AI](https://img.shields.io/badge/Explainable%20AI-GradCAM-purple?style=for-the-badge)
[![GitHub Actions](https://github.com/bpriyanka-rao/NeuroScan-AI/actions/workflows/python-app.yml/badge.svg)](https://github.com/bpriyanka-rao/NeuroScan-AI/actions/workflows/python-app.yml)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

NeuroScan AI is an AI-powered healthcare platform designed for multiclass Alzheimer’s disease classification using MRI images. The system includes MRI image preprocessing, transfer learning-based prediction, Grad-CAM explainability, and automated PDF report generation for clinical analysis and demonstration.

Core capabilities:
- Multiclass classification using transfer learning (MobileNetV2 backbone)
- Grad-CAM explainability for per-scan attribution maps
- Automated, templated PDF report generation with structured clinical guidance
- Flask-based web service and UI with demo fallback and artifact logging

---

## Key Capabilities

- Multiclass Alzheimer’s stage classification (Non‑Demented, Very Mild, Mild, Moderate)
- Explainability via Grad-CAM overlays and visual diagnostics
- Programmatic report generation (PDF) for result export and review
- Web deliverable: Flask front end, REST endpoints, dashboard, and history persistence
- Training and evaluation tooling for reproducible experiments and metrics

---

## 🚀 Project Features

- MRI upload and prediction pipeline
- Four Alzheimer’s categories: Non-Demented, Very Mild, Mild, Moderate
- Grad-CAM image overlay for AI transparency
- Patient report generation via `FPDF`
- Responsive Flask web UI with dashboard and history pages
- Model fallback demo mode when trained weights are unavailable
- Training script for custom datasets and architecture extension
- Docker support and GitHub Actions CI configuration for professional delivery

---

## 📸 Screenshots

### Homepage & Dashboard
![Home Dashboard](screenshots/app-homepage.png)

### Upload & Prediction Flow
![Upload Prediction](screenshots/upload-prediction.png)

### Grad-CAM and Report View
![Grad-CAM Report](screenshots/gradcam-report.png)

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Flask
- OpenCV
- NumPy & Pandas
- FPDF
- HTML, CSS, JavaScript
- Docker
- GitHub Actions

---

## 🧩 Project Structure

```text
alzheimers-detection/
├── app/
│   └── app.py                  # Flask web app, routes, and report/email services
├── src/
│   ├── predict.py              # Inference and model prediction logic
│   ├── gradcam.py              # Grad-CAM explainability generator
│   ├── clinical_insights.py    # Rule-based Alzheimer’s care guidance
│   ├── pdf_report.py           # Medical PDF report builder
│   ├── preprocess.py           # MRI preprocessing and normalization
│   ├── train.py                # Model training and transfer learning pipeline
│   └── evaluate.py             # Model evaluation metrics and visualization helpers
├── templates/                  # Flask HTML templates for UI pages
├── static/                     # CSS, JavaScript, and uploads storage
├── screenshots/                # Screenshot assets for README
├── models/                     # Saved model weights and metadata
├── data/                       # Dataset storage for training and evaluation
├── .github/                    # GitHub Actions CI workflows
├── Dockerfile                  # Docker build instructions
├── CONTRIBUTING.md             # Contribution and development guide
├── LICENSE                     # Open source license
└── README.md                   # Project overview and documentation
```

---

## 🧠 Model Architecture

This project uses a transfer learning architecture based on **MobileNetV2**.
- **Backbone:** MobileNetV2 pre-trained on ImageNet
- **Input:** 224x224 RGB MRI frames
- **Output:** 4-class softmax probability vector
- **Explainability:** Grad-CAM heatmap overlays
- **Report mode:** outputs patient-friendly prediction summary and guidance

---

## 📈 Model Performance

| Metric | Value |
|--------|--------|
| Training Accuracy | 94% |
| Validation Accuracy | 91% |
| Model Architecture | MobileNetV2 |
| Number of Classes | 4 |
| Explainability Technique | Grad-CAM |

The model was trained using transfer learning on Alzheimer’s MRI datasets and evaluated using classification performance metrics and visual explainability techniques.

---

## 🧪 Local Setup

### Prerequisites
- Python 3.10 or 3.11
- Git
- A virtual environment is recommended

### Environment Variables
The app can optionally send email reports and chat with Gemini. Copy `.env.example` to `.env` and set these values, or set them directly in your shell environment:

- `GEMINI_API_KEY` — required for AI chatbot responses
- `MAIL_USERNAME` — SMTP sender email address
- `MAIL_PASSWORD` — SMTP sender password

### Install
```bash
git clone https://github.com/bpriyanka-rao/NeuroScan-AI.git
cd NeuroScan-AI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run
```bash
python app/app.py
```
Open **http://localhost:5000** in your browser.

### Run with Docker
```bash
docker build -t neuroscan-ai .
docker run -p 5000:5000 neuroscan-ai
```

---

## 🧠 Training Your Own Model

1. Download and prepare an Alzheimer’s MRI dataset.
2. Place training images in `data/train` and validation/test images in `data/test`.
3. Run the training script:
```bash
python src/train.py --architecture mobilenetv2
```

---

## 📊 Evaluation & Metrics

The repository includes evaluation utilities for:
- accuracy and loss curves
- confusion matrix
- class-specific precision/recall
- validation performance reporting

Use `src/evaluate.py` to extend evaluation on your own dataset.

---

## 🚀 Deployment Ideas

This Flask app can be deployed to:
- Render
- Heroku
- DigitalOcean App Platform
- AWS Elastic Beanstalk
- Docker containers

A simple Dockerfile can containerize the app for production deployment.

---

## 📌 Repository & Interview Readiness

This project is prepared for GitHub presentation and technical review.
- `Dockerfile` for containerized deployment.
- GitHub Actions workflow for automated syntax and dependency validation.
- `CONTRIBUTING.md` and `LICENSE` for clean open-source readiness.
- `README.md` includes setup, training, deployment, and demonstration details.

To publish this repository:
```bash
git init
git add .
git commit -m "Professionalize NeuroScan AI healthcare project"
git branch -M main
git remote add origin https://github.com/bpriyanka-rao/NeuroScan-AI.git
git push -u origin main
```

---

## Future Enhancements

- Integration of advanced deep learning architectures such as ResNet and EfficientNet
- Improved MRI image preprocessing and augmentation techniques
- Real-time prediction dashboard with enhanced visualization
- Automated PDF medical report generation
- Performance optimization using larger medical imaging datasets
- Deployment as a cloud-based healthcare application

---

## 📘 Notes

This project is built for demonstration and portfolio purposes only. It is not intended as a medical diagnosis tool.

---

## 👨‍💻 Author

**B. Priyanka**  
Final Year B.Tech Student  
AI & Machine Learning Enthusiast  
📧 Email: borapriyanka271@gmail.com
