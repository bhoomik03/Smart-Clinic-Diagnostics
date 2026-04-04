# 🩺 Medical AI Diagnostic System

Intelligence-Driven Clinical Diagnostics & Analytics.

This platform integrates advanced Machine Learning models with a robust Clinical Rule Engine and Smart OCR to provide automated, high-accuracy diagnostic support for healthcare professionals.

---

## 🚀 Key Features

- **🔐 Secure Authentication:** OTP-based login (Email/SMS) with Bcrypt-protected profile management.
- **📄 Smart OCR Extraction:** Automatically digitizes patient metrics from PDF and image-based medical reports.
- **🤖 AI-Driven Predictions:** High-accuracy models for **Diabetes** and **Heart Disease** risk assessment.
- **⚖️ Clinical Rule Engine:** Evaluates 11+ conditions (Hypertension, Obesity, Kidney Function, etc.) against international WHO/ADA standards.
- **📊 Longitudinal Analytics:** Track patient health trends over time with interactive Plotly-based visualizations.
- **📋 Management & Audit:** Full Admin Dashboard for user management, system statistics, and operational auditing.

---

## 🛠️ Technology Stack

### **Core Frameworks & Languages**
- **Backend**: Python 3.12 (Core Logic & Service Layer)
- **Frontend**: Streamlit (Premium UI Framework) with Custom **Ultra-Pro CSS**
- **Database**: PostgreSQL 15+ (Relational Data Management)

### **AI & Machine Learning**
- **ML Framework**: Scikit-Learn (Random Forest, Logistic Regression)
- **OCR Engine**: Tesseract OCR (Optical Character Recognition)
- **Document Parsing**: PyMuPDF & pdf2image (High-fidelity PDF processing)
- **Image Processing**: OpenCV & Pillow (Clinical report binarization/cleanup)

### **Analytics & Visualization**
- **Data Manipulation**: Pandas & NumPy
- **Interactive Charts**: Plotly Express & Plotly Graph Objects

### **Communication & Security**
- **Authentication**: Bcrypt (Password Hashing)
- **SMS Gateway**: MSG91 API (Secure Mobile OTP)
- **Email Service**: SMTP (Secure Email OTP)
- **API Integration**: Requests Library

### **Developer Tools**
- **IDE**: Visual Studio Code
- **Version Control**: Git & GitHub
- **Environment**: Python Virtualenv

---

## 📂 Project Structure

```text
medical_ai_project/
├── backend/
│   ├── auth/           # OTP & Authentication logic
│   ├── database/       # PostgreSQL utility functions
│   ├── models/         # Trained ML models (.pkl)
│   ├── ocr/            # Tesseract OCR processing
│   ├── preprocessing/  # Data cleaning scripts
│   ├── rule_engine/    # Clinical WHO/ADA guidelines
│   └── training/       # Model training pipelines
├── data/               # Clinical datasets (CSVs)
├── frontend/
│   ├── app.py          # Main Streamlit UI
│   └── style.css       # Premium Ultra-Pro styling
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

---

## 💡 Quick Start

1. **Configure Database**: Ensure PostgreSQL is running and credentials are set in your environment.
2. **Install Dependencies**: `pip install -r requirements.txt` (inside `medical_ai_project/`).
3. **Launch Application**: Double-click `start_app.bat` or run `streamlit run frontend/app.py`.

---

## 🛡️ Medical Disclaimer
This system is an AI-assisted diagnostic tool designed for clinical decision support. All predictions and observations must be reviewed and validated by a qualified medical professional before taking any clinical action.