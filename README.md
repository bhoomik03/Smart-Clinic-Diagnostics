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

- **Backend:** Python 3.12, PostgreSQL 14+
- **Frontend:** Streamlit (Custom Premium CSS)
- **OCR Engine:** Tesseract OCR, PyMuPDF
- **ML Framework:** Scikit-Learn (Random Forest, Logistic Regression)
- **Analytics:** Pandas, Plotly Express
- **Security:** Bcrypt, SMTP (Email OTP), MSG91 (SMS OTP)

---

## 📂 Project Structure

- `frontend/app.py`: Main Streamlit application and UI logic.
- `backend/`: Core service layer for OCR, ML, Rule Engine, and Database utilities.
- `data/`: Cleaned clinical datasets used for model scaling and reference.
- `start_app.bat`: Quick-start batch script for activating the virtual environment and launching the app.

---

## 💡 Quick Start

1. **Configure Database**: Ensure PostgreSQL is running and credentials are set in your environment.
2. **Install Dependencies**: `pip install -r requirements.txt` (inside `medical_ai_project/`).
3. **Launch Application**: Double-click `start_app.bat` or run `streamlit run frontend/app.py`.

---

## 🛡️ Medical Disclaimer
This system is an AI-assisted diagnostic tool designed for clinical decision support. All predictions and observations must be reviewed and validated by a qualified medical professional before taking any clinical action.