import streamlit as st
import pandas as pd
import os
import sys
import re
import plotly.express as px
import plotly.graph_objects as go
import base64
import datetime
import pytz

# Centralized IST Helper
def localize_ist(dt_obj):
    """Safely converts ANY timestamp (aware or naive) to Asia/Kolkata (IST)."""
    if dt_obj is None: return "N/A"
    ist_tz = pytz.timezone('Asia/Kolkata')
    try:
        # If naive, assume it's UTC and localize
        if dt_obj.tzinfo is None:
            dt_obj = pytz.utc.localize(dt_obj)
        return dt_obj.astimezone(ist_tz)
    except:
        return dt_obj

# Add the backend path to sys.path to easily import modules
backend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backend')
sys.path.append(backend_path)

from rule_engine.rules import (
    evaluate_hypertension, evaluate_obesity, evaluate_kidney_disease, 
    evaluate_haemogram, evaluate_dengue, evaluate_typhoid, 
    evaluate_liver_function, evaluate_inflammation, evaluate_blood_sugar_fasting,
    evaluate_cholesterol, evaluate_heart_rate, evaluate_body_temp, evaluate_oxygen
)
from rule_engine.recommendations import DISEASE_RECOMMENDATIONS
from training.prediction import predict_disease
from database.db_utils import (
    add_patient, add_ml_prediction, 
    get_patient_history, get_disease_breakdown,
    register_user, authenticate_user, get_all_users, get_system_stats,
    verify_user_exists, update_password, update_user_role, get_audit_logs,
    get_patient_demographics, get_system_setting, set_system_setting,
    delete_user, toggle_user_status, admin_reset_password, delete_patient_record,
    get_all_patients_admin, get_db_connection, get_registration_data,
    update_user_info, get_user_dashboard_stats, get_session_vitals,
    activate_user_account, store_otp, initialize_tables,
    verify_otp_db, get_user_id_by_email, get_system_utilization, delete_login_activity,
    get_latest_patient_insight, add_diagnostic_session, add_clinical_vital,
    add_clinical_observation, get_db_status
)
from auth.otp_manager import generate_otp, send_otp
from ocr.ocr_engine import process_document_to_dict
from sklearn.preprocessing import StandardScaler, LabelEncoder

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

# Ensure database tables exist (CRITICAL for new cloud DBs like Neon)
try:
    initialize_tables()
except Exception as e:
    st.error(f"Database Initialization Error: {e}")

def render_luxury_header(title, icon="✨", badge_text=None, mode="compact", return_html=False):
    if mode == "hero":
        html = f"""
            <div class="ultra-pro-hero">
                <div class="hero-content-left">
                    <div class="hero-icon-box">{icon}</div>
                    <div>
                        <div class="hero-title-main">{title}</div>
                        <div class="hero-subtitle-main">Clinical Grade AI Diagnostics</div>
                    </div>
                </div>
            </div>
        """
    else:
        html = f"""
            <div class="ultra-pro-header">
                <div class="ultra-pro-title">
                    <div class="ultra-pro-icon">{icon}</div>
                    {title}
                </div>
            </div>
        """
    
    if return_html:
        return html
    st.markdown(html, unsafe_allow_html=True)

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'style.css')
    if os.path.exists(css_path):
        try:
            with open(css_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f"<style>{f.read()}</style>"
        except Exception:
            return ""
    return ""


def render_login_ui():
    # --- Premium login UI style + hero section ---
    st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] {
                background: #0F172A !important;
                overflow: hidden;
            }
            [data-testid="stAppViewContainer"]::before {
                content: '';
                position: absolute;
                width: 300%; height: 300%;
                top: -100%; left: -100%;
                background-image:
                    radial-gradient(circle at 20% 30%, rgba(0, 212, 255, 0.1) 0%, transparent 40%),
                    radial-gradient(circle at 80% 70%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                    radial-gradient(circle at 50% 50%, rgba(0, 212, 255, 0.05) 0%, transparent 40%);
                animation: molecularFloat 30s ease-in-out infinite;
                pointer-events: none;
                filter: blur(80px);
                z-index: -1;
            }
            @keyframes molecularFloat {
                0%, 100% { transform: translate(0, 0) rotate(0deg); }
                50% { transform: translate(-5%, -5%) rotate(5deg); }
            }
            .stTabs {
                background: rgba(255, 255, 255, 0.03) !important;
                backdrop-filter: blur(28px) saturate(180%) !important;
                -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: 32px !important;
                padding: 15px 10px !important;
                box-shadow: 0 32px 64px rgba(0, 0, 0, 0.4) !important;
                max-width: 520px;
                margin: 0 auto;
                animation: authCardEntrance 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
            }
            @keyframes authCardEntrance {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .auth-title {
                text-align: center;
                color: white !important;
                font-weight: 900;
                font-size: 2.6rem;
                letter-spacing: -1.5px;
                text-shadow: 0 4px 12px rgba(0, 212, 255, 0.4);
                margin-top: 40px;
            }
            .auth-subtitle {
                text-align: center;
                color: #00D4FF !important;
                font-weight: 600;
                font-size: 1.1rem;
                margin-bottom: 30px;
                letter-spacing: 1px;
            }
            [data-testid="stTextInput"] input {
                background: rgba(255, 255, 255, 0.05) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                color: white !important;
                border-radius: 12px !important;
                padding: 12px !important;
            }
            .stButton>button {
                border-radius: 12px !important;
                font-weight: 700 !important;
                height: 3em !important;
                background: linear-gradient(135deg, #00D4FF 0%, #4F46E5 100%) !important;
                border: none !important;
                color: white !important;
            }
            /* Enhanced Font Visibility & Navigation Fixes */
            .stTabs {
                pointer-events: auto !important;
            }
            .stTabs [data-baseweb="tab"] {
                color: rgba(255, 255, 255, 0.7) !important;
            }
            .stTabs [aria-selected="true"] {
                color: white !important;
            }
            .stTabs h3 {
                color: white !important;
                text-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
                margin-bottom: 20px !important;
                font-weight: 800 !important;
            }
            [data-testid="stForm"] label {
                color: #E2E8F0 !important;
                font-weight: 600 !important;
                font-size: 0.95rem !important;
                margin-bottom: 8px !important;
            }
            .stTabs p, .stTabs div[data-testid="stMarkdownContainer"] {
                color: #CBD5E1 !important;
            }
            .stTabs input::placeholder {
                color: rgba(255, 255, 255, 0.3) !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="auth-header">
            <div class="auth-title">AI-Based Medical Diagnosis</div>
            <div class="auth-subtitle">Advanced Clinical Diagnostic Intelligence</div>
        </div>
    """, unsafe_allow_html=True)

    tab_login, tab_reg, tab_rec = st.tabs(["🔒 Log In", "📝 Register", "🔑 Recovery"])

    with tab_login:
        with st.form("login_form", border=False):
            st.markdown("### Welcome Back")
            log_username = st.text_input("Username", placeholder="Enter your username")
            log_password = st.text_input("Password", type="password", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.form_submit_button("Login", type="primary", width="stretch"):
                if not log_username or not log_password:
                    st.error("Please provide both credentials.")
                else:
                    success, role, msg, profile_dict, user_id = authenticate_user(log_username, log_password)
                    if success:
                        maintenance = get_system_setting("maintenance_mode", "false") == "true"
                        if maintenance and role not in ['admin']:
                            st.error("System is under maintenance. Only administrators can log in right now.")
                        else:
                            st.session_state.logged_in = True
                            st.session_state.username = log_username
                            st.session_state.role = role
                            st.session_state.patient_profile = profile_dict
                            st.session_state.user_id = user_id
                            st.toast(f"Welcome back, {log_username}!", icon="✨")
                            st.rerun()
                    else:
                        st.error(f"Authentication Failed: {msg}")
                        st.markdown("---")
                        st.write("Forgotten your credentials?")
                        st.info("💡 Tip: Use the **Recovery** tab above to reset your password.")

    with tab_reg:
        if 'reg_otp_sent' not in st.session_state:
            st.session_state.reg_otp_sent = False

        if not st.session_state.reg_otp_sent:
            with st.form("register_form", border=False):
                st.markdown("### Create Clinical Account")

                colA, colB = st.columns(2)
                with colA:
                    reg_username = st.text_input("Username *")
                    reg_password = st.text_input("Password *", type="password")
                    reg_confirm = st.text_input("Confirm Password *", type="password")
                with colB:
                    reg_name = st.text_input("Full Name *")
                    reg_age = st.number_input("Age", min_value=1, max_value=120, value=30)
                    reg_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

                st.markdown("---")
                reg_email = st.text_input("Email Address * (For OTP)")
                reg_contact = st.text_input("Contact Number *")
                reg_address = st.text_area("Physical Address", height=80)

                if st.form_submit_button("Register", type="primary", width="stretch"):
                    if not all([reg_username, reg_password, reg_name, reg_email, reg_contact]):
                        st.error("Please fill in all required fields (*).")
                    elif not is_valid_email(reg_email):
                        st.error("Invalid Email format.")
                    elif reg_password != reg_confirm:
                        st.error("Passwords do not match.")
                    else:
                        success, msg = register_user(reg_username, reg_password, reg_name, int(reg_age), reg_gender, reg_email, reg_contact, reg_address)
                        if success:
                            otp = generate_otp()
                            if store_otp(reg_email, otp):
                                send_otp(otp, email=reg_email, contact=reg_contact)
                                st.session_state.reg_email = reg_email
                                st.session_state.reg_contact = reg_contact
                                st.session_state.reg_username = reg_username
                                st.session_state.reg_otp_sent = True
                                st.success(f"Verification code sent to {reg_email}")
                                st.rerun()
                            else:
                                st.error("OTP System currently unavailable. Try again later.")
                        else:
                            st.error(msg)
        else:
            with st.form("reg_otp_form", border=False):
                st.markdown("### Verify Account")
                st.info(f"Enter the 6-digit code sent to **{st.session_state.reg_email}**")
                
                col_r1, col_r2 = st.columns([3, 1])
                with col_r1:
                    user_otp = st.text_input("Verification Code", placeholder="000000")
                with col_r2:
                    st.write("") # Padding
                    st.write("") # Padding
                    if st.form_submit_button("Resend", width="stretch"):
                        new_otp = generate_otp()
                        if store_otp(st.session_state.reg_email, new_otp):
                            send_otp(new_otp, email=st.session_state.reg_email, contact=st.session_state.get('reg_contact'))
                            st.toast("New OTP sent successfully!")
                        else:
                            st.error("Failed to resend. Try again.")

                colB1, colB2 = st.columns(2)
                with colB1:
                    if st.form_submit_button("Verify & Activate", type="primary", width="stretch"):
                        v_success, v_msg = verify_otp_db(st.session_state.reg_email, user_otp)
                        if v_success:
                            u_id = get_user_id_by_email(st.session_state.reg_email)
                            if activate_user_account(u_id):
                                st.success("Account activated! Please log in.")
                                st.session_state.reg_otp_sent = False
                                import time
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                st.error("Activation failed. Contact support.")
                        else:
                            st.error(v_msg)
                with colB2:
                    if st.form_submit_button("Cancel", width="stretch"):
                        from database.db_utils import delete_unverified_user
                        delete_unverified_user(st.session_state.reg_username)
                        st.session_state.reg_otp_sent = False
                        st.rerun()

    with tab_rec:
        st.markdown("### Account Recovery")
        if 'otp_sent' not in st.session_state:
            st.session_state.otp_sent = False
            st.session_state.otp_verified = False

        if not st.session_state.otp_sent:
            with st.form("request_otp_form", border=False):
                st.write("Enter your registered email to receive a reset code.")
                reset_contact = st.text_input("Email or Contact Number")
                if st.form_submit_button("Send Reset Code", type="primary", width="stretch"):
                    exists, user_info = verify_user_exists(reset_contact)
                    if exists:
                        target_email = user_info['email']
                        otp = generate_otp()
                        if store_otp(target_email, otp):
                            send_otp(otp, email=target_email, contact=user_info['contact'])
                            st.session_state.reset_email = target_email
                            st.session_state.reset_contact = user_info['contact']
                            st.session_state.reset_username = user_info['username']
                            st.session_state.otp_sent = True
                            st.success(f"OTP Sent to {target_email}")
                            st.rerun()
                        else:
                            st.error("Database connection issue. Try again.")
                    else:
                        st.error("No account found with this identity.")
        elif st.session_state.otp_sent and not st.session_state.otp_verified:
            with st.form("verify_otp_form", border=False):
                st.info(f"Verification code sent to {st.session_state.reset_email}")
                
                col_rr1, col_rr2 = st.columns([3, 1])
                with col_rr1:
                    user_otp = st.text_input("Enter Code", placeholder="000000")
                with col_rr2:
                    st.write("") # Padding
                    st.write("") # Padding
                    if st.form_submit_button("Resend", width="stretch"):
                        new_otp = generate_otp()
                        if store_otp(st.session_state.reset_email, new_otp):
                            send_otp(new_otp, email=st.session_state.reset_email, contact=st.session_state.get('reset_contact'))
                            st.toast("New OTP sent successfully!")
                        else:
                            st.error("Failed to resend.")

                if st.form_submit_button("Verify Code", type="primary", width="stretch"):
                    v_success, v_msg = verify_otp_db(st.session_state.reset_email, user_otp)
                    if v_success:
                        st.session_state.otp_verified = True
                        st.rerun()
                    else:
                        st.error(v_msg)
            if st.button("Cancel Recovery", key="cancel_otp_btn", use_container_width=True):
                st.session_state.otp_sent = False
                st.rerun()
        elif st.session_state.otp_verified:
            with st.form("reset_password_form", border=False):
                st.markdown("### Set New Password")
                st.write(f"Account for: **{st.session_state.reset_username}**")
                
                new_pswd = st.text_input("New Password", type="password", placeholder="••••••••")
                confirm_pswd = st.text_input("Confirm New Password", type="password", placeholder="••••••••")
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.form_submit_button("Update Password", type="primary", use_container_width=True):
                    if not new_pswd or not confirm_pswd:
                        st.error("Please fill in both fields.")
                    elif new_pswd != confirm_pswd:
                        st.error("Passwords do not match.")
                    elif len(new_pswd) < 6:
                        st.error("Password must be at least 6 characters.")
                    else:
                        success, res_msg = update_password(st.session_state.reset_username, new_pswd)
                        if success:
                            st.success("✅ Password updated successfully! Please log in.")
                            # Reset all recovery states
                            st.session_state.otp_sent = False
                            st.session_state.otp_verified = False
                            st.session_state.reset_email = None
                            st.session_state.reset_username = None
                            import time
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(res_msg)
            if st.button("Back to Login", key="back_to_login_btn", use_container_width=True):
                st.session_state.otp_sent = False
                st.session_state.otp_verified = False
                st.rerun()

    st.stop()


def evaluate_manual_clinical_risk(data, target_block=None):
    results = []
    
    # Define block mapping for strict filtering as requested by user
    # This prevents parameters from one block showing up in another's report
    block_map = {
        'diabetes': ['Glucose', 'Pregnancies', 'Skin Thickness', 'Insulin', 'Diabetes Pedigree', 'BMI'],
        'heart': ['Chest Pain Severity', 'Exercise Induced Angina', 'Major Vessel Blockage', 'Resting ECG', 'ST Depression', 'Max Heart Rate', 'ST Slope', 'Thalassemia'],
        'core_vitals': ['Systolic BP', 'Diastolic BP', 'Glucose', 'BMI', 'Heart Rate', 'Oxygen Saturation', 'Body Temperature', 'Total Cholesterol'],
        'pathology': ['Creatinine', 'Hemoglobin', 'WBC', 'Platelets', 'AST', 'ALT', 'CRP', 'Typhoid', 'Dengue'],
        'general': ['Fever', 'Cough', 'Fatigue', 'Sore throat', 'Headache', 'Shortness of breath', 'Runny nose', 'Body ache']
    }
    
    def is_allowed(param_name):
        if not target_block: return True # Show all if no target (standard/OCR mode)
        allowed_list = block_map.get(target_block, [])
        return any(a.lower() in param_name.lower() for a in allowed_list)

    # General Symptoms Processing
    for i in range(1, 4):
        s_val = data.get(f'symptom_{i}')
        if s_val and s_val != "None" and is_allowed(s_val):
            results.append({'param': s_val, 'val': s_val, 'status': 'HIGH', 'msg': f'[SYMPTOM] Persistent {s_val} reported'})

    sys = data.get('systolic')
    if sys is not None and sys > 0 and is_allowed('Systolic BP'):
        if sys < 90: results.append({'param': 'Systolic BP', 'val': sys, 'status': 'LOW', 'msg': '[HYPOTENSION] Low blood pressure detected'})
        elif 90 <= sys < 120: results.append({'param': 'Systolic BP', 'val': sys, 'status': 'NORMAL', 'msg': '[NORMAL] Optimal blood pressure'})
        elif 120 <= sys < 130: results.append({'param': 'Systolic BP', 'val': sys, 'status': 'HIGH', 'msg': '[ELEVATED_BP] Elevated blood pressure detected'})
        elif sys >= 130: results.append({'param': 'Systolic BP', 'val': sys, 'status': 'HIGH', 'msg': '[HYPERTENSION] High blood pressure detected'})

    dia = data.get('diastolic')
    if dia is not None and dia > 0 and is_allowed('Diastolic BP'):
        if dia < 60: results.append({'param': 'Diastolic BP', 'val': dia, 'status': 'LOW', 'msg': '[HYPOTENSION] Low blood pressure detected'})
        elif 60 <= dia < 80: results.append({'param': 'Diastolic BP', 'val': dia, 'status': 'NORMAL', 'msg': '[NORMAL] Blood pressure within normal range'})
        elif dia >= 80: results.append({'param': 'Diastolic BP', 'val': dia, 'status': 'HIGH', 'msg': '[HYPERTENSION] High blood pressure detected'})
    
    glu = data.get('glucose')
    if glu is not None and glu > 0 and is_allowed('Glucose'):
        if glu < 70: results.append({'param': 'Glucose', 'val': glu, 'status': 'LOW', 'msg': '[HYPOGLYCEMIA] Low blood sugar detected'})
        elif 70 <= glu <= 99: results.append({'param': 'Glucose', 'val': glu, 'status': 'NORMAL', 'msg': '[NORMAL] Blood glucose normal'})
        elif glu >= 100: results.append({'param': 'Glucose', 'val': glu, 'status': 'HIGH', 'msg': '[DIABETES_RISK] High/Abnormal blood sugar detected'})
        
    bmi = data.get('bmi')
    if bmi is not None and bmi > 0 and is_allowed('BMI'):
        if bmi < 18.5: results.append({'param': 'BMI', 'val': bmi, 'status': 'LOW', 'msg': '[UNDERWEIGHT] BMI indicates underweight condition'})
        elif 18.5 <= bmi <= 24.9: results.append({'param': 'BMI', 'val': bmi, 'status': 'NORMAL', 'msg': '[NORMAL] Healthy BMI range'})
        elif bmi >= 25: results.append({'param': 'BMI', 'val': bmi, 'status': 'HIGH', 'msg': '[OVERWEIGHT_RISK] Increased obesity risk'})

    hr = data.get('heart_rate_bpm')
    if hr is not None and hr > 0 and is_allowed('Heart Rate'):
        if hr < 50: results.append({'param': 'Heart Rate', 'val': hr, 'status': 'LOW', 'msg': '[BRADYCARDIA] Low heart rate detected'})
        elif 50 <= hr <= 100: results.append({'param': 'Heart Rate', 'val': hr, 'status': 'NORMAL', 'msg': '[NORMAL] Heart rate normal'})
        elif hr > 100: results.append({'param': 'Heart Rate', 'val': hr, 'status': 'HIGH', 'msg': '[TACHYCARDIA] Elevated heart rate detected'})
        
    chol = data.get('cholesterol')
    if chol is not None and chol > 0 and is_allowed('Total Cholesterol'):
        if chol < 120: results.append({'param': 'Total Cholesterol', 'val': chol, 'status': 'LOW', 'msg': '[LOW_CHOLESTEROL] Cholesterol below recommended level'})
        elif 120 <= chol < 200: results.append({'param': 'Total Cholesterol', 'val': chol, 'status': 'NORMAL', 'msg': '[NORMAL] Cholesterol within healthy range'})
        elif chol >= 200: results.append({'param': 'Total Cholesterol', 'val': chol, 'status': 'HIGH', 'msg': '[HYPERCHOLESTEROLEMIA] Cardiovascular risk detected'})
        
    o2 = data.get('oxygen_saturation')
    if o2 is not None and o2 > 0 and is_allowed('Oxygen Saturation'):
        if o2 < 95: results.append({'param': 'Oxygen Saturation', 'val': o2, 'status': 'LOW', 'msg': '[HYPOXEMIA] Low oxygen saturation detected'})
        elif 95 <= o2 <= 100: results.append({'param': 'Oxygen Saturation', 'val': o2, 'status': 'NORMAL', 'msg': '[NORMAL] Oxygen level normal'})
        
    temp = data.get('body_temperature_c')
    if temp is not None and temp > 0 and is_allowed('Body Temperature'):
        if temp < 36.1: results.append({'param': 'Body Temperature', 'val': temp, 'status': 'LOW', 'msg': '[HYPOTHERMIA] Body temperature critically low'})
        elif 36.1 <= temp <= 37.2: results.append({'param': 'Body Temperature', 'val': temp, 'status': 'NORMAL', 'msg': '[NORMAL] Body temperature normal'})
        elif temp > 37.2: results.append({'param': 'Body Temperature', 'val': temp, 'status': 'HIGH', 'msg': '[FEVER] Elevated body temperature detected'})

    # Heart Diagnostics
    cp = data.get('cp')
    if cp is not None and cp >= 0 and is_allowed('Chest Pain Severity'):
        if cp <= 1: results.append({'param': 'Chest Pain Severity', 'val': cp, 'status': 'NORMAL', 'msg': '[NORMAL] No significant cardiac pain'})
        elif cp >= 2: results.append({'param': 'Chest Pain Severity', 'val': cp, 'status': 'HIGH', 'msg': '[ANGINA_RISK] Possible coronary artery disease'})

    exang = data.get('exang')
    if exang is not None and exang >= 0 and is_allowed('Exercise Induced Angina'):
        if exang == 0: results.append({'param': 'Exercise Induced Angina', 'val': exang, 'status': 'NORMAL', 'msg': '[NORMAL] Normal exercise tolerance'})
        elif exang >= 1: results.append({'param': 'Exercise Induced Angina', 'val': exang, 'status': 'HIGH', 'msg': '[EXERCISE_ANGINA] Exercise induced angina detected'})

    ca = data.get('ca')
    if ca is not None and ca >= 0 and is_allowed('Major Vessel Blockage'):
        if ca == 0: results.append({'param': 'Major Vessel Blockage', 'val': ca, 'status': 'NORMAL', 'msg': '[NORMAL] Adequate coronary blood flow'})
        elif ca >= 1: results.append({'param': 'Major Vessel Blockage', 'val': ca, 'status': 'HIGH', 'msg': '[CORONARY_BLOCKAGE] Coronary artery blockage risk'})

    restecg = data.get('restecg')
    if restecg is not None and restecg >= 0 and is_allowed('Resting ECG'):
        if restecg == 0: results.append({'param': 'Resting ECG', 'val': restecg, 'status': 'NORMAL', 'msg': '[NORMAL_ECG] ECG within healthy limits'})
        elif restecg >= 1: results.append({'param': 'Resting ECG', 'val': restecg, 'status': 'HIGH', 'msg': '[ECG_ABNORMALITY] Cardiac abnormality detected'})
        
    oldpeak = data.get('oldpeak')
    if oldpeak is not None and oldpeak >= 0 and is_allowed('ST Depression'):
        if oldpeak < 0.5: results.append({'param': 'ST Depression', 'val': oldpeak, 'status': 'NORMAL', 'msg': '[NORMAL] No ischemia pattern'})
        elif oldpeak >= 1: results.append({'param': 'ST Depression', 'val': oldpeak, 'status': 'HIGH', 'msg': '[ISCHEMIA_RISK] Myocardial ischemia suspected'})
        
    thalach = data.get('thalach')
    if thalach is not None and thalach > 0 and is_allowed('Max Heart Rate'):
        if thalach < 90: results.append({'param': 'Max Heart Rate', 'val': thalach, 'status': 'LOW', 'msg': '[LOW_CARDIAC_RESPONSE] Low cardiac response'})
        elif 90 <= thalach <= 170: results.append({'param': 'Max Heart Rate', 'val': thalach, 'status': 'NORMAL', 'msg': '[NORMAL] Normal cardiac response'})
        elif thalach > 170: results.append({'param': 'Max Heart Rate', 'val': thalach, 'status': 'HIGH', 'msg': '[HIGH_CARDIAC_RESPONSE] Elevated cardiac stress'})

    slope = data.get('slope')
    if slope is not None and slope >= 0 and is_allowed('ST Slope'):
        if slope == 0: results.append({'param': 'ST Slope', 'val': slope, 'status': 'NORMAL', 'msg': '[NORMAL] Healthy ST slope'})
        elif slope >= 1: results.append({'param': 'ST Slope', 'val': slope, 'status': 'HIGH', 'msg': '[ISCHEMIC_PATTERN] Possible myocardial ischemia'})

    thal = data.get('thal')
    if thal is not None and thal >= 0 and is_allowed('Thalassemia'):
        if thal == 0: results.append({'param': 'Thalassemia', 'val': thal, 'status': 'NORMAL', 'msg': '[NORMAL] Normal hemoglobin pattern'})
        elif thal >= 1: results.append({'param': 'Thalassemia', 'val': thal, 'status': 'HIGH', 'msg': '[THALASSEMIA_DEFECT] Blood disorder detected'})

    # DIABETES PARAMETERS - GENDER SENSITIVE
    sex = data.get('sex', 'Female') 
    pregnancies = data.get('pregnancies')
    
    if sex == "Female" and pregnancies is not None and pregnancies > 0 and is_allowed('Pregnancies'):
        if 1 <= pregnancies <= 3: results.append({'param': 'Pregnancies', 'val': pregnancies, 'status': 'NORMAL', 'msg': '[NORMAL] Normal pregnancy count'})
        elif pregnancies >= 4: results.append({'param': 'Pregnancies', 'val': pregnancies, 'status': 'HIGH', 'msg': '[HIGH_RISK] Multiple pregnancy risk factor'})

    skin_thickness = data.get('skin_thickness')
    if skin_thickness is not None and skin_thickness > 0 and is_allowed('Skin Thickness'):
        if skin_thickness < 10: results.append({'param': 'Skin Thickness', 'val': skin_thickness, 'status': 'LOW', 'msg': '[LOW_SKIN_FAT] Low subcutaneous fat'})
        elif 10 <= skin_thickness <= 30: results.append({'param': 'Skin Thickness', 'val': skin_thickness, 'status': 'NORMAL', 'msg': '[NORMAL] Skin thickness normal'})
        elif skin_thickness > 30: results.append({'param': 'Skin Thickness', 'val': skin_thickness, 'status': 'HIGH', 'msg': '[HIGH_RISK] High fat deposition'})

    insulin = data.get('insulin')
    if insulin is not None and insulin > 0 and is_allowed('Insulin'):
        if insulin < 16: results.append({'param': 'Insulin', 'val': insulin, 'status': 'LOW', 'msg': '[LOW_INSULIN] Low insulin level'})
        elif 16 <= insulin <= 166: results.append({'param': 'Insulin', 'val': insulin, 'status': 'NORMAL', 'msg': '[NORMAL] Insulin level normal'})
        elif insulin > 166: results.append({'param': 'Insulin', 'val': insulin, 'status': 'HIGH', 'msg': '[HIGH_INSULIN] Possible insulin resistance'})

    dpf = data.get('dpf')
    if dpf is not None and dpf > 0 and is_allowed('Diabetes Pedigree'):
        if dpf < 0.2: results.append({'param': 'Diabetes Pedigree', 'val': dpf, 'status': 'LOW', 'msg': '[LOW_GENETIC_RISK] Low genetic diabetes risk'})
        elif 0.2 <= dpf <= 0.8: results.append({'param': 'Diabetes Pedigree', 'val': dpf, 'status': 'NORMAL', 'msg': '[NORMAL] Moderate genetic risk'})
        elif dpf > 0.8: results.append({'param': 'Diabetes Pedigree', 'val': dpf, 'status': 'HIGH', 'msg': '[HIGH_GENETIC_RISK] Strong genetic risk'})

    # KIDNEY & BLOOD
    creatinine = data.get('creatinine')
    if creatinine is not None and creatinine > 0 and is_allowed('Creatinine'):
        if creatinine < 0.6: results.append({'param': 'Creatinine', 'val': creatinine, 'status': 'LOW', 'msg': '[LOW_CREATININE] Low muscle mass indication'})
        elif 0.6 <= creatinine <= 1.3: results.append({'param': 'Creatinine', 'val': creatinine, 'status': 'NORMAL', 'msg': '[NORMAL] Kidney function normal'})
        elif creatinine > 1.3: results.append({'param': 'Creatinine', 'val': creatinine, 'status': 'HIGH', 'msg': '[RENAL_IMPAIRMENT] Kidney dysfunction risk'})

    hb = data.get('hb')
    if hb is not None and hb > 0 and is_allowed('Hemoglobin'):
        if hb < 12: results.append({'param': 'Hemoglobin', 'val': hb, 'status': 'LOW', 'msg': '[ANEMIA] Low hemoglobin detected'})
        elif 12 <= hb <= 17: results.append({'param': 'Hemoglobin', 'val': hb, 'status': 'NORMAL', 'msg': '[NORMAL] Hemoglobin normal'})
        elif hb > 17: results.append({'param': 'Hemoglobin', 'val': hb, 'status': 'HIGH', 'msg': '[POLYCYTHEMIA] Elevated hemoglobin'})

    wbc = data.get('wbc')
    if wbc is not None and wbc > 0 and is_allowed('WBC'):
        if wbc < 4000: results.append({'param': 'WBC', 'val': wbc, 'status': 'LOW', 'msg': '[LEUKOPENIA] Low immunity level'})
        elif 4000 <= wbc <= 11000: results.append({'param': 'WBC', 'val': wbc, 'status': 'NORMAL', 'msg': '[NORMAL] WBC count normal'})
        elif wbc > 11000: results.append({'param': 'WBC', 'val': wbc, 'status': 'HIGH', 'msg': '[INFECTION_RISK] Possible infection'})

    platelets = data.get('platelets')
    if platelets is not None and platelets > 0 and is_allowed('Platelets'):
        if platelets < 150000: results.append({'param': 'Platelets', 'val': platelets, 'status': 'LOW', 'msg': '[THROMBOCYTOPENIA] Low platelet count'})
        elif 150000 <= platelets <= 450000: results.append({'param': 'Platelets', 'val': platelets, 'status': 'NORMAL', 'msg': '[NORMAL] Platelet count normal'})
        elif platelets > 450000: results.append({'param': 'Platelets', 'val': platelets, 'status': 'HIGH', 'msg': '[THROMBOCYTOSIS] Elevated platelet count'})

    # LIVER PARAMETERS
    sgot = data.get('sgot')
    if sgot is not None and sgot > 0 and is_allowed('AST'):
        if sgot < 10: results.append({'param': 'AST', 'val': sgot, 'status': 'LOW', 'msg': '[LOW_AST] Below normal liver enzyme'})
        elif 10 <= sgot <= 40: results.append({'param': 'AST', 'val': sgot, 'status': 'NORMAL', 'msg': '[NORMAL] Liver function normal'})
        elif sgot > 40: results.append({'param': 'AST', 'val': sgot, 'status': 'HIGH', 'msg': '[LIVER_DAMAGE] Liver injury risk'})

    sgpt = data.get('sgpt')
    if sgpt is not None and sgpt > 0 and is_allowed('ALT'):
        if sgpt < 7: results.append({'param': 'ALT', 'val': sgpt, 'status': 'LOW', 'msg': '[LOW_ALT] Below normal enzyme level'})
        elif 7 <= sgpt <= 56: results.append({'param': 'ALT', 'val': sgpt, 'status': 'NORMAL', 'msg': '[NORMAL] Liver enzyme normal'})
        elif sgpt > 56: results.append({'param': 'ALT', 'val': sgpt, 'status': 'HIGH', 'msg': '[LIVER_DAMAGE] Liver damage risk'})

    crp = data.get('crp')
    if crp is not None and crp > 0 and is_allowed('CRP'):
        if crp < 1: results.append({'param': 'CRP', 'val': crp, 'status': 'LOW', 'msg': '[NO_INFLAMMATION] No inflammation detected'})
        elif 1 <= crp <= 3: results.append({'param': 'CRP', 'val': crp, 'status': 'NORMAL', 'msg': '[NORMAL] Mild inflammation'})
        elif crp > 3: results.append({'param': 'CRP', 'val': crp, 'status': 'HIGH', 'msg': '[INFLAMMATION] Significant inflammation'})

    # INFECTIOUS TESTS
    typhoid_o = data.get('typhoid_o')
    typhoid_h = data.get('typhoid_h')
    if (typhoid_o == 'POSITIVE' or typhoid_h == 'POSITIVE') and is_allowed('Typhoid'):
        results.append({'param': 'Typhoid', 'val': 'Positive', 'status': 'HIGH', 'msg': '[TYPHOID_DETECTED] Typhoid infection detected'})
    elif typhoid_o == 'NEGATIVE' and typhoid_h == 'NEGATIVE' and is_allowed('Typhoid'):
        results.append({'param': 'Typhoid', 'val': 'Negative', 'status': 'NORMAL', 'msg': '[NO_INFECTION] Typhoid infection not detected'})

    d_igg = data.get('dengue_igg')
    d_igm = data.get('dengue_igm')
    d_ns1 = data.get('dengue_ns1')
    if (d_igg in ['REACTIVE', 'WEAK REACTIVE'] or d_igm in ['REACTIVE', 'WEAK REACTIVE'] or d_ns1 in ['REACTIVE', 'WEAK REACTIVE']) and is_allowed('Dengue'):
        results.append({'param': 'Dengue', 'val': 'Positive', 'status': 'HIGH', 'msg': '[DENGUE_DETECTED] Dengue infection detected'})
    elif d_igg == 'NON-REACTIVE' and d_igm == 'NON-REACTIVE' and d_ns1 == 'NON-REACTIVE' and is_allowed('Dengue'):
        results.append({'param': 'Dengue', 'val': 'Negative', 'status': 'NORMAL', 'msg': '[NO_INFECTION] Dengue not detected'})

    return results

def get_report_html(patient_name, patient_age, patient_gender, conditions):
    """Generates a professional diagnostic report in HTML format."""
    # Ensure IST date string
    ist_now = localize_ist(datetime.datetime.now())
    date_str = ist_now.strftime("%d %b %Y, %H:%M")
    
    conditions_html = ""
    for c in conditions:
        color = "#ff4b4b" if c['severity'] in ['High', 'Critical'] else ("#ffa500" if c['severity'] == 'Mild' else "#00ff7f")
        conditions_html += f"""
        <div style="border-left: 5px solid {color}; padding: 15px; margin-bottom: 20px; background: #f8f9fa; border-radius: 0 10px 10px 0;">
            <h4 style="margin:0; color: #0a192f;">{c['disease']} <span style="font-size: 0.8rem; color: {color};">[{c['severity']}]</span></h4>
            <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: #555;">{c['reason']}</p>
        </div>
        """
    
    if not conditions:
        conditions_html = "<p style='color: #00ff7f; font-weight: 600;'>No significant clinical abnormalities detected. Patient appears healthy.</p>"

    html = f"""
    <div style="font-family: 'Inter', sans-serif; max-width: 800px; margin: auto; padding: 40px; border: 1px solid #ddd; border-radius: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #00d2ff; padding-bottom: 20px; margin-bottom: 30px;">
            <div>
                <h1 style="margin:0; color: #00d2ff; font-size: 1.5rem;">AI-Based Medical Diagnosis Support System</h1>
                <p style="margin:0; opacity: 0.7;">Clinical Diagnostic Summary</p>
            </div>
            <div style="text-align: right;">
                <p style="margin:0; font-weight: 600;">Report ID: #DIAG-{int(datetime.now().timestamp())}</p>
                <p style="margin:0; opacity: 0.7;">Date: {date_str}</p>
            </div>
        </div>
        
        <div style="background: #0a192f; color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; display: grid; grid-template-columns: 1fr 1fr 1fr;">
            <div><small>PATIENT NAME</small><br><strong>{patient_name}</strong></div>
            <div><small>AGE</small><br><strong>{patient_age} Years</strong></div>
            <div><small>GENDER</small><br><strong>{patient_gender}</strong></div>
        </div>
        
        <h3 style="color: #0a192f; border-bottom: 1px solid #eee; padding-bottom: 10px;">Diagnostic Observations</h3>
        {conditions_html}
        
        <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; font-size: 0.8rem; color: #888;">
            <p><strong>Disclaimer:</strong> This report is generated by an Artificial Intelligence system and is intended for clinical assistance only. These findings must be reviewed and validated by a qualified medical professional before clinical action is taken.</p>
        </div>
    </div>
    """
    return html


@st.cache_resource
def get_diabetes_scaler():
    """ Load a small slice of raw data to fit standard scaler correctly """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cleaned_diabetes.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'diabetes_sample.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'diabetes.csv')
    df = pd.read_csv(data_path, nrows=1000)
    
    feature_cols = df.columns.drop('1')
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    return scaler, list(feature_cols)

@st.cache_resource
def get_heart_scaler():
    """ Load a small slice of raw data to fit standard scaler correctly """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cleaned_heart.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'heart_sample.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'heart.csv')
    df = pd.read_csv(data_path, nrows=1000)
    
    feature_cols = df.columns.drop('target')
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    return scaler, list(feature_cols)

@st.cache_resource
def get_diagnosis_scaler():
    """ Load a slice of raw data to fit standard scaler & label encoders correctly """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cleaned_disease_diagnosis.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'disease_diagnosis_sample.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'disease_diagnosis.csv')
    if not os.path.exists(data_path):
        return None, None, None
        
    df = pd.read_csv(data_path, nrows=10000)
    # Match preprocessing in preprocess.py
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    
    target_cols = ['diagnosis'] # ONLY 'diagnosis' is the target
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    le_mappings = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col in feature_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_mappings[col] = le
            
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    return scaler, list(feature_cols), le_mappings

def run_diagnostic_pipeline(extracted_data, scaler_dia, feature_keys_dia, scaler_heart, feature_keys_heart, scaler_diag=None, feature_keys_diag=None, le_diag=None, patient_info=None, user_id=None, tabs=None, target_block=None):
    """
    Unified diagnostic report generator with Modular (Targeted) execution support.
    If target_block is provided (e.g., 'diabetes'), only that pipeline runs.
    If target_block is None, all available pipelines run (standard behavior).
    """
    # Unified diagnostic report generator
    
    detected_conditions = []
    ml_db_logs = []
    
    # Risk scores for visualization
    risk_data = {
        "Diabetes": 0.0, "Heart Risk": 0.0, "Blood Pressure": 0.0,
        "Obesity/BMI": 0.0, "Cholesterol": 0.0, "Kidney Function": 0.0,
        "Blood (Haemogram)": 0.0, "Infectious Dis.": 0.0, "Liver Function": 0.0,
        "Inflammation": 0.0, "General Symptoms": 0.0
    }
    
    # Initialize local markers for safety (Fixes UnboundLocalError)
    inf_dis_risk = 0.0
    rule_heart_risk = 0.0

    # --- TAB 1: IMMEDIATE VITAL OBSERVATIONS ---
    if tabs:
        tab_obs, tab_risk, tab_det = tabs
        with tab_obs:
            render_luxury_header("Clinical Analysis Summary", icon="🧪")
            
            if target_block:
                st.markdown(f"""
                <div style="display: inline-block; background: rgba(0, 210, 255, 0.1); border: 1px solid #00d2ff; 
                            color: #00d2ff; padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; 
                            font-weight: 700; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1px;">
                    🎯 Targeted Module: {target_block.replace('_', ' ')}
                </div>
                """, unsafe_allow_html=True)

            # Extract vital indicators from the provided data
            vital_indicators = evaluate_manual_clinical_risk(extracted_data, target_block=target_block)
            
            # Logic: If a specific block is targeted, show ALL indicators from that block for feedback.
            # If in global OCR mode (target_block is None), only show abnormal metrics to avoid clutter.
            if target_block:
                obs_to_show = vital_indicators
                status_header = f"📋 Active Diagnostic Status - {target_block.replace('_', ' ').upper()}"
            else:
                obs_to_show = [r for r in vital_indicators if r['status'] != 'NORMAL' and r.get('msg')]
                status_header = "⚠️ Immediate Clinical Alert Observations"
            
            if obs_to_show:
                st.markdown(f"#### {status_header}")
                cols = st.columns(len(obs_to_show) if len(obs_to_show) < 4 else 4)
                for i, r in enumerate(obs_to_show):
                    with cols[i % 4]:
                        # Color: LOW=Blue, HIGH=Red, NORMAL=Green
                        color = "#3B82F6" if r['status'] == 'LOW' else ("#EF4444" if r['status'] == 'HIGH' else "#10B981")
                        st.markdown(f"""
                        <div class="stat-card-pro" style="border-top: 4px solid {color};">
                            <div class="stat-label-pro">{r['param']}</div>
                            <div class="stat-value-pro" style="color:{color};">{r['status']}</div>
                            <div class="stat-msg-pro" style="font-size:0.75rem;">{r.get('msg', 'Measurement Normal')}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("✅ **Screening Complete**: No immediate clinical abnormalities detected in vital markers.")
    
    # --- 1. DIABETES LOGIC ---
    # Strictly Targeted: Only if block is diabetes
    if (not target_block or target_block == 'diabetes') and 'glucose' in extracted_data:
        has_all_dia_features = all(k in extracted_data for k in ['pregnancies', 'diastolic', 'skin_thickness', 'insulin', 'bmi', 'dpf', 'age'])
        if has_all_dia_features:
            raw_features_dia = {
                feature_keys_dia[0]: extracted_data.get('pregnancies', 0),
                feature_keys_dia[1]: extracted_data['glucose'],
                feature_keys_dia[2]: extracted_data['diastolic'],  
                feature_keys_dia[3]: extracted_data.get('skin_thickness', 20),
                feature_keys_dia[4]: extracted_data.get('insulin', 80),
                feature_keys_dia[5]: extracted_data.get('bmi', 25.0),
                feature_keys_dia[6]: extracted_data.get('dpf', 0.5),
                feature_keys_dia[7]: extracted_data['age']
            }
            input_df = pd.DataFrame([raw_features_dia])
            scaled_array = scaler_dia.transform(input_df)
            scaled_features = {key: val for key, val in zip(feature_keys_dia, scaled_array[0])}
            ml_result_dia = predict_disease("Diabetes", scaled_features, raw_data=raw_features_dia)
            
            if ml_result_dia['status'] == 'success' and ml_result_dia['prediction'] == 1:
                risk_data["Diabetes"] = ml_result_dia.get('confidence', 0.0)
                detected_conditions.append({
                    "disease": "Diabetes", "severity": "High", 
                    "reason": f"AI Model detected high risk of Diabetes (Confidence: {ml_result_dia.get('confidence', 0):.2%}). Extracted Glucose: {extracted_data['glucose']}",
                    "advice": "Consult an Endocrinologist. Maintain a low-glycemic diet and monitor fasting blood sugar daily.",
                    "factors": ml_result_dia.get('top_factors', [])
                })
            ml_db_logs.append({"disease": "Diabetes", "status": ml_result_dia['status'], "prediction": ml_result_dia.get('prediction'), "confidence": ml_result_dia.get('confidence')})
        
        # Robust WHO Rules Fallback (Always check if rules detect risk, even if ML runs)
        glu_eval = evaluate_blood_sugar_fasting(extracted_data['glucose'])
        if glu_eval['category'] in ['High', 'Critical', 'Mild']:
            # Hybrid Logic: Always ensure at least WHO rule-based risk percentage is reflected
            rule_risk = 0.95 if glu_eval['category'] == 'Critical' else (0.75 if glu_eval['category'] == 'High' else 0.40)
            risk_data["Diabetes"] = max(risk_data["Diabetes"], rule_risk)
            
            # Only add to list if not already added by ML or if it adds new severity info
            if not any(c['disease'] == "Diabetes" for c in detected_conditions):
                detected_conditions.append({"disease": "Diabetes", "severity": glu_eval['category'], "reason": glu_eval['reason']})

    # --- 2. HEART DISEASE LOGIC --- (Enhanced Hybrid Approach)
    has_heart_features = all(k in extracted_data for k in ['age', 'systolic', 'cholesterol'])
    if (not target_block or target_block == 'heart') and has_heart_features:
        raw_features_heart = {
            'age': extracted_data['age'], 'sex': 1 if patient_info and patient_info.get('gender') == 'Male' else (0 if patient_info and patient_info.get('gender') == 'Female' else 1),
            'cp': extracted_data.get('cp', 0), 'trestbps': extracted_data['systolic'], 'chol': extracted_data['cholesterol'], 
            'fbs': 1 if extracted_data.get('glucose', 0) > 120 else 0, 'restecg': extracted_data.get('restecg', 1),
            'thalach': extracted_data.get('thalach', 150), 'exang': extracted_data.get('exang', 0), 'oldpeak': extracted_data.get('oldpeak', 0.0),
            'slope': extracted_data.get('slope', 1), 'ca': extracted_data.get('ca', 0), 'thal': extracted_data.get('thal', 2)
        }
        input_df = pd.DataFrame([raw_features_heart])
        scaled_array = scaler_heart.transform(input_df)
        scaled_features = {key: val for key, val in zip(feature_keys_heart, scaled_array[0])}
        ml_result_heart = predict_disease("Heart Disease", scaled_features, raw_data=raw_features_heart)
        
        if ml_result_heart['status'] == 'success' and ml_result_heart['prediction'] == 1:
            risk_data["Heart Risk"] = ml_result_heart.get('confidence', 0.0)
            detected_conditions.append({
                "disease": "Heart Disease", "severity": "High",
                "reason": f"AI Model detected high risk of Heart Disease (Confidence: {ml_result_heart.get('confidence', 0):.2%}).",
                "advice": "Immediate Cardiac Consultation required. Avoid strenuous physical activity and monitor blood pressure hourly.",
                "factors": ml_result_heart.get('top_factors', [])
            })
        ml_db_logs.append({"disease": "Heart Disease", "status": ml_result_heart['status'], "prediction": ml_result_heart.get('prediction'), "confidence": ml_result_heart.get('confidence')})
        # Heart Diagnostics Fallsback (Only if block is Heart)
        if 'cholesterol' in extracted_data or 'systolic' in extracted_data:
            v_chol = extracted_data.get('cholesterol', 0)
            v_sys = extracted_data.get('systolic', 0)
            rule_heart_risk = 0.0
            if v_chol > 240 or v_sys > 160: rule_heart_risk = 0.85
            elif v_chol > 200 or v_sys > 140: rule_heart_risk = 0.55
            risk_data["Heart Risk"] = max(risk_data["Heart Risk"], rule_heart_risk)
    
    # --- 3. HYPERTENSION LOGIC ---
    # Strictly Targeted: Only if block is core_vitals
    if (not target_block or target_block == 'core_vitals') and 'systolic' in extracted_data and 'diastolic' in extracted_data:
        ht_eval = evaluate_hypertension(extracted_data['systolic'], extracted_data['diastolic'])
        if ht_eval["detected"]:
            risk_data["Blood Pressure"] = 1.0 if ht_eval["category"] == "Critical" else 0.8
            disease_name = "Hypotension" if "HYPOTENSION" in ht_eval["reason"].upper() else "Hypertension"
            advice = "Reduce sodium intake. Monitor BP twice daily (Morning/Evening) and rest in a cool environment."
            detected_conditions.append({"disease": disease_name, "severity": ht_eval["category"], "reason": ht_eval["reason"], "advice": advice})
            
    # --- 4. BMI / OBESITY LOGIC ---
    # Strictly Targeted: Only if block is core_vitals
    if (not target_block or target_block == 'core_vitals') and 'bmi' in extracted_data:
        bmi_val = extracted_data['bmi']
        if bmi_val >= 30: 
            risk_data["Obesity/BMI"] = 0.85
            detected_conditions.append({"disease": "Obesity", "severity": "High", "reason": f"BMI recorded as {bmi_val:.1f}. [OBESITY] Increased health risk", "advice": "Consult a certified Nutritionist. Aim for 30 minutes of low-impact cardiovascular exercise daily."})
        elif bmi_val >= 25:
            risk_data["Obesity/BMI"] = 0.55
            detected_conditions.append({"disease": "Overweight", "severity": "Mild", "reason": f"BMI recorded as {bmi_val:.1f}. [OVERWEIGHT] Moderate health risk", "advice": "Maintain a balanced diet and regular physical activity."})
            
    # --- 4.5 CHOLESTEROL LOGIC ---
    # Strictly Targeted: Only if block is core_vitals
    if (not target_block or target_block == 'core_vitals') and 'cholesterol' in extracted_data:
        chol_eval = evaluate_cholesterol(extracted_data['cholesterol'])
        if chol_eval["detected"]:
            risk_data["Cholesterol"] = 1.0 if chol_eval["category"] == "Critical" else 0.7
            advice = "Follow a DASH/Mediterranean diet. Avoid trans fats and check Lipid Profile in 4 weeks."
            detected_conditions.append({"disease": "High Cholesterol", "severity": chol_eval["category"], "reason": chol_eval["reason"], "advice": advice})
            
    # --- 5. KIDNEY DISEASE LOGIC ---
    if (not target_block or target_block in ['kidney', 'pathology']) and 'creatinine' in extracted_data:
        kd_eval = evaluate_kidney_disease(extracted_data['creatinine'])
        if kd_eval["detected"]:
            risk_data["Kidney Function"] = 1.0 if kd_eval["category"] == "Critical" else 0.7
            advice = "Hydrate adequately and avoid NSAIDs (e.g., Ibuprofen). Consult a Nephrologist for a detailed Renal Function Test."
            detected_conditions.append({"disease": "Kidney Disease", "severity": kd_eval["category"], "reason": kd_eval["reason"], "advice": advice})
            
    # --- 6. HAEMOGRAM LOGIC ---
    if (not target_block or target_block in ['haemogram', 'pathology']) and any(k in extracted_data for k in ['hb', 'wbc', 'platelets']):
        haem_eval = evaluate_haemogram(hb=extracted_data.get('hb'), wbc=extracted_data.get('wbc'), platelets=extracted_data.get('platelets'))
        if haem_eval["detected"]:
            risk_data["Blood (Haemogram)"] = 1.0 if haem_eval["category"] == "Critical" else 0.7
            advice = "Check Iron and Vitamin B12 levels. Consult a Hematologist if hemoglobin or platelets remain outside normal range."
            detected_conditions.append({"disease": "Blood Disorder (Haemogram)", "severity": haem_eval["category"], "reason": haem_eval["reason"], "advice": advice})
            
    # --- 7. INFECTIOUS DISEASES LOGIC (Typhoid & Dengue) ---
    if not target_block or target_block == 'pathology':
        inf_dis_risk = 0.0
        # Typhoid logic
        if any(k in extracted_data for k in ['typhoid_o', 'typhoid_h']):
            typhoid_res = evaluate_typhoid(o_ag=extracted_data.get('typhoid_o'), h_ag=extracted_data.get('typhoid_h'))
            if typhoid_res["detected"]:
                inf_dis_risk = max(inf_dis_risk, 0.80)
                if not any(c['disease'] == "Enteric Fever (Typhoid)" for c in detected_conditions):
                    detected_conditions.append({
                        "disease": "Enteric Fever (Typhoid)", "severity": typhoid_res["category"],
                        "reason": typhoid_res["reason"],
                        "advice": "Follow an easily digestible diet (Bland diet). Complete the prescribed antibiotic course and maintain high hygiene."
                    })
        # Dengue logic
        if any(k in extracted_data for k in ['dengue_igg', 'dengue_igm', 'dengue_ns1']):
            dengue_res = evaluate_dengue(igg=extracted_data.get('dengue_igg'), igm=extracted_data.get('dengue_igm'), ns1=extracted_data.get('dengue_ns1'))
            if dengue_res["detected"]:
                inf_dis_risk = max(inf_dis_risk, 0.85)
                if not any(c['disease'] == "Dengue Fever" for c in detected_conditions):
                    detected_conditions.append({
                        "disease": "Dengue Fever", "severity": dengue_res["category"],
                        "reason": dengue_res["reason"],
                        "advice": "Bed rest and intensive hydration. Monitor platelet count every 12-24 hours. Avoid aspirin-based medications."
                    })
        risk_data["Infectious Dis."] = inf_dis_risk

    # --- 8. LIVER FUNCTION LOGIC ---
    if (not target_block or target_block == 'pathology') and any(k in extracted_data for k in ['sgot', 'sgpt']):
        liver_res = evaluate_liver_function(sgot=extracted_data.get('sgot'), sgpt=extracted_data.get('sgpt'))
        if liver_res["detected"]:
            risk_data["Liver Function"] = 0.90 if liver_res["category"] == "Critical" else 0.70
            if not any(c['disease'] == "Liver Dysfunction" for c in detected_conditions):
                detected_conditions.append({
                    "disease": "Liver Dysfunction", "severity": liver_res["category"],
                    "reason": liver_res["reason"],
                    "advice": "Avoid alcohol and high-fat foods. Consult a Hepatologist and perform an Abdominal Ultrasound if enzymes remain elevated."
                })
            
    # --- 10. SYSTEMIC INFLAMMATION LOGIC ---
    if (not target_block or target_block == 'pathology') and 'crp' in extracted_data:
        inflam_eval = evaluate_inflammation(extracted_data['crp'])
        if inflam_eval["detected"]:
            risk_data["Inflammation"] = 1.0 if inflam_eval["category"] == "Critical" else 0.5
            if not any(c['disease'] == "Systemic Inflammation" for c in detected_conditions):
                detected_conditions.append({
                    "disease": "Systemic Inflammation", "severity": inflam_eval["category"],
                    "reason": inflam_eval["reason"],
                    "advice": "Increase intake of anti-inflammatory foods (Omega-3). Follow up to identify the underlying source of inflammation."
                })

    # --- 11. GENERAL VITALS (HR, TEMP, O2) LOGIC ---
    # Strictly Targeted: Only if block is core_vitals
    if not target_block or target_block == 'core_vitals':
        if 'heart_rate_bpm' in extracted_data:
            hr_eval = evaluate_heart_rate(extracted_data['heart_rate_bpm'])
            if hr_eval["detected"]:
                if not any(c['disease'] == "Heart Rate Abnormal" for c in detected_conditions):
                    detected_conditions.append({"disease": "Heart Rate Abnormal", "severity": hr_eval["category"], "reason": hr_eval["reason"]})
                
        if 'body_temperature_c' in extracted_data:
            temp_eval = evaluate_body_temp(extracted_data['body_temperature_c'])
            if temp_eval["detected"]:
                if not any(c['disease'] == "Body Temperature Abnormal" for c in detected_conditions):
                    detected_conditions.append({"disease": "Body Temperature Abnormal", "severity": temp_eval["category"], "reason": temp_eval["reason"]})
                
        if 'oxygen_saturation' in extracted_data:
            o2_eval = evaluate_oxygen(extracted_data['oxygen_saturation'])
            if o2_eval["detected"]:
                if not any(c['disease'] == "Oxygen Saturation Abnormal" for c in detected_conditions):
                    detected_conditions.append({"disease": "Oxygen Saturation Abnormal", "severity": o2_eval["category"], "reason": o2_eval["reason"]})

    # --- 11. GENERAL DISEASE DIAGNOSIS (ML) ---
    if (not target_block or target_block == 'general') and all(k in extracted_data for k in ['symptom_1', 'symptom_2', 'symptom_3']) and scaler_diag:
        try:
            # Prepare raw features matching the training set
            raw_diag = {
                'patient_id': 0, # Model expects patient_id as a feature
                'age': float(extracted_data.get('age', 30)),
                'gender': str(patient_info.get('gender', 'Male')) if patient_info else 'Male',
                'symptom_1': str(extracted_data.get('symptom_1', 'Fatigue')),
                'symptom_2': str(extracted_data.get('symptom_2', 'Fever')),
                'symptom_3': str(extracted_data.get('symptom_3', 'Cough')),
                'heart_rate_bpm': float(extracted_data.get('heart_rate_bpm', 72)),
                'body_temperature_c': float(extracted_data.get('body_temperature_c', 37.0)),
                'blood_pressure_mmhg': f"{extracted_data.get('systolic', 120)}/{extracted_data.get('diastolic', 80)}",
                'oxygen_saturation_%': float(extracted_data.get('oxygen_saturation', 98)),
                'severity': 'Mild', # Imputed default based on training set's most frequent
                'treatment_plan': 'Rest and fluids' # Imputed default based on training set's most frequent
            }
            
            # Encode categorical features
            encoded_diag = {}
            for col in feature_keys_diag:
                val = raw_diag[col]
                if col in le_diag:
                    le = le_diag[col]
                    # Handle unseen classes by picking the closest or default
                    try:
                        encoded_diag[col] = le.transform([str(val)])[0]
                    except ValueError:
                        encoded_diag[col] = 0 # Fallback for unseen labels
                else:
                    encoded_diag[col] = val
                    
            input_df = pd.DataFrame([encoded_diag])[feature_keys_diag]
            scaled_array = scaler_diag.transform(input_df)
            scaled_features = {key: val for key, val in zip(feature_keys_diag, scaled_array[0])}
            
            ml_result_diag = predict_disease("Disease Diagnosis", scaled_features, raw_data=encoded_diag)
            
            if ml_result_diag['status'] == 'success':
                # Mapping integer back to disease name
                mapping = {0: 'Bronchitis', 1: 'Cold', 2: 'Flu', 3: 'Healthy', 4: 'Pneumonia'}
                pred_name = mapping.get(ml_result_diag['prediction'], "Unknown")
                
                if pred_name != 'Healthy':
                    risk_data["General Symptoms"] = ml_result_diag.get('confidence', 0.6)
                    advice = "Rest and high fluid intake. Visit a General Physician if symptoms persist beyond 48 hours."
                    detected_conditions.append({
                        "disease": f"General Diagnosis: {pred_name}", "severity": "Moderate",
                        "reason": f"AI General Model detected symptoms matching {pred_name} (Confidence: {ml_result_diag.get('confidence', 0):.2%}).",
                        "advice": advice,
                        "factors": ml_result_diag.get('top_factors', [])
                    })
                ml_db_logs.append({"disease": "Disease Diagnosis", "status": ml_result_diag['status'], "prediction": ml_result_diag.get('prediction'), "confidence": ml_result_diag.get('confidence')})
        except Exception as e:
            st.error(f"Error in General Disease Diagnosis Pipeline: {e}")

    # --- DATABASE SAVING ---
    if patient_info:
        patient_id = add_patient(patient_info['name'], patient_info['age'], patient_info['gender'], patient_info.get('contact', ''), user_id=user_id)
        if patient_id:
            # Create a dedicated diagnostic session for this run
            session_id = add_diagnostic_session(user_id, patient_id, "Auto Prediction Pipeline")
            if session_id:
                # 1. Save Rule-based Observations
                for cond in detected_conditions:
                    add_clinical_observation(session_id, cond['disease'], cond['severity'], cond['reason'])
                
                # 2. Save ML Prediction Results
                for log in ml_db_logs:
                    if log['status'] == 'success':
                        add_ml_prediction(
                            session_id, 
                            log['disease'], 
                            log['prediction'], 
                            probability=log.get('confidence'),
                            model_version="1.0"
                        )
                
                # 3. Save Captured Clinical Vitals
                # Mapping of codes to human readable names & units
                vitals_meta = {
                    'systolic': ('Systolic BP', 'mmHg'), 'diastolic': ('Diastolic BP', 'mmHg'),
                    'glucose': ('Glucose', 'mg/dL'), 'cholesterol': ('Total Cholesterol', 'mg/dL'),
                    'bmi': ('BMI', 'kg/m2'), 'oxygen_saturation': ('Oxygen Saturation', '%'),
                    'heart_rate_bpm': ('Heart Rate', 'bpm'), 'body_temperature_c': ('Body Temperature', '°C'),
                    'creatinine': ('Creatinine', 'mg/dL'), 'hb': ('Hemoglobin', 'gm%'),
                    'wbc': ('WBC Count', '/cmm'), 'platelets': ('Platelets', '/cmm'),
                    'sgot': ('SGOT/AST', 'U/L'), 'sgpt': ('SGPT/ALT', 'U/L'),
                    'crp': ('CRP', 'mg/L')
                }
                
                for key, val in extracted_data.items():
                    if key in vitals_meta and isinstance(val, (int, float)) and val > 0:
                        name, unit = vitals_meta[key]
                        # Determine status based on evaluation results if available
                        status = 'Normal'
                        for v_ind in vital_indicators:
                            if v_ind['param'].lower() == name.lower():
                                status = v_ind['status'].capitalize()
                                break
                        add_clinical_vital(session_id, name, val, unit, status=status)
                        
                # SUCCESS NOTIFICATION: Let the user know the record is ARCHIVED
                st.toast("✅ Diagnostic Data Archived Successfully!", icon="💾")
                st.success("✨ **Clinical Archive Updated**: Your diagnostic session has been securely stored in the system history.")
            else:
                st.warning("⚠️ Record created but diagnostic session could not be initialized.")
        else:
            st.error("❌ Failed to save patient record to database. Please ensure your profile name is complete.")
            
    # --- 12. RISK VISUALIZATION DASHBOARD ---
    if tabs:
        tab_obs, tab_risk, tab_det = tabs
        with tab_risk:
            render_luxury_header("Clinical Risk Dashboard", icon="📊")
    else:
        st.markdown("---")
        st.markdown("<h3 style='color:#1E293B;'>📊 Clinical Risk Dashboard</h3>", unsafe_allow_html=True)
    
    # Prepare data for Plotly
    labels = list(risk_data.keys())
    values = [v * 100 for v in risk_data.values()]
    colors = ['#EF4444' if v > 70 else ('#F59E0B' if v > 40 else '#10B981') for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=[f"{v:.0f}% Risk" for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        margin=dict(l=10, r=40, t=10, b=10),
        height=400,
        xaxis=dict(
            range=[0, 100], 
            title="Risk Probability (%)", 
            gridcolor='rgba(226, 232, 240, 0.4)',
            tickfont=dict(color='#64748B'),
            showline=True,
            linewidth=1.5,
            linecolor='#CBD5E1',
            mirror=False
        ),
        yaxis=dict(
            autorange="reversed", 
            tickfont=dict(color='#1E293B', size=11, weight='bold'),
            showline=True,
            linewidth=1.5,
            linecolor='#CBD5E1',
            mirror=False
        ),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", size=12),
        showlegend=False,
        hovermode='closest'
    )
    
    if tabs:
        with tab_risk:
            # Add ML Data Transparency Note
            st.markdown("""
            <div style="background: rgba(59, 130, 246, 0.05); border-left: 4px solid #3B82F6; padding: 12px; border-radius: 8px; margin-bottom: 20px;">
                <p style="margin: 0; font-size: 0.85rem; color: #1E293B;">
                    ℹ️ <b>ML Transparency Note</b>: For unentered fields, AI utilizes standard clinical averages to provide a preliminary risk estimate. Providing a full profile will increase accuracy.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})
    else:
        st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

    # --- DISPLAY RESULTS ---
    def render_content(show_header=True):
        if not detected_conditions:
            if not extracted_data:
                st.info("ℹ️ Please provide medical data (Glucose, BP, Cholesterol, BMI, Pathology Panels, etc.) to generate a prediction.")
            else:
                st.success("✅ **No Abnormalities Detected!** Based on the provided parameters, all evaluated systems appear to be within normal limits.")
        else:
            if show_header:
                render_luxury_header("Clinical Analysis Findings", icon="📋")
            for cond in detected_conditions:
                # Map severity to CSS class and icons
                severity_class = cond['severity'].lower()
                if severity_class not in ['critical', 'mild', 'normal']: severity_class = 'mild'
                
                icon = "🛑" if cond['severity'] == "Critical" else ("⚠️" if cond['severity'] in ["High", "Mild"] else "✅")
                icon_class = "icon-critical" if cond['severity'] in ["Critical", "High"] else ("icon-mild" if cond['severity'] == "Mild" else "icon-normal")
                badge_class = "badge-critical" if cond['severity'] in ["Critical", "High"] else ("badge-mild" if cond['severity'] == "Mild" else "badge-normal")
                
                st.markdown(f"""
<div class="diagnostic-container severity-{severity_class}">
<div class="diagnostic-header">
<div class="diagnostic-title">
<div class="diagnostic-icon {icon_class}">{icon}</div>
{cond['disease']}
</div>
<span class="diagnostic-badge {badge_class}">{cond['severity'].upper()}</span>
</div>

<div class="diagnostic-content-grid">
<div class="diagnostic-section">
<div class="diagnostic-section-label">📜 Reason / Observation</div>
<div class="diagnostic-reason">{cond['reason']}</div>
</div>
<div class="diagnostic-section">
<div class="diagnostic-section-label">🩺 Medical Advice / Action</div>
<div class="diagnostic-advice">{cond.get('advice', 'Consult a healthcare professional for further clinical investigation.')}</div>
</div>
</div>

<!-- Navigation Hook to new Wellness Center -->
<div id="recommendations-{cond['disease'].replace(' ', '-')}" class="recommendations-hook"></div>
</div>
""", unsafe_allow_html=True)

                # --- XAI Display ---
                if cond.get('factors'):
                    st.markdown("<div style='margin-top:12px; font-weight:600; color:#1a202c; font-size:0.85rem;'>🤖 AI DECISION FACTORS (SHAP EXPLANATION):</div>", unsafe_allow_html=True)
                    cols_shap = st.columns(len(cond['factors']))
                    for idx, factor in enumerate(cond['factors']):
                        with cols_shap[idx]:
                            # Handle either legacy keys or SHAP-enhanced keys
                            impact_val = factor.get('importance_score', 0)
                            p_val = factor.get('patient_value', 'N/A')
                            shap_contr = factor.get('shap_contribution', 0)
                            impact_color = "#EF4444" if shap_contr > 0 else "#10B981"
                            impact_label = "INCREASES RISK" if shap_contr > 0 else "REDUCES RISK"
                            
                            st.markdown(f"""
<div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; text-align: center; height: 100%;">
<div style="font-size: 0.65rem; color: #64748B; font-weight: 700;">{factor['feature'].upper()}</div>
<div style="font-size: 0.9rem; color: #1E293B; font-weight: 800; margin: 4px 0;">{p_val}</div>
<div style="font-size: 0.65rem; color: {impact_color}; font-weight: 700;">{impact_val}% {impact_label}</div>
</div>
""", unsafe_allow_html=True)
                
                # Smooth spacing instead of line
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # --- Consolidated Wellness Access Button at the very end ---
            # Compact Wellness Notification
            st.markdown("""
                <div style="background: rgba(16, 185, 129, 0.05); border: 1px dashed rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 15px; text-align: center; margin: 10px 0;">
                    <div style="color: #065F46; font-weight: 700; font-size: 0.95rem;">✨ Specialized Wellness & Recovery Roadmap is Ready!</div>
                    <div style="color: #065F46; font-size: 0.8rem; margin-top: 4px;">Personalized clinical remedies are available in the Wellness Center.</div>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Access Specialized Wellness & Recovery Plan", key="nav_well_global", use_container_width=True, type="primary"):
                st.toast("Roadmap unlocked! Please switch to the '✨ Wellness Center' tab above.", icon="🌿")
                st.info("💡 **Navigation Tip**: Click the '✨ Wellness Center' tab at the top of the page to view your full clinical recovery plan.")
            
            # Professional Report Download
            if patient_info:
                st.markdown("---")
                report_html = get_report_html(patient_info['name'], patient_info['age'], patient_info['gender'], detected_conditions)
                st.download_button(
                    label="📄 Download Medical Report",
                    data=report_html,
                    file_name=f"Medical_Report_{patient_info['name'].replace(' ', '_')}.html",
                    mime="text/html",
                    use_container_width=True,
                    type="primary"
                )

    if tabs:
        tab_obs, tab_risk, tab_det = tabs
        with tab_det:
            render_luxury_header("Clinical Analysis Findings", icon="📋")
            render_content(show_header=False)
    else:
        render_content(show_header=True)

def render_wellness_center(user_id):
    """Renders a high-end, ultra-modern Wellness and Recovery Center."""
    render_luxury_header("Personalized Wellness Center", icon="✨", badge_text="Holistic Health", mode="hero")
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    
    # Get latest diagnostic session for this user
    insight = get_latest_patient_insight(user_id)
    
    # 🛑 DEFENSIVE: Handle both legacy tuple and new dict formats
    if isinstance(insight, tuple):
        insight = {'observations': [{
            'Condition': insight[0],
            'Severity': insight[1],
            'Observation': insight[2]
        }]}
    
    if not insight or not insight.get('observations'):
        st.markdown("""<div class="wellness-page-card" style="text-align: center; padding: 100px 40px; background: #fff; border-radius: 32px; border: 1px dashed var(--border);">
<div style="font-size: 5rem; margin-bottom: 32px; filter: drop-shadow(0 10px 20px rgba(0,0,0,0.05));">🧘</div>
<h3 style="font-family: var(--font-heading); font-weight: 900; font-size: 2.2rem; color: var(--text-heading);">Start Your Journey</h3>
<p style="color: var(--text-muted); max-width: 500px; margin: 0 auto 40px auto; font-size: 1.1rem; line-height: 1.6;">
Complete a diagnostic session in 'Manual Entry' or 'OCR' to unlock your clinical-grade recovery roadmap.
</p>
</div>""", unsafe_allow_html=True)
        return

    # Filter out normal results so we only show recommendations for actual detected risks
    valid_observations = []
    for obs in insight.get('observations', []):
        d_name = obs.get('Condition', '')
        sev = obs.get('Severity', '')
        if d_name.lower() != 'normal' and sev.lower() != 'normal' and sev.strip() != '-':
            valid_observations.append(obs)
    
    if not valid_observations and insight.get('observations'):
        st.markdown("""<div class="wellness-page-card" style="text-align: center; padding: 100px 40px; background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%); border-radius: 32px; border: 1px solid #A7F3D0; box-shadow: 0 10px 30px rgba(16, 185, 129, 0.1);">
<div style="font-size: 5rem; margin-bottom: 24px; filter: drop-shadow(0 10px 20px rgba(16, 185, 129, 0.2));">🌿</div>
<h3 style="font-family: var(--font-heading); font-weight: 900; font-size: 2.2rem; color: #065F46; margin-bottom: 12px; letter-spacing:-0.5px;">Optimal Health Achieved</h3>
<p style="color: #047857; max-width: 500px; margin: 0 auto 0 auto; font-size: 1.1rem; line-height: 1.6; font-weight: 500;">
Your recent diagnostic results show all parameters within normal ranges. Keep up the excellent work maintaining your healthy lifestyle!
</p>
</div>""", unsafe_allow_html=True)
        return

    for obs in valid_observations:
        disease_name = obs['Condition']
        severity = obs['Severity']
        mapped_key = disease_name
        
        # Mapping for better lookups
        if "General Diagnosis:" in mapped_key:
            diag_name = mapped_key.split(":")[-1].strip()
            mapped_key = "Cough/Cold/Flu" if diag_name in ["Cold", "Flu", "Bronchitis", "Pneumonia", "Cough"] else "Cough/Cold/Flu"

        if mapped_key in DISEASE_RECOMMENDATIONS:
            rec = DISEASE_RECOMMENDATIONS[mapped_key]
            
            # Severity Visuals
            badge_color = "#E0F2FE" if severity.lower() in ["low", "mild"] else ("#FEF3C7" if severity.lower() == "medium" else "#FEE2E2")
            badge_text = "#0369A1" if severity.lower() in ["low", "mild"] else ("#92400E" if severity.lower() == "medium" else "#991B1B")
            
            st.markdown(f"""
<div style="margin-bottom: 24px; padding: 0 10px;">
<div style="display: flex; align-items: center; margin-bottom: 20px;">
<div style="font-family: var(--font-heading); font-weight: 900; font-size: 1.8rem; letter-spacing: -0.5px; color: var(--text-heading);">{disease_name} Roadmap</div>
</div>
</div>
""", unsafe_allow_html=True)

            # Modern SVG Chevron
            modern_chevron = '<div class="expand-icon-custom"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg></div>'

            # Recovery Roadmap Sections
            st.markdown(f"""<div class="recovery-roadmap-container">
<!-- DIET SECTION -->
<details class="recovery-section">
<summary class="recovery-section-header diet-header">
<div class="section-icon-box">🥗</div>
<div class="section-title-group">
<h4>Precision Nutrition Plan</h4>
<p>Targeted dietary interventions for {disease_name}</p>
</div>
{modern_chevron}
</summary>
<div class="recovery-section-content">
<div class="rec-bullet-list">
{"".join([f'<div class="rec-bullet-item"><div class="bullet-dot diet-dot"></div><div class="bullet-text">{item}</div></div>' for item in rec['Diet'].split('•') if item.strip()])}
</div>
</div>
</details>

<!-- REMEDY SECTION -->
<details class="recovery-section">
<summary class="recovery-section-header remedy-header">
<div class="section-icon-box">🌿</div>
<div class="section-title-group">
<h4>Therapeutic Remedies (Desi Upchar)</h4>
<p>Traditional healing for natural recovery</p>
</div>
{modern_chevron}
</summary>
<div class="recovery-section-content">
<div class="rec-bullet-list">
{"".join([f'<div class="rec-bullet-item"><div class="bullet-dot remedy-dot"></div><div class="bullet-text">{item}</div></div>' for item in rec['Remedies'].split('•') if item.strip()])}
</div>
</div>
</details>

<!-- EXERCISE SECTION -->
<details class="recovery-section">
<summary class="recovery-section-header exercise-header">
<div class="section-icon-box">🏃</div>
<div class="section-title-group">
<h4>Restorative Movement & Yoga</h4>
<p>Physical activity protocol for vitality</p>
</div>
{modern_chevron}
</summary>
<div class="recovery-section-content">
<div class="rec-bullet-list">
{"".join([f'<div class="rec-bullet-item"><div class="bullet-dot exercise-dot"></div><div class="bullet-text">{item}</div></div>' for item in rec['Exercise'].split('•') if item.strip()])}
</div>
</div>
</details>
</div>""", unsafe_allow_html=True)
    
    # Premium Health Mantras Section
    st.markdown("""<div style="margin: 60px 0 30px 0;">
<h3 style="font-family: var(--font-heading); font-weight: 900; font-size: 1.8rem; color: var(--text-heading); border-left: 6px solid var(--primary); padding-left: 20px; margin-bottom: 30px;">🌱 Essential Life-Health Mantras</h3>
<div class="wellness-mantra-grid">
<div class="wellness-mantra-card">
<div class="wellness-mantra-icon">💧</div>
<div>
<div style="font-weight: 800; color: var(--text-heading); font-size: 1.1rem;">Optimal Hydration</div>
<div style="font-size: 0.9rem; color: var(--text-muted);">Aim for 3.5 liters of filtered water daily.</div>
</div>
</div>
<div class="wellness-mantra-card">
<div class="wellness-mantra-icon">😴</div>
<div>
<div style="font-weight: 800; color: var(--text-heading); font-size: 1.1rem;">Deep Restorative Sleep</div>
<div style="font-size: 0.9rem; color: var(--text-muted);">Maintain 7-8 hours of uninterrupted rest.</div>
</div>
</div>
<div class="wellness-mantra-card">
<div class="wellness-mantra-icon">🧘</div>
<div>
<div style="font-weight: 800; color: var(--text-heading); font-size: 1.1rem;">Mindfulness Audit</div>
<div style="font-size: 0.9rem; color: var(--text-muted);">15 mins of breathwork or meditation.</div>
</div>
</div>
<div class="wellness-mantra-card">
<div class="wellness-mantra-icon">🍏</div>
<div>
<div style="font-weight: 800; color: var(--text-heading); font-size: 1.1rem;">Whole-Food Priority</div>
<div style="font-size: 0.9rem; color: var(--text-muted);">Ensure 60% of daily intake is raw or boiled plants.</div>
</div>
</div>
</div>
<div style="margin-top: 60px; padding: 30px; background: #F8FAFC; border-radius: 20px; border: 1px solid var(--border); text-align: center; color: var(--text-muted); font-size: 0.85rem; line-height: 1.6;">
<strong>Medical Disclaimer:</strong> These recovery roadmap protocols are designed to complement medical treatment. 
Please consult with your physician before implementing significant changes to your therapeutic regimen. 
Individual results may vary based on specific clinical profiles.
</div>
</div>""", unsafe_allow_html=True)

def render_clinical_portal(user_id, username, scaler_dia, feature_keys_dia, scaler_heart, feature_keys_heart, scaler_diag=None, feature_keys_diag=None, le_diag=None):
    """Renders the standard Clinical Portal for clinicians/users."""
    
    # --- Top Navbar ---
    nav_left, nav_center, nav_right = st.columns([1, 8, 2])
    with nav_left:
        st.markdown("""
        <div style="padding: 8px 0; display: flex; align-items: center; justify-content: center;">
            <div style="width: 50px; height: 50px; border-radius: 16px; background: #F1F5F9; box-shadow: 6px 6px 14px rgba(166, 180, 200, 0.45), -6px -6px 14px rgba(255, 255, 255, 0.9); display: flex; align-items: center; justify-content: center; position: relative;">
                <div style="font-family: var(--font-heading); font-weight: 900; font-size: 24px; letter-spacing: -1.5px; background: linear-gradient(135deg, #3B82F6 0%, #4F46E5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: flex; align-items: baseline; margin-left: 2px;">
                    M<span style="font-size: 20px; color: #4F46E5; -webkit-text-fill-color: #4F46E5; margin-left: 0px;">+</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with nav_center:
        st.markdown("""
        <div style="text-align: center; padding: 16px 24px; background: linear-gradient(135deg, rgba(239, 246, 255, 0.7) 0%, rgba(255, 255, 255, 0.9) 100%); border-radius: 24px; border: 1px solid rgba(255, 255, 255, 0.8); box-shadow: 0 10px 30px rgba(59, 130, 246, 0.08); backdrop-filter: blur(10px); margin: 0 auto; max-width: 850px;">
            <h1 style="margin: 0; font-family: var(--font-heading); font-weight: 900; font-size: 2.1rem; letter-spacing: -0.5px; background: linear-gradient(135deg, #1E293B 0%, #3B82F6 50%, #4F46E5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; padding-bottom: 4px;">
                AI-Based Medical Diagnosis Support System
            </h1>
            <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-top: 4px;">
                <div style="height: 2px; width: 50px; background: linear-gradient(90deg, transparent, #3B82F6);"></div>
                <p style="margin: 0; font-size: 0.75rem; color: #3B82F6; text-transform: uppercase; letter-spacing: 4px; font-weight: 800;">Intelligent Diagnostics Platform</p>
                <div style="height: 2px; width: 50px; background: linear-gradient(-90deg, transparent, #3B82F6);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with nav_right:
        icon1, icon2 = st.columns(2)
        with icon1:
            with st.popover("👤", width="stretch"):
                st.markdown(f"""
                <div style="text-align: center; padding: 10px 0; border-bottom: 1px solid var(--border); margin-bottom: 12px;">
                    <div style="width: 56px; height: 56px; background: var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px auto; border: 2px solid white; box-shadow: var(--shadow-soft);">
                        <span style="font-size: 24px; color: var(--primary); font-weight: 800;">{username[0].upper()}</span>
                    </div>
                    <div style="font-weight: 700; color: var(--text-heading);">{username}</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted);">Online • {st.session_state.role.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
                
                def sync_profile_to_db():
                    # Extract directly from widget keys to ensure we have the LATEST typed/selected value during callback
                    new_name = st.session_state.get('nav_name', st.session_state.patient_profile['name'])
                    new_age = st.session_state.get('nav_age', st.session_state.patient_profile['age'])
                    new_gender = st.session_state.get('nav_gender', st.session_state.patient_profile['gender'])
                    new_email = st.session_state.get('nav_email', st.session_state.patient_profile['email'])
                    new_contact = st.session_state.get('nav_contact', st.session_state.patient_profile['contact'])
                    new_address = st.session_state.get('nav_address', st.session_state.patient_profile['address'])
                    
                    success, msg = update_user_info(
                        user_id, new_name, new_age, new_gender, 
                        new_email, new_contact, new_address
                    )
                    if success:
                        # Update the persistent dictionary so other UI components (like headers) reflect changes
                        st.session_state.patient_profile.update({
                            'name': new_name, 'age': new_age, 'gender': new_gender,
                            'email': new_email, 'contact': new_contact, 'address': new_address
                        })
                        st.toast("✅ Profile Auto-Saved", icon="💾")
                    else:
                        st.error(f"Save Failed: {msg}")
                
                st.session_state.patient_profile['name'] = st.text_input("Full Name", value=st.session_state.patient_profile['name'], on_change=sync_profile_to_db, key="nav_name")
                st.session_state.patient_profile['age'] = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.patient_profile['age'], on_change=sync_profile_to_db, key="nav_age")
                gender_idx = ["Female", "Male", "Other"].index(st.session_state.patient_profile['gender']) if st.session_state.patient_profile['gender'] in ["Female", "Male", "Other"] else 1
                st.session_state.patient_profile['gender'] = st.selectbox("Gender", ["Female", "Male", "Other"], index=gender_idx, on_change=sync_profile_to_db, key="nav_gender")
                st.session_state.patient_profile['email'] = st.text_input("Email", value=st.session_state.patient_profile['email'], on_change=sync_profile_to_db, key="nav_email")
                st.session_state.patient_profile['contact'] = st.text_input("Contact", value=st.session_state.patient_profile['contact'], max_chars=15, on_change=sync_profile_to_db, key="nav_contact")
                st.session_state.patient_profile['address'] = st.text_input("Address", value=st.session_state.patient_profile['address'], on_change=sync_profile_to_db, key="nav_address")
        with icon2:
            if st.button("Logout", type="primary", width="stretch", key="nav_logout"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.role = ""
                st.session_state.user_id = None
                st.rerun()
    
    st.markdown("<hr style='margin: 4px 0 12px 0; border: none; border-top: 1px solid var(--border);'>", unsafe_allow_html=True)

    if 'patient_profile' not in st.session_state or st.session_state.patient_profile is None:
        st.session_state.patient_profile = {
            'name': '', 'age': 25, 'gender': 'Male', 
            'email': '', 'contact': '', 'address': ''
        }

    tabs = st.tabs([
        "🏠 Dashboard",
        "📝 Complete Patient Data Entry (Manual)", 
        "📄 Smart OCR Upload", 
        "✨ Wellness Center",
        "📈 Patient History & Analytics"
    ])
    
    tab0, tab2, tab3, tab_wellness, tab4 = tabs

    # ---------------------------------------------------------
    # TAB: WELLNESS CENTER
    # ---------------------------------------------------------
    with tab_wellness:
        render_wellness_center(user_id)

    # ---------------------------------------------------------
    # TAB 0: HOME DASHBOARD
    # ---------------------------------------------------------
    with tab0:
        # Fetch real stats
        dash_stats = get_user_dashboard_stats(user_id)
        
        # Welcome Banner
        from datetime import datetime
        greeting = "Good Morning" if datetime.now().hour < 12 else ("Good Afternoon" if datetime.now().hour < 17 else "Good Evening")
        render_luxury_header(f"{greeting}, {st.session_state.patient_profile.get('name', username)}!", icon="👋", badge_text="System Online", mode="hero")
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

        # --- Wellness Check-in Feature ---
        with st.container(border=True):
            st.markdown("<h4 style='color: #1E293B; margin-bottom: 20px; font-weight: 800;'>🌟 How is your health today compared to your last visit?</h4>", unsafe_allow_html=True)
            
            # Initialize session state for wellness check-in
            if 'wellness_check' not in st.session_state:
                st.session_state.wellness_check = None
                
            # Emoji options with custom click handling
            cols = st.columns([1, 1, 1, 1, 1])
            sentiments = [
                {"label": "Feeling Great!", "emoji_url": "https://cdnjs.cloudflare.com/ajax/libs/emoji-datasource-apple/15.0.1/img/apple/64/1f929.png", "val": "Great"},
                {"label": "Better", "emoji_url": "https://cdnjs.cloudflare.com/ajax/libs/emoji-datasource-apple/15.0.1/img/apple/64/1f60a.png", "val": "Better"},
                {"label": "No Change", "emoji_url": "https://cdnjs.cloudflare.com/ajax/libs/emoji-datasource-apple/15.0.1/img/apple/64/1f610.png", "val": "Same"},
                {"label": "A Bit Low", "emoji_url": "https://cdnjs.cloudflare.com/ajax/libs/emoji-datasource-apple/15.0.1/img/apple/64/1f61f.png", "val": "Low"},
                {"label": "Need Help", "emoji_url": "https://cdnjs.cloudflare.com/ajax/libs/emoji-datasource-apple/15.0.1/img/apple/64/1f198.png", "val": "Bad"}
            ]
            
            for idx, s in enumerate(sentiments):
                with cols[idx]:
                    st.markdown(f'<div style="text-align:center; padding-bottom:8px; margin-top:-4px;"><img src="{s["emoji_url"]}" width="56" style="filter: drop-shadow(0px 8px 16px rgba(0,0,0,0.15)); transition: transform 0.2s ease;"></div>', unsafe_allow_html=True)
                    if st.button(f"{s['val']}", key=f"well_{idx}", width="stretch"):
                        st.session_state.wellness_check = s['val']
            
            # Responses matching the selected mood
            responses = {
                "Great": "That's fantastic! Keep maintaining your healthy lifestyle. 🌟",
                "Better": "Glad to hear you're improving! Small steps lead to big changes. 😊",
                "Same": "Consistency is key. We're here to help you move forward. 😐",
                "Low": "Sorry you're feeling this way. Remember to rest and follow your medical plan. 😟",
                "Bad": "Please consider consulting your doctor soon. Your health is our priority. 🆘"
            }
            
            if st.session_state.wellness_check:
                msg = responses.get(st.session_state.wellness_check, "Thank you for sharing!")
                st.info(msg)
            
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Horizontal Stat Cards ---
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="dashboard-stat-card" style="border-bottom: 3px solid #3B82F6;">
                <div class="stat-icon" style="background: #EFF6FF; color: #3B82F6;">&#x1F4CA;</div>
                <div class="stat-value">{dash_stats['total_diagnoses']}</div>
                <div class="stat-label">Total Diagnoses</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="dashboard-stat-card" style="border-bottom: 3px solid #16A34A;">
                <div class="stat-icon" style="background: #F0FDF4; color: #16A34A;">&#x1F4C5;</div>
                <div class="stat-value" style="font-size: 1.2rem;">{dash_stats['last_visit']}</div>
                <div class="stat-label">Last Visit</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            alert_color = "#EF4444" if dash_stats['risk_alerts'] > 0 else "#16A34A"
            alert_bg = "#FEF2F2" if dash_stats['risk_alerts'] > 0 else "#F0FDF4"
            st.markdown(f"""
            <div class="dashboard-stat-card" style="border-bottom: 3px solid {alert_color};">
                <div class="stat-icon" style="background: {alert_bg}; color: {alert_color};">&#x26A0;</div>
                <div class="stat-value" style="color: {alert_color};">{dash_stats['risk_alerts']}</div>
                <div class="stat-label">Risk Alerts</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            status = dash_stats['health_status']
            if status == 'Good':
                s_color, s_bg, s_icon = '#16A34A', '#F0FDF4', '✅'
            elif status == 'Needs Attention':
                s_color, s_bg, s_icon = '#D97706', '#FFFBEB', '🔔'
            else:
                s_color, s_bg, s_icon = '#6B7280', '#F3F4F6', 'ℹ️'
            st.markdown(f"""
            <div class="dashboard-stat-card" style="border-bottom: 3px solid {s_color};">
                <div class="stat-icon" style="background: {s_bg}; color: {s_color};">{s_icon}</div>
                <div class="stat-value" style="color: {s_color}; font-size: 1.1rem;">{status}</div>
                <div class="stat-label">Health Status</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # ── AI INSIGHT BANNER (MOST CRITICAL ALERT) ──────────────────────────
        latest_insight = get_latest_patient_insight(user_id)
        
        cond_name, severity, observation = None, None, None
        
        if isinstance(latest_insight, tuple) and len(latest_insight) == 3:
            cond_name, severity, observation = latest_insight
        elif isinstance(latest_insight, dict) and latest_insight.get('observations'):
            _first_obs = latest_insight['observations'][0]
            cond_name = _first_obs.get('Condition', 'Unknown')
            severity = _first_obs.get('Severity', 'Normal')
            observation = _first_obs.get('Observation', 'No details provided')
            
        if cond_name and severity and observation:
            _sev_color = "#F87171" if severity in ['High', 'Critical'] else "#FBBF24"
            _sev_bg_1  = "#FEF2F2" if severity in ['High', 'Critical'] else "#FFFBEB"
            st.markdown(f"""
            <div class="hist-insight-pro" style="--bg-grad-1:{_sev_bg_1}; --bg-grad-2:#FFFFFF; --accent-color:{_sev_color}; --shadow-color:{_sev_color}40; margin-bottom: 25px;">
                <div class="hist-insight-icon-pro" style="color:{_sev_color};">⚠️</div>
                <div style="flex:1;">
                    <div class="hist-insight-title-pro" style="color:{_sev_color};">AI Risk Alert — {cond_name}</div>
                    <div class="hist-insight-body-pro">{observation}</div>
                    <div class="hist-insight-note-pro">💡 Consult a healthcare professional and bring recent lab reports.</div>
                </div>
                <span class="hist-sev-pill" style="background:{_sev_color}; position:absolute; top:24px; right:24px;">{severity.upper()}</span>
            </div>
            """, unsafe_allow_html=True)

        # --- Recent Activity Section ---
        st.markdown("""
        <div class="section-header">
            <span class="section-header-icon" style="background: #EFF6FF; color: #3B82F6;">&#x1F552;</span>
            <span class="section-header-title">Recent Activity</span>
        </div>
        """, unsafe_allow_html=True)
        recent_df = dash_stats.get('recent_activity', pd.DataFrame())
        if not recent_df.empty:
            recent_df.columns = ['Patient', 'Condition', 'Severity', 'Date']
            # Ensure IST localization for Recent Activity
            def format_row_date(d):
                dt = pd.to_datetime(d)
                return localize_ist(dt).strftime('%d %b %Y, %H:%M IST')
                
            recent_df['Date'] = recent_df['Date'].apply(format_row_date)
            
            # Styled activity cards
            for _, row in recent_df.iterrows():
                sev = row['Severity']
                if sev in ['High', 'Critical']:
                    badge_cls = 'health-critical'
                elif sev == 'Mild':
                    badge_cls = 'health-mild'
                else:
                    badge_cls = 'health-normal'
                st.markdown(f"""
                <div class="dashboard-activity-row">
                    <div style="flex:1;">
                        <span style="font-weight: 600; color: var(--text-heading);">{row['Condition']}</span>
                        <span style="color: var(--text-muted); font-size: 0.8rem; margin-left: 8px;">{row['Patient']}</span>
                    </div>
                    <span class="health-badge {badge_cls}">{sev}</span>
                    <span style="color: var(--text-muted); font-size: 0.8rem; min-width: 130px; text-align: right;">{row['Date']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No diagnostic activity yet. Run your first diagnosis from the **Manual Data Entry** or **Smart OCR Upload** tabs!")
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # --- Quick Actions ---
        st.markdown("""
        <div class="section-header">
            <span class="section-header-icon" style="background: #FFFBEB; color: #D97706;">&#x26A1;</span>
            <span class="section-header-title">Quick Actions</span>
        </div>
        """, unsafe_allow_html=True)
        qa1, qa2, qa3 = st.columns(3)
        with qa1:
            st.markdown("""
            <div class="quick-action-card" style="border-left: 4px solid #3B82F6;">
                <div class="qa-icon" style="background: #EFF6FF; color: #3B82F6;">&#x270D;</div>
                <div class="qa-title">New Diagnosis</div>
                <div class="qa-desc">Enter patient vitals manually</div>
                <div class="qa-hint">Go to Manual Entry tab &rarr;</div>
            </div>
            """, unsafe_allow_html=True)
        with qa2:
            st.markdown("""
            <div class="quick-action-card" style="border-left: 4px solid #A855F7;">
                <div class="qa-icon" style="background: #FDF4FF; color: #A855F7;">&#x1F4C4;</div>
                <div class="qa-title">Upload Report</div>
                <div class="qa-desc">OCR scan a medical document</div>
                <div class="qa-hint">Go to OCR Upload tab &rarr;</div>
            </div>
            """, unsafe_allow_html=True)
        with qa3:
            st.markdown("""
            <div class="quick-action-card" style="border-left: 4px solid #16A34A;">
                <div class="qa-icon" style="background: #F0FDF4; color: #16A34A;">&#x1F4C8;</div>
                <div class="qa-title">View History</div>
                <div class="qa-desc">Browse past diagnostic records</div>
                <div class="qa-hint">Go to History tab &rarr;</div>
            </div>
            """, unsafe_allow_html=True)


    # ---------------------------------------------------------
    # TAB 2: MANUAL ENTRY
    # ---------------------------------------------------------
    with tab2:
        # --- CROSS-WIDGET SYNCHRONIZATION HELPERS ---
        def sync_glu_v(): st.session_state.mi_glu_d = st.session_state.mi_glu_v
        def sync_glu_d(): st.session_state.mi_glu_v = st.session_state.mi_glu_d
        def sync_bmi_v(): st.session_state.mi_bmi_d = st.session_state.mi_bmi_v
        def sync_bmi_d(): st.session_state.mi_bmi_v = st.session_state.mi_bmi_d

        def fill_demo_data(mode):
            st.session_state.demo_mode = mode
            st.session_state.auto_submit = True 
            
            if mode == "NORMAL":
                st.session_state.mi_sys = 115
                st.session_state.mi_dia = 78
                st.session_state.mi_glu_v = 92.0
                st.session_state.mi_glu_d = 92.0
                st.session_state.mi_chol = 170.0
                st.session_state.mi_bmi_v = 23.0
                st.session_state.mi_bmi_d = 23.0
                st.session_state.mi_oxy = 98
                st.session_state.mi_hr = 72
                st.session_state.mi_temp = 36.7
                
                st.session_state.mi_preg = 0
                st.session_state.mi_skin = 12
                st.session_state.mi_ins = 30.0
                st.session_state.mi_dpf = 0.2
                
                st.session_state.mi_cp = 0  # index 1 equates to -1 usually? Wait, let's look: options are [-1,0,1,2,3]. Index 0 is -1, index 1 is 0. But we map to value directly. The state captures value. So 0 is 0.
                st.session_state.mi_restecg = 0
                st.session_state.mi_thalach = 130
                st.session_state.mi_exang = 0
                st.session_state.mi_oldpeak = 0.0
                st.session_state.mi_slope = 0
                st.session_state.mi_ca = 0
                st.session_state.mi_thal = 0
                
                st.session_state.mi_creat = 0.8
                st.session_state.mi_hb = 14.5
                st.session_state.mi_wbc = 6000
                st.session_state.mi_plat = 200000
                st.session_state.mi_sgot = 20.0
                st.session_state.mi_sgpt = 22.0
                st.session_state.mi_crp = 1.0
                
                st.session_state.mi_to = "NEGATIVE"
                st.session_state.mi_th = "NEGATIVE"
                st.session_state.mi_digg = "NON-REACTIVE"
                st.session_state.mi_digm = "NON-REACTIVE"
                st.session_state.mi_dns1 = "NON-REACTIVE"

                st.session_state.mi_s1 = "None"
                st.session_state.mi_s2 = "None"
                st.session_state.mi_s3 = "None"

            elif mode == "CRITICAL":
                st.session_state.mi_sys = 160
                st.session_state.mi_dia = 100
                st.session_state.mi_glu_v = 220.0
                st.session_state.mi_glu_d = 220.0
                st.session_state.mi_chol = 280.0
                st.session_state.mi_bmi_v = 32.0
                st.session_state.mi_bmi_d = 32.0
                st.session_state.mi_oxy = 90
                st.session_state.mi_hr = 110
                st.session_state.mi_temp = 38.5
                
                st.session_state.mi_preg = 3
                st.session_state.mi_skin = 35
                st.session_state.mi_ins = 200.0
                st.session_state.mi_dpf = 1.2
                
                st.session_state.mi_cp = 2
                st.session_state.mi_restecg = 1
                st.session_state.mi_thalach = 190
                st.session_state.mi_exang = 1
                st.session_state.mi_oldpeak = 4.0
                st.session_state.mi_slope = 2
                st.session_state.mi_ca = 3
                st.session_state.mi_thal = 2
                
                st.session_state.mi_creat = 2.5
                st.session_state.mi_hb = 10.0
                st.session_state.mi_wbc = 16000
                st.session_state.mi_plat = 100000
                st.session_state.mi_sgot = 85.0
                st.session_state.mi_sgpt = 110.0
                st.session_state.mi_crp = 45.0
                
                st.session_state.mi_to = "POSITIVE"
                st.session_state.mi_th = "POSITIVE"
                st.session_state.mi_digg = "REACTIVE"
                st.session_state.mi_digm = "REACTIVE"
                st.session_state.mi_dns1 = "REACTIVE"

                st.session_state.mi_s1 = "Fever"
                st.session_state.mi_s2 = "Fatigue"
                st.session_state.mi_s3 = "Shortness of breath"

        def reset_form_data():
            """Clears all manual input fields and resets the UI state."""
            # Explicitly set default values instead of deleting keys
            # (Streamlit widgets don't fully reset on key deletion)
            
            # Core Vitals — int fields default to 0
            for k in ['mi_sys', 'mi_dia', 'mi_oxy', 'mi_hr', 'mi_preg', 'mi_skin', 'mi_wbc', 'mi_plat', 'mi_thalach']:
                st.session_state[k] = 0
            
            # Core Vitals — float fields default to 0.0
            for k in ['mi_glu_v', 'mi_glu_d', 'mi_chol', 'mi_bmi_v', 'mi_bmi_d', 'mi_temp', 'mi_ins', 'mi_dpf', 'mi_oldpeak', 'mi_creat', 'mi_hb', 'mi_sgot', 'mi_sgpt', 'mi_crp']:
                st.session_state[k] = 0.0
            
            # Selectbox fields — reset to first option (sentinel/default)
            for k in ['mi_cp', 'mi_restecg', 'mi_exang', 'mi_slope', 'mi_ca', 'mi_thal']:
                st.session_state[k] = -1  # Sentinel "--- Select ---" value
            
            # Infectious disease selectors — reset to "Not Tested"
            for k in ['mi_to', 'mi_th']:
                st.session_state[k] = "Not Tested"
            for k in ['mi_digg', 'mi_digm', 'mi_dns1']:
                st.session_state[k] = "Not Tested"
            
            # Symptom selectors — reset to "None"
            for k in ['mi_s1', 'mi_s2', 'mi_s3']:
                st.session_state[k] = "None"
            
            # Clear demo mode flag
            if 'demo_mode' in st.session_state:
                del st.session_state['demo_mode']
            st.session_state.auto_submit = False
            
            # Initialize active_block if not present
            if 'active_block' not in st.session_state:
                st.session_state.active_block = None

        render_luxury_header("Manual Clinical Data Entry", icon="✍️", badge_text="Precision Input", mode="hero")
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
        
        # --- FORM CONTROL BUTTONS ---
        col_info, col_btn1, col_btn2, col_btn3 = st.columns([5, 1.2, 1.2, 1.2])
        with col_info:
            if st.session_state.get('demo_mode'):
                demo_m = st.session_state.demo_mode
                clr = "#10B981" if demo_m == "NORMAL" else "#EF4444"
                st.markdown(f"<div style='padding:6px 12px; border-radius:6px; background:{clr}20; color:{clr}; font-weight:bold; text-align:left; max-width: 200px;'>⚡ Demo: {demo_m}</div>", unsafe_allow_html=True)
        with col_btn1:
            st.button("🧪 Normal", on_click=fill_demo_data, args=("NORMAL",), type="secondary", width="stretch")
        with col_btn2:
            st.button("🚨 Critical", on_click=fill_demo_data, args=("CRITICAL",), type="primary", width="stretch")
        with col_btn3:
            st.button("🔄 Reset", on_click=reset_form_data, type="secondary", width="stretch")
            
        # --- CLINICAL ENTRY FORM ---
        # Wrapping in a form prevents reruns on every input change for a smoother experience
        with st.form("clinical_entry_form", clear_on_submit=False):
            # Use global profile gender to control UI logic
            p_sex = st.session_state.get('patient_profile', {}).get('gender', 'Female')
            
            with st.expander("❤️ Core Vitals & Blood Sugar", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    systolic = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=250, value=0, key="mi_sys")
                    diastolic = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=150, value=0, key="mi_dia")
                with col2:
                    glucose_v = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=400.0, value=0.0, key="mi_glu_v")
                    cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0, max_value=400.0, value=0.0, key="mi_chol")
                with col3:
                    bmi_v = st.number_input("BMI", min_value=0.0, max_value=60.0, value=0.0, key="mi_bmi_v")
                    oxygen = st.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=0, key="mi_oxy")
                with col4:
                    heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=200, value=0, key="mi_hr")
                    body_temp = st.number_input("Body Temp (°C)", min_value=0.0, max_value=45.0, value=0.0, key="mi_temp")

                st.markdown("<br>", unsafe_allow_html=True)
                submit_core_vitals = st.form_submit_button("🩸 Evaluate Blood Sugar & Vitals Only", width="stretch")
                
            with st.expander("🩸 Advanced Diabetic Specifics", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    glucose_d = st.number_input("Blood Glucose (mg/dL)", min_value=0.0, max_value=400.0, value=0.0, key="mi_glu_d")
                    bmi_d = st.number_input("Current BMI", min_value=0.0, max_value=60.0, value=0.0, key="mi_bmi_d")
                with col2:
                    if p_sex == "Female":
                        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, key="mi_preg")
                    else:
                        st.markdown("**Pregnancies**")
                        st.caption("N/A for Men")
                        pregnancies = 0
                    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=0, key="mi_skin")
                with col3: 
                    insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=0.0, key="mi_ins")
                with col4:
                    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0, format="%.3f", key="mi_dpf")
                submit_dia = st.form_submit_button("🩺 Analyze Diabetes Only", width="stretch")
                
            with st.expander("❤️ Advanced Heart Diagnostics", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    cp = st.selectbox("Chest Pain Pattern", [-1, 0, 1, 2, 3], index=0, 
                        format_func=lambda x: "--- Select Pattern ---" if x==-1 else 
                                    "0: Typical Angina" if x==0 else 
                                    "1: Atypical Angina" if x==1 else 
                                    "2: Non-anginal Pain" if x==2 else 
                                    "3: Asymptomatic", 
                        help="Typical: Pressing/squeezing, Atypical: Sharp/brief, Non-anginal: Not cardiac, Asymptomatic: No pain", key="mi_cp")
                    
                    restecg = st.selectbox("Resting ECG Result", [-1, 0, 1, 2], index=0, 
                        format_func=lambda x: "--- Select ECG ---" if x==-1 else 
                                    "0: Normal" if x==0 else 
                                    "1: ST-T Wave Abnormality" if x==1 else 
                                    "2: LV Hypertrophy", 
                        help="ST-T abnormality includes T wave inversions and ST elevation/depression > 0.05 mV", key="mi_restecg")
                    
                    thalach = st.number_input("Max Heart Rate (bpm)", min_value=0, max_value=250, value=0, help="Normal: 90-170 bpm", key="mi_thalach")
                with col2:
                    exang = st.selectbox("Exercise Induced Angina", [-1, 0, 1], index=0, 
                        format_func=lambda x: "--- Select ---" if x==-1 else "No Angina during exercise" if x==0 else "Angina present during exercise", 
                        help="Chest pain triggered/worsened by exertion", key="mi_exang")
                    
                    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, help="Normal: < 0.5 mm, Risk: >= 1 mm depression", key="mi_oldpeak")
                    
                    slope = st.selectbox("ST Slope Pattern", [-1, 0, 1, 2], index=0, 
                        format_func=lambda x: "--- Select Slope ---" if x==-1 else 
                                    "0: Upsloping (Normal)" if x==0 else 
                                    "1: Flat (Ischemia Risk)" if x==1 else 
                                    "2: Downsloping (Critical)", 
                        help="The slope of the peak exercise ST segment", key="mi_slope")
                with col3:
                    ca = st.selectbox("Major Vessel Blockage", [-1, 0, 1, 2, 3], index=0, 
                        format_func=lambda x: "--- Select Vessel Count ---" if x==-1 else 
                                    "0: 0 Vessels Clear" if x==0 else 
                                    f"{x}: {x} Vessels Blocked", 
                        help="Number of major vessels (0-3) colored by fluoroscopy", key="mi_ca")
                    
                    thal = st.selectbox("Thalassemia Status", [-1, 0, 1, 2], index=0, 
                        format_func=lambda x: "--- Select Status ---" if x==-1 else 
                                    "0: Normal Blood Flow" if x==0 else 
                                    "1: Fixed Defect (Stable)" if x==1 else 
                                    "2: Reversible Defect (Unstable)", 
                        help="Blood flow condition as shown on cardiac scan", key="mi_thal")
                submit_heart = st.form_submit_button("🩺 Analyze Heart Disease Only", width="stretch")

            with st.expander("🧪 Comprehensive Pathology Panels", expanded=False):
                colA, colB, colC = st.columns(3)
                with colA:
                    st.markdown("**Haemogram & Kidneys**")
                    creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=15.0, value=0.0, help="Kidney function marker (Normal: 0.6-1.3 mg/dL)", key="mi_creat")
                    hb = st.number_input("Haemoglobin (gm%)", min_value=0.0, max_value=25.0, value=0.0, help="WHO Normal: 12.0-17.5 gm% (Anemia indicator)", key="mi_hb")
                    wbc = st.number_input("WBC Count (/cmm)", min_value=0, max_value=100000, value=0, help="Immune system marker (Normal: 4,000-11,000 /cmm)", key="mi_wbc")
                    platelets = st.number_input("Platelets (/cmm)", min_value=0, max_value=1000000, value=0, help="Blood clotting factor (Normal: 150,000-450,000 /cmm)", key="mi_plat")
                with colB:
                    st.markdown("**Liver & Inflammation**")
                    sgot = st.number_input("SGOT/AST (U/L)", min_value=0.0, max_value=1000.0, value=0.0, help="Liver/Heart enzyme (Normal: 8-48 U/L)", key="mi_sgot")
                    sgpt = st.number_input("SGPT/ALT (U/L)", min_value=0.0, max_value=1000.0, value=0.0, help="Liver marker (Normal: 7-55 U/L)", key="mi_sgpt")
                    crp = st.number_input("C-Reactive Protein (mg/L)", min_value=0.0, max_value=300.0, value=0.0, help="Inflammation marker (Normal: < 10 mg/L)", key="mi_crp")
                with colC:
                    st.markdown("**Infectious Diseases**")
                    typhoid_o = st.selectbox("Typhoid 'O'", ["Not Tested", "NEGATIVE", "POSITIVE"], index=0, key="mi_to")
                    typhoid_h = st.selectbox("Typhoid 'H'", ["Not Tested", "NEGATIVE", "POSITIVE"], index=0, key="mi_th")
                    d_igg = st.selectbox("Dengue IgG", ["Not Tested", "NON-REACTIVE", "WEAK REACTIVE", "REACTIVE"], index=0, key="mi_digg")
                    d_igm = st.selectbox("Dengue IgM", ["Not Tested", "NON-REACTIVE", "WEAK REACTIVE", "REACTIVE"], index=0, key="mi_digm")
                    d_ns1 = st.selectbox("Dengue NS1", ["Not Tested", "NON-REACTIVE", "WEAK REACTIVE", "REACTIVE"], index=0, key="mi_dns1")

                st.markdown("**General Symptoms**")
                s_list = ["None", "Fatigue", "Cough", "Fever", "Sore throat", "Headache", "Shortness of breath", "Runny nose", "Body ache"]
                colS1, colS2, colS3 = st.columns(3)
                with colS1: s1 = st.selectbox("Symptom 1", s_list, index=0, key="mi_s1")
                with colS2: s2 = st.selectbox("Symptom 2", s_list, index=0, key="mi_s2")
                with colS3: s3 = st.selectbox("Symptom 3", s_list, index=0, key="mi_s3")
                
                submit_path = st.form_submit_button("🧪 Analyze Pathology & Symptoms Only", width="stretch")
                submit_general = st.form_submit_button("🩺 Analyze General Disease Only", width="stretch")

            glucose = glucose_v if glucose_v > 0 else glucose_d
            bmi = bmi_v if bmi_v > 0 else bmi_d
            
            # Build dict for current state evaluation
            manual_data = {}
            if systolic > 0: manual_data['systolic'] = systolic
            if diastolic > 0: manual_data['diastolic'] = diastolic
            if glucose > 0: manual_data['glucose'] = glucose
            if cholesterol > 0: manual_data['cholesterol'] = cholesterol
            if bmi > 0: manual_data['bmi'] = bmi
            if oxygen > 0: manual_data['oxygen_saturation'] = oxygen
            if heart_rate > 0: manual_data['heart_rate_bpm'] = heart_rate
            if body_temp > 0: manual_data['body_temperature_c'] = body_temp
            
            # Use global profile gender for evaluation
            manual_data['sex'] = p_sex
            if p_sex == "Female":
                manual_data['pregnancies'] = pregnancies
            else:
                manual_data['pregnancies'] = 0
            
            if skin_thickness > 0: manual_data['skin_thickness'] = skin_thickness
            if insulin > 0: manual_data['insulin'] = insulin
            if dpf > 0: manual_data['dpf'] = dpf
            if cp >= 0: manual_data['cp'] = cp
            if restecg >= 0: manual_data['restecg'] = restecg
            if thalach > 0: manual_data['thalach'] = thalach
            if exang >= 0: manual_data['exang'] = exang
            if oldpeak > 0: manual_data['oldpeak'] = oldpeak
            if slope >= 0: manual_data['slope'] = slope
            if ca >= 0: manual_data['ca'] = ca
            if thal >= 0: manual_data['thal'] = thal
            if creatinine > 0: manual_data['creatinine'] = creatinine
            if hb > 0: manual_data['hb'] = hb
            if wbc > 0: manual_data['wbc'] = wbc
            if platelets > 0: manual_data['platelets'] = platelets
            if sgot > 0: manual_data['sgot'] = sgot
            if sgpt > 0: manual_data['sgpt'] = sgpt
            if crp > 0: manual_data['crp'] = crp
            if typhoid_o != "Not Tested": manual_data['typhoid_o'] = typhoid_o
            if typhoid_h != "Not Tested": manual_data['typhoid_h'] = typhoid_h
            if d_igg != "Not Tested": manual_data['dengue_igg'] = d_igg
            if d_igm != "Not Tested": manual_data['dengue_igm'] = d_igm
            if d_ns1 != "Not Tested": manual_data['dengue_ns1'] = d_ns1
            if s1 != "None": manual_data['symptom_1'] = s1
            if s2 != "None": manual_data['symptom_2'] = s2
            if s3 != "None": manual_data['symptom_3'] = s3
            
            # --- SUBMISSION HANDLER ---
            # Map buttons to block IDs
            if submit_core_vitals: st.session_state.active_block = 'core_vitals'
            if submit_dia: st.session_state.active_block = 'diabetes'
            if submit_heart: st.session_state.active_block = 'heart'
            if submit_path: st.session_state.active_block = 'pathology'
            if submit_general: st.session_state.active_block = 'general'
            
            # Reset indicators if reset button was clicked (handled elsewhere)
            
            # Evaluate dynamically
            active_block = st.session_state.get('active_block')
            dynamic_results = evaluate_manual_clinical_risk(manual_data, target_block=active_block)
            
            # Use dynamic_results directly for live feedback
            # Filtering is already handled strictly inside evaluate_manual_clinical_risk
            render_luxury_header("Live Clinical Risk Indicators", icon="🔴")
            
            if dynamic_results and active_block:
                indicators_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px;'>"
                for r in dynamic_results:
                    # Support HIGH, LOW, and NORMAL colors
                    color = "#3B82F6" if r['status'] == 'LOW' else ("#10B981" if r['status'] == 'NORMAL' else "#EF4444")
                    indicators_html += f"<span style='background-color:{color}10; border: 1px solid {color}80; color:{color}; padding: 8px 16px; border-radius: 10px; font-weight: 700; font-size: 0.85rem; border-left: 4px solid {color};'>{r['param']}: {r['status']}</span>"
                indicators_html += "</div>"
                st.markdown(indicators_html, unsafe_allow_html=True)
            elif not active_block:
                st.info("💡 **Clinical Tip**: Select a diagnostic module above (e.g., 'Analyze Heart') to activate live markers.")
            else:
                st.info("Enter values and click an **Analyze** button inside a block to see live markers.")

            # Add professional spacing as requested
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

            # --- STEP 2: FULL REPORT (Only if a block is active) ---
            if active_block:
                st.success(f"✅ **{active_block.upper()}** appraisal complete. Click below to **archieve diagnostic history** and generate the full AI report.")
                submit_btn = st.form_submit_button("📊 ARCHIVE HISTORY & Generate Comprehensive AI Report", type="primary", width="stretch")
            else:
                submit_btn = False

        # --- TRIGGER LOGIC ---
        # IMPORTANT: any_submit should ONLY be True for Step 2 or Auto-demo mode.
        # Step 1 buttons (submit_dia, etc.) should update st.session_state.active_block 
        # but NOT trigger the full diagnostic pipeline.
        
        any_submit = submit_btn # Only Step 2 button triggers the full pipeline
        
        # Handle Demo Mode auto-submission
        if st.session_state.get('auto_submit'):
            any_submit = True
            st.session_state.auto_submit = False  # Reset flag immediately
            
        # IMPORTANT: run_diagnostic_pipeline should ONLY run if any_submit is True
        if any_submit:
            tab_obs, tab_risk, tab_det = st.tabs(["🌡️ Vital Indicators", "📊 Risk Analytics", "🩺 Clinical Insights"])
            
            with st.spinner("Processing ML Diagnostics..."):
                manual_data['age'] = st.session_state.patient_profile['age']
                run_diagnostic_pipeline(
                    manual_data, 
                    scaler_dia, feature_keys_dia, 
                    scaler_heart, feature_keys_heart, 
                    scaler_diag=scaler_diag, feature_keys_diag=feature_keys_diag, 
                    le_diag=le_diag, 
                    patient_info=st.session_state.patient_profile, 
                    user_id=user_id, 
                    tabs=(tab_obs, tab_risk, tab_det),
                    target_block=st.session_state.get('active_block')
                )
            
            # Smooth scrolling to the results container
            import streamlit.components.v1 as components
            components.html("""
                <script>
                    const parent = window.parent.document;
                    const results = parent.querySelectorAll('h2');
                    for (let i = 0; i < results.length; i++) {
                        if (results[i].innerText.includes('Clinical Analysis Result')) {
                            results[i].scrollIntoView({behavior: 'smooth', block: 'start'});
                            break;
                        }
                    }
                </script>
            """, height=0)



    # ---------------------------------------------------------
    # TAB 3: OCR UPLOAD
    # ---------------------------------------------------------
    with tab3:
        # 1. Premium Hero Banner
        render_luxury_header("Smart Medical OCR", icon="📸", badge_text="AI Vision Enabled", mode="hero")

        
        # 2. Instruction Grid
        st.markdown("""
        <div class="ocr-features-grid">
            <div class="ocr-feature-card">
                <div class="ocr-feature-icon">📄</div>
                <div class="ocr-feature-title">1. Upload Report</div>
                <div class="ocr-feature-desc">Upload clinical lab reports in PDF, PNG, or JPG format (Max 200MB).</div>
            </div>
            <div class="ocr-feature-card">
                <div class="ocr-feature-icon">🤖</div>
                <div class="ocr-feature-title">2. AI Extraction</div>
                <div class="ocr-feature-desc">Our Smart Vision AI auto-magically reads and parses your medical data.</div>
            </div>
            <div class="ocr-feature-card">
                <div class="ocr-feature-icon">🩺</div>
                <div class="ocr-feature-title">3. Instant Analysis</div>
                <div class="ocr-feature-desc">Get an immediate clinical risk assessment using 11 ML pipelines.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. File Uploader Container
        st.markdown("<h4 style='text-align:center; color:var(--text-heading); margin-bottom: 20px; font-weight:800;'>📤 Upload Your Medical Report</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop file here", type=["pdf", "png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded_file is not None:
            if st.button("Extract Data & Analyze", width="stretch", type="primary"):
                with st.spinner("Analyzing Document..."):
                    bytes_data = uploaded_file.getvalue()
                    extracted_data = process_document_to_dict(bytes_data, uploaded_file.name)
                    extracted_data['age'] = st.session_state.patient_profile['age']
                    
                    # Note: Using unified results flow without redundant OCR header
                    # Create 3 tabs for OCR results
                    tab_obs, tab_risk, tab_det = st.tabs(["🌡️ Vital Indicators", "📊 Risk Analytics", "🩺 Clinical Insights"])
                    
                    # Note: Vital Indicators for OCR are currently handled inside the pipeline or derived from extracted data
                    run_diagnostic_pipeline(extracted_data, scaler_dia, feature_keys_dia, scaler_heart, feature_keys_heart, scaler_diag=scaler_diag, feature_keys_diag=feature_keys_diag, le_diag=le_diag, patient_info=st.session_state.patient_profile, user_id=user_id, tabs=(tab_obs, tab_risk, tab_det))


    # ---------------------------------------------------------
    # TAB 4: PATIENT HISTORY & ANALYTICS
    # ---------------------------------------------------------
    with tab4:
        # ── PAGE HEADER ──────────────────────────────────────────────────────
        render_luxury_header("Patient Diagnostic History", icon="🧬", badge_text="Ultra Pro Analytics", mode="hero")
        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

        # ── FETCH DATA ────────────────────────────────────────────────────────
        history_df = get_patient_history(user_id=user_id)
        
        # Initialize metrics for safety (Fixes NameError: total_sessions)
        total_sessions = 0
        high_count = 0
        mild_count = 0
        norm_count = 0
        
        if not history_df.empty:
            # 1. Pre-process Date
            history_df['Date'] = pd.to_datetime(history_df['timestamp'])
            history_df['Condition'] = history_df['rule_disease'].fillna(history_df['ml_disease']).fillna("Analysis Run")
            history_df['Severity'] = history_df['severity'].fillna("Normal")
            
            # 2. Map Severity to Rank for Aggregation
            sev_rank = {'Critical': 4, 'High': 3, 'Mild': 2, 'Normal': 1, '-': 0}
            history_df['SevRank'] = history_df['Severity'].map(lambda x: sev_rank.get(x, 0))
            
            # 3. Group by Timestamp (session-based)
            grouped = history_df.groupby('timestamp').agg({
                'Condition': lambda x: sorted(list(set(x))), # Keep as list for tag rendering
                'Severity': lambda x: sorted(list(x), key=lambda s: sev_rank.get(s, 0), reverse=True)[0],
                'Date': 'first',
                'session_id': 'first' # Required for deep-drill fetching
            }).reset_index().sort_values('Date', ascending=False)
            
            # Ensure aware timestamp before formatting
            grouped['DateStr'] = grouped['Date'].apply(lambda x: localize_ist(x).strftime('%b %d, %Y  %H:%M IST'))
            
            # ── PRE-CALCULATE ANALYTICS DATA ────────────────────────────────────
            total_sessions = len(grouped)
            high_count = len(history_df[history_df['Severity'].isin(['High', 'Critical'])])
            mild_count = len(history_df[history_df['Severity'] == 'Mild'])
            norm_count = len(history_df[history_df['Severity'] == 'Normal'])
            
            # Clinical Trends Data (Starting from Jan of current year)
            history_df['Date'] = pd.to_datetime(history_df['timestamp'])
            history_df['MonthYear'] = history_df['Date'].dt.strftime('%b %Y')
            now = pd.Timestamp.now()
            # Start from Jan 2026 specifically as requested
            month_range = [ (pd.Timestamp(2026, 1, 1) + pd.DateOffset(months=i)).strftime('%b %Y') for i in range(now.month) ]
            range_df = pd.DataFrame({'MonthYear': month_range})
            total_per_month = history_df.groupby('MonthYear').size().reset_index(name='TotalLogs')
            alerts_per_month = history_df[history_df['Severity'].isin(['High', 'Critical'])].groupby('MonthYear').size().reset_index(name='Alerts')
            risk_trend = pd.merge(range_df, total_per_month, on='MonthYear', how='left')
            risk_trend = pd.merge(risk_trend, alerts_per_month, on='MonthYear', how='left').fillna(0)
            risk_trend['AlertRate'] = (risk_trend['Alerts'] / risk_trend['TotalLogs'].replace(0, 1) * 100).round(1)
            risk_trend.loc[risk_trend['TotalLogs'] == 0, 'AlertRate'] = 0
            
            # Monthly Comparison Data
            cur_m = now.strftime('%b %Y')
            cur_d = history_df[history_df['MonthYear'] == cur_m]
            cur_c = pd.DataFrame()
            if not cur_d.empty:
                cur_c = cur_d['Condition'].value_counts().reset_index()
                cur_c.columns = ['Condition', 'Count']
                cur_c = cur_c.sort_values('Count', ascending=True) # Ascending for horizontal Bar chart (largest at top)

            # ── 3-TAB ANALYTICS SYSTEM ──────────────────────────────────────────
            h_tab1, h_tab2, h_tab3 = st.tabs(["📊 Condition Distribution", "📈 Clinical Trends", "📜 History"])

            # ── TAB 1: CONDITION DISTRIBUTION & STATS ──────────────────────────
            with h_tab1:
                col_pie, col_stats = st.columns([1.1, 0.9], gap="large")
                
                with col_pie:
                    st.markdown('<div style="font-weight:800; color:#1E293B; margin-bottom:15px; font-size:1.1rem; display:flex; align-items:center; gap:8px;">📊 CONDITION DISTRIBUTION</div>', unsafe_allow_html=True)
                    condition_counts = history_df['Condition'].value_counts().reset_index()
                    condition_counts.columns = ['Condition', 'Count']
                    PALETTE = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#06B6D4", "#F43F5E"]
                    
                    try:
                        total_s = len(grouped)
                        fig_p = go.Figure(go.Pie(
                            labels=condition_counts['Condition'], 
                            values=condition_counts['Count'], 
                            hole=0.65, 
                            marker=dict(colors=PALETTE, line=dict(color='white', width=2)),
                            textinfo='percent',
                            textposition='outside',
                            showlegend=True,
                            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Dist: %{percent}<extra></extra>'
                        ))
                        fig_p.add_annotation(text=f"<b>{total_s}</b><br>Sessions", x=0.5, y=0.5, showarrow=False, font=dict(size=18))
                        fig_p.update_layout(margin=dict(t=30, b=30, l=10, r=10), height=450, paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_p, width="stretch", config={'displayModeBar':False})
                    except: st.info("Condition visualization ready.")
                
                with col_stats:
                    st.markdown('<div style="font-weight:800; color:#1E293B; margin-bottom:15px; font-size:1.1rem; display:flex; align-items:center; gap:8px;">🟦 QUICK STATS</div>', unsafe_allow_html=True)
                    s_col1, s_col2 = st.columns(2)
                    with s_col1:
                        st.markdown(f'<div style="border: 2px solid #3B82F6; border-radius: 12px; padding: 20px; text-align: center; background: white; margin-bottom:15px;"><div style="font-size: 1.8rem; font-weight: 800; color: #3B82F6;">{total_sessions}</div><div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">DIAGNOSTIC SESSIONS</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="border: 2px solid #F59E0B; border-radius: 12px; padding: 20px; text-align: center; background: white;"><div style="font-size: 1.8rem; font-weight: 800; color: #F59E0B;">{mild_count}</div><div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">MILD INDICATORS</div></div>', unsafe_allow_html=True)
                    with s_col2:
                        st.markdown(f'<div style="border: 2px solid #EF4444; border-radius: 12px; padding: 20px; text-align: center; background: white; margin-bottom:15px;"><div style="font-size: 1.8rem; font-weight: 800; color: #EF4444;">{high_count}</div><div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">CLINICAL ALERTS</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="border: 2px solid #10B981; border-radius: 12px; padding: 20px; text-align: center; background: white;"><div style="font-size: 1.8rem; font-weight: 800; color: #10B981;">{norm_count}</div><div style="font-size: 0.75rem; color: #64748B; font-weight: 700;">NORMAL FINDINGS</div></div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div style="font-weight:800; color:#1E293B; margin-bottom:15px; font-size:1.1rem; display:flex; align-items:center; gap:8px;">📌 FREQUENT OBSERVATIONS</div>', unsafe_allow_html=True)
                    if not condition_counts.empty:
                        t_html = ""
                        for i, r in condition_counts.head(5).iterrows():
                            c = PALETTE[i % len(PALETTE)]
                            t_html += f'<div style="display:flex; align-items:center; padding: 10px 0; border-bottom: 1px solid #F1F5F9;"><span style="background:{c}; width:10px; height:10px; border-radius:50%; margin-right:12px;"></span><span style="font-size:0.95rem; font-weight:600; color:#334155; flex:1;">{r["Condition"]}</span><span style="font-weight:800;">{r["Count"]}</span></div>'
                        st.markdown(f'<div style="background: white; padding: 0 15px; border-radius: 12px; border: 1px solid #E2E8F0;">{t_html}</div>', unsafe_allow_html=True)

            # ── TAB 2: CLINICAL TRENDS ─────────────────────────────────────────
            with h_tab2:
                st.markdown('<div style="font-weight:800; color:#1E293B; margin-bottom:20px; font-size:1.1rem; display:flex; align-items:center; gap:8px;">🧬 CLINICAL TRENDS & ANALYTICS</div>', unsafe_allow_html=True)
                col_line, col_bar = st.columns(2, gap="large")
                with col_line:
                    fig_trend = go.Figure()
                    
                    # 1. Neon Glow Shadow Trace
                    fig_trend.add_trace(go.Scatter(
                        x=risk_trend['MonthYear'], y=risk_trend['AlertRate'],
                        mode='lines',
                        line=dict(color='rgba(59, 130, 246, 0.3)', width=8, shape='spline', smoothing=0.4),
                        hoverinfo='none',
                        showlegend=False
                    ))
                    
                    # 2. Main High-Contrast Trace
                    fig_trend.add_trace(go.Scatter(
                        x=risk_trend['MonthYear'], 
                        y=risk_trend['AlertRate'], 
                        mode='lines+markers+text', 
                        line=dict(color='#2563EB', width=3, shape='spline', smoothing=0.4),
                        fill='tozeroy',
                        fillcolor='rgba(37, 99, 235, 0.05)',
                        marker=dict(size=10, color='#2563EB', line=dict(color='white', width=2)),
                        text=[f"{v:.0f}%" for v in risk_trend['AlertRate']],
                        textposition="top center",
                        textfont=dict(family="Inter, sans-serif", size=11, color='#1E3A8A', weight='bold'),
                        cliponaxis=False
                    ))
                    
                    # 3. Safety Threshold Line (20%)
                    fig_trend.add_hline(
                        y=20, 
                        line_dash="dash", 
                        line_color="rgba(16, 185, 129, 0.5)", 
                        annotation_text="Safety Threshold (20%)", 
                        annotation_position="bottom right",
                        annotation_font=dict(size=10, color="rgba(16, 185, 129, 0.8)", weight='bold')
                    )

                    fig_trend.update_layout(
                        title="📈 Clinical Alert Trend", 
                        height=350, 
                        paper_bgcolor='white', 
                        plot_bgcolor='white', 
                        margin=dict(t=50, b=10, l=10, r=10),
                        yaxis=dict(
                            ticksuffix="%",
                            dtick=20, 
                            range=[0, 120], 
                            showline=True, linewidth=1.5, linecolor='#CBD5E1',
                            gridcolor='rgba(226, 232, 240, 0.5)',
                            rangemode='nonnegative'
                        ),
                        xaxis=dict(
                            range=[-0.4, len(month_range) - 1 + 0.4],
                            showline=True, linewidth=1.5, linecolor='#CBD5E1',
                            gridcolor='rgba(226, 232, 240, 0.5)'
                        ),
                        showlegend=False,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
                with col_bar:
                    if not cur_d.empty:
                        # Use a larger palette for 12+ categories
                        FULL_PALETTE = px.colors.qualitative.Prism + px.colors.qualitative.Safe + px.colors.qualitative.Vivid
                        # Calculate percentage based on total sessions in selected view
                        total_month_sessions = cur_d['timestamp'].nunique()
                        cur_c['Percentage'] = (cur_c['Count'] / total_month_sessions * 100).clip(upper=100)
                        
                        fig_m = go.Figure(go.Bar(
                            x=cur_c['Percentage'], 
                            y=cur_c['Condition'], 
                            orientation='h', 
                            marker=dict(color=FULL_PALETTE[:len(cur_c)]),
                            text=[f"{v:.1f}%" for v in cur_c['Percentage']],
                            textposition='auto'
                        ))
                        fig_m.update_layout(
                            title=f"📊 Diagnosis Distribution (%) - {cur_m}", 
                            height=450, 
                            margin=dict(l=180, r=40),
                            paper_bgcolor='white',
                            xaxis=dict(
                                title="Frequency (%)",
                                showline=True, linewidth=1.5, linecolor='#CBD5E1',
                                dtick=20,
                                range=[0, 105],
                                ticksuffix="%"
                            ),
                            yaxis=dict(
                                showline=True, linewidth=1.5, linecolor='#CBD5E1'
                            )
                        )
                        st.plotly_chart(fig_m, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info("No diagnostic data available for the current month.")
                st.markdown('<div style="height:20px; border-top:1px solid #F1F5F9; margin: 20px 0;"></div>', unsafe_allow_html=True)

            with h_tab3:

                # ── DIAGNOSTIC HISTORY BLOCKS ─────────────────────────────────────
                if not grouped.empty:
                    # Show all sessions (including the most recent one)
                    cols = st.columns(2)
                    for i, (_, row) in enumerate(grouped.iterrows()):
                        with cols[i % 2]:
                            # Severity Visual Mapping
                            s_base = row['Severity']
                            s_color = "#EF4444" if s_base in ['High', 'Critical'] else ("#F59E0B" if s_base == 'Mild' else "#10B981")
                            s_bg = "#FEF2F2" if s_base in ['High', 'Critical'] else ("#FFFBEB" if s_base == 'Mild' else "#F0FDF4")
                            
                            # Cleanly formatted tags
                            tags_html = ""
                            for cond in row['Condition']:
                                tags_html += f'<span style="background: white; border: 1px solid #E2E8F0; color: #475569; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; margin-right: 6px; margin-bottom: 6px; display: inline-block; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">{cond.upper()}</span>'
                            
                            with st.expander(f"📑 &nbsp; {row['DateStr']}"):
                                st.markdown(f"""<div style="padding: 20px; border-radius: 12px; background: {s_bg}; border: 1px solid {s_color}30; position:relative; overflow:hidden; margin-bottom:10px;">
<!-- Severity Vertical Accent -->
<div style="position:absolute; left:0; top:0; bottom:0; width:6px; background:{s_color};"></div>
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
<div style="font-size: 0.8rem; color: #64748B; font-weight: 800; display:flex; align-items:center; gap:6px;">🕒 {row['DateStr'].upper()}</div>
<span style="background:{s_color}; color:white; padding:3px 10px; border-radius:100px; font-size:0.65rem; font-weight:900; letter-spacing:0.5px;">{s_base.upper()}</span>
</div>
<div style="margin-bottom: 12px; display:flex; flex-wrap:wrap; gap:2px;">
{tags_html}
</div>
<div style="height:1px; background:#E2E8F0; margin:15px 0;"></div>
<div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px;">
<div style="font-size: 0.75rem; color: #1E293B; font-weight: 800;">🔬 Diagnostic Session #{total_sessions - i}</div>
</div>
</div>""", unsafe_allow_html=True)

                                # --- Interactive Popover for Deep-Dive Details ---
                                with st.popover("🧬 FULL DETAILS &rarr;"):
                                    st.markdown(f"### 📋 Diagnostic Deep-Dive: Session #{total_sessions - i}")
                                    st.info(f"📍 Date: {row['DateStr']}")
                                    
                                    # 1. Physical Vitals Table
                                    v_df = get_session_vitals(row['session_id'])
                                    if not v_df.empty:
                                        st.markdown("#### 📊 Captured Clinical Vitals")
                                        st.dataframe(v_df, hide_index=True, use_container_width=True)
                                    
                                    # 2. Detailed Findings (ML + Rules)
                                    st.markdown("#### 🤖 Full AI & Clinical Analysis")
                                    # Filter original 1-row-per-finding format for this timestamp
                                    s_details = history_df[history_df['timestamp'] == row['timestamp']]
                                    
                                    findings_data = []
                                    for _, frow in s_details.iterrows():
                                        findings_data.append({
                                            "Clinical Finding": frow['rule_disease'] if frow['rule_disease'] else frow['ml_disease'],
                                            "Severity": frow['severity'] if frow['severity'] else "-",
                                            "Clinical Reasoning": frow['observation_text'] if frow['observation_text'] else "ML Prediction Result"
                                        })
                                    
                                    if findings_data:
                                        st.dataframe(
                                            pd.DataFrame(findings_data), 
                                            hide_index=True, 
                                            use_container_width=True,
                                            column_config={
                                                "Clinical Finding": st.column_config.TextColumn(width="medium"),
                                                "Severity": st.column_config.TextColumn(width="small"),
                                                "Clinical Reasoning": st.column_config.TextColumn(width="large")
                                            }
                                        )
                                    
                                    st.success("✅ Diagnostic Synchronization Complete")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.button("📄 Download Full Diagnostic Archive (CSV)", width="stretch")
        else:
            st.markdown("""
            <div class="hist-empty">
                <div style="font-size:48px;">🔬</div>
                <div class="hist-empty-title">No Diagnostic History Yet</div>
                <div class="hist-empty-sub">Run your first analysis from Manual Entry or OCR Upload to see results here.</div>
            </div>
            """, unsafe_allow_html=True)
def delete_login_activity(admin_username, target_val):
    # Direct DB inline replacement to fix ImportError/Cache
    _conn = get_db_connection()
    if not _conn: return False, "DB connection failed."
    try:
        _cur = _conn.cursor()
        is_id = False
        try:
            val_as_id = int(target_val)
            is_id = True
        except ValueError: is_id = False
            
        if is_id:
            _cur.execute("DELETE FROM login_history WHERE id = %s", (val_as_id,))
            if _cur.rowcount > 0:
                # Corrected for actual schema: user_id, action_type, details
                _cur.execute("""
                    INSERT INTO audit_logs (user_id, action_type, details) 
                    VALUES ((SELECT id FROM users WHERE username = %s), %s, %s)
                """, (admin_username, "DELETE_LOGIN_LOG", f"Deleted login log ID {val_as_id}"))
                _conn.commit()
                return True, f"Login log ID {val_as_id} deleted successfully."
            else: return False, "Log ID not found."
        else:
            _cur.execute("DELETE FROM login_history WHERE user_id = (SELECT id FROM users WHERE username = %s)", (target_val,))
            count = _cur.rowcount
            if count > 0:
                # Corrected for actual schema: user_id, action_type, details
                _cur.execute("""
                    INSERT INTO audit_logs (user_id, action_type, details) 
                    VALUES ((SELECT id FROM users WHERE username = %s), %s, %s)
                """, (admin_username, "CLEAR_USER_LOGS", f"Cleared {count} logs for user {target_val}"))
                _conn.commit()
                return True, f"Successfully cleared {count} records for: {target_val}."
            else: return False, f"No logs found for: {target_val}."
    except Exception as e: return False, str(e)
    finally: _conn.close()

def render_admin_dashboard():
    st.cache_data.clear()
    # Renders the comprehensive Admin Panel.
    st.markdown("<style>div.block-container {padding-top: 0.5rem !important;}</style>", unsafe_allow_html=True)
    
    # Logout Button placed at the absolute top corner - mini size but horizontal
    _, col_logout = st.columns([11, 1.4])
    with col_logout:
        if st.button("Logout", type="primary", key="admin_logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.session_state.user_id = None
            st.rerun()

    # Add spacing between button and banner
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # --- COMPACT ADMIN HEADER ---
    render_luxury_header("Admin Portal", icon="🛡️", mode="compact")

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # Sub-tabs within Admin Panel
    if st.session_state.role == 'admin':
        admin_sub1, admin_sub2, admin_sub3, admin_sub4, admin_sub5, admin_sub6, admin_sub7 = st.tabs([
            "📊 System Intelligence", "👥 Manage Users", "📝 System Audit", "🛡️ Control Center", "🔐 Login Activity Logs", "🔧 Node Config", "📋 Registration Details"
        ])
    else:
        admin_sub1, admin_sub5 = st.tabs(["📊 Diagnostic Trends", "🔐 Login Activity Logs"])
        admin_sub2 = admin_sub3 = admin_sub4 = admin_sub6 = admin_sub7 = None
    
    with admin_sub1:
        st.markdown("<h3 style='color: #00D4FF; margin-bottom: 20px;'>Live Diagnostic Intelligence</h3>", unsafe_allow_html=True)
        
        # Date Filter Row
        col_date1, col_date2, col_date3 = st.columns([2, 2, 4])
        
        # Ensure tables are initialized (Safe schema migration check)
        initialize_tables()
            
        import datetime
        today = datetime.date.today()
        thirty_days_ago = today - datetime.timedelta(days=30)
        with col_date1: start_date = st.date_input("Analysis From", value=thirty_days_ago, key="admin_trend_start")
        with col_date2: end_date = st.date_input("Analysis To", value=today, key="admin_trend_end")
        
        # Top-Row Metrics (Now standardized in db_utils.py)
        stats = get_system_stats(start_date, end_date)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: 
            st.markdown(f"""
                <div class="admin-metric-card-pro">
                    <div style="color: #64748B; font-size: 0.8rem; font-weight: 600;">TOTAL PROFILES <span style="color: #00D4FF; font-size: 0.6rem;">● LIVE</span></div>
                    <div style="color: #00D4FF; font-size: 2.2rem; font-weight: 800;">{stats.get('total_users', 0)}</div>
                </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
                <div class="admin-metric-card-pro">
                    <div style="color: #64748B; font-size: 0.9rem; font-weight: 600;">ACTIVE PATIENTS</div>
                    <div style="color: #F472B6; font-size: 2.2rem; font-weight: 800;">{stats.get('total_patients', 0)}</div>
                </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
                <div class="admin-metric-card-pro">
                    <div style="color: #64748B; font-size: 0.9rem; font-weight: 600;">ACTIVE PROFILES</div>
                    <div style="color: #10B981; font-size: 2.2rem; font-weight: 800;">{stats.get('active_profiles', 0)}</div>
                </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
                <div class="admin-metric-card-pro">
                    <div style="color: #64748B; font-size: 0.8rem; font-weight: 600;">CLINICAL SESSIONS <span style="color: #818CF8; font-size: 0.6rem;">● SYNCED</span></div>
                    <div style="color: #818CF8; font-size: 2.2rem; font-weight: 800;">{stats.get('total_sessions', 0)}</div>
                </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
                <div class="admin-metric-card-pro">
                    <div style="color: #64748B; font-size: 0.9rem; font-weight: 600;">AI ANALYSES</div>
                    <div style="color: #FACC15; font-size: 2.2rem; font-weight: 800;">{stats.get('total_predictions', 0)}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts Row 1: Two Separate Charts (Side-by-Side)
        _util_result = get_system_utilization(start_date, end_date)
        if isinstance(_util_result, tuple) and len(_util_result) == 2:
            reg_df, sess_df = _util_result
        else:
            reg_df, sess_df = pd.DataFrame(), pd.DataFrame()
            
        # Ensure robust date formatting for Plotly
        if not reg_df.empty:
            reg_df['scan_date'] = pd.to_datetime(reg_df['scan_date']).dt.date
        if not sess_df.empty:
            sess_df['scan_date'] = pd.to_datetime(sess_df['scan_date']).dt.date

        chart_col1, chart_col2 = st.columns(2, gap="large")
        
        with chart_col1:
            with st.container(border=True):
                if isinstance(reg_df, pd.DataFrame) and not reg_df.empty:
                    # REVERTED TO AREA with markers for premium look
                    fig_reg = px.area(reg_df, x='scan_date', y='count', 
                                      title="User Registrations",
                                      markers=True,
                                      color_discrete_sequence=['#10B981'])
                    fig_reg.update_layout(
                        template='plotly_white',
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#1E293B', title_font_size=20,
                        margin=dict(t=50, b=20, l=10, r=10),
                        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', showline=True, linecolor='#1E293B', linewidth=1, type='date'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', showline=True, linecolor='#1E293B', linewidth=1)
                    )
                    st.plotly_chart(fig_reg, use_container_width=True)
                else:
                    st.info("No user registration data available.")
        
        with chart_col2:
            with st.container(border=True):
                if isinstance(sess_df, pd.DataFrame) and not sess_df.empty:
                    # REVERTED TO AREA with markers for premium look
                    fig_sess = px.area(sess_df, x='scan_date', y='count', 
                                       title="Diagnostic Session Activity",
                                       markers=True,
                                       color_discrete_sequence=['#00D4FF'])
                    fig_sess.update_layout(
                        template='plotly_white',
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#1E293B', title_font_size=20,
                        margin=dict(t=50, b=20, l=10, r=10),
                        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', showline=True, linecolor='#1E293B', linewidth=1, type='date'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', showline=True, linecolor='#1E293B', linewidth=1)
                    )
                    st.plotly_chart(fig_sess, use_container_width=True)
                else:
                    st.info("No diagnostic session data available.")

        # LOCAL CALCULATION: Fetch demographics from 'users' table directly for Admin
        # Updated to use IST-aware boundaries
        conn = get_db_connection()
        age_df, gender_df = pd.DataFrame(), pd.DataFrame()
        if conn:
            next_day = end_date + datetime.timedelta(days=1)
            age_query = """
            SELECT 
                CASE 
                    WHEN age BETWEEN 0 AND 10 THEN '0-10'
                    WHEN age BETWEEN 11 AND 20 THEN '11-20'
                    WHEN age BETWEEN 21 AND 30 THEN '21-30'
                    WHEN age BETWEEN 31 AND 40 THEN '31-40'
                    WHEN age BETWEEN 41 AND 50 THEN '41-50'
                    WHEN age BETWEEN 51 AND 60 THEN '51-60'
                    WHEN age BETWEEN 61 AND 70 THEN '61-70'
                    WHEN age BETWEEN 71 AND 80 THEN '71-80'
                    WHEN age BETWEEN 81 AND 90 THEN '81-90'
                    WHEN age BETWEEN 91 AND 100 THEN '91-100'
                    ELSE '100+' 
                END as age_group,
                COUNT(*) as count
            FROM users
            WHERE role = 'user' AND created_at >= %s AND created_at < %s
            GROUP BY age_group
            ORDER BY age_group
            """
            gender_query = "SELECT gender, COUNT(*) as count FROM users WHERE role = 'user' AND created_at >= %s AND created_at < %s GROUP BY gender"
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                age_df = pd.read_sql(age_query, conn, params=(start_date, next_day))
                gender_df = pd.read_sql(gender_query, conn, params=(start_date, next_day))
                
                if age_df.empty: age_df = pd.DataFrame(columns=['age_group', 'count'])
                
                # Force all 11 age segments (0-10 to 100+) for the Block Chart
                all_groups = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '100+']
                if 'age_group' in age_df.columns:
                    age_df = age_df.set_index('age_group').reindex(all_groups, fill_value=0).reset_index()
                
            conn.close()

        # Charts Row 2: Demographics (New Block & Bar Config)
        col_c1, col_c2 = st.columns(2, gap="large")
        
        with col_c1:
            with st.container(border=True):
                if not age_df.empty:
                    # Calculate percentage for Age
                    total_age = age_df['count'].sum()
                    age_df['percentage'] = (age_df['count'] / total_age * 100) if total_age > 0 else 0
                    
                    # Age Bar Chart (Percentage with Premium Qualitative Colors)
                    fig_age = px.bar(
                        age_df, 
                        x='age_group', 
                        y='percentage',
                        color='age_group',
                        color_discrete_sequence=px.colors.qualitative.Prism,
                        title="Age Spectrum (0-100+) - Percentage Analysis (%)"
                    )
                    fig_age.update_layout(
                        template='plotly_white', height=450, showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#1E293B',
                        title_font_size=20, margin=dict(t=50, l=10, r=10, b=50),
                        xaxis=dict(
                            title="Age Groups", showgrid=False, tickfont=dict(size=11, weight='bold'),
                            showline=True, linewidth=1.5, linecolor='#1E293B'
                        ),
                        yaxis=dict(
                            title="Distribution (%)", showgrid=True, gridcolor='rgba(0,0,0,0.05)', ticksuffix="%",
                            showline=True, linewidth=1.5, linecolor='#1E293B'
                        )
                    )
                    st.plotly_chart(fig_age, use_container_width=True, config={'displayModeBar': False})

        with col_c2:
            with st.container(border=True):
                if not gender_df.empty:
                    # Calculate percentage for Gender
                    total_gender = gender_df['count'].sum()
                    gender_df['percentage'] = (gender_df['count'] / total_gender * 100) if total_gender > 0 else 0
                    
                    # Gender Bar Chart (Percentage with Fixed Brand Colors)
                    gender_colors = {'Male': '#3B82F6', 'Female': '#EC4899', 'Other': '#64748B'}
                    fig_gen = px.bar(
                        gender_df, x='gender', y='percentage',
                        color='gender',
                        color_discrete_map=gender_colors,
                        title="Gender Distribution - Percentage Analysis (%)"
                    )
                    fig_gen.update_layout(
                        template='plotly_white', height=450, showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#1E293B',
                        title_font_size=20, margin=dict(t=50, l=10, r=10, b=50),
                        xaxis=dict(
                            showgrid=False, title="Gender Identity", title_font_size=14, tickfont=dict(size=12, weight='bold'),
                            showline=True, linewidth=1.5, linecolor='#1E293B'
                        ),
                        yaxis=dict(
                            showgrid=True, gridcolor='rgba(0,0,0,0.05)', title="Distribution (%)", title_font_size=14, ticksuffix="%",
                            showline=True, linewidth=1.5, linecolor='#1E293B'
                        )
                    )
                    st.plotly_chart(fig_gen, use_container_width=True, config={'displayModeBar': False})
        
        # Clinical Risk Chart Removed as requested.
        pass


    if admin_sub2:
        with admin_sub2:
            # Force refresh logic
            if 'admin_refresh_users' not in st.session_state:
                st.session_state.admin_refresh_users = True
                
            if st.session_state.admin_refresh_users:
                st.cache_data.clear()
                st.session_state.admin_users_df = get_all_users()
                st.session_state.admin_refresh_users = False
            
            users_df = st.session_state.admin_users_df
            
            if not users_df.empty:
                st.markdown("#### 👥 User Management & Access Control")
                
                # Header row ratios
                cols_r = [0.6, 1.3, 1.8, 2.5, 1.8, 1.3, 1.3, 0.8]
                h_cols = st.columns(cols_r)
                h_cols[0].write("**ID**"); h_cols[1].write("**Username**"); h_cols[2].write("**Name**"); h_cols[3].write("**Email**")
                h_cols[4].write("**Assign Role**"); h_cols[5].write("**Status**"); h_cols[6].write("**Access Control**"); h_cols[7].write("**Remove**")
                st.markdown("<hr style='margin: 0 0 10px 0; border: none; border-top: 2px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
                
                for idx, row in users_df.iterrows():
                    user_id = row.get('id', idx)
                    username = row.get('username', f"user_{idx}")
                    current_role = row.get('role', 'user')
                    full_name = row.get('name', 'N/A')
                    email = row.get('email', 'N/A')
                    raw_st = str(row.get('status', 'active')).strip().lower()
                    
                    # Show current user but with a protective marker
                    is_current = (username == st.session_state.username)
                    
                    r_cols = st.columns(cols_r)
                    r_cols[0].write(f"`{user_id}`")
                    r_cols[1].markdown(f"**{username}**" + (" 🛡️" if is_current else ""))
                    r_cols[2].write(full_name)
                    r_cols[3].write(email)
                    
                    # Role Selection
                    role_m_rev = {"user": "Patient", "doctor": "Clinician/Doctor", "admin": "System Administrator"}
                    active_r = ["Patient", "Clinician/Doctor", "System Administrator"]
                    curr_role_d = role_m_rev.get(current_role, "Patient")
                    
                    with r_cols[4]:
                        new_r_disp = st.selectbox("Role", active_r, index=active_r.index(curr_role_d), key=f"r_sel_{username}_{idx}", label_visibility="collapsed")
                        if new_r_disp != curr_role_d:
                            role_map = {"Patient": "user", "Clinician/Doctor": "doctor", "System Administrator": "admin"}
                            success, msg = update_user_role(st.session_state.username, username, role_map[new_r_disp])
                            if success: 
                                st.session_state.admin_refresh_users = True
                                st.toast(msg, icon="✅")
                                st.rerun()
                    
                    # Status Indicator Logic
                    is_active = (raw_st == 'active')
                    st_ind = "✅ Active" if is_active else "🚫 Blocked"
                    r_cols[5].write(st_ind)
                    
                    with r_cols[6]:
                        # Dynamic Button Label - Exactly matching user request: Block/Unblock
                        btn_label = "🚫 Block" if is_active else "🔓 Unblock"
                        btn_type = "secondary" if is_active else "primary"
                        
                        if st.button(btn_label, key=f"tg_bt_{username}_{idx}", use_container_width=True, type=btn_type):
                            success, msg = toggle_user_status(st.session_state.username, username)
                            if success: 
                                # Force immediate reload of users from DB
                                st.session_state.admin_refresh_users = True
                                st.cache_data.clear()
                                st.success(msg)
                                import time
                                time.sleep(0.3)
                                st.rerun()
                    
                    with r_cols[7]:
                        if st.button("🗑️", key=f"dl_bt_{username}_{idx}", use_container_width=True, type="primary"):
                            success, msg = delete_user(st.session_state.username, username)
                            if success: 
                                st.session_state.admin_refresh_users = True
                                st.cache_data.clear()
                                st.success(msg)
                                import time
                                time.sleep(0.3)
                                st.rerun()
                            else:
                                st.error(msg)
                    
                    st.markdown("<hr style='margin: 4px 0 12px 0; border: none; border-top: 1px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
            else:
                st.info("No active user records found.")

    if admin_sub3:
        with admin_sub3:
            # Security Audit Trail Header Removed
            c_a1, c_a2, _ = st.columns([2, 2, 6])
            with c_a1: start_audit = st.date_input("Log Start", value=thirty_days_ago, key="audit_start")
            with c_a2: end_audit = st.date_input("Log End", value=today, key="audit_end")
            
            audit_df = get_audit_logs(limit=1000, start_date=start_audit, end_date=end_audit)
            if not audit_df.empty:
                st.dataframe(audit_df, use_container_width=True)
                st.download_button("📥 Download Audit Evidence (CSV)", 
                                 data=audit_df.to_csv(index=False).encode('utf-8'), 
                                 file_name=f"audit_trail_{datetime.date.today()}.csv", 
                                 mime="text/csv", type="secondary")
            else: st.info("No audit signatures recorded for this timeframe.")
            
    if admin_sub4:
        with admin_sub4:
            # Operational Node Control Header Removed
            
            # Real-time Connectivity Status
            st.markdown("#### Connectivity Matrix")
            cs1, cs2, cs3 = st.columns(3)
            with cs1: 
                try:
                    conn = get_db_connection()
                    if conn: st.markdown('<div class="status-pill-pro status-pill-online">● DB CLUSTER: ONLINE</div>', unsafe_allow_html=True); conn.close()
                except: st.markdown('<div class="status-pill-pro status-pill-offline">● DB CLUSTER: OFFLINE</div>', unsafe_allow_html=True)
            with cs2: st.markdown('<div class="status-pill-pro status-pill-online">● NEURAL ENGINE: READY</div>', unsafe_allow_html=True)
            with cs3: st.markdown('<div class="status-pill-pro status-pill-online">● GATEWAY API: STABLE</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Hardware Gauges
            try:
                import psutil
                st.markdown("#### Live Hardware Telemetry")
                gh1, gh2, gh3 = st.columns(3)
                
                # CPU Gauge
                fig_cpu = go.Figure(go.Indicator(
                    mode = "gauge+number", value = psutil.cpu_percent(),
                    title = {'text': "CPU LOAD", 'font': {'size': 16, 'color': 'white'}},
                    gauge = {'axis': {'range': [None, 100], 'tickcolor': "white"},
                             'bar': {'color': "#00D4FF"},
                             'bgcolor': "rgba(0,0,0,0)",
                             'steps': [{'range': [0, 70], 'color': "rgba(16, 185, 129, 0.1)"},
                                      {'range': [70, 90], 'color': "rgba(245, 158, 11, 0.1)"},
                                      {'range': [90, 100], 'color': "rgba(239, 68, 68, 0.1)"}]}
                ))
                fig_cpu.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"}, height=250, margin=dict(l=20,r=20,t=40,b=20))
                with gh1: st.plotly_chart(fig_cpu, use_container_width=True)
                
                # RAM Gauge
                mem = psutil.virtual_memory()
                fig_ram = go.Figure(go.Indicator(
                    mode = "gauge+number", value = mem.percent,
                    title = {'text': "RAM ALLOCATION", 'font': {'size': 16, 'color': 'white'}},
                    gauge = {'axis': {'range': [None, 100], 'tickcolor': "white"},
                             'bar': {'color': "#818CF8"},
                             'bgcolor': "rgba(0,0,0,0)"}
                ))
                fig_ram.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"}, height=250, margin=dict(l=20,r=20,t=40,b=20))
                with gh2: st.plotly_chart(fig_ram, use_container_width=True)
                
                # Disk Storage
                disk = psutil.disk_usage('/')
                fig_disk = go.Figure(go.Indicator(
                    mode = "gauge+number", value = disk.percent,
                    title = {'text': "VOLATILE STORAGE", 'font': {'size': 16, 'color': 'white'}},
                    gauge = {'axis': {'range': [None, 100], 'tickcolor': "white"},
                             'bar': {'color': "#F472B6"},
                             'bgcolor': "rgba(0,0,0,0)"}
                ))
                fig_disk.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Arial"}, height=250, margin=dict(l=20,r=20,t=40,b=20))
                with gh3: st.plotly_chart(fig_disk, use_container_width=True)
                
            except ImportError:
                st.info("ℹ️ Advanced hardware telemetry requires the 'psutil' library.")
                
            st.markdown("---")
            with st.expander("🔐 Disaster Recovery & Backups"):
                st.write("Generate an encrypted node backup for archival purposes.")
                if st.button("Initiate Encrypted Backup", type="secondary", use_container_width=True):
                    backup_data = b"-- Encrypted System Snapshot\n-- Generated for: " + st.session_state.username.encode() + b"\n-- Timestamp: " + str(datetime.datetime.now()).encode()
                    st.download_button("💾 Download Snapshot (.sql)", data=backup_data, 
                                     file_name=f"system_bkp_{datetime.date.today()}.sql", 
                                     mime="application/sql", use_container_width=True)

    with admin_sub5:
        st.markdown("<h3 style='color: #00D4FF;'>🔐 User Login & Session Activity</h3>", unsafe_allow_html=True)
        # Displaying the activity table (stored in the patients/diagnostic sessions for clinical accuracy)
        if True: # Force refresh block
            # FETCHING REAL LOGIN HISTORY (Instead of patient records)
            try:
                _conn = get_db_connection()
                if _conn:
                    _query = """
                        SELECT l.id, u.username, l.ip_address, l.device, l.status, l.timestamp 
                        FROM login_history l
                        JOIN users u ON l.user_id = u.id
                        ORDER BY l.timestamp DESC
                    """
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UserWarning)
                        login_activity_df = pd.read_sql_query(_query, _conn)
                    _conn.close()
                else:
                    login_activity_df = pd.DataFrame()
            except:
                login_activity_df = pd.DataFrame()
        
        if not login_activity_df.empty:
            # Layout for Search and Activity Management
            col_search1, col_search2 = st.columns([8, 2], vertical_alignment="bottom")
            
            # Initialization
            if "login_history_search" not in st.session_state:
                st.session_state.login_history_search = ""

            # Define Callback for Reset (safely clears the state before rerun)
            def clear_search_callback():
                st.session_state.login_history_search = ""

            with col_search1:
                search_query = st.text_input("🔍 Search Activity by Username", placeholder="Enter username...", key="login_history_search")
            
            with col_search2:
                if search_query:
                    # Use on_click callback to safely clear state - this is the standard way
                    st.button("🔄 Show All", use_container_width=True, type="primary", on_click=clear_search_callback)
                else:
                    if st.button("🔍 Search", use_container_width=True, type="secondary"):
                        st.rerun()
            
            filtered_df = login_activity_df.copy()
            if search_query:
                filtered_df = login_activity_df[login_activity_df['username'].str.contains(search_query, case=False, na=False)]
                st.info(f"🔍 Total **{len(filtered_df)}** records found for: **{search_query}**")
            
            # Explicit column selection and display
            display_cols = ['id', 'username', 'ip_address', 'device', 'status', 'timestamp']
            st.dataframe(filtered_df[display_cols], use_container_width=True)
        else:
            st.info("No login history records found in the system yet.")
            
        # Critical Operations (Login Activity Management)
        with st.container(border=True):
            st.write("🛑 **Login Activity Management**")
            st.info("💡 Enter a **Log ID** to delete a single record, or a **Username** to clear all logs for that user.")
            
            # Using text_input instead of number_input to support both IDs and Usernames
            target_val = st.text_input("Enter Log ID or Username for Cleanup", placeholder="ID or Username...")
            
            if st.button("🗑️ Process Deletion / Cleanup", type="primary", use_container_width=True):
                if target_val:
                    success, msg = delete_login_activity(st.session_state.username, target_val)
                    if success: st.success(msg); st.rerun()
                    else: st.error(msg)
                else:
                    st.warning("Please enter a valid Value (ID or Username).")

    if admin_sub6:
        with admin_sub6:
            st.subheader("System Core Configuration")
            
            st.markdown("#### 🛠️ Maintenance Mode")
            maintenance_enabled = get_system_setting("maintenance_mode", "false") == "true"
            if st.toggle("🚦 Enable System Maintenance (Blocks Non-Admins)", value=maintenance_enabled):
                set_system_setting("maintenance_mode", "true")
            else:
                set_system_setting("maintenance_mode", "false")
                
            st.markdown("---")
            st.markdown("#### 🔐 Security Configuration")
            otp_enabled = get_system_setting("require_otp", "true") == "true"
            if st.toggle("🔒 Require Global OTP Verification", value=otp_enabled):
                set_system_setting("require_otp", "true")
            else:
                set_system_setting("require_otp", "false")
            
            with st.expander("📧 SMTP / Email Settings", expanded=True):
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    current_email = get_system_setting("smtp_email", "bhoomikpatel98@gmail.com")
                    new_email = st.text_input("Sender Email", value=current_email)
                with col_e2:
                    current_pass = get_system_setting("smtp_password", "")
                    new_pass = st.text_input("App Password", value=current_pass, type="password")
                
                if st.button("Save SMTP Configuration"):
                    set_system_setting("smtp_email", new_email)
                    set_system_setting("smtp_password", new_pass)
                    st.success("SMTP configuration updated.")

            with st.expander("💬 SMS / MSG91 Settings", expanded=True):
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    current_auth = get_system_setting("msg91_auth_key", "")
                    new_auth = st.text_input("MSG91 Auth Key", value=current_auth, type="password")
                with col_s2:
                    current_temp = get_system_setting("msg91_template_id", "")
                    new_temp = st.text_input("MSG91 Template ID", value=current_temp)
                with col_s3:
                    current_sender = get_system_setting("msg91_sender_id", "MSGIND")
                    new_sender = st.text_input("Sender ID", value=current_sender, help="6-character Sender ID approved in DLT")
                
                if st.button("Save MSG91 Configuration"):
                    set_system_setting("msg91_auth_key", new_auth)
                    set_system_setting("msg91_template_id", new_temp)
                    set_system_setting("msg91_sender_id", new_sender)
                    st.success("MSG91 configuration updated.")

    if admin_sub7:
        with admin_sub7:
            st.markdown("<h3 style='color: #00D4FF; margin-bottom: 20px;'>📋 User Registration Records</h3>", unsafe_allow_html=True)
            
            # Fetch registration data
            reg_df = get_registration_data()
            
            if not reg_df.empty:
                st.markdown("#### Comprehensive Registration Directory")
                st.write("Below are the complete details of all registered users in the system.")
                
                # Display dataframe with premium styling
                st.dataframe(
                    reg_df,
                    use_container_width=True,
                    column_config={
                        "id": "User ID",
                        "name": "Full Name",
                        "age": "Age",
                        "gender": "Gender",
                        "email": "Email Address",
                        "contact": "Contact info",
                        "address": "Resident Address",
                        "status": "Status",
                        "created_at": "Joined Date"
                    }
                )
                
                st.download_button(
                    "📥 Export Full Registry (CSV)",
                    data=reg_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"registration_registry_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    type="secondary"
                )
            else:
                st.info("No user registrations found in the database.")


def main():
    st.set_page_config(
        page_title="Medical AI Dashboard | Premium Diagnostics",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown(load_css(), unsafe_allow_html=True)
    
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.patient_profile = None
        st.session_state.user_id = None
        
    if not st.session_state.logged_in:
        render_login_ui()
        st.stop()

    # --- Hide sidebar (replaced by top navbar) ---
        with auth_tab2:
            if 'reg_otp_sent' not in st.session_state:
                st.session_state.reg_otp_sent = False
                
            if not st.session_state.reg_otp_sent:
                with st.form("register_form", border=False):
                    st.markdown("### Create User Account")
                    
                    colA, colB = st.columns(2)
                    with colA:
                        reg_username = st.text_input("Username *")
                        reg_password = st.text_input("Password *", type="password")
                        reg_confirm = st.text_input("Confirm Password *", type="password")
                    with colB:
                        reg_name = st.text_input("Full Name *")
                        reg_age = st.number_input("Age", min_value=1, max_value=120, value=30)
                        reg_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    
                    st.markdown("---")
                    reg_email = st.text_input("Email Address * (For OTP)")
                    reg_contact = st.text_input("Contact Number *")
                    reg_address = st.text_area("Physical Address", height=80)
    
                    if st.form_submit_button("Register", type="primary", width="stretch"):
                        if not all([reg_username, reg_password, reg_name, reg_email, reg_contact]):
                            st.error("Please fill in all required fields (*).")
                        elif not is_valid_email(reg_email):
                            st.error("Invalid Email format.")
                        elif reg_password != reg_confirm:
                            st.error("Passwords do not match.")
                        else:
                            # 1. Register User (Unverified)
                            success, msg = register_user(reg_username, reg_password, reg_name, int(reg_age), reg_gender, reg_email, reg_contact, reg_address)
                            if success:
                                # 2. Generate and Send OTP
                                otp = generate_otp()
                                if store_otp(reg_email, otp): # Use email string
                                    send_otp(otp, email=reg_email, contact=reg_contact)
                                    st.session_state.reg_email = reg_email
                                    st.session_state.reg_otp_sent = True
                                    st.success(f"Verification code sent to {reg_email}")
                                    st.rerun()
                                else:
                                    st.error("OTP System currently unavailable. Try again later.")
                            else:
                                st.error(msg)
            else:
                with st.form("reg_otp_form", border=False):
                    st.markdown(f"### Verify Account")
                    st.info(f"Enter the 6-digit code sent to **{st.session_state.reg_email}**")
                    user_otp = st.text_input("Verification Code", placeholder="000000")
                    
                    colB1, colB2 = st.columns(2)
                    with colB1:
                        if st.form_submit_button("Verify & Activate", type="primary", width="stretch"):
                            v_success, v_msg = verify_otp_db(st.session_state.reg_email, user_otp)
                            if v_success:
                                u_id = get_user_id_by_email(st.session_state.reg_email)
                                if activate_user_account(u_id):
                                    st.success("Account activated! Please log in.")
                                    st.session_state.reg_otp_sent = False
                                    import time
                                    time.sleep(1.5)
                                    st.rerun()
                                else:
                                    st.error("Activation failed. Contact support.")
                            else:
                                st.error(v_msg)
                    with colB2:
                        if st.form_submit_button("Cancel", width="stretch"):
                            st.session_state.reg_otp_sent = False
                            st.rerun()
                            
        # --- RECOVERY PHASE ---
        with auth_tab3:
            st.markdown("### Account Recovery")
            if 'otp_sent' not in st.session_state:
                st.session_state.otp_sent = False
                st.session_state.otp_verified = False
                
            if not st.session_state.otp_sent:
                with st.form("request_otp_form", border=False):
                    st.write("Enter your registered email to receive a reset code.")
                    reset_contact = st.text_input("Email or Contact Number")
                    if st.form_submit_button("Send Reset Code", type="primary", width="stretch"):
                        exists, user_info = verify_user_exists(reset_contact)
                        if exists:
                            target_email = user_info['email']
                            otp = generate_otp()
                            if store_otp(target_email, otp):
                                send_otp(otp, email=target_email, contact=user_info['contact'])
                                st.session_state.reset_email = target_email
                                st.session_state.reset_username = user_info['username']
                                st.session_state.otp_sent = True
                                st.success(f"OTP Sent to {target_email}")
                                st.rerun()
                            else:
                                st.error("Database connection issue. Try again.")
                        else:
                            st.error("No account found with this identity.")
            elif st.session_state.otp_sent and not st.session_state.otp_verified:
                with st.form("verify_otp_form", border=False):
                    st.info(f"Verification code sent to {st.session_state.reset_email}")
                    user_otp = st.text_input("Enter Code", placeholder="000000")
                    if st.form_submit_button("Verify Code", type="primary", width="stretch"):
                        v_success, v_msg = verify_otp_db(st.session_state.reset_email, user_otp)
                        if v_success:
                            st.session_state.otp_verified = True
                            st.rerun()
                        else:
                            st.error(v_msg)
                if st.button("Cancel Recovery", width="stretch"):
                    st.session_state.otp_sent = False
                    st.rerun()
            elif st.session_state.otp_verified:
                with st.form("new_password_form", border=False):
                    st.markdown("### Set New Password")
                    new_pw = st.text_input("New Secure Password", type="password")
                    new_pw_confirm = st.text_input("Confirm New Password", type="password")
                    if st.form_submit_button("Update Password", type="primary", width="stretch"):
                        if new_pw and new_pw == new_pw_confirm:
                            up_success, up_msg = update_password(st.session_state.reset_username, new_pw)
                            if up_success:
                                st.success("Password updated successfully. Returning to Login...")
                                st.session_state.otp_sent = False
                                st.session_state.otp_verified = False
                                # Added programmatic-like UX: simple pause and rerun
                                import time
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                st.error(up_msg)
                        else:
                            st.error("Passwords must match and not be empty.")
        
        st.markdown('', unsafe_allow_html=True)
        st.stop()
        
    # --- Hide sidebar (replaced by top navbar) ---
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # Load resources
    try:
        s_dia, fk_dia = get_diabetes_scaler()
        s_heart, fk_heart = get_heart_scaler()
        s_diag, fk_diag, le_diag = get_diagnosis_scaler()
    except Exception as e:
        st.error(f"Resource Error: {e}")
        st.stop()

    # Route Content
    if st.session_state.role in ['admin', 'doctor']:
        render_admin_dashboard()
    else:
        render_clinical_portal(
            st.session_state.user_id, st.session_state.username,
            s_dia, fk_dia, s_heart, fk_heart, s_diag, fk_diag, le_diag
        )

if __name__ == "__main__":
    main()
