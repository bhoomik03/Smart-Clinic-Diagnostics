"""
rule_engine.py

Module 3 - Rule Engine Development
Implements clinical rule-based decision logic following WHO and AHA guidelines.
Generates explainable, reason-based outputs with risk categories:
- Normal
- Mild
- High
- Critical
"""

import numpy as np
import pandas as pd
import json

def evaluate_blood_pressure(systolic: int, diastolic: int) -> dict:
    reason = f"BP recorded as {systolic}/{diastolic} mmHg. "
    # WHO / AHA Guidelines
    if systolic > 180 or diastolic > 120:
         return {"category": "Critical", "detected": True, "reason": reason + "[HYPERTENSIVE CRISIS] Seek immediate emergency medical attention"}
    elif systolic < 90 or diastolic < 60:
         return {"category": "High", "detected": True, "reason": reason + "[HYPOTENSION] Critically low blood pressure detected"}
    elif systolic >= 140 or diastolic >= 90:
         return {"category": "High", "detected": True, "reason": reason + "[HYPERTENSION STAGE 2] High blood pressure detected"}
    elif systolic >= 130 or diastolic >= 80:
         return {"category": "Mild", "detected": True, "reason": reason + "[HYPERTENSION STAGE 1] Elevated blood pressure"}
    elif (120 <= systolic < 130) and diastolic < 80:
         return {"category": "Mild", "detected": True, "reason": reason + "[ELEVATED_BP] Elevated Blood Pressure. Monitor closely"}
    else:  
         return {"category": "Normal", "detected": False, "reason": reason + "[NORMAL] Blood pressure within healthy range"}

def evaluate_blood_sugar_fasting(glucose: float) -> dict:
    reason = f"Fasting Glucose recorded as {glucose} mg/dL. "
    # WHO Guidelines
    if glucose < 54:
        return {"category": "Critical", "detected": True, "reason": reason + "[SEVERE_HYPOGLYCEMIA] Critically low blood sugar"}
    elif glucose < 70:
        return {"category": "High", "detected": True, "reason": reason + "[HYPOGLYCEMIA] Low blood sugar detected"}
    elif glucose >= 126:
        return {"category": "High", "detected": True, "reason": reason + "[DIABETES] High blood sugar detected"}
    elif 100 <= glucose <= 125:
        return {"category": "Mild", "detected": True, "reason": reason + "[PREDIABETES] Elevated fasting glucose detected"}
    else:
        return {"category": "Normal", "detected": False, "reason": reason + "[NORMAL] Blood glucose normal"}

def evaluate_cholesterol(total_cholesterol: int) -> dict:
    reason = f"Total Cholesterol recorded as {total_cholesterol} mg/dL. "
    if total_cholesterol >= 240:
        return {"category": "High", "detected": True, "reason": reason + "[HIGH_CHOLESTEROL] Extreme cardiovascular risk detected"}
    elif total_cholesterol >= 200:
        return {"category": "Mild", "detected": True, "reason": reason + "[BORDERLINE_HIGH] Elevated cholesterol risk"}
    elif total_cholesterol < 120:
        return {"category": "Mild", "detected": True, "reason": reason + "[HYPOCHOLESTEROLEMIA] Cholesterol below recommended clinical level"}
    else:
        return {"category": "Normal", "detected": False, "reason": reason + "[NORMAL] Cholesterol within healthy range"}

def evaluate_hypertension(systolic: int, diastolic: int) -> dict:
    bp_eval = evaluate_blood_pressure(systolic, diastolic)
    if bp_eval["category"] in ["Mild", "High", "Critical"]:
        bp_eval["detected"] = True
    else:
        bp_eval["detected"] = False
    return bp_eval

def evaluate_obesity(bmi: float) -> dict:
    reason = f"BMI recorded as {bmi}. "
    # WHO Guidelines
    if bmi >= 40.0:
        return {"category": "Critical", "detected": True, "reason": reason + "[OBESITY_CLASS_III] High risk extreme obesity"}
    elif bmi >= 30.0:
        return {"category": "High", "detected": True, "reason": reason + "[OBESITY] Increased health risk"}
    elif bmi >= 25.0:
        return {"category": "Mild", "detected": True, "reason": reason + "[OVERWEIGHT] Pre-obesity range"}
    elif bmi < 18.5:
        return {"category": "High", "detected": True, "reason": reason + "[UNDERWEIGHT] Increased clinical risk from low body weight"}
    else:
        return {"category": "Normal", "detected": False, "reason": reason + "[NORMAL] Healthy BMI range"}

def evaluate_heart_rate(hr: int) -> dict:
    reason = f"Heart Rate recorded as {hr} bpm. "
    if hr > 120:
        return {"category": "Critical", "detected": True, "reason": reason + "[SEVERE_TACHYCARDIA] Dangerously high resting heart rate"}
    elif hr > 100:
        return {"category": "High", "detected": True, "reason": reason + "[TACHYCARDIA] Elevated resting heart rate detected"}
    elif hr < 50:
        return {"category": "Critical", "detected": True, "reason": reason + "[SEVERE_BRADYCARDIA] Dangerously low heart rate"}
    elif hr < 60:
        return {"category": "High", "detected": True, "reason": reason + "[BRADYCARDIA] Low resting heart rate detected"}
    else:
        return {"category": "Normal", "detected": False, "reason": reason + "[NORMAL] Heart rate normal"}

def evaluate_body_temp(temp: float) -> dict:
    reason = f"Body Temperature recorded as {temp} °C. "
    if temp >= 39.4:
        return {"category": "Critical", "detected": True, "reason": reason + "[HYPERPYREXIA] Dangerously high fever"}
    elif temp >= 38.0:
        return {"category": "High", "detected": True, "reason": reason + "[FEVER] Elevated body temperature detected"}
    elif temp < 35.0:
        return {"category": "Critical", "detected": True, "reason": reason + "[HYPOTHERMIA] Body temperature critically low"}
    elif temp < 36.1:
        return {"category": "High", "detected": True, "reason": reason + "[MILD_HYPOTHERMIA] Body temperature below normal"}
    else:
        return {"category": "Normal", "detected": False, "reason": reason + "[NORMAL] Body temperature normal"}

def evaluate_oxygen(spo2: int) -> dict:
    reason = f"Oxygen Saturation recorded as {spo2}%. "
    if spo2 < 90:
         return {"category": "Critical", "detected": True, "reason": reason + "[SEVERE_HYPOXEMIA] Blood oxygen dangerously low"}
    elif spo2 < 95:
         return {"category": "High", "detected": True, "reason": reason + "[HYPOXEMIA] Low oxygen saturation detected"}
    else:
         return {"category": "Normal", "detected": False, "reason": reason + "[NORMAL] Oxygen level normal"}

def evaluate_kidney_disease(creatinine: float) -> dict:
    reason = f"Serum Creatinine recorded as {creatinine} mg/dL. "
    if creatinine > 1.3:
        return {"category": "High", "detected": True, "reason": reason + "[RENAL_IMPAIRMENT] Kidney dysfunction risk"}
    elif creatinine < 0.6:
        return {"category": "Mild", "detected": True, "reason": reason + "[LOW_CREATININE] Low muscle mass indication"}
    elif 0.6 <= creatinine <= 1.3:
        return {"category": "Normal", "detected": False, "reason": reason + "[NORMAL] Kidney function normal"}
    return {"category": "Normal", "detected": False, "reason": reason + "[NORMAL] Kidney function normal"}

def evaluate_haemogram(hb: float = None, wbc: int = None, platelets: int = None) -> dict:
    reasons = []
    category = "Normal"
    detected = False
    
    if hb is not None:
        if hb > 17:
            category, detected = "High", True
            reasons.append("[POLYCYTHEMIA] Elevated hemoglobin")
        elif hb < 12:
            category, detected = ("High" if category != "Critical" else "Critical"), True
            reasons.append("[ANEMIA] Low hemoglobin detected")
        else:
            reasons.append("[NORMAL] Hemoglobin normal")
            
    if wbc is not None:
        if wbc > 11000:
            category, detected = "High", True
            reasons.append("[INFECTION_RISK] Possible infection")
        elif wbc < 4000:
            category, detected = "High", True
            reasons.append("[LEUKOPENIA] Low immunity level")
        else:
            reasons.append("[NORMAL] WBC count normal")
            
    if platelets is not None:
         if platelets > 450000:
             category, detected = "High", True
             reasons.append("[THROMBOCYTOSIS] Elevated platelet count")
         elif platelets < 150000:
             category, detected = "High", True
             reasons.append("[THROMBOCYTOPENIA] Low platelet count")
         else:
             reasons.append("[NORMAL] Platelet count normal")
             
    reasons_str = " | ".join(reasons) if reasons else "[NORMAL] All haemogram parameters normal"
        
    return {"category": category, "detected": detected, "reason": reasons_str}

def evaluate_dengue(igg: str = None, igm: str = None, ns1: str = None) -> dict:
    reasons = []
    detected = False
    category = "Normal"
    
    is_igg_pos = str(igg).strip().lower() == 'positive'
    is_igm_pos = str(igm).strip().lower() == 'positive'
    is_ns1_pos = str(ns1).strip().lower() == 'positive'
    
    if is_igg_pos:
        category, detected = "High", True
        reasons.append("[DENGUE_DETECTED] Dengue infection detected")
    elif igg is not None and str(igg).strip().lower() == 'negative':
        reasons.append("[NO_INFECTION] Dengue antibodies not detected")
        
    if is_igm_pos:
        category, detected = "High", True
        reasons.append("[DENGUE_DETECTED] Dengue infection detected")
    elif igm is not None and str(igm).strip().lower() == 'negative':
        reasons.append("[NO_INFECTION] Dengue infection not detected")
        
    if is_ns1_pos:
        category, detected = "High", True
        reasons.append("[DENGUE_DETECTED] Dengue infection detected")
    elif ns1 is not None and str(ns1).strip().lower() == 'negative':
        reasons.append("[NO_INFECTION] Dengue antigen not detected")
        
    reas_str = " | ".join(reasons) if reasons else "[NO_INFECTION] No Dengue markers detected."
    return {"category": category, "detected": detected, "reason": reas_str}

def evaluate_typhoid(o_ag: str = None, h_ag: str = None) -> dict:
    detected = False
    category = "Normal"
    reasons = []
    
    is_o_pos = str(o_ag).strip().lower() == 'positive'
    is_h_pos = str(h_ag).strip().lower() == 'positive'
    
    if is_o_pos:
        detected = True
        category = "High"
        reasons.append("[TYPHOID_DETECTED] Typhoid infection detected")
    elif o_ag is not None and str(o_ag).strip().lower() == 'negative':
        reasons.append("[NO_INFECTION] Typhoid infection not detected")
        
    if is_h_pos:
        detected = True
        category = "High"
        reasons.append("[TYPHOID_DETECTED] Typhoid infection detected")
    elif h_ag is not None and str(h_ag).strip().lower() == 'negative':
        reasons.append("[NO_INFECTION] Typhoid infection not detected")
    
    reason_str = " | ".join(reasons) if reasons else "[NO_INFECTION] Widal Test Negative."
         
    return {"category": category, "detected": detected, "reason": reason_str}

def evaluate_liver_function(sgot: float = None, sgpt: float = None) -> dict:
    reasons = []
    detected = False
    category = "Normal"
    
    if sgot is not None:
        if sgot > 40:
            detected, category = True, "High"
            reasons.append("[LIVER_DAMAGE] Liver injury risk")
        elif sgot < 10:
            detected = True
            if category == "Normal": category = "Mild"
            reasons.append("[LOW_AST] Below normal liver enzyme")
        else:
            reasons.append("[NORMAL] Liver function normal")
            
    if sgpt is not None:
        if sgpt > 56:
            detected, category = True, "High"
            reasons.append("[LIVER_DAMAGE] Liver damage risk")
        elif sgpt < 7:
            detected = True
            if category == "Normal": category = "Mild"
            reasons.append("[LOW_ALT] Below normal enzyme level")
        else:
            reasons.append("[NORMAL] Liver enzyme normal")
        
    reas_str = " | ".join(reasons) if reasons else "[NORMAL] Liver enzymes normal"
    return {"category": category, "detected": detected, "reason": reas_str}

def evaluate_inflammation(crp: float) -> dict:
    detected = False
    category = "Normal"
    if crp > 3.0:
        detected = True
        category = "High"
        reason = f"CRP recorded as {crp} mg/L. [INFLAMMATION] Significant inflammation"
    elif 1.0 <= crp <= 3.0:
        reason = f"CRP recorded as {crp} mg/L. [MILD_INFLAMMATION] Mild inflammation, monitor closely"
    else:
        reason = f"CRP recorded as {crp} mg/L. [NORMAL] No significant inflammation detected"
        
    return {"category": category, "detected": detected, "reason": reason}

def evaluate_advanced_diabetics(pregnancies: int = None, skin_thickness: float = None, insulin: float = None, dpf: float = None) -> dict:
    reasons = []
    detected = False
    category = "Normal"
    
    if pregnancies is not None:
        if pregnancies > 4:
            detected, category = True, "High"
            reasons.append("[HIGH_RISK] Multiple pregnancy risk factor")
        elif pregnancies == 0:
            detected = True
            if category == "Normal": category = "Mild"
            reasons.append("[NONE] No pregnancy history")
        else:
            reasons.append("[NORMAL] Normal pregnancy count")
            
    if skin_thickness is not None:
        if skin_thickness > 30:
            detected, category = True, "High"
            reasons.append("[HIGH_RISK] High fat deposition")
        elif skin_thickness < 10:
            detected = True
            if category == "Normal": category = "Mild"
            reasons.append("[LOW_SKIN_FAT] Low subcutaneous fat")
        else:
            reasons.append("[NORMAL] Skin thickness normal")
            
    if insulin is not None:
        if insulin > 166:
            detected, category = True, "High"
            reasons.append("[HIGH_INSULIN] Possible insulin resistance")
        elif insulin < 16:
            detected, category = True, "High"
            reasons.append("[LOW_INSULIN] Low insulin level")
        else:
            reasons.append("[NORMAL] Insulin level normal")
            
    if dpf is not None:
        if dpf > 0.8:
            detected, category = True, "High"
            reasons.append("[HIGH_GENETIC_RISK] Strong genetic risk")
        elif dpf < 0.2:
            detected = True
            if category == "Normal": category = "Mild"
            reasons.append("[LOW_GENETIC_RISK] Low genetic diabetes risk")
        else:
            reasons.append("[NORMAL] Moderate genetic risk")
            
    reas_str = " | ".join(reasons)
    return {"category": category, "detected": detected, "reason": reas_str}

def evaluate_advanced_heart(cp=None, exang=None, ca=None, restecg=None, oldpeak=None, thalach=None, slope=None, thal=None) -> dict:
    reasons = []
    category = "Normal"
    detected = False
    
    if cp is not None:
        if cp >= 2:
            detected, category = True, "High"
            reasons.append("[ANGINA_RISK] Possible coronary artery disease")
        else:
            reasons.append("[NORMAL] No significant cardiac pain")
            
    if exang is not None:
        if exang >= 1:
            detected, category = True, "High"
            reasons.append("[HIGH_RISK] Exercise induced angina detected")
        else:
            reasons.append("[NORMAL] Normal exercise tolerance")
            
    if ca is not None:
        if ca >= 1:
            detected, category = True, "High"
            reasons.append("[CORONARY_RISK] Coronary artery blockage risk")
        else:
            reasons.append("[NORMAL] Adequate coronary blood flow")
            
    if restecg is not None:
        if restecg >= 1:
            detected, category = True, "High"
            reasons.append("[ECG_ABNORMALITY] Possible cardiac abnormality")
        else:
            reasons.append("[NORMAL] ECG within healthy limits")
            
    if oldpeak is not None:
        if oldpeak >= 1:
            detected, category = True, "High"
            reasons.append("[ISCHEMIA_RISK] Myocardial ischemia suspected")
        else:
            reasons.append("[NORMAL] No ischemia pattern")
            
    if thalach is not None:
        if thalach > 170 or thalach < 90:
            detected, category = True, "High"
            reasons.append("[CARDIAC_RISK] Abnormal heart response")
        else:
            reasons.append("[NORMAL] Normal cardiac response")
            
    if slope is not None:
        if slope >= 1:
            detected, category = True, "High"
            reasons.append("[ISCHEMIC_PATTERN] Possible ischemia")
        else:
            reasons.append("[NORMAL] Healthy ST slope")
            
    if thal is not None:
        if thal >= 2:
            detected, category = True, "High"
            reasons.append("[THALASSEMIA_RISK] Hemoglobin disorder detected")
        else:
            reasons.append("[NORMAL] Normal hemoglobin pattern")
        
    reas_str = " | ".join(reasons)
    return {"category": category, "detected": detected, "reason": reas_str}

def patient_health_assessment(systolic: int, diastolic: int, fasting_glucose: float, total_cholesterol: int) -> dict:
    bp = evaluate_blood_pressure(systolic, diastolic)
    sugar = evaluate_blood_sugar_fasting(fasting_glucose)
    chol = evaluate_cholesterol(total_cholesterol)
    
    # Specific Disease Rules
    diabetes_risk_yes = sugar["category"] in ["High", "Critical"]
    
    # Heart disease risk is elevated if BP is High/Critical OR Cholesterol is High
    heart_disease_risk_yes = bp["category"] in ["High", "Critical"] or chol["category"] in ["High"]
    
    # Multi-disease is true if they have BOTH high risk of diabetes and heart disease
    multi_disease_risk_yes = diabetes_risk_yes and heart_disease_risk_yes
    
    categories = [bp["category"], sugar["category"], chol["category"]]
    if "Critical" in categories: overall_risk = "Critical"
    elif "High" in categories: overall_risk = "High"
    elif "Mild" in categories: overall_risk = "Mild"
    else: overall_risk = "Normal"
        
    return {
        "overall_risk": overall_risk,
        "diabetes_risk": "YES" if diabetes_risk_yes else "NO",
        "heart_disease_risk": "YES" if heart_disease_risk_yes else "NO",
        "multi_disease_risk": "YES" if multi_disease_risk_yes else "NO",
        "evaluations": {"blood_pressure": bp, "fasting_blood_sugar": sugar, "cholesterol": chol}
    }

# =====================================================================
# HIGH-PERFORMANCE VECTORIZED METHODS FOR BULK DATASET PROCESSING
# =====================================================================

def evaluate_risk_batch(df: pd.DataFrame, sys_col: str, dia_col: str, glu_col: str, chol_col: str) -> pd.DataFrame:
    df_out = df.copy()

    # Vectorized BP
    bp_conditions = [
        (df_out[sys_col] > 180) | (df_out[dia_col] > 120),
        (df_out[sys_col] >= 140) | (df_out[dia_col] >= 90),
        ((df_out[sys_col] >= 130) & (df_out[sys_col] <= 139)) | ((df_out[dia_col] >= 80) & (df_out[dia_col] <= 89)),
        ((df_out[sys_col] >= 120) & (df_out[sys_col] <= 129)) & (df_out[dia_col] < 80)
    ]
    df_out['BP_Risk'] = np.select(bp_conditions, ['Critical', 'High', 'High', 'Mild'], default='Normal')

    # Vectorized Glucose
    glu_conditions = [(df_out[glu_col] >= 250), (df_out[glu_col] >= 126), (df_out[glu_col] >= 100) & (df_out[glu_col] <= 125)]
    df_out['Glucose_Risk'] = np.select(glu_conditions, ['Critical', 'High', 'Mild'], default='Normal')

    # Vectorized Cholesterol
    chol_conditions = [(df_out[chol_col] >= 240), (df_out[chol_col] >= 200) & (df_out[chol_col] <= 239)]
    df_out['Cholesterol_Risk'] = np.select(chol_conditions, ['High', 'Mild'], default='Normal')

    # Yes/No Explicit Risks
    df_out['Diabetes_Risk'] = np.where(df_out['Glucose_Risk'].isin(['High', 'Critical']), 'YES', 'NO')
    df_out['Heart_Disease_Risk'] = np.where(
        (df_out['BP_Risk'].isin(['High', 'Critical'])) | (df_out['Cholesterol_Risk'] == 'High'), 
        'YES', 'NO'
    )
    df_out['Multi_Disease_Risk'] = np.where(
        (df_out['Diabetes_Risk'] == 'YES') & (df_out['Heart_Disease_Risk'] == 'YES'), 
        'YES', 'NO'
    )

    # Max Severity
    severity_map = {'Normal': 0, 'Mild': 1, 'High': 2, 'Critical': 3}
    reverse_map = {0: 'Normal', 1: 'Mild', 2: 'High', 3: 'Critical'}
    max_severity = np.maximum.reduce([
        df_out['BP_Risk'].map(severity_map),
        df_out['Glucose_Risk'].map(severity_map),
        df_out['Cholesterol_Risk'].map(severity_map)
    ])
    df_out['Overall_Risk'] = pd.Series(max_severity).map(reverse_map)
    
    return df_out

if __name__ == "__main__":
    print("--- Testing SINGLE Patient Assessment ---")
    assessment = patient_health_assessment(150, 95, 145, 250)
    print(json.dumps(assessment, indent=2))
    
    print("\n--- Testing HIGH-PERFORMANCE Batch Assessment ---")
    np.random.seed(42)
    test_df = pd.DataFrame({
        'sysBP': np.random.randint(90, 200, 1000000),
        'diaBP': np.random.randint(60, 130, 1000000),
        'glucose': np.random.randint(70, 300, 1000000),
        'totChol': np.random.randint(120, 300, 1000000)
    })
    
    import time
    start = time.time()
    result_df = evaluate_risk_batch(test_df, 'sysBP', 'diaBP', 'glucose', 'totChol')
    elapsed = time.time() - start
    
    print(f"Evaluated {len(test_df)} patients in {elapsed:.4f} seconds!")
    print("\nSample Output (showing YES/NO flags):")
    print(result_df[['Diabetes_Risk', 'Heart_Disease_Risk', 'Multi_Disease_Risk', 'Overall_Risk']].head(10))
