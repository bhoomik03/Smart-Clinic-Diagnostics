"""
prediction.py

Module 4 - Machine Learning Model Development (Real-Time Prediction)
Loads saved machine learning models via joblib and provides functions 
to get predictions based on patient input data.
"""

import joblib
import os
import pandas as pd

# Define paths to load the trained models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def _load_model(disease_name):
    """Utility to load a specific model."""
    model_filename = f"{disease_name.lower().replace(' ', '_')}_model.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    if not os.path.exists(model_path):
        return None
        
    return joblib.load(model_path)

def predict_disease(disease_name: str, patient_data: dict, raw_data: dict = None) -> dict:
    """
    Predicts the likelihood of a disease based on patient input data.
    
    Args:
        disease_name: "Diabetes", "Heart Disease", or "Disease Diagnosis"
        patient_data: A dictionary containing features for prediction.
                      Example: {'feature_1': 1.0, 'feature_2': -0.5, ...}
        raw_data: Optional dictionary containing the original unscaled values.
                      
    Returns:
        A dictionary containing the prediction result and the model used.
    """
    model = _load_model(disease_name)
    
    if not model:
        return {"status": "error", "message": f"Model for '{disease_name}' not found."}
    
    try:
        # Convert input dictionary into a pandas DataFrame (handles single row gracefully)
        df_input = pd.DataFrame([patient_data])
        
        # Predict
        prediction = model.predict(df_input)
        
        # In a real-world scenario, and if the model supports predict_proba,
        # we would also return confidence scores.
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_input)
            # The confidence of the predicted class
            confidence = proba[0][prediction[0]]
            
        # --- EXPLAINABLE AI (XAI) MODULE (SHAP INTEGRATION) ---
        top_factors = []
        feature_map = {
            '0': 'Pregnancies', '1': 'Glucose', '2': 'Blood Pressure', '3': 'Skin Thickness',
            '4': 'Insulin', '5': 'BMI', '6': 'Diabetes Pedigree Function', '7': 'Age',
            'age': 'Age', 'sex': 'Gender', 'cp': 'Chest Pain Type', 'trestbps': 'Resting BP',
            'chol': 'Cholesterol (mg/dl)', 'fbs': 'Fasting Blood Sugar > 120', 'restecg': 'Resting ECG',
            'thalach': 'Max Heart Rate', 'exang': 'Exercise Angina', 'oldpeak': 'ST Depression',
            'slope': 'ST Slope', 'ca': 'Major Vessels Colored', 'thal': 'Thalassemia'
        }
        
        try:
            import shap
            # Initialize SHAP TreeExplainer for the current patient prediction
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)
            
            feature_names = df_input.columns.tolist()
            
            # shape of shap_values for Random Forest Classifier is usually a list [class_0_array, class_1_array]
            # where each array is of shape (n_samples, n_features). We want the array for the predicted class.
            pred_class = int(prediction[0])
            
            if isinstance(shap_values, list):
                # Binary or multiclass Random Forest
                patient_shap_vals = shap_values[pred_class][0]
            else:
                patient_shap_vals = shap_values[0]
                
            # We care about the *absolute* magnitude of the SHAP values (which drove the decision the most)
            feature_weights = list(zip(feature_names, patient_shap_vals))
            feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Normalize to percentages relative to total absolute SHAP variations
            total_shap = sum([abs(w) for f, w in feature_weights]) or 1.0
            
            # Take top 3 factors driving the current decision
            for fname, weight in feature_weights[:3]:
                friendly_name = feature_map.get(str(fname), str(fname).replace("_", " ").title())
                
                # Assign a readable percentage impact contribution (0-100%)
                relative_impact = (abs(weight) / total_shap) * 100
                
                # We can store the detailed real SHAP weight, but importance_score shows impact proportion
                top_factors.append({
                    "feature": friendly_name,
                    "importance_score": round(relative_impact, 1),
                    "patient_value": raw_data[fname] if raw_data is not None and fname in raw_data else df_input.iloc[0][fname],
                    "shap_contribution": round(weight, 4) # Negative means it reduced risk, positive means it increased risk
                })
                
        except ImportError:
            # Fallback to Global Feature Importances if SHAP is missing
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_names = df_input.columns.tolist()
                feature_weights = list(zip(feature_names, importances))
                feature_weights.sort(key=lambda x: x[1], reverse=True)
                for fname, weight in feature_weights[:3]:
                    friendly_name = feature_map.get(str(fname), str(fname).replace("_", " ").title())
                    top_factors.append({
                        "feature": friendly_name,
                        "importance_score": round(weight * 100, 1),
                        "patient_value": raw_data[fname] if raw_data is not None and fname in raw_data else df_input.iloc[0][fname]
                    })


        return {
            "status": "success",
            "disease": disease_name,
            "prediction": int(prediction[0]),
            "confidence": round(confidence, 4) if confidence else None,
            "top_factors": top_factors
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Simple test for the module using dummy standardized data
    dummy_diabetes_data = {
        '0': 6, '1': 148, '2': 72, '3': 35, '4': 0, '5': 33.6, '6': 0.627, '7': 50
    }
    
    print("Testing Diabetes Prediction:")
    res = predict_disease("Diabetes", dummy_diabetes_data)
    print(res)
