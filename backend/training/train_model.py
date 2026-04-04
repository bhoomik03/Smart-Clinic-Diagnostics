"""
train_model.py

Module 4 - Machine Learning Model Development
Trains a single optimized Random Forest model for each disease dataset,
evaluates it on accuracy, and saves it using joblib for real-time predictions.

Model: Random Forest Classifier (single model for entire project)
"""

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate(dataset_name: str, filepath: str, target_col: str, max_rows: int = 100000):
    print(f"\n{'='*60}")
    print(f"  Training Random Forest for: {dataset_name}")
    print(f"{'='*60}")

    if not os.path.exists(filepath):
        print(f"Dataset not found at: {filepath}")
        return None

    # Load dataset (sample for speed — expanded datasets have millions of duplicated rows)
    df = pd.read_csv(filepath, nrows=max_rows)
    print(f"Loaded {dataset_name} dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # Ensure target column exists
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Split into features (X) and target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode target to integers if needed
    le = LabelEncoder()
    y = le.fit_transform(y)
    n_classes = len(set(y))
    print(f"Target classes: {n_classes}")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training: {X_train.shape[0]} samples | Testing: {X_test.shape[0]} samples\n")

    # ── Optimized Random Forest ──────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,           # More trees for better generalization
        max_depth=20,               # Prevent overfitting on noisy data
        min_samples_split=5,        # Minimum samples to split a node
        min_samples_leaf=2,         # Minimum samples at leaf
        max_features='sqrt',        # Feature subset per tree
        class_weight='balanced',    # Handle class imbalance
        random_state=42,
        n_jobs=-1                   # Use all CPU cores
    )
    
    print("Training Random Forest (200 trees, max_depth=20)...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'─'*40}")
    print(f"  ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'─'*40}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_filename = f"{dataset_name.lower().replace(' ', '_')}_model.pkl"
    model_path = os.path.join(models_dir, model_filename)
    
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    print(f"Model size: {os.path.getsize(model_path) / 1024:.1f} KB\n")
    
    return model, model_path

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data')
    
    # Cleaned dataset paths
    diabetes_data = os.path.join(data_dir, "cleaned_diabetes.csv")
    heart_data = os.path.join(data_dir, "cleaned_heart.csv")
    diagnosis_data = os.path.join(data_dir, "cleaned_disease_diagnosis.csv")
    
    print("="*60)
    print("  AI-Based Medical Diagnosis Support System")
    print("  Model: Random Forest Classifier (Unified)")
    print("="*60)
    
    # Train for Diabetes
    train_and_evaluate(
        dataset_name="Diabetes",
        filepath=diabetes_data,
        target_col="1"
    )
    
    # Train for Heart Disease
    train_and_evaluate(
        dataset_name="Heart Disease",
        filepath=heart_data,
        target_col="target"
    )
    
    # Train for General Disease Diagnosis
    train_and_evaluate(
        dataset_name="Disease Diagnosis",
        filepath=diagnosis_data,
        target_col="diagnosis"
    )
    
    print("\n" + "="*60)
    print("  All models trained and saved successfully!")
    print("  Algorithm: Random Forest (200 trees, balanced)")
    print("="*60)
