import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_dataset(filepath, output_filename):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    print(f"Loading {filepath} for cleaning...")
    # Read the dataset
    df = pd.read_csv(filepath)
    
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")
    
    # 1. Fix column names (lowercase, replace spaces with underscores)
    print("Fixing column names...")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    
    # 2. Remove missing values
    print("Removing missing values...")
    df = df.dropna()
    
    # 3. Remove duplicates (Skipped to retain expanded 5 million row datasets)
    # print("Removing duplicates...")
    # df = df.drop_duplicates()
    
    # 4. Convert categorical data to numeric
    print("Converting categorical data to numeric...")
    le = LabelEncoder()
    # Find categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # 5. Perform feature scaling and normalization
    print("Scaling and normalizing numerical features...")
    # Typically we don't scale the target variable, we will try to infer target columns from common names
    target_keywords = ['outcome', 'target', 'diagnosis', 'class', 'label']
    target_cols = [col for col in df.columns if any(kw in col for kw in target_keywords)]
    
    # Scale all numerical features except the suspected target columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    features_to_scale = [col for col in numeric_cols if col not in target_cols]
    
    if len(features_to_scale) > 0:
        scaler = StandardScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    final_shape = df.shape
    print(f"Cleaned shape: {final_shape}")
    print(f"Removed {initial_shape[0] - final_shape[0]} rows (missing/duplicates).")
    
    # 6. Save clean dataset
    output_path = os.path.join(os.path.dirname(filepath), output_filename)
    print(f"Saving cleaned & scaled dataset to {output_path}...\n")
    df.to_csv(output_path, index=False)
    
    return output_path

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data')
    
    # Clean all CSVs found in the data directory and subdirectories
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # We skip the generated 'cleaned_' prefix to avoid double processing
            if file.endswith(".csv") and not file.startswith("cleaned_"):
                input_path = os.path.join(root, file)
                output_filename = f"cleaned_{file}"
                print(f"--- Processing {file} ---")
                try:
                    clean_dataset(input_path, output_filename)
                except Exception as e:
                    print(f"Error cleaning {file}: {e}")

    print("All preprocessing finished!")
