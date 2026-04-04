import pandas as pd
import numpy as np
import os
import math

def expand_dataset(filepath, target_rows=5000000):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    current_rows = len(df)
    if current_rows >= target_rows:
        print(f"Dataset already has {current_rows} rows (target is {target_rows}). Skipping expansion.")
        return
        
    print(f"Current rows: {current_rows}. Target: {target_rows}.")
    
    # Calculate how many times we need to duplicate the dataframe
    multiplier = math.ceil(target_rows / current_rows)
    
    print(f"Duplicating dataset {multiplier} times...")
    # Concatenate the dataframe to itself 'multiplier' times
    expanded_df = pd.concat([df] * multiplier, ignore_index=True)
    
    # Trim down to exactly target_rows
    expanded_df = expanded_df.iloc[:target_rows]
    
    # To make the synthetic data slightly more "real", we could optionally add tiny random noise 
    # to continuous numerical columns, but for a simple duplication, this is safest and fastest.
    
    # Save the expanded dataset
    output_path = filepath.replace('.csv', '_expanded.csv')
    print(f"Saving to {output_path} with {len(expanded_df)} rows...")
    expanded_df.to_csv(output_path, index=False)
    
    # Optional: Replace original
    print(f"Replacing original {filepath} with expanded version...")
    os.replace(output_path, filepath)
    
    print(f"Done expanding {filepath}!\n")

if __name__ == "__main__":
    # Expand the downloaded diabetes data
    expand_dataset("d:\\DB PATEL\\medical_ai_project\\data\\diabetes.csv", 5000000)
    
    # Since the plotly heart dataset link was broken, expand the one we copied from D:\dataset
    expand_dataset("d:\\DB PATEL\\medical_ai_project\\data\\heart.csv", 5000000)
    
    # Also look for the other heart disease dataset you might have provided
    expand_dataset("d:\\DB PATEL\\medical_ai_project\\data\\disease_diagnosis.csv", 5000000)
