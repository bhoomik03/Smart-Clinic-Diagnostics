"""
ocr_engine.py

Module 7 - OCR Integration
Parses uploaded medical report images or PDFs, extracts text via Tesseract OCR, 
and uses regular expressions to find structured medical data for autofilling forms.
"""

import os
import re
import pytesseract
from PIL import Image
import pdf2image
import io
import cv2
import numpy as np

def preprocess_image_cv2(image):
    """
    Preprocess image using OpenCV to improve OCR accuracy.
    Converts to grayscale, applies blur and Otsu's thresholding.
    """
    if isinstance(image, Image.Image):
        # Convert PIL image to OpenCV format
        img_np = np.array(image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        # Assuming numpy array (already loaded via cv2)
        img_bgr = image
        
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's thresholding to binarize image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

# Explicitly set the path to tesseract.exe for Windows if needed, 
# though if it's in PATH, pytesseract should find it.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_file(file_bytes: bytes, file_name: str) -> str:
    """
    Given a file's bytes and filename, returns the extracted text string.
    Handles standard images (.png, .jpg, .jpeg) and PDFs.
    """
    text = ""
    file_name_lower = file_name.lower()
    
    try:
        if file_name_lower.endswith('.pdf'):
            # Convert PDF pages to images
            # NOTE: Poppler must be installed and in System PATH
            images = pdf2image.convert_from_bytes(file_bytes)
            for page_img in images:
                processed_img = preprocess_image_cv2(page_img)
                text += pytesseract.image_to_string(processed_img) + "\n"
        
        elif file_name_lower.endswith(('.png', '.jpg', '.jpeg')):
            # Decode file directly to OpenCV format
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            processed_img = preprocess_image_cv2(img)
            text = pytesseract.image_to_string(processed_img)
        
        else:
            raise ValueError("Unsupported file format. Please upload PDF, PNG, JPG, or JPEG.")
            
    except Exception as e:
        print(f"OCR Processing Error: {e}")
        raise e
        
    return text

def parse_medical_data(text: str) -> dict:
    """
    Parses unstructured text to find specific medical values.
    Returns a dictionary of found values, mapped to their keys.
    """
    extracted_data = {}
    
    # Pre-process text to make regex matching easier across lines and chaotic spacing
    text_clean = re.sub(r'\s+', ' ', text)
    
    # We will search line by line for better isolation of metrics instead of across the whole text block
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # 1. Blood Pressure: e.g., "120/80"
        bp_match = re.search(r'(?i)(?:bp|blood\s*pressure).*?(\d{2,3})\s*/\s*(\d{2,3})', line)
        if bp_match:
            extracted_data['systolic'] = int(bp_match.group(1))
            extracted_data['diastolic'] = int(bp_match.group(2))
        else:
            bp_raw_match = re.search(r'\b(\d{2,3})\s*/\s*(\d{2,3})\b', line)
            if bp_raw_match and 'systolic' not in extracted_data:
                sys_val = int(bp_raw_match.group(1))
                dia_val = int(bp_raw_match.group(2))
                if 80 <= sys_val <= 250 and 40 <= dia_val <= 150:
                     extracted_data['systolic'] = sys_val
                     extracted_data['diastolic'] = dia_val
                     
        # 2. Fasting Glucose / Random Blood Sugar
        # Ignores stray single digits (like "RANDOM BLOOD SUGAR 2 115.1") and finds the main metric
        if re.search(r'(?i)(?:glucose|fbs|fasting\s*sugar|random\s*blood\s*sugar|r\.b\.s\.)', line):
            # Exclude lines that are just reference ranges or purely textual criteria
            if not re.search(r'(?i)normal|impaired|reference|diabetes\s*mellitus|criteria|plasma\s*glucose\s*>=|normal\s*glycemia', line):
                # Look for numbers, potentially with OCR errors like missing decimals (e.g. 1151 instead of 115.1)
                numbers = re.findall(r'(\d{2,4}(?:\.\d+)?)', line)
                if numbers and 'glucose' not in extracted_data:
                    val_str = numbers[0]
                    # Specific fix: if a 4 digit number is found with no decimal (like 1151), assume it's meant to have 1 decimal place.
                    if len(val_str) == 4 and '.' not in val_str:
                        val = float(val_str) / 10.0
                    else:
                        val = float(val_str)
                    
                    if val > 600: val -= 600 # Simple heuristic for 1 misread as 7
                    extracted_data['glucose'] = val
                    
        # 3. Total Cholesterol
        if re.search(r'(?i)(?:cholesterol|tot chol|total\s*chol)', line):
            val_match = re.search(r'(\d{2,3}(?:\.\d+)?)', line)
            if val_match:
                extracted_data['cholesterol'] = float(val_match.group(1))

        # 4. BMI
        if re.search(r'(?i)bmi', line):
            val_match = re.search(r'(\d{2}(?:\.\d+)?)', line)
            if val_match:
                extracted_data['bmi'] = float(val_match.group(1))
                
        # 5. Age
        # Handles patterns like "Age/Sex : 20Yrs./M" or "Age: 20"
        if re.search(r'(?i)age(?:/sex)?', line):
            val_match = re.search(r'(\d{1,3})(?=\s*(?:yrs|years|y|/))', line, re.IGNORECASE)
            if not val_match:
                 val_match = re.search(r'(\d{1,3})', line)
                 
            if val_match and 'age' not in extracted_data:
                extracted_data['age'] = int(val_match.group(1))
                
        # 6. Insulin
        if re.search(r'(?i)insulin', line):
            val_match = re.search(r'(\d{1,3}(?:\.\d+)?)', line)
            if val_match:
                extracted_data['insulin'] = float(val_match.group(1))

        # 7. Serum Creatinine (Kidney)
        if re.search(r'(?i)creatinine', line):
            val_match = re.search(r'(\d{1,2}(?:\.\d+)?)', line)
            if val_match:
                extracted_data['creatinine'] = float(val_match.group(1))

        # --- NEW EXTENSIONS FOR FULL PATHOLOGY REPORTS ---
        
        # 8. Haemogram (Hb, WBC, Platelets, RBC)
        if re.search(r'(?i)haemoglobin|hemoglobin', line):
            val_match = re.search(r'(\d{1,2}(?:\.\d+)?)', line)
            if val_match and 'hb' not in extracted_data:
                extracted_data['hb'] = float(val_match.group(1))
                
        if re.search(r'(?i)wbc\s*count|white\s*blood', line):
            # Matches formats like "5,200" or "5200"
            val_match = re.search(r'(\d{1,3}(?:,\d{3})+|\d{4,6})', line)
            if val_match and 'wbc' not in extracted_data:
                extracted_data['wbc'] = int(val_match.group(1).replace(',', ''))
                
        if re.search(r'(?i)platelet', line):
            val_match = re.search(r'(\d{1,3}(?:,\d{3})+|\d{4,6})', line)
            if val_match and 'platelets' not in extracted_data:
                extracted_data['platelets'] = int(val_match.group(1).replace(',', ''))

        # 9. Liver Function Test (SGOT / SGPT)
        if re.search(r'(?i)s\.g\.o\.t|ast', line):
             val_match = re.search(r'(\d{1,3}(?:\.\d+)?)', line)
             if val_match and 'sgot' not in extracted_data:
                 extracted_data['sgot'] = float(val_match.group(1))
                 
        if re.search(r'(?i)sgpt|alt', line) and 'salt' not in line.lower():
             val_match = re.search(r'(\d{1,3}(?:\.\d+)?)', line)
             if val_match and 'sgpt' not in extracted_data:
                 extracted_data['sgpt'] = float(val_match.group(1))

        # 10. Inflammation (CRP)
        if re.search(r'(?i)c-reactive', line):
             val_match = re.search(r'(\d{1,3}(?:\.\d+)?)', line)
             if val_match and 'crp' not in extracted_data:
                 extracted_data['crp'] = float(val_match.group(1))

        # 11. Dengue Duo Test
        if re.search(r'(?i)dengue\s*(?:igg|igm)', line) or re.search(r'(?i)ns1\s*antigen', line):
            if re.search(r'(?i)weak\s*reactive', line):
                res = "WEAK REACTIVE"
            elif re.search(r'(?i)non-reactive|non\s*reactive', line):
                res = "NON-REACTIVE"
            elif re.search(r'(?i)reactive|positive', line):
                res = "REACTIVE"
            else:
                res = None
                
            if res:
                if 'igg' in line.lower(): extracted_data['dengue_igg'] = res
                if 'igm' in line.lower(): extracted_data['dengue_igm'] = res
                if 'ns1' in line.lower(): extracted_data['dengue_ns1'] = res

        # 12. Widal Test (Typhoid)
        if re.search(r'(?i)typhi\s*[\'"]?o[\'"]?', line):
             if re.search(r'(?i)no\s*agglutination', line):
                 extracted_data['typhoid_o'] = 'NEGATIVE'
             elif re.search(r'(?i)positive|reactive|1:[1-9]', line):
                 extracted_data['typhoid_o'] = 'POSITIVE'
                 
        if re.search(r'(?i)typhi\s*[\'"]?h[\'"]?', line):
             if re.search(r'(?i)no\s*agglutination', line):
                 extracted_data['typhoid_h'] = 'NEGATIVE'
             elif re.search(r'(?i)positive|reactive|1:[1-9]', line):
                 extracted_data['typhoid_h'] = 'POSITIVE'

    return extracted_data

def process_document_to_dict(file_bytes: bytes, file_name: str) -> dict:
    """Wrapper function that takes a file, runs OCR, and parsing."""
    text = extract_text_from_file(file_bytes, file_name)
    return parse_medical_data(text)
