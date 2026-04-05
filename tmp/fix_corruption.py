import os

app_path = r'd:\DB PATEL\medical_ai_project\frontend\app.py'
print(f"Opening {app_path}...")
with open(app_path, 'rb') as f:
    orig_data = f.read()

# Remove null bytes and BOM
# PowerShell -Encoding utf8 appends a BOM sometimes (EF BB BF)
# PowerShell -Encoding unicode appends null bytes (UTF-16 LE)
clean_data = orig_data.replace(b'\x00', b'').replace(b'\xff\xfe', b'').replace(b'\xef\xbb\xbf', b'')

# Strip trailing whitespace and the corrupted line starts
import re
# The corrupted line looks like "#   T r i g g e r i n g..." in binary (if it was UTF-16)
# If it's already stripped of nulls, it looks like "# Triggering fresh deployment with numerical timestamp fix"
clean_text = clean_data.decode('utf-8', errors='ignore')
clean_text = re.sub(r'#\s*T\s*r\s*i\s*g\s*g\s*e\s*r\s*i\s*n\s*g.*$', '', clean_text, flags=re.MULTILINE)
clean_text = clean_text.strip() + '\n'

with open(app_path, 'wb') as f:
    f.write(clean_text.encode('utf-8'))

print("File app.py cleaned successfully.")
