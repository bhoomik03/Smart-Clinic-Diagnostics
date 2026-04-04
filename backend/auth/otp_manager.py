import random
import time
import smtplib
import requests
import re
from email.message import EmailMessage
import os
from database.db_utils import get_system_setting

def generate_otp(length=6):
    """Generates a secure random N-digit OTP."""
    otp = ''.join([str(random.randint(0, 9)) for _ in range(length)])
    return otp

def send_otp(otp, email=None, contact=None):
    """
    Sends the OTP to the provided Email securely using SMTP and via SMS using MSG91.
    Returns (True/False, "Message string for UI")
    """
    print("\n" + "="*50)
    print("[OTP TRANSMISSION]")
    print(f"Target Email: {email}")
    print(f"Target Contact: {contact}")
    print(f"Code: {otp}")
    print("="*50)
    
    from database.db_utils import get_db_connection, log_audit_action
    
    conn = get_db_connection()
    cursor = conn.cursor() if conn else None
    
    email_success = False
    sms_success = False

    # 1. Send via MSG91 SMS if contact is provided
    if contact:
        try:
            url = "https://control.msg91.com/api/v5/otp"
            msg91_auth = get_system_setting("msg91_auth_key", "497872Ane46b3969a7300dP1")
            msg91_sender = get_system_setting("msg91_sender_id", "MSGIND")
            headers = {
                "authkey": msg91_auth,
                "Content-Type": "application/json"
            }
            # Clean mobile number: remove all non-numeric characters
            mobile_clean = re.sub(r'\D', '', str(contact).strip())
            # MSG91 requires country code. Default to 91 if it's 10 digits
            if len(mobile_clean) == 10:
                mobile_clean = "91" + mobile_clean

            msg91_template = get_system_setting("msg91_template_id", "69a73425620bd639e90057c0")
            payload = {
                "template_id": msg91_template,
                "mobile": mobile_clean,
                "otp": str(otp),
                "sender": msg91_sender
            }
            
            print(f"[DEBUG] Sending SMS to {mobile_clean} using Sender: {msg91_sender}")
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                resp_json = response.json()
                if resp_json.get('type') == 'success':
                    print(f"[SUCCESS] SMS sent securely to {mobile_clean}. Msg: {resp_json.get('message')}")
                    sms_success = True
                    if cursor:
                        log_audit_action(cursor, "OTP_SMS_SUCCESS", f"Sent SMS to {mobile_clean}")
                else:
                    print(f"[FAILED] SMS rejected by provider: {response.text}")
                    if cursor:
                        log_audit_action(cursor, "OTP_SMS_REJECTED", f"SMS rejected for {mobile_clean}: {response.text}")
            else:
                print(f"[FAILED] SMS Failed (HTTP {response.status_code}): {response.text}")
                if cursor:
                    log_audit_action(cursor, "OTP_SMS_FAILED", f"SMS HTTP error for {mobile_clean}")
        except Exception as e:
            print(f"[FAILED] SMS Exception: {str(e)}")
            if cursor:
                log_audit_action(cursor, "OTP_SMS_ERROR", f"Exception for {contact}: {str(e)}")

    # 2. Send via Email if email is provided
    if email:
        sender_email = get_system_setting("smtp_email", "bhoomikpatel98@gmail.com")
        app_password = get_system_setting("smtp_password", "YOUR_APP_PASSWORD_HERE")
        
        if app_password != "YOUR_APP_PASSWORD_HERE":
            try:
                msg = EmailMessage()
                msg.set_content(f"Your verification code is: {otp}\n\nThis code is valid for 5 minutes.\nDo not share this code with anyone.")
                msg['Subject'] = 'AI-Based Medical Diagnosis Support System - Verification Code'
                msg['From'] = f"AI-Based Medical Diagnosis Support System <{sender_email}>"
                msg['To'] = email

                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(sender_email, app_password)
                    smtp.send_message(msg)
                    
                print(f"[SUCCESS] Email sent securely to {email}")
                email_success = True
                if cursor:
                    log_audit_action(cursor, "OTP_EMAIL_SUCCESS", f"Sent precisely to {email}")
            except Exception as e:
                print(f"[FAILED] Email Failed: {e}")
                if cursor:
                    log_audit_action(cursor, "OTP_EMAIL_ERROR", f"Exception for {email}: {str(e)}")
        else:
            print("[X] SMTP Error: App Password not set.")
            if cursor:
                log_audit_action(cursor, "OTP_EMAIL_FAILED", f"SMTP not configured. Failed for {email}")

    if conn:
        conn.commit()
        conn.close()
        
    print("="*50 + "\n")

    if email_success or sms_success:
        return True, "OTP Sent Successfully via Email/SMS"
    else:
        return False, "Failed to send OTP. Please check your credentials or network."
