import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
import bcrypt
import pandas as pd
import datetime

# Database credentials (read from environment variables with safe defaults)
DB_USER = os.environ.get("MEDICAL_AI_DB_USER", "postgres")
DB_PASS = os.environ.get("MEDICAL_AI_DB_PASS", "1234")
DB_HOST = os.environ.get("MEDICAL_AI_DB_HOST", "localhost")
DB_PORT = os.environ.get("MEDICAL_AI_DB_PORT", "5432")
DB_NAME = os.environ.get("MEDICAL_AI_DB_NAME", "medical_ai")
DB_SSLMODE = os.environ.get("DB_SSLMODE", "prefer")

def create_database():
    """Creates the medical_ai database if it doesn't exist."""
    try:
        # Connect to the default 'postgres' database to create a new one
        conn = psycopg2.connect(
            dbname="postgres",
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'medical_ai'")
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            print(f"Database '{DB_NAME}' created successfully.")
        else:
            print(f"Database '{DB_NAME}' already exists.")
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")

def get_db_connection():
    """Returns a connection to the medical_ai database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT,
            sslmode=DB_SSLMODE
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def initialize_tables():
    """Creates all required tables for the AI-Based Medical Diagnosis Support System."""
    conn = get_db_connection()
    if not conn:
        return
        
    try:
        cursor = conn.cursor()
        
        # 1. Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(50) DEFAULT 'user',
                name VARCHAR(255),
                age INT,
                gender VARCHAR(50),
                email VARCHAR(255) UNIQUE NOT NULL,
                contact VARCHAR(100) UNIQUE NOT NULL,
                address TEXT,
                status VARCHAR(20) DEFAULT 'active',
                is_verified BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. Patients Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id SERIAL PRIMARY KEY,
                user_id INT REFERENCES users(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                age INT,
                gender VARCHAR(50),
                contact VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 3. Diagnostic Sessions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diagnostic_sessions (
                id SERIAL PRIMARY KEY,
                user_id INT REFERENCES users(id) ON DELETE CASCADE,
                patient_id INT REFERENCES patients(id) ON DELETE CASCADE,
                visit_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_type VARCHAR(50) NOT NULL,
                session_status VARCHAR(50) DEFAULT 'Completed'
            )
        """)
        
        # 4. Clinical Vitals Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinical_vitals (
                id SERIAL PRIMARY KEY,
                session_id INT REFERENCES diagnostic_sessions(id) ON DELETE CASCADE,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DECIMAL(10,2) NOT NULL,
                unit VARCHAR(50) NOT NULL,
                reference_min DECIMAL(10,2),
                reference_max DECIMAL(10,2),
                status VARCHAR(50) DEFAULT 'Normal',
                confidence DECIMAL(5,2)
            )
        """)

        # 5. ML Prediction Results Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id SERIAL PRIMARY KEY,
                session_id INT REFERENCES diagnostic_sessions(id) ON DELETE CASCADE,
                disease_type VARCHAR(255) NOT NULL,
                prediction VARCHAR(100) NOT NULL,
                probability DECIMAL(5,4),
                confidence_low DECIMAL(5,4),
                confidence_high DECIMAL(5,4),
                model_version VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 6. Clinical Observations Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinical_observations (
                id SERIAL PRIMARY KEY,
                session_id INT REFERENCES diagnostic_sessions(id) ON DELETE CASCADE,
                condition_name VARCHAR(255) NOT NULL,
                severity VARCHAR(50) NOT NULL,
                observation_text TEXT NOT NULL,
                guideline_ref VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 7. Audit Logs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id SERIAL PRIMARY KEY,
                user_id INT REFERENCES users(id) ON DELETE SET NULL,
                action_type VARCHAR(255) NOT NULL,
                details TEXT,
                ip_address VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 8. System Settings Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                key VARCHAR(255) PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 9. OTP Verification Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS otp_verification (
                id SERIAL PRIMARY KEY,
                email_or_contact VARCHAR(255) NOT NULL,
                otp_code VARCHAR(10) NOT NULL,
                expiry_time TIMESTAMP NOT NULL,
                failed_attempts INT DEFAULT 0,
                is_used BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 10. Login History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS login_history (
                id SERIAL PRIMARY KEY,
                user_id INT REFERENCES users(id) ON DELETE CASCADE,
                ip_address VARCHAR(50),
                device VARCHAR(255),
                status VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        print("All database tables created successfully.")
        
        # Log this database initialization
        log_audit_action(cursor, "DATABASE_INIT", "Database tables created/updated successfully.")
        
        # Ensure columns exist if table already existed
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'active'")
            cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT FALSE")
            cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS failed_logins INT DEFAULT 0")
            
            # Migrate OTP table if it was using user_id
            cursor.execute("ALTER TABLE otp_verification DROP CONSTRAINT IF EXISTS otp_verification_user_id_fkey")
            cursor.execute("ALTER TABLE otp_verification DROP COLUMN IF EXISTS user_id")
            cursor.execute("ALTER TABLE otp_verification ADD COLUMN IF NOT EXISTS email_or_contact VARCHAR(255)")
            cursor.execute("ALTER TABLE otp_verification ADD COLUMN IF NOT EXISTS failed_attempts INT DEFAULT 0")
        except Exception as e:
            print(f"Warning: ALTER TABLE migration in initialize_tables: {e}")
            conn.rollback()
            
        conn.commit()
        
    except Exception as e:
        print(f"Error initializing tables: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()

def log_audit_action(cursor, action, details, user_id=None, ip_address=None):
    """Auxiliary function to insert audit logs using an active cursor."""
    try:
        cursor.execute(
            "INSERT INTO audit_logs (user_id, action_type, details, ip_address) VALUES (%s, %s, %s, %s)",
            (user_id, action, details, ip_address)
        )
    except Exception as e:
        print(f"Error logging audit action: {e}")

def add_patient(name, age, gender, contact="", user_id=None):
    """Inserts a new patient and returns their ID."""
    conn = get_db_connection()
    if not conn: return None
    
    patient_id = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO patients (name, age, gender, contact, user_id) VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (name, age, gender, contact, user_id)
        )
        patient_id = cursor.fetchone()[0]
        log_audit_action(cursor, "ADD_PATIENT", f"Added patient '{name}' with ID {patient_id}")
        conn.commit()
    except Exception as e:
        print(f"Error adding patient: {e}")
        conn.rollback()
    finally:
        conn.close()
    return patient_id

def add_diagnostic_session(user_id, patient_id, source_type, session_status='Completed'):
    """Inserts a new diagnostic session and returns its ID."""
    conn = get_db_connection()
    if not conn: return None
    session_id = None
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO diagnostic_sessions (user_id, patient_id, source_type, session_status) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (user_id, patient_id, source_type, session_status)
        )
        session_id = cursor.fetchone()[0]
        log_audit_action(cursor, "ADD_SESSION", f"Added diagnostic session {session_id} for patient ID {patient_id}", user_id=user_id)
        conn.commit()
    except Exception as e:
        print(f"Error adding diagnostic session: {e}")
        conn.rollback()
    finally:
        conn.close()
    return session_id

def add_clinical_vital(session_id, metric_name, metric_value, unit, reference_min=None, reference_max=None, status='Normal', confidence=None):
    """Inserts a clinical vital sign for a diagnostic session."""
    conn = get_db_connection()
    if not conn: return False
    success = False
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO clinical_vitals (session_id, metric_name, metric_value, unit, reference_min, reference_max, status, confidence) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (session_id, metric_name, metric_value, unit, reference_min, reference_max, status, confidence)
        )
        conn.commit()
        success = True
    except Exception as e:
        print(f"Error adding clinical vital: {e}")
        conn.rollback()
    finally:
        conn.close()
    return success

def add_clinical_observation(session_id, condition_name, severity, observation_text, guideline_ref=None):
    """Inserts a clinical observation (rule engine result) for a diagnostic session."""
    conn = get_db_connection()
    if not conn: return False
    success = False
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO clinical_observations (session_id, condition_name, severity, observation_text, guideline_ref) "
            "VALUES (%s, %s, %s, %s, %s)",
            (session_id, condition_name, severity, observation_text, guideline_ref)
        )
        log_audit_action(cursor, "ADD_OBSERVATION", f"Added observation for session ID {session_id}: {condition_name}")
        conn.commit()
        success = True
    except Exception as e:
        print(f"Error adding clinical observation: {e}")
        conn.rollback()
    finally:
        conn.close()
    return success

def add_ml_prediction(session_id, disease_type, prediction, probability=None, confidence_low=None, confidence_high=None, model_version="1.0"):
    """Inserts an ML prediction result for a diagnostic session."""
    conn = get_db_connection()
    if not conn: return False
    success = False
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO ml_predictions (session_id, disease_type, prediction, probability, confidence_low, confidence_high, model_version) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (session_id, disease_type, str(prediction), float(probability) if probability is not None else None, 
             float(confidence_low) if confidence_low is not None else None, float(confidence_high) if confidence_high is not None else None, model_version)
        )
        log_audit_action(cursor, "ADD_ML_PREDICTION", f"Added ML prediction for session ID {session_id} for {disease_type}")
        conn.commit()
        success = True
    except Exception as e:
        print(f"Error adding ML prediction result: {e}")
        conn.rollback()
    finally:
        conn.close()
    return success


def get_patient_history(user_id=None):
    """Fetches diagnostic history including rule observations. Optionally filters by user_id."""
    conn = get_db_connection()
    if not conn: return pd.DataFrame()
    
    query = """
    SELECT 
        p.id as patient_id, 
        p.name, 
        p.age, 
        p.gender, 
        m.disease_type as ml_disease, 
        m.prediction as ml_result, 
        m.probability as ml_probability,
        o.condition_name as rule_disease, 
        o.severity, 
        o.observation_text,
        s.visit_date as timestamp, 
        s.id as session_id
    FROM patients p
    JOIN diagnostic_sessions s ON p.id = s.patient_id
    LEFT JOIN ml_predictions m ON s.id = m.session_id
    LEFT JOIN clinical_observations o ON s.id = o.session_id
    """
    
    params = []
    if user_id:
        query += " WHERE p.user_id = %s "
        params.append(user_id)
        
    query += " ORDER BY s.visit_date DESC "
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        df = pd.read_sql(query, conn, params=params)
        # Force column presence check
        if not df.empty and 'session_id' not in df.columns:
             print("CRITICAL DEBUG: session_id MISSING from sql results!")
    conn.close()
    return df

def get_session_vitals(session_id):
    """Fetches all clinical vitals captured for a specific session."""
    conn = get_db_connection()
    if not conn: return pd.DataFrame()
    
    query = """
    SELECT metric_name as "Parameter", metric_value as "Value", unit as "Unit", status as "Status"
    FROM clinical_vitals
    WHERE session_id = %s
    """
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        df = pd.read_sql(query, conn, params=(session_id,))
    conn.close()
    return df

def get_disease_breakdown(user_id=None, start_date=None, end_date=None):
    """Fetches counts of each disease detected. Optionally filters by user_id and date."""
    conn = get_db_connection()
    if not conn: return pd.DataFrame()
    
    query = """
    SELECT m.disease_type, m.prediction as result, COUNT(*) as count
    FROM ml_predictions m
    JOIN diagnostic_sessions s ON m.session_id = s.id
    JOIN patients p ON s.patient_id = p.id
    WHERE 1=1
    """
    
    params = []
    if user_id:
        query += " AND p.user_id = %s "
        params.append(user_id)
        
    if start_date and end_date:
        query += " AND s.visit_date BETWEEN %s AND %s "
        params.extend([start_date, end_date])
        
    query += " GROUP BY m.disease_type, m.prediction "
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        df = pd.read_sql(query, conn, params=params if params else None)
    conn.close()
    return df

def get_patient_demographics(start_date=None, end_date=None):
    """Fetches patient demographics for the Admin dashboard. Optionally filters by date."""
    conn = get_db_connection()
    if not conn: return pd.DataFrame(), pd.DataFrame()
    
    date_filter = ""
    params = []
    if start_date and end_date:
        date_filter = " WHERE created_at BETWEEN %s AND %s "
        params.extend([start_date, end_date])
    
    # Age Groups Query
    age_query = f"""
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
    {date_filter}
    GROUP BY age_group
    ORDER BY age_group
    """
    
    # Gender Query
    gender_query = f"""
    SELECT gender, COUNT(*) as count 
    FROM users 
    {date_filter}
    GROUP BY gender
    """
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        age_df = pd.read_sql(age_query, conn, params=params if params else None)
        gender_df = pd.read_sql(gender_query, conn, params=params if params else None)
        
    conn.close()
    return age_df, gender_df

def get_system_utilization(start_date=None, end_date=None):
    """Fetches user registrations and diagnostic sessions separately, grouped by date."""
    conn = get_db_connection()
    if not conn: return pd.DataFrame(), pd.DataFrame()
    
    # Using DATE() cast for PostgreSQL
    # Date Filtering Logic
    date_filter_reg = ""
    date_filter_sess = ""
    params = []
    if start_date and end_date:
        next_day = end_date + datetime.timedelta(days=1)
        date_filter_reg = " WHERE created_at >= %s AND created_at < %s "
        date_filter_sess = " WHERE visit_date >= %s AND visit_date < %s "
        params = [start_date, next_day]

    reg_query = f"""
    SELECT DATE(created_at) as scan_date, COUNT(*) as count
    FROM users
    {date_filter_reg}
    GROUP BY DATE(created_at)
    ORDER BY scan_date ASC
    """
    
    sess_query = f"""
    SELECT DATE(visit_date) as scan_date, COUNT(*) as count
    FROM diagnostic_sessions
    {date_filter_sess}
    GROUP BY DATE(visit_date)
    ORDER BY scan_date ASC
    """
    
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            if params:
                reg_df = pd.read_sql(reg_query, conn, params=tuple(params))
                sess_df = pd.read_sql(sess_query, conn, params=tuple(params))
            else:
                reg_df = pd.read_sql(reg_query, conn)
                sess_df = pd.read_sql(sess_query, conn)
            
            # Sort back to chronological order for charts
            if not reg_df.empty: 
                reg_df['scan_date'] = pd.to_datetime(reg_df['scan_date'])
                reg_df = reg_df.sort_values('scan_date')
            if not sess_df.empty: 
                sess_df['scan_date'] = pd.to_datetime(sess_df['scan_date'])
                sess_df = sess_df.sort_values('scan_date')
            
    except Exception as e:
        print(f"Error fetching utilization stats: {e}")
        reg_df, sess_df = pd.DataFrame(), pd.DataFrame()
    finally:
        conn.close()
        
    return reg_df, sess_df

# ---------------------------------------------------------
# AUTHENTICATION & ADMIN UTILS
# ---------------------------------------------------------

def register_user(username, raw_password, name, age, gender, email, contact, address):
    """Registers a new user including their patient demographics. The first user gets 'admin' role."""
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed"
    try:
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT is_verified FROM users WHERE username = %s", (username,))
        existing = cursor.fetchone()
        if existing:
            is_verified = existing[0]
            if is_verified:
                return False, "Username already taken."
            else:
                # User exists but not verified, allow re-registration by deleting first
                cursor.execute("DELETE FROM users WHERE username = %s", (username,))
                conn.commit()
                print(f"[RE-REGISTRATION] Deleting unverified user '{username}' to allow retry.")
                
        # Check if users table is empty to assign the first user as admin
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        role = 'admin' if count == 0 else 'user'
            
        # Hash password securely
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(raw_password.encode('utf-8'), salt).decode('utf-8')
        
        # Set is_verified = True for the first user (admin) so they can login immediately
        is_verified = True if role == 'admin' else False
            
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, name, age, gender, email, contact, address, is_verified) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (username, hashed, role, name, age, gender, email, contact, address, is_verified)
        )
        conn.commit()
        return True, "User registered successfully."
    except psycopg2.errors.UniqueViolation as e:
        conn.rollback()
        err_msg = str(e)
        if 'username' in err_msg: return False, "Username already taken."
        if 'email' in err_msg: return False, "Email already registered."
        if 'contact' in err_msg: return False, "Contact number already registered."
        return False, "User details already exist."
    except Exception as e:
        conn.rollback()
        return False, f"Error: {e}"
    finally:
        if conn:
            conn.close()

def delete_unverified_user(username):
    """Deletes an unverified user record if they cancel registration during OTP stage."""
    conn = get_db_connection()
    if not conn: return False
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = %s AND is_verified = False", (username,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting unverified user: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def authenticate_user(username, raw_password):
    """Verifies user credentials and returns (success, role, message, profile_dict, user_id)."""
    conn = get_db_connection()
    if not conn:
        return False, None, "Database connection failed", None, None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, password_hash, role, name, age, gender, email, contact, address, status, is_verified FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        
        if result:
            user_id, password_hash, role, name, age, gender, email, contact, address, status, is_verified = result
            if status == 'inactive':
                log_user_login(user_id, 'Unknown IP', 'Web Browser', 'Failed: Inactive')
                return False, None, "Account deactivated. Please contact admin.", None, None
            if status == 'locked':
                log_user_login(user_id, 'Unknown IP', 'Web Browser', 'Failed: Locked')
                return False, None, "Account LOCKED due to multiple failed logins. Contact admin.", None, None
            if status == 'blocked':
                log_user_login(user_id, 'Unknown IP', 'Web Browser', 'Failed: Blocked')
                return False, None, "Account BLOCKED by administrator for security clinical reasons.", None, None
            
            if not is_verified:
                log_user_login(user_id, 'Unknown IP', 'Web Browser', 'Failed: Unverified')
                return False, None, "Account not verified. Please verify your OTP.", None, None

            if bcrypt.checkpw(raw_password.encode('utf-8'), password_hash.encode('utf-8')):
                reset_failed_login(username)
                log_user_login(user_id, 'Unknown IP', 'Web Browser', 'Success')
                profile = {
                    'name': name, 'age': age, 'gender': gender,
                    'email': email, 'contact': contact, 'address': address
                }
                return True, role, "Login successful.", profile, user_id
            else:
                increment_failed_login(username)
                log_user_login(user_id, 'Unknown IP', 'Web Browser', 'Failed: Invalid Password')
                return False, None, "Invalid password.", None, None
        else:
            log_user_login(None, 'Unknown IP', 'Web Browser', f'Failed: Unknown User ({username})')
            return False, None, "Username not found.", None, None
    except Exception as e:
        return False, None, f"Error authenticating: {e}", None, None
    finally:
        if conn:
            conn.close()

def verify_user_exists(email_or_contact):
    """Checks if a user exists by email or contact number for OTP Auth."""
    conn = get_db_connection()
    if not conn: return False, None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username, email, contact FROM users WHERE email = %s OR contact = %s", (email_or_contact, email_or_contact))
        result = cursor.fetchone()
        if result:
            return True, {"username": result[0], "email": result[1], "contact": result[2]}
        return False, None
    except Exception as e:
        print(f"Error checking user: {e}")
        return False, None
    finally:
        if conn:
            conn.close()

def check_user_availability(username, email, contact):
    """Checks if username, email, or contact already exist before registration. Returns (True/False, reason_string)."""
    conn = get_db_connection()
    if not conn: return False, "Database connection failed."
    try:
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM users WHERE username = %s", (username,))
        if cursor.fetchone(): return False, "Username already exists."
        
        cursor.execute("SELECT 1 FROM users WHERE email = %s", (email,))
        if cursor.fetchone(): return False, "This email is already registered."
        
        cursor.execute("SELECT 1 FROM users WHERE contact = %s", (contact,))
        if cursor.fetchone(): return False, "This contact number is already registered."
        
        return True, "Available"
    except Exception as e:
        return False, f"Database error: {e}"
    finally:
        if conn: conn.close()

def update_password(username, new_raw_password):
    """Updates a user's password after OTP validation."""
    conn = get_db_connection()
    if not conn: return False, "Database connection failed"
    try:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(new_raw_password.encode('utf-8'), salt).decode('utf-8')
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password_hash = %s WHERE username = %s", (hashed, username))
        conn.commit()
        return True, "Password updated successfully."
    except Exception as e:
        conn.rollback()
        return False, f"Failed to update password: {e}"
    finally:
        if conn:
            conn.close()

def get_all_users():
    """Fetches all users for the Admin panel."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    try:
        query = "SELECT id, username, role, name, email, contact, status, created_at FROM users ORDER BY created_at DESC"
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error fetching users: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_registration_data():
    """Fetches comprehensive registration details for all users."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    try:
        query = "SELECT id, name, age, gender, email, contact, address, status, created_at FROM users ORDER BY created_at DESC"
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error fetching registration data: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_system_stats(start_date=None, end_date=None):
    """Fetches high-level metrics for the Admin dashboard."""
    conn = get_db_connection()
    if not conn:
        return {}
    try:
        cursor = conn.cursor()
        stats = {}
        
        # Prepare filters
        date_filter_users = ""
        date_filter_sessions = ""
        date_filter_preds = ""
        params = []
        
        if start_date and end_date:
            next_day = end_date + datetime.timedelta(days=1)
            date_filter_users = " AND created_at >= %s AND created_at < %s "
            date_filter_sessions = " WHERE visit_date >= %s AND visit_date < %s "
            date_filter_preds = " WHERE created_at >= %s AND created_at < %s "
            params = [start_date, next_day]
        # Patients
        query = f"SELECT COUNT(*) FROM users WHERE role = 'user' {date_filter_users}"
        cursor.execute(query, tuple(params))
        stats['total_patients'] = cursor.fetchone()[0]
        
        # AI Analyses
        query = f"SELECT COUNT(*) FROM ml_predictions {date_filter_preds}"
        cursor.execute(query, tuple(params))
        stats['total_predictions'] = cursor.fetchone()[0]
        
        # Total Profiles (Excluding Admins)
        raw_date_filter_users = ""
        if start_date and end_date:
            raw_date_filter_users = " WHERE created_at >= %s AND created_at < %s AND role NOT IN ('admin', 'System Administrator') "
        else:
            raw_date_filter_users = " WHERE role NOT IN ('admin', 'System Administrator') "
        
        query = f"SELECT COUNT(*) FROM users {raw_date_filter_users}"
        cursor.execute(query, tuple(params))
        stats['total_users'] = cursor.fetchone()[0]
        
        # Active Profiles (Excluding Admins)
        query = f"SELECT COUNT(*) FROM users WHERE status = 'active' AND role NOT IN ('admin', 'System Administrator') {date_filter_users}"
        cursor.execute(query, tuple(params))
        stats['active_profiles'] = cursor.fetchone()[0]

        # Clinical Sessions
        query = f"SELECT COUNT(*) FROM diagnostic_sessions {date_filter_sessions}"
        cursor.execute(query, tuple(params))
        stats['total_sessions'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM patients")
        stats['total_raw_records'] = cursor.fetchone()[0]
        
        return stats
    except Exception as e:
        print(f"Error fetching system stats: {e}")
        return {}
    finally:
        if conn:
            conn.close()

def update_user_role(admin_username, target_username, new_role):
    """Updates a user's role. Requires admin credentials log."""
    conn = get_db_connection()
    if not conn: return False, "Database connection failed"
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET role = %s WHERE username = %s", (new_role, target_username))
        log_audit_action(cursor, "ROLE_UPDATE", f"Admin '{admin_username}' changed user '{target_username}' role to '{new_role}'")
        conn.commit()
        return True, f"User '{target_username}' updated to '{new_role}' successfully."
    except Exception as e:
        conn.rollback()
        return False, f"Failed to update role: {e}"
    finally:
        if conn:
            conn.close()

def update_user_info(user_id, name, age, gender, email, contact, address):
    """Updates the core profile information for a user."""
    conn = get_db_connection()
    if not conn: return False, "Database connection failed"
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users 
            SET name = %s, age = %s, gender = %s, email = %s, contact = %s, address = %s 
            WHERE id = %s
        """, (name, age, gender, email, contact, address, user_id))
        conn.commit()
        return True, "Profile updated successfully."
    except Exception as e:
        if conn: conn.rollback()
        return False, f"Failed to update profile: {e}"
    finally:
        if conn: conn.close()

def get_audit_logs(limit=100, start_date=None, end_date=None):
    """Fetches recent system actions for the Admin panel. Optionally filters by date."""
    conn = get_db_connection()
    if not conn: return pd.DataFrame()
    try:
        query = "SELECT id, action_type, details, created_at FROM audit_logs "
        params = []
        if start_date and end_date:
            query += "WHERE created_at BETWEEN %s AND %s "
            params.extend([start_date, end_date])
            
        query += "ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            df = pd.read_sql_query(query, conn, params=tuple(params))
        return df
    except Exception as e:
        print(f"Error fetching audit logs: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def get_user_dashboard_stats(user_id):
    """Fetches aggregated stats for the user-side Home Dashboard."""
    conn = get_db_connection()
    if not conn:
        return {'total_diagnoses': 0, 'last_visit': None, 'risk_alerts': 0, 'health_status': 'No Data', 'recent_activity': pd.DataFrame()}
    try:
        cursor = conn.cursor()
        stats = {}

        # Total diagnostic sessions for this user
        cursor.execute("SELECT COUNT(*) FROM diagnostic_sessions WHERE user_id = %s", (user_id,))
        stats['total_diagnoses'] = cursor.fetchone()[0]

        # Last visit date
        cursor.execute("SELECT MAX(visit_date) FROM diagnostic_sessions WHERE user_id = %s", (user_id,))
        last_visit = cursor.fetchone()[0]
        stats['last_visit'] = last_visit.strftime('%d %b %Y') if last_visit else 'No visits yet'

        # High-risk alerts (High or Critical severity observations)
        cursor.execute("""
            SELECT COUNT(*) FROM clinical_observations o
            JOIN diagnostic_sessions s ON o.session_id = s.id
            WHERE s.user_id = %s AND o.severity IN ('High', 'Critical')
        """, (user_id,))
        stats['risk_alerts'] = cursor.fetchone()[0]

        # Overall health status based on most recent session
        if stats['risk_alerts'] > 0:
            stats['health_status'] = 'Needs Attention'
        elif stats['total_diagnoses'] > 0:
            stats['health_status'] = 'Good'
        else:
            stats['health_status'] = 'No Data'

        # Recent 5 activity records
        recent_query = """
            SELECT p.name, o.condition_name, o.severity, s.visit_date
            FROM clinical_observations o
            JOIN diagnostic_sessions s ON o.session_id = s.id
            JOIN patients p ON s.patient_id = p.id
            WHERE s.user_id = %s
            ORDER BY s.visit_date DESC
            LIMIT 5
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            stats['recent_activity'] = pd.read_sql(recent_query, conn, params=[user_id])

        return stats
    except Exception as e:
        print(f"Error fetching dashboard stats: {e}")
        return {'total_diagnoses': 0, 'last_visit': 'Error', 'risk_alerts': 0, 'health_status': 'Error', 'recent_activity': pd.DataFrame()}
    finally:
        if conn: conn.close()

def get_latest_patient_insight(user_id):
    """Fetches all diagnostic findings from the most recent session for a user."""
    print(f"DEBUG: get_latest_patient_insight called for user_id={user_id}")
    conn = get_db_connection()
    if not conn: return None
    try:
        cursor = conn.cursor()
        
        # 1. Get the latest session ID
        cursor.execute("""
            SELECT id FROM diagnostic_sessions 
            WHERE user_id = %s 
            ORDER BY visit_date DESC LIMIT 1
        """, (user_id,))
        session_res = cursor.fetchone()
        if not session_res: return None
        session_id = session_res[0]

        # 2. Get all observations for that session
        query = """
            SELECT condition_name, severity, observation_text
            FROM clinical_observations
            WHERE session_id = %s
        """
        cursor.execute(query, (session_id,))
        rows = cursor.fetchall()
        
        if not rows: return None
        
        observations = []
        for row in rows:
            observations.append({
                'Condition': row[0],
                'Severity': row[1],
                'Observation': row[2]
            })
            
        return {'observations': observations}
        
    except Exception as e:
        print(f"Error fetching latest insight: {e}")
        return None
    finally:
        if conn: conn.close()


def get_system_setting(key, default=None):
    """Fetches a system setting value by key."""
    conn = get_db_connection()
    if not conn: return default
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key = %s", (key,))
        result = cursor.fetchone()
        return result[0] if result else default
    except Exception as e:
        print(f"Error fetching setting {key}: {e}")
        return default
    finally:
        if conn:
            conn.close()

def set_system_setting(key, value):
    """Sets or updates a system setting."""
    conn = get_db_connection()
    if not conn: return False
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO system_settings (key, value, updated_at) 
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP
        """, (key, value))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error setting {key}: {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

# --- ADMIN CRUD & CONTROL FUNCTIONS ---

def delete_user(admin_username, target_username):
    """Deletes a user account. Requires admin authorization log."""
    conn = get_db_connection()
    if not conn: return False, "DB Connection Error"
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = %s AND role != 'admin'", (target_username,))
        if cursor.rowcount > 0:
            log_audit_action(cursor, "DELETE_USER", f"Admin {admin_username} deleted user {target_username}")
            conn.commit()
            return True, f"User {target_username} deleted successfully."
        return False, "User not found or cannot delete an admin."
    except Exception as e:
        return False, str(e)
    finally: conn.close()

def toggle_user_status(admin_username, target_username):
    """Activates or deactivates a user account."""
    conn = get_db_connection()
    if not conn: return False, "DB Connection Error"
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM users WHERE username = %s", (target_username,))
        res = cursor.fetchone()
        if res:
            current_status = str(res[0]).strip().lower()
            new_status = 'blocked' if current_status == 'active' else 'active'
            cursor.execute("UPDATE users SET status = %s WHERE username = %s", (new_status, target_username))
            log_audit_action(cursor, "USER_STATUS_TOGGLE", f"Admin {admin_username} changed {target_username} status to {new_status}")
            conn.commit()
            return True, f"User {target_username} is now {new_status}."
        return False, "User not found."
    except Exception as e:
        return False, str(e)
    finally: conn.close()

def admin_reset_password(admin_username, target_username, new_raw_password):
    """Resets a user's password. Admin override."""
    conn = get_db_connection()
    if not conn: return False, "DB Connection Error"
    try:
        cursor = conn.cursor()
        hashed = bcrypt.hashpw(new_raw_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute("UPDATE users SET password_hash = %s WHERE username = %s", (hashed, target_username))
        if cursor.rowcount > 0:
            log_audit_action(cursor, "ADMIN_PW_RESET", f"Admin {admin_username} reset password for {target_username}")
            conn.commit()
            return True, f"Password for {target_username} reset successfully."
        return False, "User not found."
    except Exception as e:
        return False, str(e)
    finally: conn.close()

def delete_patient_record(admin_username, patient_id):
    """Deletes a patient record and all associated diagnostic data."""
    conn = get_db_connection()
    if not conn: return False, "DB Connection Error"
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM patients WHERE id = %s", (patient_id,))
        if cursor.rowcount > 0:
            log_audit_action(cursor, "DELETE_PATIENT", f"Admin {admin_username} deleted patient ID {patient_id}")
            conn.commit()
            return True, f"Patient record (ID: {patient_id}) deleted successfully."
        return False, "Record not found."
    except Exception as e:
        return False, str(e)
    finally: conn.close()

def get_all_patients_admin():
    """Fetches all patient records for Administrative CRUD, including username."""
    conn = get_db_connection()
    if not conn: return pd.DataFrame()
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            # Join with users to get the account owner's username
            query = """
                SELECT p.id, u.username, p.name, p.age, p.gender, p.contact, p.created_at 
                FROM patients p
                LEFT JOIN users u ON p.user_id = u.id
                ORDER BY p.created_at DESC
            """
            return pd.read_sql_query(query, conn)
    finally: conn.close()
    
def delete_login_activity(admin_username, target_val):
    """
    Deletes login activity logs. 
    If target_val is numeric, deletes that specific log ID.
    If target_val is string (username), deletes all logs for that user.
    """
    conn = get_db_connection()
    if not conn: return False, "Database connection failed."
    try:
        cursor = conn.cursor()
        
        # Check if target_val is a numeric ID
        is_id = False
        try:
            val_as_id = int(target_val)
            is_id = True
        except ValueError:
            is_id = False
            
        if is_id:
            # Delete single record by log ID
            cursor.execute("DELETE FROM login_history WHERE id = %s", (val_as_id,))
            if cursor.rowcount > 0:
                log_audit_action(cursor, "DELETE_LOGIN_LOG", f"Admin {admin_username} deleted login log ID {val_as_id}")
                conn.commit()
                return True, f"Login log ID {val_as_id} deleted successfully."
            else:
                return False, "Log ID not found."
        else:
            # Delete all records by username
            cursor.execute("""
                DELETE FROM login_history 
                WHERE user_id = (SELECT id FROM users WHERE username = %s)
            """, (target_val,))
            count = cursor.rowcount
            if count > 0:
                log_audit_action(cursor, "CLEAR_USER_LOGS", f"Admin {admin_username} cleared {count} logs for user {target_val}")
                conn.commit()
                return True, f"Successfully cleared {count} login records for user: {target_val}."
            else:
                return False, f"No login records found for user: {target_val}."
                
    except Exception as e:
        return False, str(e)
    finally: conn.close()

def store_otp(email_or_contact, otp_code, expiry_minutes=5):
    """Stores a generated OTP for an email/contact in the database. email_or_contact is a string."""
    conn = get_db_connection()
    if not conn: return False
    try:
        from datetime import datetime, timedelta
        expiry = datetime.now() + timedelta(minutes=expiry_minutes)
        cursor = conn.cursor()
        
        # Mark previous OTPs for this email/contact as used
        cursor.execute("UPDATE otp_verification SET is_used = TRUE WHERE email_or_contact = %s", (str(email_or_contact),))
        
        cursor.execute(
            "INSERT INTO otp_verification (email_or_contact, otp_code, expiry_time) VALUES (%s, %s, %s)",
            (str(email_or_contact), str(otp_code), expiry)
        )
        log_audit_action(cursor, "OTP_GENERATION", f"Generated OTP for {email_or_contact}")
        conn.commit()
        return True
    except Exception as e:
        print(f"DEBUG: Error storing OTP: {e}")
        return False
    finally:
        if conn: conn.close()

def verify_otp_db(email_or_contact, entered_otp):
    """Verifies the latest OTP for an email/contact string."""
    conn = get_db_connection()
    if not conn: return False, "DB Connection Error"
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, otp_code, expiry_time, failed_attempts FROM otp_verification WHERE email_or_contact = %s AND is_used = FALSE ORDER BY created_at DESC LIMIT 1",
            (str(email_or_contact),)
        )
        res = cursor.fetchone()
        if res:
            otp_id, otp_code, expiry_time, failed_attempts = res
            
            if failed_attempts >= 5: # Increased to 5
                cursor.execute("UPDATE otp_verification SET is_used = TRUE WHERE id = %s", (otp_id,))
                conn.commit()
                return False, "Too many failed attempts. Request a new OTP."
                
            from datetime import datetime
            if datetime.now() > expiry_time:
                cursor.execute("UPDATE otp_verification SET is_used = TRUE WHERE id = %s", (otp_id,))
                conn.commit()
                return False, "OTP has expired."
                
            if str(otp_code) == str(entered_otp):
                cursor.execute("UPDATE otp_verification SET is_used = TRUE WHERE id = %s", (otp_id,))
                log_audit_action(cursor, "OTP_VERIFIED", f"OTP verified successfully for {email_or_contact}")
                conn.commit()
                return True, "Verified successfully."
                
            # Increment failed attempts
            cursor.execute("UPDATE otp_verification SET failed_attempts = failed_attempts + 1 WHERE id = %s", (otp_id,))
            conn.commit()
            return False, "Incorrect OTP."
            
        return False, "No active OTP found. Please request a new one."
    except Exception as e:
        return False, f"Server error: {e}"
    finally:
        if conn: conn.close()

def activate_user_account(user_id):
    """Marks a user account as verified."""
    conn = get_db_connection()
    if not conn: return False
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET is_verified = TRUE WHERE id = %s", (user_id,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error activating account: {e}")
        return False
    finally:
        if conn: conn.close()

def get_user_id_by_email(email_or_contact):
    """Fetches user_id by email or contact."""
    conn = get_db_connection()
    if not conn: return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = %s OR contact = %s", (email_or_contact, email_or_contact))
        res = cursor.fetchone()
        return res[0] if res else None
    finally:
        if conn: conn.close()

def log_user_login(user_id, ip_address, device, status):
    """Records a login attempt (success or failure)."""
    conn = get_db_connection()
    if not conn: return False
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO login_history (user_id, ip_address, device, status) VALUES (%s, %s, %s, %s)",
            (user_id, ip_address, device, status)
        )
        conn.commit()
    except Exception as e:
        print(f"Error logging login: {e}")
    finally:
        if conn: conn.close()

def get_login_history(user_id=None, limit=100):
    """Fetches login history, optionally filtered by user."""
    conn = get_db_connection()
    if not conn: return pd.DataFrame()
    try:
        query = "SELECT l.id, u.username, l.ip_address, l.device, l.status, l.timestamp FROM login_history l JOIN users u ON l.user_id = u.id "
        params = []
        if user_id:
            query += "WHERE l.user_id = %s "
            params.append(user_id)
        query += "ORDER BY l.timestamp DESC LIMIT %s"
        params.append(limit)
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            df = pd.read_sql_query(query, conn, params=tuple(params))
        return df
    finally:
        if conn: conn.close()

def get_filtered_audit_logs(search_term=None, action_type=None, start_date=None, end_date=None, limit=500):
    """Enhanced audit log fetcher with search and filters."""
    conn = get_db_connection()
    if not conn: return pd.DataFrame()
    try:
        query = "SELECT a.id, COALESCE(u.username, 'System') as username, a.action_type, a.details, a.ip_address, a.created_at FROM audit_logs a LEFT JOIN users u ON a.user_id = u.id WHERE 1=1 "
        params = []
        
        if search_term:
            query += "AND (a.details ILIKE %s OR u.username ILIKE %s) "
            params.extend([f"%%{search_term}%%", f"%%{search_term}%%"])
        if action_type and action_type != "All Action Types":
            query += "AND a.action_type = %s "
            params.append(action_type)
        if start_date and end_date:
            query += "AND a.created_at BETWEEN %s AND %s "
            params.extend([start_date, end_date])
            
        query += "ORDER BY a.created_at DESC LIMIT %s"
        params.append(limit)
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            df = pd.read_sql_query(query, conn, params=tuple(params))
        return df
    finally:
        if conn: conn.close()

def update_patient_record(patient_id, name, age, gender, contact):
    """Allows admin editing of a patient record."""
    conn = get_db_connection()
    if not conn: return False, "DB connection failed"
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE patients SET name = %s, age = %s, gender = %s, contact = %s WHERE id = %s", (name, age, gender, contact, patient_id))
        conn.commit()
        return True, "Patient record updated"
    except Exception as e:
        if conn: conn.rollback()
        return False, str(e)
    finally:
        if conn: conn.close()

def delete_diagnostic_session(session_id, admin_username):
    """Allows admin deleting of a diagnostic session."""
    conn = get_db_connection()
    if not conn: return False, "DB connection failed"
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM diagnostic_sessions WHERE id = %s", (session_id,))
        log_audit_action(cursor, "DELETE_SESSION", f"Admin {admin_username} deleted session ID {session_id}")
        conn.commit()
        return True, "Session deleted successfully"
    except Exception as e:
        if conn: conn.rollback()
        return False, str(e)
    finally:
        if conn: conn.close()

def increment_failed_login(username):
    """Increments failed logins and locks out if necessary."""
    conn = get_db_connection()
    if not conn: return False, "DB connection failed"
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET failed_logins = failed_logins + 1 WHERE username = %s RETURNING failed_logins", (username,))
        res = cursor.fetchone()
        if res and res[0] >= 5:
            cursor.execute("UPDATE users SET status = 'locked' WHERE username = %s AND role != 'admin'", (username,))
            conn.commit()
            return True, "Account locked due to multiple failed attempts."
        conn.commit()
        return True, "Failed login incremented."
    except Exception as e:
        if conn: conn.rollback()
        return False, str(e)
    finally:
        if conn: conn.close()

def reset_failed_login(username):
    """Resets failed login counter to 0."""
    conn = get_db_connection()
    if not conn: return False
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET failed_logins = 0 WHERE username = %s", (username,))
        conn.commit()
        return True
    except Exception as e:
        return False
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    print("Initializing Database...")
    create_database()
    initialize_tables()
    
    # Simple test data
    print("Testing data insertion...")
    p_id = add_patient("John Doe", 45, "Male", "john@example.com", user_id=None)
    if p_id:
        print(f"Test patient created with ID: {p_id}")
        session_id = add_diagnostic_session(None, p_id, "Manual Entry")
        if session_id:
            add_clinical_observation(session_id, "Diabetes", "Mild", "Fasting blood sugar is slightly elevated.")
            add_ml_prediction(session_id, "Diabetes", "1", probability=0.85, model_version="1.0")
            print("Test data inserted successfully. Check your pgAdmin panel.")
