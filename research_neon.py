import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))
from database.db_utils import get_db_connection
import datetime

def research_neon():
    conn = get_db_connection()
    if not conn:
        print("Failed to connect.")
        return
    
    try:
        cur = conn.cursor()
        
        # 1. Check Session Timezone
        cur.execute("SHOW timezone")
        tz = cur.fetchone()[0]
        print(f"Postgres Session Timezone: {tz}")
        
        # 2. Check current time in DB
        cur.execute("SELECT NOW()")
        now = cur.fetchone()[0]
        print(f"NOW() in DB: {now}")
        
        # 3. Check user record
        cur.execute("SELECT username, role, created_at, created_at AT TIME ZONE 'UTC' as utc_raw FROM users LIMIT 5")
        rows = cur.fetchall()
        print("\n--- User Records ---")
        for r in rows:
            print(f"User: {r[0]}, Role: {r[1]}, Created At: {r[2]}, (UTC Raw if converted: {r[3]})")
            
        # 4. Check sessions
        cur.execute("SELECT id, visit_date FROM diagnostic_sessions LIMIT 5")
        rows = cur.fetchall()
        print("\n--- Session Records ---")
        for r in rows:
            print(f"Session: {r[0]}, Visit Date: {r[1]}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    research_neon()
