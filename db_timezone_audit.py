import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))
from database.db_utils import get_db_connection
import pandas as pd

def full_db_audit():
    conn = get_db_connection()
    if not conn:
        print("Failed to connect.")
        return
    
    try:
        cur = conn.cursor()
        
        # 1. Identify all timestamp/date columns
        audit_query = """
        SELECT table_name, column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND (data_type LIKE 'timestamp%' OR column_name LIKE '%date%' OR column_name LIKE '%at%')
        ORDER BY table_name;
        """
        cur.execute(audit_query)
        columns = cur.fetchall()
        
        results = []
        for table, col, dtype in columns:
            # Get a sample and check its timezone
            cur.execute(f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY {col} DESC LIMIT 1")
            sample = cur.fetchone()
            val = sample[0] if sample else "NO DATA"
            
            results.append({
                "Table": table,
                "Column": col,
                "Type": dtype,
                "Latest Value (Raw)": val
            })
            
        print("\n=== DATABASE TIMEZONE AUDIT (SESSION: Asia/Kolkata) ===")
        print(pd.DataFrame(results).to_string(index=False))
        
        # 2. Check current time in DB
        cur.execute("SELECT NOW() as db_now, CURRENT_TIMESTAMP as db_ct, (NOW() AT TIME ZONE 'UTC') as utc_now")
        now_data = cur.fetchone()
        print("\n=== DB CLOCK CHECK ===")
        print(f"DB NOW (IST): {now_data[0]}")
        print(f"DB Current Timestamp: {now_data[1]}")
        print(f"UTC Now (from DB): {now_data[2]}")
        
        # 3. Check Session Config
        cur.execute("SHOW timezone")
        print(f"\nFinal Session Timezone Check: {cur.fetchone()[0]}")

    except Exception as e:
        print(f"Audit Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    full_db_audit()
