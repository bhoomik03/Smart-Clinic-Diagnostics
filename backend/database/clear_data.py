import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.database.db_utils import reset_entire_database, initialize_tables

def perform_reset():
    print("🚀 Starting Clinical Database Reset...")
    
    confirm = input("⚠️ WARNING: This will DELETE ALL USERS AND DIAGNOSIS DATA. Are you sure? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Reset cancelled.")
        return

    success, msg = reset_entire_database()
    if success:
        print(f"✅ {msg}")
        print("🛠️ Re-initializing tables with new timezone settings...")
        initialize_tables()
        print("✨ Database is now fresh and ready for new Admin registration.")
    else:
        print(f"❌ Error: {msg}")

if __name__ == "__main__":
    perform_reset()
