"""
Database Migration Script
Adds patient demographic columns to predictions table
"""

import sqlite3
import os

def migrate_database():
    db_path = 'instance/mediguard.db'
    
    if not os.path.exists(db_path):
        print("❌ Database not found. Please run the app first to create it.")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [col[1] for col in cursor.fetchall()]
        
        migrations_needed = []
        if 'patient_name' not in columns:
            migrations_needed.append('patient_name')
        if 'patient_age' not in columns:
            migrations_needed.append('patient_age')
        if 'patient_sex' not in columns:
            migrations_needed.append('patient_sex')
        
        if not migrations_needed:
            print("✅ Database is already up to date!")
            return
        
        print(f"📝 Adding columns: {', '.join(migrations_needed)}")
        
        # Add new columns
        if 'patient_name' in migrations_needed:
            cursor.execute('ALTER TABLE predictions ADD COLUMN patient_name VARCHAR(200)')
            print("  ✅ Added patient_name column")
        
        if 'patient_age' in migrations_needed:
            cursor.execute('ALTER TABLE predictions ADD COLUMN patient_age INTEGER')
            print("  ✅ Added patient_age column")
        
        if 'patient_sex' in migrations_needed:
            cursor.execute('ALTER TABLE predictions ADD COLUMN patient_sex VARCHAR(10)')
            print("  ✅ Added patient_sex column")
        
        conn.commit()
        print("\n✅ Database migration completed successfully!")
        print("You can now restart the Flask app.")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    migrate_database()
