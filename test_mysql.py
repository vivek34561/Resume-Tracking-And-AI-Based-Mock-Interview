"""
Test script to verify MySQL database connection and setup.
Run this before starting the main application.
"""

import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

# Configuration
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "resume_tracker")

def test_mysql_connection():
    """Test MySQL connection and database setup."""
    print("=" * 60)
    print("MySQL Database Connection Test")
    print("=" * 60)
    
    # Test 1: Connect to MySQL server
    print("\n[1] Testing MySQL server connection...")
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD
        )
        print(f"✓ Successfully connected to MySQL server at {MYSQL_HOST}:{MYSQL_PORT}")
        
        # Test 2: Create database if not exists
        print(f"\n[2] Creating database '{MYSQL_DATABASE}' if not exists...")
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
        print(f"✓ Database '{MYSQL_DATABASE}' is ready")
        
        cursor.close()
        conn.close()
        
        # Test 3: Connect to the specific database
        print(f"\n[3] Testing connection to database '{MYSQL_DATABASE}'...")
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        print(f"✓ Successfully connected to database '{MYSQL_DATABASE}'")
        
        # Test 4: Check MySQL version
        cursor = conn.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()[0]
        print(f"\n[4] MySQL Version: {version}")
        
        # Test 5: List existing tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        if tables:
            print(f"\n[5] Existing tables in '{MYSQL_DATABASE}':")
            for table in tables:
                print(f"    - {table[0]}")
        else:
            print(f"\n[5] No tables found in '{MYSQL_DATABASE}' (will be created on first run)")
        
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed! MySQL is ready to use.")
        print("=" * 60)
        print("\nYou can now run: streamlit run app.py")
        return True
        
    except mysql.connector.Error as err:
        print(f"\n✗ Error: {err}")
        print("\n" + "=" * 60)
        print("Troubleshooting Steps:")
        print("=" * 60)
        print("1. Check if MySQL service is running")
        print("2. Verify credentials in .env file:")
        print(f"   MYSQL_HOST={MYSQL_HOST}")
        print(f"   MYSQL_PORT={MYSQL_PORT}")
        print(f"   MYSQL_USER={MYSQL_USER}")
        print(f"   MYSQL_PASSWORD={'*' * len(MYSQL_PASSWORD)}")
        print(f"   MYSQL_DATABASE={MYSQL_DATABASE}")
        print("\n3. Check firewall settings")
        print("4. Verify MySQL user permissions")
        return False

if __name__ == "__main__":
    test_mysql_connection()
