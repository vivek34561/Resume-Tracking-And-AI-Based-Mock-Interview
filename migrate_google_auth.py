"""
Database Migration Script for Google OAuth Support
This script updates the existing database schema to support Google OAuth authentication.
Run this once after deploying the Google OAuth changes.
"""

import mysql.connector
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

def parse_database_config():
    """Parse MySQL configuration from env vars or DATABASE_URL."""
    database_url = os.getenv("CLEARDB_DATABASE_URL") or os.getenv("JAWSDB_URL") or os.getenv("DATABASE_URL")
    
    if database_url:
        parsed = urlparse(database_url)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 3306,
            "user": parsed.username or "root",
            "password": parsed.password or "",
            "database": parsed.path.lstrip("/") if parsed.path else "resume_tracker"
        }
    else:
        return {
            "host": os.getenv("MYSQL_HOST", "localhost"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD", ""),
            "database": os.getenv("MYSQL_DATABASE", "resume_tracker")
        }

def migrate_database():
    """Run database migrations for Google OAuth support."""
    db_config = parse_database_config()
    
    print("=" * 60)
    print("ResuMate Database Migration - Google OAuth Support")
    print("=" * 60)
    print(f"Connecting to database: {db_config['database']}")
    print(f"Host: {db_config['host']}")
    
    try:
        # Connect to database
        conn = mysql.connector.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"]
        )
        cursor = conn.cursor()
        
        print("\n✅ Connected to database successfully!")
        
        # Check if migrations are needed
        print("\n🔍 Checking current schema...")
        cursor.execute("DESCRIBE users")
        columns = [row[0] for row in cursor.fetchall()]
        
        migrations_needed = []
        
        if 'email' not in columns:
            migrations_needed.append("Add email column")
        if 'google_id' not in columns:
            migrations_needed.append("Add google_id column")
        if 'full_name' not in columns:
            migrations_needed.append("Add full_name column")
        if 'profile_picture' not in columns:
            migrations_needed.append("Add profile_picture column")
        if 'auth_type' not in columns:
            migrations_needed.append("Add auth_type column")
        if 'last_login' not in columns:
            migrations_needed.append("Add last_login column")
        
        if not migrations_needed:
            print("\n✅ Database schema is already up to date!")
            print("No migrations needed.")
            return
        
        print(f"\n📋 Migrations needed:")
        for migration in migrations_needed:
            print(f"   - {migration}")
        
        # Perform migrations
        print("\n🚀 Running migrations...")
        
        # Make username and password_hash nullable for Google OAuth users
        print("   - Making username and password_hash nullable...")
        cursor.execute("""
            ALTER TABLE users 
            MODIFY username VARCHAR(255) NULL,
            MODIFY password_hash VARCHAR(255) NULL
        """)
        
        # Add new columns if they don't exist
        if 'email' not in columns:
            print("   - Adding email column...")
            cursor.execute("""
                ALTER TABLE users 
                ADD COLUMN email VARCHAR(255) UNIQUE
            """)
        
        if 'google_id' not in columns:
            print("   - Adding google_id column...")
            cursor.execute("""
                ALTER TABLE users 
                ADD COLUMN google_id VARCHAR(255) UNIQUE
            """)
        
        if 'full_name' not in columns:
            print("   - Adding full_name column...")
            cursor.execute("""
                ALTER TABLE users 
                ADD COLUMN full_name VARCHAR(255)
            """)
        
        if 'profile_picture' not in columns:
            print("   - Adding profile_picture column...")
            cursor.execute("""
                ALTER TABLE users 
                ADD COLUMN profile_picture TEXT
            """)
        
        if 'auth_type' not in columns:
            print("   - Adding auth_type column...")
            cursor.execute("""
                ALTER TABLE users 
                ADD COLUMN auth_type ENUM('traditional', 'google') DEFAULT 'traditional'
            """)
        
        if 'last_login' not in columns:
            print("   - Adding last_login column...")
            cursor.execute("""
                ALTER TABLE users 
                ADD COLUMN last_login TIMESTAMP NULL
            """)
        
        # Add indexes
        print("   - Adding indexes...")
        try:
            cursor.execute("CREATE INDEX idx_email ON users(email)")
        except mysql.connector.Error:
            pass  # Index might already exist
        
        try:
            cursor.execute("CREATE INDEX idx_google_id ON users(google_id)")
        except mysql.connector.Error:
            pass  # Index might already exist
        
        # Commit changes
        conn.commit()
        
        print("\n✅ All migrations completed successfully!")
        print("\n" + "=" * 60)
        print("Database is now ready for Google OAuth authentication!")
        print("=" * 60)
        
    except mysql.connector.Error as err:
        print(f"\n❌ Error: {err}")
        print("Migration failed. Please check your database configuration.")
        conn.rollback()
        raise
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    try:
        migrate_database()
    except Exception as e:
        print(f"\n❌ Migration script failed: {e}")
        exit(1)
