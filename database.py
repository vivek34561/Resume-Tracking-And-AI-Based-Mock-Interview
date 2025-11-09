# database.py

import mysql.connector
from mysql.connector import pooling
import os
from passlib.hash import pbkdf2_sha256
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

# --- MySQL Configuration ---
# Support both individual env vars and DATABASE_URL (Heroku/Railway style)
def parse_database_config():
    """Parse MySQL configuration from env vars or DATABASE_URL."""
    # Check for ClearDB or JawsDB URL (Heroku add-ons)
    database_url = os.getenv("CLEARDB_DATABASE_URL") or os.getenv("JAWSDB_URL") or os.getenv("DATABASE_URL")
    
    if database_url:
        # Parse URL format: mysql://user:password@host:port/database
        parsed = urlparse(database_url)
        return {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 3306,
            "user": parsed.username or "root",
            "password": parsed.password or "",
            "database": parsed.path.lstrip("/") if parsed.path else "resume_tracker"
        }
    else:
        # Use individual environment variables
        return {
            "host": os.getenv("MYSQL_HOST", "localhost"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD", ""),
            "database": os.getenv("MYSQL_DATABASE", "resume_tracker")
        }

# Get MySQL configuration
db_config = parse_database_config()
MYSQL_HOST = db_config["host"]
MYSQL_PORT = db_config["port"]
MYSQL_USER = db_config["user"]
MYSQL_PASSWORD = db_config["password"]
MYSQL_DATABASE = db_config["database"]

# Connection pool for better performance
connection_pool = None

def init_connection_pool():
    """Initialize MySQL connection pool."""
    global connection_pool
    if connection_pool is None:
        try:
            connection_pool = pooling.MySQLConnectionPool(
                pool_name="resume_pool",
                pool_size=5,
                pool_reset_session=True,
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
                autocommit=False
            )
        except mysql.connector.Error as err:
            print(f"Error creating connection pool: {err}")
            raise

def get_db_connection():
    """Get a connection from the pool."""
    global connection_pool
    if connection_pool is None:
        init_connection_pool()
    try:
        return connection_pool.get_connection()
    except mysql.connector.Error as err:
        print(f"Error getting connection from pool: {err}")
        raise

def init_mysql_db():
    """Initialize MySQL database and create tables."""
    # First, create database if it doesn't exist
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Error creating database: {err}")
        raise
    
    # Now create tables
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_username (username)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # User settings table (JSON string)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INT PRIMARY KEY,
                settings TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # User-specific resumes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_resumes (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                filename VARCHAR(500),
                resume_hash VARCHAR(64) NOT NULL,
                resume_text LONGTEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_user_hash (user_id, resume_hash),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_user_created (user_id, created_at),
                INDEX idx_user_hash (user_id, resume_hash)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # Cache for full analysis results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_analysis (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                resume_hash VARCHAR(64) NOT NULL,
                jd_hash VARCHAR(64) NOT NULL,
                provider VARCHAR(50),
                model VARCHAR(100),
                intensity VARCHAR(50),
                result_json LONGTEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_analysis (user_id, resume_hash, jd_hash, provider, model, intensity),
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_user_time (user_id, created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # Legacy resumes table (optional - for backward compatibility)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(500) NOT NULL UNIQUE,
                resume_text LONGTEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        conn.commit()
        print("MySQL database initialized successfully!")
        
    except mysql.connector.Error as err:
        print(f"Error creating tables: {err}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


# --- User Auth Functions ---
def create_user(username: str, password: str):
    username = (username or '').strip()
    password = (password or '').strip()
    if not username or not password:
        return None
    pwd_hash = pbkdf2_sha256.hash(password)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, pwd_hash))
        conn.commit()
        return cursor.lastrowid
    except mysql.connector.IntegrityError:
        return None
    finally:
        cursor.close()
        conn.close()

def authenticate_user(username: str, password: str):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, username, password_hash FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        if not row:
            return None
        if pbkdf2_sha256.verify(password, row['password_hash']):
            return {"id": row['id'], "username": row['username']}
        return None
    finally:
        cursor.close()
        conn.close()

def get_user_by_username(username: str):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, username FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        return {"id": row['id'], "username": row['username']} if row else None
    finally:
        cursor.close()
        conn.close()

def get_user_settings(user_id: int) -> dict:
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT settings FROM user_settings WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if not row:
            return {}
        try:
            import json
            return json.loads(row['settings'] or '{}')
        except Exception:
            return {}
    finally:
        cursor.close()
        conn.close()

def save_user_settings(user_id: int, settings: dict):
    import json
    s = json.dumps(settings or {})
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Upsert behavior using INSERT ... ON DUPLICATE KEY UPDATE
        cursor.execute(
            """
            INSERT INTO user_settings (user_id, settings, updated_at) 
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON DUPLICATE KEY UPDATE settings = %s, updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, s, s)
        )
        conn.commit()
        return True
    finally:
        cursor.close()
        conn.close()

# --- User resume storage (per-user, hashed) ---
def save_user_resume(user_id: int, filename: str, resume_hash: str, resume_text: str):
    """Upsert a user's resume content keyed by content hash to avoid duplicates.

    Returns the row id (existing or new).
    """
    if not user_id or not resume_hash or not resume_text:
        return None
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Try to insert; if exists due to unique constraint, get existing id
        cursor.execute(
            "INSERT IGNORE INTO user_resumes (user_id, filename, resume_hash, resume_text) VALUES (%s, %s, %s, %s)",
            (user_id, filename, resume_hash, resume_text)
        )
        conn.commit()
        
        if cursor.rowcount == 0:
            # Already exists, retrieve id
            cursor.execute(
                "SELECT id FROM user_resumes WHERE user_id = %s AND resume_hash = %s",
                (user_id, resume_hash)
            )
            row = cursor.fetchone()
            return row[0] if row else None
        return cursor.lastrowid
    finally:
        cursor.close()
        conn.close()

def get_user_resumes(user_id: int):
    """List saved resumes for a user with metadata for sidebar selection."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT id, filename, resume_hash, created_at FROM user_resumes WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        return rows
    finally:
        cursor.close()
        conn.close()

def get_user_resume_by_id(user_id: int, user_resume_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT id, filename, resume_hash, resume_text, created_at FROM user_resumes WHERE user_id = %s AND id = %s",
            (user_id, user_resume_id)
        )
        row = cursor.fetchone()
        return row if row else None
    finally:
        cursor.close()
        conn.close()

# --- Analysis caching ---
def get_cached_analysis(user_id: int, resume_hash: str, jd_hash: str, provider: str, model: str, intensity: str):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT result_json FROM user_analysis
            WHERE user_id = %s AND resume_hash = %s AND jd_hash = %s AND provider = %s AND model = %s AND intensity = %s
            ORDER BY created_at DESC LIMIT 1
            """,
            (user_id, resume_hash, jd_hash, provider or '', model or '', intensity or 'full')
        )
        row = cursor.fetchone()
        if not row:
            return None
        import json
        try:
            return json.loads(row['result_json'])
        except Exception:
            return None
    finally:
        cursor.close()
        conn.close()

def save_cached_analysis(user_id: int, resume_hash: str, jd_hash: str, provider: str, model: str, intensity: str, result: dict):
    if not user_id or not resume_hash or not jd_hash or not result:
        return False
    import json
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        result_json = json.dumps(result)
        # Upsert using INSERT ... ON DUPLICATE KEY UPDATE
        cursor.execute(
            """
            INSERT INTO user_analysis 
            (user_id, resume_hash, jd_hash, provider, model, intensity, result_json, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON DUPLICATE KEY UPDATE result_json = %s, created_at = CURRENT_TIMESTAMP
            """,
            (
                user_id, resume_hash, jd_hash, provider or '', model or '', intensity or 'full', result_json,
                result_json
            )
        )
        conn.commit()
        return True
    finally:
        cursor.close()
        conn.close()

# --- Pinecone Functions (kept for compatibility) ---
# These can remain empty or be implemented if needed
