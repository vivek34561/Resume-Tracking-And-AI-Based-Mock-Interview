# database_postgres.py - PostgreSQL version

import psycopg2
from psycopg2 import pool, extras
import os
from passlib.hash import pbkdf2_sha256
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

# --- PostgreSQL Configuration ---
connection_pool = None

def parse_database_url():
    """Parse PostgreSQL DATABASE_URL from Heroku."""
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        # Heroku Postgres URLs start with postgres://, but psycopg2 needs postgresql://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)
        return database_url
    
    # Fallback to individual env vars (for local development)
    return f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', '')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'resume_tracker')}"

def init_connection_pool():
    """Initialize PostgreSQL connection pool."""
    global connection_pool
    if connection_pool is None:
        try:
            database_url = parse_database_url()
            connection_pool = psycopg2.pool.SimpleConnectionPool(
                1,  # minconn
                10,  # maxconn
                database_url
            )
            print("✅ PostgreSQL connection pool created successfully")
        except Exception as err:
            print(f"❌ Error creating connection pool: {err}")
            raise

def get_db_connection():
    """Get a connection from the pool."""
    global connection_pool
    if connection_pool is None:
        init_connection_pool()
    try:
        conn = connection_pool.getconn()
        return conn
    except Exception as err:
        print(f"Error getting connection from pool: {err}")
        raise

def return_connection(conn):
    """Return a connection to the pool."""
    global connection_pool
    if connection_pool and conn:
        connection_pool.putconn(conn)

def init_mysql_db():
    """Initialize PostgreSQL database and create tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Users table - Updated to support both traditional and Google OAuth
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE,
                password_hash VARCHAR(255),
                email VARCHAR(255) UNIQUE,
                google_id VARCHAR(255) UNIQUE,
                full_name VARCHAR(255),
                profile_picture TEXT,
                auth_type VARCHAR(20) DEFAULT 'traditional',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL
            )
        ''')
        
        # Create indexes for users table
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_username ON users(username)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_email ON users(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_google_id ON users(google_id)')
        
        # User settings table (JSON string)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                settings TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User-specific resumes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_resumes (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename VARCHAR(500),
                resume_hash VARCHAR(64) NOT NULL,
                resume_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, resume_hash)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_created ON user_resumes(user_id, created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_hash ON user_resumes(user_id, resume_hash)')
        
        # Cache for full analysis results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_analysis (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                resume_hash VARCHAR(64) NOT NULL,
                jd_hash VARCHAR(64) NOT NULL,
                provider VARCHAR(50),
                model VARCHAR(100),
                intensity VARCHAR(50),
                result_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (user_id, resume_hash, jd_hash, provider, model, intensity)
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_time ON user_analysis(user_id, created_at)')
        
        # Legacy resumes table (optional - for backward compatibility)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(500) NOT NULL UNIQUE,
                resume_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        print("✅ PostgreSQL database initialized successfully!")
        
    except Exception as err:
        print(f"❌ Error creating tables: {err}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        return_connection(conn)


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
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id", (username, pwd_hash))
        user_id = cursor.fetchone()[0]
        conn.commit()
        return user_id
    except psycopg2.IntegrityError:
        return None
    finally:
        cursor.close()
        return_connection(conn)

def authenticate_user(username: str, password: str):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
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
        return_connection(conn)

def get_user_by_username(username: str):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        cursor.execute("SELECT id, username FROM users WHERE username = %s", (username,))
        row = cursor.fetchone()
        return {"id": row['id'], "username": row['username']} if row else None
    finally:
        cursor.close()
        return_connection(conn)


# --- Google OAuth Functions ---
def create_or_update_google_user(email: str, google_id: str, name: str = None, picture: str = None):
    """
    Create a new user from Google OAuth or update existing user.
    Returns user dict with id, email, name, etc.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        # Check if user with this google_id already exists
        cursor.execute("SELECT * FROM users WHERE google_id = %s", (google_id,))
        user = cursor.fetchone()
        
        if user:
            # Update last login and profile info
            cursor.execute("""
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP,
                    full_name = %s,
                    profile_picture = %s
                WHERE google_id = %s
                RETURNING *
            """, (name, picture, google_id))
            user = cursor.fetchone()
            conn.commit()
        else:
            # Check if email already exists (maybe from traditional auth)
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            existing = cursor.fetchone()
            
            if existing:
                # Link Google account to existing user
                cursor.execute("""
                    UPDATE users 
                    SET google_id = %s,
                        auth_type = 'google',
                        full_name = %s,
                        profile_picture = %s,
                        last_login = CURRENT_TIMESTAMP
                    WHERE email = %s
                    RETURNING *
                """, (google_id, name, picture, email))
                user = cursor.fetchone()
                conn.commit()
            else:
                # Create new user
                cursor.execute("""
                    INSERT INTO users (email, google_id, full_name, profile_picture, auth_type, last_login)
                    VALUES (%s, %s, %s, %s, 'google', CURRENT_TIMESTAMP)
                    RETURNING *
                """, (email, google_id, name, picture))
                user = cursor.fetchone()
                conn.commit()
        
        return {
            "id": user['id'],
            "username": user.get('username') or user['email'].split('@')[0],
            "email": user['email'],
            "name": user.get('full_name'),
            "picture": user.get('profile_picture'),
            "google_id": user['google_id'],
            "auth_type": user['auth_type']
        }
    except Exception as e:
        conn.rollback()
        print(f"Error creating/updating Google user: {e}")
        return None
    finally:
        cursor.close()
        return_connection(conn)


def get_user_by_google_id(google_id: str):
    """Get user by Google ID."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        cursor.execute("SELECT * FROM users WHERE google_id = %s", (google_id,))
        user = cursor.fetchone()
        if user:
            return {
                "id": user['id'],
                "username": user.get('username') or user['email'].split('@')[0],
                "email": user['email'],
                "name": user.get('full_name'),
                "picture": user.get('profile_picture'),
                "google_id": user['google_id'],
                "auth_type": user['auth_type']
            }
        return None
    finally:
        cursor.close()
        return_connection(conn)


def get_user_by_email(email: str):
    """Get user by email."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if user:
            return {
                "id": user['id'],
                "username": user.get('username') or user['email'].split('@')[0],
                "email": user['email'],
                "name": user.get('full_name'),
                "picture": user.get('profile_picture'),
                "auth_type": user.get('auth_type')
            }
        return None
    finally:
        cursor.close()
        return_connection(conn)


def get_user_settings(user_id: int) -> dict:
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
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
        return_connection(conn)

def save_user_settings(user_id: int, settings: dict):
    import json
    s = json.dumps(settings or {})
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Upsert behavior using INSERT ... ON CONFLICT
        cursor.execute(
            """
            INSERT INTO user_settings (user_id, settings, updated_at) 
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) 
            DO UPDATE SET settings = %s, updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, s, s)
        )
        conn.commit()
        return True
    finally:
        cursor.close()
        return_connection(conn)

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
        # Try to insert; if conflict, update and return id
        cursor.execute(
            """
            INSERT INTO user_resumes (user_id, filename, resume_hash, resume_text) 
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, resume_hash) 
            DO UPDATE SET filename = %s
            RETURNING id
            """,
            (user_id, filename, resume_hash, resume_text, filename)
        )
        row_id = cursor.fetchone()[0]
        conn.commit()
        return row_id
    finally:
        cursor.close()
        return_connection(conn)

def get_user_resumes(user_id: int):
    """List saved resumes for a user with metadata for sidebar selection."""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        cursor.execute(
            "SELECT id, filename, resume_hash, created_at FROM user_resumes WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        cursor.close()
        return_connection(conn)

def get_user_resume_by_id(user_id: int, user_resume_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
    try:
        cursor.execute(
            "SELECT id, filename, resume_hash, resume_text, created_at FROM user_resumes WHERE user_id = %s AND id = %s",
            (user_id, user_resume_id)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        cursor.close()
        return_connection(conn)

# --- Analysis caching ---
def get_cached_analysis(user_id: int, resume_hash: str, jd_hash: str, provider: str, model: str, intensity: str):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
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
        return_connection(conn)

def save_cached_analysis(user_id: int, resume_hash: str, jd_hash: str, provider: str, model: str, intensity: str, result: dict):
    if not user_id or not resume_hash or not jd_hash or not result:
        return False
    import json
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        result_json = json.dumps(result)
        # Upsert using INSERT ... ON CONFLICT
        cursor.execute(
            """
            INSERT INTO user_analysis 
            (user_id, resume_hash, jd_hash, provider, model, intensity, result_json, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, resume_hash, jd_hash, provider, model, intensity)
            DO UPDATE SET result_json = %s, created_at = CURRENT_TIMESTAMP
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
        return_connection(conn)

# --- Pinecone Functions (kept for compatibility) ---
# These can remain empty or be implemented if needed
