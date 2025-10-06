# database.py

import sqlite3
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

# --- Environment Variables ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Constants ---
DB_FILE = "resumes.db"
PINECONE_INDEX_NAME = "resume-analyzer"

# --- SQLite Functions ---
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_sqlite_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            resume_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_resume_to_db(filename, resume_text):
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO resumes (filename, resume_text) VALUES (?, ?)",
            (filename, resume_text)
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        # This means a resume with this filename already exists
        return None
    finally:
        conn.close()

def get_all_resumes_from_db():
    conn = get_db_connection()
    resumes = conn.execute("SELECT id, filename FROM resumes ORDER BY filename").fetchall()
    conn.close()
    return resumes

def get_resume_text_by_id(resume_id):
    conn = get_db_connection()
    resume = conn.execute("SELECT resume_text FROM resumes WHERE id = ?", (resume_id,)).fetchone()
    conn.close()
    return resume['resume_text'] if resume else None

# --- Pinecone Functions ---
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

def upsert_vectors_to_pinecone(resume_id, resume_text):
    index = get_pinecone_index()
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    # For now, we'll embed and store the whole text as one vector.
    # This can be expanded to store chunks for RAG.
    vector = embeddings.embed_query(resume_text)
    
    # We use the resume_id from SQLite as the vector's ID in Pinecone
    index.upsert(vectors=[(str(resume_id), vector)])
    return True