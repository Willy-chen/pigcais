import os
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid
import bcrypt

# Fetch URL from environment or use default for localhost testing
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/chat_app")

def get_db_connection():
    conn = psycopg2.connect(DB_URL)
    return conn

def init_db():
    """Initialize Postgres tables."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # 1. Users Table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash BYTEA NOT NULL
        );
    ''')
    
    # 2. Sessions Table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id UUID PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    
    # 3. Messages Table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    
    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized.")

# --- User Management ---
def create_user(username, password):
    conn = get_db_connection()
    cur = conn.cursor()
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed))
        conn.commit()
        return True
    except psycopg2.errors.UniqueViolation:
        return False
    finally:
        cur.close()
        conn.close()

def verify_user(username, password):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    
    if user and bcrypt.checkpw(password.encode('utf-8'), bytes(user[1])):
        return user[0]
    return None

# --- Session Management ---
def create_session(user_id, title="New Chat"):
    session_id = str(uuid.uuid4())
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions (id, user_id, title) VALUES (%s, %s, %s)", (session_id, user_id, title))
    conn.commit()
    cur.close()
    conn.close()
    return session_id

def get_user_sessions(user_id):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id, title FROM sessions WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def delete_session(session_id, user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    # Ensure user owns the session before deleting
    cur.execute("DELETE FROM sessions WHERE id = %s AND user_id = %s", (session_id, user_id))
    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return deleted

# --- Message Management ---
def add_message(session_id, role, content):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)", (session_id, role, content))
    conn.commit()
    cur.close()
    conn.close()

def get_session_messages(session_id):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT role, content FROM messages WHERE session_id = %s ORDER BY timestamp ASC", (session_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows