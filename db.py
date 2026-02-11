import os
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import unquote

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool


# Usamos nombres no conflictivos (evita el choque que te dio con "sessions")
SESSIONS_TABLE = "chat_sessions"
MESSAGES_TABLE = "chat_messages"


DATABASE_URL = os.getenv("DATABASE_URL")  # postgresql://user:pass@host:5432/dbname?sslmode=require
if not DATABASE_URL:
    raise ValueError("Missing environment variable: DATABASE_URL")

# Azure App Settings puede dejar caracteres escapados (%24, %40, etc.)
DATABASE_URL = unquote(DATABASE_URL)

_pool: Optional[SimpleConnectionPool] = None


def _get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(
            minconn=1,
            maxconn=int(os.getenv("DB_POOL_MAX", "5")),
            dsn=DATABASE_URL,
        )
    return _pool


def get_conn():
    pool = _get_pool()
    return pool.getconn()


def put_conn(conn):
    pool = _get_pool()
    pool.putconn(conn)


def init_db() -> None:
    """
    Inicializa tablas. Idempotente y sin "sessions" para evitar conflictos.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {SESSIONS_TABLE} (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL
                );
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {MESSAGES_TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES {SESSIONS_TABLE}(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL
                );
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{MESSAGES_TABLE}_session_id_id
                ON {MESSAGES_TABLE}(session_id, id);
            """)
            conn.commit()
    finally:
        put_conn(conn)


def create_session(session_id: Optional[str] = None) -> str:
    if not session_id:
        import uuid
        session_id = uuid.uuid4().hex

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            now = datetime.utcnow()
            cur.execute(
                f"INSERT INTO {SESSIONS_TABLE} (id, created_at) VALUES (%s, %s) "
                f"ON CONFLICT (id) DO NOTHING",
                (session_id, now),
            )
            conn.commit()
    finally:
        put_conn(conn)
    return session_id


def session_exists(session_id: str) -> bool:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {SESSIONS_TABLE} WHERE id = %s", (session_id,))
            return cur.fetchone() is not None
    finally:
        put_conn(conn)


def add_message(session_id: str, role: str, content: str) -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            now = datetime.utcnow()
            cur.execute(
                f"INSERT INTO {MESSAGES_TABLE} (session_id, role, content, created_at) "
                f"VALUES (%s, %s, %s, %s)",
                (session_id, role, content, now),
            )
            conn.commit()
    finally:
        put_conn(conn)


def get_recent_messages(session_id: str, limit: int = 50) -> List[Dict]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT role, content, created_at "
                f"FROM {MESSAGES_TABLE} "
                f"WHERE session_id = %s "
                f"ORDER BY id DESC LIMIT %s",
                (session_id, limit),
            )
            rows = cur.fetchall()
            rows = list(reversed(rows))
            return [dict(r) for r in rows]
    finally:
        put_conn(conn)


def list_sessions() -> List[Dict]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT s.id,
                       s.created_at,
                       COUNT(m.id) as msg_count,
                       MAX(m.created_at) as last_message
                FROM {SESSIONS_TABLE} s
                LEFT JOIN {MESSAGES_TABLE} m ON s.id = m.session_id
                GROUP BY s.id
                ORDER BY COALESCE(MAX(m.created_at), s.created_at) DESC
            """)
            return [dict(r) for r in cur.fetchall()]
    finally:
        put_conn(conn)


def clear_session(session_id: str) -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {MESSAGES_TABLE} WHERE session_id = %s", (session_id,))
            conn.commit()
    finally:
        put_conn(conn)


def delete_session(session_id: str) -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {SESSIONS_TABLE} WHERE id = %s", (session_id,))
            conn.commit()
    finally:
        put_conn(conn)


def delete_all_sessions() -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {SESSIONS_TABLE}")
            conn.commit()
    finally:
        put_conn(conn)
