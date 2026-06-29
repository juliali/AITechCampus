import sqlite3
from pathlib import Path
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "auth.db"

_turso_client = None


class _TursoConnection:
    """Adapter that wraps libsql_client sync client to behave like sqlite3 connection."""

    def __init__(self, client):
        self._client = client

    def execute(self, sql, params=None):
        if params:
            result = self._client.execute(sql, params)
        else:
            result = self._client.execute(sql)
        return _TursoResult(result)

    def executescript(self, sql):
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._client.execute(stmt)

    def commit(self):
        pass

    def close(self):
        pass


class _TursoResult:
    """Adapter for libsql_client result sets to mimic sqlite3 cursor."""

    def __init__(self, result_set):
        self._rs = result_set

    def fetchone(self):
        if self._rs.rows:
            return _RowDict(self._rs.columns, self._rs.rows[0])
        return None

    def fetchall(self):
        return [_RowDict(self._rs.columns, row) for row in self._rs.rows]


class _RowDict:
    """Dict-like row object compatible with sqlite3.Row access patterns."""

    def __init__(self, columns, row):
        self._data = dict(zip(columns, row))

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()


def get_connection():
    global _turso_client
    turso_url = st.secrets.get("TURSO_DB_URL")
    turso_token = st.secrets.get("TURSO_AUTH_TOKEN")

    if turso_url and turso_token:
        import libsql_client
        if _turso_client is None:
            url = turso_url.replace("libsql://", "https://")
            _turso_client = libsql_client.create_client_sync(
                url, auth_token=turso_token
            )
        return _TursoConnection(_turso_client)
    else:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            ai_access_approved INTEGER DEFAULT 0,
            nickname TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS access_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            reason TEXT,
            reviewed_by INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            reviewed_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS user_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            detail TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # Migration: add nickname column for existing databases
    try:
        conn.execute("ALTER TABLE users ADD COLUMN nickname TEXT")
        conn.commit()
    except Exception:
        pass

    import hashlib
    salt = "sino_foreign_auth_2024"
    admin_hash = hashlib.sha256(f"{salt}771220".encode()).hexdigest()

    existing = conn.execute(
        "SELECT id FROM users WHERE email = ?", ("ye.julia.li@outlook.com",)
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO users (email, password_hash, is_admin, ai_access_approved) VALUES (?, ?, 1, 1)",
            ("ye.julia.li@outlook.com", admin_hash),
        )
        conn.commit()
    conn.close()
