import sqlite3
import requests
from pathlib import Path
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "auth.db"


class _TursoConnection:
    """Connect to Turso via HTTP REST API (no WebSocket, no native libs)."""

    def __init__(self, url, token):
        self._url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _post(self, sql, params=None):
        stmt = {"sql": sql}
        if params:
            if isinstance(params, (list, tuple)):
                stmt["args"] = [self._typed_arg(v) for v in params]
            elif isinstance(params, dict):
                stmt["named_args"] = [{"name": k, "value": self._typed_arg(v)} for k, v in params.items()]
        body = {"requests": [{"type": "execute", "stmt": stmt}, {"type": "close"}]}
        resp = requests.post(f"{self._url}/v3/pipeline", json=body, headers=self._headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _typed_arg(value):
        if value is None:
            return {"type": "null"}
        elif isinstance(value, int):
            return {"type": "integer", "value": str(value)}
        elif isinstance(value, float):
            return {"type": "float", "value": str(value)}
        else:
            return {"type": "text", "value": str(value)}

    def execute(self, sql, params=None):
        result = self._post(sql, params)
        return _TursoResult(result)

    def executescript(self, sql):
        reqs = [{"type": "execute", "stmt": {"sql": s.strip()}} for s in sql.split(";") if s.strip()]
        reqs.append({"type": "close"})
        body = {"requests": reqs}
        resp = requests.post(f"{self._url}/v3/pipeline", json=body, headers=self._headers, timeout=15)
        resp.raise_for_status()

    def commit(self):
        pass

    def close(self):
        pass


class _TursoResult:
    """Parse Turso HTTP API response into Row-like objects."""

    def __init__(self, response):
        self._rows = []
        self._columns = []
        if isinstance(response, list) and response:
            res = response[0].get("results", {})
            self._columns = res.get("columns", [])
            self._rows = res.get("rows", [])
        elif isinstance(response, dict):
            results = response.get("results", [])
            if results:
                res = results[0].get("response", {}).get("result", {})
                self._columns = res.get("cols", [])
                cols = [c.get("name", "") if isinstance(c, dict) else c for c in self._columns]
                self._columns = cols
                self._rows = []
                for row in res.get("rows", []):
                    self._rows.append([cell.get("value") if isinstance(cell, dict) else cell for cell in row])

    def fetchone(self):
        if self._rows:
            return _RowDict(self._columns, self._rows[0])
        return None

    def fetchall(self):
        return [_RowDict(self._columns, row) for row in self._rows]


class _RowDict:
    """Dict-like row object compatible with sqlite3.Row access patterns."""

    def __init__(self, columns, row):
        self._data = dict(zip(columns, row))
        self._values = list(row)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()


def get_connection():
    turso_url = st.secrets.get("TURSO_DB_URL")
    turso_token = st.secrets.get("TURSO_AUTH_TOKEN")

    if turso_url and turso_token:
        url = turso_url.replace("libsql://", "https://")
        return _TursoConnection(url, turso_token)
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

    admin_email = st.secrets.get("ADMIN_EMAIL")
    admin_password = st.secrets.get("ADMIN_PASSWORD")
    if admin_email and admin_password:
        import hashlib
        salt = "sino_foreign_auth_2024"
        admin_hash = hashlib.sha256(f"{salt}{admin_password}".encode()).hexdigest()
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (admin_email,)
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO users (email, password_hash, is_admin, ai_access_approved) VALUES (?, ?, 1, 1)",
                (admin_email, admin_hash),
            )
            conn.commit()
    conn.close()
