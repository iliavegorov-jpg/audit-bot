import json
import os
from typing import Dict, Any
from datetime import datetime, date

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgresql"):
    import psycopg2
    from psycopg2.extras import RealDictCursor
    USE_POSTGRES = True
else:
    import sqlite3
    USE_POSTGRES = False

def connect(path: str = None):
    if USE_POSTGRES:
        return psycopg2.connect(DATABASE_URL)
    else:
        con = sqlite3.connect(path or "audit.db")
        con.row_factory = sqlite3.Row
        return con

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def init_db(con):
    cur = con.cursor()
    if USE_POSTGRES:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS deviations (
            id SERIAL PRIMARY KEY,
            telegram_user_id BIGINT NOT NULL,
            status TEXT NOT NULL DEFAULT 'draft',
            user_input_json TEXT NOT NULL,
            selected_json TEXT,
            sections_json TEXT,
            chosen_variants_json TEXT,
            view_mode_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS authorized_users (
            telegram_user_id BIGINT PRIMARY KEY,
            authorized_date DATE NOT NULL
        )""")
    else:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS deviations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_user_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'draft',
            user_input_json TEXT NOT NULL,
            selected_json TEXT,
            sections_json TEXT,
            chosen_variants_json TEXT,
            view_mode_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS authorized_users (
            telegram_user_id INTEGER PRIMARY KEY,
            authorized_date TEXT NOT NULL
        )""")
    con.commit()

def create_deviation(con, telegram_user_id: int, user_input: Dict[str, Any]) -> int:
    ts = now_iso()
    cur = con.cursor()
    if USE_POSTGRES:
        cur.execute(
            "INSERT INTO deviations(telegram_user_id, user_input_json, created_at, updated_at) VALUES (%s,%s,%s,%s) RETURNING id",
            (telegram_user_id, json.dumps(user_input, ensure_ascii=False), ts, ts))
        row_id = cur.fetchone()[0]
    else:
        cur.execute(
            "INSERT INTO deviations(telegram_user_id, user_input_json, created_at, updated_at) VALUES (?,?,?,?)",
            (telegram_user_id, json.dumps(user_input, ensure_ascii=False), ts, ts))
        row_id = cur.lastrowid
    con.commit()
    return row_id

def update_deviation(con, deviation_id: int, selected: Dict[str, Any] | None = None, sections: Dict[str, Any] | None = None,
                     chosen_variants: Dict[str, Any] | None = None, view_mode: Dict[str, Any] | None = None):
    fields, params = [], []
    ph = "%s" if USE_POSTGRES else "?"
    
    if selected is not None:
        fields.append(f"selected_json={ph}")
        params.append(json.dumps(selected, ensure_ascii=False))
    if sections is not None:
        fields.append(f"sections_json={ph}")
        params.append(json.dumps(sections, ensure_ascii=False))
    if chosen_variants is not None:
        fields.append(f"chosen_variants_json={ph}")
        params.append(json.dumps(chosen_variants, ensure_ascii=False))
    if view_mode is not None:
        fields.append(f"view_mode_json={ph}")
        params.append(json.dumps(view_mode, ensure_ascii=False))

    fields.append(f"updated_at={ph}")
    params.append(now_iso())
    params.append(deviation_id)

    cur = con.cursor()
    cur.execute(f"UPDATE deviations SET {', '.join(fields)} WHERE id={ph}", params)
    con.commit()

def get_deviation(con, deviation_id: int) -> Dict[str, Any]:
    cur = con.cursor(cursor_factory=RealDictCursor) if USE_POSTGRES else con.cursor()
    if USE_POSTGRES:
        cur.execute("SELECT * FROM deviations WHERE id=%s", (deviation_id,))
        row = cur.fetchone()
    else:
        row = cur.execute("SELECT * FROM deviations WHERE id=?", (deviation_id,)).fetchone()
        row = dict(row) if row else None
    if not row:
        raise ValueError("deviation not found")
    return dict(row)

def _loads(s):
    return json.loads(s) if s else {}

def get_chosen_variant(row: Dict[str, Any], section_key: str) -> int:
    return int(_loads(row.get("chosen_variants_json")).get(section_key, 0))

def set_chosen_variant(con, deviation_id: int, row: Dict[str, Any], section_key: str, idx: int):
    data = _loads(row.get("chosen_variants_json"))
    data[section_key] = int(idx)
    update_deviation(con, deviation_id, chosen_variants=data)

def get_view_mode(row: Dict[str, Any], section_key: str) -> str:
    return _loads(row.get("view_mode_json")).get(section_key, "short")

def toggle_view_mode(con, deviation_id: int, row: Dict[str, Any], section_key: str):
    data = _loads(row.get("view_mode_json"))
    data[section_key] = "full" if data.get(section_key, "short") == "short" else "short"
    update_deviation(con, deviation_id, view_mode=data)

def is_user_authorized(con, telegram_user_id: int) -> bool:
    today = date.today()
    cur = con.cursor()
    if USE_POSTGRES:
        cur.execute("SELECT authorized_date FROM authorized_users WHERE telegram_user_id=%s", (telegram_user_id,))
        row = cur.fetchone()
        return row and row[0] == today
    else:
        row = cur.execute("SELECT authorized_date FROM authorized_users WHERE telegram_user_id=?", (telegram_user_id,)).fetchone()
        return row and row[0] == str(today)

def authorize_user(con, telegram_user_id: int):
    today = date.today()
    cur = con.cursor()
    if USE_POSTGRES:
        cur.execute("""
            INSERT INTO authorized_users (telegram_user_id, authorized_date) VALUES (%s, %s)
            ON CONFLICT (telegram_user_id) DO UPDATE SET authorized_date = %s
        """, (telegram_user_id, today, today))
    else:
        cur.execute("INSERT OR REPLACE INTO authorized_users (telegram_user_id, authorized_date) VALUES (?, ?)",
                    (telegram_user_id, str(today)))
    con.commit()