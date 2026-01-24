import sqlite3
import json
from typing import Dict, Any
from datetime import datetime

def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def _column_exists(con: sqlite3.Connection, table: str, col: str) -> bool:
    cur = con.cursor()
    rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == col for r in rows)

def init_db(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS deviations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        telegram_user_id INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'draft',
        user_input_json TEXT NOT NULL,
        selected_json TEXT,
        sections_json TEXT,
        chosen_variants_json TEXT,  -- {section_key: variant_index}
        view_mode_json TEXT,        -- {section_key: "short"|"full"}
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)
    con.commit()

    # migrate older dbs
    if not _column_exists(con, "deviations", "chosen_variants_json"):
        cur.execute("ALTER TABLE deviations ADD COLUMN chosen_variants_json TEXT")
    if not _column_exists(con, "deviations", "view_mode_json"):
        cur.execute("ALTER TABLE deviations ADD COLUMN view_mode_json TEXT")
    con.commit()

def create_deviation(con, telegram_user_id: int, user_input: Dict[str, Any]) -> int:
    cur = con.cursor()
    ts = now_iso()
    cur.execute(
        "INSERT INTO deviations(telegram_user_id, user_input_json, created_at, updated_at) VALUES (?,?,?,?)",
        (telegram_user_id, json.dumps(user_input, ensure_ascii=False), ts, ts),
    )
    con.commit()
    return cur.lastrowid

def update_deviation(con, deviation_id: int, selected: Dict[str, Any] | None = None, sections: Dict[str, Any] | None = None,
                     chosen_variants: Dict[str, Any] | None = None, view_mode: Dict[str, Any] | None = None):
    cur = con.cursor()
    fields = []
    params = []
    if selected is not None:
        fields.append("selected_json=?")
        params.append(json.dumps(selected, ensure_ascii=False))
    if sections is not None:
        fields.append("sections_json=?")
        params.append(json.dumps(sections, ensure_ascii=False))
    if chosen_variants is not None:
        fields.append("chosen_variants_json=?")
        params.append(json.dumps(chosen_variants, ensure_ascii=False))
    if view_mode is not None:
        fields.append("view_mode_json=?")
        params.append(json.dumps(view_mode, ensure_ascii=False))

    fields.append("updated_at=?")
    params.append(now_iso())
    params.append(deviation_id)

    cur.execute(f"UPDATE deviations SET {', '.join(fields)} WHERE id=?", params)
    con.commit()

def get_deviation(con, deviation_id: int) -> Dict[str, Any]:
    cur = con.cursor()
    row = cur.execute("SELECT * FROM deviations WHERE id=?", (deviation_id,)).fetchone()
    if not row:
        raise ValueError("deviation not found")
    return dict(row)

def _loads(s: str | None):
    if not s:
        return {}
    return json.loads(s)

def get_chosen_variant(row: Dict[str, Any], section_key: str) -> int:
    data = _loads(row.get("chosen_variants_json"))
    return int(data.get(section_key, 0))

def set_chosen_variant(con, deviation_id: int, row: Dict[str, Any], section_key: str, idx: int):
    data = _loads(row.get("chosen_variants_json"))
    data[section_key] = int(idx)
    update_deviation(con, deviation_id, chosen_variants=data)

def get_view_mode(row: Dict[str, Any], section_key: str) -> str:
    data = _loads(row.get("view_mode_json"))
    return data.get(section_key, "short")

def toggle_view_mode(con, deviation_id: int, row: Dict[str, Any], section_key: str):
    data = _loads(row.get("view_mode_json"))
    cur = data.get(section_key, "short")
    data[section_key] = "full" if cur == "short" else "short"
    update_deviation(con, deviation_id, view_mode=data)
