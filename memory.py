"""
Persistent SQLite Memory Engine.
Remembers user profiles, trading styles, favorite assets, and conversation history across server restarts.
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DB_DIR, "user_memory.sqlite")


def get_db_connection():
    """Initializes SQLite database and tables if they don't exist."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates tables for persistent user profiles and chat history."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # User Profile table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            display_name TEXT,
            preferred_assets TEXT DEFAULT '[]',
            trading_style TEXT DEFAULT 'Swing Trader',
            experience_level TEXT DEFAULT 'Intermediate',
            notes TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Chat history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# Initialize database on module load
init_db()


# ----------------------------------------------------------------------
# PROFILE & PREFERENCE MANAGEMENT
# ----------------------------------------------------------------------
def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Retrieves or creates a user profile."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (str(user_id),))
    row = cursor.fetchone()

    if not row:
        # Create new profile
        cursor.execute(
            "INSERT INTO user_profiles (user_id, display_name) VALUES (?, ?)",
            (str(user_id), "Trader"),
        )
        conn.commit()
        cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (str(user_id),))
        row = cursor.fetchone()

    profile = dict(row)
    try:
        profile["preferred_assets"] = json.loads(profile.get("preferred_assets", "[]"))
    except Exception:
        profile["preferred_assets"] = []

    conn.close()
    return profile


def update_user_profile(
    user_id: str,
    display_name: Optional[str] = None,
    username: Optional[str] = None,
    preferred_assets: Optional[List[str]] = None,
    trading_style: Optional[str] = None,
    notes: Optional[str] = None,
):
    """Upserts user profile and activity timestamp."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT user_id FROM user_profiles WHERE user_id = ?", (str(user_id),))
    exists = cursor.fetchone()

    if not exists:
        cursor.execute(
            """
            INSERT INTO user_profiles (user_id, display_name, username, preferred_assets, trading_style, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(user_id),
                display_name or "Trader",
                username or "",
                json.dumps(preferred_assets or []),
                trading_style or "Swing Trader",
                notes or "",
            ),
        )
    else:
        updates = []
        values = []
        if display_name:
            updates.append("display_name = ?")
            values.append(display_name)
        if username is not None:
            updates.append("username = ?")
            values.append(username)
        if preferred_assets is not None:
            updates.append("preferred_assets = ?")
            values.append(json.dumps(preferred_assets))
        if trading_style:
            updates.append("trading_style = ?")
            values.append(trading_style)
        if notes:
            updates.append("notes = ?")
            values.append(notes)

        updates.append("updated_at = CURRENT_TIMESTAMP")
        query = f"UPDATE user_profiles SET {', '.join(updates)} WHERE user_id = ?"
        values.append(str(user_id))
        cursor.execute(query, values)

    conn.commit()
    conn.close()


def record_activity(
    user_id: str,
    display_name: str = "",
    username: str = "",
    user_message: str = "",
    bot_response: str = "",
):
    """Records an interaction, ensuring the user profile is upserted and chat message is logged."""
    update_user_profile(user_id, display_name=display_name, username=username)
    if user_message:
        save_chat_message(user_id, "user", user_message)
    if bot_response:
        save_chat_message(user_id, "assistant", bot_response)


# ----------------------------------------------------------------------
# CHAT HISTORY PERSISTENCE
# ----------------------------------------------------------------------
def save_chat_message(user_id: str, role: str, content: str):
    """Saves a conversation turn to SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_messages (user_id, role, content) VALUES (?, ?, ?)",
        (str(user_id), role, content),
    )
    conn.commit()
    conn.close()


def get_recent_chat_history(user_id: str, limit: int = 12) -> List[Dict[str, str]]:
    """Fetches the most recent chat history for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM chat_messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (str(user_id), limit),
    )
    rows = cursor.fetchall()
    conn.close()

    # Reverse to return chronological order
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def clear_user_history(user_id: str):
    """Clears conversation history while preserving user profile preferences."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_messages WHERE user_id = ?", (str(user_id),))
    conn.commit()
    conn.close()


def build_personalized_system_context(user_id: str) -> str:
    """Constructs personalized memory prompt context for the AI."""
    profile = get_user_profile(user_id)
    name = profile.get("display_name", "Trader")
    style = profile.get("trading_style", "Swing Trader")
    assets = ", ".join(profile.get("preferred_assets", [])) or "Major pairs, Gold, Crypto"
    notes = profile.get("notes", "")

    context_str = f"USER MEMORY & PROFILE:\n- User's Name: {name}\n- Trading Style: {style}\n- Frequently Watched Assets: {assets}"
    if notes:
        context_str += f"\n- Trader Notes: {notes}"
    return context_str


# ----------------------------------------------------------------------
# SYSTEM ANALYTICS & USER TRACKING
# ----------------------------------------------------------------------
def get_system_analytics() -> Dict[str, Any]:
    """Fetches real-time user count, message volume, and active user list."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total users
    cursor.execute("SELECT COUNT(*) FROM user_profiles")
    total_users = cursor.fetchone()[0]

    # Total messages
    cursor.execute("SELECT COUNT(*) FROM chat_messages")
    total_messages = cursor.fetchone()[0]

    # Active users in last 24h
    cursor.execute("""
        SELECT COUNT(DISTINCT user_id) FROM chat_messages 
        WHERE timestamp >= datetime('now', '-1 day')
    """)
    active_24h = cursor.fetchone()[0]

    # List of recent users with message counts
    cursor.execute("""
        SELECT 
            p.user_id,
            p.display_name,
            p.username,
            p.updated_at,
            (SELECT COUNT(*) FROM chat_messages m WHERE m.user_id = p.user_id) as msg_count
        FROM user_profiles p
        ORDER BY p.updated_at DESC
        LIMIT 25
    """)
    recent_users = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return {
        "total_users": total_users,
        "total_messages": total_messages,
        "active_24h": active_24h,
        "recent_users": recent_users,
    }
