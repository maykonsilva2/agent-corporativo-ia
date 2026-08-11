"""
database.py -SQLite persistence layer for chat history.
Uses only sqllite3 (Python built-in, no extra dependencies).
"""

import sqlite3
from datetime import datetime

#  The file chat_history.db file is created automatically o the first run.
DB_PATH = "chat_history.db"

def _get_connection() -> sqlite3.Connection:

    # Create database connection, the check_same_thread=False is necessary because Streamlit runs on multiple threads(LngGraph may execute tools in different threads).
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    # Allows accessing columns by name (row["title"]) instead of numeric indices (row[0]), making the code more readable.
    conn.row_factory = sqlite3.Row

    # Enforce foreign key constraints (ON DELETE CASCADE will automatically delete dependent rows when a parent row is deleted).
    conn.execute("PRAGMA foreign_keys = ON")

    return conn

def init_db() -> None:
    """
        Create the tables if they do not exist. Call this once during ap initialization.

        ON DELETE CASCADE: When a conversation is deleted, all its messages are automatically removed by the database, we don't need to delete them manually.
    """

    conn = _get_connection()
    conn.executescript(
        """
            CREATE TABLE IF NOT EXISTS conversations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                title           TEXT    NOT NULL,
                file_names      TEXT    NOT NULL DEFAULT '',
                created_at      TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
        """
    )
    conn.commit() # save the changes
    conn.close() # close the connection

def create_conversation(title: str, file_names: list) -> int:
    """
        Creates a new convesation and returns its ID.

        file_names is a list of filenames that will be joined with ';'.
        E.g., ["politica_privacidade.txt", "faq_suporte.txt"] -> "politica_privacidade.txt;faq_suporte.txt"

        This makes it possible to identify which documents a conversation used, so they can be reindexed when the user reopens the conversation.
    """

    conn = _get_connection()
    cursor = conn.execute(
        "INSERT INTO conversations (title, file_names, created_at) VALUES (?, ?, ?)",
        (title, ";".join(file_names), datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conv_id = cursor.lastrowid # It returns the id of the last row inserted.
    conn.close()
    return conv_id
