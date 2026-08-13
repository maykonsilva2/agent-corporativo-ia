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

    # `raise` -> raising this exception stops the application and displays the error message to the user
    if conv_id is None:
        raise RuntimeError("Failed to create conversation - noID returned by SQLite.") 

    conn.close()
    return conv_id

def list_conversations() -> list:
    """Returns all conversations, from newest to oldest.

        LEFT JOIN with messages + COUNT: counts how many messages each conversation has,
        useful for display in the sidebar (e.g., "Support FAQ (5 messages)").

        ON - ON defines how the two tables connect. It's the rule that says: "for each conversation, find the messages that belong to it".

        Why LEFT JOIN?
        LEFT JOIN means: "show ALL conversations, even those with ZERO messages". (With a normal JOIN, conversations without messages would disappear from the result.)

        AS — gives a nickname to the column

        .fetchall() — collect all the results
        .fetchone() - only 1 row
        .fetcmay(5) - N row (e.g., 5)
    """
    conn = _get_connection()
    rows = conn.execute(
        """
        SELECT c.id, c.title, c.file_names, c.created_at, COUNT(m.id) AS message_count 
        FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
        GROUP BY c.id
        ORDER BY c.created_at  DESC
        """
    ).fetchall()

    conn.close()

    # dict(row) converts each row into a dictionary
    # Remember: because of conn.row_factory = sqlite3.Row, each row is a special sqlite3.Row object. It works, but it's not a normal Python dictionary.
    # So [dict(row) for row in rows] means:
    # "For EACH row in the list, convert it to a dictionary — and return a new list with all of them."
    return [dict(row) for row in rows]

def get_messages(conversation_id: int) -> list:
    """Returns all messages from a conversation in chronological order.

    ORDER BY id ASC (not by timestamp): ensures the correct order even if
    two messages are saved in the same second.

    Instead of inserting the variable directly into the SQL text string, you put a ? as a placeholder. Then, you pass the variable as a separate item in a tuple (). This is called parameterization and prevents SQL injection attacks. It is the correct way to pass variables into SQL queries in Python.

    why the common in end (conversation_id,) -> The common comma in (conversation_id,) is to ensure that the tuple has at least one element.

    Parentheses alone don't make a tuple: 
    x = (conversation_id)      ← this is NOT a tuple!
    type(x)                    <class 'int'>

    y = (conversation_id,)     ← THIS is a tuple
    type(y)                    <class 'tuple'>
    """
    conn = _get_connection()
    rows = conn.execute(
        """
            SELECT role, content FROM WHERE conversation_id = ?
            ORDER BY id ASC
        """, (conversation_id,),
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]

def save_message(conversation_id: int, role: str, content: str) -> None:
    """Saves a single message (user question or agent response)."""

    conn = _get_connection()
    conn.execute(
        """INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)
        """, (conversation_id, role, content, datetime.now().isoformat(timespec="seconds")),
    )

    conn.commit()
    conn.close()

def delete_conversation(conversation_id: int) -> None:
    """Deletes a conversation and all its messages."""
    conn = _get_connection()
    conn.execute(
        "DELETE FROM conversations WHERE id = ?", (conversation_id,)
    )
    conn.commit()
    conn.close()