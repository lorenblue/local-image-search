from __future__ import annotations

import sqlite3
from pathlib import Path

from local_image_search.db import connect, connect_readonly, init_db


def test_init_db_does_not_store_embedding_json(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    with connect(db_path) as conn:
        init_db(conn)

        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(images)").fetchall()
        }

    assert "embedding_json" not in columns


def test_connect_readonly_reads_without_allowing_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "images.db"
    with connect(db_path) as conn:
        init_db(conn)

    with connect_readonly(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM images").fetchone()
        assert row["count"] == 0

        try:
            conn.execute("INSERT INTO schema_meta (key, value) VALUES ('x', 'y')")
        except sqlite3.OperationalError as exc:
            assert "readonly" in str(exc).lower() or "read-only" in str(exc).lower()
        else:
            raise AssertionError("read-only connection unexpectedly allowed a write")
