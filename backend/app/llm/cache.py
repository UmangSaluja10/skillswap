"""Tiny on-disk response cache (SQLite). Makes seeded demo data instant and free."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path


def make_key(kind: str, system: str | None, prompt: str, **extra) -> str:
    blob = json.dumps([kind, system or "", prompt, extra], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS cache("
                             "key TEXT PRIMARY KEY, value TEXT NOT NULL, meta TEXT, created REAL)")
                conn.commit()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def get(self, key: str) -> tuple[str, dict] | None:
        if not self.enabled:
            return None
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT value, meta FROM cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        return row[0], json.loads(row[1] or "{}")

    def set(self, key: str, value: str, meta: dict | None = None) -> None:
        if not self.enabled:
            return
        with closing(self._connect()) as conn:
            conn.execute("INSERT OR REPLACE INTO cache(key, value, meta, created) VALUES (?,?,?,?)",
                         (key, value, json.dumps(meta or {}), time.time()))
            conn.commit()

    def delete(self, key: str) -> None:
        if not self.enabled:
            return
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM cache WHERE key=?", (key,))
            conn.commit()

    def clear(self) -> None:
        if not self.enabled:
            return
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
