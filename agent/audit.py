"""Append-only, hash-chained decision log: editing or deleting any past entry breaks verify()."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time


class DecisionLog:
    def __init__(self, path: str = ":memory:") -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS decisions (seq INTEGER PRIMARY KEY, body TEXT, hash TEXT)")
        self._lock = threading.Lock()

    @staticmethod
    def _digest(body: str, prev: str) -> str:
        return hashlib.sha256((prev + body).encode()).hexdigest()

    def append(self, **entry) -> str:
        with self._lock:
            row = self._db.execute("SELECT hash FROM decisions ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else "genesis"
            body = json.dumps({"ts": round(time.time(), 3), **entry}, sort_keys=True, default=str)
            h = self._digest(body, prev)
            self._db.execute("INSERT INTO decisions (body, hash) VALUES (?, ?)", (body, h))
            self._db.commit()
            return h

    def entries(self) -> list[dict]:
        return [json.loads(b) for (b,) in self._db.execute("SELECT body FROM decisions ORDER BY seq")]

    def verify(self) -> bool:
        prev = "genesis"
        for body, h in self._db.execute("SELECT body, hash FROM decisions ORDER BY seq"):
            if self._digest(body, prev) != h:
                return False
            prev = h
        return True
