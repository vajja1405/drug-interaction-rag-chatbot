"""
Memory and context engineering.

PatientMemory   long-term memory in SQLite: current medication list plus a log of past reviews, so a
                follow-up ("patient P-201 started ibuprofen") is checked against what the patient
                already takes without the clinician restating it.
ContextBuilder  turns retrieved label passages into a prompt under a token budget: keep only the
                sentences that name the partner drug (or its class) or carry risk language, rank them,
                and cut at the budget. Before/after token counts are returned so the saving is measured.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time

from agent.policy import HIGH_RISK
from agent.tools import mention_pattern

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # offline without the BPE file: ~4 characters per token
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


class PatientMemory:
    def __init__(self, path: str = ":memory:") -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS patients (patient_id TEXT PRIMARY KEY, medications TEXT, updated REAL);
            CREATE TABLE IF NOT EXISTS reviews (patient_id TEXT, case_id TEXT, summary TEXT, ts REAL);
        """)

    def seed(self, patient_id: str, medications: list[str]) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO patients VALUES (?, ?, ?)",
                             (patient_id, json.dumps(sorted(set(medications))), time.time()))
            self._db.commit()

    def get(self, patient_id: str, recent: int = 3) -> dict:
        with self._lock:
            row = self._db.execute("SELECT medications FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
            reviews = self._db.execute(
                "SELECT case_id, summary FROM reviews WHERE patient_id = ? ORDER BY ts DESC LIMIT ?",
                (patient_id, recent)).fetchall()
        return {"patient_id": patient_id, "known": row is not None,
                "medications": json.loads(row[0]) if row else [],
                "recent_reviews": [{"case_id": c, "summary": s} for c, s in reviews]}

    def record(self, patient_id: str, medications: list[str], case_id: str, summary: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO patients VALUES (?, ?, ?)",
                             (patient_id, json.dumps(sorted(set(medications))), time.time()))
            self._db.execute("INSERT INTO reviews VALUES (?, ?, ?, ?)", (patient_id, case_id, summary[:500], time.time()))
            self._db.commit()


_SENT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


class ContextBuilder:
    def __init__(self, budget_tokens: int = 700, compress: bool = True) -> None:
        self.budget_tokens, self.compress = budget_tokens, compress

    def _sentences(self, ev: dict, pair: tuple[str, str]) -> list[tuple[float, str]]:
        partner = pair[1] if ev["label"] == pair[0] else pair[0]
        pat = mention_pattern(partner)
        out = []
        for s in _SENT.split(ev["text"]):
            s = s.strip()
            if len(s) < 25:
                continue
            names, risk = bool(pat.search(s)), bool(HIGH_RISK.search(s))
            if names or risk:
                out.append(((2.0 if names else 0.0) + (1.0 if risk else 0.0) - 0.01 * ev["rank"], s))
        return out

    def build(self, evidence: dict[str, list[dict]]) -> tuple[str, dict]:
        """evidence: {"a|b": [passages]} → prompt block citing passage ids, plus token accounting."""
        raw = "\n\n".join(f"## {k.replace('|', ' + ')}\n" + "\n".join(f"[{e['id']}] {e['text']}" for e in evs)
                          for k, evs in evidence.items())
        raw_tokens = count_tokens(raw) if raw else 0
        if not self.compress:
            return raw, {"raw_tokens": raw_tokens, "context_tokens": raw_tokens}
        per_pair = max(120, self.budget_tokens // max(1, len(evidence)))
        blocks = []
        for key, evs in evidence.items():
            pair = tuple(key.split("|"))
            ranked = sorted(((score, e["id"], s) for e in evs for score, s in self._sentences(e, pair)),
                            key=lambda t: -t[0])
            lines, used, seen = [], 0, set()
            for _, pid, s in ranked:
                if s in seen:
                    continue
                t = count_tokens(s) + 6
                if used + t > per_pair and lines:
                    break
                lines.append(f"[{pid}] {s}")
                used += t
                seen.add(s)
            blocks.append(f"## {pair[0]} + {pair[1]}\n" + "\n".join(lines))
        text = "\n\n".join(blocks)
        return text, {"raw_tokens": raw_tokens, "context_tokens": count_tokens(text)}
