"""Per-process admission controls. Provider billing caps remain separate."""
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Lock
class AdmissionDenied(Exception): pass
class AnalysisBudget:
    def __init__(self, concurrency, daily, clock=None):
        self.limit=concurrency; self.daily=daily; self.lock=Lock(); self.active=0; self.used=0
        self.clock=clock or (lambda: datetime.now(timezone.utc).date()); self.day=self.clock()
    @contextmanager
    def reserve(self):
        with self.lock:
            day=self.clock()
            if day!=self.day: self.day=day; self.used=0
            if self.active>=self.limit: raise AdmissionDenied('The demo is busy. Please retry shortly.')
            if self.used>=self.daily: raise AdmissionDenied('The daily demo allowance is exhausted. Please try another day.')
            self.active+=1; self.used+=1
        try: yield
        finally:
            with self.lock: self.active-=1
