"""
In-memory rolling performance tracker for latency and throughput.
Thread-safe. Bounded ring buffer avoids unbounded memory growth.
"""
from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Deque, Dict


class PerfTracker:
    def __init__(self, window: int = 500):
        self._buf: Deque[tuple[float, float]] = deque(maxlen=window)   # (ts, ms)
        self._lock = Lock()

    def record(self, ms: float) -> None:
        with self._lock:
            self._buf.append((time.time(), ms))

    def stats(self) -> Dict[str, float]:
        with self._lock:
            pts = list(self._buf)
        if not pts:
            return {"p50": 0, "p90": 0, "p99": 0, "mean": 0, "max": 0,
                    "throughput_fps": 0, "samples": 0}

        ms_vals = sorted(ms for _, ms in pts)
        n = len(ms_vals)
        p = lambda q: ms_vals[min(n - 1, int(q * n))]
        mean = sum(ms_vals) / n

        # Throughput: 1/average over the observation window
        ts_span = max(pts[-1][0] - pts[0][0], 1e-6)
        throughput = n / ts_span if ts_span > 1 else n / 1.0

        return {
            "p50": round(p(0.50), 2),
            "p90": round(p(0.90), 2),
            "p99": round(p(0.99), 2),
            "mean": round(mean, 2),
            "max": round(ms_vals[-1], 2),
            "throughput_fps": round(throughput, 2),
            "samples": n,
        }


# Module-level singleton
TRACKER = PerfTracker()
