"""
SQLite persistence layer for the Biomedical Waste AI system.

Tables:
  detections         — one row per detection (analytics)
  model_metrics      — latest test-set metrics for EfficientNet-B0
  confusion_matrix   — 6x6 CM + labels (JSON)
  model_comparison   — rows for EfficientNet-B0 vs ResNet50 vs YOLO-only
  failure_cases      — misclassified examples for the dashboard viewer
  confidence_hist    — binned confidence distribution for calibration
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from config import DB_PATH, CATEGORIES, BIN_MAP, CONFIDENCE_THRESHOLD

_LOCK = threading.Lock()


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _LOCK, _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS detections (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            TEXT NOT NULL,
                category      TEXT NOT NULL,
                bin           TEXT NOT NULL,
                confidence    REAL NOT NULL,
                det_conf      REAL NOT NULL,
                alerts        TEXT NOT NULL,
                bbox          TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_det_ts ON detections(ts);

            CREATE TABLE IF NOT EXISTS model_metrics (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT NOT NULL,
                accuracy   REAL, precision REAL, recall REAL, f1 REAL,
                per_class  TEXT
            );

            CREATE TABLE IF NOT EXISTS confusion_matrix (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT NOT NULL,
                labels    TEXT NOT NULL,          -- JSON list of class names
                matrix    TEXT NOT NULL,          -- JSON 2D list (rows=true, cols=pred)
                model     TEXT NOT NULL DEFAULT 'efficientnet_b0'
            );

            CREATE TABLE IF NOT EXISTS model_comparison (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT NOT NULL,
                model           TEXT NOT NULL,       -- efficientnet_b0, resnet50, yolo_only
                accuracy        REAL, precision REAL, recall REAL, f1 REAL,
                params_m        REAL,                -- millions
                inference_ms    REAL,                -- mean latency ms
                notes           TEXT
            );

            CREATE TABLE IF NOT EXISTS failure_cases (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            TEXT NOT NULL,
                image_path    TEXT NOT NULL,
                true_label    TEXT NOT NULL,
                pred_label    TEXT NOT NULL,
                confidence    REAL NOT NULL,
                reason        TEXT,
                image_b64     TEXT                    -- small thumbnail
            );

            CREATE TABLE IF NOT EXISTS confidence_hist (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT NOT NULL,
                bins      TEXT NOT NULL,   -- JSON list of bin edges
                counts    TEXT NOT NULL,   -- JSON list of counts
                correct   TEXT             -- JSON list of counts of correct preds per bin
            );

            CREATE TABLE IF NOT EXISTS ablation_results (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT NOT NULL,
                setting   TEXT NOT NULL,
                accuracy  REAL, f1 REAL,
                notes     TEXT
            );
        """)


# ──────────────────────────────────────────────────────────
# Detections
# ──────────────────────────────────────────────────────────
def save_detections(detections: List[dict]) -> None:
    if not detections:
        return
    rows = [
        (
            datetime.utcnow().isoformat(),
            d["category"], d["bin"],
            float(d["classification_confidence"]),
            float(d["detection_confidence"]),
            json.dumps(d.get("alerts", [])),
            json.dumps(d["bbox"]),
        )
        for d in detections
    ]
    with _LOCK, _conn() as c:
        c.executemany(
            "INSERT INTO detections (ts, category, bin, confidence, det_conf, alerts, bbox) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )


def recent_history(limit: int = 50) -> List[dict]:
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT id, ts, category, bin, confidence, alerts FROM detections "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"id": r["id"], "ts": r["ts"], "category": r["category"], "bin": r["bin"],
         "confidence": r["confidence"], "alerts": json.loads(r["alerts"])}
        for r in rows
    ]


def aggregate_metrics() -> dict:
    """Resilient aggregation — every field has a sensible default on empty DB."""
    try:
        with _LOCK, _conn() as c:
            total = c.execute("SELECT COUNT(*) FROM detections").fetchone()[0] or 0

            # Initialise with every known key so the frontend always has full data
            counts_cat = {cat: 0 for cat in CATEGORIES}
            counts_bin = {b: 0 for b in set(BIN_MAP.values())}

            for r in c.execute(
                "SELECT category, bin, COUNT(*) AS n FROM detections "
                "GROUP BY category, bin"
            ).fetchall():
                if r["category"] in counts_cat:
                    counts_cat[r["category"]] += r["n"]
                if r["bin"] in counts_bin:
                    counts_bin[r["bin"]] += r["n"]

            avg_conf = c.execute(
                "SELECT AVG(confidence) FROM detections"
            ).fetchone()[0] or 0.0
            hazardous = c.execute(
                "SELECT COUNT(*) FROM detections WHERE alerts LIKE '%HAZARDOUS%'"
            ).fetchone()[0] or 0
            low_conf = c.execute(
                "SELECT COUNT(*) FROM detections WHERE confidence < ?",
                (CONFIDENCE_THRESHOLD,),
            ).fetchone()[0] or 0

            since = (datetime.utcnow() - timedelta(days=7)).isoformat()
            ts_rows = c.execute(
                "SELECT substr(ts,1,13) AS hour, COUNT(*) AS n "
                "FROM detections WHERE ts >= ? GROUP BY hour ORDER BY hour",
                (since,),
            ).fetchall()
    except sqlite3.OperationalError:
        # DB file doesn't exist yet — return empty shell
        total = 0; avg_conf = 0.0; hazardous = 0; low_conf = 0
        counts_cat = {cat: 0 for cat in CATEGORIES}
        counts_bin = {b: 0 for b in set(BIN_MAP.values())}
        ts_rows = []

    hazard_rate = (hazardous / total) if total else 0.0
    return {
        "total_detections": int(total),
        "counts_by_category": counts_cat,
        "counts_by_bin": counts_bin,
        "avg_confidence": round(float(avg_conf), 4),
        "hazard_rate": round(hazard_rate, 4),
        "low_confidence_count": int(low_conf),
        "timeseries": [{"hour": r["hour"], "count": r["n"]} for r in ts_rows],
    }


# ──────────────────────────────────────────────────────────
# Model metrics / CM / comparison / confidence hist / failures
# ──────────────────────────────────────────────────────────
def save_model_metrics(m: dict) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO model_metrics (ts, accuracy, precision, recall, f1, per_class) "
            "VALUES (?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), m.get("accuracy"), m.get("precision"),
             m.get("recall"), m.get("f1"), json.dumps(m.get("per_class", {}))),
        )


def latest_model_metrics() -> dict:
    with _LOCK, _conn() as c:
        row = c.execute(
            "SELECT accuracy, precision, recall, f1, per_class "
            "FROM model_metrics ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "per_class": {}}
    return {
        "accuracy": row["accuracy"] or 0.0, "precision": row["precision"] or 0.0,
        "recall": row["recall"] or 0.0, "f1": row["f1"] or 0.0,
        "per_class": json.loads(row["per_class"] or "{}"),
    }


def save_confusion_matrix(labels: list, matrix: list, model: str = "efficientnet_b0") -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO confusion_matrix (ts, labels, matrix, model) VALUES (?,?,?,?)",
            (datetime.utcnow().isoformat(), json.dumps(labels), json.dumps(matrix), model),
        )


def latest_confusion_matrix(model: str = "efficientnet_b0") -> dict:
    with _LOCK, _conn() as c:
        row = c.execute(
            "SELECT labels, matrix, model, ts FROM confusion_matrix "
            "WHERE model = ? ORDER BY id DESC LIMIT 1",
            (model,),
        ).fetchone()
    if not row:
        return {"labels": CATEGORIES, "matrix": [[0] * len(CATEGORIES)] * len(CATEGORIES),
                "model": model, "ts": None}
    return {"labels": json.loads(row["labels"]), "matrix": json.loads(row["matrix"]),
            "model": row["model"], "ts": row["ts"]}


def save_comparison_row(r: dict) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO model_comparison (ts, model, accuracy, precision, recall, f1, "
            "params_m, inference_ms, notes) VALUES (?,?,?,?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), r["model"],
             r.get("accuracy"), r.get("precision"), r.get("recall"), r.get("f1"),
             r.get("params_m"), r.get("inference_ms"), r.get("notes", "")),
        )


def latest_comparison() -> list:
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT model, MAX(ts) AS ts, accuracy, precision, recall, f1, params_m, "
            "inference_ms, notes FROM model_comparison GROUP BY model ORDER BY f1 DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def save_confidence_hist(bins: list, counts: list, correct: list | None = None) -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO confidence_hist (ts, bins, counts, correct) VALUES (?,?,?,?)",
            (datetime.utcnow().isoformat(), json.dumps(bins), json.dumps(counts),
             json.dumps(correct) if correct else None),
        )


def latest_confidence_hist() -> dict:
    with _LOCK, _conn() as c:
        row = c.execute(
            "SELECT bins, counts, correct FROM confidence_hist ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return {"bins": [], "counts": [], "correct": None}
    return {
        "bins": json.loads(row["bins"]),
        "counts": json.loads(row["counts"]),
        "correct": json.loads(row["correct"]) if row["correct"] else None,
    }


def save_failure_cases(cases: list[dict]) -> None:
    if not cases:
        return
    rows = [
        (datetime.utcnow().isoformat(), c["image_path"], c["true_label"],
         c["pred_label"], float(c["confidence"]), c.get("reason", ""),
         c.get("image_b64"))
        for c in cases
    ]
    with _LOCK, _conn() as c:
        c.executemany(
            "INSERT INTO failure_cases (ts, image_path, true_label, pred_label, "
            "confidence, reason, image_b64) VALUES (?,?,?,?,?,?,?)",
            rows,
        )


def get_failure_cases(limit: int = 24) -> list[dict]:
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT id, image_path, true_label, pred_label, confidence, reason, image_b64 "
            "FROM failure_cases ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        # Ensure image_b64 has the data URI prefix so browsers can render it
        b64 = d.get("image_b64") or ""
        if b64 and not b64.startswith("data:"):
            b64 = "data:image/jpeg;base64," + b64
        d["image_b64"] = b64
        result.append(d)
    return result


def save_ablation(setting: str, accuracy: float, f1: float, notes: str = "") -> None:
    with _LOCK, _conn() as c:
        c.execute(
            "INSERT INTO ablation_results (ts, setting, accuracy, f1, notes) VALUES (?,?,?,?,?)",
            (datetime.utcnow().isoformat(), setting, accuracy, f1, notes),
        )


def latest_ablation() -> list[dict]:
    with _LOCK, _conn() as c:
        rows = c.execute(
            "SELECT setting, MAX(ts) AS ts, accuracy, f1, notes "
            "FROM ablation_results GROUP BY setting ORDER BY accuracy DESC"
        ).fetchall()
    return [dict(r) for r in rows]
