"""
PostgreSQL connection pool and query helpers for the fraud detection POC.

Reads connection details from environment variables (see .env.example).
Uses a small connection pool since FastAPI can serve concurrent requests --
a single shared connection would serialize every /score call on DB I/O.
"""

import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db"
)

# Lazy, not created at import time. A hard-crash on import would take down
# the whole API (model scoring included) if Postgres just happens to be
# down or not-yet-started when the API boots -- exactly the kind of single
# point of failure the fail-open design elsewhere (model errors, DB write
# errors) is meant to avoid. Every caller already handles get_cursor()
# raising (see api.py's try/except around db.* calls), so raising here on
# a connection failure is safe -- it degrades that one call, not the process.
_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)
    return _pool


@contextmanager
def get_cursor():
    """
    Yields a dict-cursor within a transaction; commits on success,
    rolls back and re-raises on any exception, always returns the
    connection to the pool. Raises if the DB is unreachable -- callers
    (api.py) catch this and fail open rather than crashing the request.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
def insert_transaction(txn_id: str, amount: float, merchant: str, device_id: str, ip: str, timestamp):
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO transactions (id, amount, merchant, device_id, ip, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (txn_id, amount, merchant, device_id, ip, timestamp),
        )


def insert_decision(txn_id: str, fraud_score: float, decision: str, reason: str, true_label: int | None):
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO decisions (txn_id, fraud_score, decision, reason, true_label)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (txn_id, fraud_score, decision, reason, true_label),
        )


def insert_model_metrics(precision: float, recall: float, f1: float, threshold: float,
                          review_threshold: float | None = None, auc_roc: float | None = None):
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_metrics (precision, recall, f1, threshold, review_threshold, auc_roc)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (precision, recall, f1, threshold, review_threshold, auc_roc),
        )


# ---------------------------------------------------------------------------
# Reads (back the dashboard)
# ---------------------------------------------------------------------------
def get_recent_decisions(limit: int = 100):
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT d.txn_id AS transaction_id, t.amount, d.fraud_score, d.decision,
                   d.reason, d.true_label, d.created_at AS logged_at
            FROM decisions d
            JOIN transactions t ON t.id = d.txn_id
            ORDER BY d.created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()


def get_summary():
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE decision = 'block') AS block,
                COUNT(*) FILTER (WHERE decision = 'flag') AS flag,
                COUNT(*) FILTER (WHERE decision = 'allow') AS allow,
                COUNT(*) FILTER (WHERE true_label IS NOT NULL) AS labeled_count,
                COUNT(*) FILTER (WHERE true_label = 1 AND decision IN ('block','flag')) AS tp,
                COUNT(*) FILTER (WHERE true_label = 0 AND decision IN ('block','flag')) AS fp,
                COUNT(*) FILTER (WHERE true_label = 0 AND decision = 'allow') AS tn,
                COUNT(*) FILTER (WHERE true_label = 1 AND decision = 'allow') AS fn
            FROM decisions
            """
        )
        row = cur.fetchone()

    total = row["total"] or 0
    block, flag, allow = row["block"] or 0, row["flag"] or 0, row["allow"] or 0
    tp, fp, tn, fn = row["tp"] or 0, row["fp"] or 0, row["tn"] or 0, row["fn"] or 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    return {
        "total": total,
        "block": block,
        "flag": flag,
        "allow": allow,
        "flagged_rate": round((block + flag) / total, 4) if total else 0.0,
        "labeled_count": row["labeled_count"] or 0,
        "live_precision": round(precision, 4) if precision is not None else None,
        "live_recall": round(recall, 4) if recall is not None else None,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def get_latest_model_metrics():
    with get_cursor() as cur:
        cur.execute("SELECT * FROM model_metrics ORDER BY updated_at DESC LIMIT 1")
        return cur.fetchone()


def health_check() -> bool:
    try:
        with get_cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False