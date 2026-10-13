"""
FastAPI fraud-scoring service.

*** READ THIS FIRST -- a real gap between your LLD's API spec and the model ***
Your LLD's /score request has: transaction_id, amount, merchant, device_id,
ip, timestamp. But the model was trained on those PLUS V1-V28 -- the
anonymized PCA components that carry most of the actual fraud signal in the
ULB dataset. Those 28 values represent proprietary, already-derived risk
features (from the card network's own fraud models) that a real production
system would either (a) receive from a card network/processor integration
you don't have in a POC, or (b) replace with your own engineered risk
features built from years of transaction history you don't have either.

For this POC, the request schema is extended with an optional `features`
field carrying V1-V28, sourced from the dataset by stream.py during replay.
This keeps the demo's model performance meaningful. It's flagged here, in
stream.py, and should be flagged in your LLD/demo narrative -- it is a
POC-only simulation detail, not something a production integration would do.

Run:
    uvicorn src.api:app --reload --port 8000
"""

import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.features import BASE_TIMESTAMP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraud-api")

MODEL_PATH = "models/lgbm_fraud_model.pkl"
FEATURE_LIST_PATH = "models/feature_columns.json"
METRICS_PATH = "models/model_metrics.json"
MERCHANT_RISK_PATH = "models/merchant_risk_map.json"
FEATURE_STATS_PATH = "models/feature_stats.json"

VELOCITY_WINDOW_SECONDS = 60 * 60  # 60 minutes, per LLD §4

# ---------------------------------------------------------------------------
# Load artifacts once at startup (not per-request)
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)

with open(FEATURE_LIST_PATH) as f:
    FEATURE_COLUMNS = json.load(f)

with open(METRICS_PATH) as f:
    METRICS = json.load(f)
BLOCK_THRESHOLD = METRICS["block_threshold"]
REVIEW_THRESHOLD = METRICS["review_threshold"]

with open(MERCHANT_RISK_PATH) as f:
    _risk_data = json.load(f)
MERCHANT_RISK_MAP = _risk_data["merchant_risk_map"]
GLOBAL_FRAUD_RATE = _risk_data["global_fraud_rate"]

with open(FEATURE_STATS_PATH) as f:
    FEATURE_STATS = json.load(f)

logger.info(f"Loaded model with {len(FEATURE_COLUMNS)} features")
logger.info(f"Thresholds: block>={BLOCK_THRESHOLD}, flag>={REVIEW_THRESHOLD}")

# ---------------------------------------------------------------------------
# In-memory state for velocity / frequency features
# ---------------------------------------------------------------------------
# POC-only: this is per-process memory, so it resets on restart and won't
# work across multiple API workers/replicas. In production (or once Postgres
# is wired up per LLD §7) this state comes from querying the `transactions`
# table for the trailing window, not from a Python dict living in the
# process -- flagged here as a known POC limitation, not a design decision.
device_history: dict[str, deque] = defaultdict(deque)  # device_id -> deque[(timestamp, amount)]
device_counts: dict[str, int] = defaultdict(int)
ip_counts: dict[str, int] = defaultdict(int)

# In-memory decision log -- stands in for the `decisions` Postgres table
# (LLD §7) until that's wired up. Every decision is logged here for
# explainability per HLD §7 ("Log explainability for every decision").
decision_log: list[dict] = []


# ---------------------------------------------------------------------------
# Request / response schemas (per LLD §3, extended with `features`)
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    transaction_id: str
    amount: float
    merchant: str
    device_id: str
    ip: str
    timestamp: str  # ISO 8601, e.g. "2025-01-01T10:00:00Z"
    # POC-only extension -- see module docstring. V1-V28 from the source
    # dataset; optional so /health and schema docs don't require it, but
    # scoring degrades (see fail-open note below) without it.
    features: Optional[dict[str, float]] = Field(
        default=None, description="V1-V28 anonymized features from the source dataset (POC only)"
    )
    # DEMO/POC INSTRUMENTATION ONLY -- not part of the real LLD contract.
    # Real production scoring never has the true label at request time (fraud
    # is usually confirmed later via chargebacks, days later). This exists
    # purely so stream.py can replay labeled historical data and the
    # dashboard can show a LIVE precision/recall for demo purposes, alongside
    # (not instead of) the real offline test-set metrics in model_metrics.json.
    # It has zero effect on scoring/decision -- it's logged and nothing else.
    true_label: Optional[int] = Field(default=None, description="Demo-only ground truth for live dashboard metrics")


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    decision: str  # "block" | "flag" | "allow"
    reason: str


# ---------------------------------------------------------------------------
# Feature computation for a single live transaction
# ---------------------------------------------------------------------------
def build_feature_vector(txn: Transaction) -> pd.DataFrame:
    ts = pd.to_datetime(txn.timestamp)
    row = {}

    # Raw V1-V28 (POC-only, see module docstring). Missing -> 0.0, which the
    # model was NOT trained to expect -- this is a degraded-input case, not
    # a silently-fine default. See the fail-open handling in the endpoint.
    for i in range(1, 29):
        row[f"V{i}"] = txn.features.get(f"V{i}", 0.0) if txn.features else 0.0

    row["Time"] = (ts - BASE_TIMESTAMP).total_seconds()
    row["Amount"] = txn.amount
    row["amount_log"] = np.log1p(txn.amount)

    # Time-of-day features (mirrors features.py add_time_features)
    row["hour_of_day"] = ts.hour
    row["day_of_week"] = ts.dayofweek
    row["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    row["is_night"] = int(0 <= ts.hour <= 5)

    # Velocity: purge entries older than 60 min, then count/sum INCLUDING
    # this transaction -- same inclusive-window semantics as training.
    hist = device_history[txn.device_id]
    cutoff = ts - pd.Timedelta(seconds=VELOCITY_WINDOW_SECONDS)
    while hist and hist[0][0] < cutoff:
        hist.popleft()
    velocity_count = len(hist) + 1
    velocity_sum = sum(a for _, a in hist) + txn.amount
    row["device_id_velocity_count_60min"] = velocity_count
    row["device_id_velocity_sum_60min"] = velocity_sum

    # Frequency: cumulative count including this transaction
    row["device_id_frequency"] = device_counts[txn.device_id] + 1
    row["ip_frequency"] = ip_counts[txn.ip] + 1

    # Merchant risk: fixed map from training, fallback to global rate for
    # unseen merchants (see features.py add_merchant_risk leakage warning)
    row["merchant_risk_score"] = MERCHANT_RISK_MAP.get(txn.merchant, GLOBAL_FRAUD_RATE)

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def update_state(txn: Transaction):
    """Called only after a successful score, so state doesn't drift on errors."""
    ts = pd.to_datetime(txn.timestamp)
    device_history[txn.device_id].append((ts, txn.amount))
    device_counts[txn.device_id] += 1
    ip_counts[txn.ip] += 1


def generate_reason(feature_row: pd.DataFrame) -> str:
    """
    Plain-English reason from the interpretable features' z-scores
    (V1-V28 are excluded -- anonymized components have no human meaning).
    Picks the top 2 features by |z-score| and maps them to short tags,
    matching the style of your LLD's example: "high_amount_velocity_anomaly".
    """
    tag_map = {
        "amount_log": "high_amount",
        "device_id_velocity_count_60min": "velocity_anomaly",
        "device_id_velocity_sum_60min": "velocity_anomaly",
        "device_id_frequency": "device_activity_anomaly",
        "ip_frequency": "ip_activity_anomaly",
        "merchant_risk_score": "high_risk_merchant",
    }
    z_scores = {}
    for col, stats in FEATURE_STATS.items():
        val = feature_row.iloc[0][col]
        z_scores[col] = abs((val - stats["mean"]) / (stats["std"] or 1.0))

    top = sorted(z_scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
    tags = list(dict.fromkeys(tag_map[col] for col, _ in top))  # dedupe, preserve order
    return "_".join(tags) if tags else "model_score"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Fraud Detection Scoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "db_connected": False,
        "n_features": len(FEATURE_COLUMNS),
        "block_threshold": BLOCK_THRESHOLD,
        "review_threshold": REVIEW_THRESHOLD,
        # in-memory count still reported for visibility, but is no longer the
        # source of truth once the DB is up -- see /decisions/summary
        "total_scored_in_memory": len(decision_log),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):
    # Fail-open per HLD §7: "Fail-open fallback if the model errors" -- a
    # scoring outage should never silently block legitimate transactions.
    # We still return HTTP 200 with an explicit "allow" + reason, because a
    # 500 error would leave the caller with no decision at all.
    try:
        feature_row = build_feature_vector(txn)
        fraud_score = float(model.predict_proba(feature_row)[0, 1])

        if fraud_score >= BLOCK_THRESHOLD:
            decision = "block"
        elif fraud_score >= REVIEW_THRESHOLD:
            decision = "flag"
        else:
            decision = "allow"

        reason = generate_reason(feature_row) if decision != "allow" else "low_risk"

        update_state(txn)

    except Exception:
        logger.exception(f"Scoring failed for transaction_id={txn.transaction_id}, failing open")
        fraud_score = 0.0
        decision = "allow"
        reason = "model_error_fail_open"

    result = ScoreResponse(
        transaction_id=txn.transaction_id,
        fraud_score=round(fraud_score, 4),
        decision=decision,
        reason=reason,
    )

    # Explainability logging per HLD §7 -- every decision is logged.
    # Kept in memory too (not just DB) so the API/dashboard degrade
    # gracefully instead of going dark if Postgres is briefly unavailable --
    # this is a DIFFERENT fail-open than the model's: a DB outage should
    # never block scoring or crash the request either.
    decision_log.append({
        **result.model_dump(),
        "amount": txn.amount,
        "merchant": getattr(txn, "merchant", "unknown"),
        "device_id": getattr(txn, "device_id", "unknown"),
        "ip": getattr(txn, "ip", "unknown"),
        "true_label": txn.true_label,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    })

    logger.info(f"txn={txn.transaction_id} score={result.fraud_score} decision={decision} reason={reason}")

    return result


def _in_memory_summary():
    """Fallback used only if Postgres is unreachable -- see decisions_summary()."""
    total = len(decision_log)
    block = sum(1 for d in decision_log if d["decision"] == "block")
    flag = sum(1 for d in decision_log if d["decision"] == "flag")
    allow = total - block - flag

    labeled = [d for d in decision_log if d.get("true_label") is not None]
    tp = sum(1 for d in labeled if d["true_label"] == 1 and d["decision"] in ("block", "flag"))
    fp = sum(1 for d in labeled if d["true_label"] == 0 and d["decision"] in ("block", "flag"))
    fn = sum(1 for d in labeled if d["true_label"] == 1 and d["decision"] == "allow")
    tn = sum(1 for d in labeled if d["true_label"] == 0 and d["decision"] == "allow")
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    return {
        "total": total, "block": block, "flag": flag, "allow": allow,
        "flagged_rate": round((block + flag) / total, 4) if total else 0.0,
        "labeled_count": len(labeled),
        "live_precision": round(precision, 4) if precision is not None else None,
        "live_recall": round(recall, 4) if recall is not None else None,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


@app.get("/decisions/summary")
def decisions_summary():
    """
    POC implementation using in-memory decisions.
    PostgreSQL can be added later without changing the API contract.
    """
    return _in_memory_summary()

@app.get("/decisions/recent")
def recent_decisions(limit: int = 20):
    """
    POC implementation using the in-memory decision log.
    """
    limit = max(1, min(limit, 200))
    return decision_log[-limit:]