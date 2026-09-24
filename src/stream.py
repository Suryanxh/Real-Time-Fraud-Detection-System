"""
Streaming replay: simulates live transaction events from test data.

Reads transactions from data/processed/test_features.csv (ground truth dataset),
extracts them one-by-one with a configurable delay, and sends them to the
FastAPI /score endpoint. Tracks fraud predictions, false positives, false
negatives, and updates running metrics.

Useful for:
- Live testing the API with realistic transaction patterns
- Observing decision-threshold behavior in real-time
- Monitoring latency and API reliability
- Building a decision audit trail (for LLD §7)

Run:
    python src/stream.py

(Make sure the API is running in another terminal:)
    python -m uvicorn src.api:app --reload --port 8000
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from collections import defaultdict


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_DATA_PATH = "data/processed/test_features.csv"
API_BASE_URL = "http://localhost:8000"
SCORE_ENDPOINT = f"{API_BASE_URL}/score"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Delay between events (seconds). Set to 0 for instant replay, or 0.5-1.0
# for realistic pacing.
EVENT_DELAY = 0.01  # 100 ms between transactions

# Subset size (for quick testing). Use 0 or None to replay entire dataset.
# Set to 100 for a quick 10-second test, 1000 for ~100 seconds, etc.
LIMIT_ROWS = None  # None = full dataset; int = limit

# Decision thresholds (should match API's model_metrics.json)
BLOCK_THRESHOLD = 0.9995
REVIEW_THRESHOLD = 0.7995


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------
class StreamMetrics:
    """Track running fraud detection metrics."""

    def __init__(self):
        self.total_events = 0
        self.true_positives = 0  # fraud predicted, actually fraud
        self.false_positives = 0  # fraud predicted, actually legit
        self.true_negatives = 0  # legit predicted, actually legit
        self.false_negatives = 0  # legit predicted, actually fraud
        self.blocked_count = 0  # high confidence fraud
        self.review_count = 0  # medium confidence fraud
        self.latencies = []  # API response times
        self.decision_log = []

    def record_decision(self, decision: dict):
        """Record a single decision and update metrics."""
        self.total_events += 1
        
        is_fraud_actual = decision["is_fraud_actual"]
        is_fraud_predicted = decision["is_fraud_predicted"]
        fraud_probability = decision["fraud_probability"]
        
        # Update confusion matrix
        if is_fraud_predicted and is_fraud_actual:
            self.true_positives += 1
        elif is_fraud_predicted and not is_fraud_actual:
            self.false_positives += 1
        elif not is_fraud_predicted and is_fraud_actual:
            self.false_negatives += 1
        else:
            self.true_negatives += 1
        
        # Update action counts
        if fraud_probability >= BLOCK_THRESHOLD:
            self.blocked_count += 1
        elif fraud_probability >= REVIEW_THRESHOLD:
            self.review_count += 1
        
        # Track latency
        if "latency_ms" in decision:
            self.latencies.append(decision["latency_ms"])
        
        self.decision_log.append(decision)

    def print_summary(self):
        """Print running metrics every N events or at end."""
        print("\n" + "=" * 70)
        print(f"STREAM METRICS (after {self.total_events} events)")
        print("=" * 70)
        
        # Confusion matrix
        print(f"True Positives:  {self.true_positives:5d}")
        print(f"False Positives: {self.false_positives:5d}")
        print(f"True Negatives:  {self.true_negatives:5d}")
        print(f"False Negatives: {self.false_negatives:5d}")
        
        # Detection rates
        if self.true_positives + self.false_negatives > 0:
            recall = self.true_positives / (self.true_positives + self.false_negatives)
            print(f"\nRecall (sensitivity):         {recall:.4f}")
        
        if self.true_positives + self.false_positives > 0:
            precision = self.true_positives / (self.true_positives + self.false_positives)
            print(f"Precision:                    {precision:.4f}")
        
        # Action distribution
        print(f"\nACTIONS:")
        print(f"  Blocked (p>={BLOCK_THRESHOLD}):       {self.blocked_count:5d}")
        print(f"  Flagged (p>={REVIEW_THRESHOLD}):        {self.review_count:5d}")
        print(f"  Allowed:                      {self.total_events - self.blocked_count - self.review_count:5d}")
        
        # Latency stats
        if self.latencies:
            avg_latency = sum(self.latencies) / len(self.latencies)
            max_latency = max(self.latencies)
            min_latency = min(self.latencies)
            print(f"\nAPI LATENCY (ms):")
            print(f"  Min:     {min_latency:.1f}")
            print(f"  Avg:     {avg_latency:.1f}")
            print(f"  Max:     {max_latency:.1f}")
        
        print("=" * 70)

    def save_decision_log(self, filepath: str = "models/stream_decisions.json"):
        """Save decision log for audit trail."""
        with open(filepath, "w") as f:
            json.dump(self.decision_log, f, indent=2, default=str)
        print(f"\nDecision log saved to {filepath}")


# ---------------------------------------------------------------------------
# Stream replay
# ---------------------------------------------------------------------------
def load_test_data(limit: int | None = None) -> pd.DataFrame:
    """Load test features CSV and optionally limit rows."""
    df = pd.read_csv(TEST_DATA_PATH)
    if limit:
        df = df.head(limit)
    return df


def wait_for_api(max_retries: int = 10, delay: float = 1.0):
    """Poll the API health endpoint until it's ready."""
    print(f"Waiting for API at {API_BASE_URL}...")
    for attempt in range(max_retries):
        try:
            resp = requests.get(HEALTH_ENDPOINT, timeout=2)
            if resp.status_code == 200:
                print(f"[OK] API is ready\n")
                return
        except requests.ConnectionError:
            pass
        
        if attempt < max_retries - 1:
            print(f"  Retry {attempt + 1}/{max_retries}...")
            time.sleep(delay)
    
    raise RuntimeError(
        f"API at {API_BASE_URL} did not become ready after {max_retries} attempts. "
        "Make sure to run: python -m uvicorn src.api:app --reload --port 8000"
    )


def stream_transactions(df: pd.DataFrame, metrics: StreamMetrics):
    """Replay transactions row-by-row to the API."""
    print(f"Replaying {len(df)} transactions...")
    print(f"Event delay: {EVENT_DELAY} seconds")
    print(f"Estimated duration: {len(df) * EVENT_DELAY / 60:.1f} minutes\n")
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        # Build request payload (matches Transaction schema in api.py)
        payload = {
            "transaction_id": f"txn_{idx:06d}",
            "amount": float(row["Amount"]),
            "merchant": row.get("merchant", "unknown"),
            "device_id": row.get("device_id", "dev_unknown"),
            "ip": row.get("ip", "ip_unknown"),
            "timestamp": row.get("timestamp", datetime.now(timezone.utc).isoformat()),
            # Include V1-V28 (POC-only extension, per api.py docstring)
            "features": {
                col: float(row[col])
                for col in [f"V{i}" for i in range(1, 29)]
                if col in row.index
            },
        }
        
        # Call API
        request_start = time.time()
        try:
            resp = requests.post(SCORE_ENDPOINT, json=payload, timeout=10)
            latency_ms = (time.time() - request_start) * 1000
            
            if resp.status_code == 200:
                result = resp.json()
                
                # Build decision record
                decision = {
                    "transaction_id": payload["transaction_id"],
                    "amount": payload["amount"],
                    "timestamp": payload["timestamp"],
                    "fraud_probability": result.get("fraud_probability", 0),
                    "decision": result.get("decision", "unknown"),
                    "is_fraud_actual": int(row.get("Class", 0)) == 1,
                    "is_fraud_predicted": result.get("decision") in ["BLOCK", "REVIEW"],
                    "latency_ms": latency_ms,
                }
                
                metrics.record_decision(decision)
                
                # Print running updates every 100 transactions
                if (idx + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    print(
                        f"[{idx + 1:6d}] "
                        f"{decision['decision']:6s} "
                        f"(p={decision['fraud_probability']:.4f}) "
                        f"actual={decision['is_fraud_actual']} "
                        f"{rate:.1f} tx/sec"
                    )
            else:
                print(f"[FAIL] Request failed (status {resp.status_code}): {resp.text}")
        
        except requests.RequestException as e:
            print(f"[ERROR] Request error: {e}")
        
        # Sleep before next event
        time.sleep(EVENT_DELAY)
    
    elapsed = time.time() - start_time
    print(f"\nReplay complete in {elapsed:.1f} seconds")


def main():
    """Main entry point."""
    print("=" * 70)
    print("FRAUD DETECTION SYSTEM - STREAMING REPLAY")
    print("=" * 70)
    print()
    
    # Wait for API
    wait_for_api()
    
    # Load data
    df = load_test_data(limit=LIMIT_ROWS)
    print(f"Loaded {len(df)} transactions from {TEST_DATA_PATH}\n")
    
    # Stream
    metrics = StreamMetrics()
    try:
        stream_transactions(df, metrics)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    # Summary
    metrics.print_summary()
    
    # Save decision log
    metrics.save_decision_log()


if __name__ == "__main__":
    main()
