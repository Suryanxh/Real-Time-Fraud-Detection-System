"""
Streaming replay: simulates live transaction events from test data.

Reads transactions from data/processed/test_features.csv (ground truth dataset),
extracts them one-by-one with a configurable delay, and sends them to the
FastAPI /score endpoint over HTTP. This is a demo replay, not a Kafka consumer.
Tracks fraud predictions, false positives, false negatives, and API latency.

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
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_DATA_PATH = "data/processed/test_features.csv"
API_BASE_URL = "http://localhost:8000"
SCORE_ENDPOINT = f"{API_BASE_URL}/score"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Delay between events (seconds), paced slowly enough to follow in the demo.
EVENT_DELAY = 0.5

# Keep the default replay short and balanced for a live demo. Set to None to
# replay the full test set.
LIMIT_ROWS = 50


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
        self.allowed_count = 0
        self.failed_events = 0
        self.latencies = []  # API response times
        self.decision_log = []

    def record_decision(self, decision: dict):
        """Record a single decision and update metrics."""
        self.total_events += 1

        is_fraud_actual = decision["is_fraud_actual"]
        is_fraud_predicted = decision["is_fraud_predicted"]

        # Update confusion matrix
        if is_fraud_predicted and is_fraud_actual:
            self.true_positives += 1
        elif is_fraud_predicted and not is_fraud_actual:
            self.false_positives += 1
        elif not is_fraud_predicted and is_fraud_actual:
            self.false_negatives += 1
        else:
            self.true_negatives += 1

        # Use the API's decision directly so replay metrics follow its thresholds.
        if decision["decision"] == "block":
            self.blocked_count += 1
        elif decision["decision"] == "flag":
            self.review_count += 1
        elif decision["decision"] == "allow":
            self.allowed_count += 1

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
        print(f"  Blocked:                      {self.blocked_count:5d}")
        print(f"  Flagged:                      {self.review_count:5d}")
        print(f"  Allowed:                      {self.allowed_count:5d}")
        print(f"  Failed requests:              {self.failed_events:5d}")

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
    """Load a chronological demo sample with both classes, or the full dataset."""
    df = pd.read_csv(TEST_DATA_PATH)
    if limit is None:
        return df.sort_values("timestamp").reset_index(drop=True)
    if limit <= 0:
        return df.iloc[0:0].copy()

    # Labels select a representative demo mix only; they are sent as separate
    # evaluation metadata and are not model inputs.
    fraud_count = min((limit + 1) // 2, int((df["Class"] == 1).sum()))
    legitimate_count = min(limit - fraud_count, int((df["Class"] == 0).sum()))
    fraud = df[df["Class"] == 1].sample(n=fraud_count, random_state=42)
    legitimate_pool = df[df["Class"] == 0]
    flagged_demo = legitimate_pool.nlargest(1, "Amount")
    other_legitimate = legitimate_pool.drop(index=flagged_demo.index)
    legitimate = pd.concat(
        [
            flagged_demo,
            other_legitimate.sample(
                n=legitimate_count - len(flagged_demo),
                random_state=42,
            ),
        ]
    )
    return (
        pd.concat([fraud, legitimate])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


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
    
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        # Build request payload (matches Transaction schema in api.py)
        payload = {
            "transaction_id": f"txn_{row_number:06d}",
            "amount": float(row["Amount"]),
            "merchant": row.get("merchant", "unknown"),
            "device_id": row.get("device_id", "dev_unknown"),
            "ip": row.get("ip", "ip_unknown"),
            "timestamp": str(row.get("timestamp", datetime.now(timezone.utc).isoformat())),
            "true_label": int(row["Class"]) if "Class" in row.index else None,
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
                api_decision = str(result["decision"]).lower()
                fraud_score = float(result["fraud_score"])

                # Build decision record
                decision = {
                    "transaction_id": payload["transaction_id"],
                    "amount": payload["amount"],
                    "timestamp": payload["timestamp"],
                    "fraud_score": fraud_score,
                    "decision": api_decision,
                    "is_fraud_actual": payload["true_label"] == 1,
                    "is_fraud_predicted": api_decision in ("block", "flag"),
                    "latency_ms": latency_ms,
                }

                metrics.record_decision(decision)
                print(
                    f"[{row_number:3d}/{len(df)}] "
                    f"{decision['transaction_id']} "
                    f"amount=${decision['amount']:.2f} "
                    f"score={fraud_score:.4f} "
                    f"decision={api_decision} "
                    f"actual_fraud={int(decision['is_fraud_actual'])} "
                    f"latency={latency_ms:.1f}ms"
                )
            else:
                metrics.failed_events += 1
                print(f"[FAIL] Request failed (status {resp.status_code}): {resp.text}")

        except requests.RequestException as e:
            metrics.failed_events += 1
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
