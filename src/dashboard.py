"""
Live Streamlit dashboard for the fraud detection POC.

Polls the FastAPI service's /health, /decisions/summary, and /decisions/recent
endpoints and re-renders on an interval -- no shared DB access needed, no
duplicated business logic. All numbers you see here are computed by api.py,
this file only displays them (per LLD §1: Dashboard's responsibility is
"Live feed, alerts, metrics", not scoring or aggregation).

Two kinds of metrics are shown, deliberately kept visually separate:
  - OFFLINE metrics (from model_metrics.json via the API): precision/recall/F1
    /AUC computed once on the held-out test set during training. This is the
    real, trustworthy measure of model quality.
  - LIVE metrics (from /decisions/summary): precision/recall computed only
    over transactions replayed with a demo-only true_label (see api.py /
    stream.py). This is a demo convenience, not a production capability --
    real fraud labels arrive later via chargebacks, not at scoring time.

Run (with the API already running separately):
    streamlit run src/dashboard.py
"""

import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Fraud Detection - Live Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar controls (read once per script run; Streamlit preserves widget
# state across the st.rerun()-driven refresh loop below)
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
api_base = st.sidebar.text_input("API base URL", "http://localhost:8000")
refresh_seconds = st.sidebar.slider("Refresh interval (s)", 1, 10, 2)
row_limit = st.sidebar.slider("Transactions shown in feed", 10, 200, 50)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
decision_filter = st.sidebar.multiselect(
    "Filter feed by decision", ["block", "flag", "allow"], default=["block", "flag", "allow"]
)

st.title("🛡️ Real-Time Fraud Detection — Live Dashboard")


# ---------------------------------------------------------------------------
# Data fetching -- isolated so a down API degrades gracefully, not a crash
# ---------------------------------------------------------------------------
def safe_get(path: str, params: dict | None = None):
    try:
        resp = requests.get(f"{api_base}{path}", params=params, timeout=3)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        return None, str(e)


health, health_err = safe_get("/health")
summary, summary_err = safe_get("/decisions/summary")
recent, recent_err = safe_get("/decisions/recent", params={"limit": row_limit})

if health_err:
    st.error(
        f"Can't reach the API at {api_base} -- is it running? "
        f"(`uvicorn src.api:app --port 8000`)\n\nDetail: {health_err}"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Top row: model status + offline (real) evaluation metrics
# ---------------------------------------------------------------------------
st.subheader("Model status")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model loaded", "Yes" if health["model_loaded"] else "No")
c2.metric("Block threshold", f"{health['block_threshold']:.3f}")
c3.metric("Review/flag threshold", f"{health['review_threshold']:.3f}")
if summary and "total" in summary:
    c4.metric("Total scored (lifetime)", summary["total"])
else:
    c4.metric("Total scored (lifetime)", "N/A")

st.caption(
    "Offline test-set metrics below are from training (model_metrics.json) -- "
    "this is the trustworthy measure of model quality, not the live numbers further down."
)

# ---------------------------------------------------------------------------
# Live operational metrics (from /decisions/summary)
# ---------------------------------------------------------------------------
st.subheader("Live stream metrics")

if summary_err or summary is None:
    st.warning("Could not load live summary metrics.")
else:
    total = summary["total"]
    fraud_rate = summary["flagged_rate"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions scored", total)
    m2.metric("Flagged rate (block+flag)", f"{fraud_rate * 100:.2f}%")
    m3.metric("Blocked", summary["block"])
    m4.metric("Flagged for review", summary["flag"])

    if summary["labeled_count"] > 0:
        st.caption(
            f"Live precision/recall below are computed from {summary['labeled_count']} "
            f"demo-labeled transactions (sent by stream.py with a ground-truth true_label). "
            f"This is a POC convenience -- real production scoring never has the true label "
            f"at request time, so this metric can't exist outside a labeled replay demo."
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Live precision", f"{summary['live_precision']*100:.1f}%" if summary["live_precision"] is not None else "N/A")
        p2.metric("Live recall", f"{summary['live_recall']*100:.1f}%" if summary["live_recall"] is not None else "N/A")
        p3.metric("Confusion (TP/FP/TN/FN)", f"{summary['tp']}/{summary['fp']}/{summary['tn']}/{summary['fn']}")
    else:
        st.info("No labeled transactions yet -- run stream.py to see live precision/recall.")

st.divider()

# ---------------------------------------------------------------------------
# Recent alerts (block/flag only) -- the "highlight fraud" requirement
# ---------------------------------------------------------------------------
st.subheader("🚨 Recent fraud alerts")

if recent_err or not recent:
    st.info("No transactions scored yet.")
else:
    alerts = [d for d in recent if d["decision"] in ("block", "flag")]
    if not alerts:
        st.success("No fraud alerts in the current feed window.")
    else:
        alert_df = pd.DataFrame(alerts)[["transaction_id", "amount", "fraud_score", "decision", "reason", "logged_at"]]
        alert_df = alert_df.sort_values("logged_at", ascending=False)

        def highlight_decision(row):
            color = "#ffcccc" if row["decision"] == "block" else "#fff3cd"
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            alert_df.style.apply(highlight_decision, axis=1),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# Live transaction feed (all decisions, filterable)
# ---------------------------------------------------------------------------
st.subheader("Live transaction feed")

if recent_err or not recent:
    st.info("No transactions scored yet -- start stream.py to see live data here.")
else:
    feed_df = pd.DataFrame(recent)
    feed_df = feed_df[feed_df["decision"].isin(decision_filter)]
    feed_df = feed_df.sort_values("logged_at", ascending=False)

    display_cols = ["transaction_id", "amount", "fraud_score", "decision", "reason", "true_label", "logged_at"]
    display_cols = [c for c in display_cols if c in feed_df.columns]

    st.dataframe(feed_df[display_cols], use_container_width=True, hide_index=True, height=400)

    # Simple decision-mix chart -- quick visual read on stream composition
    if not feed_df.empty:
        counts = feed_df["decision"].value_counts()
        st.bar_chart(counts)

st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ---------------------------------------------------------------------------
# Auto-refresh: full script rerun on a timer. Simple and dependency-free --
# Streamlit preserves sidebar widget state across reruns automatically, so
# this doesn't reset your settings each cycle.
# ---------------------------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()