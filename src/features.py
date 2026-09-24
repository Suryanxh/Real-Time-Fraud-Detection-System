"""
Feature engineering for the fraud detection POC.

IMPORTANT DATA GAP: the ULB Credit Card Fraud dataset (V1-V28, Time, Amount, Class)
has NO merchant, device_id, or ip columns -- those were stripped out along with
everything else during the dataset's PCA anonymization. But your LLD's /score API
spec expects them, and your feature list (device/IP frequency, merchant risk)
needs them to mean anything.

So this module does two jobs:
  1. synthesize_entity_fields()  -- generates plausible merchant/device_id/ip
     values so the rest of the pipeline (and your API payloads) work end-to-end
     for the demo. This is a POC-only step -- in production these fields come
     from the real transaction event, you would NOT synthesize them.
  2. engineer_features()          -- computes the actual model features from
     amount, time, and those (real or synthetic) identity fields.

Usage:
    from src.features import synthesize_entity_fields, engineer_features

    df = synthesize_entity_fields(df, seed=42)
    train_feats, merchant_risk_map = engineer_features(df_train, is_train=True)
    test_feats, _ = engineer_features(df_test, is_train=False, merchant_risk_map=merchant_risk_map)
"""

import numpy as np
import pandas as pd

BASE_TIMESTAMP = pd.Timestamp("2024-01-01 00:00:00")


# ---------------------------------------------------------------------------
# 1. Synthesize merchant / device_id / ip (POC-only -- see module docstring)
# ---------------------------------------------------------------------------
def synthesize_entity_fields(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Adds synthetic 'merchant', 'device_id', 'ip', and 'timestamp' columns.

    Design choices, so the synthetic data is at least *directionally* realistic:
      - A small pool of devices/IPs is reused across many transactions (real
        fraud rings reuse infrastructure) rather than one unique ID per row.
      - Fraud rows are biased toward a smaller sub-pool of devices/IPs/merchants,
        so velocity and frequency features actually have signal to learn from
        instead of being pure noise.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    is_fraud = df["Class"].values.astype(bool)

    # Real timestamp from the 'Time' column (seconds since first transaction)
    df["timestamp"] = BASE_TIMESTAMP + pd.to_timedelta(df["Time"], unit="s")

    # Device/IP pools: fraud transactions draw from a much smaller pool,
    # simulating reused devices/IPs in fraud rings.
    normal_devices = [f"dev_{i:05d}" for i in range(3000)]
    fraud_devices = [f"dev_{i:05d}" for i in range(3000, 3050)]  # small reused pool
    normal_ips = [f"ip_{i:05d}" for i in range(2500)]
    fraud_ips = [f"ip_{i:05d}" for i in range(2500, 2540)]

    device_id = np.where(
        is_fraud,
        rng.choice(fraud_devices, size=n),
        rng.choice(normal_devices, size=n),
    )
    ip = np.where(
        is_fraud,
        rng.choice(fraud_ips, size=n),
        rng.choice(normal_ips, size=n),
    )

    # Merchant categories with different baseline risk (mirrors real-world
    # patterns: electronics/gift cards see more fraud than groceries).
    merchants = ["grocery", "electronics", "travel", "gift_cards", "clothing", "utilities"]
    merchant_weights_normal = [0.30, 0.15, 0.10, 0.05, 0.25, 0.15]
    merchant_weights_fraud = [0.05, 0.35, 0.15, 0.30, 0.10, 0.05]

    merchant = np.where(
        is_fraud,
        rng.choice(merchants, size=n, p=merchant_weights_fraud),
        rng.choice(merchants, size=n, p=merchant_weights_normal),
    )

    df["device_id"] = device_id
    df["ip"] = ip
    df["merchant"] = merchant
    return df


# ---------------------------------------------------------------------------
# 2. Time-of-day features
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hour-of-day and day-of-week, plus cyclical (sin/cos) encodings.

    Why cyclical encoding: hour 23 and hour 0 are one hour apart, but a raw
    integer feature makes them look 23 apart. sin/cos encoding preserves that
    adjacency, which helps tree splits near midnight (relevant for fraud,
    which often clusters in late-night hours).
    """
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)
    return df


# ---------------------------------------------------------------------------
# 3. Velocity features -- count/sum of transactions in the last 60 minutes
# ---------------------------------------------------------------------------
def add_velocity_features(df: pd.DataFrame, entity_col: str = "device_id") -> pd.DataFrame:
    """
    For each transaction, count and sum of that entity's (device_id by default)
    transactions in the trailing 60-minute window, INCLUDING the current one.

    Causality: pandas' time-based rolling window for a row at time t only ever
    looks at [t - 60min, t] -- it physically cannot see future rows. That's
    what makes this safe to compute the same way on train and test/live data,
    unlike the merchant risk score below.
    """
    df = df.sort_values([entity_col, "timestamp"]).copy()
    indexed = df.set_index("timestamp")

    grouped = indexed.groupby(entity_col)["Amount"]
    velocity_count = grouped.rolling("60min").count().reset_index(level=0, drop=True)
    velocity_sum = grouped.rolling("60min").sum().reset_index(level=0, drop=True)

    df[f"{entity_col}_velocity_count_60min"] = velocity_count.values
    df[f"{entity_col}_velocity_sum_60min"] = velocity_sum.values

    # Restore original row order (rolling required sorting by entity+time)
    return df.sort_index()


# ---------------------------------------------------------------------------
# 4. Device / IP frequency -- how often has this device/IP been seen so far
# ---------------------------------------------------------------------------
def add_entity_frequency(df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """
    Cumulative count of prior (+ current) transactions from this device/IP,
    up to and including the current transaction's timestamp.

    Why cumulative and not "total count across the whole dataset": a total
    count uses future information relative to any given transaction (leakage).
    A brand-new device on its first-ever transaction should show frequency=1,
    not its eventual lifetime total -- that's the realistic signal a live
    system would actually have at scoring time.
    """
    df = df.sort_values([entity_col, "timestamp"]).copy()
    df[f"{entity_col}_frequency"] = df.groupby(entity_col).cumcount() + 1
    return df.sort_index()


# ---------------------------------------------------------------------------
# 5. Merchant risk score -- historical fraud rate per merchant category
# ---------------------------------------------------------------------------
def add_merchant_risk(
    df: pd.DataFrame,
    is_train: bool,
    merchant_risk_map: dict | None = None,
    global_fraud_rate: float | None = None,
):
    """
    Target-encodes 'merchant' as its historical fraud rate.

    LEAKAGE WARNING -- this is the one feature in this module where causality
    isn't automatic:
      - On TRAIN: the risk map is *fit* here, using only training labels.
      - On TEST/LIVE: you must pass in the map fit on train (never recompute
        it from test data, and never let the model see test labels here).
      - Unseen merchants at inference time fall back to the global fraud rate
        rather than crashing or silently becoming NaN.
    """
    df = df.copy()

    if is_train:
        merchant_risk_map = df.groupby("merchant")["Class"].mean().to_dict()
        global_fraud_rate = df["Class"].mean()
    else:
        if merchant_risk_map is None or global_fraud_rate is None:
            raise ValueError(
                "merchant_risk_map and global_fraud_rate (from training) "
                "are required when is_train=False"
            )

    df["merchant_risk_score"] = (
        df["merchant"].map(merchant_risk_map).fillna(global_fraud_rate)
    )
    return df, merchant_risk_map, global_fraud_rate


# ---------------------------------------------------------------------------
# 6. Amount transform
# ---------------------------------------------------------------------------
def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """log1p transform to tame Amount's right skew (see EDA step)."""
    df = df.copy()
    df["amount_log"] = np.log1p(df["Amount"])
    return df


# ---------------------------------------------------------------------------
# 7. Orchestrator -- run the full pipeline
# ---------------------------------------------------------------------------
def engineer_features(
    df: pd.DataFrame,
    is_train: bool,
    merchant_risk_map: dict | None = None,
    global_fraud_rate: float | None = None,
):
    """
    Runs the full feature pipeline in the correct order and returns
    (engineered_df, merchant_risk_map, global_fraud_rate).

    Pass the returned merchant_risk_map/global_fraud_rate from the train call
    into the test/live call -- see module docstring for usage.
    """
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_velocity_features(df, entity_col="device_id")
    df = add_entity_frequency(df, entity_col="device_id")
    df = add_entity_frequency(df, entity_col="ip")
    df, merchant_risk_map, global_fraud_rate = add_merchant_risk(
        df, is_train=is_train,
        merchant_risk_map=merchant_risk_map,
        global_fraud_rate=global_fraud_rate,
    )
    return df, merchant_risk_map, global_fraud_rate


if __name__ == "__main__":
    # Quick smoke test on the processed splits from eda.py
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    train_df = X_train.copy()
    train_df["Class"] = y_train.values
    test_df = X_test.copy()
    test_df["Class"] = y_test.values

    train_df = synthesize_entity_fields(train_df, seed=42)
    test_df = synthesize_entity_fields(test_df, seed=43)  # different seed = different pool draw order

    train_feats, risk_map, global_rate = engineer_features(train_df, is_train=True)
    test_feats, _, _ = engineer_features(
        test_df, is_train=False, merchant_risk_map=risk_map, global_fraud_rate=global_rate
    )

    print("Train features shape:", train_feats.shape)
    print("Test features shape:", test_feats.shape)
    print("\nNew feature columns:")
    new_cols = [c for c in train_feats.columns if c not in X_train.columns]
    print(new_cols)

    train_feats.to_csv("data/processed/train_features.csv", index=False)
    test_feats.to_csv("data/processed/test_features.csv", index=False)
    print("\nSaved to data/processed/train_features.csv and test_features.csv")

    # ---- Persist artifacts api.py needs at live-scoring time ----------------
    # merchant_risk_map/global_fraud_rate were fit on TRAIN ONLY (see
    # add_merchant_risk's leakage warning) -- api.py must reuse these exact
    # values for every live transaction, never recompute them.
    import json
    import os

    os.makedirs("models", exist_ok=True)
    with open("models/merchant_risk_map.json", "w") as f:
        json.dump({"merchant_risk_map": risk_map, "global_fraud_rate": global_rate}, f, indent=2)

    # Mean/std of the human-interpretable engineered features, from TRAIN only.
    # api.py uses these to z-score an incoming transaction's features and
    # generate a plain-English "reason" string (e.g. "high_amount_velocity_anomaly").
    # V1-V28 are deliberately excluded -- they're anonymized PCA components with
    # no real-world meaning, so they can't produce a human-readable reason.
    interpretable_cols = [
        "amount_log",
        "device_id_velocity_count_60min",
        "device_id_velocity_sum_60min",
        "device_id_frequency",
        "ip_frequency",
        "merchant_risk_score",
    ]
    feature_stats = {
        col: {"mean": float(train_feats[col].mean()), "std": float(train_feats[col].std() or 1.0)}
        for col in interpretable_cols
    }
    with open("models/feature_stats.json", "w") as f:
        json.dump(feature_stats, f, indent=2)

    print("Saved models/merchant_risk_map.json and models/feature_stats.json (needed by api.py)")