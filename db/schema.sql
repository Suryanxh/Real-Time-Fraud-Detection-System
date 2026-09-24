-- Fraud detection POC — PostgreSQL schema
-- Matches LLD §7's table/column spec exactly for the required fields;
-- additions beyond that spec (types, keys, indexes, a couple of extra
-- columns) are called out inline so it's clear what's LLD-mandated vs
-- pragmatic engineering.

-- ---------------------------------------------------------------------------
-- transactions: one row per incoming transaction event (LLD: id, amount,
-- merchant, device_id, ip, timestamp)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transactions (
    id          TEXT PRIMARY KEY,       -- transaction_id from the request payload
                                         -- TEXT, not SERIAL: the ID is generated
                                         -- upstream (stream.py / the real event
                                         -- source), not by the database.
    amount      NUMERIC(12, 2) NOT NULL,
    merchant    TEXT NOT NULL,
    device_id   TEXT NOT NULL,
    ip          TEXT NOT NULL,
    timestamp   TIMESTAMPTZ NOT NULL,   -- event time (from the transaction), not insert time
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now()   -- addition: audit/debug column
);

-- Velocity/frequency features are computed per device_id and ip over a
-- trailing time window -- these indexes are what make that fast once the
-- API queries this table instead of its in-memory dict (see api.py notes).
CREATE INDEX IF NOT EXISTS idx_transactions_device_id_timestamp ON transactions (device_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_ip_timestamp ON transactions (ip, timestamp);

-- ---------------------------------------------------------------------------
-- decisions: one row per scoring decision (LLD: txn_id, fraud_score,
-- decision, reason, created_at)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS decisions (
    id          SERIAL PRIMARY KEY,     -- addition: surrogate key (a transaction
                                         -- could in principle be re-scored)
    txn_id      TEXT NOT NULL REFERENCES transactions(id),
    fraud_score NUMERIC(6, 4) NOT NULL,
    decision    TEXT NOT NULL CHECK (decision IN ('block', 'flag', 'allow')),
    reason      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- addition: demo-only ground truth, mirrors api.py's Transaction.true_label.
    -- Nullable because real production traffic never has this at scoring time.
    true_label  INTEGER CHECK (true_label IN (0, 1))
);

-- Powers the dashboard's "recent decisions" feed and live summary aggregation.
CREATE INDEX IF NOT EXISTS idx_decisions_created_at ON decisions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_decision ON decisions (decision);

-- ---------------------------------------------------------------------------
-- model_metrics: one row per trained model version (LLD: precision, recall,
-- f1, threshold, updated_at)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_metrics (
    id                SERIAL PRIMARY KEY,   -- addition: lets you keep history across retrains
    precision         NUMERIC(5, 4) NOT NULL,
    recall            NUMERIC(5, 4) NOT NULL,
    f1                NUMERIC(5, 4) NOT NULL,
    threshold         NUMERIC(5, 4) NOT NULL,  -- maps to train.py's block_threshold
    -- addition: train.py's decision logic actually needs two thresholds
    -- (block vs flag/review) per LLD §6 -- added here so the DB can be the
    -- single source of truth for both, not just the one the LLD table listed.
    review_threshold  NUMERIC(5, 4),
    auc_roc           NUMERIC(5, 4),          -- addition: see train.py's evaluation notes
                                               -- on why AUC-ROC alone is insufficient here
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_model_metrics_updated_at ON model_metrics (updated_at DESC);