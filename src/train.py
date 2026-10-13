"""
Train a LightGBM binary classifier for fraud detection.

Handles class imbalance via scale_pos_weight (recommended default -- see
explanation below). Evaluates with precision, recall, F1, and AUC-ROC, since
accuracy is meaningless on a ~578:1 imbalanced dataset (per LLD §5).

Run from your project root (after eda.py and features.py have produced
data/processed/train_features.csv and test_features.csv):
    python src/train.py
"""

import json
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

if __package__:
    from .evaluation import select_threshold
else:
    from evaluation import select_threshold

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_PATH = "data/processed/train_features.csv"
TEST_PATH = "data/processed/test_features.csv"
MODEL_PATH = "models/lgbm_fraud_model.pkl"
FEATURE_LIST_PATH = "models/feature_columns.json"
METRICS_PATH = "models/model_metrics.json"

# Columns that are identifiers/raw strings/leakage risks, not model inputs.
# (merchant/device_id/ip are already encoded into merchant_risk_score and the
# *_frequency / *_velocity_* features by features.py -- feeding the raw
# strings too would just be noise LightGBM can't use directly.)
DROP_COLS = ["Class", "timestamp", "device_id", "ip", "merchant"]


# ---------------------------------------------------------------------------
# 1. Load engineered features
# ---------------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

X = train_df[feature_cols]
y = train_df["Class"]
X_test = test_df[feature_cols]
y_test = test_df["Class"]

print(f"Training features ({len(feature_cols)}): {feature_cols}")

# ---------------------------------------------------------------------------
# 2. Carve out a validation set from train (for early stopping)
# ---------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Handle class imbalance
# ---------------------------------------------------------------------------
# scale_pos_weight tells LightGBM to weight each positive (fraud) example's
# gradient more heavily, roughly proportional to the imbalance ratio. This is
# the recommended default over SMOTE for a POC:
#   - No synthetic rows are created, so nothing to accidentally leak across
#     train/val/test if you re-split later.
#   - LightGBM computes it natively -- no extra dependency (imbalanced-learn).
#   - It tends to perform comparably to SMOTE on tabular fraud data in
#     practice, with far less complexity.
# SMOTE is a reasonable alternative if you want to demonstrate it explicitly
# (it's mentioned in your LLD) -- see the commented block below for how it
# would slot in instead.
raw_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# IMPORTANT: don't feed the raw imbalance ratio straight into scale_pos_weight.
# On a ~600:1 dataset like this one, scale_pos_weight=600 makes the loss so
# dominated by the ~300 fraud rows that the very first few trees already
# push almost every prediction's probability up, then additional trees just
# overfit noise in that tiny fraud set. Symptom: early stopping fires after
# 1-2 iterations and you get high recall but near-zero precision (thousands
# of false positives) -- the model never learned real structure, it just
# memorized a shortcut.
# Capping it (empirically 10-20 works well for this kind of ratio) keeps the
# gradient signal strong without letting fraud examples dominate every split.
scale_pos_weight = min(raw_ratio, 15)
print(f"\nRaw imbalance ratio: {raw_ratio:.1f}  (i.e. {(y_train==1).sum()} fraud vs {(y_train==0).sum()} legit in training fold)")
print(f"Using capped scale_pos_weight = {scale_pos_weight:.1f}")

# --- SMOTE alternative (uncomment to use instead of scale_pos_weight) ------
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)
# # If using SMOTE, set scale_pos_weight=1 in the model params below
# ----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
# min_child_samples / subsample / colsample_bytree / reg_lambda are added as
# regularization -- without them, the capped scale_pos_weight can still let
# a handful of trees overfit narrow splits around individual fraud rows.
model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    # average_precision (PR-AUC), not auc, is what to watch for early stopping
    # on data this imbalanced -- see note in the metrics section below on why
    # AUC-ROC can look fine while precision is unusable.
    eval_metric="average_precision",
    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
)

print(f"\nBest iteration: {model.best_iteration_}")
if model.best_iteration_ <= 5:
    print(
        "WARNING: best_iteration is very low. This usually means "
        "scale_pos_weight is still too aggressive for this data, or the "
        "features have little real signal. Check the feature importances "
        "below before trusting these metrics."
    )

# ---------------------------------------------------------------------------
# 5. Evaluate on the held-out test set
# ---------------------------------------------------------------------------
y_proba = model.predict_proba(X_test)[:, 1]

# Default 0.5 threshold, for a baseline reading
y_pred_default = (y_proba >= 0.5).astype(int)

auc_roc = roc_auc_score(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)  # area under PR curve
precision_default = precision_score(y_test, y_pred_default)
recall_default = recall_score(y_test, y_pred_default)
f1_default = f1_score(y_test, y_pred_default)

print("\n" + "=" * 60)
print("TEST SET METRICS (threshold = 0.5)")
print("=" * 60)
print(f"AUC-ROC:              {auc_roc:.4f}")
print(f"Average Precision:    {avg_precision:.4f}  (area under precision-recall curve)")
print(f"Precision:            {precision_default:.4f}")
print(f"Recall:               {recall_default:.4f}")
print(f"F1:                   {f1_default:.4f}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_default).ravel()
print(f"\nConfusion matrix @ 0.5: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
print(f"  -> caught {tp}/{tp+fn} fraud cases, missed {fn}")
print(f"  -> {fp} legitimate transactions would be wrongly blocked/flagged")

# Why AUC-ROC can look fine while precision is unusable: ROC-AUC is measured
# against the FALSE POSITIVE RATE (FP / total negatives), and with ~200k
# negatives even 14,000+ false positives is only a ~7% FPR -- which barely
# dents ROC-AUC. But precision is measured against total PREDICTED positives,
# and with only ~95 real fraud cases in the test set, 14,000 false alarms
# drown them out completely (precision collapses to <1%). This is exactly
# why your LLD calls out precision/recall over accuracy -- but note it
# actually needs to go one step further: AUC-ROC alone isn't sufficient
# either on data this imbalanced. Average Precision (area under the
# precision-recall curve) is the more honest single-number summary here.

# ---------------------------------------------------------------------------
# 6. Choose operating thresholds on validation data, not the held-out test set
# ---------------------------------------------------------------------------
y_val_proba = model.predict_proba(X_val)[:, 1]
block_threshold, block_target_met = select_threshold(y_val, y_val_proba, 0.90)
review_threshold, review_target_met = select_threshold(y_val, y_val_proba, 0.10)
review_threshold = min(review_threshold, block_threshold)

print("\nThresholds selected on validation data:")
print(f"  BLOCK: {block_threshold:.6f} (minimum precision target: 0.90)")
print(f"  REVIEW: {review_threshold:.6f} (minimum precision target: 0.10)")
if not block_target_met:
    print("WARNING: validation data could not meet the block precision target.")
if not review_target_met:
    print("WARNING: validation data could not meet the review precision target.")

# Evaluate both operating points once on the untouched test set.
y_pred_block = (y_proba >= block_threshold).astype(int)
y_pred_review = (y_proba >= review_threshold).astype(int)
precision_chosen = precision_score(y_test, y_pred_block, zero_division=0)
recall_chosen = recall_score(y_test, y_pred_block, zero_division=0)
f1_chosen = f1_score(y_test, y_pred_block, zero_division=0)
precision_review = precision_score(y_test, y_pred_review, zero_division=0)
recall_review = recall_score(y_test, y_pred_review, zero_division=0)
f1_review = f1_score(y_test, y_pred_review, zero_division=0)

tn_block, fp_block, fn_block, tp_block = confusion_matrix(
    y_test, y_pred_block
).ravel()
tn_review, fp_review, fn_review, tp_review = confusion_matrix(
    y_test, y_pred_review
).ravel()
print("\nTEST SET AT BLOCK THRESHOLD")
print(f"Precision: {precision_chosen:.4f}  Recall: {recall_chosen:.4f}  F1: {f1_chosen:.4f}")
print(f"TP={tp_block}  FP={fp_block}  TN={tn_block}  FN={fn_block}")
print("\nTEST SET AT REVIEW THRESHOLD")
print(f"Precision: {precision_review:.4f}  Recall: {recall_review:.4f}  F1: {f1_review:.4f}")
print(f"TP={tp_review}  FP={fp_review}  TN={tn_review}  FN={fn_review}")

# ---------------------------------------------------------------------------
# 7. Feature importance (sanity check -- do the engineered features matter?)
# ---------------------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 feature importances:")
print(importances.head(10))

# ---------------------------------------------------------------------------
# 8. Save the model artifact + feature list + metrics
# ---------------------------------------------------------------------------
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

with open(FEATURE_LIST_PATH, "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"Feature list saved to {FEATURE_LIST_PATH}")

metrics = {
    "precision": round(precision_chosen, 4),
    "recall": round(recall_chosen, 4),
    "f1": round(f1_chosen, 4),
    "review_precision": round(precision_review, 4),
    "review_recall": round(recall_review, 4),
    "review_f1": round(f1_review, 4),
    "auc_roc": round(auc_roc, 4),
    "average_precision": round(avg_precision, 4),
    "block_threshold": block_threshold,
    "review_threshold": review_threshold,
    "block_target_precision": 0.90,
    "review_target_precision": 0.10,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {METRICS_PATH}")

# Best-effort: also record this training run in Postgres (LLD §7's
# model_metrics table), if it's reachable. Training is an offline/batch
# step, not a live request path, so there's no fail-open requirement here
# like api.py has -- but training shouldn't hard-fail just because Postgres
# happens to be down, so this is wrapped rather than left to crash the script.
try:
    import db
    db.insert_model_metrics(
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        threshold=metrics["block_threshold"],
        review_threshold=metrics["review_threshold"],
        auc_roc=metrics["auc_roc"],
    )
    print("Also recorded this run in Postgres (model_metrics table)")
except Exception as e:
    print(f"Note: could not write to Postgres model_metrics table ({e}). "
          f"models/model_metrics.json is still saved and used by api.py either way.")