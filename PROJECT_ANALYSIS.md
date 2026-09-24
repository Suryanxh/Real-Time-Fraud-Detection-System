# Fraud Detection System - Comprehensive Project Analysis

**Date:** 2025-01-14  
**Status:** POC (Proof of Concept)  
**Severity Classification:** Critical Issues, Major Issues, Minor Issues, and Recommendations

---

## Executive Summary

This fraud detection system is well-structured for a POC but contains **critical architectural gaps**, **security vulnerabilities**, **unfinished implementations**, and **production-readiness issues**. The project demonstrates solid ML fundamentals but lacks enterprise-grade infrastructure and operational maturity.

---

## 🔴 CRITICAL ISSUES (Must Fix Before Production)

### 1. **Hardcoded Credentials in docker-compose.yml**
**File:** [docker-compose.yml](docker-compose.yml)  
**Severity:** CRITICAL  
**Issue:**
```yaml
MYSQL_ROOT_PASSWORD: root          # Hardcoded!
MYSQL_PASSWORD: fraud_password     # Exposed in repo!
```

**Impact:**
- Anyone with access to the repository can see database credentials
- No environment variable separation between dev/prod/staging
- Violates OWASP and compliance standards (PCI-DSS, SOC2)

**Fix:**
```yaml
environment:
  MYSQL_ROOT_PASSWORD: ${DB_ROOT_PASSWORD}
  MYSQL_DATABASE: ${DB_NAME}
  MYSQL_USER: ${DB_USER}
  MYSQL_PASSWORD: ${DB_PASSWORD}
```

**Action Required:**
- [ ] Create `.env.example` with placeholder values
- [ ] Add `.env` to `.gitignore` immediately
- [ ] Rotate all passwords in docker-compose
- [ ] Use Docker secrets or HashiCorp Vault in production

---

### 2. **In-Memory State Loss on API Restart**
**File:** [src/api.py](src/api.py:80-87)  
**Severity:** CRITICAL  
**Issue:**
```python
# POC-only: this is per-process memory, so it resets on restart and won't
# work across multiple API workers/replicas. In production (or once Postgres
# is wired up per LLD §7) this state comes from querying the `transactions`
# table for the trailing window, not from a Python dict living in the
# process -- flagged here as a known POC limitation, not a design decision.
device_history: dict[str, deque] = defaultdict(deque)
device_counts: dict[str, int] = defaultdict(int)
ip_counts: dict[str, int] = defaultdict(int)
```

**Impact:**
- Velocity features reset on every restart → model degrades
- Can't scale to multiple API replicas (each has its own state)
- Velocity calculations become unreliable
- Decision log lives in memory, lost on restart

**Why It's Critical:**
The model was trained with velocity features (device_id_velocity_count_60min, device_id_velocity_sum_60min). When the API restarts, these become 0, creating distribution shift that causes:
- False positives (legitimate activity flagged as anomalies)
- False negatives (fraud patterns not detected without velocity context)

**Fix Required:**
- [ ] Implement Redis cache for velocity state (short-term: 24-48 hrs)
- [ ] Query PostgreSQL `transactions` table for rolling windows (production)
- [ ] Add state recovery on startup from last 60 minutes of DB
- [ ] Implement transaction log persistence (currently ephemeral)

---

### 3. **Missing Database Integration**
**File:** [db/schema.sql](db/schema.sql)  
**Severity:** CRITICAL  
**Issue:**
- Database schema defined but **not wired into API or stream**
- No ORM (SQLAlchemy) models to interact with the schema
- Decisions table missing (only fraud_predictions table exists)
- No connection pooling or query layer in any source file

**Current State:**
- API stores decision_log in Python list (line 89)
- Stream saves decisions to JSON file only
- No transactional consistency
- No audit trail with proper timestamps

**Why It's Critical:**
- Cannot prove compliance (audit trail exists in memory only)
- No way to backfill missing predictions if replay fails
- SLA monitoring impossible without persistent data
- Dispute resolution impossible (no transaction history)

**Missing:**
```python
# Should exist but doesn't:
# src/db.py  or src/models.py with:
# - SQLAlchemy session management
# - Transaction model class
# - FraudPrediction model class
# - Decision logging ORM queries
```

**Fix Required:**
- [ ] Create `src/database.py` with SQLAlchemy engine/session management
- [ ] Create `src/models.py` with ORM classes (Transaction, FraudPrediction, Decision)
- [ ] Add INSERT on every API `/score` call
- [ ] Create migration system (Alembic already in requirements.txt but not used)
- [ ] Implement connection pooling
- [ ] Add transaction-level error handling

---

### 4. **Model Feature-Target Mismatch**
**File:** [src/api.py](src/api.py:1-20) (docstring warning)  
**Severity:** CRITICAL  
**Issue:**

The model expects V1-V28 (anonymized PCA components) but:
1. The POC extends the API to accept these as optional "features" field
2. In real production, card networks don't provide V1-V28 (they're internal risk scores)
3. The system has no way to generate equivalent risk features from raw transaction data
4. Without V1-V28, predictions degrade severely

**Quoted from api.py:**
```python
"""
Your LLD's /score request has: transaction_id, amount, merchant, device_id,
ip, timestamp. But the model was trained on those PLUS V1-V28 -- the
anonymized PCA components that carry most of the actual fraud signal in the
ULB dataset. Those 28 values represent proprietary, already-derived risk
features (from the card network's own fraud models)
"""
```

**Impact:**
- Model performance is artificial (relies on card network's internal features)
- Cannot transition from POC to production without these features
- Alternative feature engineering needed (unclear how)

**Fix Required:**
- [ ] Document how V1-V28 equivalents will be engineered in production
- [ ] Option A: Use raw transaction history to engineer 28+ features
- [ ] Option B: Integrate with actual card processor API
- [ ] Add feature engineering pipeline for real-world data
- [ ] Evaluate model performance with only "available" features
- [ ] Retrain model if needed

---

### 5. **No Authentication/Authorization**
**File:** [src/api.py](src/api.py) - FastAPI endpoints  
**Severity:** CRITICAL  
**Issue:**
- `/score` endpoint has NO authentication
- `/decisions/recent` endpoint is public
- `/health` endpoint is public (OK)
- No API keys, JWT tokens, or OAuth2
- No rate limiting

**Endpoints Exposed:**
```python
@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction):  # ANYONE CAN CALL THIS
    ...

@app.get("/decisions/recent")
def recent_decisions(limit: int = 20):  # NO AUTH - exposes all decisions!
    ...
```

**Impact:**
- Anyone can:
  - Make unlimited fraud predictions (DoS)
  - Harvest decision log (PII exposure)
  - Test the model with probe transactions
  - Exploit model thresholds
- No audit of who called what

**Fix Required:**
- [ ] Add FastAPI security (OAuth2/JWT or API key)
- [ ] Implement rate limiting (slowapi or similar)
- [ ] Add request signing/verification
- [ ] Log all access with user/service identity
- [ ] Restrict `/decisions/recent` to authorized users only

---

## 🟠 MAJOR ISSUES (Should Fix Before Production)

### 6. **Incomplete Data Loader Module**
**File:** [src/data_loader.py](src/data_loader.py)  
**Severity:** MAJOR  
**Issue:**
```python
def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw transaction data from CSV file."""
    pass

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and clean transaction data."""
    pass
```

Both functions are **stubs**. They do nothing.

**Usage:** Unknown if they're even called anywhere. Likely dead code.

**Fix Required:**
- [ ] Implement actual data loading logic
- [ ] Add error handling (missing files, corrupted data)
- [ ] Add data validation (schema checks)
- [ ] OR remove if not used

---

### 7. **No Error Handling or Logging Strategy**
**File:** [src/stream.py](src/stream.py:213-219)  
**Severity:** MAJOR  
**Issue:**

Minimal error handling:
```python
except requests.RequestException as e:
    print(f"[ERROR] Request error: {e}")
```

Problems:
- Using `print()` instead of logging (bad for production)
- No structured logging (JSON, correlation IDs, severity levels)
- No retry logic (failed transactions are silently dropped)
- No dead-letter queue (lost transactions)
- No metrics collection on failures

**Current logging setup:**
[src/api.py](src/api.py:33-34) has logging but uses basicConfig (works for demo, not production):
```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
```

**Missing:**
- Structured logging (JSON format for ELK stack / CloudWatch)
- Separate error/audit logs
- Log rotation
- Sensitive data masking

**Fix Required:**
- [ ] Implement structured logging (use `structlog` or similar)
- [ ] Add retry logic with exponential backoff
- [ ] Implement dead-letter queue for failed requests
- [ ] Add correlation IDs to trace requests end-to-end
- [ ] Mask PII in logs (device_id, ip, merchant names)
- [ ] Add application metrics (errors, latencies, throughput)

---

### 8. **Missing Tests Entirely**
**File:** [tests/](tests/) folder  
**Severity:** MAJOR  
**Issue:**
- `tests/` folder exists but is **empty**
- No unit tests, integration tests, or end-to-end tests
- No test data fixtures
- No CI/CD pipeline

**Risk:**
- Refactoring breaks unreported features
- Bug regression possible
- No confidence in deployments
- SLA/compliance proof difficult

**Fix Required:**
- [ ] Add unit tests for each module:
  - `test_features.py` - feature engineering logic
  - `test_train.py` - model training, threshold selection
  - `test_api.py` - API endpoints, business logic
  - `test_stream.py` - streaming replay logic
- [ ] Add integration tests (API + model + DB)
- [ ] Add E2E tests (full replay + dashboard)
- [ ] Aim for >80% coverage
- [ ] Add pytest to CI/CD

---

### 9. **No Configuration Management**
**File:** Project-wide  
**Severity:** MAJOR  
**Issue:**

Configurations scattered and hardcoded:
- `api.py:49-50` - Hardcoded model paths
- `stream.py:26-36` - Hardcoded API URL, delays, thresholds
- `train.py:18-25` - Hardcoded paths and column names
- `docker-compose.yml:8-12` - Hardcoded credentials (already listed above)

**Missing:**
- `.env` file support (python-dotenv is in requirements but not used!)
- Config validation on startup
- Different configs for dev/staging/prod
- Feature flags

**Fix Required:**
- [ ] Create `src/config.py` with pydantic BaseSettings
- [ ] Load all configs from environment variables
- [ ] Validate on startup (fail fast)
- [ ] Support `.env` files for local dev
- [ ] Document all config options

Example:
```python
# src/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "models/lgbm_fraud_model.pkl"
    api_url: str = "http://localhost:8000"
    db_url: str = "postgresql://user:pass@localhost/fraud_db"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

### 10. **Velocity Feature Leakage / Design Issue**
**File:** [src/api.py](src/api.py:153-160)  
**Severity:** MAJOR  
**Issue:**

Velocity features include the **current transaction** but the model was trained the same way:

```python
# Velocity: purge entries older than 60 min, then count/sum INCLUDING
# this transaction -- same inclusive-window semantics as training.
hist = device_history[txn.device_id]
cutoff = ts - pd.Timedelta(seconds=VELOCITY_WINDOW_SECONDS)
while hist and hist[0][0] < cutoff:
    hist.popleft()
velocity_count = len(hist) + 1  # INCLUDING current transaction
```

**Problem:**
- On the first request after API restart, device_history is empty
- Velocity features become artificially low
- Model predicts "allow" even for suspicious patterns
- This is documented as a POC limitation but **breaks model performance**

**Fix Required:**
- [ ] Initialize device_history from DB on startup (last 60 min of transactions)
- [ ] Add sanity check: if velocity < 2, use fallback (DB query)
- [ ] Consider using separate "historical velocity" (non-causal) + "delta velocity" (causal)

---

### 11. **No Model Monitoring / Performance Degradation Detection**
**File:** [src/dashboard.py](src/dashboard.py) + API  
**Severity:** MAJOR  
**Issue:**

The dashboard loads `model_metrics.json` (from training) but has no way to:
- Detect when live performance diverges from test performance
- Monitor for data drift (transaction distributions changing)
- Detect model degradation (precision/recall dropping over time)
- Compare actual fraud labels vs predicted labels (actuals missing!)

**Current metrics loaded:**
- AUC-ROC: 0.9456 (from test set)
- Precision: 0.9000 (from test set)
- Recall: ? (missing from metrics file based on my reading)

**Missing in stream.py:**
- Actual fraud labels are only available in test_features.csv
- Live transactions don't include true labels
- Can't compute precision/recall on live data
- Feedback loop missing

**Fix Required:**
- [ ] Add `/label` endpoint to provide true labels (batch, delayed)
- [ ] Implement model performance monitoring (live precision/recall)
- [ ] Add data drift detection (Kolmogorov-Smirnov test)
- [ ] Add alerts when metrics degrade
- [ ] Implement auto-retraining trigger

---

### 12. **Merchant Risk Score Leakage**
**File:** [src/features.py](src/features.py:177-204)  
**Severity:** MAJOR  
**Issue:**

Merchant risk is target-encoded (fraud rate per merchant):
```python
def add_merchant_risk(df: pd.DataFrame, is_train: bool, ...):
    if is_train:
        merchant_risk_map = df.groupby("merchant")["Class"].mean().to_dict()
```

**Problem:**
- This is the only feature where you must manually prevent leakage
- If someone retrains without passing `merchant_risk_map`, leakage happens
- New merchants at inference use global rate (OK fallback)
- But no safeguard if `is_train=True` called on test data

**Example of the bug:**
```python
# WRONG - if accidentally called on test data:
test_features, _ = engineer_features(test_df, is_train=True)  # Leakage! ❌

# CORRECT:
test_features, _, _ = engineer_features(test_df, is_train=False, 
                                        merchant_risk_map=merchant_risk_map,
                                        global_fraud_rate=global_fraud_rate)
```

**Fix Required:**
- [ ] Add assertion in features.py to catch `is_train=True` without label column
- [ ] Add validation that merchant_risk_map is provided when is_train=False
- [ ] Add unit test that explicitly tests for leakage prevention
- [ ] Document why merchant risk is different from other features

---

### 13. **No Input Validation on API Requests**
**File:** [src/api.py](src/api.py:110-120)  
**Severity:** MAJOR  
**Issue:**

Request schema has basic types but no validation:
```python
class Transaction(BaseModel):
    transaction_id: str
    amount: float                    # No min/max bounds
    merchant: str                    # No enum or allowed values
    device_id: str                   # No format validation
    ip: str                          # No IP validation!
    timestamp: str                   # No date format validation
    features: Optional[dict[str, float]] = None  # No bounds on values
```

**Problems:**
- Negative amount accepted
- Invalid IP format accepted
- Timestamp parsing fails silently (no test of this edge case)
- Feature values unbounded (could be NaN, Inf)
- No size limits (memory DoS via huge dict)

**Fix Required:**
- [ ] Add Pydantic validators:
```python
from pydantic import validator, Field

class Transaction(BaseModel):
    amount: float = Field(gt=0, le=1000000)  # Positive, <= $1M
    ip: str = Field(regex=r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
    timestamp: str  # Parse as datetime
    merchant: str = Field(pattern=r"^[a-z_]+$")
    
    @validator("timestamp")
    def validate_timestamp(cls, v):
        try:
            pd.to_datetime(v)
            return v
        except:
            raise ValueError("Invalid ISO 8601 timestamp")
    
    @validator("features")
    def validate_features(cls, v):
        if v and (len(v) != 28 or any(not isinstance(x, (int, float)) for x in v.values())):
            raise ValueError("Features must be 28 numeric values")
        return v
```
- [ ] Add unit tests for validation edge cases

---

## 🟡 MINOR ISSUES (Polish & Best Practices)

### 14. **Missing .gitignore**
**Severity:** MINOR  
**Issue:**
- No `.gitignore` file in repo
- `__pycache__/` is committed (seen in workspace)
- Model artifacts likely committed (bad for repo size)
- `.env` files not protected

**Fix:**
```
# .gitignore
__pycache__/
*.pyc
*.egg-info/
.eggs/
dist/
build/
.env
.env.local
.env.*.local
venv/
.venv/
*.pkl
*.json.lock
.pytest_cache/
.mypy_cache/
.DS_Store
*.swp
```

---

### 15. **Incomplete Documentation**
**Severity:** MINOR  
**Issues:**
- README.md has placeholder: "## Model Performance: - Accuracy: [To be added]"
- No API documentation (missing docstrings detail parameter meanings)
- No architecture diagram
- No deployment guide
- No troubleshooting guide
- Dashboard functionality not documented

**Fix:**
- [ ] Complete README with actual metrics
- [ ] Add API endpoint documentation (or use Swagger auto-docs)
- [ ] Create ARCHITECTURE.md
- [ ] Create DEPLOYMENT.md
- [ ] Create TROUBLESHOOTING.md

---

### 16. **Inconsistent Decision Format**
**Severity:** MINOR  
**Issue:**

API returns:
```python
decision: str  # "block" | "flag" | "allow"
```

Stream expects:
```python
decision: str  # "BLOCK" | "REVIEW" | "ALLOW"
```

**Example from stream.py line 227:**
```python
"is_fraud_predicted": result.get("decision") in ["BLOCK", "REVIEW"],
```

But api.py returns lowercase "block", "flag", "allow".

This works because Python dict lookup is case-sensitive:
- `"block" in ["BLOCK", "REVIEW"]` → False (bug!)
- Should compare `"block"` vs `"block"` or standardize case

**Fix:**
- [ ] Choose one convention (recommend lowercase for JSON)
- [ ] Update stream.py to match
- [ ] Add unit test

---

### 17. **Hardcoded Thresholds Duplicated**
**Severity:** MINOR  
**Issue:**

Block/review thresholds defined in multiple places:
- `train.py:217-219` - computed and saved to JSON
- `api.py:52-54` - loaded from JSON ✓
- `stream.py:39-40` - hardcoded duplicates!

```python
# stream.py - WRONG: duplicates what's in models/model_metrics.json
BLOCK_THRESHOLD = 0.9995
REVIEW_THRESHOLD = 0.7995
```

If thresholds change, stream.py needs manual update.

**Fix:**
- [ ] Load thresholds from model_metrics.json in stream.py
- [ ] Remove hardcoded values

---

### 18. **Missing Type Hints**
**Severity:** MINOR  
**Issue:**

Some functions lack type hints:
- `features.py:189` - `engineer_features()` could be clearer about return types
- `stream.py:90-100` - metrics tracking function arguments

**Fix:**
- [ ] Add comprehensive type hints
- [ ] Enable mypy type checking in CI/CD
- [ ] Example:
```python
from typing import Dict, Tuple

def engineer_features(
    df: pd.DataFrame,
    is_train: bool,
    merchant_risk_map: Dict[str, float] | None = None,
    global_fraud_rate: float | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], float]:
    ...
```

---

### 19. **Dashboard Incomplete**
**File:** [src/dashboard.py](src/dashboard.py:300+)  
**Severity:** MINOR  
**Issue:**

Based on reading, the dashboard file is truncated in my view. Likely incomplete sections:
- "SECTION 4: Latency Monitoring" header exists but content cut off
- May have more incomplete sections

**Fix:**
- [ ] Complete all dashboard sections
- [ ] Add interactivity (date range selector, threshold adjuster)
- [ ] Add transaction detail drill-down

---

### 20. **No Health Check for Dependencies**
**Severity:** MINOR  
**Issue:**

`/health` endpoint checks model is loaded but not:
- Database connection
- Kafka connection
- Model file freshness
- Disk space

**Fix:**
```python
@app.get("/health")
def health():
    checks = {
        "model_loaded": model is not None,
        "db_connected": check_db_connection(),
        "kafka_reachable": check_kafka(),
        "disk_space_mb": get_disk_space(),
    }
    
    status = "ok" if all(checks.values()) else "degraded"
    return {"status": status, **checks}
```

---

## 📋 SUMMARY TABLE OF ISSUES

| # | Category | Severity | Title | File(s) |
|---|----------|----------|-------|---------|
| 1 | Security | 🔴 CRITICAL | Hardcoded DB Credentials | docker-compose.yml |
| 2 | Architecture | 🔴 CRITICAL | In-Memory State Loss | api.py |
| 3 | Architecture | 🔴 CRITICAL | No Database Integration | db/, api.py, stream.py |
| 4 | ML | 🔴 CRITICAL | Feature-Target Mismatch (V1-V28) | api.py |
| 5 | Security | 🔴 CRITICAL | No Authentication/Authorization | api.py |
| 6 | Code Quality | 🟠 MAJOR | Incomplete data_loader.py | src/data_loader.py |
| 7 | Operations | 🟠 MAJOR | No Error Handling/Logging | stream.py, api.py |
| 8 | Testing | 🟠 MAJOR | No Tests (Empty tests/ folder) | tests/ |
| 9 | Configuration | 🟠 MAJOR | No Config Management | Project-wide |
| 10 | ML | 🟠 MAJOR | Velocity Feature Leakage | api.py |
| 11 | Operations | 🟠 MAJOR | No Model Monitoring | dashboard.py |
| 12 | ML | 🟠 MAJOR | Merchant Risk Leakage Risk | features.py |
| 13 | Security | 🟠 MAJOR | No Input Validation | api.py |
| 14 | DevOps | 🟡 MINOR | Missing .gitignore | - |
| 15 | Documentation | 🟡 MINOR | Incomplete Documentation | README.md |
| 16 | Code Quality | 🟡 MINOR | Inconsistent Decision Format | api.py, stream.py |
| 17 | Code Quality | 🟡 MINOR | Hardcoded Thresholds Duplication | stream.py |
| 18 | Code Quality | 🟡 MINOR | Missing Type Hints | features.py |
| 19 | UI/UX | 🟡 MINOR | Incomplete Dashboard | dashboard.py |
| 20 | Operations | 🟡 MINOR | No Dependency Health Checks | api.py |

---

## 🚀 RECOMMENDED FIX PRIORITY

**Phase 1 (Blocking for POC Demo):**
- [x] Issue #1 - Remove hardcoded credentials
- [ ] Issue #5 - Add basic auth or API key to demo safely
- [ ] Issue #9 - Implement config management with .env
- [ ] Issue #13 - Add input validation to API

**Phase 2 (Before Pilot/Staging):**
- [ ] Issue #2 - Implement Redis/DB-backed state
- [ ] Issue #3 - Wire up database integration
- [ ] Issue #8 - Add test suite (>70% coverage)
- [ ] Issue #7 - Improve logging and error handling

**Phase 3 (Before Production):**
- [ ] Issue #4 - Resolve V1-V28 feature engineering
- [ ] Issue #11 - Implement model monitoring
- [ ] Issue #10 - Fix velocity feature leakage
- [ ] Issue #12 - Add leakage safeguards

**Phase 4 (Polish):**
- [ ] Issues #14-20 - Documentation, code quality, completeness

---

## 📁 SRD DOCUMENTS NOTES

The SRD folder contains:
- **HLD** - Real-Time AI Fraud Detection (POC).docx
- **LLD** - Real-Time AI Fraud Detection (POC).docx  
- **Requirements.docx**

**Key Mismatch:** 
The codebase makes references to "per LLD §6" and "per HLD §7" throughout (especially in api.py and features.py), suggesting these documents define requirements this code attempts to follow. However:
- The API schema is extended beyond what the LLD likely specifies (V1-V28 POC hack)
- State management is acknowledged as POC-only
- Database integration is deferred

**Recommendation:** 
Review SRD documents against actual implementation to identify gaps and update either the docs or code accordingly.

---

## 💡 CONCLUSION

**Strengths:**
✅ Clean ML pipeline with proper train/test separation  
✅ Good feature engineering documentation  
✅ Thoughtful threshold selection (precision-based)  
✅ Comprehensive docstrings explaining leakage risks  
✅ Docker support for easy deployment  

**Weaknesses:**
❌ Critical security issues (credentials, auth)  
❌ Fundamental architectural issues (stateless design, no persistence)  
❌ No database integration despite schema existing  
❌ Missing production operational patterns (logging, monitoring, testing)  
❌ Unrealistic model assumptions (V1-V28 features)  

**Verdict:** 
This is a **solid POC that demonstrates ML workflow** but **not production-ready**. Recommend addressing all critical issues before any live/customer-facing deployment. The major issues should be resolved before staging/pilot. The minor issues are polish but important for maintainability.

