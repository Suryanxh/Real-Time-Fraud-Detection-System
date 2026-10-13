import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  CheckCircle2,
  Clock3,
  RefreshCw,
  Shield,
  ShieldAlert,
  Wifi,
  WifiOff,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const REFRESH_MS = 2000;

async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${path} returned HTTP ${response.status}`);
  }
  return response.json();
}

function formatPercent(value) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function StatCard({ label, value, detail, icon: Icon, tone = "blue" }) {
  return (
    <article className="stat-card">
      <div className="stat-topline">
        <span className="eyebrow">{label}</span>
        <span className={`stat-icon ${tone}`}><Icon size={18} /></span>
      </div>
      <strong className="stat-value">{value}</strong>
      <span className="stat-detail">{detail}</span>
    </article>
  );
}

function DecisionBadge({ decision }) {
  const value = (decision || "unknown").toLowerCase();
  return <span className={`decision-badge ${value}`}>{value}</span>;
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [summary, setSummary] = useState(null);
  const [decisions, setDecisions] = useState([]);
  const [apiError, setApiError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [filter, setFilter] = useState("all");
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const [healthData, summaryData, recentData] = await Promise.all([
        getJson("/health"),
        getJson("/decisions/summary"),
        getJson("/decisions/recent?limit=50"),
      ]);
      setHealth(healthData);
      setSummary(summaryData);
      setDecisions(Array.isArray(recentData) ? [...recentData].reverse() : []);
      setApiError("");
      setLastUpdated(new Date());
    } catch (error) {
      setApiError(
        `${error.message}. Start the FastAPI server at ${API_BASE} and refresh.`,
      );
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const filteredDecisions = decisions.filter(
    (item) => filter === "all" || item.decision === filter,
  );
  const total = summary?.total ?? 0;
  const flaggedRate = summary?.flagged_rate ?? 0;
  const maxCount = Math.max(summary?.block ?? 0, summary?.flag ?? 0, summary?.allow ?? 0, 1);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Shield size={21} /></div>
          <div>
            <span className="brand-name">SENTINEL</span>
            <span className="brand-caption">FRAUD INTELLIGENCE</span>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>
        <div className="nav-item active"><Activity size={17} /> Live monitor</div>
        <div className="sidebar-spacer" />
        <div className="sidebar-status">
          <span className={`status-dot ${health?.status === "ok" ? "online" : ""}`} />
          <span>{health?.status === "ok" ? "API connected" : "Connecting to API"}</span>
        </div>
        <div className="sidebar-footer">FRAUD DETECTION POC <span>v1.0</span></div>
      </aside>

      <section className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Monitoring <span>/</span> Live overview</div>
          <div className="topbar-right">
            <div className={`connection-pill ${apiError ? "offline" : ""}`}>
              {apiError ? <WifiOff size={15} /> : <Wifi size={15} />}
              {apiError ? "Offline" : "Live"}
            </div>
            <button className="refresh-button" onClick={refresh} disabled={refreshing} aria-label="Refresh dashboard">
              <RefreshCw size={16} className={refreshing ? "spin" : ""} />
              <span>Refresh</span>
            </button>
          </div>
        </header>

        <div className="page">
          <section className="page-heading">
            <div>
              <div className="overline"><span className="live-dot" /> REAL-TIME OPERATIONS</div>
              <h1>Fraud monitor</h1>
              <p>Transaction decisions and model performance at a glance.</p>
            </div>
            <div className="updated-at"><Clock3 size={15} /> Updated {formatTime(lastUpdated)}</div>
          </section>

          {apiError && (
            <div className="error-banner" role="alert">
              <AlertTriangle size={18} />
              <div><strong>Can’t reach the scoring API</strong><span>{apiError}</span></div>
            </div>
          )}

          <section className="stats-grid" aria-label="Transaction summary">
            <StatCard label="Transactions scored" value={total.toLocaleString()} detail="Since API started" icon={Activity} />
            <StatCard label="Blocked" value={(summary?.block ?? 0).toLocaleString()} detail="High-risk transactions" icon={ShieldAlert} tone="red" />
            <StatCard label="Flagged for review" value={(summary?.flag ?? 0).toLocaleString()} detail="Requires attention" icon={AlertTriangle} tone="amber" />
            <StatCard label="Allowed" value={(summary?.allow ?? 0).toLocaleString()} detail="Passed by the model" icon={CheckCircle2} tone="green" />
          </section>

          <section className="insight-grid">
            <article className="panel distribution-panel">
              <div className="panel-heading">
                <div><span className="eyebrow">DECISION BREAKDOWN</span><h2>Transaction outcomes</h2></div>
                <span className="panel-tag">LIVE</span>
              </div>
              <div className="bar-chart">
                {[
                  { label: "Allowed", value: summary?.allow ?? 0, color: "green" },
                  { label: "Flagged", value: summary?.flag ?? 0, color: "amber" },
                  { label: "Blocked", value: summary?.block ?? 0, color: "red" },
                ].map((bar) => (
                  <div className="bar-row" key={bar.label}>
                    <span>{bar.label}</span>
                    <div className="bar-track"><div className={`bar-fill ${bar.color}`} style={{ width: `${(bar.value / maxCount) * 100}%` }} /></div>
                    <strong>{bar.value}</strong>
                  </div>
                ))}
              </div>
              <div className="chart-foot"><span>Flagged rate</span><strong>{formatPercent(flaggedRate)}</strong></div>
            </article>

            <article className="panel model-panel">
              <div className="panel-heading">
                <div><span className="eyebrow">MODEL STATUS</span><h2>Scoring engine</h2></div>
                <span className={`model-indicator ${health?.model_loaded ? "ready" : ""}`}>
                  <span className="status-dot" /> {health?.model_loaded ? "Ready" : "Waiting"}
                </span>
              </div>
              <div className="model-row"><span>Features used</span><strong>{health?.n_features ?? "—"}</strong></div>
              <div className="model-row"><span>Block threshold</span><strong>{health?.block_threshold?.toFixed(4) ?? "—"}</strong></div>
              <div className="model-row"><span>Review threshold</span><strong>{health?.review_threshold?.toFixed(4) ?? "—"}</strong></div>
              <div className="model-note"><Shield size={15} /> Scores are calculated by the trained model via FastAPI.</div>
            </article>
          </section>

          <section className="panel feed-panel">
            <div className="panel-heading feed-heading">
              <div><span className="eyebrow">LIVE ACTIVITY</span><h2>Recent transactions</h2></div>
              <div className="feed-controls">
                <span className="feed-count">{filteredDecisions.length} shown</span>
                <select value={filter} onChange={(event) => setFilter(event.target.value)} aria-label="Filter by decision">
                  <option value="all">All decisions</option>
                  <option value="block">Blocked</option>
                  <option value="flag">Flagged</option>
                  <option value="allow">Allowed</option>
                </select>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>TRANSACTION</th><th>AMOUNT</th><th>RISK SCORE</th><th>DECISION</th><th>REASON</th><th>TIME</th></tr></thead>
                <tbody>
                  {filteredDecisions.map((item, index) => (
                    <tr key={`${item.transaction_id}-${index}`}>
                      <td className="transaction-id">{item.transaction_id}</td>
                      <td>${Number(item.amount ?? 0).toFixed(2)}</td>
                      <td><div className="score-cell"><span className="score-track"><span style={{ width: `${Math.min(Number(item.fraud_score ?? 0) * 100, 100)}%` }} /></span><strong>{(Number(item.fraud_score ?? 0) * 100).toFixed(1)}%</strong></div></td>
                      <td><DecisionBadge decision={item.decision} /></td>
                      <td className="reason-cell">{(item.reason || "—").replaceAll("_", " ")}</td>
                      <td className="time-cell">{formatTime(item.logged_at)}</td>
                    </tr>
                  ))}
                  {filteredDecisions.length === 0 && (
                    <tr><td colSpan="6" className="empty-state">
                      {apiError ? "Waiting for the API connection…" : "No transactions yet. Start the API and replay transactions with src/stream.py."}
                    </td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <section className="performance-row">
            <div className="performance-title"><span className="eyebrow">LABELED DEMO METRICS</span><span>Calculated from replayed transactions</span></div>
            <div className="performance-metric"><span>Precision</span><strong>{formatPercent(summary?.live_precision)}</strong><ArrowUpRight size={15} /></div>
            <div className="performance-metric"><span>Recall</span><strong>{formatPercent(summary?.live_recall)}</strong><ArrowDownRight size={15} /></div>
            <div className="performance-caption">Demo-only live metrics; use the offline test-set results to discuss model quality.</div>
          </section>

          <footer className="page-footer">FRAUD DETECTION SYSTEM <span>·</span> COLLEGE PROJECT DEMO</footer>
        </div>
      </section>
    </main>
  );
}
