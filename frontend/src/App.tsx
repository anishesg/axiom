import { useState, useEffect, useRef, useCallback } from "react";

// ── Types ──────────────────────────────────────────────────────

interface BrainState {
  engagement: number;
  focus: number;
  valence: number;
  cognitive_load: number;
  relaxation: number;
  jaw_clench: boolean;
  error_response: number;
  hmm_state: string;
  hmm_probs: Record<string, number>;
  eegnet_features: number[];
}

interface GazeState {
  screen_x: number;
  screen_y: number;
  viewport_x: number;
  viewport_y: number;
  fixating: boolean;
  fixation_duration: number;
  element: { id: string; role: string; name: string } | null;
  confidence: number;
}

interface PageState {
  url: string;
  title: string;
  domain: string;
  element_count: number;
  elements: { id: string; role: string; name: string; bbox: Record<string, number> }[];
}

interface IntentState {
  element_id: string | null;
  element_name: string | null;
  probability: number;
  commitment_level: string;
  action_prediction: string | null;
  action_confidence: number;
}

interface ActionEntry {
  action: string;
  target: string;
  element_role?: string;
  confidence: number;
  reason: string;
  timestamp: number;
  result?: string;
  errp_undone?: boolean;
}

interface ErrPState {
  detected: boolean;
  probability: number;
  undoing: string | null;
  target: string | null;
}

interface CalState {
  type: string;
  targets?: { x: number; y: number }[];
  total?: number;
  success?: boolean;
}

interface BanditActionStat {
  count: number;
  avg_reward: number;
  total_reward: number;
}

// ── Constants ──────────────────────────────────────────────────

const ZERO_BRAIN: BrainState = {
  engagement: 0, focus: 0, valence: 0.5,
  cognitive_load: 0, relaxation: 0,
  jaw_clench: false, error_response: 0,
  hmm_state: "browsing",
  hmm_probs: { browsing: 0.6, searching: 0.1, intending: 0.1, resting: 0.1, error: 0.1 },
  eegnet_features: [],
};

const BRAIN_METRICS: { key: keyof BrainState; label: string; color: string }[] = [
  { key: "engagement", label: "Engagement", color: "#00d4ff" },
  { key: "focus", label: "Focus", color: "#a855f7" },
  { key: "valence", label: "Valence", color: "#ff6b9d" },
  { key: "cognitive_load", label: "Cog Load", color: "#ffaa00" },
  { key: "relaxation", label: "Relaxation", color: "#00ff88" },
];

const HMM_STATES: { key: string; label: string; color: string }[] = [
  { key: "browsing", label: "BROWSE", color: "#00d4ff" },
  { key: "searching", label: "SEARCH", color: "#ff6b9d" },
  { key: "intending", label: "INTENT", color: "#00ff88" },
  { key: "resting", label: "REST", color: "#888888" },
  { key: "error", label: "ERROR", color: "#ff4444" },
];

const ROLE_COLORS: Record<string, string> = {
  link: "#00d4ff",
  button: "#00ff88",
  textbox: "#a855f7",
  searchbox: "#a855f7",
  heading: "#ffaa00",
  checkbox: "#ff6b9d",
  combobox: "#ff6b9d",
  tab: "#00d4ff",
  menuitem: "#00d4ff",
  radio: "#ff6b9d",
  img: "#ffaa00",
  listitem: "#888888",
};

function getRoleColor(role: string): string {
  return ROLE_COLORS[role] || "#5a6894";
}

function getCommitmentColor(p: number): string {
  if (p >= 0.9) return "#ffffff";
  if (p >= 0.7) return "#00ff88";
  if (p >= 0.4) return "#00d4ff";
  if (p >= 0.15) return "#5a6894";
  return "#1a1a3e";
}

function getCommitmentGlow(p: number): string {
  if (p >= 0.9) return "0 0 30px #ffffff60, 0 0 60px #00ff8840";
  if (p >= 0.7) return "0 0 20px #00ff8840";
  if (p >= 0.4) return "0 0 15px #00d4ff30";
  return "none";
}

// ── App ────────────────────────────────────────────────────────

export default function App() {
  const [connected, setConnected] = useState(false);
  const [phase, setPhase] = useState("startup");
  const [sim, setSim] = useState(false);
  const [brain, setBrain] = useState<BrainState>(ZERO_BRAIN);
  const [gaze, setGaze] = useState<GazeState | null>(null);
  const [page, setPage] = useState<PageState | null>(null);
  const [intent, setIntent] = useState<IntentState | null>(null);
  const [actions, setActions] = useState<ActionEntry[]>([]);
  const [errpFlash, setErrpFlash] = useState(false);
  const [errpInfo, setErrpInfo] = useState<string | null>(null);
  const [calState, setCalState] = useState<CalState | null>(null);
  const [calPointIdx, setCalPointIdx] = useState(0);
  const [systemLog, setSystemLog] = useState<string[]>([]);
  const [banditStats, setBanditStats] = useState<Record<string, BanditActionStat | number>>({});
  const wsRef = useRef<WebSocket | null>(null);

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    function connect() {
      const ws = new WebSocket("ws://127.0.0.1:8765");
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        timer = setTimeout(connect, 2000);
      };
      ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        switch (d.type) {
          case "init":
            setPhase(d.phase);
            setSim(d.sim);
            setActions(d.action_log || []);
            break;
          case "system":
            setPhase(d.phase);
            setSystemLog((prev) => [...prev.slice(-40), d.status]);
            break;
          case "brain":
            setBrain({
              engagement: d.engagement,
              focus: d.focus,
              valence: d.valence,
              cognitive_load: d.cognitive_load,
              relaxation: d.relaxation,
              jaw_clench: d.jaw_clench,
              error_response: d.error_response,
              hmm_state: d.hmm_state || "browsing",
              hmm_probs: d.hmm_probs || {},
              eegnet_features: d.eegnet_features || [],
            });
            break;
          case "gaze":
            setGaze(d as GazeState);
            break;
          case "page":
            setPage(d as PageState);
            break;
          case "intent":
            setIntent(d as IntentState);
            break;
          case "action":
            setActions((prev) => [...prev.slice(-50), d as ActionEntry]);
            break;
          case "errp": {
            const errp = d as ErrPState;
            if (errp.detected) {
              setErrpFlash(true);
              setErrpInfo(`ErrP: undoing ${errp.undoing} on ${errp.target}`);
              setTimeout(() => setErrpFlash(false), 800);
              setTimeout(() => setErrpInfo(null), 3000);
              // Mark the last action as undone
              setActions((prev) => {
                const copy = [...prev];
                if (copy.length > 0) {
                  copy[copy.length - 1] = { ...copy[copy.length - 1], errp_undone: true };
                }
                return copy;
              });
            }
            break;
          }
          case "calibration":
            setCalState(d as CalState);
            if (d.type === "gaze_done" || d.type === "eeg_done") {
              setPhase("live");
            }
            break;
        }
      };
    }
    connect();
    return () => {
      clearTimeout(timer);
      wsRef.current?.close();
    };
  }, []);

  // Periodically request bandit stats (piggybacked on intent updates)
  useEffect(() => {
    // The bandit stats come from the server; we track them from action broadcasts
    // For now, compute local stats from action log
    const stats: Record<string, BanditActionStat> = {};
    for (const a of actions) {
      if (!stats[a.action]) {
        stats[a.action] = { count: 0, avg_reward: 0, total_reward: 0 };
      }
      stats[a.action].count += 1;
      const reward = a.errp_undone ? -1 : 1;
      stats[a.action].total_reward += reward;
      stats[a.action].avg_reward =
        stats[a.action].total_reward / stats[a.action].count;
    }
    setBanditStats(stats);
  }, [actions]);

  const gazeCalTargets = calState?.targets || [];
  const isGazeCal = phase === "gaze_cal" && gazeCalTargets.length > 0;

  const intentP = intent?.probability ?? 0;
  const commitColor = getCommitmentColor(intentP);
  const commitGlow = getCommitmentGlow(intentP);
  const circumference = 2 * Math.PI * 52;

  return (
    <div className="app">
      {/* ErrP flash overlay */}
      {errpFlash && <div className="errp-flash" />}

      {/* Header */}
      <header className="header">
        <div className="header-left">
          <h1 className="title">
            AXIOM<span className="title-accent"> v2</span>
          </h1>
          <span className="subtitle">Universal Brain-Computer Interface</span>
        </div>
        <div className="header-center">
          {page && (
            <div className="header-url">
              <span className="header-domain">{page.domain || "..."}</span>
              <span className="header-page-title">{page.title?.slice(0, 60) || ""}</span>
            </div>
          )}
        </div>
        <div className="header-right">
          <span className={`phase-badge phase-${phase}`}>{phase.toUpperCase()}</span>
          <span className={`conn-dot ${connected ? "on" : "off"}`} />
          <span className="conn-label">{connected ? (sim ? "SIM" : "LIVE") : "OFFLINE"}</span>
        </div>
      </header>

      <div className="main">
        {/* ── Left Panel: Neural State ─────────────────────── */}
        <div className="left-panel">
          <h3 className="panel-title">NEURAL STATE</h3>
          {BRAIN_METRICS.map(({ key, label, color }) => {
            const v = typeof brain[key] === "number" ? (brain[key] as number) : 0;
            return (
              <div key={key} className="brain-metric">
                <div className="bm-head">
                  <span className="bm-label">{label}</span>
                  <span className="bm-val" style={{ color }}>
                    {Math.round(v * 100)}%
                  </span>
                </div>
                <div className="bm-track">
                  <div
                    className="bm-fill"
                    style={{
                      width: `${v * 100}%`,
                      background: color,
                      boxShadow: `0 0 10px ${color}50`,
                    }}
                  />
                </div>
              </div>
            );
          })}

          {/* HMM State */}
          <div className="section-divider" />
          <h3 className="panel-title">HMM STATE</h3>
          <div className="hmm-states">
            {HMM_STATES.map(({ key, label, color }) => {
              const prob = brain.hmm_probs?.[key] ?? 0;
              const isActive = brain.hmm_state === key;
              return (
                <div key={key} className={`hmm-row ${isActive ? "hmm-active" : ""}`}>
                  <div className="hmm-dot" style={{ background: isActive ? color : "#1a1a3e", boxShadow: isActive ? `0 0 8px ${color}` : "none" }} />
                  <span className="hmm-label" style={{ color: isActive ? color : "#5a6894" }}>{label}</span>
                  <div className="hmm-bar-track">
                    <div className="hmm-bar-fill" style={{ width: `${prob * 100}%`, background: color, opacity: isActive ? 1 : 0.4 }} />
                  </div>
                  <span className="hmm-prob">{Math.round(prob * 100)}%</span>
                </div>
              );
            })}
          </div>

          {/* Indicators */}
          <div className="indicator-row">
            {brain.jaw_clench && <div className="clench-indicator">JAW CLENCH</div>}
            {brain.error_response > 0.5 && <div className="errp-indicator">ErrP {Math.round(brain.error_response * 100)}%</div>}
          </div>

          {/* Gaze */}
          <div className="section-divider" />
          <h3 className="panel-title">GAZE</h3>
          {gaze ? (
            <div className="gaze-info">
              <div className="gaze-row">
                <span>Position</span>
                <span>{Math.round(gaze.screen_x)}, {Math.round(gaze.screen_y)}</span>
              </div>
              <div className="gaze-row">
                <span>Fixating</span>
                <span className={gaze.fixating ? "gaze-fix" : "gaze-nofix"}>
                  {gaze.fixating ? `YES (${gaze.fixation_duration.toFixed(1)}s)` : "NO"}
                </span>
              </div>
              <div className="gaze-row">
                <span>Confidence</span>
                <span>{Math.round(gaze.confidence * 100)}%</span>
              </div>
              {gaze.element && (
                <div className="gaze-element">
                  <span className="ge-role" style={{ color: getRoleColor(gaze.element.role) }}>
                    {gaze.element.role}
                  </span>
                  <span className="ge-name">{gaze.element.name.slice(0, 40)}</span>
                </div>
              )}
            </div>
          ) : (
            <p className="gaze-uncal">Not calibrated</p>
          )}

          {/* Calibration buttons */}
          <div className="section-divider" />
          <div className="panel-actions">
            <button className="btn btn-sm btn-s" onClick={() => send({ command: "start_gaze_cal" })}>
              Calibrate Gaze
            </button>
            <button className="btn btn-sm btn-s" onClick={() => send({ command: "start_eeg_cal" })}>
              Calibrate EEG
            </button>
            <button className="btn btn-sm btn-s" onClick={() => send({ command: "set_phase", phase: "live" })}>
              Go Live
            </button>
          </div>
        </div>

        {/* ── Center Panel ─────────────────────────────────── */}
        <div className="center-panel">
          {/* Gaze calibration overlay */}
          {isGazeCal && (
            <div className="gaze-cal-overlay">
              <p className="cal-instruction">Look at each dot and click when ready</p>
              {gazeCalTargets.map((t, i) => (
                <div
                  key={i}
                  className={`cal-dot ${i === calPointIdx ? "active" : i < calPointIdx ? "done" : ""}`}
                  style={{ left: `${(t.x / 2560) * 100}%`, top: `${(t.y / 1440) * 100}%` }}
                  onClick={() => {
                    send({ command: "gaze_point", x: t.x, y: t.y });
                    setCalPointIdx(i + 1);
                  }}
                />
              ))}
            </div>
          )}

          {/* EEG calibration */}
          {phase === "calibrate" && (
            <div className="eeg-cal">
              <h2>EEG Calibration</h2>
              <p>Browse naturally while we record your brain patterns for HMM training.</p>
              <div className="cal-actions">
                {["browsing", "searching", "intending", "resting", "reading"].map((a) => (
                  <button key={a} className="btn btn-cal" onClick={() => send({ command: "cal_action", action: a })}>
                    Record: {a}
                  </button>
                ))}
              </div>
              <button className="btn btn-p" onClick={() => send({ command: "end_eeg_cal" })}>
                Finish Calibration
              </button>
            </div>
          )}

          {/* Startup */}
          {phase === "startup" && (
            <div className="startup">
              <div className="startup-icon">
                <svg viewBox="0 0 80 80" width="80" height="80">
                  <circle cx="40" cy="40" r="35" fill="none" stroke="#00d4ff" strokeWidth="2" opacity="0.3" />
                  <circle cx="40" cy="40" r="25" fill="none" stroke="#00d4ff" strokeWidth="1.5" opacity="0.5" />
                  <circle cx="40" cy="40" r="15" fill="none" stroke="#00d4ff" strokeWidth="1" opacity="0.7" />
                  <circle cx="40" cy="40" r="4" fill="#00d4ff" />
                </svg>
              </div>
              <h2>AXIOM v2</h2>
              <p>Initializing universal brain-computer interface...</p>
              <div className="startup-log">
                {systemLog.slice(-5).map((msg, i) => (
                  <div key={i} className="startup-log-entry">{msg}</div>
                ))}
              </div>
            </div>
          )}

          {/* Live mode */}
          {phase === "live" && (
            <div className="live-view">
              {/* Page info bar */}
              <div className="page-info-bar">
                <div className="pi-domain">{page?.domain || "No page"}</div>
                <div className="pi-title">{page?.title?.slice(0, 80) || "Loading..."}</div>
                <div className="pi-count">{page?.element_count || 0} elements</div>
              </div>

              {/* Intent Commitment Ring — centerpiece */}
              <div className="commitment-section">
                <div className="commitment-ring" style={{ boxShadow: commitGlow }}>
                  <svg viewBox="0 0 120 120" className="cr-svg">
                    {/* Background circle */}
                    <circle cx="60" cy="60" r="52" className="cr-bg" />
                    {/* Progress arc */}
                    <circle
                      cx="60" cy="60" r="52"
                      className="cr-fg"
                      style={{
                        stroke: commitColor,
                        strokeDasharray: `${circumference}`,
                        strokeDashoffset: `${circumference * (1 - intentP)}`,
                        filter: intentP > 0.7 ? `drop-shadow(0 0 6px ${commitColor})` : "none",
                      }}
                    />
                    {/* Center text */}
                    <text x="60" y="54" className="cr-pct" style={{ fill: commitColor }}>
                      {Math.round(intentP * 100)}%
                    </text>
                    <text x="60" y="70" className="cr-level" style={{ fill: commitColor, opacity: 0.7 }}>
                      {intent?.commitment_level?.toUpperCase() || "NONE"}
                    </text>
                  </svg>
                </div>
                <div className="commitment-info">
                  {intent?.element_name ? (
                    <>
                      <div className="ci-element">{intent.element_name.slice(0, 60)}</div>
                      <div className="ci-action">
                        {intent.action_prediction && (
                          <>
                            <span className="ci-action-label">Predicted:</span>
                            <span className="ci-action-name">{intent.action_prediction}</span>
                            <span className="ci-action-conf">({Math.round(intent.action_confidence * 100)}%)</span>
                          </>
                        )}
                      </div>
                      <div className="ci-commitment">
                        <span className="ci-level-dot" style={{ background: commitColor }} />
                        <span>{intent.commitment_level}</span>
                      </div>
                    </>
                  ) : (
                    <div className="ci-empty">No element focused</div>
                  )}
                </div>
              </div>

              {/* Element list */}
              {page && page.elements.length > 0 && (
                <div className="element-list">
                  <h3 className="panel-title">PAGE ELEMENTS</h3>
                  <div className="el-scroll">
                    {page.elements
                      .filter((e) => ["link", "button", "textbox", "searchbox", "checkbox", "combobox", "tab", "menuitem", "radio", "heading"].includes(e.role))
                      .slice(0, 20)
                      .map((e) => {
                        const isFocused = gaze?.element?.id === e.id;
                        const isIntent = intent?.element_id === e.id;
                        return (
                          <div
                            key={e.id}
                            className={`el-row ${isFocused ? "el-focused" : ""} ${isIntent ? "el-intent" : ""}`}
                          >
                            <span
                              className="el-role-badge"
                              style={{
                                background: `${getRoleColor(e.role)}20`,
                                color: getRoleColor(e.role),
                              }}
                            >
                              {e.role}
                            </span>
                            <span className="el-name">{e.name.slice(0, 60)}</span>
                            {isIntent && (
                              <span className="el-intent-p">{Math.round(intentP * 100)}%</span>
                            )}
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Right Panel: Actions + Bandit + Log ──────────── */}
        <div className="right-panel">
          <h3 className="panel-title">ACTION FEED</h3>
          <div className="action-scroll">
            {actions
              .slice(-15)
              .reverse()
              .map((a, i) => (
                <div key={i} className={`action-entry ${a.errp_undone ? "action-undone" : ""}`}>
                  <div className="ae-head">
                    <span className="ae-action">{a.action}</span>
                    <span className="ae-conf">{Math.round(a.confidence * 100)}%</span>
                  </div>
                  <div className="ae-target">{a.target?.slice(0, 50)}</div>
                  <div className="ae-reason">{a.reason?.slice(0, 80)}</div>
                  {a.result && <div className="ae-result">{a.result}</div>}
                  {a.errp_undone && <div className="ae-errp">UNDONE (ErrP)</div>}
                </div>
              ))}
            {actions.length === 0 && <p className="no-actions">No actions yet</p>}
          </div>

          {/* Bandit Stats */}
          <div className="section-divider" />
          <h3 className="panel-title">BANDIT STATS</h3>
          <div className="bandit-stats">
            {Object.entries(banditStats)
              .filter(([k]) => !k.startsWith("_"))
              .filter(([, v]) => typeof v === "object" && v !== null && "count" in v && (v as BanditActionStat).count > 0)
              .map(([action, stat]) => {
                const s = stat as BanditActionStat;
                const avgR = s.avg_reward;
                const barColor = avgR > 0 ? "#00ff88" : avgR < 0 ? "#ff4444" : "#5a6894";
                return (
                  <div key={action} className="bandit-row">
                    <span className="bandit-action">{action}</span>
                    <span className="bandit-count">{s.count}x</span>
                    <div className="bandit-bar-track">
                      <div
                        className="bandit-bar-fill"
                        style={{
                          width: `${Math.min(100, Math.abs(avgR) * 100)}%`,
                          background: barColor,
                        }}
                      />
                    </div>
                    <span className="bandit-reward" style={{ color: barColor }}>
                      {avgR > 0 ? "+" : ""}{avgR.toFixed(2)}
                    </span>
                  </div>
                );
              })}
            {Object.keys(banditStats).length === 0 && (
              <p className="no-actions">No data yet</p>
            )}
          </div>

          {/* System Log */}
          <div className="section-divider" />
          <h3 className="panel-title">SYSTEM LOG</h3>
          <div className="log-scroll">
            {systemLog
              .slice(-12)
              .reverse()
              .map((msg, i) => (
                <div key={i} className="log-entry">
                  {msg}
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* ErrP Toast */}
      {errpInfo && <div className="errp-toast">{errpInfo}</div>}
    </div>
  );
}
