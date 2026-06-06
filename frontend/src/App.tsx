import { useState, useEffect, useRef, useCallback } from "react";

interface BrainState {
  engagement: number;
  focus: number;
  valence: number;
  cognitive_load: number;
  relaxation: number;
  jaw_clench: boolean;
  error_response: number;
}

interface GazeState {
  screen_x: number;
  screen_y: number;
  viewport_x: number;
  viewport_y: number;
  fixating: boolean;
  fixation_duration: number;
  element: { id: string; type: string; label: string } | null;
  confidence: number;
}

interface GmailState {
  view: string;
  element_count: number;
  unread_count: number;
  selected_email: Record<string, string>;
  elements: { id: string; type: string; label: string; bbox: Record<string, number> }[];
}

interface ActionEntry {
  action: string;
  target: string;
  confidence: number;
  reason: string;
  timestamp: number;
  result?: string;
}

interface IntentRing {
  element_id: string;
  element_label: string;
  progress: number;
  intent: string;
  confidence: number;
  engagement: number;
}

interface CalState {
  type: string;
  targets?: { x: number; y: number }[];
  total?: number;
  stats?: Record<string, number>;
  results?: Record<string, unknown>;
  action?: string;
}

const ZERO_BRAIN: BrainState = {
  engagement: 0, focus: 0, valence: 0.5,
  cognitive_load: 0, relaxation: 0,
  jaw_clench: false, error_response: 0,
};

const BRAIN_METRICS: { key: keyof BrainState; label: string; color: string; }[] = [
  { key: "engagement", label: "Engagement", color: "#00d4ff" },
  { key: "focus", label: "Focus", color: "#a855f7" },
  { key: "valence", label: "Valence", color: "#ff6b9d" },
  { key: "cognitive_load", label: "Cog Load", color: "#ffaa00" },
  { key: "relaxation", label: "Relaxation", color: "#00ff88" },
];

export default function App() {
  const [connected, setConnected] = useState(false);
  const [phase, setPhase] = useState("startup");
  const [brain, setBrain] = useState<BrainState>(ZERO_BRAIN);
  const [gaze, setGaze] = useState<GazeState | null>(null);
  const [gmail, setGmail] = useState<GmailState | null>(null);
  const [actions, setActions] = useState<ActionEntry[]>([]);
  const [intentRing, setIntentRing] = useState<IntentRing | null>(null);
  const [calState, setCalState] = useState<CalState | null>(null);
  const [calPointIdx, setCalPointIdx] = useState(0);
  const [systemLog, setSystemLog] = useState<string[]>([]);
  const [errp, setErrp] = useState<string | null>(null);
  const [sim, setSim] = useState(false);
  const [, setThresholds] = useState<Record<string, number>>({});
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
      ws.onclose = () => { setConnected(false); timer = setTimeout(connect, 2000); };
      ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        switch (d.type) {
          case "init":
            setPhase(d.phase); setSim(d.sim);
            setThresholds(d.thresholds || {});
            setActions(d.action_log || []);
            break;
          case "system":
            setPhase(d.phase);
            setSystemLog(prev => [...prev.slice(-30), d.status]);
            break;
          case "brain":
            setBrain({
              engagement: d.engagement, focus: d.focus, valence: d.valence,
              cognitive_load: d.cognitive_load, relaxation: d.relaxation,
              jaw_clench: d.jaw_clench, error_response: d.error_response,
            });
            break;
          case "gaze":
            setGaze(d as GazeState);
            break;
          case "gmail":
            setGmail(d as GmailState);
            break;
          case "action":
            setActions(prev => [...prev.slice(-50), d as ActionEntry]);
            break;
          case "intent_ring":
            setIntentRing(d as IntentRing);
            break;
          case "errp":
            setErrp(`Undoing: ${d.undoing} on ${d.target}`);
            setTimeout(() => setErrp(null), 3000);
            break;
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
    return () => { clearTimeout(timer); wsRef.current?.close(); };
  }, []);

  const gazeCalTargets = calState?.targets || [];
  const isGazeCal = phase === "gaze_cal" && gazeCalTargets.length > 0;

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1 className="title">
            AXIOM<span className="title-accent">.GMAIL</span>
          </h1>
          <span className="subtitle">Brain-Computer Interface</span>
        </div>
        <div className="header-right">
          <span className={`phase-badge phase-${phase}`}>{phase.toUpperCase()}</span>
          <span className={`conn-dot ${connected ? "on" : "off"}`} />
          <span className="conn-label">{connected ? (sim ? "SIM" : "LIVE") : "OFFLINE"}</span>
        </div>
      </header>

      <div className="main">
        {/* Brain State Panel */}
        <div className="brain-panel">
          <h3 className="panel-title">Neural State</h3>
          {BRAIN_METRICS.map(({ key, label, color }) => {
            const v = typeof brain[key] === "number" ? (brain[key] as number) : 0;
            return (
              <div key={key} className="brain-metric">
                <div className="bm-head">
                  <span className="bm-label">{label}</span>
                  <span className="bm-val" style={{ color }}>{Math.round(v * 100)}%</span>
                </div>
                <div className="bm-track">
                  <div className="bm-fill" style={{ width: `${v * 100}%`, background: color, boxShadow: `0 0 10px ${color}50` }} />
                </div>
              </div>
            );
          })}

          {brain.jaw_clench && <div className="clench-indicator">JAW CLENCH</div>}
          {brain.error_response > 0.5 && <div className="errp-indicator">ErrP DETECTED</div>}

          <div className="brain-divider" />
          <h3 className="panel-title">Gaze</h3>
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
              {gaze.element && (
                <div className="gaze-element">
                  <span className="ge-type">{gaze.element.type}</span>
                  <span className="ge-label">{gaze.element.label.slice(0, 40)}</span>
                </div>
              )}
            </div>
          ) : (
            <p className="gaze-uncal">Not calibrated</p>
          )}

          <div className="brain-divider" />
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

        {/* Center Content */}
        <div className="content">
          {/* Gaze Calibration */}
          {isGazeCal && (
            <div className="gaze-cal-overlay">
              <p className="cal-instruction">Look at each dot and press SPACE when ready</p>
              {gazeCalTargets.map((t, i) => (
                <div key={i} className={`cal-dot ${i === calPointIdx ? "active" : i < calPointIdx ? "done" : ""}`}
                  style={{ left: `${(t.x / 2560) * 100}%`, top: `${(t.y / 1440) * 100}%` }}
                  onClick={() => {
                    send({ command: "gaze_point", x: t.x, y: t.y });
                    setCalPointIdx(i + 1);
                  }} />
              ))}
            </div>
          )}

          {/* EEG Calibration */}
          {phase === "calibrate" && (
            <div className="eeg-cal">
              <h2>EEG Calibration</h2>
              <p>Perform these actions in Gmail while we record your brain patterns:</p>
              <div className="cal-actions">
                {["interested", "disinterested", "reading", "scanning", "idle"].map(a => (
                  <button key={a} className="btn btn-cal" onClick={() => send({ command: "cal_action", action: a })}>
                    Record: {a}
                  </button>
                ))}
              </div>
              <div className="cal-stats">
                {calState?.stats && Object.entries(calState.stats).map(([k, v]) => (
                  <span key={k} className="cal-stat">{k}: {v as number}</span>
                ))}
              </div>
              <button className="btn btn-p" onClick={() => send({ command: "end_eeg_cal" })}>
                Finish Calibration
              </button>
            </div>
          )}

          {/* Live Mode */}
          {phase === "live" && (
            <div className="live-view">
              {/* Gmail Status */}
              <div className="gmail-status">
                <div className="gs-view">
                  <span className="gs-icon">{gmail?.view === "inbox" ? "\u{1F4E5}" : gmail?.view === "email" ? "\u{1F4E7}" : "\u{1F4DD}"}</span>
                  <span className="gs-label">{gmail?.view?.toUpperCase() || "LOADING"}</span>
                  {gmail && gmail.unread_count > 0 && <span className="gs-unread">{gmail.unread_count} unread</span>}
                  <span className="gs-elements">{gmail?.element_count || 0} elements</span>
                </div>
              </div>

              {/* Intent Ring */}
              {intentRing && intentRing.progress > 0 && (
                <div className="intent-ring-display">
                  <svg viewBox="0 0 120 120" className="ir-svg">
                    <circle cx="60" cy="60" r="50" className="ir-bg" />
                    <circle cx="60" cy="60" r="50" className="ir-fg"
                      strokeDasharray={`${2 * Math.PI * 50}`}
                      strokeDashoffset={`${2 * Math.PI * 50 * (1 - intentRing.progress)}`} />
                  </svg>
                  <div className="ir-info">
                    <span className="ir-intent">{intentRing.intent}</span>
                    <span className="ir-target">{intentRing.element_label.slice(0, 50)}</span>
                    <span className="ir-conf">{Math.round(intentRing.confidence * 100)}%</span>
                  </div>
                </div>
              )}

              {/* Element Minimap */}
              {gmail && gmail.elements.length > 0 && (
                <div className="element-list">
                  <h3 className="panel-title">Gmail Elements</h3>
                  <div className="el-scroll">
                    {gmail.elements.filter(e => e.type !== "other").slice(0, 15).map(e => (
                      <div key={e.id} className={`el-row ${gaze?.element?.id === e.id ? "el-focused" : ""}`}>
                        <span className={`el-type el-type-${e.type.split("_")[0]}`}>{e.type.replace("_", " ")}</span>
                        <span className="el-label">{e.label.slice(0, 60)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Startup */}
          {phase === "startup" && (
            <div className="startup">
              <div className="startup-icon">{"\u{1F9E0}"}</div>
              <h2>AXIOM.GMAIL</h2>
              <p>Initializing brain-computer interface...</p>
            </div>
          )}
        </div>

        {/* Action Feed */}
        <div className="action-panel">
          <h3 className="panel-title">Action Feed</h3>
          <div className="action-scroll">
            {actions.slice(-15).reverse().map((a, i) => (
              <div key={i} className={`action-entry ${a.action}`}>
                <div className="ae-head">
                  <span className="ae-action">{a.action}</span>
                  <span className="ae-conf">{Math.round(a.confidence * 100)}%</span>
                </div>
                <div className="ae-target">{a.target?.slice(0, 50)}</div>
                <div className="ae-reason">{a.reason?.slice(0, 80)}</div>
                {a.result && <div className="ae-result">{a.result}</div>}
              </div>
            ))}
            {actions.length === 0 && <p className="no-actions">No actions yet</p>}
          </div>

          <div className="brain-divider" />
          <h3 className="panel-title">System Log</h3>
          <div className="log-scroll">
            {systemLog.slice(-10).reverse().map((msg, i) => (
              <div key={i} className="log-entry">{msg}</div>
            ))}
          </div>
        </div>
      </div>

      {/* ErrP Toast */}
      {errp && <div className="errp-toast">{errp}</div>}
    </div>
  );
}
