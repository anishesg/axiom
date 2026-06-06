import { useState, useEffect, useRef, useCallback } from "react";

interface Agent {
  id: string;
  name: string;
  emoji: string;
  color: string;
}
interface BrainState {
  engagement: number;
  focus: number;
  valence: number;
  cognitive_load: number;
  relaxation: number;
}
interface AgentResponse extends Agent {
  text: string;
}
interface Reward {
  agent_id: string;
  engagement: number;
  valence: number;
  focus: number;
  cognitive_load: number;
  relaxation: number;
  total: number;
  n_samples: number;
}
interface RoundResult {
  round: number;
  winner: string | null;
  rewards: Record<string, number>;
  topic?: string;
}

const ZERO_BRAIN: BrainState = {
  engagement: 0,
  focus: 0,
  valence: 0.5,
  cognitive_load: 0,
  relaxation: 0,
};

const BRAIN_METRICS: {
  key: keyof BrainState;
  label: string;
  color: string;
}[] = [
  { key: "engagement", label: "Engagement", color: "#00d4ff" },
  { key: "focus", label: "Focus", color: "#a855f7" },
  { key: "valence", label: "Valence", color: "#ff6b9d" },
  { key: "cognitive_load", label: "Cog Load", color: "#ffaa00" },
  { key: "relaxation", label: "Relaxation", color: "#00ff88" },
];

export default function App() {
  const [connected, setConnected] = useState(false);
  const [phase, setPhase] = useState("idle");
  const [brain, setBrain] = useState<BrainState>(ZERO_BRAIN);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [responses, setResponses] = useState<AgentResponse[]>([]);
  const [readingTarget, setReadingTarget] = useState<string | null>(null);
  const [readingIdx, setReadingIdx] = useState(0);
  const [readingTotal, setReadingTotal] = useState(0);
  const [readingProgress, setReadingProgress] = useState(0);
  const [readingRemaining, setReadingRemaining] = useState(0);
  const [rewards, setRewards] = useState<Reward[]>([]);
  const [winner, setWinner] = useState<string | null>(null);
  const [round, setRound] = useState(0);
  const [maxRounds, setMaxRounds] = useState(5);
  const [history, setHistory] = useState<RoundResult[]>([]);
  const [topics, setTopics] = useState<string[]>([]);
  const [topicInput, setTopicInput] = useState("");
  const [calProgress, setCalProgress] = useState(0);
  const [adaptations, setAdaptations] = useState<Record<string, string>>({});
  const [agentStats, setAgentStats] = useState<
    Record<string, { wins: number; rounds: number; avg_reward: number }>
  >({});
  const [simMode, setSimMode] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof setTimeout>;
    function tryConnect() {
      const ws = new WebSocket("ws://127.0.0.1:8080");
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        reconnectTimer = setTimeout(tryConnect, 3000);
      };
      ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        switch (d.type) {
          case "init":
            setAgents(d.agents);
            setMaxRounds(d.max_rounds);
            setRound(d.round);
            setSimMode(d.sim_mode);
            if (d.topics) setTopics(d.topics);
            if (d.history) setHistory(d.history);
            if (d.phase && d.phase !== "idle") setPhase(d.phase);
            break;
          case "phase":
            setPhase(d.phase);
            if (d.round) setRound(d.round);
            if (d.history) setHistory(d.history);
            if (d.agent_stats) setAgentStats(d.agent_stats);
            break;
          case "brain":
            setBrain({
              engagement: d.engagement,
              focus: d.focus,
              valence: d.valence,
              cognitive_load: d.cognitive_load,
              relaxation: d.relaxation,
            });
            break;
          case "responses":
            setResponses(d.responses);
            if (d.round) setRound(d.round);
            break;
          case "reading_target":
            setReadingTarget(d.agent_id);
            setReadingIdx(d.agent_idx);
            setReadingTotal(d.total_agents);
            setReadingProgress(0);
            setReadingRemaining(d.duration);
            break;
          case "reading_progress":
            setReadingProgress(d.progress);
            setReadingRemaining(d.remaining);
            break;
          case "rewards":
            setRewards(d.rewards);
            setWinner(d.winner);
            setPhase("results");
            break;
          case "round_complete":
            setAdaptations(d.adaptations || {});
            if (d.history) setHistory(d.history);
            if (d.agent_stats) setAgentStats(d.agent_stats);
            break;
          case "calibration_progress":
            setCalProgress(d.progress);
            break;
          case "error":
            setError(d.message);
            setTimeout(() => setError(null), 5000);
            break;
        }
      };
    }
    tryConnect();
    return () => {
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, []);

  const readingResp = responses.find((r) => r.id === readingTarget);
  const showBrain = phase !== "idle" && phase !== "complete";

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1 className="title">
            NEURAL<span className="title-accent">RLHF</span>
          </h1>
          <span className="subtitle">Your Brain as the Reward Model</span>
        </div>
        <div className="header-right">
          {round > 0 && (
            <span className="round-badge">
              ROUND {round}/{maxRounds}
            </span>
          )}
          <span className={`conn-dot ${connected ? "on" : "off"}`} />
          <span className="conn-label">
            {connected ? (simMode ? "SIM" : "LIVE") : "OFFLINE"}
          </span>
        </div>
      </header>

      <div className="main">
        {showBrain && (
          <div className="brain-panel">
            {BRAIN_METRICS.map(({ key, label, color }) => (
              <div key={key} className="brain-metric">
                <div className="bm-head">
                  <span className="bm-label">{label}</span>
                  <span className="bm-val" style={{ color }}>
                    {Math.round(brain[key] * 100)}%
                  </span>
                </div>
                <div className="bm-track">
                  <div
                    className="bm-fill"
                    style={{
                      width: `${brain[key] * 100}%`,
                      background: color,
                      boxShadow: `0 0 10px ${color}50`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="content">
          {phase === "idle" && (
            <div className="idle">
              <div className="idle-icon">{"\u{1F9E0}"}</div>
              <h2>NEURAL RLHF</h2>
              <p>
                Put on your Muse S headband. Three AI agents will compete to
                explain topics &mdash; your brain&apos;s neural response picks
                the winner. No clicking, no typing. Just thinking.
              </p>
              <div className="idle-btns">
                <button
                  className="btn btn-p"
                  onClick={() => send({ command: "start_calibration" })}
                >
                  Calibrate Baseline
                </button>
                <button
                  className="btn btn-s"
                  onClick={() => send({ command: "skip_calibration" })}
                >
                  Skip &rarr; Demo
                </button>
              </div>
            </div>
          )}

          {phase === "calibrating" && (
            <div className="calibrate">
              <div className="cal-ring-wrap">
                <svg className="cal-ring-bg" viewBox="0 0 140 140">
                  <circle cx="70" cy="70" r="60" />
                </svg>
                <svg className="cal-ring-fg" viewBox="0 0 140 140">
                  <circle
                    cx="70"
                    cy="70"
                    r="60"
                    strokeDasharray={`${2 * Math.PI * 60}`}
                    strokeDashoffset={`${
                      2 * Math.PI * 60 * (1 - calProgress)
                    }`}
                  />
                </svg>
                <span className="cal-pct">
                  {Math.round(calProgress * 100)}%
                </span>
              </div>
              <h2>Calibrating Baseline</h2>
              <p>Relax and breathe normally. Recording resting brain state.</p>
            </div>
          )}

          {phase === "ready" && (
            <div className="ready">
              <h2>{round === 0 ? "Choose a Topic" : `Round ${round + 1}`}</h2>
              <p>
                Three AI agents will compete to explain this. Your brain picks
                the winner.
              </p>
              <div className="topic-row">
                <input
                  className="topic-input"
                  value={topicInput}
                  onChange={(e) => setTopicInput(e.target.value)}
                  placeholder="Type a topic or pick one below..."
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && topicInput.trim())
                      send({ command: "start_round", topic: topicInput });
                  }}
                />
                <button
                  className="btn btn-p"
                  disabled={!topicInput.trim()}
                  onClick={() =>
                    send({ command: "start_round", topic: topicInput })
                  }
                >
                  Go
                </button>
              </div>
              <div className="topic-presets">
                {topics.map((t, i) => (
                  <button
                    key={i}
                    className="topic-btn"
                    onClick={() => {
                      setTopicInput(t);
                      send({ command: "start_round", topic: t });
                    }}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>
          )}

          {phase === "generating" && (
            <div className="generating">
              <h2>Agents are thinking...</h2>
              <div className="gen-agents">
                {agents.map((a) => (
                  <div
                    key={a.id}
                    className="gen-agent"
                    style={{ borderColor: a.color }}
                  >
                    <span className="gen-emoji">{a.emoji}</span>
                    <span className="gen-name">{a.name}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {phase === "reading" && readingResp && (
            <div className="reading">
              <div className="reading-head">
                <span className="reading-label">READING</span>
                <span
                  className="reading-agent"
                  style={{ color: readingResp.color }}
                >
                  {readingResp.emoji} {readingResp.name}
                </span>
                <span className="reading-count">
                  {readingIdx + 1} of {readingTotal}
                </span>
              </div>
              <div
                className="reading-card"
                style={{
                  borderColor: readingResp.color,
                  boxShadow: `0 0 30px ${readingResp.color}15`,
                }}
              >
                {readingResp.text}
              </div>
              <div className="timer-track">
                <div
                  className="timer-fill"
                  style={{
                    width: `${readingProgress * 100}%`,
                    background: readingResp.color,
                    boxShadow: `0 0 8px ${readingResp.color}60`,
                  }}
                />
              </div>
              <div className="timer-row">
                <span className="timer-text">
                  {Math.ceil(readingRemaining)}s remaining
                </span>
                <button
                  className="btn btn-s btn-sm"
                  onClick={() => send({ command: "next_response" })}
                >
                  Skip &rarr;
                </button>
              </div>
            </div>
          )}

          {phase === "scoring" && (
            <div className="scoring">
              <h2>Computing Neural Rewards...</h2>
              <div className="scoring-anim" />
            </div>
          )}

          {(phase === "results" || phase === "adapting") && (
            <div className="results">
              <h2>Round {round} Results</h2>
              <div className="reward-bars">
                {rewards.map((r) => {
                  const agent = agents.find((a) => a.id === r.agent_id);
                  const isW = r.agent_id === winner;
                  return (
                    <div
                      key={r.agent_id}
                      className={`reward-row${isW ? " winner" : ""}`}
                    >
                      <div className="rw-agent">
                        <span className="rw-emoji">{agent?.emoji}</span>
                        <span className="rw-name">{agent?.name}</span>
                        {isW && (
                          <span className="winner-badge">
                            {"★"} WINNER
                          </span>
                        )}
                      </div>
                      <div className="rw-track">
                        <div
                          className="rw-fill"
                          style={{
                            width: `${r.total * 100}%`,
                            background: agent?.color,
                            boxShadow: isW
                              ? `0 0 16px ${agent?.color}50`
                              : "none",
                          }}
                        />
                      </div>
                      <span className="rw-score">
                        {Math.round(r.total * 100)}
                      </span>
                    </div>
                  );
                })}
              </div>

              {Object.keys(adaptations).length > 0 && (
                <div className="adaptations">
                  <h3>Neural Feedback &rarr; Agent Adaptation</h3>
                  {Object.entries(adaptations).map(([aid, text]) => {
                    const ag = agents.find((a) => a.id === aid);
                    return (
                      <div key={aid} className="adapt-row">
                        <span
                          className="adapt-agent"
                          style={{ color: ag?.color }}
                        >
                          {ag?.emoji} {ag?.name}:
                        </span>{" "}
                        <span className="adapt-text">{text}</span>
                      </div>
                    );
                  })}
                </div>
              )}

              {round < maxRounds && (
                <button
                  className="btn btn-p"
                  onClick={() => setPhase("ready")}
                >
                  Next Round &rarr;
                </button>
              )}
              {round >= maxRounds && (
                <button
                  className="btn btn-p"
                  onClick={() => setPhase("complete")}
                >
                  View Summary
                </button>
              )}
            </div>
          )}

          {phase === "complete" && (
            <div className="complete">
              <h2>SESSION COMPLETE</h2>
              <p>
                Over {history.length} rounds, your brain guided agents toward
                your neural preferences.
              </p>
              {history.length > 0 && (
                <Chart
                  history={history}
                  agents={agents}
                  stats={agentStats}
                  full
                />
              )}
              <button
                className="btn btn-p"
                style={{ marginTop: 24 }}
                onClick={() => window.location.reload()}
              >
                New Session
              </button>
            </div>
          )}
        </div>

        {history.length > 0 && phase !== "complete" && phase !== "idle" && (
          <Chart history={history} agents={agents} stats={agentStats} />
        )}
      </div>

      {error && <div className="error-toast">{error}</div>}
    </div>
  );
}

function Chart({
  history,
  agents,
  stats,
  full,
}: {
  history: RoundResult[];
  agents: Agent[];
  stats: Record<
    string,
    { wins: number; rounds: number; avg_reward: number }
  >;
  full?: boolean;
}) {
  if (!history.length) return null;
  const w = full
    ? Math.max(history.length * 120, 240)
    : Math.max(history.length * 80, 160);
  const h = full ? 150 : 70;
  const px = 40;

  return (
    <div className={full ? "chart-box" : "mini-chart"}>
      {full ? <h3>Neural Reward Over Time</h3> : <h4>Reward History</h4>}
      <svg
        viewBox={`0 0 ${w} ${h + 20}`}
        className={full ? "chart-svg" : "mini-svg"}
        style={{ width: "100%", height: full ? 180 : 80 }}
      >
        {agents.map((agent) => {
          const pts = history.map((r, i) => ({
            x:
              px +
              (i * (w - px * 2)) / Math.max(history.length - 1, 1),
            y: h - (r.rewards[agent.id] ?? 0.5) * (h - 10) + 5,
          }));
          if (pts.length === 1) {
            return (
              <circle
                key={agent.id}
                cx={pts[0].x}
                cy={pts[0].y}
                r={full ? 5 : 3}
                fill={agent.color}
              />
            );
          }
          const d = pts
            .map((p, i) => `${i ? "L" : "M"} ${p.x} ${p.y}`)
            .join(" ");
          return (
            <g key={agent.id}>
              <path
                d={d}
                fill="none"
                stroke={agent.color}
                strokeWidth={full ? 2.5 : 1.5}
                opacity={0.85}
              />
              {pts.map((p, i) => (
                <circle
                  key={i}
                  cx={p.x}
                  cy={p.y}
                  r={full ? 4 : 3}
                  fill={agent.color}
                />
              ))}
            </g>
          );
        })}
        {full &&
          history.map((_, i) => (
            <text
              key={i}
              x={
                px +
                (i * (w - px * 2)) / Math.max(history.length - 1, 1)
              }
              y={h + 16}
              textAnchor="middle"
              fill="#5a6894"
              fontSize="11"
            >
              R{i + 1}
            </text>
          ))}
      </svg>
      {full && (
        <div className="chart-legend">
          {agents.map((a) => (
            <span key={a.id} className="legend-item">
              <span className="legend-dot" style={{ background: a.color }} />
              {a.emoji} {a.name}
              {stats[a.id] && (
                <span className="legend-wins">
                  ({stats[a.id].wins} wins)
                </span>
              )}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
