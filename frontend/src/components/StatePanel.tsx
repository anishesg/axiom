import type { RLState } from "../hooks/useMuseStream";

interface Props {
  state: RLState | null;
}

function Metric({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "4px 0" }}>
      <span style={{ fontSize: 10, color: "#5a5a7a" }}>{label}</span>
      <span style={{ fontSize: 11, color, fontWeight: 600 }}>{value}</span>
    </div>
  );
}

function QualityDot({ name, value }: { name: string; value: number }) {
  const color = value >= 0.6 ? "#00e68a" : value >= 0.3 ? "#f7a84d" : "#f74d6a";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ width: 6, height: 6, borderRadius: "50%", background: color, boxShadow: `0 0 6px ${color}60` }} />
      <span style={{ fontSize: 10, color: "#6a6a8a" }}>{name}</span>
    </div>
  );
}

export function StatePanel({ state }: Props) {
  if (!state) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#3a3a5a", fontSize: 11 }}>
        Waiting for state...
      </div>
    );
  }

  const r = state.band_ratios;
  const focusLevel = r.beta_alpha > 2.0 ? "HIGH" : r.beta_alpha > 1.0 ? "MED" : "LOW";
  const focusColor = r.beta_alpha > 2.0 ? "#00e68a" : r.beta_alpha > 1.0 ? "#f7a84d" : "#f74d6a";

  const relaxLevel = r.alpha_theta > 1.5 ? "HIGH" : r.alpha_theta > 0.8 ? "MED" : "LOW";
  const relaxColor = r.alpha_theta > 1.5 ? "#00e68a" : r.alpha_theta > 0.8 ? "#f7a84d" : "#f74d6a";

  const valence = state.asymmetry > 0.1 ? "POSITIVE" : state.asymmetry < -0.1 ? "NEGATIVE" : "NEUTRAL";
  const valColor = state.asymmetry > 0.1 ? "#00e68a" : state.asymmetry < -0.1 ? "#f74d6a" : "#6a6a8a";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <span style={{ fontSize: 11, color: "#4a4a6a", letterSpacing: 2 }}>RL STATE</span>

      <div style={{ borderBottom: "1px solid #1e1e3a", paddingBottom: 10 }}>
        <span style={{ fontSize: 9, color: "#4a4a6a", letterSpacing: 1 }}>COGNITIVE</span>
        <Metric label="Focus (β/α)" value={`${r.beta_alpha.toFixed(2)} · ${focusLevel}`} color={focusColor} />
        <Metric label="Relax (α/θ)" value={`${r.alpha_theta.toFixed(2)} · ${relaxLevel}`} color={relaxColor} />
        <Metric label="Valence" value={`${state.asymmetry.toFixed(3)} · ${valence}`} color={valColor} />
      </div>

      <div style={{ borderBottom: "1px solid #1e1e3a", paddingBottom: 10 }}>
        <span style={{ fontSize: 9, color: "#4a4a6a", letterSpacing: 1 }}>COHERENCE</span>
        {Object.entries(state.coherence).map(([pair, val]) => (
          <Metric key={pair} label={pair} value={val.toFixed(3)} color={val > 0.5 ? "#4d8ef7" : "#3a3a5a"} />
        ))}
      </div>

      <div>
        <span style={{ fontSize: 9, color: "#4a4a6a", letterSpacing: 1 }}>SIGNAL</span>
        <div style={{ display: "flex", gap: 12, marginTop: 6, flexWrap: "wrap" }}>
          {Object.entries(state.quality).map(([ch, val]) => (
            <QualityDot key={ch} name={ch} value={val} />
          ))}
        </div>
      </div>

      <div style={{ marginTop: "auto", fontSize: 9, color: "#2a2a4a" }}>
        dim={state.state.length} · {state.totalSamples.toLocaleString()} samples
      </div>
    </div>
  );
}
