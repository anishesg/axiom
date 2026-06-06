import type { BrainState } from "../hooks/useMuseStream";

const STATES = [
  { key: "engagement", label: "Engagement", color: "#4d8ef7" },
  { key: "focus", label: "Focus", color: "#00e68a" },
  { key: "relaxation", label: "Relaxation", color: "#a855f7" },
  { key: "cognitive_load", label: "Load", color: "#f74d6a" },
  { key: "valence", label: "Valence", color: "#fbbf24" },
] as const;

export function BrainStatePanel({ state }: { state: BrainState | null }) {
  if (!state) {
    return (
      <div style={{ color: "#4a4a6a", fontSize: 12 }}>Waiting for brain state...</div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <span style={{ fontSize: 11, color: "#4a4a6a", letterSpacing: 2 }}>
        BRAIN STATE
      </span>
      {STATES.map(({ key, label, color }) => {
        const val = state[key] as number;
        return (
          <div key={key}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 11,
                color: "#8a8aa0",
                marginBottom: 2,
              }}
            >
              <span>{label}</span>
              <span style={{ color }}>{(val * 100).toFixed(0)}%</span>
            </div>
            <div
              style={{
                height: 6,
                background: "#1a1a30",
                borderRadius: 3,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${val * 100}%`,
                  height: "100%",
                  background: `linear-gradient(90deg, ${color}40, ${color})`,
                  borderRadius: 3,
                  transition: "width 0.2s ease",
                  boxShadow: `0 0 8px ${color}60`,
                }}
              />
            </div>
          </div>
        );
      })}

      {/* Signals */}
      <div
        style={{
          display: "flex",
          gap: 12,
          marginTop: 4,
          fontSize: 11,
        }}
      >
        <Signal label="Clench" active={state.jaw_clench} color="#f74d6a" />
        <Signal label="2x Clench" active={state.double_clench} color="#ff0055" />
        <Signal
          label="Ctx Switch"
          active={state.context_switch > 0.3}
          color="#fbbf24"
        />
        <Signal
          label="Error"
          active={state.error_response > 0.3}
          color="#ff6b35"
        />
      </div>
    </div>
  );
}

function Signal({
  label,
  active,
  color,
}: {
  label: string;
  active: boolean;
  color: string;
}) {
  return (
    <span
      style={{
        color: active ? color : "#2a2a40",
        fontWeight: active ? 700 : 400,
        transition: "color 0.15s",
      }}
    >
      ● {label}
    </span>
  );
}
