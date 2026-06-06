import { useEffect, useRef } from "react";
import type { AgentAction, LearningMetrics } from "../hooks/useMuseStream";

const ACTION_COLORS: Record<string, string> = {
  click: "#00e68a",
  open: "#00e68a",
  switch_app: "#4d8ef7",
  undo: "#f74d6a",
  scroll_down: "#8a8aa0",
  scroll_up: "#8a8aa0",
  intent_ring_start: "#fbbf24",
  intent_ring_progress: "#fbbf24",
  intent_ring_cancel: "#4a4a6a",
  ghost_actions: "#a855f7",
  suggest_apps: "#4d8ef7",
  sleep: "#2a2a40",
  wake: "#00e68a",
  none: "#1a1a30",
};

export function ActionFeed({
  lastAction,
  learning,
}: {
  lastAction: AgentAction | null;
  learning: LearningMetrics | null;
}) {
  const feedRef = useRef<HTMLDivElement>(null);
  const actionsRef = useRef<AgentAction[]>([]);

  useEffect(() => {
    if (lastAction && lastAction.action_type !== "none") {
      actionsRef.current = [...actionsRef.current.slice(-19), lastAction];
      if (feedRef.current) {
        feedRef.current.scrollTop = feedRef.current.scrollHeight;
      }
    }
  }, [lastAction]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, height: "100%" }}>
      <span style={{ fontSize: 11, color: "#4a4a6a", letterSpacing: 2 }}>
        AGENT · {learning?.mode?.toUpperCase() ?? "—"}
      </span>

      {/* Stats row */}
      <div style={{ display: "flex", gap: 16, fontSize: 11, color: "#6a6a80" }}>
        <span>
          Actions:{" "}
          <span style={{ color: "#00e68a" }}>{learning?.total_actions ?? 0}</span>
        </span>
        <span>
          Accuracy:{" "}
          <span style={{ color: "#4d8ef7" }}>
            {learning ? `${(learning.accuracy * 100).toFixed(0)}%` : "—"}
          </span>
        </span>
        <span>
          Undone:{" "}
          <span style={{ color: "#f74d6a" }}>{learning?.undone_actions ?? 0}</span>
        </span>
      </div>

      {/* Accuracy curve */}
      {learning?.accuracy_history && learning.accuracy_history.length > 1 && (
        <AccuracyCurve data={learning.accuracy_history} />
      )}

      {/* Action feed */}
      <div
        ref={feedRef}
        style={{
          flex: 1,
          overflow: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 4,
          fontSize: 11,
        }}
      >
        {actionsRef.current.map((a, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              gap: 8,
              padding: "3px 0",
              borderBottom: "1px solid #1a1a30",
            }}
          >
            <span
              style={{
                color: ACTION_COLORS[a.action_type] ?? "#6a6a80",
                fontWeight: 600,
                minWidth: 80,
                textTransform: "uppercase",
                fontSize: 10,
              }}
            >
              {a.action_type.replace(/_/g, " ")}
            </span>
            <span style={{ color: "#8a8aa0", flex: 1 }}>
              {a.target || "—"}
            </span>
            <span style={{ color: "#4a4a6a" }}>
              {(a.confidence * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>

      {/* Demo description */}
      {learning?.description && (
        <div
          style={{
            padding: "6px 10px",
            background: "#1a1a30",
            borderRadius: 6,
            fontSize: 11,
            color: "#8a8aa0",
            fontStyle: "italic",
          }}
        >
          {learning.description}
        </div>
      )}
    </div>
  );
}

function AccuracyCurve({ data }: { data: number[] }) {
  const w = 200;
  const h = 40;
  const n = data.length;
  if (n < 2) return null;

  const points = data
    .map((v, i) => `${(i / (n - 1)) * w},${h - v * h}`)
    .join(" ");

  return (
    <svg width={w} height={h} style={{ flexShrink: 0 }}>
      <polyline
        points={points}
        fill="none"
        stroke="#4d8ef7"
        strokeWidth={1.5}
        strokeLinejoin="round"
      />
      <line x1={0} y1={h} x2={w} y2={h} stroke="#1a1a30" strokeWidth={0.5} />
    </svg>
  );
}
