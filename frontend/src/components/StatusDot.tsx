import type { ConnectionStatus } from "../hooks/useMuseStream";

interface Props {
  status: ConnectionStatus;
  sampleCount: number;
}

const STATUS_COLOR: Record<ConnectionStatus, string> = {
  disconnected: "#f74d6a",
  connecting: "#f7a84d",
  connected: "#00e68a",
};

const STATUS_LABEL: Record<ConnectionStatus, string> = {
  disconnected: "DISCONNECTED",
  connecting: "CONNECTING...",
  connected: "STREAMING",
};

export function StatusDot({ status, sampleCount }: Props) {
  const color = STATUS_COLOR[status];
  const seconds = (sampleCount / 256).toFixed(0);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: color,
          boxShadow: `0 0 8px ${color}80`,
          animation: status === "connected" ? "pulse 2s infinite" : undefined,
        }}
      />
      <span style={{ fontSize: 12, color: "#6a6a8a", letterSpacing: 1 }}>
        {STATUS_LABEL[status]}
      </span>
      {status === "connected" && (
        <span style={{ fontSize: 11, color: "#4a4a6a", marginLeft: 8 }}>
          {Number(sampleCount).toLocaleString()} samples · {seconds}s
        </span>
      )}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
}
