interface Props {
  bands: Record<string, number> | null;
}

const BAND_CONFIG: { key: string; label: string; color: string }[] = [
  { key: "delta", label: "Delta", color: "#4d8ef7" },
  { key: "theta", label: "Theta", color: "#f74d6a" },
  { key: "alpha", label: "Alpha", color: "#00e68a" },
  { key: "beta", label: "Beta", color: "#a855f7" },
  { key: "gamma", label: "Gamma", color: "#f7a84d" },
];

export function BandBars({ bands }: Props) {
  const maxVal = bands
    ? Math.max(...Object.values(bands), 0.001)
    : 1;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%", justifyContent: "center" }}>
      {BAND_CONFIG.map(({ key, label, color }) => {
        const val = bands?.[key] ?? 0;
        const pct = Math.min((val / maxVal) * 100, 100);
        return (
          <div key={key} style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ width: 52, fontSize: 11, color: "#6a6a8a", textAlign: "right" }}>
              {label}
            </span>
            <div
              style={{
                flex: 1,
                height: 18,
                background: "#1a1a2e",
                borderRadius: 4,
                overflow: "hidden",
                position: "relative",
              }}
            >
              <div
                style={{
                  width: `${pct}%`,
                  height: "100%",
                  background: `linear-gradient(90deg, ${color}cc, ${color})`,
                  borderRadius: 4,
                  transition: "width 0.15s ease-out",
                  boxShadow: `0 0 12px ${color}40`,
                }}
              />
            </div>
            <span style={{ width: 48, fontSize: 10, color: "#6a6a8a", textAlign: "left" }}>
              {val > 0 ? val.toFixed(1) : "—"}
            </span>
          </div>
        );
      })}
    </div>
  );
}
