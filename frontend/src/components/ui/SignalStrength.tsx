interface SignalStrengthProps {
  level: number; // Current signal level
  maxLevel?: number; // Maximum level (default 4)
  showLabel?: boolean;
}

export function SignalStrength({ level, maxLevel = 4, showLabel = false }: SignalStrengthProps) {
  const bars = Array.from({ length: maxLevel }, (_, i) => i + 1);

  return (
    <div className="flex items-center gap-2">
      {showLabel && (
        <span className="text-label-sm text-on-surface-variant">SIGNAL</span>
      )}
      <div className="flex gap-[2px] items-end h-4">
        {bars.map((bar) => (
          <div
            key={bar}
            className={`w-1 transition-colors ${
              bar <= level ? 'bg-primary' : 'bg-surface-container-highest'
            }`}
            style={{ height: `${(bar / maxLevel) * 100}%` }}
          />
        ))}
      </div>
    </div>
  );
}
