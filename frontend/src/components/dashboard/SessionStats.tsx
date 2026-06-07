interface SessionStatsProps {
  sessionStartTime: number | null;
  totalSignals: number;
  accuracy?: number;
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

export function SessionStats({ sessionStartTime, totalSignals, accuracy }: SessionStatsProps) {
  const duration = sessionStartTime ? Date.now() - sessionStartTime : 0;

  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-6">
      <h3 className="text-label-lg text-on-surface-variant uppercase tracking-widest mb-4">
        Session Stats
      </h3>

      <div className="grid grid-cols-3 gap-4">
        {/* Duration */}
        <div className="text-center">
          <div className="text-headline-md text-primary font-mono">
            {sessionStartTime ? formatDuration(duration) : '--:--'}
          </div>
          <div className="text-label-sm text-on-surface-variant">Duration</div>
        </div>

        {/* Signals */}
        <div className="text-center">
          <div className="text-headline-md text-primary font-mono">{totalSignals}</div>
          <div className="text-label-sm text-on-surface-variant">Signals</div>
        </div>

        {/* Accuracy */}
        <div className="text-center">
          <div className="text-headline-md text-primary font-mono">
            {accuracy !== undefined ? `${Math.round(accuracy * 100)}%` : '--'}
          </div>
          <div className="text-label-sm text-on-surface-variant">Accuracy</div>
        </div>
      </div>
    </div>
  );
}
