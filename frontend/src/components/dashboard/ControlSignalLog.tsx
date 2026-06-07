import type { SignalHistoryEntry, ControlSignal } from '../../stores/sessionStore';

interface ControlSignalLogProps {
  history: SignalHistoryEntry[];
  lastSignal: ControlSignal | null;
  lastTimestamp: number | null;
}

const SIGNAL_NAMES: Record<ControlSignal, string> = {
  double_blink: 'Double Blink',
  triple_blink: 'Triple Blink',
  jaw_clench: 'Jaw Clench',
  long_jaw_clench: 'Long Jaw Clench',
};

export function ControlSignalLog({ history, lastSignal, lastTimestamp }: ControlSignalLogProps) {
  // Count signals by type
  const counts = history.reduce(
    (acc, entry) => {
      acc[entry.signal] = (acc[entry.signal] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  // Calculate time since last signal
  const timeSinceLast = lastTimestamp ? Math.round((Date.now() - lastTimestamp) / 1000) : null;

  const signalTypes: ControlSignal[] = ['double_blink', 'triple_blink', 'jaw_clench', 'long_jaw_clench'];

  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-6">
      <h3 className="text-label-lg text-on-surface-variant uppercase tracking-widest mb-4">
        Control Signals
      </h3>

      {/* Signal counts */}
      <div className="space-y-2 mb-4">
        {signalTypes.map((signal) => {
          const count = counts[signal] || 0;
          const isActive = lastSignal === signal && timeSinceLast !== null && timeSinceLast < 2;

          return (
            <div
              key={signal}
              className={`flex items-center justify-between px-3 py-2 rounded transition-colors ${
                isActive ? 'bg-primary text-on-primary' : 'bg-surface-container'
              }`}
            >
              <span className={`text-body-md ${isActive ? 'font-semibold' : ''}`}>
                {SIGNAL_NAMES[signal]}
              </span>
              <span className="text-label-lg font-mono">{count}</span>
            </div>
          );
        })}
      </div>

      {/* Last signal */}
      <div className="pt-4 border-t border-outline-variant">
        <div className="text-label-sm text-on-surface-variant mb-1">Last Signal</div>
        {lastSignal && timeSinceLast !== null ? (
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary">bolt</span>
            <span className="text-body-md text-on-surface">{SIGNAL_NAMES[lastSignal]}</span>
            <span className="text-label-sm text-on-surface-variant ml-auto">
              {timeSinceLast}s ago
            </span>
          </div>
        ) : (
          <div className="text-body-md text-on-surface-variant">None detected</div>
        )}
      </div>
    </div>
  );
}
