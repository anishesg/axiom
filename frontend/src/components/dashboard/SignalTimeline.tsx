import type { SignalHistoryEntry, ControlSignal } from '../../stores/sessionStore';

interface SignalTimelineProps {
  history: SignalHistoryEntry[];
  duration?: number; // Duration in seconds to show (default 30)
}

const SIGNAL_LABELS: Record<ControlSignal, { short: string; icon: string }> = {
  double_blink: { short: 'B2', icon: 'visibility' },
  triple_blink: { short: 'B3', icon: 'visibility' },
  jaw_clench: { short: 'JC', icon: 'sentiment_very_satisfied' },
  long_jaw_clench: { short: 'LJC', icon: 'sentiment_very_satisfied' },
};

export function SignalTimeline({ history, duration = 30 }: SignalTimelineProps) {
  const now = Date.now();
  const startTime = now - duration * 1000;

  // Filter events to the time window
  const visibleEvents = history.filter((e) => e.timestamp >= startTime);

  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-label-lg text-on-surface-variant uppercase tracking-widest">
          Signal Timeline
        </h3>
        <span className="text-label-sm text-on-surface-variant">
          Last {duration}s
        </span>
      </div>

      {/* Timeline */}
      <div className="relative h-16">
        {/* Background track */}
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-[2px] bg-surface-container-highest" />

        {/* Time markers */}
        <div className="absolute inset-x-0 bottom-0 flex justify-between text-[10px] text-on-surface-variant">
          <span>-{duration}s</span>
          <span>-{duration / 2}s</span>
          <span>Now</span>
        </div>

        {/* Event dots */}
        {visibleEvents.map((event) => {
          const elapsed = now - event.timestamp;
          const position = 100 - (elapsed / (duration * 1000)) * 100;
          const { short, icon } = SIGNAL_LABELS[event.signal];

          return (
            <div
              key={event.id}
              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 flex flex-col items-center"
              style={{ left: `${position}%` }}
            >
              <div className="w-6 h-6 rounded-full bg-primary flex items-center justify-center">
                <span className="material-symbols-outlined text-on-primary text-[14px]">
                  {icon}
                </span>
              </div>
              <span className="text-[10px] text-on-surface font-mono mt-1">{short}</span>
            </div>
          );
        })}

        {/* Empty state */}
        {visibleEvents.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center text-on-surface-variant text-body-md">
            No signals detected
          </div>
        )}
      </div>
    </div>
  );
}
