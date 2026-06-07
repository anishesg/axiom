export type AttemptResult = 'success' | 'failure' | 'pending';

interface AttemptTrackerProps {
  attempts: AttemptResult[];
  totalAttempts: number;
  currentAttempt: number;
}

export function AttemptTracker({ attempts, totalAttempts, currentAttempt }: AttemptTrackerProps) {
  const successCount = attempts.filter((a) => a === 'success').length;
  const completedCount = attempts.filter((a) => a !== 'pending').length;
  const score = completedCount > 0 ? Math.round((successCount / completedCount) * 100) : 0;

  return (
    <div className="space-y-4">
      {/* Attempt dots */}
      <div className="flex items-center justify-center gap-2 flex-wrap">
        {Array.from({ length: totalAttempts }).map((_, index) => {
          const result = attempts[index] || 'pending';
          const isCurrent = index === currentAttempt;

          return (
            <div
              key={index}
              className={`w-8 h-8 rounded-full flex items-center justify-center text-sm transition-all ${
                result === 'success'
                  ? 'bg-primary text-on-primary'
                  : result === 'failure'
                    ? 'bg-error text-on-error'
                    : isCurrent
                      ? 'border-2 border-primary bg-surface-container-lowest'
                      : 'bg-surface-container-highest text-on-surface-variant'
              } ${isCurrent ? 'scale-125 ring-2 ring-primary ring-offset-2' : ''}`}
            >
              {result === 'success' ? (
                <span className="material-symbols-outlined text-[16px]">check</span>
              ) : result === 'failure' ? (
                <span className="material-symbols-outlined text-[16px]">close</span>
              ) : (
                <span className="text-[12px]">{index + 1}</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Progress text */}
      <div className="flex items-center justify-between text-label-sm">
        <span className="text-on-surface-variant">
          Attempt {Math.min(currentAttempt + 1, totalAttempts)} of {totalAttempts}
        </span>
        <span className="text-primary font-semibold">Score: {score}%</span>
      </div>
    </div>
  );
}
