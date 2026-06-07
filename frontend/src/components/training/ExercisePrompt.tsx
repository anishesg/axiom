import { useEffect, useState } from 'react';

export type PromptState = 'ready' | 'countdown' | 'action' | 'success' | 'failure' | 'waiting';

interface ExercisePromptProps {
  state: PromptState;
  actionText: string;
  countdown?: number;
  timeRemaining?: number;
  timeWindow?: number;
  icon?: string;
}

export function ExercisePrompt({
  state,
  actionText,
  countdown = 3,
  timeRemaining,
  timeWindow = 3,
  icon = 'visibility',
}: ExercisePromptProps) {
  const [displayCountdown, setDisplayCountdown] = useState(countdown);

  useEffect(() => {
    if (state === 'countdown') {
      setDisplayCountdown(countdown);
    }
  }, [state, countdown]);

  const progressPercent = timeRemaining !== undefined && timeWindow > 0
    ? (timeRemaining / timeWindow) * 100
    : 100;

  return (
    <div className="flex flex-col items-center justify-center p-8">
      {/* Main prompt area */}
      <div
        className={`w-64 h-64 rounded-full flex flex-col items-center justify-center transition-all duration-300 ${
          state === 'success'
            ? 'bg-primary text-on-primary scale-110'
            : state === 'failure'
              ? 'bg-error text-on-error scale-95'
              : state === 'action'
                ? 'bg-surface-container-lowest border-4 border-primary animate-pulse'
                : state === 'countdown'
                  ? 'bg-surface-container border-2 border-outline'
                  : 'bg-surface-container-low border border-outline-variant'
        }`}
      >
        {state === 'ready' && (
          <>
            <span className="material-symbols-outlined text-[48px] text-on-surface-variant mb-2">
              hourglass_empty
            </span>
            <span className="text-headline-md text-on-surface-variant">Ready</span>
          </>
        )}

        {state === 'countdown' && (
          <>
            <span className="text-display text-primary">{displayCountdown}</span>
            <span className="text-body-md text-on-surface-variant">Get ready...</span>
          </>
        )}

        {state === 'action' && (
          <>
            <span className="material-symbols-outlined text-[64px] text-primary mb-2">
              {icon}
            </span>
            <span className="text-headline-md text-primary font-bold">{actionText}</span>
            <span className="text-body-md text-on-surface-variant mt-1">NOW!</span>
          </>
        )}

        {state === 'success' && (
          <>
            <span className="material-symbols-outlined text-[64px]">check_circle</span>
            <span className="text-headline-md font-bold">Great!</span>
          </>
        )}

        {state === 'failure' && (
          <>
            <span className="material-symbols-outlined text-[64px]">cancel</span>
            <span className="text-headline-md font-bold">Missed</span>
          </>
        )}

        {state === 'waiting' && (
          <>
            <span className="material-symbols-outlined text-[48px] text-on-surface-variant animate-spin">
              sync
            </span>
            <span className="text-body-md text-on-surface-variant mt-2">Processing...</span>
          </>
        )}
      </div>

      {/* Time progress bar (only during action) */}
      {state === 'action' && timeRemaining !== undefined && (
        <div className="w-64 mt-6">
          <div className="h-2 bg-surface-container-highest rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-100"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          <div className="text-center text-label-sm text-on-surface-variant mt-1">
            {timeRemaining.toFixed(1)}s remaining
          </div>
        </div>
      )}
    </div>
  );
}
