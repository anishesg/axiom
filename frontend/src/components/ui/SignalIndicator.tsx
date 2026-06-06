type SignalStatus = 'connected' | 'streaming' | 'disconnected' | 'ready' | 'error';

interface SignalIndicatorProps {
  status: SignalStatus;
  label?: string;
  size?: 'sm' | 'md';
}

const statusConfig: Record<SignalStatus, { pulse: boolean; color: string; defaultLabel: string }> = {
  connected: { pulse: true, color: 'bg-primary', defaultLabel: 'Connected' },
  streaming: { pulse: true, color: 'bg-primary', defaultLabel: 'Streaming' },
  ready: { pulse: true, color: 'bg-primary', defaultLabel: 'Ready' },
  disconnected: { pulse: false, color: 'bg-outline', defaultLabel: 'Disconnected' },
  error: { pulse: false, color: 'bg-error', defaultLabel: 'Error' },
};

export function SignalIndicator({
  status,
  label,
  size = 'md'
}: SignalIndicatorProps) {
  const config = statusConfig[status];
  const dotSize = size === 'sm' ? 'w-1.5 h-1.5' : 'w-2 h-2';

  return (
    <div className="flex items-center gap-4 py-3 px-6 border border-surface-variant rounded-full">
      <div
        className={`${dotSize} rounded-full ${config.color} ${config.pulse ? 'signal-pulse' : ''}`}
      />
      <span className="text-label-sm text-on-surface-variant uppercase tracking-widest">
        {label || config.defaultLabel}
      </span>
    </div>
  );
}
