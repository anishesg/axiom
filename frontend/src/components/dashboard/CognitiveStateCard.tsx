interface CognitiveStateCardProps {
  focus: number;
  relaxation: number;
  engagement: number;
}

export function CognitiveStateCard({ focus, relaxation, engagement }: CognitiveStateCardProps) {
  const metrics = [
    { label: 'Focus', value: focus, color: 'bg-primary' },
    { label: 'Relaxation', value: relaxation, color: 'bg-primary' },
    { label: 'Engagement', value: engagement, color: 'bg-primary' },
  ];

  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-6">
      <h3 className="text-label-lg text-on-surface-variant uppercase tracking-widest mb-4">
        Cognitive State
      </h3>
      <div className="space-y-4">
        {metrics.map(({ label, value, color }) => (
          <div key={label} className="space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-body-md text-on-surface">{label}</span>
              <span className="text-label-lg text-primary font-mono">
                {Math.round(value * 100)}%
              </span>
            </div>
            <div className="h-3 bg-surface-container-highest rounded-full overflow-hidden">
              <div
                className={`h-full ${color} transition-all duration-300 ease-out`}
                style={{ width: `${value * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
