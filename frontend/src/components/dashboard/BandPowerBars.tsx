import type { BandPowers } from '../../stores/sessionStore';

interface BandPowerBarsProps {
  bandPowers: BandPowers;
}

const BAND_INFO = [
  { key: 'delta' as const, label: 'Delta', range: '0.5-4 Hz', description: 'Deep sleep' },
  { key: 'theta' as const, label: 'Theta', range: '4-8 Hz', description: 'Relaxation' },
  { key: 'alpha' as const, label: 'Alpha', range: '8-13 Hz', description: 'Calm focus' },
  { key: 'beta' as const, label: 'Beta', range: '13-30 Hz', description: 'Active thinking' },
  { key: 'gamma' as const, label: 'Gamma', range: '30-45 Hz', description: 'High processing' },
];

export function BandPowerBars({ bandPowers }: BandPowerBarsProps) {
  // Normalize band powers to 0-1 range (assuming max power is around 1.0)
  const maxPower = Math.max(...Object.values(bandPowers), 0.01);

  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-6">
      <h3 className="text-label-lg text-on-surface-variant uppercase tracking-widest mb-4">
        Band Powers
      </h3>
      <div className="flex items-end justify-between gap-2 h-32">
        {BAND_INFO.map(({ key, label, range }) => {
          const value = bandPowers[key];
          const normalizedHeight = (value / maxPower) * 100;

          return (
            <div key={key} className="flex-1 flex flex-col items-center gap-2">
              <div className="w-full h-24 bg-surface-container-highest rounded relative flex items-end">
                <div
                  className="w-full bg-primary rounded transition-all duration-300"
                  style={{ height: `${Math.max(normalizedHeight, 2)}%` }}
                />
              </div>
              <div className="text-center">
                <div className="text-label-sm text-on-surface font-medium">{label}</div>
                <div className="text-[10px] text-on-surface-variant">{range}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
