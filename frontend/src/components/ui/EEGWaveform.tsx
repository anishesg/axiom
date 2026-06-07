import { useEffect, useRef, useMemo } from 'react';

interface EEGWaveformProps {
  data: number[][]; // 4 channels x N samples
  height?: number;
  showLabels?: boolean;
  className?: string;
  animate?: boolean;
}

const CHANNEL_LABELS = ['TP9', 'AF7', 'AF8', 'TP10'];
const CHANNEL_COLORS = [
  'rgba(0, 0, 0, 0.8)',      // TP9 - black
  'rgba(59, 130, 246, 0.8)', // AF7 - blue
  'rgba(16, 185, 129, 0.8)', // AF8 - green
  'rgba(139, 92, 246, 0.8)', // TP10 - purple
];

export function EEGWaveform({
  data,
  height = 200,
  showLabels = true,
  className = '',
  animate = true,
}: EEGWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const offsetRef = useRef(0);

  // Calculate dimensions
  const channelHeight = height / 4;
  const padding = { left: showLabels ? 50 : 10, right: 10, top: 5, bottom: 5 };

  // Normalize data for each channel
  const normalizedData = useMemo(() => {
    if (!data || data.length === 0) return [];

    return data.map((channel) => {
      if (!channel || channel.length === 0) return [];

      // Find min/max for auto-scaling
      const validValues = channel.filter((v) => !isNaN(v) && isFinite(v));
      if (validValues.length === 0) return channel.map(() => 0.5);

      const min = Math.min(...validValues);
      const max = Math.max(...validValues);
      const range = max - min || 1;

      // Normalize to 0-1 range with some padding
      return channel.map((v) => {
        if (isNaN(v) || !isFinite(v)) return 0.5;
        return 0.1 + 0.8 * ((v - min) / range);
      });
    });
  }, [data]);

  // Draw the waveforms
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size with device pixel ratio for sharpness
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const drawHeight = rect.height;

    const draw = () => {
      // Clear canvas
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, drawHeight);

      // Draw channel separators
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        const y = i * channelHeight;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
      }

      // Draw grid lines
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.setLineDash([2, 2]);
      const gridSpacing = 50;
      for (let x = padding.left; x < width - padding.right; x += gridSpacing) {
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, drawHeight - padding.bottom);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // Draw channel labels
      if (showLabels) {
        ctx.font = '11px Inter, system-ui, sans-serif';
        ctx.textBaseline = 'middle';
        for (let i = 0; i < 4; i++) {
          const y = i * channelHeight + channelHeight / 2;
          ctx.fillStyle = CHANNEL_COLORS[i];
          ctx.fillText(CHANNEL_LABELS[i], 8, y);
        }
      }

      // Draw waveforms
      const drawWidth = width - padding.left - padding.right;

      normalizedData.forEach((channel, channelIndex) => {
        if (!channel || channel.length === 0) return;

        const yOffset = channelIndex * channelHeight;
        const centerY = yOffset + channelHeight / 2;

        ctx.strokeStyle = CHANNEL_COLORS[channelIndex];
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        // Apply animation offset for scrolling effect
        const offset = animate ? Math.floor(offsetRef.current) % channel.length : 0;

        for (let i = 0; i < channel.length; i++) {
          const dataIndex = (i + offset) % channel.length;
          const x = padding.left + (i / channel.length) * drawWidth;
          const normalizedValue = channel[dataIndex];
          const y = yOffset + padding.top + normalizedValue * (channelHeight - padding.top - padding.bottom);

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        ctx.stroke();
      });

      // Update offset for animation
      if (animate) {
        offsetRef.current += 2;
        animationRef.current = requestAnimationFrame(draw);
      }
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [normalizedData, channelHeight, padding, showLabels, animate]);

  // Handle no data state
  if (!data || data.length === 0 || data.every((ch) => ch.length === 0)) {
    return (
      <div
        className={`flex items-center justify-center bg-surface-container-lowest border border-outline-variant rounded-lg ${className}`}
        style={{ height }}
      >
        <div className="text-center text-on-surface-variant">
          <span className="material-symbols-outlined text-4xl opacity-30 block mb-2">
            show_chart
          </span>
          <p className="text-label-sm">Waiting for EEG data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      <canvas
        ref={canvasRef}
        className="w-full rounded-lg border border-outline-variant"
        style={{ height }}
      />
      {/* Legend */}
      {showLabels && (
        <div className="absolute top-2 right-2 flex gap-3 bg-white/80 px-2 py-1 rounded text-xs">
          {CHANNEL_LABELS.map((label, i) => (
            <div key={label} className="flex items-center gap-1">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: CHANNEL_COLORS[i] }}
              />
              <span className="text-on-surface-variant">{label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Static waveform for displaying a snapshot (no animation)
export function EEGWaveformSnapshot({
  data,
  height = 150,
  showLabels = false,
  className = '',
}: Omit<EEGWaveformProps, 'animate'>) {
  return (
    <EEGWaveform
      data={data}
      height={height}
      showLabels={showLabels}
      className={className}
      animate={false}
    />
  );
}
