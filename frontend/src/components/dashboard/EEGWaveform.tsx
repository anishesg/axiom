import { useRef, useEffect } from 'react';
import { useSessionStore } from '../../stores/sessionStore';

const CHANNEL_LABELS = ['TP9', 'AF7', 'AF8', 'TP10'];
const CHANNEL_COLORS = [
  '#FFFFFF', // TP9 - White (primary)
  '#A0A0A0', // AF7 - Gray
  '#A0A0A0', // AF8 - Gray
  '#FFFFFF', // TP10 - White (primary)
];

export function EEGWaveform() {
  const rawEEG = useSessionStore((state) => state.rawEEG);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Get device pixel ratio for sharp rendering
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    // Set canvas size accounting for device pixel ratio
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.fillStyle = '#0A0A0A';
    ctx.fillRect(0, 0, rect.width, rect.height);

    // If no data, show placeholder
    if (rawEEG.length === 0 || rawEEG[0]?.length === 0) {
      ctx.fillStyle = '#666666';
      ctx.font = '14px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for EEG data...', rect.width / 2, rect.height / 2);
      return;
    }

    const numChannels = Math.min(rawEEG.length, 4);
    const channelHeight = rect.height / numChannels;
    const padding = 10;

    // Draw each channel
    for (let ch = 0; ch < numChannels; ch++) {
      const channelData = rawEEG[ch];
      if (!channelData || channelData.length === 0) continue;

      const yCenter = channelHeight * ch + channelHeight / 2;

      // Draw channel separator line
      if (ch > 0) {
        ctx.strokeStyle = '#333333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, channelHeight * ch);
        ctx.lineTo(rect.width, channelHeight * ch);
        ctx.stroke();
      }

      // Normalize data for display
      const min = Math.min(...channelData);
      const max = Math.max(...channelData);
      const range = max - min || 1;

      // Draw waveform
      ctx.strokeStyle = CHANNEL_COLORS[ch];
      ctx.lineWidth = 1.5;
      ctx.beginPath();

      const xStep = (rect.width - padding * 2) / (channelData.length - 1);

      for (let i = 0; i < channelData.length; i++) {
        const x = padding + i * xStep;
        // Normalize to -0.5 to 0.5 range, then scale to channel height
        const normalized = (channelData[i] - min) / range - 0.5;
        const y = yCenter - normalized * (channelHeight - padding * 2);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Draw channel label
      ctx.fillStyle = '#666666';
      ctx.font = '11px system-ui, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(CHANNEL_LABELS[ch], 4, channelHeight * ch + 14);
    }
  }, [rawEEG]);

  return (
    <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-6">
      <h3 className="text-label-lg text-on-surface-variant uppercase tracking-widest mb-4">
        Raw EEG
      </h3>
      <div className="relative">
        <canvas
          ref={canvasRef}
          className="w-full h-96 rounded bg-black"
          style={{ imageRendering: 'pixelated' }}
        />
      </div>
      <div className="flex justify-between mt-3 text-label-sm text-on-surface-variant">
        {CHANNEL_LABELS.map((label, i) => (
          <span key={label} className="flex items-center gap-1">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: CHANNEL_COLORS[i] }}
            />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
