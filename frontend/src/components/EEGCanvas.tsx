import { useRef, useEffect, useCallback } from "react";
import type { EEGFrame } from "../hooks/useMuseStream";

const COLORS = ["#4d8ef7", "#f74d6a", "#00e68a", "#a855f7"];
const NAMES = ["TP9", "AF7", "AF8", "TP10"];
const BUFFER_LEN = 512;

interface Props {
  frame: EEGFrame | null;
}

export function EEGCanvas({ frame }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bufRef = useRef<number[][]>([[], [], [], []]);

  // Append incoming data to ring buffers
  useEffect(() => {
    if (!frame?.data) return;
    const buf = bufRef.current;
    for (let ch = 0; ch < 4; ch++) {
      const incoming = frame.data[ch] ?? [];
      buf[ch] = [...buf[ch], ...incoming].slice(-BUFFER_LEN);
    }
  }, [frame]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const parent = canvas.parentElement;
    if (!parent) return;

    const dpr = window.devicePixelRatio || 1;
    const W = parent.clientWidth;
    const H = parent.clientHeight;

    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      canvas.style.width = W + "px";
      canvas.style.height = H + "px";
    }

    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const laneH = H / 4;
    const padL = 56;

    // Clear
    ctx.fillStyle = "#0d0d1a";
    ctx.fillRect(0, 0, W, H);

    for (let ch = 0; ch < 4; ch++) {
      const yMid = laneH * ch + laneH / 2;
      const data = bufRef.current[ch];

      // Divider
      if (ch > 0) {
        ctx.strokeStyle = "#1a1a35";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, laneH * ch);
        ctx.lineTo(W, laneH * ch);
        ctx.stroke();
      }

      // Label
      ctx.fillStyle = COLORS[ch];
      ctx.font = "bold 12px 'JetBrains Mono', monospace";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(NAMES[ch], 8, yMid);

      if (data.length < 4) continue;

      // Scale based on standard deviation
      const n = data.length;
      let sum = 0;
      for (let i = 0; i < n; i++) sum += data[i];
      const mean = sum / n;

      let sqSum = 0;
      for (let i = 0; i < n; i++) sqSum += (data[i] - mean) ** 2;
      const std = Math.sqrt(sqSum / n) || 50;

      const amplitude = laneH * 0.38;
      const scale = amplitude / (std * 2.5);

      // Waveform
      const drawW = W - padL;
      const xStep = drawW / BUFFER_LEN;
      const xOff = BUFFER_LEN - n;

      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = padL + (xOff + i) * xStep;
        const y = yMid - (data[i] - mean) * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      // Glow
      ctx.strokeStyle = COLORS[ch] + "20";
      ctx.lineWidth = 6;
      ctx.lineJoin = "round";
      ctx.stroke();

      // Main line
      ctx.strokeStyle = COLORS[ch];
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, []);

  // Animation loop
  useEffect(() => {
    let id: number;
    const loop = () => {
      draw();
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      style={{ display: "block", width: "100%", height: "100%" }}
    />
  );
}
