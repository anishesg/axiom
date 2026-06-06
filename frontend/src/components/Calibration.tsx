import { useState, useEffect, useRef, useCallback } from "react";
import type { BrainState } from "../hooks/useMuseStream";

type Stage = "welcome" | "baseline_rest" | "baseline_focus" | "results" | "done";

interface CalibrationData {
  baselineAlpha: number;
  baselineBeta: number;
  focusScore: number;
  relaxScore: number;
}

export function Calibration({
  brainState,
  onComplete,
  sendCommand: _sendCommand,
}: {
  brainState: BrainState | null;
  onComplete: (data: CalibrationData) => void;
  sendCommand: (cmd: object) => void;
}) {
  const [stage, setStage] = useState<Stage>("welcome");
  const [countdown, setCountdown] = useState(3);
  const [focusHistory, setFocusHistory] = useState<number[]>([]);
  const [relaxHistory, setRelaxHistory] = useState<number[]>([]);
  const [progress, setProgress] = useState(0);
  const [score, setScore] = useState(0);
  const stageStartRef = useRef(Date.now());

  const nextStage = useCallback((s: Stage) => {
    setStage(s);
    stageStartRef.current = Date.now();
    setProgress(0);
  }, []);

  useEffect(() => {
    if (stage === "baseline_rest" || stage === "baseline_focus") {
      const duration = stage === "baseline_rest" ? 10 : 8;
      const interval = setInterval(() => {
        const elapsed = (Date.now() - stageStartRef.current) / 1000;
        setProgress(Math.min(1, elapsed / duration));
        setCountdown(Math.max(0, Math.ceil(duration - elapsed)));
        if (elapsed >= duration) {
          clearInterval(interval);
          if (stage === "baseline_rest") {
            setScore((s) => s + 200);
            nextStage("baseline_focus");
          } else {
            setScore((s) => s + 300);
            nextStage("results");
          }
        }
      }, 100);
      return () => clearInterval(interval);
    }
  }, [stage, nextStage]);

  useEffect(() => {
    if (!brainState) return;
    if (stage === "baseline_rest") {
      setRelaxHistory((h) => [...h, brainState.relaxation]);
    } else if (stage === "baseline_focus") {
      setFocusHistory((h) => [...h, brainState.focus]);
    }
  }, [brainState, stage]);

  const avgFocus =
    focusHistory.length > 0
      ? focusHistory.reduce((a, b) => a + b, 0) / focusHistory.length
      : 0;

  const avgRelax =
    relaxHistory.length > 0
      ? relaxHistory.reduce((a, b) => a + b, 0) / relaxHistory.length
      : 0;

  return (
    <div style={fullscreen}>
      {stage === "welcome" && (
        <div style={centerCol}>
          <h1 style={{ fontSize: 48, fontWeight: 800, color: "#00e68a", margin: 0 }}>
            AXIOM
          </h1>
          <p style={{ fontSize: 16, color: "#8a8aa0", margin: "8px 0 0" }}>
            NEURAL INTERFACE CALIBRATION
          </p>
          <div style={{ ...card, marginTop: 40, maxWidth: 500 }}>
            <p style={{ color: "#c0c0d0", lineHeight: 1.6, margin: 0 }}>
              We'll measure your brain's baseline signals.
              Two quick stages — just relax and then focus.
            </p>
            <div style={{ marginTop: 20, display: "flex", flexDirection: "column", gap: 12 }}>
              <StagePreview n={1} label="Relax Baseline" desc="Close your eyes and relax for 10 seconds" color="#a855f7" />
              <StagePreview n={2} label="Focus Baseline" desc="Focus intently on a dot for 8 seconds" color="#4d8ef7" />
            </div>
            <div style={{ marginTop: 16, padding: "10px 14px", background: "#1a1a2e", borderRadius: 8, border: "1px solid #2a2a40" }}>
              <span style={{ color: "#6a6a80", fontSize: 12 }}>
                CURSOR MODE — mouse position is used as gaze input for now
              </span>
            </div>
          </div>
          <button onClick={() => nextStage("baseline_rest")} style={btnGreen}>
            START CALIBRATION
          </button>
        </div>
      )}

      {stage === "baseline_rest" && (
        <div style={centerCol}>
          <h2 style={{ fontSize: 28, color: "#a855f7", margin: 0 }}>
            STAGE 1 · RELAX
          </h2>
          <p style={{ color: "#8a8aa0", margin: "12px 0 30px" }}>
            Close your eyes and relax. We're measuring your alpha baseline.
          </p>
          <div style={{ fontSize: 64, color: "#a855f7", fontWeight: 700 }}>
            {countdown}
          </div>
          <ProgressBar progress={progress} color="#a855f7" />
          <div style={{ marginTop: 20, fontSize: 13, color: "#6a6a80" }}>
            Relaxation: {brainState ? `${(brainState.relaxation * 100).toFixed(0)}%` : "—"}
          </div>
        </div>
      )}

      {stage === "baseline_focus" && (
        <div style={centerCol}>
          <h2 style={{ fontSize: 28, color: "#4d8ef7", margin: 0 }}>
            STAGE 2 · FOCUS
          </h2>
          <p style={{ color: "#8a8aa0", margin: "12px 0 30px" }}>
            Focus intently on this dot. Think hard about a math problem.
          </p>
          <div
            style={{
              width: 24,
              height: 24,
              borderRadius: "50%",
              background: "#4d8ef7",
              boxShadow: "0 0 20px #4d8ef780",
              margin: "20px 0",
            }}
          />
          <div style={{ fontSize: 48, color: "#4d8ef7", fontWeight: 700 }}>
            {countdown}
          </div>
          <ProgressBar progress={progress} color="#4d8ef7" />
          <div style={{ marginTop: 20, fontSize: 13, color: "#6a6a80" }}>
            Focus: {brainState ? `${(brainState.focus * 100).toFixed(0)}%` : "—"}
          </div>
        </div>
      )}

      {stage === "results" && (
        <div style={centerCol}>
          <h1 style={{ fontSize: 36, color: "#00e68a", margin: 0 }}>
            CALIBRATION COMPLETE
          </h1>
          <div style={{ fontSize: 64, color: "#00e68a", fontWeight: 800, margin: "20px 0" }}>
            {score}
          </div>
          <p style={{ color: "#4a4a6a", fontSize: 13, letterSpacing: 2 }}>SCORE</p>

          <div style={{ ...card, marginTop: 30, width: 400 }}>
            <ResultRow label="Avg Relaxation" value={`${(avgRelax * 100).toFixed(0)}%`} color="#a855f7" />
            <ResultRow label="Avg Focus" value={`${(avgFocus * 100).toFixed(0)}%`} color="#4d8ef7" />
            <ResultRow label="Input Mode" value="Cursor" color="#00e68a" />
          </div>

          <button
            onClick={() => {
              onComplete({
                baselineAlpha: avgRelax,
                baselineBeta: avgFocus,
                focusScore: avgFocus,
                relaxScore: avgRelax,
              });
            }}
            style={{ ...btnGreen, marginTop: 30 }}
          >
            LAUNCH AXIOM
          </button>
        </div>
      )}
    </div>
  );
}

function StagePreview({
  n,
  label,
  desc,
  color,
}: {
  n: number;
  label: string;
  desc: string;
  color: string;
}) {
  return (
    <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
      <div
        style={{
          width: 28,
          height: 28,
          borderRadius: "50%",
          background: `${color}20`,
          border: `2px solid ${color}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 12,
          fontWeight: 700,
          color,
          flexShrink: 0,
        }}
      >
        {n}
      </div>
      <div>
        <div style={{ color, fontSize: 13, fontWeight: 600 }}>{label}</div>
        <div style={{ color: "#6a6a80", fontSize: 11 }}>{desc}</div>
      </div>
    </div>
  );
}

function ProgressBar({ progress, color }: { progress: number; color: string }) {
  return (
    <div
      style={{
        width: 300,
        height: 6,
        background: "#1a1a30",
        borderRadius: 3,
        overflow: "hidden",
        marginTop: 20,
      }}
    >
      <div
        style={{
          width: `${progress * 100}%`,
          height: "100%",
          background: `linear-gradient(90deg, ${color}60, ${color})`,
          borderRadius: 3,
          transition: "width 0.2s",
          boxShadow: `0 0 10px ${color}40`,
        }}
      />
    </div>
  );
}

function ResultRow({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        padding: "8px 0",
        borderBottom: "1px solid #1a1a30",
        fontSize: 14,
      }}
    >
      <span style={{ color: "#8a8aa0" }}>{label}</span>
      <span style={{ color, fontWeight: 700 }}>{value}</span>
    </div>
  );
}

const fullscreen: React.CSSProperties = {
  width: "100vw",
  height: "100vh",
  background: "#0a0a14",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};

const centerCol: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  textAlign: "center",
};

const card: React.CSSProperties = {
  background: "#12122a",
  borderRadius: 16,
  border: "1px solid #1e1e3a",
  padding: 24,
};

const btnGreen: React.CSSProperties = {
  marginTop: 24,
  padding: "14px 40px",
  fontSize: 14,
  fontWeight: 700,
  letterSpacing: 2,
  color: "#0a0a14",
  background: "#00e68a",
  border: "none",
  borderRadius: 8,
  cursor: "pointer",
  transition: "all 0.2s",
};
