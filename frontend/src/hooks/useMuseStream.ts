import { useEffect, useRef, useCallback, useState } from "react";

export interface EEGFrame {
  channels: string[];
  data: number[][];
  bands: Record<string, number>;
  sampleRate: number;
  totalSamples: number;
  timestamp: number;
}

export interface RLState {
  state: number[];
  bands: Record<string, number> | null;
  band_ratios: Record<string, number>;
  asymmetry: number;
  coherence: Record<string, number>;
  quality: Record<string, number>;
  timestamp: number;
  totalSamples: number;
}

export interface BrainState {
  engagement: number;
  focus: number;
  relaxation: number;
  cognitive_load: number;
  valence: number;
  jaw_clench: boolean;
  double_clench: boolean;
  context_switch: number;
  error_response: number;
  timestamp: number;
}

export interface AgentAction {
  action_type: string;
  target: string;
  confidence: number;
  reason: string;
  timestamp: number;
}

export interface LearningMetrics {
  accuracy: number;
  accuracy_history: number[];
  total_actions: number;
  undone_actions: number;
  thresholds: Record<string, number>;
  decision_log: Array<{
    action_type: string;
    target: string;
    confidence: number;
    reason: string;
  }>;
  mode: string;
  description?: string;
  timestamp: number;
}

export type ConnectionStatus = "disconnected" | "connecting" | "connected";

const WS_URL = "ws://127.0.0.1:8080";
const RECONNECT_MS = 2000;

export function useMuseStream() {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [frame, setFrame] = useState<EEGFrame | null>(null);
  const [rlState, setRlState] = useState<RLState | null>(null);
  const [brainState, setBrainState] = useState<BrainState | null>(null);
  const [lastAction, setLastAction] = useState<AgentAction | null>(null);
  const [learning, setLearning] = useState<LearningMetrics | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setStatus("connecting");
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setStatus("connected");

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "eeg") setFrame(msg as EEGFrame);
      else if (msg.type === "state") setRlState(msg as RLState);
      else if (msg.type === "brain") setBrainState(msg as BrainState);
      else if (msg.type === "action") setLastAction(msg as AgentAction);
      else if (msg.type === "learning") setLearning(msg as LearningMetrics);
    };

    ws.onclose = () => {
      setStatus("disconnected");
      wsRef.current = null;
      reconnectTimer.current = setTimeout(connect, RECONNECT_MS);
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { status, frame, rlState, brainState, lastAction, learning };
}
