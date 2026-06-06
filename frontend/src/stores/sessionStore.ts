import { create } from 'zustand';

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'streaming' | 'error';

export type ControlSignal = 'double_blink' | 'triple_blink' | 'jaw_clench' | 'long_jaw_clench';

interface AttentionData {
  focus: number;
  relaxation: number;
}

interface SessionState {
  // Connection
  connectionState: ConnectionState;
  signalStrength: number;

  // Attention metrics
  attention: AttentionData;

  // Latest control signal
  lastControlSignal: ControlSignal | null;
  lastSignalTimestamp: number | null;

  // Calibration
  isCalibrated: boolean;
  calibrationProgress: number;

  // Actions
  setConnectionState: (state: ConnectionState) => void;
  setSignalStrength: (strength: number) => void;
  setAttention: (attention: AttentionData) => void;
  setControlSignal: (signal: ControlSignal) => void;
  setCalibrated: (calibrated: boolean) => void;
  setCalibrationProgress: (progress: number) => void;
  reset: () => void;
}

const initialState = {
  connectionState: 'disconnected' as ConnectionState,
  signalStrength: 0,
  attention: { focus: 0, relaxation: 0 },
  lastControlSignal: null,
  lastSignalTimestamp: null,
  isCalibrated: false,
  calibrationProgress: 0,
};

export const useSessionStore = create<SessionState>((set) => ({
  ...initialState,

  setConnectionState: (connectionState) => set({ connectionState }),

  setSignalStrength: (signalStrength) => set({ signalStrength }),

  setAttention: (attention) => set({ attention }),

  setControlSignal: (signal) =>
    set({
      lastControlSignal: signal,
      lastSignalTimestamp: Date.now(),
    }),

  setCalibrated: (isCalibrated) => set({ isCalibrated }),

  setCalibrationProgress: (calibrationProgress) => set({ calibrationProgress }),

  reset: () => set(initialState),
}));
