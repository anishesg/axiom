import { create } from 'zustand';

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'streaming' | 'error';

// Mapped from axiom server's jaw_clench/double_clench to Eleven's control signals
export type ControlSignal = 'double_blink' | 'triple_blink' | 'jaw_clench' | 'long_jaw_clench';

// Brain state from axiom server (brain_state.py)
export interface BrainState {
  engagement: number;
  focus: number;
  relaxation: number;
  cognitive_load: number;
  valence: number;
  jaw_clench: boolean;
  double_clench: boolean;
  context_switch: boolean;
  error_response: boolean;
}

// EEG band powers
export interface BandPowers {
  delta: number;
  theta: number;
  alpha: number;
  beta: number;
  gamma: number;
}

// Learning/agent metrics
export interface LearningMetrics {
  accuracy: number;
  totalActions: number;
  undoneActions: number;
  mode: string;
}

// Agent action
export interface AgentAction {
  action_type: string;
  target: string | null;
  confidence: number;
  reason: string;
  timestamp: number;
}

// Signal history entry for timeline
export interface SignalHistoryEntry {
  id: number;
  signal: ControlSignal;
  timestamp: number;
}

interface SessionState {
  // Connection
  connectionState: ConnectionState;
  signalStrength: number;

  // Brain state (from axiom server)
  brainState: BrainState;
  bandPowers: BandPowers;

  // Learning metrics
  learningMetrics: LearningMetrics;

  // Latest agent action
  lastAction: AgentAction | null;

  // Latest control signal (mapped from brain state)
  lastControlSignal: ControlSignal | null;
  lastSignalTimestamp: number | null;

  // Calibration
  isCalibrated: boolean;
  calibrationProgress: number;

  // Signal history for dashboard/training
  signalHistory: SignalHistoryEntry[];
  sessionStartTime: number | null;
  totalSignalsDetected: number;

  // Raw EEG buffer for visualization (4 channels x samples)
  rawEEG: number[][];

  // Server info
  serverInfo: {
    channels: string[];
    sampleRate: number;
    gazeEnabled: boolean;
    llmEnabled: boolean;
    osActionsEnabled: boolean;
  } | null;

  // Actions
  setConnectionState: (state: ConnectionState) => void;
  setSignalStrength: (strength: number) => void;
  setBrainState: (state: BrainState) => void;
  setBandPowers: (bands: BandPowers) => void;
  setLearningMetrics: (metrics: LearningMetrics) => void;
  setLastAction: (action: AgentAction) => void;
  setControlSignal: (signal: ControlSignal) => void;
  setCalibrated: (calibrated: boolean) => void;
  setCalibrationProgress: (progress: number) => void;
  setServerInfo: (info: SessionState['serverInfo']) => void;
  addRawEEG: (channels: number[][]) => void;
  startSession: () => void;
  clearSignalHistory: () => void;
  reset: () => void;
}

const initialBrainState: BrainState = {
  engagement: 0,
  focus: 0,
  relaxation: 0,
  cognitive_load: 0,
  valence: 0,
  jaw_clench: false,
  double_clench: false,
  context_switch: false,
  error_response: false,
};

const initialBandPowers: BandPowers = {
  delta: 0,
  theta: 0,
  alpha: 0,
  beta: 0,
  gamma: 0,
};

const initialLearningMetrics: LearningMetrics = {
  accuracy: 0,
  totalActions: 0,
  undoneActions: 0,
  mode: 'passive',
};

const initialState = {
  connectionState: 'disconnected' as ConnectionState,
  signalStrength: 0,
  brainState: initialBrainState,
  bandPowers: initialBandPowers,
  learningMetrics: initialLearningMetrics,
  lastAction: null,
  lastControlSignal: null,
  lastSignalTimestamp: null,
  isCalibrated: false,
  calibrationProgress: 0,
  signalHistory: [] as SignalHistoryEntry[],
  sessionStartTime: null as number | null,
  totalSignalsDetected: 0,
  rawEEG: [] as number[][],
  serverInfo: null,
};

let signalIdCounter = 0;

export const useSessionStore = create<SessionState>((set) => ({
  ...initialState,

  setConnectionState: (connectionState) => set({ connectionState }),

  setSignalStrength: (signalStrength) => set({ signalStrength }),

  setBrainState: (brainState) => set({ brainState }),

  setBandPowers: (bandPowers) => set({ bandPowers }),

  setLearningMetrics: (learningMetrics) => set({ learningMetrics }),

  setLastAction: (lastAction) => set({ lastAction }),

  setControlSignal: (signal) =>
    set((state) => {
      const now = Date.now();
      const newEntry: SignalHistoryEntry = {
        id: signalIdCounter++,
        signal,
        timestamp: now,
      };
      // Keep last 30 seconds of history (assuming ~1 signal per second max, keep 60 entries)
      const cutoff = now - 30000;
      const filteredHistory = state.signalHistory.filter((e) => e.timestamp > cutoff);
      return {
        lastControlSignal: signal,
        lastSignalTimestamp: now,
        signalHistory: [...filteredHistory, newEntry],
        totalSignalsDetected: state.totalSignalsDetected + 1,
      };
    }),

  setCalibrated: (isCalibrated) => set({ isCalibrated }),

  setCalibrationProgress: (calibrationProgress) => set({ calibrationProgress }),

  setServerInfo: (serverInfo) => set({ serverInfo }),

  addRawEEG: (channels) =>
    set((state) => {
      // Each channel is ~64 samples (256ms window at 256Hz)
      // Keep last 512 samples per channel (~2 seconds of data)
      const MAX_SAMPLES = 512;
      const newRawEEG = channels.map((newChannel, i) => {
        const existing = state.rawEEG[i] || [];
        const combined = [...existing, ...newChannel];
        // Keep only the last MAX_SAMPLES
        return combined.slice(-MAX_SAMPLES);
      });
      return { rawEEG: newRawEEG };
    }),

  startSession: () =>
    set({
      sessionStartTime: Date.now(),
      signalHistory: [],
      totalSignalsDetected: 0,
    }),

  clearSignalHistory: () =>
    set({
      signalHistory: [],
      totalSignalsDetected: 0,
    }),

  reset: () => {
    signalIdCounter = 0;
    set(initialState);
  },
}));

// Selector for attention (derived from brain state for backward compatibility)
export const selectAttention = (state: SessionState) => ({
  focus: state.brainState.focus,
  relaxation: state.brainState.relaxation,
});
