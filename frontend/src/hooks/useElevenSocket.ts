import { useEffect, useRef, useCallback, useState } from 'react';
import { useSessionStore } from '../stores/sessionStore';
import type { BrainState, BandPowers, LearningMetrics, AgentAction } from '../stores/sessionStore';

// Eleven server runs on port 8000
const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';
const RECONNECT_DELAY = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

interface WebSocketMessage {
  type: string;
  timestamp?: number;
  data?: Record<string, unknown>;
}

export interface CalibrationStep {
  step: 'rest' | 'natural_blinks' | 'deliberate_blinks' | 'jaw_clenches';
  duration_seconds: number;
}

export interface EnhancedCalibrationStep {
  id: string;
  name: string;
  duration: number;
  instruction: string;
}

export interface UserProfile {
  user_id: string;
  created_at: string;
  calibration_complete: boolean;
  baseline_collected: boolean;
  intents_trained: string[];
  total_sessions: number;
}

export interface ConnectionError {
  type: 'connection' | 'api' | 'websocket';
  message: string;
  code?: string;
  action?: string;
  recoverable?: boolean;
  collected?: Record<string, number>;
  missing?: string[];
}

export function useElevenSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const reconnectAttemptsRef = useRef(0);
  const connectingRef = useRef(false);  // Prevents duplicate connection attempts

  const [error, setError] = useState<ConnectionError | null>(null);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [calibrationStep, setCalibrationStep] = useState<string | null>(null);

  const {
    setConnectionState,
    setSignalStrength,
    setBrainState,
    setBandPowers,
    setLearningMetrics,
    setLastAction,
    setControlSignal,
    setServerInfo,
    setCalibrationProgress: setStoreCalibrationProgress,
    setCalibrated,
    addRawEEG,
  } = useSessionStore();

  // Clear error after a timeout
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // API helper with error handling
  const apiCall = useCallback(async (
    endpoint: string,
    method: 'GET' | 'POST' = 'POST',
    body?: Record<string, unknown>
  ) => {
    try {
      const response = await fetch(`${API_URL}${endpoint}`, {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
        body: body ? JSON.stringify(body) : undefined,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));

        // Parse structured error response
        const detail = errorData.detail;
        const isStructured = detail && typeof detail === 'object';

        const error: ConnectionError = {
          type: 'api',
          message: isStructured ? detail.message : (detail || `API error: ${response.status}`),
          code: isStructured ? detail.code : undefined,
          action: isStructured ? detail.action : undefined,
          recoverable: isStructured ? detail.recoverable : true,
          collected: isStructured ? detail.collected : undefined,
          missing: isStructured ? detail.missing : undefined,
        };

        setError(error);
        throw error;
      }

      return await response.json();
    } catch (err) {
      // If it's already a ConnectionError, just re-throw
      if (err && typeof err === 'object' && 'type' in err) {
        throw err;
      }
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError({ type: 'api', message, recoverable: true });
      throw err;
    }
  }, []);

  // WebSocket message handler
  const handleMessage = useCallback(
    (message: WebSocketMessage) => {
      const { type, data } = message;

      switch (type) {
        case 'connection':
          if (data?.status === 'connected') {
            setConnectionState('connected');
            reconnectAttemptsRef.current = 0;
            setError(null);
          } else if (data?.status === 'disconnected') {
            setConnectionState('disconnected');
          }
          break;

        case 'streaming':
          if (data?.status === 'started') {
            setConnectionState('streaming');
          } else if (data?.status === 'stopped') {
            setConnectionState('connected');
          }
          break;

        case 'heartbeat':
          // Update state from heartbeat
          if (data?.state) {
            const state = data.state as string;
            if (state === 'streaming') {
              setConnectionState('streaming');
            } else if (state === 'connected') {
              setConnectionState('connected');
            } else if (state === 'calibrating') {
              setIsCalibrating(true);
            }
          }
          break;

        case 'control_signal':
          // Map eleven signals to store types
          const signalMap: Record<string, 'double_blink' | 'triple_blink' | 'jaw_clench' | 'long_jaw_clench'> = {
            'single_blink': 'double_blink', // Treat single as confirmation
            'double_blink': 'double_blink',
            'triple_blink': 'triple_blink',
            'jaw_clench': 'jaw_clench',
            'long_jaw_clench': 'long_jaw_clench',
          };
          const signal = data?.signal as string;
          if (signal && signalMap[signal]) {
            setControlSignal(signalMap[signal]);
            console.log('[Eleven] Control signal:', signal);
          }
          break;

        case 'attention':
          // Attention/focus metrics
          if (data) {
            const brainState: BrainState = {
              engagement: (data.engagement as number) || 0,
              focus: (data.focus as number) || 0,
              relaxation: (data.relaxation as number) || 0,
              cognitive_load: (data.cognitive_load as number) || 0,
              valence: (data.valence as number) || 0,
              jaw_clench: false,
              double_clench: false,
              context_switch: false,
              error_response: false,
            };
            setBrainState(brainState);

            // Estimate signal quality from attention confidence
            const quality = Math.min(4, Math.round(((data.focus as number) || 0) * 4));
            setSignalStrength(quality);
          }
          break;

        case 'eeg':
        case 'eeg_raw':
          // Filtered EEG data for waveform visualization
          // Handle both new format (data.data) and old format (data.channels)
          const eegChannels = data?.data || data?.channels;
          if (eegChannels && Array.isArray(eegChannels) && eegChannels.length > 0) {
            addRawEEG(eegChannels as number[][]);
          }
          break;

        case 'calibration':
          if (data?.status === 'started') {
            setIsCalibrating(true);
            setCalibrationProgress(0);
          } else if (data?.status === 'completed') {
            setIsCalibrating(false);
            setCalibrationProgress(100);
            setCalibrated(true);
            setStoreCalibrationProgress(100);
          }
          break;

        case 'calibration_step':
          if (data?.status === 'started') {
            setCalibrationStep(data.step as string);
            setCalibrationProgress(0);
          } else if (data?.status === 'completed') {
            setCalibrationStep(null);
          }
          // Update progress from samples
          if (data?.progress !== undefined) {
            setCalibrationProgress((data.progress as number) * 100);
            setStoreCalibrationProgress((data.progress as number) * 100);
          }
          break;

        case 'error':
          setError({
            type: 'websocket',
            message: (data?.message as string) || 'Unknown error',
            code: data?.code as string,
          });
          break;

        default:
          console.log('[Eleven] Unknown message type:', type);
      }
    },
    [setConnectionState, setBrainState, setControlSignal, setSignalStrength, setCalibrated, setStoreCalibrationProgress, addRawEEG]
  );

  // Connect WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[Eleven] WebSocket connected');
      reconnectAttemptsRef.current = 0;
      setError(null);
    };

    ws.onclose = (event) => {
      console.log('[Eleven] WebSocket disconnected', event.code, event.reason);

      // Auto-reconnect with backoff
      if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttemptsRef.current++;
        const delay = RECONNECT_DELAY * Math.pow(1.5, reconnectAttemptsRef.current - 1);
        console.log(`[Eleven] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`);

        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, delay);
      } else {
        setError({
          type: 'websocket',
          message: 'Unable to connect to EEG server. Please check if the server is running.',
        });
        setConnectionState('error');
      }
    };

    ws.onerror = () => {
      console.error('[Eleven] WebSocket error');
      // Error details come through onclose
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        handleMessage(message);
      } catch (err) {
        console.error('[Eleven] Failed to parse message:', err);
      }
    };
  }, [handleMessage, setConnectionState]);

  // Ensure WebSocket is connected (without creating a new session)
  // Use this when navigating to a page that needs WebSocket events but session may already exist
  const ensureWebSocket = useCallback(async () => {
    // Connect WebSocket if not already connected
    connectWebSocket();

    // Check current session state from server
    try {
      const status = await apiCall('/session/state', 'GET');
      if (status?.state === 'streaming') {
        setConnectionState('streaming');
      } else if (status?.state === 'connected') {
        setConnectionState('connected');
      } else if (status?.state === 'disconnected') {
        setConnectionState('disconnected');
      }
      return true;
    } catch {
      // Server might be down, but WebSocket will handle reconnection
      return false;
    }
  }, [apiCall, connectWebSocket, setConnectionState]);

  // Connect to Muse via REST API
  const connect = useCallback(async (useSimulation = false) => {
    // Prevent duplicate connection attempts
    if (connectingRef.current) {
      console.log('[Eleven] Connection already in progress, ignoring duplicate call');
      return false;
    }

    connectingRef.current = true;
    setConnectionState('connecting');
    setError(null);

    try {
      await apiCall('/session/connect', 'POST', {
        use_simulation: useSimulation,
        notch_freq: 60.0,
      });

      // Connect WebSocket for events
      connectWebSocket();

      setConnectionState('connected');
      return true;
    } catch (err) {
      // If session already active, just connect WebSocket
      if (err && typeof err === 'object' && 'type' in err) {
        const connErr = err as { message?: string };
        if (connErr.message?.includes('already active') || connErr.message?.includes('streaming')) {
          console.log('[Eleven] Session already active, connecting WebSocket only');
          connectWebSocket();
          setConnectionState('streaming');
          return true;
        }
      }
      setConnectionState('error');
      return false;
    } finally {
      connectingRef.current = false;
    }
  }, [apiCall, connectWebSocket, setConnectionState]);

  // Start streaming
  const startStreaming = useCallback(async () => {
    try {
      await apiCall('/session/start');
      setConnectionState('streaming');
      return true;
    } catch {
      return false;
    }
  }, [apiCall, setConnectionState]);

  // Stop streaming
  const stopStreaming = useCallback(async () => {
    try {
      await apiCall('/session/stop');
      setConnectionState('connected');
      return true;
    } catch {
      return false;
    }
  }, [apiCall, setConnectionState]);

  // Disconnect
  const disconnect = useCallback(async () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    try {
      await apiCall('/session/disconnect');
    } catch {
      // Ignore disconnect errors
    }

    setConnectionState('disconnected');
  }, [apiCall, setConnectionState]);

  // Calibration methods
  const startCalibration = useCallback(async () => {
    try {
      await apiCall('/calibration/start');
      setIsCalibrating(true);
      setCalibrationProgress(0);
      return true;
    } catch {
      return false;
    }
  }, [apiCall]);

  const runCalibrationStep = useCallback(async (step: CalibrationStep) => {
    try {
      await apiCall('/calibration/step', 'POST', step);
      setCalibrationStep(step.step);
      return true;
    } catch {
      return false;
    }
  }, [apiCall]);

  const endCalibrationStep = useCallback(async () => {
    try {
      await apiCall('/calibration/end-step');
      setCalibrationStep(null);
      return true;
    } catch {
      return false;
    }
  }, [apiCall]);

  const completeCalibration = useCallback(async () => {
    try {
      const result = await apiCall('/calibration/complete');
      setIsCalibrating(false);
      setCalibrationProgress(100);
      setCalibrated(true);
      return result;
    } catch {
      return null;
    }
  }, [apiCall, setCalibrated]);

  const getCalibrationStatus = useCallback(async () => {
    try {
      return await apiCall('/calibration/status', 'GET');
    } catch {
      return null;
    }
  }, [apiCall]);

  // Enhanced calibration methods (RVQ-based)
  const startEnhancedCalibration = useCallback(async (userId: string = 'default') => {
    try {
      const result = await apiCall(`/calibration/enhanced/start/${userId}`);
      setIsCalibrating(true);
      setCalibrationProgress(0);
      return result;
    } catch {
      return null;
    }
  }, [apiCall]);

  const getEnhancedCalibrationSteps = useCallback(async (): Promise<EnhancedCalibrationStep[]> => {
    try {
      const result = await apiCall('/calibration/enhanced/steps', 'GET');
      return result.steps || [];
    } catch {
      return [];
    }
  }, [apiCall]);

  const startEnhancedCalibrationStep = useCallback(async (stepId: string) => {
    try {
      const result = await apiCall('/calibration/enhanced/step', 'POST', { step_id: stepId });
      setCalibrationStep(stepId);
      return result;
    } catch {
      return null;
    }
  }, [apiCall]);

  const endEnhancedCalibrationStep = useCallback(async () => {
    try {
      const result = await apiCall('/calibration/enhanced/end-step');
      setCalibrationStep(null);
      return result;
    } catch {
      return null;
    }
  }, [apiCall]);

  const trainVQTokenizer = useCallback(async (useRvq: boolean = true) => {
    try {
      return await apiCall('/calibration/enhanced/train', 'POST', { use_rvq: useRvq });
    } catch {
      return null;
    }
  }, [apiCall]);

  const completeEnhancedCalibration = useCallback(async () => {
    try {
      const result = await apiCall('/calibration/enhanced/complete');
      setIsCalibrating(false);
      setCalibrationProgress(100);
      setCalibrated(true);
      return result;
    } catch {
      return null;
    }
  }, [apiCall, setCalibrated]);

  // Profile management
  const getProfile = useCallback(async (userId: string): Promise<UserProfile | null> => {
    try {
      return await apiCall(`/profiles/${userId}`, 'GET');
    } catch {
      return null;
    }
  }, [apiCall]);

  const createProfile = useCallback(async (userId: string): Promise<UserProfile | null> => {
    try {
      return await apiCall(`/profiles/${userId}`, 'POST');
    } catch {
      return null;
    }
  }, [apiCall]);

  const activateProfile = useCallback(async (userId: string) => {
    try {
      return await apiCall(`/profiles/${userId}/activate`, 'POST');
    } catch {
      return null;
    }
  }, [apiCall]);

  const listProfiles = useCallback(async (): Promise<string[]> => {
    try {
      const result = await apiCall('/profiles', 'GET');
      return result.profiles || [];
    } catch {
      return [];
    }
  }, [apiCall]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    // Connection
    connect,
    ensureWebSocket,
    disconnect,
    startStreaming,
    stopStreaming,

    // Basic Calibration
    startCalibration,
    runCalibrationStep,
    endCalibrationStep,
    completeCalibration,
    getCalibrationStatus,
    isCalibrating,
    calibrationProgress,
    calibrationStep,

    // Enhanced Calibration (RVQ)
    startEnhancedCalibration,
    getEnhancedCalibrationSteps,
    startEnhancedCalibrationStep,
    endEnhancedCalibrationStep,
    trainVQTokenizer,
    completeEnhancedCalibration,

    // Profile Management
    getProfile,
    createProfile,
    activateProfile,
    listProfiles,

    // Error handling
    error,
    clearError,
  };
}
