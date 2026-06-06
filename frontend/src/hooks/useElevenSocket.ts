import { useEffect, useRef, useCallback } from 'react';
import { useSessionStore } from '../stores/sessionStore';
import type { BrainState, BandPowers, LearningMetrics, AgentAction } from '../stores/sessionStore';

// Axiom server runs on port 8080
const WS_URL = 'ws://localhost:8080';
const RECONNECT_DELAY = 3000;

interface WebSocketMessage {
  type: string;
  // Info message
  channels?: string[];
  sampleRate?: number;
  gaze_enabled?: boolean;
  llm_enabled?: boolean;
  os_actions_enabled?: boolean;
  // EEG message
  bands?: Record<string, number>;
  totalSamples?: number;
  // Brain state message
  engagement?: number;
  focus?: number;
  relaxation?: number;
  cognitive_load?: number;
  valence?: number;
  jaw_clench?: boolean;
  double_clench?: boolean;
  context_switch?: boolean;
  error_response?: boolean;
  // Action message
  action_type?: string;
  target?: string;
  confidence?: number;
  reason?: string;
  // Learning message
  accuracy?: number;
  total_actions?: number;
  undone_actions?: number;
  mode?: string;
  // Common
  timestamp?: number;
}

export function useElevenSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const {
    setConnectionState,
    setSignalStrength,
    setBrainState,
    setBandPowers,
    setLearningMetrics,
    setLastAction,
    setControlSignal,
    setServerInfo,
  } = useSessionStore();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionState('connecting');

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[Axiom] WebSocket connected to', WS_URL);
      setConnectionState('connected');
    };

    ws.onclose = () => {
      console.log('[Axiom] WebSocket disconnected');
      setConnectionState('disconnected');

      // Auto-reconnect
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, RECONNECT_DELAY);
    };

    ws.onerror = (error) => {
      console.error('[Axiom] WebSocket error:', error);
      setConnectionState('error');
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        handleMessage(message);
      } catch (err) {
        console.error('[Axiom] Failed to parse message:', err);
      }
    };
  }, [setConnectionState]);

  const handleMessage = useCallback(
    (message: WebSocketMessage) => {
      switch (message.type) {
        case 'info':
          // Server info on connection
          console.log('[Axiom] Server info:', message);
          setServerInfo({
            channels: message.channels || [],
            sampleRate: message.sampleRate || 256,
            gazeEnabled: message.gaze_enabled || false,
            llmEnabled: message.llm_enabled || false,
            osActionsEnabled: message.os_actions_enabled || false,
          });
          setConnectionState('streaming');
          break;

        case 'eeg':
          // EEG data with band powers (20Hz)
          if (message.bands) {
            const bands: BandPowers = {
              delta: message.bands.delta || 0,
              theta: message.bands.theta || 0,
              alpha: message.bands.alpha || 0,
              beta: message.bands.beta || 0,
              gamma: message.bands.gamma || 0,
            };
            setBandPowers(bands);

            // Estimate signal quality from alpha presence
            const totalPower = bands.alpha + bands.beta + bands.theta;
            const quality = Math.min(4, Math.round((totalPower / 50) * 4));
            setSignalStrength(quality);
          }
          break;

        case 'brain':
          // Brain state (2Hz)
          const brainState: BrainState = {
            engagement: message.engagement || 0,
            focus: message.focus || 0,
            relaxation: message.relaxation || 0,
            cognitive_load: message.cognitive_load || 0,
            valence: message.valence || 0,
            jaw_clench: message.jaw_clench || false,
            double_clench: message.double_clench || false,
            context_switch: message.context_switch || false,
            error_response: message.error_response || false,
          };
          setBrainState(brainState);

          // Map brain signals to Eleven control signals
          if (message.double_clench) {
            // Double jaw clench = YES (like double blink)
            setControlSignal('double_blink');
            console.log('[Axiom] Control signal: double_clench → YES');
          } else if (message.jaw_clench) {
            // Single jaw clench = SELECT
            setControlSignal('jaw_clench');
            console.log('[Axiom] Control signal: jaw_clench → SELECT');
          }
          break;

        case 'action':
          // Agent action
          if (message.action_type && message.action_type !== 'none') {
            const action: AgentAction = {
              action_type: message.action_type,
              target: message.target || null,
              confidence: message.confidence || 0,
              reason: message.reason || '',
              timestamp: message.timestamp || Date.now(),
            };
            setLastAction(action);
            console.log('[Axiom] Agent action:', action.action_type, action.target);
          }
          break;

        case 'learning':
          // Learning metrics (every 5s)
          const metrics: LearningMetrics = {
            accuracy: message.accuracy || 0,
            totalActions: message.total_actions || 0,
            undoneActions: message.undone_actions || 0,
            mode: message.mode || 'passive',
          };
          setLearningMetrics(metrics);
          break;

        case 'state':
          // RL state vector (1Hz) - we don't use this directly in the UI
          break;

        default:
          console.log('[Axiom] Unknown message type:', message.type);
      }
    },
    [setConnectionState, setBrainState, setBandPowers, setLearningMetrics, setLastAction, setControlSignal, setSignalStrength, setServerInfo]
  );

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionState('disconnected');
  }, [setConnectionState]);

  const send = useCallback((command: string, data?: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command, ...data }));
    } else {
      console.warn('[Axiom] Cannot send - WebSocket not connected');
    }
  }, []);

  // Commands to axiom server
  const enableOsActions = useCallback((enabled: boolean) => send('enable_os_actions', { enabled }), [send]);
  const enableLlm = useCallback((enabled: boolean) => send('enable_llm', { enabled }), [send]);
  const recordOutcome = useCallback((outcome: 'confirmed' | 'undone') => send('record_outcome', { outcome }), [send]);
  const adjustThreshold = useCallback((key: string, value: number) => send('adjust_threshold', { key, value }), [send]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connect,
    disconnect,
    send,
    enableOsActions,
    enableLlm,
    recordOutcome,
    adjustThreshold,
  };
}
