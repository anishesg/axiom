import { useEffect, useRef, useCallback } from 'react';
import { useSessionStore } from '../stores/sessionStore';
import type { ControlSignal } from '../stores/sessionStore';

const WS_URL = 'ws://localhost:8000/ws';
const RECONNECT_DELAY = 3000;

interface WebSocketMessage {
  type: string;
  data?: Record<string, unknown>;
}

export function useElevenSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const {
    setConnectionState,
    setSignalStrength,
    setAttention,
    setControlSignal,
    setCalibrationProgress,
  } = useSessionStore();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionState('connecting');

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[Eleven] WebSocket connected');
      setConnectionState('connected');
    };

    ws.onclose = () => {
      console.log('[Eleven] WebSocket disconnected');
      setConnectionState('disconnected');

      // Auto-reconnect
      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, RECONNECT_DELAY);
    };

    ws.onerror = (error) => {
      console.error('[Eleven] WebSocket error:', error);
      setConnectionState('error');
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        handleMessage(message);
      } catch (err) {
        console.error('[Eleven] Failed to parse message:', err);
      }
    };
  }, [setConnectionState]);

  const handleMessage = useCallback(
    (message: WebSocketMessage) => {
      switch (message.type) {
        case 'heartbeat':
          // Update connection state based on backend status
          if (message.data?.state === 'streaming') {
            setConnectionState('streaming');
          } else if (message.data?.state === 'connected') {
            setConnectionState('connected');
          }
          break;

        case 'control_signal':
          // EEG control signal detected (blink, clench, etc.)
          if (message.data?.signal) {
            setControlSignal(message.data.signal as ControlSignal);
            console.log('[Eleven] Control signal:', message.data.signal);
          }
          break;

        case 'attention':
          // Attention/focus metrics
          if (message.data) {
            setAttention({
              focus: (message.data.focus as number) || 0,
              relaxation: (message.data.relaxation as number) || 0,
            });
          }
          break;

        case 'signal_quality':
          // Signal strength update
          if (typeof message.data?.quality === 'number') {
            // Convert 0-1 to 0-4 scale for UI
            setSignalStrength(Math.round((message.data.quality as number) * 4));
          }
          break;

        case 'calibration_progress':
          // Calibration progress update
          if (typeof message.data?.progress === 'number') {
            setCalibrationProgress(message.data.progress as number);
          }
          break;

        default:
          console.log('[Eleven] Unknown message type:', message.type);
      }
    },
    [setConnectionState, setControlSignal, setAttention, setSignalStrength, setCalibrationProgress]
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
      console.warn('[Eleven] Cannot send - WebSocket not connected');
    }
  }, []);

  // Commands
  const startStreaming = useCallback(() => send('start'), [send]);
  const stopStreaming = useCallback(() => send('stop'), [send]);
  const startCalibration = useCallback(() => send('calibrate'), [send]);

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
    startStreaming,
    stopStreaming,
    startCalibration,
  };
}
