import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { SignalIndicator } from '../components/ui/SignalIndicator';
import { SignalStrength } from '../components/ui/SignalStrength';
import { useElevenSocket } from '../hooks/useElevenSocket';
import { useSessionStore } from '../stores/sessionStore';

export function ConnectDevice() {
  const navigate = useNavigate();
  const [useSimulation, setUseSimulation] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);

  // Get connection methods and state
  const { connect, disconnect, startStreaming, error, clearError } = useElevenSocket();
  const { connectionState, signalStrength } = useSessionStore();

  // Handle successful connection - navigate to calibration
  useEffect(() => {
    if (connectionState === 'connected' || connectionState === 'streaming') {
      const timer = setTimeout(() => {
        navigate('/calibrate');
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [connectionState, navigate]);

  const handleConnect = async () => {
    if (isConnecting || connectionState === 'connected' || connectionState === 'streaming') {
      return;
    }

    setIsConnecting(true);
    clearError();

    try {
      // Ensure any stale session is cleared first
      await disconnect();

      const connected = await connect(useSimulation);
      if (connected) {
        // Start streaming after connection
        await startStreaming();
      }
    } catch (err) {
      console.error('Connection failed:', err);
    } finally {
      setIsConnecting(false);
    }
  };

  const getConnectionStatus = () => {
    if (isConnecting || connectionState === 'connecting') return 'connecting';
    if (connectionState === 'connected' || connectionState === 'streaming') return 'connected';
    if (connectionState === 'error') return 'error';
    return 'disconnected';
  };

  const status = getConnectionStatus();

  const getButtonContent = () => {
    switch (status) {
      case 'connecting':
        return (
          <>
            <span className="material-symbols-outlined animate-spin">sync</span>
            <span>Scanning for Muse S...</span>
          </>
        );
      case 'connected':
        return (
          <>
            <span className="material-symbols-outlined">check_circle</span>
            <span>Connected</span>
          </>
        );
      case 'error':
        return (
          <>
            <span className="material-symbols-outlined">error</span>
            <span>Retry Connection</span>
          </>
        );
      default:
        return <span>Connect via Bluetooth</span>;
    }
  };

  const getStatusLabel = () => {
    switch (status) {
      case 'connecting':
        return useSimulation ? 'Starting simulation...' : 'Scanning for Muse S...';
      case 'connected':
        return connectionState === 'streaming' ? 'Streaming' : 'Connected';
      case 'error':
        return 'Connection Failed';
      default:
        return 'Disconnected';
    }
  };

  return (
    <Layout showBackButton backTo="/" backLabel="Back">
      <main className="flex-grow flex items-center justify-center px-gutter py-stack-lg">
        <div className="w-full max-w-[480px] bg-surface-container-lowest border border-on-background p-stack-lg flex flex-col items-center gap-stack-md">
          {/* Connection Status */}
          <SignalIndicator
            status={status === 'connected' ? 'connected' : 'disconnected'}
            label={getStatusLabel()}
            size="sm"
          />

          {/* Header Section */}
          <div className="text-center space-y-2">
            <h1 className="text-headline-lg text-primary tracking-tight">
              Connect your headband.
            </h1>
            <p className="text-body-md text-on-surface-variant max-w-[320px] mx-auto">
              {useSimulation
                ? 'Running in simulation mode with synthetic EEG data.'
                : 'Initialize your Muse S neural interface via Bluetooth.'}
            </p>
          </div>

          {/* Central Pulsing Circle */}
          <div className="relative w-48 h-48 flex items-center justify-center">
            {/* Pulse rings */}
            {status === 'connecting' && (
              <>
                <div className="pulse-ring absolute w-48 h-48 border border-primary rounded-full animate-pulse-ring" />
                <div
                  className="pulse-ring absolute w-48 h-48 border border-primary rounded-full animate-pulse-ring"
                  style={{ animationDelay: '1s' }}
                />
                <div
                  className="pulse-ring absolute w-48 h-48 border border-primary rounded-full animate-pulse-ring"
                  style={{ animationDelay: '2s' }}
                />
              </>
            )}

            {/* Central circle */}
            <div
              className={`z-10 w-32 h-32 rounded-full border flex items-center justify-center bg-background transition-all duration-300 ${
                status === 'connected'
                  ? 'border-primary bg-primary'
                  : status === 'error'
                    ? 'border-error'
                    : 'border-primary'
              }`}
            >
              <span
                className={`material-symbols-outlined transition-all duration-300 ${
                  status === 'connected'
                    ? 'text-on-primary text-[64px]'
                    : status === 'error'
                      ? 'text-error text-[64px]'
                      : 'text-primary text-[64px]'
                }`}
                style={{ fontVariationSettings: "'wght' 200" }}
              >
                {status === 'connected'
                  ? 'check'
                  : status === 'error'
                    ? 'error_outline'
                    : useSimulation
                      ? 'psychology'
                      : 'bluetooth_searching'}
              </span>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="w-full p-4 bg-error-container rounded border border-error">
              <p className="text-body-md text-on-error-container text-center">
                {error.message}
              </p>
              <p className="text-label-sm text-on-error-container/70 text-center mt-2">
                Make sure the Eleven server is running: <code className="bg-error/20 px-1 rounded">python -m eleven.cli server</code>
              </p>
            </div>
          )}

          {/* Action Section */}
          <div className="w-full space-y-stack-sm text-center">
            <button
              onClick={handleConnect}
              disabled={status === 'connected'}
              className={`w-full min-h-target-min bg-primary text-on-primary text-label-lg uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${
                status === 'connecting' ? 'opacity-80' : ''
              } ${
                status === 'connected'
                  ? 'bg-primary/50 cursor-not-allowed'
                  : 'hover:opacity-90 active:scale-[0.98]'
              }`}
            >
              {getButtonContent()}
            </button>

            {/* Simulation Toggle */}
            <div className="flex items-center justify-center gap-3 py-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useSimulation}
                  onChange={(e) => setUseSimulation(e.target.checked)}
                  disabled={status === 'connecting' || status === 'connected'}
                  className="w-4 h-4 accent-primary"
                />
                <span className="text-label-sm text-on-surface-variant">
                  Use simulation mode (no device needed)
                </span>
              </label>
            </div>

            <p className="text-label-sm text-on-surface-variant px-stack-sm">
              {useSimulation
                ? 'Simulation mode generates synthetic EEG signals for testing.'
                : 'Ensure your Muse S headband is powered on and within range.'}
            </p>
          </div>

          {/* Signal Strength */}
          <div
            className={`w-full pt-stack-sm border-t border-surface-container flex justify-between items-center transition-opacity ${
              status === 'connected' ? 'opacity-100' : 'opacity-40'
            }`}
          >
            <span className="text-label-sm text-on-surface-variant">SIGNAL STRENGTH</span>
            <SignalStrength
              level={status === 'connected' ? signalStrength : 0}
              maxLevel={4}
            />
          </div>

          {/* Device Info (shown when connected) */}
          {status === 'connected' && (
            <div className="w-full pt-stack-sm border-t border-surface-container">
              <div className="flex justify-between items-center">
                <span className="text-label-sm text-on-surface-variant">DEVICE</span>
                <span className="text-label-sm text-primary">
                  {useSimulation ? 'Simulation' : 'Muse S'}
                </span>
              </div>
              <div className="flex justify-between items-center mt-2">
                <span className="text-label-sm text-on-surface-variant">STATUS</span>
                <span className="text-label-sm text-primary flex items-center gap-1">
                  <span className="w-2 h-2 bg-primary rounded-full animate-pulse" />
                  {connectionState === 'streaming' ? 'Streaming EEG' : 'Ready'}
                </span>
              </div>
            </div>
          )}
        </div>
      </main>
    </Layout>
  );
}
