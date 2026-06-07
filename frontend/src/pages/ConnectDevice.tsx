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
        <div className="w-full max-w-[420px] bg-white border border-outline-variant rounded-2xl p-8 flex flex-col items-center gap-7 shadow-sm">
          {/* Connection Status */}
          <SignalIndicator
            status={status === 'connected' ? 'connected' : 'disconnected'}
            label={getStatusLabel()}
            size="sm"
          />

          {/* Header Section */}
          <div className="text-center space-y-2">
            <h1 className="text-headline-lg text-primary tracking-tight">
              Connect your headband
            </h1>
            <p className="text-body-md text-on-surface-variant max-w-[300px] mx-auto leading-relaxed">
              {useSimulation
                ? 'Running in simulation mode with synthetic EEG data.'
                : 'Initialize your Muse S neural interface via Bluetooth.'}
            </p>
          </div>

          {/* Central Pulsing Circle */}
          <div className="relative w-40 h-40 flex items-center justify-center my-2">
            {/* Pulse rings */}
            {status === 'connecting' && (
              <>
                <div className="absolute w-40 h-40 border border-primary/30 rounded-full animate-pulse-ring" />
                <div
                  className="absolute w-40 h-40 border border-primary/30 rounded-full animate-pulse-ring"
                  style={{ animationDelay: '1s' }}
                />
                <div
                  className="absolute w-40 h-40 border border-primary/30 rounded-full animate-pulse-ring"
                  style={{ animationDelay: '2s' }}
                />
              </>
            )}

            {/* Central circle */}
            <div
              className={`z-10 w-28 h-28 rounded-full flex items-center justify-center transition-all duration-500 ease-out ${
                status === 'connected'
                  ? 'bg-primary shadow-lg'
                  : status === 'error'
                    ? 'bg-error-container border-2 border-error'
                    : 'bg-surface-container-low border-2 border-outline-variant'
              }`}
            >
              <span
                className={`material-symbols-outlined transition-all duration-300 ${
                  status === 'connected'
                    ? 'text-on-primary text-[48px]'
                    : status === 'error'
                      ? 'text-error text-[48px]'
                      : 'text-on-surface-variant text-[48px]'
                }`}
                style={{ fontVariationSettings: "'wght' 300" }}
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
            <div className="w-full p-4 bg-error-container rounded-xl">
              <p className="text-body-md text-on-error-container text-center">
                {error.message}
              </p>
              <p className="text-[11px] text-on-error-container/70 text-center mt-2">
                Run: <code className="bg-error/10 px-1.5 py-0.5 rounded font-mono">python -m eleven.cli server</code>
              </p>
            </div>
          )}

          {/* Action Section */}
          <div className="w-full space-y-4 text-center">
            <button
              onClick={handleConnect}
              disabled={status === 'connected'}
              className={`w-full py-3.5 bg-primary text-on-primary text-[14px] font-medium rounded-lg transition-all flex items-center justify-center gap-2.5 ${
                status === 'connecting' ? 'opacity-70' : ''
              } ${
                status === 'connected'
                  ? 'opacity-50 cursor-not-allowed'
                  : 'hover:shadow-md hover:-translate-y-0.5 active:translate-y-0 active:shadow-sm'
              }`}
            >
              {getButtonContent()}
            </button>

            {/* Simulation Toggle */}
            <label className="flex items-center justify-center gap-2.5 cursor-pointer py-1">
              <input
                type="checkbox"
                checked={useSimulation}
                onChange={(e) => setUseSimulation(e.target.checked)}
                disabled={status === 'connecting' || status === 'connected'}
                className="w-4 h-4 accent-primary rounded"
              />
              <span className="text-[13px] text-on-surface-variant">
                Use simulation mode
              </span>
            </label>

            <p className="text-[12px] text-on-surface-variant/70 leading-relaxed">
              {useSimulation
                ? 'Generates synthetic EEG signals for testing.'
                : 'Ensure your Muse S is powered on and nearby.'}
            </p>
          </div>

          {/* Signal Strength */}
          <div
            className={`w-full pt-5 border-t border-outline-variant/50 flex justify-between items-center transition-all duration-300 ${
              status === 'connected' ? 'opacity-100' : 'opacity-40'
            }`}
          >
            <span className="text-[11px] text-on-surface-variant uppercase tracking-wider">Signal</span>
            <SignalStrength
              level={status === 'connected' ? signalStrength : 0}
              maxLevel={4}
            />
          </div>

          {/* Device Info (shown when connected) */}
          {status === 'connected' && (
            <div className="w-full pt-4 border-t border-outline-variant/50 space-y-3 animate-fade-in">
              <div className="flex justify-between items-center">
                <span className="text-[11px] text-on-surface-variant uppercase tracking-wider">Device</span>
                <span className="text-[13px] font-medium text-primary">
                  {useSimulation ? 'Simulation' : 'Muse S'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-[11px] text-on-surface-variant uppercase tracking-wider">Status</span>
                <span className="text-[13px] font-medium text-primary flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 bg-primary rounded-full signal-pulse" />
                  {connectionState === 'streaming' ? 'Streaming' : 'Ready'}
                </span>
              </div>
            </div>
          )}
        </div>
      </main>
    </Layout>
  );
}
