import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { SignalIndicator } from '../components/ui/SignalIndicator';
import { SignalStrength } from '../components/ui/SignalStrength';

type ConnectionState = 'disconnected' | 'connecting' | 'connected';

export function ConnectDevice() {
  const navigate = useNavigate();
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');

  const handleConnect = async () => {
    if (connectionState === 'connecting') return;

    setConnectionState('connecting');

    // Simulate connection process (will be replaced with real WebSocket connection)
    setTimeout(() => {
      setConnectionState('connected');
      // Navigate to calibration after successful connection
      setTimeout(() => {
        navigate('/calibrate');
      }, 1000);
    }, 3000);
  };

  const getButtonContent = () => {
    switch (connectionState) {
      case 'connecting':
        return (
          <>
            <span className="material-symbols-outlined animate-spin">sync</span>
            <span>Searching...</span>
          </>
        );
      case 'connected':
        return (
          <>
            <span className="material-symbols-outlined">check_circle</span>
            <span>Connected</span>
          </>
        );
      default:
        return <span>Connect</span>;
    }
  };

  const getStatusLabel = () => {
    switch (connectionState) {
      case 'connecting':
        return 'Searching...';
      case 'connected':
        return 'Connected';
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
            status={connectionState === 'connected' ? 'connected' : 'disconnected'}
            label={getStatusLabel()}
            size="sm"
          />

          {/* Header Section */}
          <div className="text-center space-y-2">
            <h1 className="text-headline-lg text-primary tracking-tight">
              Connect your headband.
            </h1>
            <p className="text-body-md text-on-surface-variant max-w-[320px] mx-auto">
              Initialize your neural interface to begin communication.
            </p>
          </div>

          {/* Central Pulsing Circle */}
          <div className="relative w-48 h-48 flex items-center justify-center">
            {/* Pulse rings */}
            {connectionState === 'connecting' && (
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
                connectionState === 'connected'
                  ? 'border-primary bg-primary'
                  : 'border-primary'
              }`}
            >
              <span
                className={`material-symbols-outlined transition-all duration-300 ${
                  connectionState === 'connected'
                    ? 'text-on-primary text-[64px]'
                    : 'text-primary text-[64px]'
                }`}
                style={{ fontVariationSettings: "'wght' 200" }}
              >
                {connectionState === 'connected' ? 'check' : 'radio_button_checked'}
              </span>
            </div>
          </div>

          {/* Action Section */}
          <div className="w-full space-y-stack-sm text-center">
            <button
              onClick={handleConnect}
              disabled={connectionState === 'connected'}
              className={`w-full min-h-target-min bg-primary text-on-primary text-label-lg uppercase tracking-widest transition-all flex items-center justify-center gap-2 ${
                connectionState === 'connecting' ? 'opacity-80' : ''
              } ${
                connectionState === 'connected'
                  ? 'bg-primary/50 cursor-not-allowed'
                  : 'hover:opacity-90 active:scale-[0.98]'
              }`}
            >
              {getButtonContent()}
            </button>
            <p className="text-label-sm text-on-surface-variant px-stack-sm">
              Ensure your EEG headband is powered on and within range.
            </p>
          </div>

          {/* Signal Strength */}
          <div
            className={`w-full pt-stack-sm border-t border-surface-container flex justify-between items-center transition-opacity ${
              connectionState === 'connected' ? 'opacity-100' : 'opacity-40'
            }`}
          >
            <span className="text-label-sm text-on-surface-variant">SIGNAL STRENGTH</span>
            <SignalStrength
              level={connectionState === 'connected' ? 3 : 0}
              maxLevel={4}
            />
          </div>
        </div>
      </main>
    </Layout>
  );
}
