import { useEffect } from 'react';
import { Layout } from '../components/layout/Layout';
import { SignalStrength } from '../components/ui/SignalStrength';
import {
  CognitiveStateCard,
  BandPowerBars,
  SignalTimeline,
  ControlSignalLog,
  SessionStats,
  EEGWaveform,
} from '../components/dashboard';
import { useElevenSocket } from '../hooks/useElevenSocket';
import { useSessionStore } from '../stores/sessionStore';

export function Dashboard() {
  const { connect, ensureWebSocket, startStreaming, disconnect } = useElevenSocket();
  const {
    connectionState,
    signalStrength,
    brainState,
    bandPowers,
    signalHistory,
    lastControlSignal,
    lastSignalTimestamp,
    sessionStartTime,
    totalSignalsDetected,
    learningMetrics,
    startSession,
  } = useSessionStore();

  // Ensure WebSocket is connected when mounting
  // This allows the Dashboard to receive events from an existing session
  useEffect(() => {
    const init = async () => {
      startSession();
      // Always ensure WebSocket is connected to receive events
      // This works whether session is new or already exists
      await ensureWebSocket();
    };
    init();

    return () => {
      // Don't disconnect on unmount - let user navigate back
    };
  }, []);

  const isConnected = connectionState === 'streaming' || connectionState === 'connected';

  return (
    <Layout showBackButton backTo="/communicate" backLabel="Back" showNav={false}>
      <main className="flex-grow px-margin-page py-stack-md max-w-[1400px] mx-auto w-full">
        {/* Header */}
        <div className="flex items-center justify-between mb-stack-md">
          <div>
            <h1 className="text-headline-lg text-primary">Brain Activity Dashboard</h1>
            <p className="text-body-md text-on-surface-variant">
              Real-time EEG visualization and monitoring
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-4 py-2 bg-surface-container-low rounded-full">
              <SignalStrength level={signalStrength} maxLevel={4} />
              <span className="text-label-sm text-on-surface-variant">
                {connectionState === 'streaming' ? 'Live' : connectionState}
              </span>
            </div>
            <button
              onClick={() => (isConnected ? disconnect() : connect(true).then(() => startStreaming()))}
              className={`px-6 py-2 text-label-lg rounded transition-all ${
                isConnected
                  ? 'border border-outline text-on-surface-variant hover:bg-surface-container'
                  : 'bg-primary text-on-primary hover:opacity-90'
              }`}
            >
              {isConnected ? 'Disconnect' : 'Connect'}
            </button>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Cognitive State & Control Signals */}
          <div className="space-y-6">
            <CognitiveStateCard
              focus={brainState.focus}
              relaxation={brainState.relaxation}
              engagement={brainState.engagement}
            />
            <ControlSignalLog
              history={signalHistory}
              lastSignal={lastControlSignal}
              lastTimestamp={lastSignalTimestamp}
            />
          </div>

          {/* Center Column - Band Powers & Timeline */}
          <div className="space-y-6">
            <BandPowerBars bandPowers={bandPowers} />
            <SessionStats
              sessionStartTime={sessionStartTime}
              totalSignals={totalSignalsDetected}
              accuracy={learningMetrics.accuracy > 0 ? learningMetrics.accuracy : undefined}
            />
          </div>

          {/* Right Column - Additional Info */}
          <div className="space-y-6">
            {/* Raw Values Card */}
            <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-6">
              <h3 className="text-label-lg text-on-surface-variant uppercase tracking-widest mb-4">
                Raw Values
              </h3>
              <div className="space-y-2 font-mono text-sm">
                <div className="flex justify-between">
                  <span className="text-on-surface-variant">Focus</span>
                  <span className="text-on-surface">{brainState.focus.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-on-surface-variant">Relaxation</span>
                  <span className="text-on-surface">{brainState.relaxation.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-on-surface-variant">Engagement</span>
                  <span className="text-on-surface">{brainState.engagement.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-on-surface-variant">Cognitive Load</span>
                  <span className="text-on-surface">{brainState.cognitive_load.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-on-surface-variant">Valence</span>
                  <span className="text-on-surface">{brainState.valence.toFixed(3)}</span>
                </div>
              </div>
            </div>

            {/* Status Indicators */}
            <div className="bg-surface-container-lowest border border-outline-variant rounded-lg p-6">
              <h3 className="text-label-lg text-on-surface-variant uppercase tracking-widest mb-4">
                Detection Status
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <StatusIndicator label="Jaw Clench" active={brainState.jaw_clench} />
                <StatusIndicator label="Double Clench" active={brainState.double_clench} />
                <StatusIndicator label="Context Switch" active={brainState.context_switch} />
                <StatusIndicator label="Error Response" active={brainState.error_response} />
              </div>
            </div>
          </div>
        </div>

        {/* Raw EEG Waveform - Full Width */}
        <div className="mt-6">
          <EEGWaveform />
        </div>

        {/* Signal Timeline - Full Width */}
        <div className="mt-6">
          <SignalTimeline history={signalHistory} duration={30} />
        </div>
      </main>
    </Layout>
  );
}

function StatusIndicator({ label, active }: { label: string; active: boolean }) {
  return (
    <div
      className={`px-3 py-2 rounded text-center text-label-sm transition-colors ${
        active
          ? 'bg-primary text-on-primary'
          : 'bg-surface-container text-on-surface-variant'
      }`}
    >
      {label}
    </div>
  );
}
