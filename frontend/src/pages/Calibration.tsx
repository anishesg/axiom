import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { SignalStrength } from '../components/ui/SignalStrength';
import { useElevenSocket } from '../hooks/useElevenSocket';
import { useSessionStore } from '../stores/sessionStore';

interface CalibrationStepConfig {
  id: string;
  prompt: string;
  description: string;
  duration: number; // seconds
  showNowPrompt?: boolean; // Show "NOW" prompts for action steps
}

// Research-style calibration steps matching backend EnhancedCalibrationSession.STEPS
const CALIBRATION_STEPS: CalibrationStepConfig[] = [
  {
    id: 'relax_baseline',
    prompt: 'Relax Baseline',
    description: 'Close your eyes and relax. We\'re measuring your alpha baseline.',
    duration: 10,
  },
  {
    id: 'focus_baseline',
    prompt: 'Focus Baseline',
    description: 'Focus intently on the dot. Think hard about a math problem.',
    duration: 8,
  },
  {
    id: 'natural_blinks',
    prompt: 'Natural Blinks',
    description: 'Blink naturally, don\'t try to blink more or less.',
    duration: 20,
  },
  {
    id: 'deliberate_blinks',
    prompt: 'Deliberate Blinks',
    description: 'Blink firmly when you see "NOW".',
    duration: 30,
    showNowPrompt: true,
  },
  {
    id: 'jaw_clenches',
    prompt: 'Jaw Clenches',
    description: 'Clench your jaw firmly when you see "NOW".',
    duration: 30,
    showNowPrompt: true,
  },
];

export function Calibration() {
  const navigate = useNavigate();
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [stepProgress, setStepProgress] = useState(0);
  const [stepStartTime, setStepStartTime] = useState<number | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);
  const [status, setStatus] = useState<'idle' | 'running' | 'waiting' | 'complete' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [warningMessage, setWarningMessage] = useState<string | null>(null);
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);
  const [wasConnected, setWasConnected] = useState(false);
  const [lastErrorRecoverable, setLastErrorRecoverable] = useState(true);

  const {
    connect,
    startStreaming,
    startEnhancedCalibration,
    startEnhancedCalibrationStep,
    endEnhancedCalibrationStep,
    trainVQTokenizer,
    completeEnhancedCalibration,
    error,
    clearError,
  } = useElevenSocket();

  const { connectionState, signalStrength } = useSessionStore();

  const totalSteps = CALIBRATION_STEPS.length;
  const currentStep = CALIBRATION_STEPS[currentStepIndex];
  const overallProgress = ((currentStepIndex + stepProgress / 100) / totalSteps) * 100;

  // Initialize enhanced calibration with VQ training
  useEffect(() => {
    const init = async () => {
      // If not connected, connect first (with simulation for testing)
      if (connectionState === 'disconnected') {
        const connected = await connect(true); // Use simulation
        if (!connected) {
          setStatus('error');
          setErrorMessage('Failed to connect to EEG device');
          return;
        }
      }

      // Start streaming if not already
      if (connectionState === 'connected') {
        await startStreaming();
      }

      // Start enhanced calibration session (with VQ/RVQ support)
      const result = await startEnhancedCalibration('default');
      if (!result) {
        setStatus('error');
        setErrorMessage('Failed to start enhanced calibration');
        return;
      }

      setStatus('idle');
    };

    init();
  }, []);

  // Handle errors from the hook
  useEffect(() => {
    if (error) {
      setStatus('error');
      setErrorMessage(error.action ? `${error.message}. ${error.action}` : error.message);
      setLastErrorRecoverable(error.recoverable ?? true);
    }
  }, [error]);

  // Detect connection drop during calibration
  useEffect(() => {
    if (connectionState === 'streaming') {
      setWasConnected(true);
    } else if (wasConnected && connectionState !== 'streaming' && status === 'running') {
      setStatus('error');
      setErrorMessage('Connection lost during calibration. Data may be incomplete.');
      setLastErrorRecoverable(true);
    }
  }, [connectionState, wasConnected, status]);

  // Progress timer for current step
  useEffect(() => {
    if (status !== 'running' || !stepStartTime) return;

    const interval = setInterval(() => {
      const elapsed = (Date.now() - stepStartTime) / 1000;
      const progress = Math.min(100, (elapsed / currentStep.duration) * 100);
      setStepProgress(progress);

      // Show "NOW" prompts for action steps
      if (currentStep.showNowPrompt) {
        // Show prompt every 2 seconds
        const shouldShow = Math.floor(elapsed) % 2 === 1 && elapsed < currentStep.duration;
        setShowPrompt(shouldShow);
      }

      // Step complete
      if (progress >= 100) {
        clearInterval(interval);
        handleStepComplete();
      }
    }, 100);

    return () => clearInterval(interval);
  }, [status, stepStartTime, currentStep]);

  const startStep = useCallback(async () => {
    setStatus('running');
    setStepProgress(0);
    setShowPrompt(false);

    // Use enhanced calibration step
    const result = await startEnhancedCalibrationStep(currentStep.id);

    if (!result) {
      setStatus('error');
      setErrorMessage('Failed to start calibration step');
      return;
    }

    setStepStartTime(Date.now());
  }, [currentStep, startEnhancedCalibrationStep]);

  const handleStepComplete = useCallback(async () => {
    setStatus('waiting');
    setShowPrompt(false);
    setWarningMessage(null);

    // End the current step on the server and check results
    const stepResult = await endEnhancedCalibrationStep();

    if (!stepResult) {
      setStatus('error');
      setErrorMessage('Failed to complete step - please retry');
      setLastErrorRecoverable(true);
      return;
    }

    // Check data quality from results
    const results = stepResult.results;
    if (results?.warning) {
      setWarningMessage(results.warning);
    }

    // For state steps, warn if data quality is low
    if (results && !results.sufficient_data && currentStepIndex < 4) {
      setWarningMessage(`Only ${results.samples_collected || 0} samples collected. Consider repeating this step.`);
    }

    // Mark step as completed
    setCompletedSteps(prev => [...prev, currentStep.id]);

    // Check if this was the last step
    if (currentStepIndex >= totalSteps - 1) {
      // Train the VQ tokenizer with timeout
      setWarningMessage('Training personalized model...');

      const trainingTimeout = new Promise<null>((_, reject) =>
        setTimeout(() => reject(new Error('Training timeout')), 60000)
      );

      try {
        const trainResult = await Promise.race([
          trainVQTokenizer(true), // Use RVQ
          trainingTimeout
        ]);

        if (!trainResult) {
          setStatus('error');
          setErrorMessage('Training failed - insufficient calibration data. Please restart and complete all state steps.');
          setLastErrorRecoverable(false);
          return;
        }
      } catch (err) {
        if (err instanceof Error && err.message === 'Training timeout') {
          setStatus('error');
          setErrorMessage('Training is taking too long. Please restart calibration.');
          setLastErrorRecoverable(false);
          return;
        }
        throw err;
      }

      setWarningMessage(null);

      // Complete calibration and save profile
      const result = await completeEnhancedCalibration();
      if (result) {
        setStatus('complete');
        setTimeout(() => {
          navigate('/communicate');
        }, 1500);
      } else {
        setStatus('error');
        setErrorMessage('Failed to complete calibration');
        setLastErrorRecoverable(false);
      }
    }
    // Keep status as 'waiting' for non-final steps so user clicks "Next" to advance
  }, [currentStepIndex, currentStep, totalSteps, endEnhancedCalibrationStep, trainVQTokenizer, completeEnhancedCalibration, navigate]);

  const handleNext = () => {
    if (status === 'idle') {
      startStep();
    } else if (status === 'waiting') {
      setCurrentStepIndex(currentStepIndex + 1);
      setStepProgress(0);
      setStatus('idle');
    }
  };

  const handleRetry = async () => {
    clearError();
    setErrorMessage(null);
    setWarningMessage(null);

    // If we have completed steps and error is recoverable, try to resume
    if (completedSteps.length > 0 && lastErrorRecoverable) {
      setStatus('waiting');
      // Don't reset currentStepIndex - let user continue from where they left off
    } else {
      // Full restart needed
      setStatus('idle');
      setCurrentStepIndex(0);
      setStepProgress(0);
      setCompletedSteps([]);
      setWasConnected(false);

      // Re-initialize calibration
      const result = await startEnhancedCalibration('default');
      if (!result) {
        setStatus('error');
        setErrorMessage('Failed to restart calibration');
        setLastErrorRecoverable(true);
      }
    }
  };

  // SVG circle properties
  const radius = 48;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - (stepProgress / 100) * circumference;

  return (
    <Layout showBackButton backTo="/connect" backLabel="Back" showNav={false}>
      {/* Progress Bar */}
      <div className="fixed top-0 left-0 w-full h-[2px] bg-secondary-container z-50">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${overallProgress}%` }}
        />
      </div>

      <main className="flex-grow flex flex-col items-center justify-center px-margin-page py-stack-lg max-w-[1200px] mx-auto w-full">
        {/* Step Indicator */}
        <div className="mb-stack-md text-center">
          <span className="text-label-lg text-on-surface-variant tracking-widest uppercase">
            Step {currentStepIndex + 1} of {totalSteps}: {currentStep.id.replace('_', ' ')}
          </span>
        </div>

        {/* Error State */}
        {status === 'error' && (
          <div className="w-full max-w-md mb-stack-md p-6 border border-error bg-error-container text-on-error-container rounded-lg">
            <div className="flex items-start gap-4">
              <span className="material-symbols-outlined text-error text-[24px]">error</span>
              <div className="flex-1">
                <h3 className="text-headline-md font-semibold mb-2">Calibration Error</h3>
                <p className="text-body-md mb-4">{errorMessage || 'An unknown error occurred'}</p>
                {completedSteps.length > 0 && (
                  <p className="text-body-sm text-on-error-container/70 mb-4">
                    Completed steps: {completedSteps.join(', ')}
                  </p>
                )}
                <button
                  onClick={handleRetry}
                  className="px-6 py-2 bg-error text-on-error text-label-lg rounded hover:opacity-90"
                >
                  {lastErrorRecoverable && completedSteps.length > 0 ? 'Resume' : 'Restart'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Warning Message */}
        {warningMessage && status !== 'error' && (
          <div className="w-full max-w-md mb-stack-md p-4 border border-tertiary bg-tertiary-container text-on-tertiary-container rounded-lg">
            <div className="flex items-center gap-3">
              <span className="material-symbols-outlined text-tertiary text-[20px]">info</span>
              <p className="text-body-md">{warningMessage}</p>
            </div>
          </div>
        )}

        {/* Complete State */}
        {status === 'complete' && (
          <div className="text-center space-y-stack-md">
            <span className="material-symbols-outlined text-primary text-[80px]">
              check_circle
            </span>
            <h1 className="text-display text-primary">Calibration Complete</h1>
            <p className="text-body-lg text-on-surface-variant">
              Your personalized settings have been saved. Redirecting...
            </p>
          </div>
        )}

        {/* Normal Calibration UI */}
        {status !== 'error' && status !== 'complete' && (
          <div className="relative flex flex-col items-center justify-center w-full max-w-2xl text-center space-y-stack-lg">
            {/* Prompt */}
            <div className="space-y-stack-sm">
              <h1 className="text-display text-primary transition-all duration-500">
                {showPrompt ? 'NOW' : currentStep.prompt}
              </h1>
              <p className="text-body-lg text-on-surface-variant max-w-md mx-auto">
                {currentStep.description}
              </p>
            </div>

            {/* Progress Ring */}
            <div className="relative w-64 h-64 md:w-80 md:h-80 flex items-center justify-center">
              <div className="absolute inset-0 rounded-full border border-surface-container-highest" />

              <svg className="w-full h-full" viewBox="0 0 100 100">
                <circle
                  className="text-primary"
                  cx="50"
                  cy="50"
                  r={radius}
                  fill="transparent"
                  stroke="currentColor"
                  strokeWidth="1"
                  strokeLinecap="round"
                  style={{
                    strokeDasharray: circumference,
                    strokeDashoffset: strokeDashoffset,
                    transition: 'stroke-dashoffset 0.1s linear',
                    transform: 'rotate(-90deg)',
                    transformOrigin: '50% 50%',
                  }}
                />
              </svg>

              {/* Inner circle */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className={`w-48 h-48 md:w-56 md:h-56 rounded-full border border-surface-container flex items-center justify-center ${
                  status === 'running' ? 'animate-pulse' : ''
                }`}>
                  <span
                    className={`material-symbols-outlined text-primary ${showPrompt ? 'opacity-100' : 'opacity-20'}`}
                    style={{ fontSize: '48px' }}
                  >
                    {showPrompt ? 'radio_button_checked' : 'neurology'}
                  </span>
                </div>
              </div>

              {/* Progress text */}
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-headline-md text-on-surface-variant">
                  {status === 'running' ? `${Math.round(stepProgress)}%` : ''}
                  {status === 'idle' ? 'Ready' : ''}
                  {status === 'waiting' ? 'Done!' : ''}
                </span>
              </div>
            </div>

            {/* Signal Strength */}
            <div className="flex items-center gap-2 px-4 py-2 bg-surface-container-low rounded-full">
              <SignalStrength level={signalStrength} maxLevel={4} />
              <span className="text-label-sm text-on-surface-variant">
                {connectionState === 'streaming' ? 'Signal Active' : 'Connecting...'}
              </span>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      {status !== 'complete' && status !== 'error' && (
        <footer className="w-full bg-background border-t border-secondary-container">
          <div className="flex flex-col md:flex-row justify-between items-center px-margin-page py-stack-md max-w-[1200px] mx-auto gap-stack-sm">
            <div className="order-2 md:order-1 flex items-center gap-4">
              <p className="text-label-sm text-on-surface-variant">
                {status === 'running' ? 'Collecting data...' : ''}
                {status === 'idle' ? 'Press Start to begin this step' : ''}
                {status === 'waiting' ? 'Step complete! Press Next to continue' : ''}
              </p>
              <button
                onClick={() => navigate('/dashboard')}
                className="text-on-surface-variant hover:text-primary text-label-sm underline"
              >
                Skip calibration
              </button>
            </div>
            <div className="order-1 md:order-2">
              <button
                onClick={handleNext}
                disabled={status === 'running'}
                className={`px-12 min-h-target-min min-w-[200px] bg-primary text-on-primary text-label-lg transition-all flex items-center justify-center gap-2 ${
                  status === 'running'
                    ? 'opacity-50 cursor-not-allowed'
                    : 'hover:opacity-90 active:scale-95'
                }`}
              >
                {status === 'idle' && 'Start'}
                {status === 'running' && 'Recording...'}
                {status === 'waiting' && (currentStepIndex < totalSteps - 1 ? 'Next' : 'Complete')}
                <span className="material-symbols-outlined">
                  {status === 'running' ? 'pending' : 'arrow_forward'}
                </span>
              </button>
            </div>
          </div>
        </footer>
      )}
    </Layout>
  );
}
