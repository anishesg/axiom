import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { SignalStrength } from '../components/ui/SignalStrength';

interface CalibrationStep {
  prompt: string;
  description: string;
}

const CALIBRATION_STEPS: CalibrationStep[] = [
  { prompt: "Think 'yes'", description: 'Focus on the affirmative response.' },
  { prompt: "Think 'no'", description: 'Focus on the negative response.' },
  { prompt: 'Relax', description: 'Clear your mind and breathe normally.' },
];

export function Calibration() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [signalLevel, setSignalLevel] = useState(3);

  const totalSteps = CALIBRATION_STEPS.length;
  const step = CALIBRATION_STEPS[currentStep];
  const overallProgress = ((currentStep + progress / 100) / totalSteps) * 100;

  // Simulate calibration progress
  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          return prev;
        }
        return prev + 0.5;
      });
    }, 50);

    return () => clearInterval(interval);
  }, [currentStep]);

  // Simulate signal fluctuation
  useEffect(() => {
    const interval = setInterval(() => {
      setSignalLevel(Math.floor(Math.random() * 2) + 2); // 2-3
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const handleNext = () => {
    if (currentStep < totalSteps - 1) {
      setCurrentStep(currentStep + 1);
      setProgress(0);
    } else {
      // Calibration complete, go to communication hub
      navigate('/communicate');
    }
  };

  // Calculate SVG circle properties
  const radius = 48;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  return (
    <Layout showBackButton backTo="/connect" backLabel="Back" showNav={false}>
      {/* Thin Progress Bar at Top */}
      <div className="fixed top-0 left-0 w-full h-[2px] bg-secondary-container z-50">
        <div
          className="h-full bg-primary transition-all duration-300 ease-in-out"
          style={{ width: `${overallProgress}%` }}
        />
      </div>

      <main className="flex-grow flex flex-col items-center justify-center px-margin-page py-stack-lg max-w-[1200px] mx-auto w-full">
        {/* Step Indicator */}
        <div className="mb-stack-md text-center">
          <span className="text-label-lg text-on-surface-variant tracking-widest uppercase">
            Step {currentStep + 1} of {totalSteps}: Intent Calibration
          </span>
        </div>

        <div className="relative flex flex-col items-center justify-center w-full max-w-2xl text-center space-y-stack-lg">
          {/* Centered Large Prompt */}
          <div className="space-y-stack-sm">
            <h1 className="text-display text-primary transition-all duration-500">
              {step.prompt}
            </h1>
            <p className="text-body-lg text-on-surface-variant max-w-md mx-auto">
              {step.description}
            </p>
          </div>

          {/* Large Circular Progress Indicator */}
          <div className="relative w-64 h-64 md:w-80 md:h-80 flex items-center justify-center">
            {/* Static Outer Ring */}
            <div className="absolute inset-0 rounded-full border border-surface-container-highest" />

            {/* Animated Progress Ring */}
            <svg className="w-full h-full" viewBox="0 0 100 100">
              <circle
                className="text-primary progress-ring-circle"
                cx="50"
                cy="50"
                r={radius}
                fill="transparent"
                stroke="currentColor"
                strokeWidth="1"
                style={{
                  strokeDasharray: circumference,
                  strokeDashoffset: strokeDashoffset,
                }}
              />
            </svg>

            {/* Inner Signal Visualizer */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-48 h-48 md:w-56 md:h-56 rounded-full border border-surface-container flex items-center justify-center animate-pulse-thin">
                <span
                  className="material-symbols-outlined text-primary opacity-20"
                  style={{ fontSize: '48px' }}
                >
                  neurology
                </span>
              </div>
            </div>

            {/* Progress Percentage */}
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-headline-md text-on-surface-variant">
                {Math.round(progress)}%
              </span>
            </div>
          </div>

          {/* Signal Strength Indicator */}
          <div className="flex items-center gap-2 px-4 py-2 bg-surface-container-low rounded-full">
            <SignalStrength level={signalLevel} maxLevel={4} />
            <span className="text-label-sm text-on-surface-variant">
              Signal {signalLevel >= 3 ? 'Stable' : 'Weak'}
            </span>
          </div>
        </div>
      </main>

      {/* Footer Action Bar */}
      <footer className="w-full bg-background border-t border-secondary-container">
        <div className="flex flex-col md:flex-row justify-between items-center px-margin-page py-stack-md max-w-[1200px] mx-auto gap-stack-sm">
          <div className="order-2 md:order-1">
            <p className="text-label-sm text-on-surface-variant">
              {currentStep + 1} of {totalSteps} calibrations complete
            </p>
          </div>
          <div className="order-1 md:order-2 flex flex-col md:flex-row items-center gap-stack-md w-full md:w-auto">
            <button
              onClick={handleNext}
              disabled={progress < 100}
              className={`w-full md:w-auto px-12 min-h-target-min min-w-[200px] bg-primary text-on-primary text-label-lg transition-all flex items-center justify-center gap-2 group ${
                progress < 100
                  ? 'opacity-50 cursor-not-allowed'
                  : 'hover:opacity-90 active:scale-95'
              }`}
            >
              {currentStep < totalSteps - 1 ? 'Next' : 'Complete'}
              <span className="material-symbols-outlined transition-transform group-hover:translate-x-1">
                arrow_forward
              </span>
            </button>
          </div>
        </div>
      </footer>
    </Layout>
  );
}
