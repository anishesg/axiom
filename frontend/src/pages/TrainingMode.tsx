import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { SignalStrength } from '../components/ui/SignalStrength';
import {
  AttemptTracker,
  ExercisePrompt,
  ExerciseSelector,
  type AttemptResult,
  type PromptState,
  type Exercise,
} from '../components/training';
import { useElevenSocket } from '../hooks/useElevenSocket';
import { useSessionStore, type ControlSignal } from '../stores/sessionStore';

// Exercise configurations
const EXERCISES: (Exercise & {
  targetSignal: ControlSignal;
  attempts: number;
  timeWindow: number;
})[] = [
  {
    id: 'double_blink',
    name: 'Double Blink',
    description: 'Blink twice quickly when prompted',
    icon: 'visibility',
    difficulty: 'easy',
    targetSignal: 'double_blink',
    attempts: 10,
    timeWindow: 3,
  },
  {
    id: 'triple_blink',
    name: 'Triple Blink',
    description: 'Blink three times quickly when prompted',
    icon: 'visibility',
    difficulty: 'medium',
    targetSignal: 'triple_blink',
    attempts: 10,
    timeWindow: 4,
  },
  {
    id: 'jaw_clench',
    name: 'Jaw Clench',
    description: 'Clench your jaw firmly when prompted',
    icon: 'sentiment_very_satisfied',
    difficulty: 'easy',
    targetSignal: 'jaw_clench',
    attempts: 10,
    timeWindow: 3,
  },
  {
    id: 'long_clench',
    name: 'Long Jaw Clench',
    description: 'Clench your jaw and hold for 1 second',
    icon: 'sentiment_very_satisfied',
    difficulty: 'hard',
    targetSignal: 'long_jaw_clench',
    attempts: 10,
    timeWindow: 4,
  },
];

type ExercisePhase = 'select' | 'running' | 'complete';

export function TrainingMode() {
  const navigate = useNavigate();
  const { connect, startStreaming } = useElevenSocket();
  const { connectionState, signalStrength, lastControlSignal, lastSignalTimestamp } = useSessionStore();

  // Exercise state
  const [phase, setPhase] = useState<ExercisePhase>('select');
  const [selectedExercise, setSelectedExercise] = useState<typeof EXERCISES[0] | null>(null);
  const [currentAttempt, setCurrentAttempt] = useState(0);
  const [attempts, setAttempts] = useState<AttemptResult[]>([]);
  const [promptState, setPromptState] = useState<PromptState>('ready');
  const [countdown, setCountdown] = useState(3);
  const [timeRemaining, setTimeRemaining] = useState(3);

  // Refs for timing
  const actionStartRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastProcessedSignalRef = useRef<number | null>(null);

  // Connect on mount
  useEffect(() => {
    const init = async () => {
      if (connectionState === 'disconnected') {
        const connected = await connect(true);
        if (connected) {
          await startStreaming();
        }
      }
    };
    init();
  }, []);

  // Cleanup timers
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // Handle signal detection during action phase
  useEffect(() => {
    if (
      promptState !== 'action' ||
      !selectedExercise ||
      !lastControlSignal ||
      !lastSignalTimestamp
    ) {
      return;
    }

    // Prevent processing the same signal twice
    if (lastProcessedSignalRef.current === lastSignalTimestamp) {
      return;
    }

    // Check if signal matches target
    if (lastControlSignal === selectedExercise.targetSignal) {
      lastProcessedSignalRef.current = lastSignalTimestamp;
      handleSuccess();
    }
  }, [lastControlSignal, lastSignalTimestamp, promptState, selectedExercise]);

  const handleSelectExercise = (exerciseId: string) => {
    const exercise = EXERCISES.find((e) => e.id === exerciseId);
    if (!exercise) return;

    setSelectedExercise(exercise);
    setPhase('running');
    setCurrentAttempt(0);
    setAttempts([]);
    startAttempt();
  };

  const startAttempt = useCallback(() => {
    setPromptState('countdown');
    setCountdown(3);

    // Countdown timer
    let count = 3;
    const countdownInterval = setInterval(() => {
      count--;
      setCountdown(count);

      if (count <= 0) {
        clearInterval(countdownInterval);
        startAction();
      }
    }, 1000);
  }, []);

  const startAction = useCallback(() => {
    if (!selectedExercise) return;

    setPromptState('action');
    setTimeRemaining(selectedExercise.timeWindow);
    actionStartRef.current = Date.now();
    lastProcessedSignalRef.current = null;

    // Action timer
    timerRef.current = setInterval(() => {
      if (!selectedExercise) return;

      const elapsed = (Date.now() - (actionStartRef.current || 0)) / 1000;
      const remaining = selectedExercise.timeWindow - elapsed;

      if (remaining <= 0) {
        if (timerRef.current) clearInterval(timerRef.current);
        handleFailure();
      } else {
        setTimeRemaining(remaining);
      }
    }, 100);
  }, [selectedExercise]);

  const handleSuccess = () => {
    if (timerRef.current) clearInterval(timerRef.current);

    // Play success sound
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance('Good!');
      utterance.rate = 1.2;
      speechSynthesis.speak(utterance);
    }

    setPromptState('success');
    setAttempts((prev) => [...prev, 'success']);

    setTimeout(() => {
      nextAttempt();
    }, 1000);
  };

  const handleFailure = () => {
    if (timerRef.current) clearInterval(timerRef.current);

    setPromptState('failure');
    setAttempts((prev) => [...prev, 'failure']);

    setTimeout(() => {
      nextAttempt();
    }, 1000);
  };

  const nextAttempt = useCallback(() => {
    if (!selectedExercise) return;

    const nextAttemptNum = currentAttempt + 1;

    if (nextAttemptNum >= selectedExercise.attempts) {
      // Exercise complete
      setPhase('complete');
      setPromptState('ready');
    } else {
      setCurrentAttempt(nextAttemptNum);
      startAttempt();
    }
  }, [currentAttempt, selectedExercise, startAttempt]);

  const handleRestart = () => {
    if (!selectedExercise) return;
    setCurrentAttempt(0);
    setAttempts([]);
    setPhase('running');
    startAttempt();
  };

  const handleBackToSelect = () => {
    setPhase('select');
    setSelectedExercise(null);
    setAttempts([]);
    setCurrentAttempt(0);
    setPromptState('ready');
  };

  // Calculate final score
  const successCount = attempts.filter((a) => a === 'success').length;
  const finalScore = attempts.length > 0 ? Math.round((successCount / attempts.length) * 100) : 0;

  return (
    <Layout showBackButton backTo="/communicate" backLabel="Back" showNav={false}>
      <main className="flex-grow flex flex-col px-margin-page py-stack-md max-w-[1200px] mx-auto w-full">
        {/* Header */}
        <div className="flex items-center justify-between mb-stack-md">
          <div>
            <h1 className="text-headline-lg text-primary">Training Mode</h1>
            <p className="text-body-md text-on-surface-variant">
              {phase === 'select' && 'Practice your EEG control signals'}
              {phase === 'running' && selectedExercise && `Exercise: ${selectedExercise.name}`}
              {phase === 'complete' && 'Exercise Complete!'}
            </p>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-surface-container-low rounded-full">
            <SignalStrength level={signalStrength} maxLevel={4} />
            <span className="text-label-sm text-on-surface-variant">
              {connectionState === 'streaming' ? 'Live' : 'Offline'}
            </span>
          </div>
        </div>

        {/* Content */}
        <div className="flex-grow flex flex-col items-center justify-center">
          {/* Select Phase */}
          {phase === 'select' && (
            <ExerciseSelector exercises={EXERCISES} onSelect={handleSelectExercise} />
          )}

          {/* Running Phase */}
          {phase === 'running' && selectedExercise && (
            <div className="w-full max-w-lg space-y-8">
              <ExercisePrompt
                state={promptState}
                actionText={selectedExercise.name.toUpperCase()}
                countdown={countdown}
                timeRemaining={timeRemaining}
                timeWindow={selectedExercise.timeWindow}
                icon={selectedExercise.icon}
              />

              <AttemptTracker
                attempts={attempts}
                totalAttempts={selectedExercise.attempts}
                currentAttempt={currentAttempt}
              />

              {/* Tip */}
              <div className="bg-surface-container-low border border-outline-variant rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <span className="material-symbols-outlined text-primary">lightbulb</span>
                  <div>
                    <p className="text-body-md text-on-surface">
                      <strong>Tip:</strong> {selectedExercise.description}
                    </p>
                    <p className="text-body-md text-on-surface-variant mt-1">
                      Wait for the green checkmark before preparing for the next attempt.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Complete Phase */}
          {phase === 'complete' && (
            <div className="text-center space-y-8">
              <div className="w-32 h-32 mx-auto rounded-full bg-primary flex items-center justify-center">
                <span className="material-symbols-outlined text-on-primary text-[64px]">
                  {finalScore >= 80 ? 'emoji_events' : finalScore >= 50 ? 'thumb_up' : 'trending_up'}
                </span>
              </div>

              <div>
                <h2 className="text-display text-primary">{finalScore}%</h2>
                <p className="text-body-lg text-on-surface-variant">
                  {finalScore >= 80
                    ? 'Excellent! You\'re doing great!'
                    : finalScore >= 50
                      ? 'Good progress! Keep practicing.'
                      : 'Keep practicing to improve your accuracy.'}
                </p>
              </div>

              <div className="flex items-center justify-center gap-4">
                <button
                  onClick={handleRestart}
                  className="px-8 min-h-target-min bg-primary text-on-primary text-label-lg rounded flex items-center gap-2 hover:opacity-90"
                >
                  <span className="material-symbols-outlined">replay</span>
                  Try Again
                </button>
                <button
                  onClick={handleBackToSelect}
                  className="px-8 min-h-target-min border border-outline text-on-surface text-label-lg rounded flex items-center gap-2 hover:bg-surface-container"
                >
                  <span className="material-symbols-outlined">menu</span>
                  Other Exercises
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer controls during exercise */}
        {phase === 'running' && (
          <footer className="mt-8 pt-4 border-t border-outline-variant">
            <div className="flex items-center justify-between">
              <button
                onClick={handleBackToSelect}
                className="px-6 py-2 text-label-lg text-on-surface-variant hover:text-on-surface"
              >
                End Exercise
              </button>
              <div className="text-label-sm text-on-surface-variant">
                Press the signal when you see "NOW!"
              </div>
            </div>
          </footer>
        )}
      </main>
    </Layout>
  );
}
