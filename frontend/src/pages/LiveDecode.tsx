import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { SignalStrength } from '../components/ui/SignalStrength';
import { useElevenSocket } from '../hooks/useElevenSocket';
import { useSessionStore } from '../stores/sessionStore';

interface DecodedWord {
  id: number;
  text: string;
  confidence: number;
  timestamp: number;
}

export function LiveDecode() {
  const navigate = useNavigate();
  const [decodedWords, setDecodedWords] = useState<DecodedWord[]>([]);
  const [currentIntent, setCurrentIntent] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [showCursor, setShowCursor] = useState(true);
  const wordIdRef = useRef(0);
  const outputRef = useRef<HTMLDivElement>(null);

  const { connect, startStreaming, disconnect } = useElevenSocket();
  const { connectionState, signalStrength, controlSignal, brainState } = useSessionStore();

  // Cursor blink effect
  useEffect(() => {
    const interval = setInterval(() => {
      setShowCursor(prev => !prev);
    }, 530);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [decodedWords]);

  // Handle control signals -> map to words/intents
  useEffect(() => {
    if (!controlSignal || !isListening) return;

    const intentMap: Record<string, string> = {
      'double_blink': 'Yes',
      'triple_blink': 'No',
      'jaw_clench': 'Help',
      'long_jaw_clench': 'Stop',
    };

    const word = intentMap[controlSignal];
    if (word) {
      setCurrentIntent(controlSignal);
      addWord(word, 0.95);

      // Clear intent indicator after animation
      setTimeout(() => setCurrentIntent(null), 500);
    }
  }, [controlSignal, isListening]);

  const addWord = (text: string, confidence: number) => {
    const newWord: DecodedWord = {
      id: wordIdRef.current++,
      text,
      confidence,
      timestamp: Date.now(),
    };
    setDecodedWords(prev => [...prev, newWord]);

    // Speak the word
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      speechSynthesis.speak(utterance);
    }
  };

  const handleStart = async () => {
    if (connectionState === 'disconnected') {
      await connect(true); // Use simulation
    }
    if (connectionState === 'connected' || connectionState === 'disconnected') {
      await startStreaming();
    }
    setIsListening(true);
  };

  const handleStop = () => {
    setIsListening(false);
  };

  const handleClear = () => {
    setDecodedWords([]);
  };

  const handleSpeak = () => {
    if ('speechSynthesis' in window && decodedWords.length > 0) {
      const fullText = decodedWords.map(w => w.text).join(' ');
      const utterance = new SpeechSynthesisUtterance(fullText);
      utterance.rate = 0.85;
      speechSynthesis.speak(utterance);
    }
  };

  // Demo mode - simulate random words for testing
  const handleDemo = () => {
    const demoWords = ['Hello', 'I', 'need', 'water', 'please', 'Thank', 'you'];
    let index = 0;

    setIsListening(true);
    const interval = setInterval(() => {
      if (index >= demoWords.length) {
        clearInterval(interval);
        setIsListening(false);
        return;
      }
      addWord(demoWords[index], 0.85 + Math.random() * 0.15);
      index++;
    }, 1500);
  };

  return (
    <Layout showBackButton backTo="/communicate" backLabel="Back" showNav={false}>
      <main className="flex-grow flex flex-col px-margin-page py-stack-md max-w-[1200px] mx-auto w-full">
        {/* Header */}
        <div className="flex items-center justify-between mb-stack-md">
          <div>
            <h1 className="text-headline-lg text-primary">Live Decode</h1>
            <p className="text-body-md text-on-surface-variant">
              Brain signals → Text → Speech
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-4 py-2 bg-surface-container-low rounded-full">
              <SignalStrength level={signalStrength} maxLevel={4} />
              <span className="text-label-sm text-on-surface-variant">
                {connectionState === 'streaming' ? 'Live' : 'Offline'}
              </span>
            </div>
          </div>
        </div>

        {/* Main Output Area */}
        <div className="flex-grow flex flex-col bg-surface-container-lowest border border-outline-variant rounded-lg overflow-hidden">
          {/* Output Text */}
          <div
            ref={outputRef}
            className="flex-grow p-8 overflow-y-auto min-h-[300px]"
          >
            {decodedWords.length === 0 ? (
              <div className="h-full flex items-center justify-center">
                <p className="text-body-lg text-on-surface-variant opacity-50">
                  {isListening ? 'Listening for brain signals...' : 'Press Start to begin decoding'}
                </p>
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {decodedWords.map((word, index) => (
                  <span
                    key={word.id}
                    className="text-display text-primary animate-fade-in"
                    style={{
                      animationDelay: `${index * 50}ms`,
                      opacity: word.confidence,
                    }}
                  >
                    {word.text}
                  </span>
                ))}
                {isListening && (
                  <span
                    className={`text-display text-primary transition-opacity ${
                      showCursor ? 'opacity-100' : 'opacity-0'
                    }`}
                  >
                    |
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Intent Indicator */}
          {currentIntent && (
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <div className="px-8 py-4 bg-primary text-on-primary rounded-lg text-headline-md animate-pulse">
                {currentIntent.replace('_', ' ').toUpperCase()}
              </div>
            </div>
          )}

          {/* Brain State Indicators */}
          <div className="px-8 py-4 border-t border-outline-variant bg-surface-container">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-8">
                <div className="flex items-center gap-2">
                  <span className="text-label-sm text-on-surface-variant">Focus</span>
                  <div className="w-24 h-2 bg-surface-container-highest rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${(brainState?.focus || 0) * 100}%` }}
                    />
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-label-sm text-on-surface-variant">Relax</span>
                  <div className="w-24 h-2 bg-surface-container-highest rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${(brainState?.relaxation || 0) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
              <div className="text-label-sm text-on-surface-variant">
                {decodedWords.length} words decoded
              </div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-center gap-4 mt-stack-md">
          {!isListening ? (
            <button
              onClick={handleStart}
              className="px-12 min-h-target-min bg-primary text-on-primary text-label-lg rounded flex items-center gap-2 hover:opacity-90 active:scale-95 transition-all"
            >
              <span className="material-symbols-outlined">mic</span>
              Start Listening
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="px-12 min-h-target-min bg-error text-on-error text-label-lg rounded flex items-center gap-2 hover:opacity-90 active:scale-95 transition-all"
            >
              <span className="material-symbols-outlined">stop</span>
              Stop
            </button>
          )}

          <button
            onClick={handleSpeak}
            disabled={decodedWords.length === 0}
            className="px-8 min-h-target-min border border-primary text-primary text-label-lg rounded flex items-center gap-2 hover:bg-surface-container active:scale-95 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <span className="material-symbols-outlined">volume_up</span>
            Speak All
          </button>

          <button
            onClick={handleClear}
            disabled={decodedWords.length === 0}
            className="px-8 min-h-target-min border border-outline text-on-surface-variant text-label-lg rounded flex items-center gap-2 hover:bg-surface-container active:scale-95 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <span className="material-symbols-outlined">backspace</span>
            Clear
          </button>

          <button
            onClick={handleDemo}
            disabled={isListening}
            className="px-8 min-h-target-min border border-outline text-on-surface-variant text-label-lg rounded flex items-center gap-2 hover:bg-surface-container active:scale-95 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <span className="material-symbols-outlined">play_circle</span>
            Demo
          </button>
        </div>

        {/* Quick Phrases */}
        <div className="mt-stack-md">
          <p className="text-label-sm text-on-surface-variant mb-2">Quick phrases (click to add):</p>
          <div className="flex flex-wrap gap-2">
            {['Hello', 'Yes', 'No', 'Help', 'Thank you', 'Water', 'Pain', 'Tired'].map(phrase => (
              <button
                key={phrase}
                onClick={() => addWord(phrase, 1.0)}
                className="px-4 py-2 bg-surface-container text-on-surface text-label-lg rounded hover:bg-surface-container-high active:scale-95 transition-all"
              >
                {phrase}
              </button>
            ))}
          </div>
        </div>
      </main>
    </Layout>
  );
}
