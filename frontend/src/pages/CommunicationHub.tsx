import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { SignalStrength } from '../components/ui/SignalStrength';
import { ConfirmationModal } from '../components/ui/ConfirmationModal';
import { useSessionStore } from '../stores/sessionStore';
import { useElevenSocket } from '../hooks/useElevenSocket';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'partner';
  timestamp: Date;
}

interface Phrase {
  id: string;
  text: string;
  category: 'common' | 'needs';
}

const PHRASES: Phrase[] = [
  { id: '1', text: 'Yes', category: 'common' },
  { id: '2', text: 'No', category: 'common' },
  { id: '3', text: 'Help', category: 'common' },
  { id: '4', text: 'Thank you', category: 'common' },
  { id: '5', text: "I need...", category: 'common' },
  { id: '6', text: 'More', category: 'common' },
  { id: '7', text: 'Water', category: 'needs' },
  { id: '8', text: 'Food', category: 'needs' },
  { id: '9', text: 'Rest', category: 'needs' },
  { id: '10', text: 'Medicine', category: 'needs' },
  { id: '11', text: 'Pain', category: 'needs' },
  { id: '12', text: 'Call nurse', category: 'needs' },
];

const INITIAL_MESSAGES: Message[] = [
  {
    id: '1',
    text: 'How are you feeling today?',
    sender: 'partner',
    timestamp: new Date(),
  },
  {
    id: '2',
    text: 'I am well.',
    sender: 'user',
    timestamp: new Date(),
  },
];

export function CommunicationHub() {
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES);
  const [activeTab, setActiveTab] = useState<'common' | 'needs'>('common');
  const [selectedPhrase, setSelectedPhrase] = useState<Phrase | null>(null);
  const [focusedPhraseId, setFocusedPhraseId] = useState<string | null>(null);
  const [showConfirmation, setShowConfirmation] = useState(false);

  // Axiom connectivity
  const { connect } = useElevenSocket();
  const { connectionState, signalStrength, brainState, lastControlSignal, lastSignalTimestamp } = useSessionStore();

  // Connect to axiom server on mount
  useEffect(() => {
    connect();
  }, [connect]);

  // Handle control signals from EEG
  useEffect(() => {
    if (!lastControlSignal || !lastSignalTimestamp) return;

    // Only react to recent signals (within last 2 seconds)
    const age = Date.now() - lastSignalTimestamp;
    if (age > 2000) return;

    if (showConfirmation) {
      // In confirmation mode
      if (lastControlSignal === 'double_blink' || lastControlSignal === 'jaw_clench') {
        handleConfirm();
      } else if (lastControlSignal === 'triple_blink' || lastControlSignal === 'long_jaw_clench') {
        handleCancel();
      }
    } else if (focusedPhraseId) {
      // Phrase is focused - select it on jaw clench
      if (lastControlSignal === 'jaw_clench') {
        const phrase = PHRASES.find(p => p.id === focusedPhraseId);
        if (phrase) handlePhraseSelect(phrase);
      }
    }
  }, [lastControlSignal, lastSignalTimestamp]);

  const filteredPhrases = PHRASES.filter((p) => p.category === activeTab);

  const handlePhraseSelect = (phrase: Phrase) => {
    setSelectedPhrase(phrase);
    setShowConfirmation(true);
  };

  const handleConfirm = () => {
    if (selectedPhrase) {
      const newMessage: Message = {
        id: Date.now().toString(),
        text: selectedPhrase.text,
        sender: 'user',
        timestamp: new Date(),
      };
      setMessages([...messages, newMessage]);

      // Speak the phrase using Web Speech API
      if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(selectedPhrase.text);
        utterance.rate = 0.9;
        speechSynthesis.speak(utterance);
      }
    }
    setShowConfirmation(false);
    setSelectedPhrase(null);
  };

  const handleCancel = () => {
    setShowConfirmation(false);
    setSelectedPhrase(null);
  };

  const handleEmergency = () => {
    // Immediate emergency action
    const emergencyMessage: Message = {
      id: Date.now().toString(),
      text: 'EMERGENCY HELP NEEDED',
      sender: 'user',
      timestamp: new Date(),
    };
    setMessages([...messages, emergencyMessage]);

    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance('Emergency help needed');
      utterance.rate = 1.2;
      utterance.volume = 1;
      speechSynthesis.speak(utterance);
    }
  };

  const connectionLabel = {
    disconnected: 'Disconnected',
    connecting: 'Connecting...',
    connected: 'Connected',
    streaming: 'Streaming',
    error: 'Error',
  }[connectionState];

  return (
    <div className="bg-background text-on-background overflow-hidden h-screen flex flex-col">
      {/* TopAppBar */}
      <header className="w-full top-0 bg-background border-b border-on-background flex justify-between items-center px-margin-page py-4 max-w-[1200px] mx-auto z-40">
        <div className="flex items-center gap-4">
          <span className="text-headline-md text-primary font-bold tracking-tighter">
            ELEVEN
          </span>
        </div>

        {/* Brain State Indicators */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${connectionState === 'streaming' ? 'bg-primary signal-pulse' : 'bg-secondary-container'}`} />
            <span className="text-label-sm uppercase tracking-widest">{connectionLabel}</span>
          </div>

          {connectionState === 'streaming' && (
            <div className="flex items-center gap-4 text-label-sm text-on-surface-variant">
              <span>Focus: <strong className="text-primary">{Math.round(brainState.focus * 100)}%</strong></span>
              <span>Relax: <strong className="text-primary">{Math.round(brainState.relaxation * 100)}%</strong></span>
              {brainState.jaw_clench && (
                <span className="text-primary font-bold uppercase animate-pulse">JAW</span>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-gutter">
          <SignalStrength level={signalStrength} maxLevel={4} />
          <Link
            to="/dashboard"
            className="material-symbols-outlined text-primary p-2 hover:bg-surface-container rounded transition-colors"
            title="Dashboard"
          >
            monitoring
          </Link>
          <Link
            to="/train"
            className="material-symbols-outlined text-primary p-2 hover:bg-surface-container rounded transition-colors"
            title="Training Mode"
          >
            school
          </Link>
          <button className="material-symbols-outlined text-primary p-2">
            volume_up
          </button>
        </div>
      </header>

      <main className="flex-1 flex flex-col max-w-[1200px] mx-auto w-full overflow-hidden">
        {/* Upper Pane: Chat Thread */}
        <section className="h-[55%] flex flex-col p-gutter overflow-y-auto no-scrollbar gap-stack-md border-b border-secondary-container">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex flex-col max-w-[80%] ${
                message.sender === 'user' ? 'items-end self-end' : 'items-start'
              }`}
            >
              <div
                className={`p-6 rounded-xl ${
                  message.sender === 'user'
                    ? 'bg-surface-container text-primary border-transparent'
                    : 'bg-white text-primary border border-secondary-container'
                }`}
              >
                <p className="text-body-lg">{message.text}</p>
              </div>
              <span className="text-label-sm text-on-surface-variant mt-2 mx-1">
                {message.sender === 'user' ? 'You' : 'Healthcare Partner'}
              </span>
            </div>
          ))}
        </section>

        {/* Lower Pane: Phrase Grid */}
        <section className="h-[45%] p-gutter bg-white flex flex-col">
          {/* Tabs */}
          <div className="flex gap-4 mb-4 border-b border-secondary-container pb-2">
            <button
              onClick={() => setActiveTab('common')}
              className={`text-label-lg uppercase tracking-widest transition-colors ${
                activeTab === 'common'
                  ? 'text-primary border-b-2 border-primary'
                  : 'text-on-surface-variant'
              }`}
            >
              Common
            </button>
            <button
              onClick={() => setActiveTab('needs')}
              className={`text-label-lg uppercase tracking-widest transition-colors ${
                activeTab === 'needs'
                  ? 'text-primary border-b-2 border-primary'
                  : 'text-on-surface-variant'
              }`}
            >
              Needs
            </button>
          </div>

          {/* Phrase Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 flex-1">
            {filteredPhrases.map((phrase) => (
              <button
                key={phrase.id}
                onClick={() => handlePhraseSelect(phrase)}
                onMouseEnter={() => setFocusedPhraseId(phrase.id)}
                onMouseLeave={() => setFocusedPhraseId(null)}
                className={`relative flex items-center justify-center rounded-lg min-h-target-min transition-all overflow-hidden ${
                  focusedPhraseId === phrase.id
                    ? 'bg-primary-container text-on-primary border-primary'
                    : 'border border-secondary-container bg-white hover:bg-surface-container-low'
                }`}
              >
                <span className="text-headline-md uppercase tracking-wide">
                  {phrase.text}
                </span>

                {/* Dwell progress indicator (visible when focused) */}
                {focusedPhraseId === phrase.id && (
                  <div className="dwell-progress" style={{ width: '0%' }} />
                )}
              </button>
            ))}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="w-full bottom-0 bg-background border-t border-secondary-container py-stack-sm mt-auto">
        <div className="flex flex-col md:flex-row justify-between items-center w-full px-margin-page max-w-[1200px] mx-auto gap-4">
          <span className="text-label-sm text-on-surface-variant uppercase tracking-wider">
            © 2024 ELEVEN MEDICAL SYSTEMS. ALL RIGHTS RESERVED.
          </span>
          <div className="flex gap-stack-md">
            <a
              className="text-label-sm text-on-surface-variant hover:text-primary transition-colors"
              href="#"
            >
              Privacy
            </a>
            <a
              className="text-label-sm text-on-surface-variant hover:text-primary transition-colors"
              href="#"
            >
              Terms
            </a>
            <a
              className="text-label-sm text-on-surface-variant hover:text-primary transition-colors"
              href="#"
            >
              Support
            </a>
          </div>
        </div>
      </footer>

      {/* Emergency Help Button */}
      <button
        onClick={handleEmergency}
        className="fixed bottom-24 right-8 z-50 bg-error text-on-error px-8 py-4 rounded-full text-headline-md uppercase tracking-widest shadow-lg active:scale-95 transition-transform"
      >
        Emergency Help
      </button>

      {/* Confirmation Modal */}
      <ConfirmationModal
        isOpen={showConfirmation}
        phrase={selectedPhrase?.text || ''}
        onConfirm={handleConfirm}
        onCancel={handleCancel}
      />
    </div>
  );
}
