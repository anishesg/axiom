import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '../components/layout/Layout';
import { SignalStrength } from '../components/ui/SignalStrength';
import { EEGWaveform, EEGWaveformSnapshot } from '../components/ui/EEGWaveform';
import { useSpeechRecognition } from '../hooks/useSpeechRecognition';
import { useElevenSocket } from '../hooks/useElevenSocket';
import { useSessionStore } from '../stores/sessionStore';

// ElevenLabs TTS helper
async function speakWithElevenLabs(text: string, voiceId: string = '21m00Tcm4TlvDq8ikWAM'): Promise<void> {
  const API_BASE = 'http://localhost:8000';

  try {
    const response = await fetch(`${API_BASE}/api/text-to-speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice_id: voiceId }),
    });

    if (!response.ok) {
      console.warn('ElevenLabs TTS failed, falling back to Web Speech API');
      fallbackSpeak(text);
      return;
    }

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);

    audio.onended = () => {
      URL.revokeObjectURL(audioUrl);
    };

    await audio.play();
  } catch (err) {
    console.warn('ElevenLabs TTS error, falling back to Web Speech API:', err);
    fallbackSpeak(text);
  }
}

// Fallback to Web Speech API
function fallbackSpeak(text: string): void {
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    speechSynthesis.speak(utterance);
  }
}

interface CognitiveState {
  emotion: string;
  engagement: number;
  valence: number;
  focus: number;
  relaxation: number;
}

interface ConversationMessage {
  id: string;
  sender: 'speaker' | 'eeg_user';
  text: string;
  timestamp: Date;
  cognitiveState?: CognitiveState;
  eegSnapshot?: number[][];
}

type TurnState = 'idle' | 'speaker_speaking' | 'listening_eeg' | 'generating_response';

const API_BASE = 'http://localhost:8000';

export function ConversationMode() {
  const navigate = useNavigate();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Conversation state
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [turnState, setTurnState] = useState<TurnState>('idle');
  const [conversationActive, setConversationActive] = useState(false);
  const [speakerName, setSpeakerName] = useState('Speaker');
  const [eegUserName, setEegUserName] = useState('User');
  const [showNameModal, setShowNameModal] = useState(true);
  const [eegListeningProgress, setEegListeningProgress] = useState(0);

  // EEG connection
  const { connect, startStreaming, disconnect } = useElevenSocket();
  const { connectionState, signalStrength, brainState, rawEEG } = useSessionStore();

  // Speech recognition
  const {
    transcript,
    interimTranscript,
    isListening,
    isSupported: speechSupported,
    error: speechError,
    startListening,
    stopListening,
    resetTranscript,
  } = useSpeechRecognition({
    continuous: true,
    interimResults: true,
  });

  // Auto-scroll messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initialize EEG connection
  useEffect(() => {
    if (conversationActive && connectionState === 'disconnected') {
      connect(true); // Use simulation
    }
  }, [conversationActive, connectionState, connect]);

  // Start streaming when connected
  useEffect(() => {
    if (conversationActive && connectionState === 'connected') {
      startStreaming();
    }
  }, [conversationActive, connectionState, startStreaming]);

  // EEG listening timer
  useEffect(() => {
    if (turnState !== 'listening_eeg') {
      setEegListeningProgress(0);
      return;
    }

    const duration = 5000; // 5 seconds
    const interval = 50; // Update every 50ms
    let elapsed = 0;

    const timer = setInterval(() => {
      elapsed += interval;
      const progress = Math.min(100, (elapsed / duration) * 100);
      setEegListeningProgress(progress);

      if (elapsed >= duration) {
        clearInterval(timer);
        generateEEGResponse();
      }
    }, interval);

    return () => clearInterval(timer);
  }, [turnState]);

  // Generate response from EEG state
  const generateEEGResponse = useCallback(async () => {
    setTurnState('generating_response');

    const lastSpeakerMessage = messages
      .filter((m) => m.sender === 'speaker')
      .pop();

    try {
      const response = await fetch(`${API_BASE}/api/generate-response`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          speaker_message: lastSpeakerMessage?.text || '',
          cognitive_state: {
            engagement: brainState.engagement,
            focus: brainState.focus,
            relaxation: brainState.relaxation,
            cognitive_load: brainState.cognitive_load,
            valence: brainState.valence,
          },
          conversation_history: messages.map((m) => ({
            sender: m.sender,
            text: m.text,
          })),
          eeg_user_name: eegUserName,
        }),
      });

      const data = await response.json();

      // Capture EEG snapshot
      const eegSnapshot = rawEEG.length > 0 ? rawEEG.map((ch) => [...ch]) : [];

      // Add EEG user response
      const newMessage: ConversationMessage = {
        id: Date.now().toString(),
        sender: 'eeg_user',
        text: data.response,
        timestamp: new Date(),
        cognitiveState: {
          emotion: data.emotion,
          engagement: brainState.engagement,
          valence: brainState.valence,
          focus: brainState.focus,
          relaxation: brainState.relaxation,
        },
        eegSnapshot,
      };

      setMessages((prev) => [...prev, newMessage]);

      // Speak the response using ElevenLabs (with fallback to Web Speech API)
      speakWithElevenLabs(data.response);
    } catch (err) {
      console.error('Failed to generate response:', err);
      // Fallback response
      const newMessage: ConversationMessage = {
        id: Date.now().toString(),
        sender: 'eeg_user',
        text: "I'm having trouble responding right now.",
        timestamp: new Date(),
        cognitiveState: {
          emotion: 'Neutral',
          engagement: brainState.engagement,
          valence: brainState.valence,
          focus: brainState.focus,
          relaxation: brainState.relaxation,
        },
      };
      setMessages((prev) => [...prev, newMessage]);
    }

    setTurnState('idle');
  }, [messages, brainState, rawEEG, eegUserName]);

  // Handle speaker finished speaking
  const handleSpeakerFinished = useCallback(() => {
    stopListening();

    if (transcript.trim()) {
      const newMessage: ConversationMessage = {
        id: Date.now().toString(),
        sender: 'speaker',
        text: transcript.trim(),
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, newMessage]);
      resetTranscript();

      // Switch to EEG listening
      setTurnState('listening_eeg');
    } else {
      setTurnState('idle');
    }
  }, [transcript, stopListening, resetTranscript]);

  // Start speaking
  const handleStartSpeaking = () => {
    resetTranscript();
    startListening();
    setTurnState('speaker_speaking');
  };

  // Start conversation
  const handleStartConversation = () => {
    setShowNameModal(false);
    setConversationActive(true);
    setTurnState('idle');
  };

  // End conversation
  const handleEndConversation = () => {
    stopListening();
    setConversationActive(false);
    setTurnState('idle');
    speechSynthesis.cancel();
  };

  // Get status text
  const getStatusText = () => {
    switch (turnState) {
      case 'speaker_speaking':
        return `${speakerName} is speaking...`;
      case 'listening_eeg':
        return `Reading ${eegUserName}'s brain signals...`;
      case 'generating_response':
        return 'Generating response...';
      default:
        return conversationActive
          ? `${speakerName}'s turn to speak`
          : 'Start a conversation';
    }
  };

  return (
    <Layout showBackButton backTo="/communicate" backLabel="Back" showNav={false}>
      {/* Name Setup Modal */}
      {showNameModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-8 max-w-md w-full mx-4 shadow-xl">
            <h2 className="text-headline-md text-primary mb-6">Set Up Conversation</h2>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-label-sm text-on-surface-variant mb-2">
                  Speaker Name (person who can speak)
                </label>
                <input
                  type="text"
                  value={speakerName}
                  onChange={(e) => setSpeakerName(e.target.value)}
                  className="w-full px-4 py-3 border border-outline-variant rounded-lg text-body-md focus:outline-none focus:border-primary"
                  placeholder="Adam"
                />
              </div>

              <div>
                <label className="block text-label-sm text-on-surface-variant mb-2">
                  EEG User Name (person using brain signals)
                </label>
                <input
                  type="text"
                  value={eegUserName}
                  onChange={(e) => setEegUserName(e.target.value)}
                  className="w-full px-4 py-3 border border-outline-variant rounded-lg text-body-md focus:outline-none focus:border-primary"
                  placeholder="Josh"
                />
              </div>
            </div>

            <div className="flex gap-4">
              <button
                onClick={() => navigate('/communicate')}
                className="flex-1 px-6 py-3 border border-outline text-on-surface-variant rounded-lg hover:bg-surface-container transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleStartConversation}
                disabled={!speakerName.trim() || !eegUserName.trim()}
                className="flex-1 px-6 py-3 bg-primary text-on-primary rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50"
              >
                Start Conversation
              </button>
            </div>
          </div>
        </div>
      )}

      <main className="flex-grow flex flex-col h-[calc(100vh-120px)] max-w-[900px] mx-auto w-full px-margin-page py-stack-sm">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-headline-md text-primary">Conversation Mode</h1>
            <p className="text-body-md text-on-surface-variant">{getStatusText()}</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-4 py-2 bg-surface-container-low rounded-full">
              <SignalStrength level={signalStrength} maxLevel={4} />
              <span className="text-label-sm text-on-surface-variant">
                {connectionState === 'streaming' ? 'EEG Active' : 'Connecting...'}
              </span>
            </div>
          </div>
        </div>

        {/* Live EEG Visualization (when listening) */}
        {turnState === 'listening_eeg' && (
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-label-sm text-on-surface-variant">
                Reading {eegUserName}'s brain activity
              </span>
              <span className="text-label-sm text-primary font-medium">
                {Math.round(eegListeningProgress)}%
              </span>
            </div>
            <div className="h-1 bg-surface-container-high rounded-full overflow-hidden mb-2">
              <div
                className="h-full bg-primary transition-all duration-100"
                style={{ width: `${eegListeningProgress}%` }}
              />
            </div>
            <EEGWaveform data={rawEEG} height={120} showLabels={false} animate={true} />
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
          {messages.length === 0 && conversationActive && (
            <div className="flex items-center justify-center h-full text-on-surface-variant">
              <div className="text-center">
                <span className="material-symbols-outlined text-5xl opacity-30 block mb-2">
                  chat
                </span>
                <p className="text-body-md">
                  Click the microphone to start speaking to {eegUserName}
                </p>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.sender === 'speaker' ? 'justify-start' : 'justify-end'
              }`}
            >
              <div
                className={`max-w-[80%] rounded-xl overflow-hidden ${
                  message.sender === 'speaker'
                    ? 'bg-surface-container'
                    : 'bg-primary-container'
                }`}
              >
                {/* EEG User: Show waveform snapshot */}
                {message.sender === 'eeg_user' && message.eegSnapshot && (
                  <div className="p-3 bg-surface-container-lowest">
                    <EEGWaveformSnapshot
                      data={message.eegSnapshot}
                      height={100}
                      showLabels={false}
                    />
                  </div>
                )}

                {/* Message Content */}
                <div className="p-4">
                  {/* Cognitive state for EEG user */}
                  {message.sender === 'eeg_user' && message.cognitiveState && (
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-label-sm text-on-surface-variant">
                        Cognitive State:
                      </span>
                      <span className="px-2 py-0.5 bg-primary/10 text-primary text-label-sm rounded-full font-medium">
                        {message.cognitiveState.emotion}
                      </span>
                    </div>
                  )}

                  {/* Response label for EEG user */}
                  {message.sender === 'eeg_user' && (
                    <div className="text-label-sm text-on-surface-variant mb-1">
                      Response:
                    </div>
                  )}

                  <p className="text-body-lg text-on-surface">{message.text}</p>

                  <div className="flex items-center justify-between mt-2">
                    <span className="text-label-sm text-on-surface-variant">
                      {message.sender === 'speaker' ? speakerName : eegUserName}
                    </span>
                    <span className="text-label-sm text-on-surface-variant opacity-60">
                      {message.timestamp.toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Speech Recognition Status */}
        {turnState === 'speaker_speaking' && (
          <div className="mb-4 p-4 bg-surface-container-low rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 bg-error rounded-full animate-pulse" />
              <span className="text-label-sm text-on-surface-variant">Recording...</span>
            </div>
            <p className="text-body-lg text-primary min-h-[1.5em]">
              {transcript || interimTranscript || 'Listening...'}
            </p>
          </div>
        )}

        {/* Speech Error */}
        {speechError && (
          <div className="mb-4 p-4 bg-error-container text-on-error-container rounded-lg">
            <p className="text-body-md">{speechError}</p>
          </div>
        )}

        {/* Controls */}
        <div className="flex items-center justify-center gap-4">
          {conversationActive && turnState === 'idle' && (
            <button
              onClick={handleStartSpeaking}
              disabled={!speechSupported}
              className="flex items-center gap-3 px-8 py-4 min-h-target-min bg-primary text-on-primary rounded-full hover:opacity-90 active:scale-95 transition-all disabled:opacity-50"
            >
              <span className="material-symbols-outlined text-2xl">mic</span>
              <span className="text-label-lg">Speak to {eegUserName}</span>
            </button>
          )}

          {turnState === 'speaker_speaking' && (
            <button
              onClick={handleSpeakerFinished}
              className="flex items-center gap-3 px-8 py-4 min-h-target-min bg-error text-on-error rounded-full hover:opacity-90 active:scale-95 transition-all"
            >
              <span className="material-symbols-outlined text-2xl">send</span>
              <span className="text-label-lg">Done Speaking</span>
            </button>
          )}

          {(turnState === 'listening_eeg' || turnState === 'generating_response') && (
            <div className="flex items-center gap-3 px-8 py-4 min-h-target-min bg-surface-container text-on-surface-variant rounded-full">
              <span className="material-symbols-outlined text-2xl animate-pulse">
                neurology
              </span>
              <span className="text-label-lg">
                {turnState === 'listening_eeg'
                  ? `Reading ${eegUserName}'s signals...`
                  : 'Generating response...'}
              </span>
            </div>
          )}

          {conversationActive && (
            <button
              onClick={handleEndConversation}
              className="flex items-center gap-2 px-6 py-3 border border-outline text-on-surface-variant rounded-full hover:bg-surface-container transition-colors"
            >
              <span className="material-symbols-outlined">close</span>
              <span className="text-label-lg">End</span>
            </button>
          )}
        </div>
      </main>
    </Layout>
  );
}
