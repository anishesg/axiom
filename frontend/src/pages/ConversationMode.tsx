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
  console.log('Using Web Speech API fallback for TTS');
  if ('speechSynthesis' in window) {
    // Cancel any ongoing speech
    speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.volume = 1.0;

    // Try to use a natural voice if available
    const voices = speechSynthesis.getVoices();
    const naturalVoice = voices.find(v => v.name.includes('Natural') || v.name.includes('Samantha'));
    if (naturalVoice) {
      utterance.voice = naturalVoice;
    }

    speechSynthesis.speak(utterance);
  } else {
    console.error('Web Speech API not supported in this browser');
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
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
          <div className="bg-white rounded-2xl p-7 max-w-[380px] w-full mx-4 shadow-2xl animate-scale-in">
            <h2 className="text-headline-md text-primary mb-1">Set Up Conversation</h2>
            <p className="text-body-md text-on-surface-variant mb-6">Enter names for both participants</p>

            <div className="space-y-5 mb-7">
              <div>
                <label className="block text-[12px] font-medium text-on-surface-variant mb-2 uppercase tracking-wide">
                  Speaker Name
                </label>
                <input
                  type="text"
                  value={speakerName}
                  onChange={(e) => setSpeakerName(e.target.value)}
                  className="w-full px-4 py-3 border border-outline-variant rounded-xl text-body-md focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10 transition-all"
                  placeholder="e.g. Adam"
                />
              </div>

              <div>
                <label className="block text-[12px] font-medium text-on-surface-variant mb-2 uppercase tracking-wide">
                  EEG User Name
                </label>
                <input
                  type="text"
                  value={eegUserName}
                  onChange={(e) => setEegUserName(e.target.value)}
                  className="w-full px-4 py-3 border border-outline-variant rounded-xl text-body-md focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/10 transition-all"
                  placeholder="e.g. Josh"
                />
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => navigate('/communicate')}
                className="flex-1 px-5 py-3 border border-outline-variant text-on-surface-variant rounded-xl hover:bg-surface-container-low transition-all text-[14px] font-medium"
              >
                Cancel
              </button>
              <button
                onClick={handleStartConversation}
                disabled={!speakerName.trim() || !eegUserName.trim()}
                className="flex-1 px-5 py-3 bg-primary text-on-primary rounded-xl hover:shadow-md hover:-translate-y-0.5 active:translate-y-0 transition-all disabled:opacity-40 disabled:hover:shadow-none disabled:hover:translate-y-0 text-[14px] font-medium"
              >
                Start
              </button>
            </div>
          </div>
        </div>
      )}

      <main className="flex-grow flex flex-col h-[calc(100vh-120px)] max-w-[800px] mx-auto w-full px-6 py-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div>
            <h1 className="text-[22px] font-semibold text-primary tracking-tight">Conversation Mode</h1>
            <p className="text-[14px] text-on-surface-variant mt-0.5">{getStatusText()}</p>
          </div>
          <div className="flex items-center gap-2.5 px-4 py-2 bg-surface-container-low rounded-full border border-outline-variant/50">
            <SignalStrength level={signalStrength} maxLevel={4} />
            <span className="text-[12px] text-on-surface-variant font-medium">
              {connectionState === 'streaming' ? 'EEG Active' : 'Connecting...'}
            </span>
          </div>
        </div>

        {/* Live EEG Visualization (when listening) */}
        {turnState === 'listening_eeg' && (
          <div className="mb-5 p-4 bg-surface-container-low rounded-xl border border-outline-variant/30 animate-fade-in">
            <div className="flex items-center justify-between mb-3">
              <span className="text-[12px] font-medium text-on-surface-variant uppercase tracking-wide">
                Reading {eegUserName}'s brain activity
              </span>
              <span className="text-[13px] text-primary font-semibold">
                {Math.round(eegListeningProgress)}%
              </span>
            </div>
            <div className="h-1.5 bg-outline-variant/30 rounded-full overflow-hidden mb-3">
              <div
                className="h-full bg-primary rounded-full transition-all duration-150 ease-out"
                style={{ width: `${eegListeningProgress}%` }}
              />
            </div>
            <EEGWaveform data={rawEEG} height={100} showLabels={false} animate={true} />
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-1 scrollbar-thin">
          {messages.length === 0 && conversationActive && (
            <div className="flex items-center justify-center h-full text-on-surface-variant">
              <div className="text-center py-12">
                <span className="material-symbols-outlined text-[48px] text-on-surface-variant/20 block mb-3">
                  forum
                </span>
                <p className="text-[15px] text-on-surface-variant/60">
                  Click the microphone to start speaking to {eegUserName}
                </p>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex animate-fade-in ${
                message.sender === 'speaker' ? 'justify-start' : 'justify-end'
              }`}
            >
              <div
                className={`max-w-[75%] rounded-2xl overflow-hidden shadow-sm ${
                  message.sender === 'speaker'
                    ? 'bg-surface-container-low border border-outline-variant/30'
                    : 'bg-primary-container'
                }`}
              >
                {/* EEG User: Show waveform snapshot */}
                {message.sender === 'eeg_user' && message.eegSnapshot && (
                  <div className="p-3 bg-white/5 border-b border-white/10">
                    <EEGWaveformSnapshot
                      data={message.eegSnapshot}
                      height={80}
                      showLabels={false}
                    />
                  </div>
                )}

                {/* Message Content */}
                <div className="p-4">
                  {/* Cognitive state for EEG user */}
                  {message.sender === 'eeg_user' && message.cognitiveState && (
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-[11px] text-white/60 uppercase tracking-wide">
                        State
                      </span>
                      <span className="px-2.5 py-1 bg-white/15 text-white text-[12px] rounded-full font-medium">
                        {message.cognitiveState.emotion}
                      </span>
                    </div>
                  )}

                  <p className={`text-[16px] leading-relaxed ${message.sender === 'eeg_user' ? 'text-white' : 'text-on-surface'}`}>
                    {message.text}
                  </p>

                  <div className="flex items-center justify-between mt-3 pt-2 border-t border-white/10">
                    <span className={`text-[12px] font-medium ${message.sender === 'eeg_user' ? 'text-white/70' : 'text-on-surface-variant'}`}>
                      {message.sender === 'speaker' ? speakerName : eegUserName}
                    </span>
                    <span className={`text-[11px] ${message.sender === 'eeg_user' ? 'text-white/40' : 'text-on-surface-variant/50'}`}>
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
          <div className="mb-4 p-4 bg-surface-container-low rounded-xl border border-outline-variant/30 animate-fade-in">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2.5 h-2.5 bg-error rounded-full signal-pulse" />
              <span className="text-[12px] font-medium text-on-surface-variant uppercase tracking-wide">Recording</span>
            </div>
            <p className="text-[17px] text-primary min-h-[1.5em] leading-relaxed">
              {transcript || interimTranscript || 'Listening...'}
            </p>
          </div>
        )}

        {/* Speech Error */}
        {speechError && (
          <div className="mb-4 p-4 bg-error-container text-on-error-container rounded-xl">
            <p className="text-[14px]">{speechError}</p>
          </div>
        )}

        {/* Controls */}
        <div className="flex items-center justify-center gap-3 pt-2">
          {conversationActive && turnState === 'idle' && (
            <button
              onClick={handleStartSpeaking}
              disabled={!speechSupported}
              className="flex items-center gap-2.5 px-6 py-3.5 bg-primary text-on-primary rounded-xl hover:shadow-lg hover:-translate-y-0.5 active:translate-y-0 active:shadow-md transition-all disabled:opacity-40 disabled:hover:shadow-none disabled:hover:translate-y-0"
            >
              <span className="material-symbols-outlined text-[22px]">mic</span>
              <span className="text-[14px] font-medium">Speak to {eegUserName}</span>
            </button>
          )}

          {turnState === 'speaker_speaking' && (
            <button
              onClick={handleSpeakerFinished}
              className="flex items-center gap-2.5 px-6 py-3.5 bg-primary text-on-primary rounded-xl hover:shadow-lg hover:-translate-y-0.5 active:translate-y-0 transition-all"
            >
              <span className="material-symbols-outlined text-[22px]">send</span>
              <span className="text-[14px] font-medium">Done Speaking</span>
            </button>
          )}

          {(turnState === 'listening_eeg' || turnState === 'generating_response') && (
            <div className="flex items-center gap-2.5 px-6 py-3.5 bg-surface-container-low text-on-surface-variant rounded-xl border border-outline-variant/30">
              <span className="material-symbols-outlined text-[22px] animate-pulse-thin">
                neurology
              </span>
              <span className="text-[14px] font-medium">
                {turnState === 'listening_eeg'
                  ? `Reading ${eegUserName}'s signals...`
                  : 'Generating response...'}
              </span>
            </div>
          )}

          {conversationActive && (
            <button
              onClick={handleEndConversation}
              className="flex items-center gap-2 px-5 py-3 border border-outline-variant text-on-surface-variant rounded-xl hover:bg-surface-container-low transition-all text-[14px] font-medium"
            >
              <span className="material-symbols-outlined text-[18px]">close</span>
              End
            </button>
          )}
        </div>
      </main>
    </Layout>
  );
}
