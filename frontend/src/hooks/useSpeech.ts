import { useCallback } from 'react';

interface SpeakOptions {
  rate?: number;
  pitch?: number;
  volume?: number;
}

export function useSpeech() {
  const speak = useCallback((text: string, options: SpeakOptions = {}) => {
    if (!('speechSynthesis' in window)) {
      console.warn('[Speech] Web Speech API not supported');
      return;
    }

    // Cancel any ongoing speech
    speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = options.rate ?? 0.9;
    utterance.pitch = options.pitch ?? 1;
    utterance.volume = options.volume ?? 1;

    // Use a clear, natural voice if available
    const voices = speechSynthesis.getVoices();
    const preferredVoice = voices.find(
      (v) => v.lang.startsWith('en') && v.name.includes('Natural')
    ) || voices.find((v) => v.lang.startsWith('en'));

    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    speechSynthesis.speak(utterance);
  }, []);

  const stop = useCallback(() => {
    if ('speechSynthesis' in window) {
      speechSynthesis.cancel();
    }
  }, []);

  const isSpeaking = useCallback(() => {
    return 'speechSynthesis' in window && speechSynthesis.speaking;
  }, []);

  return { speak, stop, isSpeaking };
}
