// Default phrases for the communication board
export const DEFAULT_PHRASES = {
  common: [
    { id: 'yes', text: 'Yes' },
    { id: 'no', text: 'No' },
    { id: 'help', text: 'Help' },
    { id: 'thank-you', text: 'Thank you' },
    { id: 'i-need', text: "I need..." },
    { id: 'more', text: 'More' },
  ],
  needs: [
    { id: 'water', text: 'Water' },
    { id: 'food', text: 'Food' },
    { id: 'rest', text: 'Rest' },
    { id: 'medicine', text: 'Medicine' },
    { id: 'pain', text: 'Pain' },
    { id: 'call-nurse', text: 'Call nurse' },
  ],
  emotions: [
    { id: 'happy', text: 'Happy' },
    { id: 'sad', text: 'Sad' },
    { id: 'tired', text: 'Tired' },
    { id: 'uncomfortable', text: 'Uncomfortable' },
    { id: 'anxious', text: 'Anxious' },
    { id: 'calm', text: 'Calm' },
  ],
} as const;

// Timing configurations
export const TIMING = {
  // Dwell time for selection (ms)
  DWELL_TIME: 2000,

  // Confirmation modal timeout (ms)
  CONFIRMATION_TIMEOUT: 5000,

  // WebSocket reconnection delay (ms)
  WS_RECONNECT_DELAY: 3000,

  // Signal quality update interval (ms)
  SIGNAL_UPDATE_INTERVAL: 1000,
} as const;

// Control signal mappings
export const CONTROL_SIGNALS = {
  double_blink: {
    label: 'Double Blink',
    action: 'yes',
    description: 'Confirm / Yes',
  },
  triple_blink: {
    label: 'Triple Blink',
    action: 'no',
    description: 'Cancel / No',
  },
  jaw_clench: {
    label: 'Jaw Clench',
    action: 'select',
    description: 'Select current item',
  },
  long_jaw_clench: {
    label: 'Long Jaw Clench',
    action: 'cancel',
    description: 'Cancel / Go back',
  },
} as const;

// Calibration steps
export const CALIBRATION_STEPS = [
  {
    id: 'yes',
    prompt: "Think 'yes'",
    description: 'Focus on the affirmative response.',
  },
  {
    id: 'no',
    prompt: "Think 'no'",
    description: 'Focus on the negative response.',
  },
  {
    id: 'relax',
    prompt: 'Relax',
    description: 'Clear your mind and breathe normally.',
  },
] as const;
