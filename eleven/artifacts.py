"""
Artifact detection for deliberate control signals.

This module detects intentional "artifacts" that serve as reliable control signals:
- Eye blinks (single, double, triple patterns)
- Jaw clenches (short, long)

These are the most reliable signals from consumer EEG (95%+ accuracy)
and form the foundation of the Eleven communication system.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
from collections import deque
import time


class ControlSignal(Enum):
    """Detected control signals from EEG artifacts."""
    NONE = "none"
    SINGLE_BLINK = "single_blink"
    DOUBLE_BLINK = "double_blink"
    TRIPLE_BLINK = "triple_blink"
    JAW_CLENCH = "jaw_clench"
    LONG_JAW_CLENCH = "long_jaw_clench"


@dataclass
class DetectionConfig:
    """Configuration for artifact detection."""

    # Blink detection (on AF7/AF8 frontal channels)
    blink_threshold: float = 50.0      # μV - amplitude threshold for blink
    blink_min_duration: float = 0.05   # seconds - minimum blink duration
    blink_max_duration: float = 0.4    # seconds - maximum blink duration
    blink_refractory: float = 0.2      # seconds - minimum time between blinks

    # Multi-blink pattern detection
    double_blink_window: float = 0.8   # seconds - max time for double blink
    triple_blink_window: float = 1.2   # seconds - max time for triple blink
    pattern_timeout: float = 0.6       # seconds - wait time after last blink to confirm pattern

    # Jaw clench detection (EMG on all channels, especially TP9/TP10)
    clench_threshold: float = 30.0     # μV - high-frequency power threshold
    clench_freq_low: float = 20.0      # Hz - EMG frequency band low
    clench_freq_high: float = 40.0     # Hz - EMG frequency band high
    clench_min_duration: float = 0.15  # seconds - minimum clench duration
    long_clench_duration: float = 0.8  # seconds - threshold for "long" clench

    # General
    sample_rate: int = 256


@dataclass
class BlinkEvent:
    """Represents a detected eye blink."""
    timestamp: float
    amplitude: float
    duration: float
    channel: str  # "AF7" or "AF8" or "both"


@dataclass
class ClenchEvent:
    """Represents a detected jaw clench."""
    timestamp: float
    duration: float
    intensity: float


class BlinkDetector:
    """
    Detects eye blinks from frontal EEG channels (AF7, AF8).

    Eye blinks appear as large amplitude deflections (50-200 μV)
    lasting 100-400ms on the frontal channels.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

        # State tracking
        self._in_blink = False
        self._blink_start_time: Optional[float] = None
        self._blink_peak_amplitude = 0.0
        self._last_blink_time = 0.0

        # Recent blinks for pattern detection
        self._recent_blinks: deque[BlinkEvent] = deque(maxlen=5)

        # Pattern state
        self._waiting_for_pattern = False
        self._pattern_start_time = 0.0

    def detect(self, data: np.ndarray, timestamp: float) -> Optional[BlinkEvent]:
        """
        Detect blink in current data window.

        Args:
            data: Shape (4, n_samples) filtered EEG data [TP9, AF7, AF8, TP10]
            timestamp: Current timestamp

        Returns:
            BlinkEvent if blink detected, None otherwise.
        """
        # Focus on frontal channels (AF7=index 1, AF8=index 2)
        af7 = data[1, :]
        af8 = data[2, :]

        # Use peak-to-peak amplitude
        af7_amplitude = np.max(af7) - np.min(af7)
        af8_amplitude = np.max(af8) - np.min(af8)
        max_amplitude = max(af7_amplitude, af8_amplitude)

        # Check refractory period
        if timestamp - self._last_blink_time < self.config.blink_refractory:
            return None

        # Detect blink onset
        if not self._in_blink and max_amplitude > self.config.blink_threshold:
            self._in_blink = True
            self._blink_start_time = timestamp
            self._blink_peak_amplitude = max_amplitude
            return None

        # Track ongoing blink
        if self._in_blink:
            self._blink_peak_amplitude = max(self._blink_peak_amplitude, max_amplitude)

            # Check if blink ended (amplitude dropped)
            if max_amplitude < self.config.blink_threshold * 0.5:
                duration = timestamp - self._blink_start_time

                # Validate blink duration
                if (self.config.blink_min_duration <= duration <= self.config.blink_max_duration):
                    # Valid blink detected
                    self._in_blink = False
                    self._last_blink_time = timestamp

                    # Determine which channel
                    if af7_amplitude > af8_amplitude * 1.5:
                        channel = "AF7"
                    elif af8_amplitude > af7_amplitude * 1.5:
                        channel = "AF8"
                    else:
                        channel = "both"

                    blink = BlinkEvent(
                        timestamp=timestamp,
                        amplitude=self._blink_peak_amplitude,
                        duration=duration,
                        channel=channel,
                    )

                    self._recent_blinks.append(blink)
                    return blink

                else:
                    # Invalid duration - reset
                    self._in_blink = False

            # Check for timeout (blink too long)
            elif timestamp - self._blink_start_time > self.config.blink_max_duration:
                self._in_blink = False

        return None

    def get_pattern(self) -> Optional[ControlSignal]:
        """
        Analyze recent blinks to detect multi-blink patterns.

        Should be called periodically (e.g., every 100ms) to check
        if a pattern is complete.

        Returns:
            ControlSignal for detected pattern, or None if still waiting.
        """
        if len(self._recent_blinks) == 0:
            return None

        current_time = time.time()
        latest_blink = self._recent_blinks[-1]

        # Wait for pattern timeout after last blink
        time_since_last = current_time - latest_blink.timestamp
        if time_since_last < self.config.pattern_timeout:
            return None  # Still waiting for more blinks

        # Count blinks within pattern windows
        blinks_in_window = [
            b for b in self._recent_blinks
            if current_time - b.timestamp < self.config.triple_blink_window + self.config.pattern_timeout
        ]

        n_blinks = len(blinks_in_window)
        self._recent_blinks.clear()  # Reset for next pattern

        if n_blinks >= 3:
            # Check timing for triple blink
            if blinks_in_window[-1].timestamp - blinks_in_window[-3].timestamp < self.config.triple_blink_window:
                return ControlSignal.TRIPLE_BLINK
            elif blinks_in_window[-1].timestamp - blinks_in_window[-2].timestamp < self.config.double_blink_window:
                return ControlSignal.DOUBLE_BLINK
            else:
                return ControlSignal.SINGLE_BLINK
        elif n_blinks == 2:
            if blinks_in_window[-1].timestamp - blinks_in_window[-2].timestamp < self.config.double_blink_window:
                return ControlSignal.DOUBLE_BLINK
            else:
                return ControlSignal.SINGLE_BLINK
        elif n_blinks == 1:
            return ControlSignal.SINGLE_BLINK

        return None


class JawClenchDetector:
    """
    Detects jaw clenches from EMG activity.

    Jaw clenches produce high-frequency EMG bursts (20-40 Hz)
    visible on all channels but especially temporal (TP9, TP10).
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()

        # State
        self._in_clench = False
        self._clench_start_time: Optional[float] = None
        self._clench_intensity = 0.0

    def detect(self, data: np.ndarray, timestamp: float) -> Optional[ClenchEvent]:
        """
        Detect jaw clench in current data window.

        Args:
            data: Shape (4, n_samples) filtered EEG data
            timestamp: Current timestamp

        Returns:
            ClenchEvent if clench ended, None otherwise.
        """
        # Compute high-frequency power (EMG band)
        # Simple approach: variance of the signal (correlates with EMG activity)
        # More sophisticated: bandpass 20-40 Hz then compute power

        # Use temporal channels primarily (TP9=0, TP10=3)
        tp9_var = np.var(data[0, :])
        tp10_var = np.var(data[3, :])
        emg_power = (tp9_var + tp10_var) / 2

        # Also check frontal for confirmation
        af_var = (np.var(data[1, :]) + np.var(data[2, :])) / 2

        # EMG should be elevated on all channels
        combined_power = (emg_power + af_var) / 2

        # Detect clench onset
        if not self._in_clench and combined_power > self.config.clench_threshold:
            self._in_clench = True
            self._clench_start_time = timestamp
            self._clench_intensity = combined_power
            return None

        # Track ongoing clench
        if self._in_clench:
            self._clench_intensity = max(self._clench_intensity, combined_power)

            # Check if clench ended
            if combined_power < self.config.clench_threshold * 0.5:
                duration = timestamp - self._clench_start_time

                if duration >= self.config.clench_min_duration:
                    self._in_clench = False

                    return ClenchEvent(
                        timestamp=timestamp,
                        duration=duration,
                        intensity=self._clench_intensity,
                    )
                else:
                    # Too short - reset
                    self._in_clench = False

        return None

    def classify_clench(self, event: ClenchEvent) -> ControlSignal:
        """
        Classify clench as short or long.

        Args:
            event: Detected clench event

        Returns:
            JAW_CLENCH or LONG_JAW_CLENCH
        """
        if event.duration >= self.config.long_clench_duration:
            return ControlSignal.LONG_JAW_CLENCH
        return ControlSignal.JAW_CLENCH


class ArtifactDetector:
    """
    Combined artifact detector for all control signals.

    Integrates blink and clench detection into a unified interface.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.blink_detector = BlinkDetector(config)
        self.clench_detector = JawClenchDetector(config)

        # Callbacks
        self._on_signal: Optional[Callable[[ControlSignal], None]] = None

        # State
        self._last_check_time = 0.0
        self._pending_blink_pattern = False

    def set_callback(self, callback: Callable[[ControlSignal], None]):
        """Set callback for detected control signals."""
        self._on_signal = callback

    def process(self, data: np.ndarray, timestamp: float) -> Optional[ControlSignal]:
        """
        Process EEG window and detect control signals.

        Args:
            data: Shape (4, n_samples) filtered EEG data
            timestamp: Current timestamp

        Returns:
            Detected ControlSignal, or None.
        """
        detected_signal = None

        # Check for blinks
        blink = self.blink_detector.detect(data, timestamp)
        if blink is not None:
            self._pending_blink_pattern = True

        # Check for completed blink patterns (periodically)
        if self._pending_blink_pattern:
            pattern = self.blink_detector.get_pattern()
            if pattern is not None:
                detected_signal = pattern
                self._pending_blink_pattern = False

        # Check for jaw clench
        clench = self.clench_detector.detect(data, timestamp)
        if clench is not None:
            detected_signal = self.clench_detector.classify_clench(clench)

        # Fire callback if signal detected
        if detected_signal is not None and self._on_signal is not None:
            self._on_signal(detected_signal)

        return detected_signal


@dataclass
class CalibrationData:
    """Stores calibration data for personalized detection thresholds."""

    blink_amplitudes: list[float] = field(default_factory=list)
    clench_intensities: list[float] = field(default_factory=list)
    natural_blink_amplitudes: list[float] = field(default_factory=list)  # Involuntary blinks

    def compute_thresholds(self) -> DetectionConfig:
        """
        Compute personalized thresholds from calibration data.

        Returns:
            DetectionConfig with personalized thresholds.
        """
        config = DetectionConfig()

        if self.blink_amplitudes:
            # Set threshold to discriminate deliberate from natural blinks
            deliberate_mean = np.mean(self.blink_amplitudes)
            deliberate_std = np.std(self.blink_amplitudes)

            if self.natural_blink_amplitudes:
                natural_mean = np.mean(self.natural_blink_amplitudes)
                # Threshold halfway between natural and deliberate
                config.blink_threshold = (natural_mean + deliberate_mean) / 2
            else:
                # Default: 2 std below mean deliberate blink
                config.blink_threshold = max(30, deliberate_mean - 2 * deliberate_std)

        if self.clench_intensities:
            clench_mean = np.mean(self.clench_intensities)
            clench_std = np.std(self.clench_intensities)
            # Threshold at 2 std below mean
            config.clench_threshold = max(20, clench_mean - 2 * clench_std)

        return config


class CalibrationSession:
    """
    Guides user through calibration to learn their signal patterns.

    Calibration steps:
    1. Rest (baseline)
    2. Natural blinks (involuntary)
    3. Deliberate strong blinks
    4. Jaw clenches
    """

    def __init__(self):
        self.data = CalibrationData()
        self._current_step = 0
        self._collecting = False
        self._temp_amplitudes: list[float] = []

    def start_step(self, step_name: str):
        """Start collecting data for a calibration step."""
        self._collecting = True
        self._temp_amplitudes = []

    def add_sample(self, amplitude: float):
        """Add a detected amplitude during calibration."""
        if self._collecting:
            self._temp_amplitudes.append(amplitude)

    def end_step(self, step_name: str):
        """End current calibration step and store data."""
        self._collecting = False

        if step_name == "natural_blinks":
            self.data.natural_blink_amplitudes.extend(self._temp_amplitudes)
        elif step_name == "deliberate_blinks":
            self.data.blink_amplitudes.extend(self._temp_amplitudes)
        elif step_name == "jaw_clenches":
            self.data.clench_intensities.extend(self._temp_amplitudes)

    def get_config(self) -> DetectionConfig:
        """Get personalized detection config from calibration data."""
        return self.data.compute_thresholds()


# CLI for testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

    from eleven.acquisition import create_acquisition, SimulatedMuse
    from eleven.preprocessing import EEGPreprocessor

    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing artifact detection with simulated data...")
    print("Simulated blinks will be injected randomly.")
    print()

    muse = SimulatedMuse()
    preprocessor = EEGPreprocessor()
    detector = ArtifactDetector()

    def on_signal(signal: ControlSignal):
        print(f"  >> DETECTED: {signal.value}")

    detector.set_callback(on_signal)

    muse.connect()
    muse.start_stream()

    try:
        for _ in range(50):  # Run for ~12 seconds
            data = muse.get_data(64)
            if data is not None:
                windows = preprocessor.process(data)
                for window in windows:
                    timestamp = time.time()
                    signal = detector.process(window, timestamp)

            time.sleep(0.25)

    except KeyboardInterrupt:
        pass

    muse.disconnect()
    print("\nDone.")
