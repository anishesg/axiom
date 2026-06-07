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
from pathlib import Path
import time


def _sanitize_for_json(obj):
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    return obj


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


class EnhancedCalibrationSession:
    """
    Enhanced calibration session with baseline collection and RVQ training.

    Calibration phases:
    1. Baseline: 2 min eyes-open rest → establish normalization statistics
    2. State Calibration: Focus, relax, neutral (30s each) → train attention classifier
    3. Action Calibration: Blinks, clenches → learn detection thresholds
    4. VQ Training: Train RVQ codebook on all collected data
    """

    # Calibration step definitions (from research branch)
    # Two-phase baseline calibration (~18 seconds) for alpha/beta measurement
    # Plus artifact detection steps for control signals
    STEPS = [
        # Research-style baseline calibration (Phase 1: ~18 seconds)
        {
            "id": "relax_baseline",
            "name": "Relax Baseline",
            "duration": 10,
            "instruction": "Close your eyes and relax. We're measuring your alpha baseline.",
        },
        {
            "id": "focus_baseline",
            "name": "Focus Baseline",
            "duration": 8,
            "instruction": "Focus intently on the dot. Think hard about a math problem.",
        },
        # Artifact detection calibration (Phase 2: control signals)
        {
            "id": "natural_blinks",
            "name": "Natural Blinks",
            "duration": 20,
            "instruction": "Blink naturally, don't try to blink more or less.",
        },
        {
            "id": "deliberate_blinks",
            "name": "Deliberate Blinks",
            "duration": 30,
            "instruction": "Blink firmly when you see 'NOW'.",
        },
        {
            "id": "jaw_clenches",
            "name": "Jaw Clenches",
            "duration": 30,
            "instruction": "Clench your jaw firmly when you see 'NOW'.",
        },
    ]

    def __init__(self, user_id: str = "default"):
        from eleven.features import MultiScaleFeatureExtractor
        from eleven.user_profile import UserProfile

        self.user_id = user_id
        self.feature_extractor = MultiScaleFeatureExtractor()
        self.profile = UserProfile(user_id=user_id)

        # Current step tracking
        self._current_step_index = -1
        self._step_start_time: Optional[float] = None
        self._is_collecting = False

        # Collected data per step
        self._collected_windows: dict[str, list[np.ndarray]] = {
            step["id"]: [] for step in self.STEPS
        }
        self._collected_amplitudes: dict[str, list[float]] = {
            "natural_blinks": [],
            "deliberate_blinks": [],
            "jaw_clenches": [],
        }

        # Artifact detector for amplitude collection
        self._detector = ArtifactDetector()

        # RVQ training data
        self._training_data: dict[str, list[np.ndarray]] = {}

    def _compute_band_power(self, windows: list[np.ndarray], low_freq: float, high_freq: float) -> float:
        """
        Compute average power in a frequency band across all windows.

        Args:
            windows: List of EEG windows, each shape (4, n_samples)
            low_freq: Lower frequency bound (Hz)
            high_freq: Upper frequency bound (Hz)

        Returns:
            Average power in the specified band (μV²)
        """
        from brainflow.data_filter import DataFilter

        sample_rate = 256  # Muse sample rate
        all_powers = []

        for window in windows:
            # Average across the 4 channels
            for ch in range(4):
                channel_data = window[ch, :].copy()
                if len(channel_data) < 16:  # Need enough samples for FFT
                    continue

                # Use BrainFlow's band power calculation
                try:
                    power = DataFilter.get_band_power(
                        DataFilter.perform_fft(channel_data, 0),  # 0 = no windowing
                        sample_rate,
                        low_freq,
                        high_freq
                    )
                    if not np.isnan(power) and not np.isinf(power):
                        all_powers.append(power)
                except Exception:
                    # Fall back to simple variance-based power estimate
                    all_powers.append(np.var(channel_data))

        if all_powers:
            return float(np.mean(all_powers))
        return 0.0

    @property
    def current_step(self) -> Optional[dict]:
        """Get current calibration step info."""
        if 0 <= self._current_step_index < len(self.STEPS):
            return self.STEPS[self._current_step_index]
        return None

    @property
    def progress(self) -> float:
        """Get overall calibration progress (0-1)."""
        if self._current_step_index < 0:
            return 0.0
        return (self._current_step_index + 1) / len(self.STEPS)

    @property
    def is_complete(self) -> bool:
        """Check if calibration is complete."""
        return self._current_step_index >= len(self.STEPS) - 1 and not self._is_collecting

    def start_step(self, step_id: str) -> bool:
        """
        Start a calibration step.

        Args:
            step_id: ID of the step to start

        Returns:
            True if step started successfully
        """
        step_index = next(
            (i for i, s in enumerate(self.STEPS) if s["id"] == step_id),
            None
        )

        if step_index is None:
            return False

        self._current_step_index = step_index
        self._step_start_time = time.time()
        self._is_collecting = True

        return True

    def add_window(self, window: np.ndarray, timestamp: float):
        """
        Add an EEG window during calibration.

        Args:
            window: Shape (4, n_samples) filtered EEG data
            timestamp: Current timestamp
        """
        if not self._is_collecting or self.current_step is None:
            return

        step_id = self.current_step["id"]

        # Store window
        self._collected_windows[step_id].append(window.copy())

        # For artifact steps, detect amplitudes
        if step_id in ["natural_blinks", "deliberate_blinks"]:
            blink = self._detector.blink_detector.detect(window, timestamp)
            if blink is not None:
                self._collected_amplitudes[step_id].append(blink.amplitude)

        elif step_id == "jaw_clenches":
            clench = self._detector.clench_detector.detect(window, timestamp)
            if clench is not None:
                self._collected_amplitudes["jaw_clenches"].append(clench.intensity)

    def end_step(self) -> dict:
        """
        End current calibration step and process collected data.

        Returns:
            Dict with step results including quality metrics
        """
        if not self._is_collecting or self.current_step is None:
            return {"error": "No step in progress", "code": "NO_STEP"}

        step_id = self.current_step["id"]
        duration = time.time() - self._step_start_time if self._step_start_time else 0
        self._is_collecting = False

        samples_collected = len(self._collected_windows[step_id])

        # Calculate quality score based on expected samples
        # At 256Hz with 64-sample windows, we expect ~4 windows/second
        expected_samples = int(duration * 256 / 64) if duration > 0 else 1
        quality_score = min(1.0, samples_collected / max(expected_samples, 1))
        sufficient_data = quality_score >= 0.7

        results = {
            "step_id": step_id,
            "duration": duration,
            "n_windows": samples_collected,
            "samples_collected": samples_collected,
            "quality_score": round(quality_score, 2),
            "sufficient_data": sufficient_data,
            "warning": None if sufficient_data else "Low sample count - consider repeating step",
        }

        # Process research-style baseline steps
        if step_id == "relax_baseline":
            windows = self._collected_windows["relax_baseline"]
            if windows:
                # Compute alpha power (8-13 Hz) during relaxation
                alpha_power = self._compute_band_power(windows, 8.0, 13.0)
                self.profile.baseline_alpha = alpha_power
                results["alpha_power"] = alpha_power
                results["baseline_type"] = "relaxation"

        elif step_id == "focus_baseline":
            windows = self._collected_windows["focus_baseline"]
            if windows:
                # Compute beta power (13-30 Hz) during focus
                beta_power = self._compute_band_power(windows, 13.0, 30.0)
                self.profile.baseline_beta = beta_power
                results["beta_power"] = beta_power
                results["baseline_type"] = "focus"

        # Process legacy baseline (if still used)
        elif step_id == "baseline":
            windows = self._collected_windows["baseline"]
            if windows:
                self.feature_extractor.collect_baseline(windows)
                self.profile.update_baseline(windows, duration)
                results["baseline_features"] = self.feature_extractor.feature_dim

        # Process state steps (for VQ training)
        elif step_id in ["rest", "focus", "relax"]:
            windows = self._collected_windows[step_id]
            if windows:
                # Extract features for VQ training
                self._training_data[step_id] = windows
                results["samples_collected"] = len(windows)

        # Process amplitude steps
        elif step_id in ["natural_blinks", "deliberate_blinks", "jaw_clenches"]:
            amplitudes = self._collected_amplitudes.get(step_id, [])
            results["detections"] = len(amplitudes)
            if amplitudes:
                results["mean_amplitude"] = float(np.mean(amplitudes))
                results["std_amplitude"] = float(np.std(amplitudes))

        return results

    def update_thresholds(self):
        """Update detection thresholds from collected amplitude data."""
        self.profile.update_thresholds(
            natural_blinks=self._collected_amplitudes.get("natural_blinks", []),
            deliberate_blinks=self._collected_amplitudes.get("deliberate_blinks", []),
            jaw_clenches=self._collected_amplitudes.get("jaw_clenches", []),
        )

    def train_tokenizer(self, use_rvq: bool = True) -> dict:
        """
        Train the EEG tokenizer on collected calibration data.

        Args:
            use_rvq: Whether to use Residual VQ (recommended)

        Returns:
            Dict with training results
        """
        from eleven.vq_encoder import EEGTokenizer, RVQConfig

        # Prepare training data
        training_data = {}
        for intent, windows in self._training_data.items():
            if windows:
                training_data[intent] = windows

        if not training_data:
            return {"error": "No training data collected"}

        # Create tokenizer
        if use_rvq:
            rvq_config = RVQConfig(
                n_levels=self.profile.rvq_n_levels,
                codebook_sizes=self.profile.rvq_codebook_sizes,
                feature_dim=self.feature_extractor.feature_dim,
            )
            tokenizer = EEGTokenizer(use_rvq=True, rvq_config=rvq_config)
        else:
            tokenizer = EEGTokenizer(use_rvq=False, codebook_size=64)

        # Train tokenizer
        tokenizer.calibrate(training_data)

        # Get training statistics
        stats = tokenizer.get_stats()

        # Mark profile as trained
        self.profile.intent_names = list(training_data.keys())
        self.profile.mark_calibration_complete()

        return _sanitize_for_json({
            "tokenizer_stats": stats,
            "intents_trained": list(training_data.keys()),
            "total_samples": sum(len(w) for w in training_data.values()),
        })

    def get_training_readiness(self) -> dict:
        """
        Check if enough data has been collected for training.

        Returns:
            Dict with readiness status and details
        """
        state_steps = ["rest", "focus", "relax"]
        collected_states = {k: len(v) for k, v in self._training_data.items() if v}

        total_samples = sum(collected_states.values())
        missing_states = [s for s in state_steps if s not in collected_states]

        # Ready if we have at least one state with data
        ready = total_samples > 0

        return {
            "ready": ready,
            "total_samples": total_samples,
            "collected_states": collected_states,
            "missing_states": missing_states,
            "recommendation": "Complete at least one state step (rest, focus, or relax)" if total_samples == 0 else None
        }

    def get_detection_config(self) -> DetectionConfig:
        """Get personalized detection config from collected data."""
        config = DetectionConfig()

        # Update blink threshold
        natural = self._collected_amplitudes.get("natural_blinks", [])
        deliberate = self._collected_amplitudes.get("deliberate_blinks", [])

        if natural and deliberate:
            natural_max = np.percentile(natural, 90)
            deliberate_min = np.percentile(deliberate, 10)
            config.blink_threshold = (natural_max + deliberate_min) / 2
        elif deliberate:
            config.blink_threshold = np.percentile(deliberate, 10) * 0.8

        # Update clench threshold
        clenches = self._collected_amplitudes.get("jaw_clenches", [])
        if clenches:
            config.clench_threshold = np.percentile(clenches, 10) * 0.8

        return config

    def save_results(self, directory: Path):
        """
        Save all calibration results to directory.

        Args:
            directory: Directory to save results
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save profile
        self.profile.save(directory)

        # Save feature extractor baseline
        self.feature_extractor.save_baseline(directory / "features")

        # Save collected data for debugging/retraining
        import json
        summary = {
            "user_id": self.user_id,
            "steps_completed": [
                {
                    "id": s["id"],
                    "n_windows": len(self._collected_windows.get(s["id"], [])),
                }
                for s in self.STEPS
            ],
            "amplitudes": {
                k: {"count": len(v), "mean": float(np.mean(v)) if v else 0}
                for k, v in self._collected_amplitudes.items()
            },
        }
        with open(directory / "calibration_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    @classmethod
    def load_profile(cls, directory: Path, user_id: str) -> "EnhancedCalibrationSession":
        """
        Load a calibration session from saved profile.

        Args:
            directory: Directory containing saved profile
            user_id: User ID

        Returns:
            EnhancedCalibrationSession with loaded profile
        """
        from eleven.user_profile import UserProfile

        session = cls(user_id=user_id)
        session.profile = UserProfile.load(directory)
        session.feature_extractor.load_baseline(directory / "features")

        return session


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
