"""Neural Processing Pipeline for Muse S EEG (4-channel, 256 Hz).

Replaces the simple band-power + RandomForest system with:
  1. EEGNet feature extractor (learned spatial-temporal filters)
  2. HMM brain state estimator (latent state with temporal dynamics)
  3. ErrP detector (error-related potential from frontal electrodes)
  4. Bayesian intent accumulator (evidence integration over gaze + EEG)
  5. NeuralUCB contextual bandit (online RL for action selection)

Each class is self-contained and can be tested in isolation.

Dependencies: numpy, scipy. Optional: torch (for EEGNet; falls back to numpy).
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal
from scipy.special import softmax
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import PyTorch; fall back to numpy-only EEGNet if unavailable
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch not available; EEGNet will use numpy-only implementation.")


# ============================================================================
# 1. EEGNet Feature Extractor
# ============================================================================

if HAS_TORCH:

    class _EEGNetTorch(nn.Module):
        """EEGNet (Lawhern et al., JNER 2018) — PyTorch implementation.

        ~2.6K parameters. Extracts a 32-dim feature vector from 1 s of
        4-channel EEG at 256 Hz.

        Architecture:
          Block 1: Temporal conv (1,128) x F1=8 → BatchNorm → Depthwise spatial (C,1) x D=2 →
                   BatchNorm → ELU → AvgPool (1,4) → Dropout
          Block 2: Separable conv (1,16) x F2=16 → BatchNorm → ELU → AvgPool (1,8) → Dropout
          Classifier: Flatten → Linear → n_features
        """

        def __init__(self, n_channels: int = 4, n_samples: int = 256,
                     n_features: int = 32, dropout: float = 0.25):
            super().__init__()
            F1, D, F2 = 8, 2, 16

            # Block 1 — temporal convolution
            self.conv_temporal = nn.Conv2d(1, F1, (1, 128), padding=(0, 64), bias=False)
            self.bn_temporal = nn.BatchNorm2d(F1)

            # Block 1 — depthwise spatial convolution
            self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                          groups=F1, bias=False)
            self.bn_spatial = nn.BatchNorm2d(F1 * D)
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout)

            # Block 2 — separable convolution
            self.conv_sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16),
                                            padding=(0, 8), groups=F1 * D, bias=False)
            self.conv_sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
            self.bn_sep = nn.BatchNorm2d(F2)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout)

            # Compute flattened size by doing a dummy forward pass
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_samples)
                dummy = self._forward_features(dummy)
                flat_size = dummy.shape[1]

            self.fc = nn.Linear(flat_size, n_features)

            self._n_channels = n_channels
            self._n_samples = n_samples
            self._n_features = n_features

        def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, 1, C, T)
            # Block 1
            x = self.conv_temporal(x)
            x = self.bn_temporal(x)
            x = self.conv_spatial(x)
            x = self.bn_spatial(x)
            x = F.elu(x)
            x = self.pool1(x)
            x = self.drop1(x)

            # Block 2
            x = self.conv_sep_depth(x)
            x = self.conv_sep_point(x)
            x = self.bn_sep(x)
            x = F.elu(x)
            x = self.pool2(x)
            x = self.drop2(x)

            # Flatten
            x = x.flatten(start_dim=1)
            return x

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: (batch, n_channels, n_samples) raw EEG tensor.

            Returns:
                (batch, n_features) feature vectors.
            """
            # Add channel dim for Conv2d: (B, 1, C, T)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            features = self._forward_features(x)
            features = self.fc(features)
            return features


class EEGNetExtractor:
    """High-level wrapper around EEGNet for feature extraction.

    Handles numpy ↔ torch conversion, device placement, and model loading.
    Falls back to a pure-numpy PCA-like feature extractor if PyTorch is
    unavailable.
    """

    def __init__(self, n_channels: int = 4, n_samples: int = 256,
                 n_features: int = 32):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_features = n_features
        self._use_torch = HAS_TORCH

        if self._use_torch:
            self._model = _EEGNetTorch(n_channels, n_samples, n_features)
            self._model.eval()
            self._device = torch.device("cpu")
            self._model.to(self._device)
            n_params = sum(p.numel() for p in self._model.parameters())
            logger.info("EEGNet initialized with %d parameters (torch)", n_params)
        else:
            # Numpy fallback: random (but deterministic) projection matrix
            # This gives a consistent 32-dim embedding. Weights will be loaded
            # from a pretrained file if available.
            rng = np.random.RandomState(42)
            self._proj_temporal = rng.randn(n_samples, 64).astype(np.float32) * 0.01
            self._proj_spatial = rng.randn(n_channels * 64, 128).astype(np.float32) * 0.01
            self._proj_final = rng.randn(128, n_features).astype(np.float32) * 0.01
            self._bias_final = np.zeros(n_features, dtype=np.float32)
            logger.info("EEGNet initialized with numpy fallback")

    @classmethod
    def from_pretrained(cls, path: str, n_channels: int = 4,
                        n_samples: int = 256, n_features: int = 32) -> "EEGNetExtractor":
        """Load pretrained weights from disk.

        Supports both .pt (torch) and .npz (numpy) formats.
        """
        extractor = cls(n_channels, n_samples, n_features)
        p = Path(path)

        if not p.exists():
            logger.warning("Pretrained weights not found at %s; using random init", path)
            return extractor

        if HAS_TORCH and p.suffix == ".pt":
            state = torch.load(path, map_location="cpu", weights_only=True)
            extractor._model.load_state_dict(state)
            extractor._model.eval()
            logger.info("Loaded pretrained EEGNet (torch) from %s", path)
        elif p.suffix == ".npz":
            data = np.load(path)
            if HAS_TORCH:
                # Load numpy weights into torch model
                state_dict = {k: torch.from_numpy(v) for k, v in data.items()}
                extractor._model.load_state_dict(state_dict)
                extractor._model.eval()
            else:
                extractor._proj_temporal = data.get("proj_temporal", extractor._proj_temporal)
                extractor._proj_spatial = data.get("proj_spatial", extractor._proj_spatial)
                extractor._proj_final = data.get("proj_final", extractor._proj_final)
                extractor._bias_final = data.get("bias_final", extractor._bias_final)
            logger.info("Loaded pretrained EEGNet (numpy) from %s", path)
        else:
            logger.warning("Unrecognized weight format: %s", p.suffix)

        return extractor

    def extract_features(self, eeg_window: np.ndarray) -> np.ndarray:
        """Extract a 32-dim feature vector from a (4, 256) EEG window.

        Args:
            eeg_window: (n_channels, n_samples) numpy array. Will be
                zero-padded or truncated to (n_channels, n_samples) if needed.

        Returns:
            (n_features,) numpy array of features.
        """
        # Validate and reshape
        x = np.asarray(eeg_window, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(self.n_channels, -1)
        if x.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {x.shape[0]}")

        # Pad or truncate temporal dimension
        T = x.shape[1]
        if T < self.n_samples:
            pad = np.zeros((self.n_channels, self.n_samples - T), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
        elif T > self.n_samples:
            x = x[:, -self.n_samples:]

        # Standardize per channel (zero-mean, unit-var) for stability
        means = x.mean(axis=1, keepdims=True)
        stds = x.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-6, 1.0, stds)
        x = (x - means) / stds

        if self._use_torch:
            return self._extract_torch(x)
        else:
            return self._extract_numpy(x)

    def _extract_torch(self, x: np.ndarray) -> np.ndarray:
        """Extract features using the PyTorch model."""
        with torch.no_grad():
            tensor = torch.from_numpy(x).unsqueeze(0).to(self._device)  # (1, C, T)
            features = self._model(tensor)  # (1, n_features)
            return features.squeeze(0).cpu().numpy()

    def _extract_numpy(self, x: np.ndarray) -> np.ndarray:
        """Extract features using numpy projection (fallback).

        Applies a simple two-stage linear projection with ReLU,
        mimicking the temporal → spatial → pointwise structure of EEGNet.
        """
        # Temporal projection: (C, T) @ (T, 64) → (C, 64)
        h_temporal = x @ self._proj_temporal  # (C, 64)
        h_temporal = np.maximum(h_temporal, 0)  # ReLU

        # Flatten spatial + temporal: (C*64,)
        h_flat = h_temporal.flatten()

        # Spatial mixing: (C*64,) @ (C*64, 128) → (128,)
        h_spatial = h_flat @ self._proj_spatial
        h_spatial = np.maximum(h_spatial, 0)  # ReLU

        # Final projection: (128,) @ (128, n_features) → (n_features,)
        features = h_spatial @ self._proj_final + self._bias_final
        return features.astype(np.float32)

    def save_weights(self, path: str) -> None:
        """Save current weights to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if self._use_torch:
            torch.save(self._model.state_dict(), path)
        else:
            np.savez(path,
                     proj_temporal=self._proj_temporal,
                     proj_spatial=self._proj_spatial,
                     proj_final=self._proj_final,
                     bias_final=self._bias_final)
        logger.info("Saved EEGNet weights to %s", path)

    @property
    def param_count(self) -> int:
        if self._use_torch:
            return sum(p.numel() for p in self._model.parameters())
        else:
            return (self._proj_temporal.size + self._proj_spatial.size +
                    self._proj_final.size + self._bias_final.size)


# ============================================================================
# 2. HMM Brain State Estimator
# ============================================================================

class BrainStateHMM:
    """Time-Delay Embedded Hidden Markov Model for latent brain-state estimation.

    Uses online forward algorithm (alpha-filtering) to estimate P(state|observations)
    in real time. Each state emits observations via a multivariate Gaussian.

    States:
        browsing   - casually looking around, moderate engagement
        searching  - actively looking for something specific
        intending  - about to take a deliberate action
        resting    - disengaged, idle
        error      - detected something wrong, surprise/frustration
    """

    STATES = ["browsing", "searching", "intending", "resting", "error"]

    def __init__(self, n_features: int = 38, sticky_factor: float = 10.0):
        """
        Args:
            n_features: observation dimensionality.
                Default 38 = 32 (EEGNet) + 6 (band power ratios).
            sticky_factor: Dirichlet concentration on the diagonal of the
                transition matrix. Higher = states persist longer.
        """
        self.n_states = len(self.STATES)
        self.n_features = n_features
        self._state_idx = {s: i for i, s in enumerate(self.STATES)}

        # --- Transition matrix A[i,j] = P(state_j | state_i) ---
        # Initialize with sticky prior: strong diagonal + plausible transitions
        self._A = self._init_transition_matrix(sticky_factor)

        # --- Initial state distribution ---
        self._pi = np.array([0.3, 0.1, 0.05, 0.5, 0.05], dtype=np.float64)

        # --- Emission parameters: per-state mean and covariance ---
        # Initialize with mild separation so states are distinguishable
        self._means = np.zeros((self.n_states, n_features), dtype=np.float64)
        self._covars = np.zeros((self.n_states, n_features, n_features), dtype=np.float64)
        self._init_emission_params()

        # --- Online state: forward variable alpha ---
        self._alpha = self._pi.copy()
        self._current_state = "resting"
        self._n_updates = 0

        # Observation history for parameter updates (bounded)
        self._obs_history: list[np.ndarray] = []
        self._state_history: list[int] = []
        self._max_history = 500

        # Regularization for covariance updates
        self._cov_reg = 1e-3

    def _init_transition_matrix(self, sticky: float) -> np.ndarray:
        """Build initial transition matrix with sticky diagonal and
        structured off-diagonal priors encoding plausible state flows.

        Flow priors:
            resting → browsing → searching → intending → resting
            any → error (low probability)
            error → resting (recovery)
        """
        # Start with uniform off-diagonal
        A = np.ones((self.n_states, self.n_states), dtype=np.float64)

        # State indices
        BROWSING, SEARCHING, INTENDING, RESTING, ERROR = range(5)

        # Encode plausible transition structure (higher = more likely)
        transitions = {
            (RESTING, BROWSING): 3.0,
            (BROWSING, SEARCHING): 2.5,
            (BROWSING, RESTING): 2.0,
            (SEARCHING, INTENDING): 3.0,
            (SEARCHING, BROWSING): 2.0,
            (INTENDING, RESTING): 2.5,    # after executing intent
            (INTENDING, BROWSING): 2.0,   # changed mind
            (ERROR, RESTING): 3.0,        # recovery
            (ERROR, BROWSING): 2.0,       # recovery
        }
        for (i, j), weight in transitions.items():
            A[i, j] = weight

        # Make error state reachable from anywhere (but unlikely)
        for i in range(self.n_states):
            if i != ERROR:
                A[i, ERROR] = 0.3

        # Apply sticky prior (boost diagonal)
        for i in range(self.n_states):
            A[i, i] = sticky

        # Row-normalize
        A /= A.sum(axis=1, keepdims=True)
        return A

    def _init_emission_params(self) -> None:
        """Initialize emission Gaussians with mild separation.

        Each state gets a slightly different mean in a few feature dimensions
        so the model can start discriminating before adaptation.
        """
        rng = np.random.RandomState(123)

        for i in range(self.n_states):
            # Small random offsets for state differentiation
            self._means[i] = rng.randn(self.n_features) * 0.1

            # Identity covariance with some noise
            self._covars[i] = np.eye(self.n_features, dtype=np.float64) * 1.0

        # Give each state a distinguishing signature in the first few dims
        # These correspond to high-level EEGNet features
        # browsing: moderate activation
        self._means[0, :4] = [0.3, 0.2, -0.1, 0.0]
        # searching: high beta-like features
        self._means[1, :4] = [0.5, 0.6, 0.3, 0.2]
        # intending: high engagement features
        self._means[2, :4] = [0.7, 0.4, 0.5, 0.3]
        # resting: low activation, high alpha-like
        self._means[3, :4] = [-0.3, -0.2, -0.4, 0.5]
        # error: sharp deviation
        self._means[4, :4] = [-0.5, 0.3, -0.2, -0.6]

    def update(self, features: np.ndarray) -> dict:
        """Perform one step of forward filtering given a new observation.

        Args:
            features: (n_features,) observation vector (EEGNet features +
                band power ratios).

        Returns:
            dict with:
                state: str — most likely current state
                probabilities: dict[str, float] — P(state) for each state
                confidence: float — max probability
        """
        obs = np.asarray(features, dtype=np.float64).ravel()

        # Truncate or pad to expected dimensionality
        if len(obs) < self.n_features:
            obs = np.concatenate([obs, np.zeros(self.n_features - len(obs))])
        elif len(obs) > self.n_features:
            obs = obs[:self.n_features]

        # Compute emission likelihoods: P(obs | state_i) for each state
        emission_probs = np.zeros(self.n_states, dtype=np.float64)
        for i in range(self.n_states):
            emission_probs[i] = self._emission_likelihood(obs, i)

        # Forward step: alpha_t = normalize( emission * (A^T @ alpha_{t-1}) )
        predicted = self._A.T @ self._alpha  # (n_states,)
        self._alpha = emission_probs * predicted

        # Normalize (avoid underflow)
        alpha_sum = self._alpha.sum()
        if alpha_sum < 1e-300:
            # All likelihoods collapsed — reset to uniform
            self._alpha = np.ones(self.n_states) / self.n_states
        else:
            self._alpha /= alpha_sum

        # Determine current state
        best_idx = int(np.argmax(self._alpha))
        self._current_state = self.STATES[best_idx]

        # Track history for online parameter updates
        self._n_updates += 1
        if len(self._obs_history) < self._max_history:
            self._obs_history.append(obs)
            self._state_history.append(best_idx)
        else:
            idx = self._n_updates % self._max_history
            self._obs_history[idx] = obs
            self._state_history[idx] = best_idx

        # Periodically update emission parameters from recent history
        if self._n_updates > 50 and self._n_updates % 25 == 0:
            self._update_emission_params()

        return {
            "state": self._current_state,
            "probabilities": self.get_probabilities(),
            "confidence": float(self._alpha[best_idx]),
        }

    def _emission_likelihood(self, obs: np.ndarray, state_idx: int) -> float:
        """Compute P(obs | state) using multivariate Gaussian.

        Uses a diagonal approximation for efficiency + numerical stability.
        """
        diff = obs - self._means[state_idx]
        # Use diagonal of covariance for fast computation
        var_diag = np.diag(self._covars[state_idx])
        var_diag = np.maximum(var_diag, 1e-6)  # floor

        # Log-likelihood of diagonal Gaussian
        log_prob = -0.5 * np.sum(diff ** 2 / var_diag) - 0.5 * np.sum(np.log(var_diag))
        # Convert to probability (unnormalized is fine, we normalize alpha)
        # Clamp to avoid overflow
        log_prob = np.clip(log_prob, -500, 0)
        return np.exp(log_prob)

    def _update_emission_params(self) -> None:
        """Update emission means and covariances from recent observations.

        Uses soft assignments (alpha-filtered state probabilities) as weights.
        """
        obs_arr = np.array(self._obs_history, dtype=np.float64)
        state_arr = np.array(self._state_history, dtype=np.int32)
        n = len(obs_arr)
        if n < 20:
            return

        for s in range(self.n_states):
            mask = state_arr == s
            count = mask.sum()
            if count < 5:
                continue

            state_obs = obs_arr[mask]

            # Exponential recency weighting
            weights = np.exp(np.linspace(-2, 0, count))
            weights /= weights.sum()

            # Weighted mean
            new_mean = np.average(state_obs, axis=0, weights=weights)

            # Weighted covariance (diagonal only for stability)
            diffs = state_obs - new_mean
            new_var = np.average(diffs ** 2, axis=0, weights=weights)
            new_var = np.maximum(new_var, self._cov_reg)

            # Blend with existing params (slow adaptation)
            blend = 0.1
            self._means[s] = (1 - blend) * self._means[s] + blend * new_mean
            np.fill_diagonal(self._covars[s],
                             (1 - blend) * np.diag(self._covars[s]) + blend * new_var)

    def get_state(self) -> str:
        """Return the current most-likely state name."""
        return self._current_state

    def get_probabilities(self) -> dict[str, float]:
        """Return P(state) for all states."""
        return {s: float(self._alpha[i]) for i, s in enumerate(self.STATES)}

    def reset(self) -> None:
        """Reset the forward variable to the initial distribution."""
        self._alpha = self._pi.copy()
        self._current_state = "resting"
        self._n_updates = 0
        self._obs_history.clear()
        self._state_history.clear()

    def set_transition_matrix(self, A: np.ndarray) -> None:
        """Override the transition matrix (e.g., after learning from data)."""
        assert A.shape == (self.n_states, self.n_states)
        row_sums = A.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Rows must sum to 1"
        self._A = A.copy()

    def save(self, path: str) -> None:
        """Save HMM parameters to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
                 A=self._A, pi=self._pi,
                 means=self._means, covars=self._covars)
        logger.info("Saved HMM parameters to %s", path)

    def load(self, path: str) -> bool:
        """Load HMM parameters from disk. Returns True on success."""
        p = Path(path)
        if not p.exists():
            logger.info("No saved HMM parameters at %s", path)
            return False
        try:
            data = np.load(path)
            self._A = data["A"]
            self._pi = data["pi"]
            self._means = data["means"]
            self._covars = data["covars"]
            logger.info("Loaded HMM parameters from %s", path)
            return True
        except Exception as e:
            logger.error("Failed to load HMM parameters: %s", e)
            return False


# ============================================================================
# 3. ErrP Detector (Error-Related Potential)
# ============================================================================

class ErrPDetector:
    """Detects Error-Related Potentials (ErrP) from frontal EEG electrodes.

    ErrP components:
        ERN (Error-Related Negativity): negative peak at ~250-350 ms post-action
        Pe (Error Positivity): positive peak at ~350-500 ms post-action

    Uses AF7 (channel 1) and AF8 (channel 2) on the Muse S headband.
    Applies bandpass filtering (1-10 Hz) and template matching against a
    canonical ErrP waveform.
    """

    # Channel indices for frontal electrodes on Muse S
    FRONTAL_CHANNELS = [1, 2]  # AF7, AF8

    def __init__(self, sample_rate: int = 256, threshold: float = 0.65):
        self._sr = sample_rate
        self._threshold = threshold

        # Time window of interest after action (seconds)
        self._window_start = 0.200  # 200 ms
        self._window_end = 0.500    # 500 ms

        # Build canonical ErrP template
        self._template = self._build_template()

        # Bandpass filter coefficients (1-10 Hz, 4th order Butterworth)
        nyq = sample_rate / 2.0
        low, high = 1.0 / nyq, min(10.0 / nyq, 0.99)
        self._b, self._a = scipy_signal.butter(4, [low, high], btype="band")

        # Running baseline for normalization
        self._baseline_power = 1.0
        self._baseline_ema = 0.02

    def _build_template(self) -> np.ndarray:
        """Build canonical ErrP waveform template.

        The template covers the 200-500 ms post-stimulus window:
            - Negative peak (ERN) centered at ~300 ms
            - Positive peak (Pe) centered at ~400 ms

        Parameterized as a sum of two Gaussians.
        """
        n_samples = int((self._window_end - self._window_start) * self._sr)
        t = np.linspace(self._window_start, self._window_end, n_samples)

        # ERN: negative Gaussian centered at 300 ms, sigma ~30 ms
        ern = -1.0 * np.exp(-0.5 * ((t - 0.300) / 0.030) ** 2)

        # Pe: positive Gaussian centered at 400 ms, sigma ~40 ms
        pe = 0.7 * np.exp(-0.5 * ((t - 0.400) / 0.040) ** 2)

        template = ern + pe
        # Normalize to unit energy
        template /= np.sqrt(np.sum(template ** 2) + 1e-10)
        return template

    def detect(self, eeg_window: np.ndarray, action_time: float,
               current_time: Optional[float] = None) -> float:
        """Detect error-related potential after an action.

        Args:
            eeg_window: (4, N) array of recent EEG data. The last sample
                corresponds to current_time.
            action_time: timestamp (seconds) when the action occurred.
            current_time: current timestamp. If None, uses time.time().

        Returns:
            P(error) as a float in [0, 1]. Values above self._threshold
            indicate a likely error.
        """
        if current_time is None:
            current_time = time.time()

        eeg = np.asarray(eeg_window, dtype=np.float64)
        if eeg.ndim != 2 or eeg.shape[0] < max(self.FRONTAL_CHANNELS) + 1:
            return 0.0

        n_total = eeg.shape[1]
        elapsed = current_time - action_time

        # We need at least up to 500 ms after action to detect ErrP
        if elapsed < self._window_end:
            return 0.0

        # Determine the sample range corresponding to the 200-500 ms window
        # The last sample in eeg_window is at current_time
        samples_ago_start = int((elapsed - self._window_start) * self._sr)
        samples_ago_end = int((elapsed - self._window_end) * self._sr)

        # Convert to array indices (end of array = most recent)
        idx_start = n_total - samples_ago_start
        idx_end = n_total - samples_ago_end

        # Bounds check
        if idx_start < 0:
            idx_start = 0
        if idx_end > n_total:
            idx_end = n_total
        if idx_end <= idx_start:
            return 0.0

        # Extract frontal channels in the window
        frontal_segment = eeg[self.FRONTAL_CHANNELS, idx_start:idx_end]

        # Apply bandpass filter
        filtered = np.zeros_like(frontal_segment)
        for i in range(len(self.FRONTAL_CHANNELS)):
            ch_data = frontal_segment[i]
            if len(ch_data) < 15:  # filter needs minimum samples
                return 0.0
            # Pad to avoid edge effects
            padlen = min(3 * max(len(self._b), len(self._a)), len(ch_data) - 1)
            if padlen < 1:
                return 0.0
            filtered[i] = scipy_signal.filtfilt(self._b, self._a, ch_data,
                                                 padlen=padlen)

        # Average the two frontal channels
        avg_signal = filtered.mean(axis=0)

        # Update baseline power (for normalization)
        sig_power = np.mean(avg_signal ** 2) + 1e-10
        self._baseline_power = ((1 - self._baseline_ema) * self._baseline_power +
                                self._baseline_ema * sig_power)

        # Normalize signal
        avg_signal /= np.sqrt(self._baseline_power + 1e-10)

        # Template matching via normalized cross-correlation
        p_error = self._template_match(avg_signal)

        return float(np.clip(p_error, 0.0, 1.0))

    def _template_match(self, signal: np.ndarray) -> float:
        """Match signal against canonical ErrP template.

        Uses normalized cross-correlation, allowing for slight temporal
        jitter (up to +/- 30 ms).
        """
        template = self._template
        sig_len = len(signal)
        tmpl_len = len(template)

        if sig_len < tmpl_len:
            # Resample signal to match template length
            signal = np.interp(
                np.linspace(0, 1, tmpl_len),
                np.linspace(0, 1, sig_len),
                signal,
            )
            sig_len = tmpl_len

        if sig_len == tmpl_len:
            # Direct correlation
            sig_norm = signal / (np.sqrt(np.sum(signal ** 2)) + 1e-10)
            correlation = float(np.dot(sig_norm, template))
            # Map from [-1, 1] to [0, 1], with bias toward positive match
            p_error = (correlation + 1.0) / 2.0
            # Apply sigmoid-like shaping for better discrimination
            p_error = 1.0 / (1.0 + np.exp(-8.0 * (p_error - 0.5)))
            return p_error

        # Sliding cross-correlation to handle temporal jitter
        jitter_samples = int(0.030 * self._sr)  # +/- 30 ms
        best_corr = -1.0

        start = max(0, (sig_len - tmpl_len) // 2 - jitter_samples)
        end = min(sig_len - tmpl_len, (sig_len - tmpl_len) // 2 + jitter_samples) + 1

        for offset in range(start, end):
            segment = signal[offset:offset + tmpl_len]
            seg_norm = segment / (np.sqrt(np.sum(segment ** 2)) + 1e-10)
            corr = float(np.dot(seg_norm, template))
            if corr > best_corr:
                best_corr = corr

        p_error = (best_corr + 1.0) / 2.0
        p_error = 1.0 / (1.0 + np.exp(-8.0 * (p_error - 0.5)))
        return p_error

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = float(np.clip(value, 0.0, 1.0))


# ============================================================================
# 4. Bayesian Intent Accumulator
# ============================================================================

class BayesianIntentAccumulator:
    """Continuously updates P(intent to act on element) by integrating
    evidence from gaze, EEG engagement, frontal alpha asymmetry, and
    HMM brain-state estimates.

    Each UI element gets an independent belief P(intent | evidence_1:t)
    that accumulates over time while fixated and decays when not.

    Evidence sources and their effects:
        - Fixation on element: primary evidence accumulator
        - High engagement (from EEGNet/BrainState): faster accumulation
        - Positive FAA (approach motivation): faster accumulation
        - HMM in 'intending' state: multiplicative boost
        - Not fixating: exponential decay toward 0
    """

    # Commitment thresholds
    THRESHOLDS = {
        "none": 0.0,
        "subtle": 0.3,
        "medium": 0.5,
        "strong": 0.7,
        "execute": 0.9,
    }

    def __init__(self, decay_rate: float = 0.95,
                 base_accumulation_rate: float = 0.02,
                 max_elements: int = 50):
        """
        Args:
            decay_rate: per-frame multiplicative decay for unfixated elements.
            base_accumulation_rate: base rate of evidence accumulation per update.
            max_elements: maximum tracked elements (LRU eviction).
        """
        self._element_beliefs: dict[str, float] = {}
        self._element_last_seen: dict[str, float] = {}
        self._decay_rate = decay_rate
        self._base_rate = base_accumulation_rate
        self._max_elements = max_elements
        self._update_count = 0

    def update(self,
               fixated_element_id: Optional[str],
               fixation_duration: float,
               engagement: float,
               faa: float,
               hmm_state: str,
               hmm_confidence: float) -> dict[str, float]:
        """Perform one Bayesian update step.

        Args:
            fixated_element_id: ID of the currently fixated element, or None.
            fixation_duration: how long the current fixation has lasted (seconds).
            engagement: engagement score from EEG [0, 1].
            faa: frontal alpha asymmetry (positive = approach motivation).
            hmm_state: current HMM state name.
            hmm_confidence: HMM confidence in current state [0, 1].

        Returns:
            dict mapping element_id to P(intent) for all tracked elements.
        """
        self._update_count += 1
        now = time.time()

        # --- Decay all non-fixated elements ---
        for eid in list(self._element_beliefs.keys()):
            if eid != fixated_element_id:
                self._element_beliefs[eid] *= self._decay_rate
                # Prune negligible beliefs
                if self._element_beliefs[eid] < 1e-4:
                    del self._element_beliefs[eid]
                    self._element_last_seen.pop(eid, None)

        # --- Accumulate evidence for fixated element ---
        if fixated_element_id is not None:
            self._element_last_seen[fixated_element_id] = now

            # Initialize if new
            if fixated_element_id not in self._element_beliefs:
                self._element_beliefs[fixated_element_id] = 0.0
                self._maybe_evict()

            current_p = self._element_beliefs[fixated_element_id]

            # Compute evidence rate from multiple signals
            rate = self._compute_evidence_rate(
                fixation_duration=fixation_duration,
                engagement=engagement,
                faa=faa,
                hmm_state=hmm_state,
                hmm_confidence=hmm_confidence,
            )

            # Bayesian update: P(intent) = P(intent) + rate * (1 - P(intent))
            # This naturally saturates toward 1.0
            new_p = current_p + rate * (1.0 - current_p)
            self._element_beliefs[fixated_element_id] = float(
                np.clip(new_p, 0.0, 1.0)
            )

        return self.get_all_beliefs()

    def _compute_evidence_rate(self, fixation_duration: float,
                                engagement: float, faa: float,
                                hmm_state: str,
                                hmm_confidence: float) -> float:
        """Compute the rate of evidence accumulation from all signals.

        Each signal contributes a multiplicative factor to the base rate.
        """
        rate = self._base_rate

        # Fixation duration: longer fixation = stronger evidence
        # Logarithmic scaling so first 500ms matter most
        fixation_factor = 1.0 + 0.5 * np.log1p(fixation_duration * 2.0)
        rate *= fixation_factor

        # Engagement: higher engagement = faster accumulation
        # Scale: 0.5 at engagement=0, 2.0 at engagement=1
        engagement_factor = 0.5 + 1.5 * float(np.clip(engagement, 0, 1))
        rate *= engagement_factor

        # Frontal alpha asymmetry: positive = approach motivation
        # Maps FAA to a factor in [0.5, 2.0]
        faa_clamped = float(np.clip(faa, -1.0, 1.0))
        faa_factor = 1.0 + 0.5 * faa_clamped
        rate *= faa_factor

        # HMM state modulation
        state_factors = {
            "intending": 2.5,
            "searching": 1.5,
            "browsing": 1.0,
            "resting": 0.3,
            "error": 0.1,
        }
        hmm_factor = state_factors.get(hmm_state, 1.0)
        # Scale by HMM confidence
        hmm_factor = 1.0 + (hmm_factor - 1.0) * float(np.clip(hmm_confidence, 0, 1))
        rate *= hmm_factor

        return float(np.clip(rate, 0.0, 0.5))  # cap to prevent instant jumps

    def get_all_beliefs(self) -> dict[str, float]:
        """Return P(intent) for all tracked elements."""
        return dict(self._element_beliefs)

    def get_top_intent(self) -> tuple[Optional[str], float]:
        """Return (element_id, probability) of the highest-confidence intent.

        Returns (None, 0.0) if no elements are tracked.
        """
        if not self._element_beliefs:
            return None, 0.0
        best_id = max(self._element_beliefs, key=self._element_beliefs.get)
        return best_id, self._element_beliefs[best_id]

    def reset_element(self, element_id: str) -> None:
        """Reset belief for an element (after action execution or error)."""
        self._element_beliefs.pop(element_id, None)
        self._element_last_seen.pop(element_id, None)

    def reset_all(self) -> None:
        """Clear all tracked elements."""
        self._element_beliefs.clear()
        self._element_last_seen.clear()

    def get_commitment_level(self, element_id: str) -> str:
        """Return the commitment level for an element based on P(intent).

        Returns one of: 'none', 'subtle', 'medium', 'strong', 'execute'
        """
        p = self._element_beliefs.get(element_id, 0.0)

        if p >= self.THRESHOLDS["execute"]:
            return "execute"
        elif p >= self.THRESHOLDS["strong"]:
            return "strong"
        elif p >= self.THRESHOLDS["medium"]:
            return "medium"
        elif p >= self.THRESHOLDS["subtle"]:
            return "subtle"
        else:
            return "none"

    def _maybe_evict(self) -> None:
        """Evict least-recently-seen elements if we exceed max_elements."""
        if len(self._element_beliefs) <= self._max_elements:
            return

        # Sort by last-seen time, evict oldest
        sorted_elems = sorted(self._element_last_seen.items(), key=lambda x: x[1])
        n_evict = len(self._element_beliefs) - self._max_elements
        for eid, _ in sorted_elems[:n_evict]:
            self._element_beliefs.pop(eid, None)
            self._element_last_seen.pop(eid, None)


# ============================================================================
# 5. NeuralUCB Contextual Bandit
# ============================================================================

class NeuralUCBBandit:
    """Online contextual bandit using NeuralUCB for action selection.

    Each action arm has its own small neural network that predicts expected
    reward given context. Action selection uses Upper Confidence Bound (UCB)
    to balance exploitation and exploration.

    Context vector (48 dims):
        [0:32]   EEGNet features
        [32:37]  HMM state one-hot (5 states)
        [37:45]  element role one-hot (8 roles)
        [45]     engagement score
        [46]     frontal alpha asymmetry
        [47]     fixation duration
    """

    ACTIONS = [
        "click", "scroll_down", "scroll_up", "navigate_back",
        "open_link", "type_text", "noop",
    ]

    # Element roles for one-hot encoding
    ELEMENT_ROLES = [
        "link", "button", "input", "text", "image",
        "list_item", "heading", "other",
    ]

    def __init__(self, context_dim: int = 48, hidden_dim: int = 64,
                 alpha: float = 1.0, learning_rate: float = 0.01):
        """
        Args:
            context_dim: dimension of the context vector.
            hidden_dim: hidden layer size for each arm's network.
            alpha: UCB exploration parameter. Higher = more exploration.
            learning_rate: SGD learning rate for online updates.
        """
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.lr = learning_rate
        self.n_actions = len(self.ACTIONS)
        self._action_idx = {a: i for i, a in enumerate(self.ACTIONS)}

        # Initialize per-arm networks (2-layer MLPs)
        # Each arm: context_dim → hidden_dim → 1
        rng = np.random.RandomState(42)
        self._W1 = []  # List of (hidden_dim, context_dim) weight matrices
        self._b1 = []  # List of (hidden_dim,) bias vectors
        self._W2 = []  # List of (1, hidden_dim) weight matrices
        self._b2 = []  # List of (1,) bias vectors

        for _ in range(self.n_actions):
            # Xavier initialization
            scale1 = np.sqrt(2.0 / (context_dim + hidden_dim))
            scale2 = np.sqrt(2.0 / (hidden_dim + 1))
            self._W1.append(rng.randn(hidden_dim, context_dim).astype(np.float64) * scale1)
            self._b1.append(np.zeros(hidden_dim, dtype=np.float64))
            self._W2.append(rng.randn(1, hidden_dim).astype(np.float64) * scale2)
            self._b2.append(np.zeros(1, dtype=np.float64))

        # Per-arm uncertainty estimation using a running covariance approximation
        # We track the diagonal of the gradient outer product (cheap approximation)
        self._grad_sq_sum = [np.ones(context_dim, dtype=np.float64) for _ in range(self.n_actions)]

        # Action counts for UCB exploration bonus
        self._action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self._total_updates = 0

        # History for batch diagnostics
        self._reward_history: list[float] = []
        self._max_reward_history = 1000

    def _forward(self, context: np.ndarray, arm: int) -> tuple[float, np.ndarray]:
        """Forward pass through arm's network.

        Returns (predicted_reward, hidden_activations).
        """
        ctx = np.nan_to_num(context.astype(np.float64), nan=0.0, posinf=1e6, neginf=-1e6)

        # Suppress spurious BLAS matmul warnings (numpy 2.x on Python 3.14
        # can emit these even when all inputs/outputs are finite)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # Layer 1: linear + ReLU
            h = self._W1[arm] @ ctx + self._b1[arm]
            h = np.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
            h_act = np.maximum(h, 0)  # ReLU

            # Layer 2: linear (no activation — regression output)
            raw = self._W2[arm] @ h_act + self._b2[arm]

        out = float(np.clip(np.nan_to_num(raw, nan=0.0), -100.0, 100.0).item())

        return out, h_act

    def _compute_uncertainty(self, context: np.ndarray, arm: int) -> float:
        """Compute exploration bonus for the given arm and context.

        Uses the diagonal Fisher information approximation:
            uncertainty ~ sqrt(context^T @ diag(1/grad_sq_sum) @ context)
        """
        ctx = context.astype(np.float64)
        inv_fisher = 1.0 / (self._grad_sq_sum[arm] + 1e-6)
        # Project context through inverse Fisher diagonal
        uncertainty = np.sqrt(np.sum(ctx ** 2 * inv_fisher))
        # Scale by inverse sqrt of action count for UCB-like decay
        count = max(1, self._action_counts[arm])
        uncertainty *= np.sqrt(np.log(max(2, self._total_updates + 1)) / count)
        return float(uncertainty)

    def select_action(self, context: np.ndarray) -> tuple[str, float]:
        """Select action using NeuralUCB.

        Args:
            context: (context_dim,) numpy array.

        Returns:
            (action_name, confidence) where confidence is the UCB score
            normalized to [0, 1].
        """
        ctx = self._validate_context(context)

        scores = np.zeros(self.n_actions, dtype=np.float64)
        rewards = np.zeros(self.n_actions, dtype=np.float64)

        for arm in range(self.n_actions):
            pred_reward, _ = self._forward(ctx, arm)
            uncertainty = self._compute_uncertainty(ctx, arm)
            ucb_score = pred_reward + self.alpha * uncertainty
            scores[arm] = ucb_score
            rewards[arm] = pred_reward

        best_arm = int(np.argmax(scores))
        action = self.ACTIONS[best_arm]

        # Confidence: softmax over scores, take max
        probs = softmax(scores)
        confidence = float(probs[best_arm])

        return action, confidence

    def update(self, context: np.ndarray, action: str, reward: float) -> None:
        """Online gradient update after observing reward.

        Args:
            context: (context_dim,) the context when action was selected.
            action: the action that was executed.
            reward: observed reward (e.g., 1.0 for success, -1.0 for error,
                    derived from ErrP detector or user feedback).
        """
        if action not in self._action_idx:
            logger.warning("Unknown action '%s' in bandit update", action)
            return

        arm = self._action_idx[action]
        ctx = self._validate_context(context)

        # Forward pass
        pred_reward, h_act = self._forward(ctx, arm)
        error = reward - pred_reward

        # Backward pass (gradient of MSE loss)
        # d_loss/d_W2 = error * h_act
        # d_loss/d_b2 = error
        d_W2 = error * h_act.reshape(1, -1)
        d_b2 = np.array([error])

        # d_loss/d_h = error * W2^T (only where ReLU was active)
        d_h = (error * self._W2[arm].flatten()) * (h_act > 0).astype(np.float64)

        # d_loss/d_W1 = d_h outer ctx
        d_W1 = np.outer(d_h, ctx)
        d_b1 = d_h

        # SGD update
        self._W2[arm] += self.lr * d_W2
        self._b2[arm] += self.lr * d_b2
        self._W1[arm] += self.lr * d_W1
        self._b1[arm] += self.lr * d_b1

        # Update Fisher diagonal approximation (gradient magnitude tracking)
        # Use the context magnitude as a proxy for how informative this update was
        self._grad_sq_sum[arm] += ctx ** 2 * error ** 2

        # Bookkeeping
        self._action_counts[arm] += 1
        self._total_updates += 1

        if len(self._reward_history) < self._max_reward_history:
            self._reward_history.append(reward)
        else:
            self._reward_history[self._total_updates % self._max_reward_history] = reward

    def _validate_context(self, context: np.ndarray) -> np.ndarray:
        """Validate, resize, and sanitize context vector."""
        ctx = np.asarray(context, dtype=np.float64).ravel()
        if len(ctx) < self.context_dim:
            ctx = np.concatenate([ctx, np.zeros(self.context_dim - len(ctx))])
        elif len(ctx) > self.context_dim:
            ctx = ctx[:self.context_dim]
        # Sanitize: replace NaN/inf, clamp to reasonable range
        ctx = np.nan_to_num(ctx, nan=0.0, posinf=1e4, neginf=-1e4)
        ctx = np.clip(ctx, -1e4, 1e4)
        return ctx

    def build_context(self,
                      eegnet_features: np.ndarray,
                      hmm_state: str,
                      element_role: str,
                      engagement: float,
                      faa: float,
                      fixation_duration: float) -> np.ndarray:
        """Build a context vector from component signals.

        Convenience method that assembles the 48-dim context vector from
        individual inputs.

        Args:
            eegnet_features: (32,) from EEGNetExtractor
            hmm_state: one of BrainStateHMM.STATES
            element_role: one of ELEMENT_ROLES
            engagement: [0, 1] engagement score
            faa: frontal alpha asymmetry value
            fixation_duration: current fixation duration in seconds

        Returns:
            (48,) context vector
        """
        ctx = np.zeros(self.context_dim, dtype=np.float64)

        # EEGNet features [0:32]
        feat = np.asarray(eegnet_features, dtype=np.float64).ravel()
        n = min(32, len(feat))
        ctx[:n] = feat[:n]

        # HMM state one-hot [32:37]
        hmm_states = BrainStateHMM.STATES
        if hmm_state in hmm_states:
            ctx[32 + hmm_states.index(hmm_state)] = 1.0

        # Element role one-hot [37:45]
        if element_role in self.ELEMENT_ROLES:
            ctx[37 + self.ELEMENT_ROLES.index(element_role)] = 1.0
        else:
            ctx[37 + self.ELEMENT_ROLES.index("other")] = 1.0

        # Scalar features [45:48]
        ctx[45] = float(np.clip(engagement, 0.0, 1.0))
        ctx[46] = float(np.clip(faa, -2.0, 2.0))
        ctx[47] = float(np.clip(fixation_duration, 0.0, 10.0))

        return ctx

    def get_action_values(self, context: np.ndarray) -> dict[str, float]:
        """Return predicted reward for each action (no exploration bonus)."""
        ctx = self._validate_context(context)
        values = {}
        for arm, action in enumerate(self.ACTIONS):
            pred, _ = self._forward(ctx, arm)
            values[action] = float(pred)
        return values

    def get_stats(self) -> dict:
        """Return diagnostic statistics."""
        recent_n = min(100, len(self._reward_history))
        recent_rewards = self._reward_history[-recent_n:] if recent_n > 0 else []

        return {
            "total_updates": self._total_updates,
            "action_counts": {
                a: int(self._action_counts[i])
                for i, a in enumerate(self.ACTIONS)
            },
            "recent_avg_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            "recent_reward_std": float(np.std(recent_rewards)) if recent_rewards else 0.0,
            "alpha": self.alpha,
            "learning_rate": self.lr,
        }

    def save(self, path: str) -> None:
        """Save bandit parameters to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "context_dim": self.context_dim,
            "hidden_dim": self.hidden_dim,
            "alpha": self.alpha,
            "lr": self.lr,
            "action_counts": self._action_counts,
            "total_updates": self._total_updates,
        }
        # Save per-arm weights
        for arm in range(self.n_actions):
            data[f"W1_{arm}"] = self._W1[arm]
            data[f"b1_{arm}"] = self._b1[arm]
            data[f"W2_{arm}"] = self._W2[arm]
            data[f"b2_{arm}"] = self._b2[arm]
            data[f"grad_sq_{arm}"] = self._grad_sq_sum[arm]

        np.savez(path, **data)
        logger.info("Saved NeuralUCB bandit to %s (%d updates)",
                     path, self._total_updates)

    def load(self, path: str) -> bool:
        """Load bandit parameters from disk. Returns True on success."""
        p = Path(path)
        if not p.exists():
            logger.info("No saved bandit at %s", path)
            return False

        try:
            data = np.load(path, allow_pickle=True)
            self._action_counts = data["action_counts"]
            self._total_updates = int(data["total_updates"])

            for arm in range(self.n_actions):
                self._W1[arm] = data[f"W1_{arm}"]
                self._b1[arm] = data[f"b1_{arm}"]
                self._W2[arm] = data[f"W2_{arm}"]
                self._b2[arm] = data[f"b2_{arm}"]
                self._grad_sq_sum[arm] = data[f"grad_sq_{arm}"]

            logger.info("Loaded NeuralUCB bandit from %s (%d updates)",
                         path, self._total_updates)
            return True
        except Exception as e:
            logger.error("Failed to load bandit: %s", e)
            return False


# ============================================================================
# Integration: NeuralPipeline (orchestrates all 5 components)
# ============================================================================

class NeuralPipeline:
    """Orchestrates the full neural processing pipeline.

    Wires together EEGNet, HMM, ErrP detector, intent accumulator, and
    contextual bandit into a single update loop.

    Usage:
        pipe = NeuralPipeline()

        # Each frame (~10 Hz):
        result = pipe.process(
            eeg_window=eeg_4ch,           # (4, 256) raw EEG
            band_powers=band_dict,         # from BrainStateEngine
            fixated_element_id="btn_42",
            fixation_duration=0.8,
            element_role="button",
            engagement=0.7,
            faa=0.15,
            last_action_time=1234567890.0,
        )
        # result = {
        #     "features": np.array (32,),
        #     "hmm": {"state": "intending", "probabilities": {...}, "confidence": 0.8},
        #     "errp": 0.12,
        #     "intent": {"btn_42": 0.65, ...},
        #     "top_intent": ("btn_42", 0.65),
        #     "commitment": "medium",
        #     "action": ("click", 0.82),
        # }
    """

    def __init__(self, eegnet_weights_path: Optional[str] = None,
                 hmm_params_path: Optional[str] = None,
                 bandit_params_path: Optional[str] = None):

        # 1. EEGNet feature extractor
        if eegnet_weights_path and Path(eegnet_weights_path).exists():
            self.eegnet = EEGNetExtractor.from_pretrained(eegnet_weights_path)
        else:
            self.eegnet = EEGNetExtractor()

        # 2. HMM brain state estimator
        self.hmm = BrainStateHMM(n_features=38)
        if hmm_params_path:
            self.hmm.load(hmm_params_path)

        # 3. ErrP detector
        self.errp = ErrPDetector()

        # 4. Bayesian intent accumulator
        self.intent = BayesianIntentAccumulator()

        # 5. NeuralUCB bandit
        self.bandit = NeuralUCBBandit()
        if bandit_params_path:
            self.bandit.load(bandit_params_path)

    def process(self,
                eeg_window: np.ndarray,
                band_powers: Optional[dict] = None,
                fixated_element_id: Optional[str] = None,
                fixation_duration: float = 0.0,
                element_role: str = "other",
                engagement: float = 0.0,
                faa: float = 0.0,
                last_action_time: Optional[float] = None,
                current_time: Optional[float] = None) -> dict:
        """Run one full pipeline update.

        Args:
            eeg_window: (4, 256) raw EEG from Muse S.
            band_powers: dict with alpha, beta, theta keys per hemisphere
                (from BrainStateEngine). If None, uses zeros.
            fixated_element_id: currently fixated UI element ID.
            fixation_duration: how long the fixation has lasted (seconds).
            element_role: semantic role of the fixated element.
            engagement: engagement score [0, 1].
            faa: frontal alpha asymmetry.
            last_action_time: timestamp of the most recent action (for ErrP).
            current_time: current timestamp (defaults to time.time()).

        Returns:
            dict with all pipeline outputs.
        """
        if current_time is None:
            current_time = time.time()

        # 1. EEGNet features
        features = self.eegnet.extract_features(eeg_window)

        # 2. Build HMM observation: EEGNet features (32) + band ratios (6)
        band_ratios = self._extract_band_ratios(band_powers)
        hmm_obs = np.concatenate([features, band_ratios])
        hmm_result = self.hmm.update(hmm_obs)

        # 3. ErrP detection
        errp_score = 0.0
        if last_action_time is not None:
            errp_score = self.errp.detect(eeg_window, last_action_time, current_time)

        # 4. Intent accumulation
        intent_beliefs = self.intent.update(
            fixated_element_id=fixated_element_id,
            fixation_duration=fixation_duration,
            engagement=engagement,
            faa=faa,
            hmm_state=hmm_result["state"],
            hmm_confidence=hmm_result["confidence"],
        )
        top_id, top_p = self.intent.get_top_intent()
        commitment = self.intent.get_commitment_level(top_id) if top_id else "none"

        # 5. Action selection
        context = self.bandit.build_context(
            eegnet_features=features,
            hmm_state=hmm_result["state"],
            element_role=element_role,
            engagement=engagement,
            faa=faa,
            fixation_duration=fixation_duration,
        )
        action, action_conf = self.bandit.select_action(context)

        return {
            "features": features,
            "hmm": hmm_result,
            "errp": errp_score,
            "intent": intent_beliefs,
            "top_intent": (top_id, top_p),
            "commitment": commitment,
            "action": (action, action_conf),
            "context_vector": context,
        }

    def record_reward(self, context: np.ndarray, action: str, reward: float) -> None:
        """Feed reward back to the bandit after action execution."""
        self.bandit.update(context, action, reward)

    def record_error(self, element_id: Optional[str] = None) -> None:
        """Record that the last action was an error.

        Resets intent for the element and provides negative reward signal.
        """
        if element_id:
            self.intent.reset_element(element_id)

    def _extract_band_ratios(self, band_powers: Optional[dict]) -> np.ndarray:
        """Extract 6-dim band ratio vector from band power dict.

        Returns [alpha_left, alpha_right, beta_left, beta_right,
                 theta_left, theta_right] — or zeros if no data.
        """
        if band_powers is None:
            return np.zeros(6, dtype=np.float32)

        # Try to extract per-hemisphere values; fall back to averaged
        alpha = float(band_powers.get("alpha", 0.0))
        beta = float(band_powers.get("beta", 0.0))
        theta = float(band_powers.get("theta", 0.0))

        # If per-hemisphere data is available
        alpha_l = float(band_powers.get("alpha_left", alpha))
        alpha_r = float(band_powers.get("alpha_right", alpha))
        beta_l = float(band_powers.get("beta_left", beta))
        beta_r = float(band_powers.get("beta_right", beta))
        theta_l = float(band_powers.get("theta_left", theta))
        theta_r = float(band_powers.get("theta_right", theta))

        return np.array([alpha_l, alpha_r, beta_l, beta_r, theta_l, theta_r],
                        dtype=np.float32)

    def save_all(self, directory: str) -> None:
        """Save all component parameters to a directory."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        self.eegnet.save_weights(str(d / "eegnet_weights.pt"))
        self.hmm.save(str(d / "hmm_params.npz"))
        self.bandit.save(str(d / "bandit_params.npz"))
        logger.info("Saved all pipeline parameters to %s", directory)

    def load_all(self, directory: str) -> None:
        """Load all component parameters from a directory."""
        d = Path(directory)
        eeg_path = d / "eegnet_weights.pt"
        hmm_path = d / "hmm_params.npz"
        bandit_path = d / "bandit_params.npz"

        if eeg_path.exists():
            self.eegnet = EEGNetExtractor.from_pretrained(str(eeg_path))
        if hmm_path.exists():
            self.hmm.load(str(hmm_path))
        if bandit_path.exists():
            self.bandit.load(str(bandit_path))
        logger.info("Loaded pipeline parameters from %s", directory)
