"""BrainReward — EEG signals as a reward function.

Converts continuous brain state into scalar reward for RL agents.
Works with any EEG source that provides engagement, valence, focus,
relaxation, cognitive_load, error_response, and jaw_clench.

Reward = weighted combination of brain state CHANGES after an action,
not absolute values. This is key — we care about the brain's REACTION
to what the agent did, not the resting state.

Supports:
  - Delta-based reward (change in engagement/valence after action)
  - ErrP penalty (error-related potential detection)
  - Jaw clench bonus (explicit confirmation signal)
  - Baseline-subtracted reward (advantage estimation)
  - Configurable reward weights per application
"""

import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BrainSnapshot:
    engagement: float = 0.0
    focus: float = 0.0
    valence: float = 0.5
    cognitive_load: float = 0.0
    relaxation: float = 0.0
    error_response: float = 0.0
    jaw_clench: bool = False
    band_powers: dict = field(default_factory=dict)
    asymmetry: float = 0.0
    timestamp: float = 0.0

    def to_context(self):
        """12-dim context vector for the RL policy."""
        bp = self.band_powers
        alpha = bp.get("alpha", 0) + 1e-6
        theta = bp.get("theta", 0) + 1e-6
        beta = bp.get("beta", 0) + 1e-6
        gamma = bp.get("gamma", 0) + 1e-6
        delta = bp.get("delta", 0) + 1e-6
        return np.array([
            self.engagement, self.focus, self.valence,
            self.cognitive_load, self.relaxation, self.asymmetry,
            alpha / theta, beta / alpha, gamma / beta,
            theta / alpha, delta / theta, self.error_response,
        ], dtype=np.float64)


DEFAULT_WEIGHTS = {
    "engagement_delta": 0.35,
    "valence_delta": 0.30,
    "focus": 0.10,
    "no_error": 0.15,
    "calm": 0.10,
}


class BrainReward:
    """Computes scalar reward from brain state changes."""

    def __init__(self, weights=None, clench_bonus=0.4, errp_penalty=-0.5):
        self.weights = weights or DEFAULT_WEIGHTS
        self.clench_bonus = clench_bonus
        self.errp_penalty = errp_penalty

        self._pre_action: BrainSnapshot | None = None
        self._baseline_reward = 0.0
        self._baseline_ema = 0.1
        self._reward_history: list[float] = []

    def mark_pre_action(self, snapshot: BrainSnapshot):
        """Call RIGHT BEFORE the agent acts. Captures baseline brain state."""
        self._pre_action = snapshot

    def compute(self, post_snapshot: BrainSnapshot) -> float:
        """Call AFTER the agent acts. Returns reward from brain reaction."""
        if self._pre_action is None:
            return 0.0

        pre = self._pre_action
        post = post_snapshot
        w = self.weights

        eng_delta = post.engagement - pre.engagement
        val_delta = post.valence - pre.valence
        no_error = 1.0 - post.error_response

        raw_reward = (
            w["engagement_delta"] * np.clip(eng_delta * 3, -1, 1)
            + w["valence_delta"] * np.clip(val_delta * 3, -1, 1)
            + w["focus"] * post.focus
            + w["no_error"] * no_error
            + w["calm"] * (1.0 - post.cognitive_load)
        )

        if post.jaw_clench:
            raw_reward += self.clench_bonus
        if post.error_response > 0.5:
            raw_reward += self.errp_penalty

        reward = float(np.clip(raw_reward, -1, 1))

        advantage = reward - self._baseline_reward
        self._baseline_reward += self._baseline_ema * (reward - self._baseline_reward)
        self._reward_history.append(reward)

        self._pre_action = None
        return advantage

    def get_context(self, snapshot: BrainSnapshot) -> np.ndarray:
        """Extract context vector from brain snapshot for the policy."""
        return snapshot.to_context()

    @property
    def total_updates(self):
        return len(self._reward_history)

    @property
    def recent_avg(self):
        if not self._reward_history:
            return 0.0
        n = min(20, len(self._reward_history))
        return float(np.mean(self._reward_history[-n:]))

    @property
    def baseline(self):
        return self._baseline_reward

    def get_history(self):
        return list(self._reward_history)
