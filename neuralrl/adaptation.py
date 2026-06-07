"""AdaptationTracker — Measures how well the system has learned this brain.

Tracks the RL agent's confidence by monitoring:
  - Reward variance (dropping = learning)
  - Exploration rate (dropping = exploiting)
  - Posterior width (shrinking = confident)
  - Reward trend (rising = improving)

Outputs a single 0-100% "adaptation score" that the demo visualization
shows climbing in real-time. This IS the demo's narrative arc.
"""

import numpy as np
from collections import deque


class AdaptationTracker:
    """Computes a 0-100% adaptation score from RL convergence signals."""

    def __init__(self, window=15):
        self._window = window
        self._rewards = deque(maxlen=200)
        self._exploration_rates = deque(maxlen=200)
        self._posterior_widths = deque(maxlen=200)
        self._score = 0.0
        self._score_history = []

    def update(self, reward: float, exploration_rate: float,
               posterior_stds: np.ndarray) -> float:
        """Update and return adaptation score 0-100."""
        self._rewards.append(reward)
        self._exploration_rates.append(exploration_rate)
        self._posterior_widths.append(float(np.mean(posterior_stds)))

        if len(self._rewards) < 3:
            self._score = 0.0
            self._score_history.append(0.0)
            return 0.0

        n = min(self._window, len(self._rewards))

        # 1. Reward variance dropping = learning (0-30 points)
        recent_var = float(np.var(list(self._rewards)[-n:]))
        early_var = float(np.var(list(self._rewards)[:max(3, n)])) if len(self._rewards) > n else recent_var + 0.1
        var_improvement = np.clip((early_var - recent_var) / (early_var + 1e-6), 0, 1)
        var_score = var_improvement * 30

        # 2. Exploration rate dropping = exploiting (0-25 points)
        current_explore = self._exploration_rates[-1]
        initial_explore = self._exploration_rates[0] if self._exploration_rates else 0.5
        explore_drop = np.clip((initial_explore - current_explore) / (initial_explore + 1e-6), 0, 1)
        explore_score = explore_drop * 25

        # 3. Posterior width shrinking = confident (0-25 points)
        current_width = self._posterior_widths[-1]
        initial_width = self._posterior_widths[0] if self._posterior_widths else 0.5
        width_drop = np.clip((initial_width - current_width) / (initial_width + 1e-6), 0, 1)
        width_score = width_drop * 25

        # 4. Reward trending up = improving (0-20 points)
        recent_avg = float(np.mean(list(self._rewards)[-n:]))
        early_avg = float(np.mean(list(self._rewards)[:max(3, n)]))
        reward_improvement = np.clip((recent_avg - early_avg + 0.5) / 1.0, 0, 1)
        trend_score = reward_improvement * 20

        self._score = float(np.clip(var_score + explore_score + width_score + trend_score, 0, 100))
        self._score_history.append(self._score)

        return self._score

    @property
    def score(self):
        return self._score

    @property
    def score_history(self):
        return list(self._score_history)

    def get_breakdown(self) -> dict:
        """Return score components for debugging/display."""
        if len(self._rewards) < 3:
            return {"total": 0, "variance": 0, "exploration": 0, "confidence": 0, "trend": 0}

        n = min(self._window, len(self._rewards))
        recent_var = float(np.var(list(self._rewards)[-n:]))
        recent_avg = float(np.mean(list(self._rewards)[-n:]))

        return {
            "total": round(self._score, 1),
            "reward_variance": round(recent_var, 4),
            "exploration_rate": round(float(self._exploration_rates[-1]), 4),
            "posterior_width": round(float(self._posterior_widths[-1]), 4),
            "recent_avg_reward": round(recent_avg, 4),
            "n_updates": len(self._rewards),
        }

    def reset(self):
        self._rewards.clear()
        self._exploration_rates.clear()
        self._posterior_widths.clear()
        self._score = 0.0
        self._score_history = []
