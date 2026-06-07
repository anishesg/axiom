"""CTS — Contextual Thompson Sampling for neural RL.

A continuous-action contextual bandit that maintains Gaussian posteriors
over action parameters, conditioned on brain-state context.

Each action dimension has:
  - A mean (what we think is best for this brain)
  - A variance (how uncertain we are)

On each step:
  1. Observe brain context (12-dim)
  2. Compute context-adjusted mean: mu + W @ context
  3. Sample action from N(adjusted_mean, variance)
  4. After reward, update mean/variance via REINFORCE gradient

The posteriors VISIBLY collapse as the system learns — wide distributions
(high uncertainty, lots of exploration) → narrow peaks (confident, exploiting).
This convergence IS the demo.
"""

import numpy as np
from pathlib import Path


class CTS:
    """Contextual Thompson Sampling with Gaussian posteriors."""

    def __init__(self, action_dim, context_dim=12, param_names=None):
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.param_names = param_names or [f"a{i}" for i in range(action_dim)]

        # Posterior parameters
        self.mu = np.full(action_dim, 0.5, dtype=np.float64)
        self.log_var = np.zeros(action_dim, dtype=np.float64)

        # Context-dependent adjustment
        self.W = np.zeros((action_dim, context_dim), dtype=np.float64)

        # Learning rates
        self.lr_mu = 0.12
        self.lr_var = 0.04
        self.lr_W = 0.01

        # State
        self._pending_action = None
        self._pending_context = None
        self._pending_noise = None
        self._total_updates = 0

        # Full trajectory for visualization
        self._mu_history = []
        self._var_history = []
        self._action_history = []

    @property
    def std(self):
        return np.exp(0.5 * self.log_var)

    @property
    def exploration_rate(self):
        return float(np.mean(self.std))

    def sample(self, context: np.ndarray) -> np.ndarray:
        """Sample an action from the posterior given brain context."""
        ctx = self._prep_context(context)

        adjusted_mu = self.mu + self.W @ ctx
        std = np.clip(self.std, 0.03, 0.5)

        noise = np.random.randn(self.action_dim)
        action = adjusted_mu + std * noise
        action = np.clip(action, 0, 1)

        self._pending_action = action.copy()
        self._pending_context = ctx.copy()
        self._pending_noise = noise.copy()

        return action

    def update(self, reward: float):
        """Update posteriors using REINFORCE gradient after observing reward."""
        if self._pending_context is None:
            return

        ctx = self._pending_context
        noise = self._pending_noise
        std = np.clip(self.std, 0.03, 0.5)

        # REINFORCE: d_log_pi/d_mu = noise / std
        grad_mu = reward * noise / (std + 1e-8)
        self.mu += self.lr_mu * grad_mu
        self.mu = np.clip(self.mu, 0.0, 1.0)

        # Context weight update
        self.W += self.lr_W * np.outer(grad_mu, ctx)

        # Variance update: shrink when reward is consistent, grow when noisy
        grad_logvar = reward * (noise ** 2 - 1) * 0.5
        self.log_var += self.lr_var * grad_logvar
        self.log_var = np.clip(self.log_var, -4, 0.5)

        # Record history
        self._total_updates += 1
        self._mu_history.append(self.mu.copy())
        self._var_history.append(self.std.copy())
        self._action_history.append(self._pending_action.copy())

        self._pending_context = None
        self._pending_action = None
        self._pending_noise = None

    def get_posteriors(self) -> dict:
        """Return current posterior state for visualization."""
        return {
            name: {
                "mean": float(self.mu[i]),
                "std": float(self.std[i]),
                "low": float(np.clip(self.mu[i] - 2 * self.std[i], 0, 1)),
                "high": float(np.clip(self.mu[i] + 2 * self.std[i], 0, 1)),
            }
            for i, name in enumerate(self.param_names)
        }

    def get_trajectories(self) -> dict:
        """Return full history for learning graph visualization."""
        return {
            "means": {
                name: [float(h[i]) for h in self._mu_history]
                for i, name in enumerate(self.param_names)
            },
            "stds": {
                name: [float(h[i]) for h in self._var_history]
                for i, name in enumerate(self.param_names)
            },
            "actions": {
                name: [float(h[i]) for h in self._action_history]
                for i, name in enumerate(self.param_names)
            },
        }

    def get_stats(self) -> dict:
        return {
            "total_updates": self._total_updates,
            "exploration_rate": self.exploration_rate,
            "posteriors": self.get_posteriors(),
        }

    def _prep_context(self, ctx):
        ctx = np.asarray(ctx, dtype=np.float64).ravel()
        if len(ctx) < self.context_dim:
            ctx = np.concatenate([ctx, np.zeros(self.context_dim - len(ctx))])
        elif len(ctx) > self.context_dim:
            ctx = ctx[:self.context_dim]
        return np.nan_to_num(ctx, nan=0.0)

    def save(self, path):
        np.savez(path, mu=self.mu, log_var=self.log_var, W=self.W,
                 total=self._total_updates)

    def load(self, path):
        if not Path(path).exists():
            return False
        d = np.load(path)
        self.mu = d["mu"]
        self.log_var = d["log_var"]
        self.W = d["W"]
        self._total_updates = int(d["total"])
        return True
