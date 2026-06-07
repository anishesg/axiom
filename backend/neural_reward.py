"""Neural reward computation from EEG brain state snapshots.

During each reading phase, brain state is sampled at ~2Hz.
After reading all responses, snapshots are aggregated into reward scores.
Reward = weighted combination of engagement, valence, focus minus cognitive load and relaxation.
"""

import numpy as np
from dataclasses import dataclass, asdict

try:
    import weave
    HAS_WEAVE = True
except ImportError:
    HAS_WEAVE = False


@dataclass
class NeuralReward:
    agent_id: str
    engagement: float = 0.0
    valence: float = 0.0
    focus: float = 0.0
    cognitive_load: float = 0.0
    relaxation: float = 0.0
    total: float = 0.0
    n_samples: int = 0

    def to_dict(self):
        return asdict(self)


class NeuralRewardComputer:
    """Maps brain state snapshots to scalar reward scores per agent."""

    WEIGHTS = {
        "engagement": 0.35,
        "valence": 0.25,
        "focus": 0.25,
        "cognitive_load": -0.10,
        "relaxation": -0.05,
    }

    def __init__(self):
        self.baseline = None

    def set_baseline(self, snapshots: list[dict]):
        if not snapshots:
            self.baseline = {
                "engagement": 0.33, "valence": 0.5, "focus": 0.25,
                "cognitive_load": 0.3, "relaxation": 0.5,
            }
            return
        self.baseline = {
            key: float(np.mean([s.get(key, 0) for s in snapshots]))
            for key in self.WEIGHTS
        }

    def compute_reward(self, agent_id: str, snapshots: list[dict]) -> NeuralReward:
        reward = NeuralReward(agent_id=agent_id, n_samples=len(snapshots))
        if not snapshots:
            return reward

        for key in self.WEIGHTS:
            values = [s.get(key, 0) for s in snapshots]
            setattr(reward, key, round(float(np.mean(values)), 4))

        total = 0.0
        for key, weight in self.WEIGHTS.items():
            val = getattr(reward, key)
            baseline_val = self.baseline.get(key, 0) if self.baseline else 0
            total += weight * (val - baseline_val)

        reward.total = round(float(np.clip(0.5 + total * 2, 0.05, 0.99)), 4)
        return reward

    def rank_rewards(self, rewards: list[NeuralReward]) -> list[NeuralReward]:
        return sorted(rewards, key=lambda r: r.total, reverse=True)
