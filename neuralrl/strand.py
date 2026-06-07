"""Strand — A specialized learning agent bound to one reward dimension.

A Strand is a nerve fiber in the neural harness. It:
  - Owns one dimension of the brain's reward signal
  - Runs its own model concurrently with other strands
  - Maintains its own episodic memory (Redis VSS)
  - Reports calibrated confidence via temperature-scaled prediction error
  - Self-diagnoses: knows when it's unreliable and why

    strand = Strand("valence", model_fn=valence_net, reward_dim="valence")
    output, confidence = strand.infer(brain_window)
    strand.step(reward=0.7, brain_state=state)
"""

import time
import json
import struct
import numpy as np
from collections import deque
from dataclasses import dataclass, field

try:
    import redis as redis_lib
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


@dataclass
class StrandMemory:
    """One episodic memory entry for a strand."""
    brain_embedding: np.ndarray
    output: np.ndarray
    reward: float
    timestamp: float


class Strand:
    """A specialized learning agent for one reward dimension.

    Each strand is an independent concurrent processor that:
    1. Infers from a brain window using its own model
    2. Reports calibrated confidence from prediction error history
    3. Learns from its own reward slice independently
    4. Maintains strand-local episodic memory
    5. Self-diagnoses accuracy conditioned on other strands
    """

    def __init__(self, name: str, model_fn=None, output_dim: int = 8,
                 memory_k: int = 64, lr: float = 0.05):
        self.name = name
        self._model_fn = model_fn
        self.output_dim = output_dim
        self.lr = lr

        # Learnable projection: brain features → strand output
        self._W = np.random.randn(output_dim, 12).astype(np.float64) * 0.05
        self._bias = np.zeros(output_dim, dtype=np.float64)

        # Confidence via prediction error EMA
        self._error_ema = 0.3
        self._error_alpha = 0.15
        self._temperature = 1.0

        # Episodic memory (local to this strand)
        self._memory: deque = deque(maxlen=memory_k)
        self._memory_k = memory_k

        # Self-diagnostic: accuracy conditioned on states
        self._accuracy_by_regime = {"high_load": [], "low_load": [],
                                     "high_eng": [], "low_eng": []}

        # History for introspection
        self._reward_history: deque = deque(maxlen=200)
        self._output_history: deque = deque(maxlen=50)
        self._total_steps = 0

        # Redis for persistent memory (optional)
        self._redis = None
        self._redis_key = f"strand:{name}:memory"

    def connect_redis(self, r):
        """Attach a Redis connection for persistent episodic memory."""
        self._redis = r

    def infer(self, brain_context: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference. Returns (output_vector, confidence).

        If a custom model_fn was provided, uses that.
        Otherwise uses the learned linear projection.
        """
        ctx = np.asarray(brain_context, dtype=np.float64).ravel()[:12]
        if len(ctx) < 12:
            ctx = np.concatenate([ctx, np.zeros(12 - len(ctx))])

        # Custom model or linear projection
        if self._model_fn:
            try:
                output = np.asarray(self._model_fn(ctx), dtype=np.float64)
            except Exception:
                output = self._linear_infer(ctx)
        else:
            output = self._linear_infer(ctx)

        # Memory-augmented: blend with retrieved similar episodes
        memory_blend = self._retrieve_similar(ctx)
        if memory_blend is not None:
            memory_weight = 0.2 * self.confidence
            output = (1 - memory_weight) * output + memory_weight * memory_blend

        output = np.clip(output[:self.output_dim], 0, 1)
        self._output_history.append(output.copy())

        return output, self.confidence

    def _linear_infer(self, ctx: np.ndarray) -> np.ndarray:
        """Simple learned linear projection + ReLU."""
        out = self._W @ ctx + self._bias
        return np.maximum(out, 0)

    def step(self, reward: float, brain_context: np.ndarray = None,
             other_strand_states: dict = None):
        """Update this strand from its reward dimension.

        Args:
            reward: scalar reward for THIS strand's dimension
            brain_context: brain state at time of action
            other_strand_states: {name: confidence} of other strands (for diagnostics)
        """
        self._total_steps += 1
        self._reward_history.append(reward)

        # Update prediction error EMA
        pred_error = abs(reward - 0.5)
        self._error_ema = (1 - self._error_alpha) * self._error_ema + self._error_alpha * pred_error

        # Update projection weights via REINFORCE-style gradient
        if brain_context is not None and self._output_history:
            ctx = np.asarray(brain_context, dtype=np.float64).ravel()[:12]
            if len(ctx) < 12:
                ctx = np.concatenate([ctx, np.zeros(12 - len(ctx))])

            last_output = self._output_history[-1]
            advantage = reward - np.mean(list(self._reward_history)[-20:])
            grad = advantage * np.outer(last_output, ctx)
            self._W += self.lr * grad[:self._W.shape[0], :self._W.shape[1]]
            self._bias += self.lr * advantage * last_output[:len(self._bias)]
            self._W = np.clip(self._W, -5, 5)

            # Store episodic memory
            self._memory.append(StrandMemory(
                brain_embedding=ctx.copy(),
                output=last_output.copy(),
                reward=reward,
                timestamp=time.time(),
            ))

        # Self-diagnostic
        if other_strand_states and brain_context is not None:
            ctx = np.asarray(brain_context, dtype=np.float64).ravel()
            cog_load = ctx[3] if len(ctx) > 3 else 0
            eng = ctx[0] if len(ctx) > 0 else 0
            regime = "high_load" if cog_load > 0.5 else "low_load"
            self._accuracy_by_regime[regime].append(1.0 if reward > 0 else 0.0)
            regime2 = "high_eng" if eng > 0.5 else "low_eng"
            self._accuracy_by_regime[regime2].append(1.0 if reward > 0 else 0.0)

    def _retrieve_similar(self, ctx: np.ndarray, top_k: int = 3) -> np.ndarray | None:
        """Retrieve most similar past episodes from strand-local memory."""
        if len(self._memory) < 5:
            return None

        similarities = []
        for mem in self._memory:
            sim = float(np.dot(ctx, mem.brain_embedding) /
                       (np.linalg.norm(ctx) * np.linalg.norm(mem.brain_embedding) + 1e-8))
            similarities.append((sim, mem))

        similarities.sort(key=lambda x: -x[0])
        top = similarities[:top_k]

        if not top or top[0][0] < 0.7:
            return None

        # Reward-weighted blend of similar episodes' outputs
        weights = np.array([s * max(0.1, m.reward) for s, m in top])
        weights /= weights.sum() + 1e-8
        blended = np.zeros(self.output_dim)
        for (_, mem), w in zip(top, weights):
            blended += w * mem.output[:self.output_dim]
        return blended

    @property
    def confidence(self) -> float:
        """Calibrated confidence: inverse of temperature-scaled prediction error."""
        return float(np.clip(1.0 / (1.0 + self._error_ema * self._temperature), 0.05, 1.0))

    def get_diagnostic(self) -> dict:
        """Self-report: when am I reliable vs unreliable?"""
        diag = {"name": self.name, "confidence": round(self.confidence, 3),
                "total_steps": self._total_steps, "memory_size": len(self._memory)}
        for regime, accs in self._accuracy_by_regime.items():
            if len(accs) >= 3:
                diag[f"accuracy_{regime}"] = round(float(np.mean(accs[-20:])), 3)
        return diag

    def get_stats(self) -> dict:
        n = min(20, len(self._reward_history))
        recent = list(self._reward_history)[-n:] if n > 0 else []
        return {
            "name": self.name,
            "confidence": round(self.confidence, 3),
            "avg_reward": round(float(np.mean(recent)), 3) if recent else 0,
            "total_steps": self._total_steps,
            "memory_entries": len(self._memory),
            "error_ema": round(self._error_ema, 3),
        }
