"""Braid — Confidence-gated fusion of multiple Strands.

A Braid weaves strands together. Each strand produces an output and a
confidence score. The braid fuses them using competitive attention gating:
strands that have been accurate get more influence. Strands that have been
wrong automatically lose weight — no manual tuning.

    braid = Braid()
    braid.add(Strand("content", output_dim=8))
    braid.add(Strand("emotion", output_dim=8))
    braid.add(Strand("style", output_dim=8))

    output = braid.fuse(brain_context)         # parallel inference + fusion
    braid.step(reward_vector, brain_context)   # each strand learns its slice
"""

import asyncio
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from neuralrl.strand import Strand

try:
    import redis as redis_lib
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class Braid:
    """Confidence-gated multi-strand fusion with parallel inference.

    The core abstraction: multiple specialized agents run concurrently
    on the same brain data, each producing an output + confidence.
    The braid fuses them into one output vector that's better than
    any individual strand because we know which dimensions to trust.

    Technical innovation: competitive attention gating where weights
    are the reciprocal of each strand's EMA prediction error. This is
    online meta-learning — the system automatically routes trust toward
    whichever strand is currently most accurate for THIS brain.
    """

    def __init__(self, output_dim: int = 8, max_workers: int = 4):
        self._strands: dict[str, Strand] = {}
        self._output_dim = output_dim
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Fusion history
        self._fusion_history = []
        self._total_fusions = 0

        # Redis for streaming pipeline
        self._redis = None
        self._stream_key = "braid:fused"

    def add(self, strand: Strand) -> 'Braid':
        """Add a strand. Returns self for chaining."""
        self._strands[strand.name] = strand
        return self

    def connect_redis(self, host="localhost", port=6379):
        """Connect Redis for streaming output + strand memory."""
        if not HAS_REDIS:
            return self
        try:
            self._redis = redis_lib.Redis(host=host, port=port, decode_responses=False)
            self._redis.ping()
            for strand in self._strands.values():
                strand.connect_redis(self._redis)
            print(f"[BRAID] Redis connected, {len(self._strands)} strands", flush=True)
        except Exception:
            self._redis = None
        return self

    def fuse(self, brain_context: np.ndarray) -> np.ndarray:
        """Run all strands in parallel, fuse outputs with confidence gating.

        This is synchronous but internally parallelized. Each strand runs
        in its own thread on the thread pool — true concurrency for I/O-bound
        models, GIL-friendly for numpy compute.
        """
        ctx = np.asarray(brain_context, dtype=np.float64).ravel()

        # Run all strands concurrently
        futures = {}
        for name, strand in self._strands.items():
            futures[name] = self._executor.submit(strand.infer, ctx)

        # Collect results
        outputs = {}
        confidences = {}
        for name, future in futures.items():
            try:
                output, conf = future.result(timeout=0.5)
                outputs[name] = output[:self._output_dim]
                confidences[name] = conf
            except Exception:
                outputs[name] = np.full(self._output_dim, 0.5)
                confidences[name] = 0.1

        # Competitive attention gating
        fused = self._attention_fuse(outputs, confidences, ctx)
        self._total_fusions += 1

        # Publish to Redis stream
        if self._redis:
            try:
                self._redis.xadd(self._stream_key, {
                    "fused": fused.tobytes(),
                    "confidences": str({k: round(v, 3) for k, v in confidences.items()}),
                }, maxlen=1000)
            except Exception:
                pass

        return fused

    async def fuse_async(self, brain_context: np.ndarray) -> np.ndarray:
        """Async version — runs strands in executor, doesn't block event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fuse, brain_context)

    def _attention_fuse(self, outputs: dict, confidences: dict,
                        brain_context: np.ndarray) -> np.ndarray:
        """Competitive attention gating.

        Query = brain state, Keys = strand identities, Values = strand outputs.
        Weights = confidence scores (reciprocal of EMA prediction error).

        When strands agree, output is strong. When they disagree, the more
        confident strand dominates. This captures interactions without
        learned parameters that would overfit on small BCI datasets.
        """
        if not outputs:
            return np.full(self._output_dim, 0.5)

        names = list(outputs.keys())
        vectors = np.array([outputs[n] for n in names])
        weights = np.array([confidences[n] for n in names])

        # Temperature scaling: sharpen or soften based on agreement
        agreement = 1.0 - np.mean(np.std(vectors, axis=0))
        temperature = 0.5 + 0.5 * agreement  # high agreement → sharper weights

        weights = weights ** (1.0 / max(temperature, 0.1))
        weights /= weights.sum() + 1e-8

        fused = np.average(vectors, axis=0, weights=weights)

        # Record fusion state for diagnostics
        self._fusion_history.append({
            "weights": {n: round(float(w), 3) for n, w in zip(names, weights)},
            "agreement": round(float(agreement), 3),
            "temperature": round(float(temperature), 3),
        })
        if len(self._fusion_history) > 100:
            self._fusion_history = self._fusion_history[-100:]

        return np.clip(fused, 0, 1)

    def step(self, reward_vector: dict[str, float], brain_context: np.ndarray):
        """Update each strand from its reward dimension.

        Args:
            reward_vector: {"content": 0.7, "emotion": 0.3, "style": 0.8}
                          Each strand gets its own reward slice.
            brain_context: brain state at time of action
        """
        ctx = np.asarray(brain_context, dtype=np.float64).ravel()
        strand_states = {n: s.confidence for n, s in self._strands.items()}

        for name, strand in self._strands.items():
            reward = reward_vector.get(name, 0.0)
            strand.step(reward, ctx, other_strand_states=strand_states)

    def decompose_reward(self, brain_snapshot: dict, overall_reward: float) -> dict[str, float]:
        """Decompose a scalar brain reward into strand-specific rewards.

        Uses the brain signal dimensions to attribute reward:
          - engagement delta → content reward
          - valence delta → emotion reward
          - focus → style reward (focused = style is working)
          - cognitive_load inverse → complexity reward

        Override this for custom decomposition.
        """
        eng = brain_snapshot.get("engagement", 0.5)
        val = brain_snapshot.get("valence", 0.5)
        focus = brain_snapshot.get("focus", 0.5)
        cog = brain_snapshot.get("cognitive_load", 0.5)

        rewards = {}
        strand_names = list(self._strands.keys())

        # Default mapping: distribute based on brain dimensions
        if len(strand_names) >= 1:
            rewards[strand_names[0]] = overall_reward * (0.5 + 0.5 * eng)
        if len(strand_names) >= 2:
            rewards[strand_names[1]] = overall_reward * (0.5 + 0.5 * (val - 0.5) * 2)
        if len(strand_names) >= 3:
            rewards[strand_names[2]] = overall_reward * (0.5 + 0.5 * focus)
        if len(strand_names) >= 4:
            rewards[strand_names[3]] = overall_reward * (0.5 + 0.5 * (1 - cog))

        # Fill remaining with overall
        for name in strand_names[4:]:
            rewards[name] = overall_reward

        return {k: float(np.clip(v, -1, 1)) for k, v in rewards.items()}

    # ── Introspection ──────────────────────────────────────────

    def get_strand_stats(self) -> dict:
        return {name: strand.get_stats() for name, strand in self._strands.items()}

    def get_diagnostics(self) -> dict:
        return {name: strand.get_diagnostic() for name, strand in self._strands.items()}

    def get_fusion_state(self) -> dict:
        if not self._fusion_history:
            return {"weights": {}, "agreement": 0, "temperature": 1}
        return self._fusion_history[-1]

    def get_confidence_map(self) -> dict[str, float]:
        return {name: strand.confidence for name, strand in self._strands.items()}

    @property
    def strand_count(self) -> int:
        return len(self._strands)

    @property
    def total_fusions(self) -> int:
        return self._total_fusions
