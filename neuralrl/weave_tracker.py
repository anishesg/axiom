"""WeaveTracker — Brain-as-evaluation-function via W&B Weave.

Deeply integrates Weave into the neural RL loop:
  1. Traces: every pipeline call (intent → style → generate → evaluate → speak)
  2. RL Model versioning: policy snapshots as weave.Model, compare v1 vs v20
  3. Brain episodic memory: every (brain, response, reward) stored as dataset
  4. Neural evaluation: brain signals AS the scorer function — the brain judges AI
  5. Feedback: attach brain reward/engagement/valence to every trace

The brain is the evaluation function. Weave tracks how well AI learns to
speak for this specific brain, measured by that brain's own neural signals.
"""

import time
import asyncio
import numpy as np
from typing import Optional

import weave


# ── 1. Traced Operations ──────────────────────────────────────

@weave.op()
def trace_intent(brain_features: dict) -> dict:
    """Trace brain-to-intent classification."""
    return brain_features

@weave.op()
def trace_style_sample(brain_context: list, style_params: dict) -> dict:
    """Trace CTS posterior sampling."""
    return {"context": brain_context[:4], "style": style_params}

@weave.op()
def trace_generate(intent: str, style: dict, candidates: list) -> dict:
    """Trace LLM response generation."""
    return {"intent": intent, "n_candidates": len(candidates), "candidates": candidates}

@weave.op()
def trace_ab_eval(score_a: float, score_b: float, winner: str,
                  brain_a: dict, brain_b: dict) -> dict:
    """Trace the A/B neural evaluation — brain picks the winner."""
    return {
        "score_a": score_a, "score_b": score_b, "winner": winner,
        "margin": abs(score_a - score_b),
        "brain_a_engagement": brain_a.get("engagement", 0),
        "brain_b_engagement": brain_b.get("engagement", 0),
    }

@weave.op()
def trace_reward(reward: float, adaptation: float, policy_version: int) -> dict:
    """Trace RL reward computation and policy update."""
    return {"reward": reward, "adaptation": adaptation, "policy_version": policy_version}


# ── 2. RL Policy as Weave Model ──────────────────────────────

class NeuroBanditModel(weave.Model):
    """RL policy tracked as versioned Weave Model.

    Every publish creates a new version. Weave UI shows policy evolution.
    Compare v1 (random) vs v20 (adapted to this brain) side-by-side.
    """
    mean: list[float]
    log_var: list[float]
    total_updates: int
    reward_baseline: float
    param_names: list[str]

    @weave.op()
    def predict(self, brain_context: list[float]) -> dict:
        ctx = np.array(brain_context)
        mu = np.array(self.mean)
        std = np.clip(np.exp(0.5 * np.array(self.log_var)), 0.03, 0.5)
        sampled = np.clip(mu + std * np.random.randn(len(mu)), 0, 1)
        return {p: round(float(sampled[i]), 3) for i, p in enumerate(self.param_names)}


# ── 3. Neural Evaluation Scorer ──────────────────────────────

@weave.op()
def neural_scorer(output: dict, target: dict) -> dict:
    """The brain IS the scorer. Did RL's top pick match brain preference?"""
    brain_preferred_first = target.get("winner", "A") == "A"
    return {
        "brain_agreed_with_rl": brain_preferred_first,
        "reward": target.get("reward", 0),
        "confidence_margin": target.get("margin", 0),
        "adaptation_pct": target.get("adaptation", 0),
    }

@weave.op()
def reward_trend_scorer(output: dict, target: dict) -> dict:
    """Track if reward is trending upward — is the system improving?"""
    return {"reward": target.get("reward", 0)}


# ── 4. WeaveTracker — orchestrates everything ────────────────

class WeaveTracker:
    """Manages all Weave integrations for the neural RL loop."""

    def __init__(self, project="neurovoice/brain-voice"):
        self._project = project
        self._episodes = []
        self._initialized = False
        self._publish_every = 10

    def init(self):
        try:
            weave.init(self._project)
            self._initialized = True
            print(f"[WEAVE] Initialized: {self._project}", flush=True)
        except Exception as e:
            print(f"[WEAVE] Init failed: {e}", flush=True)

    def trace_pipeline(self, brain_features, intent, style_params,
                       candidates, score_a, score_b, winner,
                       brain_samples_a, brain_samples_b,
                       reward, adaptation, policy_version):
        """Trace the full brain-to-voice pipeline as nested Weave ops."""
        if not self._initialized:
            return

        try:
            trace_intent(brain_features)
            trace_style_sample(
                list(brain_features.values())[:4] if isinstance(brain_features, dict) else [],
                style_params)
            trace_generate(intent, style_params, candidates)

            avg_a = {"engagement": np.mean([s.get("engagement", 0) for s in brain_samples_a])} if brain_samples_a else {}
            avg_b = {"engagement": np.mean([s.get("engagement", 0) for s in brain_samples_b])} if brain_samples_b else {}
            trace_ab_eval(score_a, score_b, winner, avg_a, avg_b)
            trace_reward(reward, adaptation, policy_version)
        except Exception as e:
            print(f"[WEAVE] Trace error: {e}", flush=True)

    def log_episode(self, brain_context, style_dict, candidates,
                    score_a, score_b, winner, reward, adaptation,
                    policy_version, intent):
        """Store a brain episode for memory + evaluation."""
        if not self._initialized:
            return

        episode = {
            "brain_context": brain_context if isinstance(brain_context, list) else brain_context.tolist(),
            "style": style_dict,
            "candidate_a": candidates[0] if candidates else "",
            "candidate_b": candidates[1] if len(candidates) > 1 else "",
            "score_a": round(score_a, 4),
            "score_b": round(score_b, 4),
            "winner": winner,
            "margin": round(abs(score_a - score_b), 4),
            "reward": round(reward, 4),
            "adaptation": round(adaptation, 1),
            "policy_version": policy_version,
            "intent": intent,
            "timestamp": time.time(),
        }
        self._episodes.append(episode)

        if len(self._episodes) % self._publish_every == 0:
            self._publish_dataset()

    def _publish_dataset(self):
        try:
            dataset = weave.Dataset(name="neural-ab-tests", rows=self._episodes)
            weave.publish(dataset)
            print(f"[WEAVE] Published {len(self._episodes)} episodes", flush=True)
        except Exception as e:
            print(f"[WEAVE] Dataset publish error: {e}", flush=True)

    def publish_policy(self, policy, param_names):
        """Snapshot the RL policy as a versioned Weave Model."""
        if not self._initialized:
            return
        try:
            model = NeuroBanditModel(
                mean=policy.mu.tolist(),
                log_var=policy.log_var.tolist(),
                total_updates=policy._total_updates,
                reward_baseline=float(getattr(policy, '_reward_baseline', 0)),
                param_names=param_names,
            )
            weave.publish(model)
            print(f"[WEAVE] Policy v{policy._total_updates} published", flush=True)
        except Exception as e:
            print(f"[WEAVE] Policy publish error: {e}", flush=True)

    def run_eval(self, policy, param_names):
        """Run brain-as-scorer evaluation against current policy."""
        if not self._initialized or len(self._episodes) < 5:
            return

        try:
            model = NeuroBanditModel(
                mean=policy.mu.tolist(),
                log_var=policy.log_var.tolist(),
                total_updates=policy._total_updates,
                reward_baseline=float(getattr(policy, '_reward_baseline', 0)),
                param_names=param_names,
            )
            dataset = weave.Dataset(name="neural-ab-tests", rows=self._episodes[-20:])
            evaluation = weave.Evaluation(
                name="brain-judges-ai",
                dataset=dataset,
                scorers=[neural_scorer, reward_trend_scorer],
            )
            asyncio.get_event_loop().run_in_executor(
                None, lambda: asyncio.run(evaluation.evaluate(model)))
            print(f"[WEAVE] Eval triggered ({len(self._episodes)} episodes)", flush=True)
        except Exception as e:
            print(f"[WEAVE] Eval error: {e}", flush=True)

    def find_similar_brain_state(self, current_ctx, top_k=3):
        """Episodic memory: find past brain states similar to current."""
        if len(self._episodes) < 5:
            return []

        current = np.array(current_ctx[:12]) if len(current_ctx) >= 12 else np.array(current_ctx)
        scored = []
        for ep in self._episodes:
            past = np.array(ep["brain_context"][:len(current)])
            if len(past) != len(current):
                continue
            sim = float(np.dot(current, past) / (np.linalg.norm(current) * np.linalg.norm(past) + 1e-8))
            scored.append((sim, ep))

        scored.sort(key=lambda x: -x[0])
        return [(s, ep["style"], ep["reward"]) for s, ep in scored[:top_k]]

    @property
    def episode_count(self):
        return len(self._episodes)
