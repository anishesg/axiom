"""BrainOrchestrator — EEG-driven multi-agent operating system.

The brain is the kernel. EEG metrics map to OS-level scheduling decisions:
  - Engagement  → CPU allocation (high = keep current agent, low = context switch)
  - Focus       → Process lock (high = suppress interrupts, low = allow switching)
  - Cognitive Load → Memory pressure (high = reduce agents, simplify output)
  - Valence     → Reward signal (positive shift = reinforce, negative = penalize)
  - Error (ErrP) → Hardware interrupt (triggers correction handler)

Architecture: Reactive Blackboard over Redis pub/sub.
  Brain state IS the blackboard. Agents subscribe and react.
  The scheduler decides who's active based on neural signals.

    orchestrator = BrainOrchestrator()
    orchestrator.register(VoiceAgent())
    orchestrator.register(RoverAgent())
    await orchestrator.run(brain_stream)

Novel contribution: First consumer-EEG multi-agent OS.
All prior work (NEURO-LOOP, ErrP-RL) is single-agent.
"""

import asyncio
import time
import json
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

try:
    import redis as redis_lib
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


@dataclass
class NeuralState:
    """Brain state snapshot for orchestration decisions."""
    engagement: float = 0.0
    focus: float = 0.0
    valence: float = 0.5
    cognitive_load: float = 0.0
    relaxation: float = 0.0
    error_detected: bool = False
    jaw_clench: bool = False
    timestamp: float = 0.0


@dataclass
class AgentRecord:
    """Tracks an agent's activation history and learned lessons."""
    agent: 'BrainAgent' = None
    activation_score: float = 0.0
    total_activations: int = 0
    total_reward: float = 0.0
    avg_engagement_during: float = 0.0
    lessons: list = field(default_factory=list)
    last_activated: float = 0.0
    is_active: bool = False


class BrainAgent(ABC):
    """Base class for any brain-orchestrated agent.

    Implement these 4 methods to plug any agent into the brain OS.
    """

    @abstractmethod
    async def on_activate(self, state: NeuralState, context: dict) -> None:
        """Brain scheduler activates this agent.

        Args:
            state: Current brain state
            context: Dict with 'lessons' (list of past learnings),
                     'reason' (why this agent was activated)
        """

    @abstractmethod
    async def on_deactivate(self) -> None:
        """Brain scheduler deactivates this agent (another takes over)."""

    @abstractmethod
    async def on_correct(self, state: NeuralState, error_context: dict) -> None:
        """Brain error response (ErrP) detected — fix what you just did.

        Args:
            state: Brain state at time of error
            error_context: Dict with 'last_action', 'lessons'
        """

    @abstractmethod
    def get_priority(self, state: NeuralState) -> float:
        """Return 0-1 priority given current brain state.

        The scheduler calls this on all agents and activates the highest.
        Use brain metrics to determine relevance.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier."""


# ── Scheduling Policies ────────────────────────────────────────

ENGAGEMENT_LOCK_THRESHOLD = 0.65
ENGAGEMENT_SWITCH_THRESHOLD = 0.25
FOCUS_LOCK_THRESHOLD = 0.7
COGNITIVE_OVERLOAD_THRESHOLD = 0.8
SWITCH_COOLDOWN = 3.0
MAX_LESSONS_PER_AGENT = 50


class BrainOrchestrator:
    """Brain-as-OS multi-agent scheduler.

    Manages agent lifecycle based on continuous neural signals.
    Publishes orchestration events to Redis for observability.
    Maintains per-agent lessons for self-improvement.
    """

    def __init__(self, redis_host="localhost", redis_port=6379):
        self._agents: dict[str, AgentRecord] = {}
        self._active_agent: Optional[str] = None
        self._state_history: deque = deque(maxlen=100)
        self._last_switch_time: float = 0.0
        self._total_decisions: int = 0
        self._callbacks: dict[str, list] = {
            "activate": [], "deactivate": [], "correct": [],
            "reward": [], "switch": [],
        }

        self._redis = None
        if HAS_REDIS:
            try:
                self._redis = redis_lib.Redis(
                    host=redis_host, port=redis_port, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    def register(self, agent: BrainAgent) -> None:
        """Register an agent with the orchestrator."""
        self._agents[agent.name] = AgentRecord(agent=agent)

    def on(self, event: str, callback) -> None:
        """Register a callback for orchestration events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def process_state(self, state: NeuralState) -> dict:
        """Process one brain state update. Returns orchestration decision.

        Call this at ~10Hz from your brain loop.
        """
        self._state_history.append(state)
        self._total_decisions += 1

        decision = {"action": "none", "agent": self._active_agent}

        # Priority 1: Error correction (hardware interrupt)
        if state.error_detected and self._active_agent:
            await self._handle_error(state)
            decision = {"action": "correct", "agent": self._active_agent}

        # Priority 2: Cognitive overload — reduce complexity
        elif state.cognitive_load > COGNITIVE_OVERLOAD_THRESHOLD:
            decision = {"action": "simplify", "agent": self._active_agent,
                       "cognitive_load": state.cognitive_load}

        # Priority 3: Engagement-driven scheduling
        elif state.engagement < ENGAGEMENT_SWITCH_THRESHOLD:
            if time.time() - self._last_switch_time > SWITCH_COOLDOWN:
                new_agent = await self._find_best_agent(state)
                if new_agent and new_agent != self._active_agent:
                    await self._switch_agent(new_agent, state, "low_engagement")
                    decision = {"action": "switch", "agent": new_agent,
                               "reason": "low_engagement"}

        # Priority 4: Focus lock — don't interrupt
        elif state.focus > FOCUS_LOCK_THRESHOLD and self._active_agent:
            decision = {"action": "locked", "agent": self._active_agent}

        # Priority 5: Normal scheduling — best priority wins
        else:
            best = await self._find_best_agent(state)
            if best and best != self._active_agent:
                if time.time() - self._last_switch_time > SWITCH_COOLDOWN:
                    await self._switch_agent(best, state, "priority")
                    decision = {"action": "switch", "agent": best, "reason": "priority"}

        # Track reward via valence shifts
        self._track_reward(state)

        # Publish to Redis
        self._publish_decision(decision, state)

        return decision

    async def _find_best_agent(self, state: NeuralState) -> Optional[str]:
        """Find the highest-priority agent for current brain state."""
        if not self._agents:
            return None
        priorities = {}
        for name, record in self._agents.items():
            priorities[name] = record.agent.get_priority(state)
        return max(priorities, key=priorities.get)

    async def _switch_agent(self, new_name: str, state: NeuralState, reason: str):
        """Deactivate current agent, activate new one."""
        if self._active_agent and self._active_agent in self._agents:
            old_record = self._agents[self._active_agent]
            old_record.is_active = False
            try:
                await old_record.agent.on_deactivate()
            except Exception as e:
                print(f"[ORCH] Deactivate error ({self._active_agent}): {e}", flush=True)
            for cb in self._callbacks["deactivate"]:
                await cb(self._active_agent)

        self._active_agent = new_name
        self._last_switch_time = time.time()
        record = self._agents[new_name]
        record.is_active = True
        record.total_activations += 1
        record.last_activated = time.time()

        context = {
            "reason": reason,
            "lessons": record.lessons[-10:],
            "brain_state": asdict(state),
        }
        try:
            await record.agent.on_activate(state, context)
        except Exception as e:
            print(f"[ORCH] Activate error ({new_name}): {e}", flush=True)

        for cb in self._callbacks["activate"]:
            await cb(new_name, state, reason)
        for cb in self._callbacks["switch"]:
            await cb(self._active_agent, new_name, reason)

    async def _handle_error(self, state: NeuralState):
        """Brain detected an error — trigger correction on active agent."""
        if not self._active_agent or self._active_agent not in self._agents:
            return
        record = self._agents[self._active_agent]
        error_context = {
            "lessons": record.lessons[-5:],
            "state_at_error": asdict(state),
        }
        try:
            await record.agent.on_correct(state, error_context)
        except Exception as e:
            print(f"[ORCH] Correct error ({self._active_agent}): {e}", flush=True)

        record.lessons.append({
            "type": "error_correction",
            "timestamp": time.time(),
            "engagement_at_error": state.engagement,
            "valence_at_error": state.valence,
        })
        if len(record.lessons) > MAX_LESSONS_PER_AGENT:
            record.lessons = record.lessons[-MAX_LESSONS_PER_AGENT:]

        for cb in self._callbacks["correct"]:
            await cb(self._active_agent, state)

    def _track_reward(self, state: NeuralState):
        """Track valence shifts as implicit reward for the active agent."""
        if not self._active_agent or len(self._state_history) < 2:
            return
        prev = self._state_history[-2]
        valence_delta = state.valence - prev.valence

        record = self._agents[self._active_agent]

        # Update running engagement average
        n = max(1, record.total_activations)
        record.avg_engagement_during = (
            record.avg_engagement_during * (n - 1) + state.engagement) / n

        # Significant valence shift = learning signal
        if abs(valence_delta) > 0.05:
            reward = 1.0 if valence_delta > 0 else -0.5
            record.total_reward += reward

            if valence_delta < -0.1:
                record.lessons.append({
                    "type": "negative_valence_shift",
                    "timestamp": time.time(),
                    "magnitude": round(valence_delta, 3),
                })
                if len(record.lessons) > MAX_LESSONS_PER_AGENT:
                    record.lessons = record.lessons[-MAX_LESSONS_PER_AGENT:]

            for cb in self._callbacks["reward"]:
                asyncio.ensure_future(cb(self._active_agent, reward, valence_delta))

    def _publish_decision(self, decision: dict, state: NeuralState):
        """Publish orchestration decision to Redis for observability."""
        if not self._redis:
            return
        try:
            self._redis.publish("axiom:orchestrator", json.dumps({
                **decision,
                "engagement": round(state.engagement, 3),
                "focus": round(state.focus, 3),
                "cognitive_load": round(state.cognitive_load, 3),
                "valence": round(state.valence, 3),
                "tick": self._total_decisions,
            }))
        except Exception:
            pass

    # ── Introspection ──────────────────────────────────────────

    def get_status(self) -> dict:
        """Return orchestrator state for dashboard display."""
        agents = {}
        for name, record in self._agents.items():
            agents[name] = {
                "is_active": record.is_active,
                "activations": record.total_activations,
                "total_reward": round(record.total_reward, 2),
                "avg_engagement": round(record.avg_engagement_during, 3),
                "lessons_count": len(record.lessons),
                "last_activated": record.last_activated,
            }
        return {
            "active_agent": self._active_agent,
            "total_decisions": self._total_decisions,
            "agents": agents,
        }

    def get_lessons(self, agent_name: str) -> list:
        """Get learned lessons for a specific agent."""
        if agent_name in self._agents:
            return self._agents[agent_name].lessons
        return []

    @property
    def active_agent(self) -> Optional[str]:
        return self._active_agent

    @property
    def agent_count(self) -> int:
        return len(self._agents)
