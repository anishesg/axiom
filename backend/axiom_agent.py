"""Axiom Agent — the decision-making core.

Consumes BrainState + screen context, decides actions, learns from outcomes.

Architecture:
  - Fast path: threshold-based reactive decisions (jaw clench → click, gaze dwell → select)
  - Slow path: LLM-powered context understanding (draft replies, predict next app)
  - RL path: online policy that adjusts confidence thresholds from experience

The agent maintains an action log and runs self-improvement every N actions.
"""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from brain_state import BrainState


@dataclass
class ScreenContext:
    """What's currently on screen (from accessibility API or screenshot OCR)."""
    active_app: str = ""
    focused_element: str = ""
    element_type: str = ""  # button, text_field, link, app_icon, notification
    visible_text: str = ""
    available_actions: list = field(default_factory=list)
    gaze_region: str = ""  # "top-left", "center", "dock", etc.
    gaze_target: str = ""  # specific element under gaze


@dataclass
class AgentAction:
    """An action the agent decides to take."""
    action_type: str  # click, scroll, switch_app, open, close, draft_text, undo, ghost_actions, none
    target: str = ""
    confidence: float = 0.0
    reason: str = ""
    timestamp: float = 0.0
    brain_state: dict = field(default_factory=dict)
    screen_context: dict = field(default_factory=dict)
    outcome: Optional[str] = None  # "confirmed", "undone", "ignored"


class AxiomAgent:
    """The Axiom decision engine."""

    def __init__(self):
        # Confidence thresholds (these get adjusted by the RL loop)
        self.thresholds = {
            "intent_ring_engagement": 0.65,    # engagement needed to start intent ring
            "intent_ring_duration": 1.5,       # seconds of sustained engagement to confirm
            "auto_action_confidence": 0.80,    # confidence to take action without clench
            "scroll_engagement_fast": 0.3,     # below this = fast scroll (skimming)
            "scroll_engagement_slow": 0.7,     # above this = slow scroll (reading)
            "context_switch_threshold": 0.5,   # brain pattern strength to suggest app switch
            "error_detection_threshold": 0.7,  # asymmetry shift to trigger undo
            "sleep_alpha_threshold": 0.8,      # sustained relaxation to enter sleep
        }

        # State tracking
        self._action_log: list[AgentAction] = []
        self._intent_ring_start: float = 0.0
        self._intent_ring_target: str = ""
        self._intent_ring_active: bool = False
        self._current_mode: str = "passive"  # passive, tracking, acting, sleeping
        self._actions_since_improve: int = 0
        self._scroll_velocity: float = 0.0

        # Learning metrics
        self.total_actions = 0
        self.correct_actions = 0
        self.undone_actions = 0
        self.accuracy_history: list[float] = []

    @property
    def accuracy(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.correct_actions / self.total_actions

    def decide(self, brain: BrainState, ctx: ScreenContext, timestamp: float = None) -> AgentAction:
        """Main decision loop. Returns the action to take (or none)."""
        now = timestamp if timestamp is not None else time.time()

        # --- SLEEP MODE ---
        if self._current_mode == "sleeping":
            if brain.relaxation < 0.5 and brain.engagement > 0.3:
                self._current_mode = "passive"
                return AgentAction("wake", reason="User re-engaged", confidence=0.9, timestamp=now)
            return AgentAction("none", reason="Sleeping", timestamp=now)

        if brain.relaxation > self.thresholds["sleep_alpha_threshold"] and brain.engagement < 0.1:
            self._current_mode = "sleeping"
            return AgentAction("sleep", reason="Sustained relaxation, no engagement", confidence=0.9, timestamp=now)

        # --- ERROR DETECTION (auto-undo) ---
        if brain.error_response > self.thresholds["error_detection_threshold"]:
            if self._action_log and now - self._action_log[-1].timestamp < 2.0:
                last = self._action_log[-1]
                last.outcome = "undone"
                self.undone_actions += 1
                return self._log_action(AgentAction("undo", target=last.target,
                                                     reason=f"Error response detected after {last.action_type}",
                                                     confidence=brain.error_response, timestamp=now), brain, ctx)

        # --- CONTEXT SWITCH PREDICTION ---
        if brain.context_switch > self.thresholds["context_switch_threshold"]:
            return self._log_action(AgentAction("suggest_apps", reason="Context switch pattern detected",
                                                 confidence=brain.context_switch, timestamp=now), brain, ctx)

        # --- INTENT RING (engagement-based selection) ---
        if brain.engagement > self.thresholds["intent_ring_engagement"] and ctx.gaze_target:
            if not self._intent_ring_active or ctx.gaze_target != self._intent_ring_target:
                self._intent_ring_active = True
                self._intent_ring_start = now
                self._intent_ring_target = ctx.gaze_target
                return AgentAction("intent_ring_start", target=ctx.gaze_target,
                                   confidence=brain.engagement, timestamp=now,
                                   reason="Engagement + gaze dwell → intent ring started")

            # Ring filling...
            elapsed = now - self._intent_ring_start
            progress = elapsed / self.thresholds["intent_ring_duration"]

            if progress >= 1.0 and brain.engagement > self.thresholds["intent_ring_engagement"]:
                self._intent_ring_active = False
                return self._log_action(AgentAction("click", target=self._intent_ring_target,
                                                     reason=f"Intent ring completed ({elapsed:.1f}s sustained engagement)",
                                                     confidence=min(0.95, brain.engagement), timestamp=now), brain, ctx)
            else:
                return AgentAction("intent_ring_progress", target=self._intent_ring_target,
                                   confidence=progress, timestamp=now,
                                   reason=f"Intent ring {progress*100:.0f}%")
        else:
            if self._intent_ring_active:
                self._intent_ring_active = False
                return AgentAction("intent_ring_cancel", target=self._intent_ring_target,
                                   confidence=0, timestamp=now, reason="Engagement dropped, ring cancelled")

        # --- SCROLL (passive) ---
        if ctx.gaze_region in ("bottom-third", "bottom-edge"):
            speed = 1.0 if brain.engagement < self.thresholds["scroll_engagement_fast"] else 0.3
            if ctx.gaze_region == "bottom-edge":
                speed *= 2.0
            return AgentAction("scroll_down", confidence=0.7, timestamp=now,
                               reason=f"Gaze at bottom, scroll speed={speed:.1f}")

        # --- GHOST ACTIONS (show after action) ---
        if self._action_log and now - self._action_log[-1].timestamp < 3.0:
            last = self._action_log[-1]
            if last.action_type in ("click", "open") and brain.engagement < 0.4:
                return AgentAction("ghost_actions", target=last.target,
                                   confidence=0.6, timestamp=now,
                                   reason="Engagement dropping after action, show options")

        # --- PASSIVE ---
        self._current_mode = "passive"
        return AgentAction("none", timestamp=now, reason="Passive — watching")

    def _log_action(self, action: AgentAction, brain: BrainState, ctx: ScreenContext) -> AgentAction:
        action.brain_state = brain.to_dict()
        action.screen_context = asdict(ctx)
        self._action_log.append(action)
        self.total_actions += 1
        self._actions_since_improve += 1

        # Track accuracy
        if action.outcome != "undone":
            self.correct_actions += 1

        self.accuracy_history.append(self.accuracy)
        return action

    def record_outcome(self, outcome: str):
        """Record whether the last action was confirmed, undone, or ignored."""
        if self._action_log:
            self._action_log[-1].outcome = outcome
            if outcome == "undone":
                self.undone_actions += 1
                self.correct_actions = max(0, self.correct_actions - 1)

    def should_improve(self) -> bool:
        return self._actions_since_improve >= 20 or (
            self._actions_since_improve >= 5 and self.total_actions > 0 and self.accuracy < 0.6
        )

    def get_improvement_context(self) -> str:
        """Get recent action log as context for the LLM self-improvement step."""
        recent = self._action_log[-20:]
        lines = []
        for a in recent:
            lines.append(f"  [{a.action_type}] target={a.target} conf={a.confidence:.2f} "
                         f"outcome={a.outcome} reason={a.reason}")
        return (
            f"Current accuracy: {self.accuracy:.1%} ({self.correct_actions}/{self.total_actions})\n"
            f"Current thresholds: {json.dumps(self.thresholds, indent=2)}\n"
            f"Recent actions:\n" + "\n".join(lines)
        )

    def apply_improvements(self, updates: dict[str, float]):
        """Apply threshold updates from the self-improvement engine."""
        for key, val in updates.items():
            if key in self.thresholds:
                old = self.thresholds[key]
                self.thresholds[key] = val
                print(f"  [IMPROVE] {key}: {old:.3f} → {val:.3f}")
        self._actions_since_improve = 0

    def get_decision_log(self, n: int = 10) -> list[dict]:
        """Get the last N actions as dicts for the dashboard."""
        return [asdict(a) for a in self._action_log[-n:]]
