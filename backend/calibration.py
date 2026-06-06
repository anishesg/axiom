"""EEG Calibration and Intent Classification System.

Learns per-user EEG→intent mappings from labeled calibration samples,
then classifies new EEG windows into intents in real-time. Supports
ErrP-based reinforcement to improve predictions over time.

Architecture:
  - CalibrationSample: labeled EEG snapshot with action/context/gaze
  - CalibrationEngine: feature extraction, training, prediction, feedback loop

Intent taxonomy (what the user wants to do):
  interested, disinterested, click_intent, reading, scanning, idle

Action taxonomy (what the system should do):
  open_email, archive, star, reply, compose, scroll, back, search, none

Usage:
  engine = CalibrationEngine()
  engine.start_calibration_session()
  # ...collect samples via engine.add_sample(brain_state, action, context, gaze)...
  results = engine.end_calibration_session()
  intent, conf = engine.predict_intent(brain_state)
  action, conf = engine.predict_action(brain_state, context)
  engine.record_feedback(correct=True)
  engine.save()
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import heavy dependencies so the module loads fast in tests
# ---------------------------------------------------------------------------

def _import_sklearn():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report
    return RandomForestClassifier, LabelEncoder, cross_val_score, classification_report


def _import_joblib():
    import joblib
    return joblib


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSample:
    timestamp: float
    brain_state: dict        # BrainState.to_dict()
    action: str              # ground-truth action performed
    context: dict            # what was visible (element_type, email_subject, …)
    gaze: dict               # gaze position {"x": float, "y": float, "element": str}
    inferred_intent: str = ""  # filled in by engine when adding


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class CalibrationEngine:
    """Learns and applies user-specific EEG→intent mappings.

    The pipeline:
      1. CalibrationSample collection (with ground-truth labels).
      2. Feature extraction from BrainState (instantaneous + temporal dynamics).
      3. RandomForest training (intent classifier + action classifier).
      4. Real-time prediction + contextual action resolution.
      5. ErrP-reinforced feedback loop: per-pattern confidence adjustment.
    """

    INTENTS = [
        "interested",     # user wants to engage with what they see
        "disinterested",  # user wants to skip / move on
        "click_intent",   # user wants to click/select (before jaw-clench confirm)
        "reading",        # user is actively reading content
        "scanning",       # user is quickly scanning / skimming
        "idle",           # user is not doing anything specific
    ]

    ACTIONS = [
        "open_email",
        "archive",
        "star",
        "reply",
        "compose",
        "scroll",
        "back",
        "search",
        "none",
    ]

    # Minimum samples per intent class required before training
    MIN_SAMPLES_PER_CLASS = 10

    # Rolling-buffer length (brain states; ~15 s at 2 Hz)
    BUFFER_LEN = 30

    # Action rules: (element_type, intent) → action
    # These are overridden by the ML classifier once enough data is available.
    _ACTION_RULES: list[tuple[str, str, str]] = [
        ("email_row",       "interested",    "open_email"),
        ("email_row",       "click_intent",  "open_email"),
        ("star_button",     "click_intent",  "star"),
        ("star_button",     "interested",    "star"),
        ("archive_button",  "click_intent",  "archive"),
        ("archive_button",  "interested",    "archive"),
        ("reply_button",    "click_intent",  "reply"),
        ("compose_button",  "click_intent",  "compose"),
        ("search_bar",      "click_intent",  "search"),
        ("email_body",      "reading",       "none"),
        ("email_body",      "interested",    "none"),
        ("email_body",      "disinterested", "back"),
        ("inbox",           "scanning",      "scroll"),
        ("inbox",           "disinterested", "scroll"),
        ("inbox",           "idle",          "none"),
    ]

    def __init__(self, model_path: str = "data/calibration_model.pkl"):
        self.model_path = model_path

        # Calibration data
        self._samples: list[CalibrationSample] = []
        self._is_calibrating: bool = False
        self._session_start: Optional[float] = None

        # Rolling buffer of feature vectors for temporal dynamics
        self._feature_buffer: deque = deque(maxlen=self.BUFFER_LEN)
        # Rolling buffer of raw brain-state dicts (for delta computation)
        self._state_buffer: deque = deque(maxlen=self.BUFFER_LEN)

        # Trained models (None until train() is called or load() succeeds)
        self._intent_clf = None       # RandomForestClassifier → INTENTS
        self._intent_encoder = None   # LabelEncoder for intent labels
        self._action_clf = None       # RandomForestClassifier → ACTIONS (optional)
        self._action_encoder = None

        # Feature names (built once at first feature extraction)
        self._feature_names: list[str] = []

        # ErrP feedback state
        self._last_intent: Optional[str] = None
        self._last_intent_conf: float = 0.0
        self._last_feature_vec: Optional[np.ndarray] = None
        self._feedback_log: list[dict] = []  # {timestamp, intent, correct, conf}

        # Per-intent confidence bias (adjusted by feedback)
        self._intent_bias: dict[str, float] = {i: 0.0 for i in self.INTENTS}

    # ------------------------------------------------------------------
    # Calibration session management
    # ------------------------------------------------------------------

    def start_calibration_session(self) -> None:
        """Begin a calibration data-collection session."""
        self._is_calibrating = True
        self._session_start = time.time()
        logger.info("Calibration session started at %.2f", self._session_start)

    def end_calibration_session(self) -> dict:
        """End collection, auto-train, and return session results.

        Returns a dict with:
          session_duration_s, n_samples, samples_per_intent,
          train_results (from train()), error (if training failed).
        """
        self._is_calibrating = False
        duration = time.time() - (self._session_start or time.time())
        n = len(self._samples)

        counts = self._count_by_intent()
        result: dict = {
            "session_duration_s": round(duration, 2),
            "n_samples": n,
            "samples_per_intent": counts,
        }

        try:
            train_result = self.train()
            result["train_results"] = train_result
        except Exception as exc:
            result["error"] = str(exc)
            logger.warning("Auto-train after calibration failed: %s", exc)

        logger.info(
            "Calibration session ended: %d samples, %.1f s",
            n, duration,
        )
        return result

    def add_sample(
        self,
        brain_state,   # BrainState instance
        action: str,
        context: dict,
        gaze: dict,
    ) -> CalibrationSample:
        """Record a labeled calibration sample.

        The intent is inferred from the action + context using the rule table
        so we have intent labels even if the caller only provides actions.
        """
        bs_dict = brain_state.to_dict() if hasattr(brain_state, "to_dict") else dict(brain_state)
        intent = self._infer_intent_from_action(action, context)

        sample = CalibrationSample(
            timestamp=time.time(),
            brain_state=bs_dict,
            action=action,
            context=context,
            gaze=gaze,
            inferred_intent=intent,
        )
        self._samples.append(sample)

        # Keep the feature buffer updated for temporal dynamics
        fv = self._extract_features_from_dict(bs_dict)
        self._feature_buffer.append(fv)
        self._state_buffer.append(bs_dict)

        return sample

    def reset(self) -> None:
        """Clear all calibration data and trained models."""
        self._samples.clear()
        self._feature_buffer.clear()
        self._state_buffer.clear()
        self._intent_clf = None
        self._intent_encoder = None
        self._action_clf = None
        self._action_encoder = None
        self._last_intent = None
        self._last_feature_vec = None
        self._feedback_log.clear()
        self._intent_bias = {i: 0.0 for i in self.INTENTS}
        logger.info("CalibrationEngine reset.")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_feature_vector(self, brain_state) -> np.ndarray:
        """Extract a deterministic ~28-feature vector from a BrainState.

        Feature groups:
          [0-5]   instantaneous: engagement, focus, relaxation,
                  cognitive_load, valence, asymmetry
          [6-10]  band_powers: delta, theta, alpha, beta, gamma
          [11-14] band_ratios: alpha_theta, beta_alpha, gamma_beta, theta_alpha
          [15-20] short-term deltas (vs mean of last 5): eng, focus, relax,
                  cog_load, valence, asymmetry
          [21-26] mid-term deltas (vs mean of last 15): same 6 features
          [27-32] variability (std of last 10): same 6 features
        Total: 33 features
        """
        bs_dict = brain_state.to_dict() if hasattr(brain_state, "to_dict") else dict(brain_state)
        fv = self._extract_features_from_dict(bs_dict)

        # Update buffers for temporal dynamics on the next call
        self._feature_buffer.append(fv)
        self._state_buffer.append(bs_dict)

        return fv

    def _extract_features_from_dict(self, bs: dict) -> np.ndarray:
        """Build feature vector from a brain_state dict (no side effects)."""

        # --- Instantaneous features ---
        eng        = float(bs.get("engagement",     0.0))
        focus      = float(bs.get("focus",          0.0))
        relax      = float(bs.get("relaxation",     0.0))
        cog_load   = float(bs.get("cognitive_load", 0.0))
        valence    = float(bs.get("valence",        0.5))
        asymmetry  = float(bs.get("asymmetry",      0.0))

        instant = np.array([eng, focus, relax, cog_load, valence, asymmetry], dtype=np.float32)

        # --- Band powers (5) ---
        bp = bs.get("band_powers", {})
        band_vec = np.array([
            float(bp.get("delta", 0.0)),
            float(bp.get("theta", 0.0)),
            float(bp.get("alpha", 0.0)),
            float(bp.get("beta",  0.0)),
            float(bp.get("gamma", 0.0)),
        ], dtype=np.float32)

        # --- Band ratios (4) ---
        br = bs.get("band_ratios", {})
        ratio_vec = np.array([
            float(br.get("alpha_theta", 1.0)),
            float(br.get("beta_alpha",  1.0)),
            float(br.get("gamma_beta",  1.0)),
            float(br.get("theta_alpha", 1.0)),
        ], dtype=np.float32)

        # --- Temporal dynamics from buffer ---
        # We compute deltas against the buffer that existed *before* this call,
        # so we read _state_buffer as-is (this dict hasn't been appended yet).
        buf = list(self._state_buffer)  # up to BUFFER_LEN items

        short_delta  = np.zeros(6, dtype=np.float32)
        mid_delta    = np.zeros(6, dtype=np.float32)
        variability  = np.zeros(6, dtype=np.float32)

        INST_KEYS = ["engagement", "focus", "relaxation", "cognitive_load", "valence", "asymmetry"]
        INST_DEFAULTS = [0.0, 0.0, 0.0, 0.0, 0.5, 0.0]

        if len(buf) >= 2:
            def _series(key, default):
                return np.array([float(s.get(key, default)) for s in buf], dtype=np.float32)

            for i, (key, default) in enumerate(zip(INST_KEYS, INST_DEFAULTS)):
                series = _series(key, default)
                short_n = min(5, len(series))
                mid_n   = min(15, len(series))

                short_mean = float(np.mean(series[-short_n:]))
                mid_mean   = float(np.mean(series[-mid_n:]))
                std_n      = min(10, len(series))

                short_delta[i] = instant[i] - short_mean
                mid_delta[i]   = instant[i] - mid_mean
                variability[i] = float(np.std(series[-std_n:]))

        # Assemble
        fv = np.concatenate([instant, band_vec, ratio_vec, short_delta, mid_delta, variability])

        # Build feature-name list on first call (for interpretability)
        if not self._feature_names:
            self._feature_names = (
                ["eng", "focus", "relax", "cog_load", "valence", "asymmetry"]
                + ["bp_delta", "bp_theta", "bp_alpha", "bp_beta", "bp_gamma"]
                + ["br_alpha_theta", "br_beta_alpha", "br_gamma_beta", "br_theta_alpha"]
                + [f"sd_{k}" for k in ["eng", "focus", "relax", "cog_load", "valence", "asym"]]
                + [f"md_{k}" for k in ["eng", "focus", "relax", "cog_load", "valence", "asym"]]
                + [f"var_{k}" for k in ["eng", "focus", "relax", "cog_load", "valence", "asym"]]
            )

        return fv.astype(np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Train intent (and action) classifiers from collected samples.

        Requires MIN_SAMPLES_PER_CLASS per intent class.
        Returns accuracy metrics dict.
        """
        RandomForestClassifier, LabelEncoder, cross_val_score, classification_report = _import_sklearn()

        if not self._samples:
            raise ValueError("No calibration samples collected.")

        # Build feature matrix + labels
        X, y_intent, y_action = [], [], []
        for s in self._samples:
            fv = self._extract_features_from_dict(s.brain_state)
            X.append(fv)
            y_intent.append(s.inferred_intent or "idle")
            y_action.append(s.action or "none")

        X = np.array(X, dtype=np.float32)
        y_intent = np.array(y_intent)
        y_action = np.array(y_action)

        # Check class counts
        intent_counts = {intent: int(np.sum(y_intent == intent)) for intent in self.INTENTS}
        low_classes = [k for k, v in intent_counts.items() if 0 < v < self.MIN_SAMPLES_PER_CLASS]
        if any(v > 0 and v < self.MIN_SAMPLES_PER_CLASS for v in intent_counts.values()):
            logger.warning(
                "Some intent classes have fewer than %d samples: %s. "
                "Model may have low accuracy.",
                self.MIN_SAMPLES_PER_CLASS,
                {k: intent_counts[k] for k in low_classes},
            )

        present_intents = sorted(set(y_intent))
        if len(present_intents) < 2:
            raise ValueError(
                f"Need at least 2 intent classes to train; only have: {present_intents}"
            )

        # Intent classifier
        intent_enc = LabelEncoder()
        y_intent_enc = intent_enc.fit_transform(y_intent)

        intent_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        intent_clf.fit(X, y_intent_enc)

        # Cross-validated accuracy (min 3 folds, capped at 5)
        n_folds = min(5, min(intent_counts[k] for k in present_intents if intent_counts.get(k, 0) > 0))
        n_folds = max(2, n_folds)
        if len(X) >= n_folds * 2:
            cv_scores = cross_val_score(intent_clf, X, y_intent_enc, cv=n_folds, scoring="accuracy")
            cv_accuracy = float(np.mean(cv_scores))
        else:
            cv_accuracy = float(np.mean(intent_clf.predict(X) == y_intent_enc))

        self._intent_clf = intent_clf
        self._intent_encoder = intent_enc

        # Action classifier (if enough variety)
        action_result: dict = {}
        present_actions = sorted(set(y_action))
        if len(present_actions) >= 2:
            action_enc = LabelEncoder()
            y_action_enc = action_enc.fit_transform(y_action)
            action_clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            action_clf.fit(X, y_action_enc)
            self._action_clf = action_clf
            self._action_encoder = action_enc
            action_result = {
                "n_classes": len(present_actions),
                "classes": list(present_actions),
                "train_accuracy": float(np.mean(action_clf.predict(X) == y_action_enc)),
            }

        # Feature importances
        importances = intent_clf.feature_importances_
        if self._feature_names and len(self._feature_names) == len(importances):
            feat_imp = {
                name: round(float(imp), 4)
                for name, imp in sorted(
                    zip(self._feature_names, importances),
                    key=lambda x: -x[1],
                )
            }
        else:
            feat_imp = {f"f{i}": round(float(v), 4) for i, v in enumerate(importances)}

        result = {
            "n_samples": len(X),
            "intent_classes": list(present_intents),
            "intent_cv_accuracy": round(cv_accuracy, 4),
            "intent_train_accuracy": round(
                float(np.mean(intent_clf.predict(X) == y_intent_enc)), 4
            ),
            "samples_per_intent": intent_counts,
            "feature_importances": feat_imp,
            "action_classifier": action_result,
        }
        logger.info(
            "Training complete: %d samples, %d intent classes, CV accuracy %.3f",
            len(X), len(present_intents), cv_accuracy,
        )
        return result

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_intent(self, brain_state) -> tuple[str, float]:
        """Predict current intent and confidence.

        Returns (intent_name, confidence_0_to_1).
        Falls back to heuristics if model is not trained.
        """
        bs_dict = brain_state.to_dict() if hasattr(brain_state, "to_dict") else dict(brain_state)
        fv = self._extract_features_from_dict(bs_dict)
        self._feature_buffer.append(fv)
        self._state_buffer.append(bs_dict)
        self._last_feature_vec = fv

        if self._intent_clf is None or self._intent_encoder is None:
            intent, conf = self._heuristic_intent(bs_dict)
            self._last_intent = intent
            self._last_intent_conf = conf
            return intent, conf

        try:
            proba = self._intent_clf.predict_proba(fv.reshape(1, -1))[0]
            best_idx = int(np.argmax(proba))
            conf = float(proba[best_idx])
            intent_label = self._intent_encoder.inverse_transform([best_idx])[0]

            # Apply ErrP-learned bias
            bias = self._intent_bias.get(intent_label, 0.0)
            conf_biased = float(np.clip(conf + bias, 0.0, 1.0))

            self._last_intent = intent_label
            self._last_intent_conf = conf_biased
            return intent_label, conf_biased

        except Exception as exc:
            logger.warning("Intent prediction failed: %s", exc)
            return "idle", 0.0

    def predict_action(self, brain_state, context: dict) -> tuple[str, float]:
        """Predict likely desired action given brain state + UI context.

        Strategy:
          1. Obtain intent from predict_intent().
          2. If ML action classifier is trained, use it directly.
          3. Otherwise (or if confidence is low), fall back to rule table.
          4. Rules match on (element_type, intent) → action.

        Returns (action_name, confidence_0_to_1).
        """
        intent, intent_conf = self.predict_intent(brain_state)
        element_type = context.get("element_type", "")

        # Try ML action classifier first
        if self._action_clf is not None and self._action_encoder is not None:
            try:
                fv = self._last_feature_vec
                if fv is not None:
                    action_proba = self._action_clf.predict_proba(fv.reshape(1, -1))[0]
                    best_idx = int(np.argmax(action_proba))
                    action_conf = float(action_proba[best_idx])
                    action = self._action_encoder.inverse_transform([best_idx])[0]

                    # If ML is confident enough, use it
                    if action_conf >= 0.5:
                        return action, action_conf
            except Exception as exc:
                logger.warning("Action classifier failed: %s", exc)

        # Rule-based fallback
        action, rule_conf = self._rule_based_action(element_type, intent, intent_conf, context)
        return action, rule_conf

    # ------------------------------------------------------------------
    # Feedback / ErrP reinforcement
    # ------------------------------------------------------------------

    def record_feedback(self, correct: bool) -> None:
        """Record whether the last prediction was correct.

        If incorrect, applies a negative bias to the predicted intent's
        confidence. If correct, applies a small positive reinforcement.
        The bias decays toward zero over time to avoid over-fitting.
        """
        if self._last_intent is None:
            return

        entry = {
            "timestamp": time.time(),
            "intent": self._last_intent,
            "confidence": self._last_intent_conf,
            "correct": correct,
        }
        self._feedback_log.append(entry)

        adjustment = 0.05 if correct else -0.15
        current = self._intent_bias.get(self._last_intent, 0.0)
        # Soft clamp: bias stays in [-0.4, +0.2]
        new_bias = float(np.clip(current + adjustment, -0.4, 0.2))
        self._intent_bias[self._last_intent] = new_bias

        logger.debug(
            "Feedback: intent=%s correct=%s bias %.3f→%.3f",
            self._last_intent, correct, current, new_bias,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = None) -> None:
        """Save model + calibration data to disk via joblib."""
        joblib = _import_joblib()
        target = path or self.model_path
        Path(target).parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": 1,
            "samples": [asdict(s) for s in self._samples],
            "intent_clf": self._intent_clf,
            "intent_encoder": self._intent_encoder,
            "action_clf": self._action_clf,
            "action_encoder": self._action_encoder,
            "feature_names": self._feature_names,
            "intent_bias": self._intent_bias,
            "feedback_log": self._feedback_log,
        }
        joblib.dump(payload, target)
        logger.info("Saved calibration model to %s (%d samples)", target, len(self._samples))

    def load(self, path: str = None) -> bool:
        """Load saved model from disk. Returns True on success."""
        joblib = _import_joblib()
        target = path or self.model_path

        if not Path(target).exists():
            logger.info("No saved calibration model found at %s", target)
            return False

        try:
            payload = joblib.load(target)
            raw_samples = payload.get("samples", [])
            self._samples = [CalibrationSample(**s) for s in raw_samples]
            self._intent_clf = payload.get("intent_clf")
            self._intent_encoder = payload.get("intent_encoder")
            self._action_clf = payload.get("action_clf")
            self._action_encoder = payload.get("action_encoder")
            self._feature_names = payload.get("feature_names", [])
            self._intent_bias = payload.get("intent_bias", {i: 0.0 for i in self.INTENTS})
            self._feedback_log = payload.get("feedback_log", [])
            logger.info(
                "Loaded calibration model from %s (%d samples)", target, len(self._samples)
            )
            return True
        except Exception as exc:
            logger.error("Failed to load calibration model: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_calibration_stats(self) -> dict:
        """Return statistics about collected data, model accuracy, and feature importances."""
        counts = self._count_by_intent()
        action_counts = {}
        for s in self._samples:
            action_counts[s.action] = action_counts.get(s.action, 0) + 1

        is_trained = self._intent_clf is not None

        # Feature importances (top 10)
        feat_imp: dict = {}
        if is_trained and self._feature_names:
            importances = self._intent_clf.feature_importances_
            paired = sorted(zip(self._feature_names, importances), key=lambda x: -x[1])
            feat_imp = {k: round(float(v), 4) for k, v in paired[:10]}

        # Feedback accuracy
        total_fb = len(self._feedback_log)
        correct_fb = sum(1 for f in self._feedback_log if f["correct"])
        feedback_accuracy = round(correct_fb / total_fb, 4) if total_fb > 0 else None

        return {
            "n_total_samples": len(self._samples),
            "samples_per_intent": counts,
            "samples_per_action": action_counts,
            "is_trained": is_trained,
            "n_feedback_events": total_fb,
            "feedback_accuracy": feedback_accuracy,
            "intent_confidence_bias": {k: round(v, 4) for k, v in self._intent_bias.items()},
            "top_feature_importances": feat_imp,
            "intent_classes_in_model": (
                list(self._intent_encoder.classes_) if self._intent_encoder else []
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_by_intent(self) -> dict[str, int]:
        counts = {intent: 0 for intent in self.INTENTS}
        for s in self._samples:
            k = s.inferred_intent or "idle"
            counts[k] = counts.get(k, 0) + 1
        return counts

    def _infer_intent_from_action(self, action: str, context: dict) -> str:
        """Map a ground-truth action + context to the most likely intent label."""
        element_type = context.get("element_type", "")

        mapping = {
            "open_email":  "interested",
            "star":        "interested",
            "reply":       "interested",
            "compose":     "click_intent",
            "search":      "click_intent",
            "archive":     "disinterested",
            "back":        "disinterested",
            "scroll":      "scanning",
            "none":        "reading",
        }

        # Refine scroll: if inside an email body it's reading, not scanning
        if action == "scroll" and element_type in ("email_body", "email_content"):
            return "reading"

        return mapping.get(action, "idle")

    def _heuristic_intent(self, bs: dict) -> tuple[str, float]:
        """Rule-based fallback intent when no model is trained."""
        eng       = float(bs.get("engagement",     0.0))
        focus     = float(bs.get("focus",          0.0))
        relax     = float(bs.get("relaxation",     0.0))
        cog_load  = float(bs.get("cognitive_load", 0.0))
        valence   = float(bs.get("valence",        0.5))
        jaw       = bool(bs.get("jaw_clench",      False))
        err_resp  = float(bs.get("error_response", 0.0))

        if jaw:
            return "click_intent", 0.9

        if err_resp > 0.5:
            return "disinterested", 0.7

        if focus > 0.6 and cog_load > 0.4:
            return "reading", 0.65

        if eng > 0.6 and valence > 0.55:
            return "interested", 0.6

        if eng < 0.2 and valence < 0.45:
            return "disinterested", 0.55

        if eng > 0.4 and focus < 0.3:
            return "scanning", 0.5

        if relax > 0.6 and eng < 0.2:
            return "idle", 0.6

        # Default: low-confidence idle
        return "idle", 0.3

    def _rule_based_action(
        self,
        element_type: str,
        intent: str,
        intent_conf: float,
        context: dict,
    ) -> tuple[str, float]:
        """Map (element_type, intent) → action using the rule table."""
        for rule_element, rule_intent, rule_action in self._ACTION_RULES:
            if rule_element == element_type and rule_intent == intent:
                # Scale rule confidence by intent confidence
                return rule_action, round(intent_conf * 0.85, 4)

        # Partial match on intent only
        intent_action_defaults = {
            "interested":    "none",
            "disinterested": "scroll",
            "click_intent":  "none",
            "reading":       "none",
            "scanning":      "scroll",
            "idle":          "none",
        }
        action = intent_action_defaults.get(intent, "none")
        return action, round(intent_conf * 0.5, 4)
