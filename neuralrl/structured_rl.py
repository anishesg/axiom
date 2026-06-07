"""StructuredRL — OpenAI gpt-5.5-instant with grounded brain interpretation.

The key insight: the LLM must be CONSTRAINED by brain signals, not just
informed. We do this by:
1. Mapping brain metrics to concrete behavioral instructions (not vibes)
2. Feeding back high-reward past responses as few-shot examples
3. Using strand confidence to weight which dimensions to emphasize
4. Strict structured output so the RL loop always gets clean data
"""

from pydantic import BaseModel, Field
from openai import OpenAI

MODEL = "gpt-5.5-instant"


class BrainInterpretation(BaseModel):
    intent: str = Field(description="One of: agree, disagree, elaborate, question, acknowledge, express_emotion, correct")
    response: str = Field(description="Full natural conversational response. Match length to the question depth.")
    alternative: str = Field(description="A different response with a different angle or tone, also full length.")
    confidence: float = Field(description="0-1 confidence in this interpretation")
    emotion: str = Field(description="One of: neutral, happy, concerned, excited, thoughtful, frustrated")


class BrainInterpreter:
    """Grounded brain-to-speech with episodic memory feedback."""

    def __init__(self, api_key=None, model=MODEL):
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._model = model
        self._high_reward_examples = []
        self._low_reward_examples = []
        self._max_examples = 5

    def record_outcome(self, response_text: str, reward: float, brain_features: dict):
        """Feed back what worked and what didn't — grounds future generation."""
        entry = {"text": response_text, "reward": round(reward, 3),
                 "engagement": round(brain_features.get("engagement", 0), 2),
                 "valence": round(brain_features.get("valence", 0.5), 2)}

        if reward > 0.3:
            self._high_reward_examples.append(entry)
            if len(self._high_reward_examples) > self._max_examples:
                self._high_reward_examples.pop(0)
        elif reward < -0.1:
            self._low_reward_examples.append(entry)
            if len(self._low_reward_examples) > self._max_examples:
                self._low_reward_examples.pop(0)

    def interpret(self, brain_features: dict, conversation_context: str,
                  style_params: dict = None, strand_confidences: dict = None) -> BrainInterpretation:

        # Build concrete instructions from brain metrics, not vague descriptions
        instructions = self._brain_to_instructions(brain_features, style_params, strand_confidences)
        grounding = self._build_grounding()

        system = (
            "You are the voice of Axiom, a real-time brain-computer interface being demoed at "
            "WeaveHacks 4 (Weights & Biases hackathon, judges from OpenAI, Google DeepMind, Cursor, Apple).\n\n"
            "A user wears a Muse S EEG headband. Their brain signals (engagement, valence, focus, "
            "cognitive load) are read in real-time. You speak FOR them.\n\n"
            "The system uses Contextual Thompson Sampling to learn communication style from brain reward, "
            "a Braid of 4 dimension-specialized strands (content, emotion, style, context) each learning "
            "independently, OpenAI gpt-5.5-instant with structured outputs, ElevenLabs v3 TTS with "
            "emotion tags from EEG, Redis Streams for brain pub/sub, and W&B Weave with brain-as-scorer.\n\n"
            "IF asked about the project/technology/how it works: explain impressively but accessibly. "
            "The novel insight is the brain IS the reward function — not human labels. The system learns "
            "WHY you like a response (content vs emotion vs style) not just that you liked it. "
            "Each strand specializes in one dimension and they fuse via confidence-weighted gating.\n\n"
            "IF the question is unrelated: just respond naturally as a warm, thoughtful person.\n\n"
            f"{instructions}\n"
            f"{grounding}\n"
            "RULES:\n"
            "- Respond to what was JUST said. Be specific and substantive.\n"
            "- Deep question = detailed answer. Simple question = brief.\n"
            "- Sound confident, knowledgeable, human.\n"
            "- The 'alternative' must take a genuinely different angle."
        )

        resp = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Conversation:\n{conversation_context}\n\nRespond:"},
            ],
            response_format=BrainInterpretation,
            temperature=0.7,
        )
        return resp.choices[0].message.parsed

    def _brain_to_instructions(self, brain: dict, style: dict = None,
                                strand_conf: dict = None) -> str:
        """Convert brain metrics to CONCRETE behavioral instructions."""
        eng = brain.get("engagement", 0.5)
        val = brain.get("valence", 0.5)
        focus = brain.get("focus", 0.5)
        cog = brain.get("cognitive_load", 0.5)
        intent = brain.get("intent", "acknowledge")

        parts = []

        # Intent → concrete instruction
        intent_map = {
            "agree": "You AGREE with what was said. Express genuine agreement and build on their point.",
            "disagree": "You DISAGREE. Push back respectfully but clearly. Say why.",
            "elaborate": "You want to EXPLAIN your thinking in detail. Go deep.",
            "question": "You want to ASK a follow-up question. Be curious and specific.",
            "acknowledge": "You want to briefly acknowledge what was said, then continue naturally.",
            "express_emotion": "You want to share how you FEEL about this. Be emotionally honest.",
            "correct": "You want to CORRECT something. Be clear about what's wrong and why.",
        }
        parts.append(intent_map.get(intent, "Respond naturally."))

        # Engagement → response depth
        if eng > 0.7:
            parts.append("You're highly engaged — give a substantive, detailed response.")
        elif eng < 0.3:
            parts.append("You're not very engaged — keep it brief and to the point.")

        # Valence → emotional tone
        if val > 0.65:
            parts.append("You're feeling positive — let warmth come through.")
        elif val < 0.35:
            parts.append("You're feeling negative or uneasy — don't fake positivity.")

        # Cognitive load → complexity
        if cog > 0.6:
            parts.append("You're thinking hard — your response should reflect careful thought.")

        # Style params → concrete adjustments
        if style:
            if style.get("formality", 0.5) > 0.7:
                parts.append("Use formal language.")
            elif style.get("formality", 0.5) < 0.3:
                parts.append("Be very casual — slang is fine.")
            if style.get("humor", 0.5) > 0.6:
                parts.append("Include a touch of humor if appropriate.")
            if style.get("empathy", 0.5) > 0.7:
                parts.append("Show deep empathy — acknowledge their feelings.")

        # Strand confidence → which dimensions to trust
        if strand_conf:
            most_confident = max(strand_conf, key=strand_conf.get)
            least_confident = min(strand_conf, key=strand_conf.get)
            if strand_conf[most_confident] > 0.8:
                parts.append(f"Prioritize {most_confident} — the system is most confident about that dimension.")

        return "\n".join(parts)

    def _build_grounding(self) -> str:
        """Build few-shot grounding from past high/low reward responses."""
        if not self._high_reward_examples and not self._low_reward_examples:
            return ""

        parts = ["GROUNDING (what this brain responded to in the past):"]

        if self._high_reward_examples:
            parts.append("Brain LIKED these responses (high engagement/positive valence):")
            for ex in self._high_reward_examples[-3:]:
                parts.append(f'  ✓ "{ex["text"]}" (reward={ex["reward"]})')

        if self._low_reward_examples:
            parts.append("Brain DISLIKED these (low engagement/negative valence):")
            for ex in self._low_reward_examples[-2:]:
                parts.append(f'  ✗ "{ex["text"]}" (reward={ex["reward"]})')

        return "\n".join(parts)
