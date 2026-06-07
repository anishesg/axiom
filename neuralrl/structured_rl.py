"""StructuredRL — OpenAI Structured Outputs for guaranteed-parseable RL signals.

Uses Pydantic models + OpenAI's structured output mode to ensure every
brain-to-intent interpretation is type-safe. No regex, no parse failures.
The RL loop gets clean, typed data every single time.

    interpreter = BrainInterpreter()
    result = interpreter.interpret(brain_features, conversation_context)
    # result.intent, result.phrase, result.confidence, result.alternatives
"""

from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI


class BrainInterpretation(BaseModel):
    """Structured output from brain signal interpretation."""
    intent: str = Field(description="Detected intent: agree, disagree, elaborate, question, acknowledge, express_emotion, correct")
    phrase: str = Field(description="The best response phrase to speak")
    confidence: float = Field(description="Confidence in this interpretation, 0-1")
    emotion: str = Field(description="Detected emotion: neutral, happy, concerned, excited, thoughtful, frustrated")
    alternatives: list[str] = Field(description="3 alternative response phrases")
    style_feedback: str = Field(description="Brief feedback on what communication style traits this brain pattern suggests: formal/casual, brief/verbose, warm/cool")


class StyleUpdate(BaseModel):
    """Structured output for RL style parameter updates."""
    formality: float = Field(description="0=very casual, 1=very formal")
    enthusiasm: float = Field(description="0=subdued, 1=energetic")
    verbosity: float = Field(description="0=brief, 1=detailed")
    empathy: float = Field(description="0=matter-of-fact, 1=deeply empathetic")
    humor: float = Field(description="0=serious, 1=playful")
    assertiveness: float = Field(description="0=tentative, 1=confident")
    expressiveness: float = Field(description="0=restrained, 1=expressive")
    warmth: float = Field(description="0=cool/professional, 1=warm/personal")
    reasoning: str = Field(description="Brief explanation of why these values based on the brain signals")


class BrainInterpreter:
    """Uses OpenAI Structured Outputs to interpret brain signals into actions."""

    def __init__(self, api_key=None, model="gpt-4o-mini"):
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._model = model

    def interpret(self, brain_features: dict, conversation_context: str,
                  style_params: dict = None) -> BrainInterpretation:
        brain_str = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in brain_features.items())
        style_str = ""
        if style_params:
            style_str = f"\nCurrent style: {', '.join(f'{k}={v:.2f}' for k,v in style_params.items())}"

        resp = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content":
                    "You interpret EEG brain signals to generate speech for a non-verbal user. "
                    "Based on the brain state and conversation context, determine what the user "
                    "wants to say. Generate natural, human responses. Keep the main phrase 1-2 sentences."},
                {"role": "user", "content":
                    f"Brain state: {brain_str}\n"
                    f"Conversation:\n{conversation_context}"
                    f"{style_str}\n\n"
                    f"What does this person want to say?"},
            ],
            response_format=BrainInterpretation,
        )
        return resp.choices[0].message.parsed

    def suggest_style(self, brain_features: dict, reward_history: list[float],
                      current_style: dict) -> StyleUpdate:
        brain_str = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in brain_features.items())
        recent_rewards = reward_history[-10:] if reward_history else []

        resp = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content":
                    "You are an RL style optimizer. Based on brain signals and reward history, "
                    "suggest optimal communication style parameters for this user. "
                    "Higher rewards mean the user's brain responded positively to that style."},
                {"role": "user", "content":
                    f"Brain state: {brain_str}\n"
                    f"Current style: {current_style}\n"
                    f"Recent rewards: {recent_rewards}\n"
                    f"Suggest optimal style parameters:"},
            ],
            response_format=StyleUpdate,
        )
        return resp.choices[0].message.parsed
