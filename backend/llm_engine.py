"""Axiom LLM Engine — self-improvement via AWS Bedrock Claude.

Two tiers:
  - Fast (Haiku): per-action context understanding when RL confidence is low
  - Strategy (Sonnet): periodic threshold/policy review every 5 min or 20 actions

Uses the Converse API with tool_use for structured action suggestions.
"""

import json
import boto3
from dataclasses import dataclass

HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
SONNET_MODEL = "us.anthropic.claude-sonnet-4-6-20250514-v1:0"

ACTION_TOOL = {
    "toolSpec": {
        "name": "suggest_action",
        "description": "Suggest an OS action based on brain state and screen context",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["click", "scroll_down", "scroll_up", "switch_app",
                                 "open", "close", "draft_text", "undo", "none"],
                    },
                    "target": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"},
                    "draft_text": {"type": "string"},
                },
                "required": ["action", "confidence", "reasoning"],
            }
        },
    }
}

THRESHOLD_TOOL = {
    "toolSpec": {
        "name": "update_thresholds",
        "description": "Update agent decision thresholds based on performance analysis",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "thresholds": {
                        "type": "object",
                        "description": "Updated threshold values",
                    },
                    "observations": {
                        "type": "string",
                        "description": "What patterns you noticed in the action log",
                    },
                    "recommendations": {
                        "type": "string",
                        "description": "High-level strategy recommendations",
                    },
                },
                "required": ["thresholds", "observations"],
            }
        },
    }
}


@dataclass
class LLMAction:
    action: str = "none"
    target: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    draft_text: str = ""


class AxiomLLM:
    def __init__(self, region: str = "us-west-2"):
        self._client = boto3.client("bedrock-runtime", region_name=region)
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def suggest_action(self, brain_state: dict, screen_context: dict,
                       recent_actions: list[dict] = None) -> LLMAction:
        """Fast path: ask Haiku for an action suggestion."""
        context_parts = [
            f"Brain state: {json.dumps(brain_state, default=str)}",
            f"Screen: app={screen_context.get('active_app', '?')}, "
            f"gaze_target={screen_context.get('gaze_target', '?')}, "
            f"element={screen_context.get('element_type', '?')}, "
            f"text={screen_context.get('visible_text', '')[:200]}",
        ]
        if recent_actions:
            context_parts.append(
                f"Recent actions: {json.dumps(recent_actions[-5:], default=str)}"
            )

        messages = [{
            "role": "user",
            "content": [{"text": (
                "You are Axiom, a brain-computer interface agent. "
                "Based on the user's brain state (from EEG) and what's on screen, "
                "suggest the best action. The user controls the computer through "
                "thought (engagement/focus), gaze (where they look), and jaw clenches (click).\n\n"
                + "\n".join(context_parts)
            )}],
        }]

        try:
            response = self._client.converse(
                modelId=HAIKU_MODEL,
                messages=messages,
                toolConfig={"tools": [ACTION_TOOL]},
                inferenceConfig={"maxTokens": 256, "temperature": 0.2},
            )
            self._track_usage(response)
            return self._parse_action(response)
        except Exception as e:
            return LLMAction(action="none", reasoning=f"LLM error: {e}")

    def review_strategy(self, action_log: str, current_thresholds: dict) -> dict:
        """Slow path: ask Sonnet to review performance and adjust thresholds."""
        messages = [{
            "role": "user",
            "content": [{"text": (
                "You are the self-improvement engine for Axiom, a brain-computer interface. "
                "Review the recent action log and current thresholds. "
                "Analyze patterns: which actions succeeded/failed, what thresholds should change, "
                "and what strategy adjustments would improve accuracy.\n\n"
                f"Current accuracy and log:\n{action_log}\n\n"
                "Adjust thresholds carefully — small changes (±0.05) are better than large swings. "
                "Only change thresholds where you see clear evidence of under/over-triggering."
            )}],
        }]

        try:
            response = self._client.converse(
                modelId=SONNET_MODEL,
                messages=messages,
                toolConfig={"tools": [THRESHOLD_TOOL]},
                inferenceConfig={"maxTokens": 512, "temperature": 0.1},
            )
            self._track_usage(response)
            return self._parse_thresholds(response)
        except Exception as e:
            return {"error": str(e)}

    def draft_reply(self, visible_text: str, context: str = "") -> str:
        """Generate a reply draft based on what the user is reading."""
        messages = [{
            "role": "user",
            "content": [{"text": (
                "The user is reading this on screen and wants to reply. "
                "Draft a brief, natural reply. Keep it concise (1-2 sentences max).\n\n"
                f"Message: {visible_text}\n"
                f"Context: {context}" if context else f"Message: {visible_text}"
            )}],
        }]

        try:
            response = self._client.converse(
                modelId=HAIKU_MODEL,
                messages=messages,
                inferenceConfig={"maxTokens": 128, "temperature": 0.5},
            )
            self._track_usage(response)
            content = response.get("output", {}).get("message", {}).get("content", [])
            for block in content:
                if "text" in block:
                    return block["text"]
            return ""
        except Exception as e:
            return f"(draft error: {e})"

    def _parse_action(self, response: dict) -> LLMAction:
        content = response.get("output", {}).get("message", {}).get("content", [])
        for block in content:
            if "toolUse" in block:
                inp = block["toolUse"].get("input", {})
                return LLMAction(
                    action=inp.get("action", "none"),
                    target=inp.get("target", ""),
                    confidence=inp.get("confidence", 0.0),
                    reasoning=inp.get("reasoning", ""),
                    draft_text=inp.get("draft_text", ""),
                )
        return LLMAction(action="none", reasoning="No tool use in response")

    def _parse_thresholds(self, response: dict) -> dict:
        content = response.get("output", {}).get("message", {}).get("content", [])
        for block in content:
            if "toolUse" in block:
                return block["toolUse"].get("input", {})
        return {}

    def _track_usage(self, response: dict):
        self._call_count += 1
        usage = response.get("usage", {})
        self._total_input_tokens += usage.get("inputTokens", 0)
        self._total_output_tokens += usage.get("outputTokens", 0)

    @property
    def stats(self) -> dict:
        return {
            "calls": self._call_count,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
        }
