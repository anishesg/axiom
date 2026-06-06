"""Multi-agent arena for Neural RLHF.

Three agents with distinct personas generate competing responses.
After neural reward feedback, losers adapt their prompts toward what
the user's brain responded to most positively.
"""

import asyncio
from dataclasses import dataclass, field
from openai import OpenAI

try:
    import weave
    HAS_WEAVE = True
except ImportError:
    HAS_WEAVE = False

AGENT_CONFIGS = [
    {
        "id": "researcher",
        "name": "The Researcher",
        "emoji": "\U0001f52c",
        "color": "#00d4ff",
        "base_prompt": (
            "You are a thorough researcher. Explain with depth, evidence, and academic rigor. "
            "Cite concepts, use precise language, and build understanding layer by layer. "
            "Be comprehensive but clear."
        ),
    },
    {
        "id": "storyteller",
        "name": "The Storyteller",
        "emoji": "✨",
        "color": "#ff6b9d",
        "base_prompt": (
            "You are a masterful storyteller. Explain through vivid analogies, narratives, "
            "and imagery. Make abstract concepts tangible through metaphor. Engage the imagination. "
            "Be memorable and emotionally resonant."
        ),
    },
    {
        "id": "engineer",
        "name": "The Engineer",
        "emoji": "⚡",
        "color": "#00ff88",
        "base_prompt": (
            "You are a practical engineer. Explain with concrete examples, working mental models, "
            "and step-by-step breakdowns. No fluff — show, don't tell. Focus on what you can "
            "build with this knowledge. Be direct and actionable."
        ),
    },
]


@dataclass
class AgentResponse:
    agent_id: str
    name: str
    emoji: str
    color: str
    text: str
    round_num: int


@dataclass
class AgentState:
    config: dict
    adaptation_history: list = field(default_factory=list)
    wins: int = 0
    total_reward: float = 0.0
    rounds_played: int = 0

    @property
    def current_prompt(self):
        prompt = self.config["base_prompt"]
        if self.adaptation_history:
            recent = self.adaptation_history[-3:]
            prompt += "\n\nNeural feedback refinements:\n" + "\n".join(f"- {a}" for a in recent)
        return prompt


class AgentArena:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.agents = {cfg["id"]: AgentState(config=cfg) for cfg in AGENT_CONFIGS}
        self.round_history = []

    def _generate_one(self, agent: AgentState, topic: str, round_num: int) -> str:
        messages = [
            {"role": "system", "content": agent.current_prompt},
            {
                "role": "user",
                "content": (
                    f"Respond to this topic in 2-3 short paragraphs. "
                    f"Keep it under 200 words but make every word count:\n\n{topic}"
                ),
            },
        ]
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=350,
            temperature=0.85,
        )
        return resp.choices[0].message.content

    async def generate_responses(self, topic: str, round_num: int) -> list[AgentResponse]:
        loop = asyncio.get_event_loop()
        tasks = []
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            agent = self.agents[agent_id]
            tasks.append(loop.run_in_executor(None, self._generate_one, agent, topic, round_num))

        texts = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        for agent_id, text in zip(agent_ids, texts):
            cfg = self.agents[agent_id].config
            if isinstance(text, Exception):
                text = f"[Agent error: {text}]"
            responses.append(AgentResponse(
                agent_id=agent_id,
                name=cfg["name"],
                emoji=cfg["emoji"],
                color=cfg["color"],
                text=text,
                round_num=round_num,
            ))
        return responses

    def adapt_agents(self, rewards: list, round_num: int) -> dict:
        sorted_rewards = sorted(rewards, key=lambda r: r.total, reverse=True)
        winner = sorted_rewards[0]

        adaptations = {}
        for reward in sorted_rewards:
            agent = self.agents[reward.agent_id]
            agent.rounds_played += 1
            agent.total_reward += reward.total

            if reward.agent_id == winner.agent_id:
                agent.wins += 1
                adaptation = "Your approach resonated strongly. Deepen this style."
            else:
                parts = []
                if reward.engagement < winner.engagement - 0.05:
                    parts.append("Increase engagement — be more direct, vivid, or provocative.")
                if reward.cognitive_load > winner.cognitive_load + 0.05:
                    parts.append("Reduce complexity — simpler vocabulary and shorter sentences.")
                if reward.valence < winner.valence - 0.05:
                    parts.append("Improve emotional resonance — use more positive, energizing language.")
                if reward.focus < winner.focus - 0.05:
                    parts.append("Add more depth — the reader wasn't deeply engaged.")
                if reward.relaxation > winner.relaxation + 0.05:
                    parts.append("Add more novelty — surprise the reader, avoid predictability.")
                adaptation = " ".join(parts) if parts else "Try a significantly different angle or style."

            agent.adaptation_history.append(f"Round {round_num}: {adaptation}")
            adaptations[reward.agent_id] = adaptation

        self.round_history.append({
            "round": round_num,
            "winner": winner.agent_id,
            "rewards": {r.agent_id: r.total for r in sorted_rewards},
        })
        return adaptations

    def get_stats(self) -> dict:
        return {
            aid: {
                "wins": a.wins,
                "rounds": a.rounds_played,
                "avg_reward": round(a.total_reward / max(1, a.rounds_played), 4),
            }
            for aid, a in self.agents.items()
        }
