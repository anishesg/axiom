"""BrainStream — Redis pub/sub for real-time brain state across agents.

One EEG source publishes brain state. Multiple agents (voice, rover,
computer use) subscribe and react. Decouples EEG from any single agent.

    publisher = BrainStream.publisher()
    publisher.publish(brain_snapshot)

    subscriber = BrainStream.subscriber()
    async for state in subscriber.listen():
        # react to brain state
"""

import json
import asyncio
from typing import AsyncIterator

try:
    import redis
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

from neuralrl.brain_reward import BrainSnapshot

CHANNEL = "axiom:brain"
RL_CHANNEL = "axiom:rl"
STREAM_KEY = "axiom:brain:stream"


class BrainPublisher:
    """Publishes brain state to Redis for all agents to consume."""

    def __init__(self, host="localhost", port=6379):
        self._r = None
        self._host = host
        self._port = port

    def connect(self):
        if not HAS_REDIS:
            return False
        try:
            self._r = redis.Redis(host=self._host, port=self._port, decode_responses=True)
            self._r.ping()
            return True
        except Exception:
            self._r = None
            return False

    def publish_brain(self, snap: BrainSnapshot):
        if not self._r:
            return
        data = {
            "engagement": round(snap.engagement, 4),
            "focus": round(snap.focus, 4),
            "valence": round(snap.valence, 4),
            "cognitive_load": round(snap.cognitive_load, 4),
            "relaxation": round(snap.relaxation, 4),
            "error_response": round(snap.error_response, 4),
            "jaw_clench": snap.jaw_clench,
            "asymmetry": round(snap.asymmetry, 4),
            "timestamp": snap.timestamp,
        }
        try:
            self._r.publish(CHANNEL, json.dumps(data))
            self._r.xadd(STREAM_KEY, data, maxlen=5000)
        except Exception:
            pass

    def publish_rl_update(self, reward, posteriors, adaptation):
        if not self._r:
            return
        try:
            self._r.publish(RL_CHANNEL, json.dumps({
                "reward": reward,
                "posteriors": posteriors,
                "adaptation": adaptation,
            }))
        except Exception:
            pass


class BrainSubscriber:
    """Subscribes to brain state updates from Redis."""

    def __init__(self, host="localhost", port=6379):
        self._host = host
        self._port = port

    async def listen(self) -> AsyncIterator[dict]:
        if not HAS_REDIS:
            return
        r = aioredis.Redis(host=self._host, port=self._port, decode_responses=True)
        ps = r.pubsub()
        await ps.subscribe(CHANNEL)
        try:
            async for msg in ps.listen():
                if msg["type"] == "message":
                    yield json.loads(msg["data"])
        finally:
            await ps.unsubscribe(CHANNEL)
            await r.aclose()
