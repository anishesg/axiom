#!/usr/bin/env python3
"""Neural RLHF Server — Your Brain as the Reward Model.

Streams EEG from Muse S (or simulation), runs multi-agent arena,
computes neural rewards from brain state during reading, and
adapts agents based on what your brain responds to.

WebSocket API on ws://127.0.0.1:8080
"""

import asyncio
import json
import time
import sys
import os
import numpy as np

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    HAS_BRAINFLOW = True
except ImportError:
    HAS_BRAINFLOW = False

import websockets
from brain_state import BrainStateEngine
from neural_reward import NeuralRewardComputer
from agent_arena import AgentArena

HAS_WEAVE = False
try:
    import weave
    if os.environ.get("WANDB_API_KEY"):
        weave.init("neural-rlhf")
        HAS_WEAVE = True
except Exception:
    pass

try:
    import redis as redis_lib
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


READING_DURATION = 12
CALIBRATION_DURATION = 20
EEG_SR = 256

TOPICS = [
    "Explain how neural networks learn, in a way that would blow someone's mind",
    "What would cities look like if we redesigned them from scratch with AI?",
    "How could brain-computer interfaces revolutionize education?",
    "What's the most underrated breakthrough in science right now?",
    "If you could build any app that doesn't exist yet, what would change the world?",
]


class EEGSource:
    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
        self.board = None
        self.eeg_channels = None
        self.ring = np.zeros((4, EEG_SR * 10))
        self.ring_pos = 0
        self.total_samples = 0
        self._sim_phase = 0.0
        self._sim_bias = {"engagement": 0.5, "valence": 0.5}

    def connect(self):
        if self.sim_mode:
            print("Simulation mode — no Muse S required", flush=True)
            return
        if not HAS_BRAINFLOW:
            raise RuntimeError("BrainFlow not installed — use --sim for simulation mode")
        params = BrainFlowInputParams()
        params.timeout = 5
        self.board = BoardShim(BoardIds.MUSE_S_BOARD.value, params)
        print("Scanning for Muse S...", flush=True)
        self.board.prepare_session()
        print("Connected!", flush=True)
        self.board.start_stream(num_samples=450000)
        self.eeg_channels = BoardShim.get_eeg_channels(BoardIds.MUSE_S_BOARD.value)[:4]
        print("Streaming EEG at 256 Hz", flush=True)

    def set_sim_bias(self, engagement=0.5, valence=0.5):
        self._sim_bias = {"engagement": engagement, "valence": valence}

    def poll(self):
        if self.sim_mode:
            self._generate_sim()
        else:
            raw = self.board.get_board_data(num_samples=64)
            if raw.shape[1] > 0:
                self._push(raw[self.eeg_channels, :])

    def get_window(self, seconds=1.0):
        n = int(EEG_SR * seconds)
        cap = self.ring.shape[1]
        n = min(n, self.total_samples, cap)
        if n <= 0:
            return np.zeros((4, 0))
        end = self.ring_pos % cap
        if n <= end:
            return self.ring[:, end - n:end].copy()
        return np.concatenate([self.ring[:, cap - (n - end):], self.ring[:, :end]], axis=1)

    def _push(self, eeg):
        cap = self.ring.shape[1]
        for i in range(eeg.shape[1]):
            self.ring[:, self.ring_pos % cap] = eeg[:, i]
            self.ring_pos += 1
        self.total_samples += eeg.shape[1]

    def _generate_sim(self):
        n = 32
        t = np.arange(n) / EEG_SR + self._sim_phase
        self._sim_phase += n / EEG_SR
        eng = self._sim_bias["engagement"]
        val = self._sim_bias["valence"]

        eeg = np.zeros((4, n))
        for ch in range(4):
            phase_offset = ch * 0.5
            alpha_amp = 15 * (1 - eng * 0.5) + np.random.randn() * 1.5
            eeg[ch] += alpha_amp * np.sin(2 * np.pi * 10 * t + phase_offset)
            beta_amp = 8 * (0.3 + eng * 0.7) + np.random.randn() * 1.0
            eeg[ch] += beta_amp * np.sin(2 * np.pi * 22 * t + phase_offset * 0.7)
            theta_amp = 10 * (0.5 + (1 - eng) * 0.3) + np.random.randn() * 1.0
            eeg[ch] += theta_amp * np.sin(2 * np.pi * 6 * t + phase_offset * 1.2)
            eeg[ch] += 18 * np.sin(2 * np.pi * 2 * t + phase_offset * 0.3)
            eeg[ch] += np.random.randn(n) * 4

        val_shift = (val - 0.5) * 6
        eeg[1] += val_shift * np.sin(2 * np.pi * 10 * t)
        eeg[2] -= val_shift * np.sin(2 * np.pi * 10 * t)

        self._push(eeg)

    def cleanup(self):
        if self.board:
            try:
                self.board.stop_stream()
                self.board.release_session()
            except Exception:
                pass


class NeuralRLHFServer:
    def __init__(self, sim_mode=False):
        self.eeg = EEGSource(sim_mode=sim_mode)
        self.brain_engine = BrainStateEngine()
        self.reward_computer = NeuralRewardComputer()
        self.arena = AgentArena()
        self.clients: set = set()

        self.phase = "idle"
        self.current_round = 0
        self.max_rounds = 5
        self.current_topic = ""
        self.current_responses = []
        self.reading_target_idx = -1
        self.reading_snapshots: dict[str, list[dict]] = {}
        self.round_history: list[dict] = []
        self.reading_start_time = 0.0
        self.latest_brain: dict = {}

        self.redis_client = None
        if HAS_REDIS:
            try:
                rc = redis_lib.Redis(host="localhost", port=6379, decode_responses=True)
                rc.ping()
                self.redis_client = rc
                print("Redis connected", flush=True)
            except Exception:
                print("Redis not available — running without", flush=True)

    async def broadcast(self, data: dict):
        msg = json.dumps(data)
        dead = set()
        for ws in list(self.clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        for d in dead:
            self.clients.discard(d)

        if self.redis_client and data.get("type") in ("brain", "rewards", "round_complete"):
            try:
                self.redis_client.xadd(
                    f"neural:{data['type']}", {"data": json.dumps(data)}, maxlen=10000
                )
            except Exception:
                pass

    async def handle_ws(self, ws):
        self.clients.add(ws)
        print(f"Client connected ({len(self.clients)})", flush=True)
        try:
            from agent_arena import AGENT_CONFIGS
            await ws.send(json.dumps({
                "type": "init",
                "phase": self.phase,
                "topics": TOPICS,
                "agents": [
                    {"id": c["id"], "name": c["name"], "emoji": c["emoji"], "color": c["color"]}
                    for c in AGENT_CONFIGS
                ],
                "round": self.current_round,
                "max_rounds": self.max_rounds,
                "sim_mode": self.eeg.sim_mode,
                "history": self.round_history,
            }))
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    await self.handle_command(data)
                except json.JSONDecodeError:
                    pass
        finally:
            self.clients.discard(ws)
            print(f"Client disconnected ({len(self.clients)})", flush=True)

    async def handle_command(self, data: dict):
        cmd = data.get("command")
        if cmd == "start_calibration":
            asyncio.create_task(self.run_calibration())
        elif cmd == "skip_calibration":
            self.reward_computer.set_baseline([])
            self.phase = "ready"
            await self.broadcast({"type": "phase", "phase": "ready"})
        elif cmd == "start_round":
            topic = data.get("topic", TOPICS[self.current_round % len(TOPICS)])
            asyncio.create_task(self.run_round(topic))
        elif cmd == "next_response":
            asyncio.create_task(self.advance_reading())

    async def brain_loop(self):
        while True:
            self.eeg.poll()
            window = self.eeg.get_window(1.0)
            if window.shape[1] >= EEG_SR:
                state = self.brain_engine.process(window, timestamp=time.time())
                brain_data = {
                    "type": "brain",
                    "engagement": round(state.engagement, 4),
                    "focus": round(state.focus, 4),
                    "valence": round(state.valence, 4),
                    "cognitive_load": round(state.cognitive_load, 4),
                    "relaxation": round(state.relaxation, 4),
                    "signal_quality": [round(q, 2) for q in state.signal_quality],
                    "timestamp": time.time(),
                }
                self.latest_brain = brain_data
                await self.broadcast(brain_data)

                if self.phase == "reading" and 0 <= self.reading_target_idx < len(self.current_responses):
                    agent_id = self.current_responses[self.reading_target_idx].agent_id
                    if agent_id in self.reading_snapshots:
                        self.reading_snapshots[agent_id].append({
                            "engagement": state.engagement,
                            "focus": state.focus,
                            "valence": state.valence,
                            "cognitive_load": state.cognitive_load,
                            "relaxation": state.relaxation,
                        })

            await asyncio.sleep(0.5)

    async def run_calibration(self):
        self.phase = "calibrating"
        snapshots = []
        await self.broadcast({"type": "phase", "phase": "calibrating", "duration": CALIBRATION_DURATION})

        start = time.time()
        while time.time() - start < CALIBRATION_DURATION:
            if self.latest_brain:
                snapshots.append({
                    k: self.latest_brain[k]
                    for k in ("engagement", "focus", "valence", "cognitive_load", "relaxation")
                })
            elapsed = time.time() - start
            await self.broadcast({
                "type": "calibration_progress",
                "progress": min(1.0, elapsed / CALIBRATION_DURATION),
                "seconds_remaining": max(0, int(CALIBRATION_DURATION - elapsed)),
            })
            await asyncio.sleep(1.0)

        self.reward_computer.set_baseline(snapshots)
        self.phase = "ready"
        await self.broadcast({"type": "phase", "phase": "ready"})
        print(f"Calibration complete ({len(snapshots)} samples)", flush=True)

    async def run_round(self, topic: str):
        self.current_round += 1
        self.current_topic = topic
        self.reading_snapshots = {}

        self.phase = "generating"
        await self.broadcast({
            "type": "phase", "phase": "generating",
            "round": self.current_round, "topic": topic,
        })

        try:
            if HAS_WEAVE:
                with weave.attributes({"round": self.current_round, "topic": topic}):
                    self.current_responses = await self.arena.generate_responses(topic, self.current_round)
            else:
                self.current_responses = await self.arena.generate_responses(topic, self.current_round)
        except Exception as e:
            print(f"Generation error: {e}", flush=True)
            await self.broadcast({"type": "error", "message": str(e)})
            self.phase = "ready"
            self.current_round -= 1
            return

        await self.broadcast({
            "type": "responses",
            "responses": [
                {"id": r.agent_id, "name": r.name, "emoji": r.emoji, "color": r.color, "text": r.text}
                for r in self.current_responses
            ],
            "round": self.current_round,
        })

        self.phase = "reading"
        await self.start_reading(0)

    async def start_reading(self, idx: int):
        if idx >= len(self.current_responses):
            await self.finish_round()
            return

        resp = self.current_responses[idx]
        self.reading_target_idx = idx
        self.reading_snapshots[resp.agent_id] = []
        self.reading_start_time = time.time()

        if self.eeg.sim_mode:
            biases = {
                "researcher": (0.4 + self.current_round * 0.04, 0.5),
                "storyteller": (0.6 + self.current_round * 0.03, 0.65 + self.current_round * 0.02),
                "engineer": (0.5 + self.current_round * 0.035, 0.55),
            }
            eng, val = biases.get(resp.agent_id, (0.5, 0.5))
            eng = np.clip(eng + np.random.uniform(-0.08, 0.08), 0.1, 0.95)
            val = np.clip(val + np.random.uniform(-0.08, 0.08), 0.1, 0.95)
            self.eeg.set_sim_bias(float(eng), float(val))

        await self.broadcast({
            "type": "reading_target",
            "agent_id": resp.agent_id,
            "agent_name": resp.name,
            "agent_idx": idx,
            "total_agents": len(self.current_responses),
            "duration": READING_DURATION,
        })

        asyncio.create_task(self._reading_countdown(idx))

    async def _reading_countdown(self, idx: int):
        start = time.time()
        while time.time() - start < READING_DURATION:
            if self.reading_target_idx != idx:
                return
            elapsed = time.time() - start
            await self.broadcast({
                "type": "reading_progress",
                "agent_id": self.current_responses[idx].agent_id,
                "elapsed": round(elapsed, 1),
                "remaining": round(max(0, READING_DURATION - elapsed), 1),
                "progress": round(min(1.0, elapsed / READING_DURATION), 3),
            })
            await asyncio.sleep(0.5)

        if self.reading_target_idx == idx:
            await self.advance_reading()

    async def advance_reading(self):
        next_idx = self.reading_target_idx + 1
        self.reading_target_idx = -1
        await self.start_reading(next_idx)

    async def finish_round(self):
        self.phase = "scoring"
        self.reading_target_idx = -1
        await self.broadcast({"type": "phase", "phase": "scoring"})
        await asyncio.sleep(1.5)

        rewards = []
        for resp in self.current_responses:
            snapshots = self.reading_snapshots.get(resp.agent_id, [])
            reward = self.reward_computer.compute_reward(resp.agent_id, snapshots)
            rewards.append(reward)

        ranked = self.reward_computer.rank_rewards(rewards)
        winner_id = ranked[0].agent_id if ranked else None

        await self.broadcast({
            "type": "rewards",
            "rewards": [r.to_dict() for r in ranked],
            "winner": winner_id,
            "round": self.current_round,
        })

        if self.redis_client:
            try:
                self.redis_client.xadd("neural:round_rewards", {
                    "round": str(self.current_round),
                    "winner": winner_id or "",
                    "data": json.dumps({r.agent_id: r.total for r in ranked}),
                })
            except Exception:
                pass

        await asyncio.sleep(2)

        self.phase = "adapting"
        adaptations = self.arena.adapt_agents(rewards, self.current_round)

        self.round_history.append({
            "round": self.current_round,
            "topic": self.current_topic,
            "winner": winner_id,
            "rewards": {r.agent_id: r.total for r in rewards},
        })

        await self.broadcast({
            "type": "round_complete",
            "round": self.current_round,
            "winner": winner_id,
            "adaptations": adaptations,
            "history": self.round_history,
            "agent_stats": self.arena.get_stats(),
        })

        await asyncio.sleep(1)

        if self.current_round >= self.max_rounds:
            self.phase = "complete"
            await self.broadcast({
                "type": "phase", "phase": "complete",
                "history": self.round_history,
                "agent_stats": self.arena.get_stats(),
            })
        else:
            self.phase = "ready"
            await self.broadcast({"type": "phase", "phase": "ready"})

        print(
            f"Round {self.current_round} complete — winner: {winner_id} "
            f"({', '.join(f'{r.agent_id}={r.total:.2f}' for r in ranked)})",
            flush=True,
        )


async def main():
    sim_mode = "--sim" in sys.argv or not HAS_BRAINFLOW
    port = int(os.environ.get("PORT", "8080"))

    server = NeuralRLHFServer(sim_mode=sim_mode)
    server.eeg.connect()

    ws_server = await websockets.serve(server.handle_ws, "127.0.0.1", port)
    print(f"\nNeural RLHF server on ws://127.0.0.1:{port}", flush=True)
    print(f"  Mode:  {'SIMULATION' if sim_mode else 'LIVE (Muse S)'}", flush=True)
    print(f"  Weave: {'ON' if HAS_WEAVE else 'OFF'}", flush=True)
    print(f"  Redis: {'ON' if server.redis_client else 'OFF'}", flush=True)
    print(f"\nWaiting for frontend connection...\n", flush=True)

    brain_task = asyncio.create_task(server.brain_loop())

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        brain_task.cancel()
        server.eeg.cleanup()
        ws_server.close()


if __name__ == "__main__":
    asyncio.run(main())
