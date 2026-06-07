"""NeuralRL — Brain-as-reward-model reinforcement learning framework.

A unified RL library where EEG brain signals ARE the reward function.
Any agent (voice, rover, computer use) plugs in the same way:

    from neuralrl import BrainReward, CTS, NeuralFingerprint

    reward = BrainReward(eeg_source)
    policy = CTS(action_dim=8, context_dim=12)

    # Agent loop:
    ctx = reward.get_context()           # brain state → context vector
    action = policy.sample(ctx)          # Thompson sample from posterior
    # ... execute action ...
    r = reward.compute()                 # brain response → scalar reward
    policy.update(ctx, action, r)        # posterior update

The brain provides reward through engagement, valence, and error signals.
The policy learns what actions/parameters maximize brain reward for THIS user.
"""

from neuralrl.brain_reward import BrainReward
from neuralrl.cts import CTS
from neuralrl.fingerprint import NeuralFingerprint
from neuralrl.adaptation import AdaptationTracker
from neuralrl.brain_stream import BrainPublisher, BrainSubscriber
from neuralrl.structured_rl import BrainInterpreter, BrainInterpretation, StyleUpdate
from neuralrl.weave_tracker import WeaveTracker
from neuralrl.orchestrator import BrainOrchestrator, BrainAgent, NeuralState

__all__ = [
    "BrainReward", "CTS", "NeuralFingerprint", "AdaptationTracker",
    "BrainPublisher", "BrainSubscriber",
    "BrainInterpreter", "BrainInterpretation", "StyleUpdate",
    "WeaveTracker",
    "BrainOrchestrator", "BrainAgent", "NeuralState",
]
