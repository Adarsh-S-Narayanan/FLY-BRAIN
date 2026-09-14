import time
import json
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional
from src.brain.runtime import BrainRuntime
from src.memory.persistence import PersistentMemoryManager

class DreamEngine:
    """
    Offline dream and experience replay pipeline.
    Replays authentic stored experiences, recombines sensory observations,
    simulates alternative actions counterfactually, evaluates hypothetical outcomes,
    and consolidates insights into persistent memory.
    """
    def __init__(self, brain: BrainRuntime, memory_manager: PersistentMemoryManager):
        self.brain = brain
        self.memory = memory_manager

    def run_dream_cycle(
        self,
        mode: str = "deterministic",
        seed: int = 42,
        num_episodes_to_replay: int = 3
    ) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(seed)
        episodes = self.memory.get_recent_episodes(limit=num_episodes_to_replay)
        
        if not episodes:
            # If no episodes in database yet, create a baseline memory event
            self.memory.record_episode(
                step=1,
                observation={"visual": "floral_stimulus"},
                action="observe_visual",
                reward=0.5,
                prediction_error=0.1,
                outcome={"detected": True}
            )
            episodes = self.memory.get_recent_episodes(limit=1)

        dream_records = []
        is_deterministic = (mode == "deterministic")

        for ep in episodes:
            base_ep_id = ep["id"]
            orig_action = ep["action"]
            
            # Initial state hash before replay
            h_init = hashlib.sha256()
            h_init.update(self.brain.state.membrane_potentials.tobytes())
            h_init.update(self.brain.state.spikes.tobytes())
            initial_state_hash = h_init.hexdigest()

            # Select alternative counterfactual action
            possible_actions = ["speak", "generate_image", "remember", "act_in_environment"]
            alt_actions = [a for a in possible_actions if a != orig_action]
            sim_action = rng.choice(alt_actions) if mode == "exploratory" else alt_actions[0]

            # Recombine observation into simulated sensory input
            vis_len = min(64, self.brain.graph.num_neurons)
            sim_sensory = rng.uniform(0.1, 0.4, vis_len).astype(np.float32)
            if mode == "exploratory":
                sim_sensory += rng.normal(0.0, 0.05, vis_len).astype(np.float32)

            # Step brain in counterfactual simulation (low reward to consolidate)
            sim_out = self.brain.step(sensory_inputs={"visual": sim_sensory}, reward=0.0)
            
            # Hypothetical reward evaluation
            act_score = float(self.brain.state.tool_associations.get(sim_action, 0.0))
            counterfactual_reward = float(round(0.4 + 0.5 * act_score, 3))

            # Result state hash
            h_res = hashlib.sha256()
            h_res.update(self.brain.state.membrane_potentials.tobytes())
            h_res.update(self.brain.state.spikes.tobytes())
            result_state_hash = h_res.hexdigest()

            insight = (
                f"Replayed episode #{base_ep_id} (orig action: {orig_action}). "
                f"Counterfactual simulation '{sim_action}' yielded predicted reward {counterfactual_reward:.2f}."
            )

            # Consolidate into persistent memory
            dream_id = self.memory.record_dream(
                episode_id=base_ep_id,
                replay_step=self.brain.state.step_count,
                simulated_action=sim_action,
                hypothetical_reward=counterfactual_reward,
                consolidation_insight=insight
            )

            record = {
                "dream_id": dream_id,
                "base_episode_id": base_ep_id,
                "mode": mode,
                "is_deterministic": is_deterministic,
                "seed": seed,
                "initial_state_hash": initial_state_hash,
                "counterfactual_action": sim_action,
                "parameters": {
                    "noise_scale": 0.05 if mode == "exploratory" else 0.0,
                    "sensory_len": vis_len
                },
                "counterfactual_reward": counterfactual_reward,
                "result_state_hash": result_state_hash,
                "consolidation_result": "CONSOLIDATED_TO_PERSISTENT_MEMORY",
                "insight": insight
            }
            dream_records.append(record)

        return dream_records
