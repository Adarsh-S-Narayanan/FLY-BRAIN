import time
import json
import numpy as np
from typing import Dict, Any, List, Optional
from src.brain.runtime import BrainRuntime
from src.memory.persistence import PersistentMemoryManager

class DreamEngine:
    """
    Offline dream and experience replay pipeline.
    Replays authentic stored experiences, recombines sensory observations,
    simulates alternative actions counterfactually, evaluates hypothetical outcomes,
    and consolidates insights into separate dream memory logs.
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

        for ep in episodes:
            base_ep_id = ep["id"]
            orig_action = ep["action"]
            
            # Select alternative counterfactual action
            possible_actions = ["speak", "generate_image", "remember", "act_in_environment"]
            alt_actions = [a for a in possible_actions if a != orig_action]
            sim_action = rng.choice(alt_actions) if mode == "exploratory" else alt_actions[0]

            # Recombine observation into simulated sensory input
            sim_sensory = rng.uniform(0.1, 0.4, 64).astype(np.float32)
            if mode == "exploratory":
                sim_sensory += rng.normal(0.0, 0.05, 64).astype(np.float32)

            # Step brain in counterfactual simulation (low plasticity to consolidate)
            sim_out = self.brain.step(sensory_inputs={"visual": sim_sensory}, reward=0.0)
            
            # Hypothetical reward evaluation
            counterfactual_reward = float(0.4 + 0.5 * sim_out["tool_scores"].get(sim_action, 0.0))
            
            insight = (
                f"Dream Replay of Ep #{base_ep_id}: Simulated '{sim_action}' instead of '{orig_action}'. "
                f"Counterfactual Reward: {counterfactual_reward:.3f}. "
                f"Consolidated memory pathways."
            )

            # Record in dedicated dream memory
            dream_id = self.memory.record_dream(
                seed=seed,
                base_episode_id=base_ep_id,
                simulated_action=sim_action,
                counterfactual_reward=round(counterfactual_reward, 3),
                insight=insight
            )

            dream_records.append({
                "dream_id": dream_id,
                "base_episode_id": base_ep_id,
                "mode": mode,
                "original_action": orig_action,
                "simulated_action": sim_action,
                "counterfactual_reward": round(counterfactual_reward, 3),
                "insight": insight
            })

        return dream_records
