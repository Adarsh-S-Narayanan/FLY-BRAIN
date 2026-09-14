from typing import Dict, Any, Optional
import numpy as np
from src.tools.base import ToolConnector
from src.memory.persistence import PersistentMemoryManager

class RememberConnector(ToolConnector):
    def __init__(self, memory_manager: PersistentMemoryManager):
        super().__init__(
            name="remember",
            description="Persists an observation, experience, or concept into persistent episodic and semantic memory.",
            timeout_sec=5.0
        )
        self.mem = memory_manager

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "step": {"type": "integer"},
                "observation": {"type": "object"},
                "action": {"type": "string"},
                "reward": {"type": "number"},
                "outcome": {"type": "object"},
                "concept": {"type": "string"}
            },
            "required": ["step", "action"]
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "memory_id": {"type": "integer"},
                "status": {"type": "string"}
            },
            "required": ["memory_id", "status"]
        }

    def _execute(self, params: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        step = int(params["step"])
        action = str(params["action"])
        obs = params.get("observation", {})
        reward = float(params.get("reward", 0.0))
        outcome = params.get("outcome", {})

        ep_id = self.mem.record_episode(
            step=step,
            observation=obs,
            action=action,
            reward=reward,
            prediction_error=0.0,
            outcome=outcome
        )

        if "concept" in params:
            concept = str(params["concept"])
            dummy_emb = np.random.RandomState(step).randn(16).astype(np.float32)
            self.mem.store_concept(concept, f"Concept formed at step {step}", dummy_emb, associations=obs)

        return {
            "memory_id": ep_id,
            "status": "STORED"
        }

class RetrieveMemoryConnector(ToolConnector):
    def __init__(self, memory_manager: PersistentMemoryManager):
        super().__init__(
            name="retrieve_memory",
            description="Retrieves relevant past experiences and semantic concepts from persistent memory.",
            timeout_sec=5.0
        )
        self.mem = memory_manager

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 3}
            }
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "recent_episodes": {"type": "array"},
                "retrieved_count": {"type": "integer"}
            },
            "required": ["recent_episodes", "retrieved_count"]
        }

    def _execute(self, params: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        limit = int(params.get("limit", 3))
        episodes = self.mem.get_recent_episodes(limit=limit)
        return {
            "recent_episodes": episodes,
            "retrieved_count": len(episodes)
        }
