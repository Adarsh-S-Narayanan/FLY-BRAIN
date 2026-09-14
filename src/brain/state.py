from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Any, List, Optional

@dataclass
class HomeostaticDrives:
    energy: float = 1.0           # Depletes over time, replenished by sleep/reward
    curiosity: float = 0.8        # Increases with novel inputs, decreases with habituation
    social: float = 0.5           # Drives communication / speech
    integrity: float = 1.0        # Reduced by error/damage, recovered by rest

    def step(self, activity_level: float, novelty: float):
        # Metabolism consumes energy
        self.energy = max(0.0, min(1.0, self.energy - 0.001 * (1.0 + activity_level)))
        # Curiosity increases with novelty, decays with stagnation
        self.curiosity = max(0.0, min(1.0, self.curiosity * 0.99 + 0.1 * novelty))
        # Social drive gently drifts up to encourage interaction
        self.social = max(0.0, min(1.0, self.social + 0.0005))
        # Integrity recovers if energy is adequate
        if self.energy > 0.3:
            self.integrity = min(1.0, self.integrity + 0.001)

@dataclass
class BrainState:
    num_neurons: int
    membrane_potentials: np.ndarray    # [N] float32
    spikes: np.ndarray                 # [N] float32 (0.0 or 1.0)
    refractory_steps: np.ndarray       # [N] int32
    activations: np.ndarray            # [N] float32 (filtered rate / display)
    attention: np.ndarray              # [N] float32 attention focus
    prediction_error: float = 0.0
    predicted_reward: float = 0.0
    current_reward: float = 0.0
    drives: HomeostaticDrives = field(default_factory=HomeostaticDrives)
    active_goal: str = "explore"
    goal_embedding: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float32))
    tool_associations: Dict[str, float] = field(default_factory=dict)
    active_memory_refs: List[str] = field(default_factory=list)
    step_count: int = 0
    total_spikes: int = 0

    @classmethod
    def create_initial(cls, num_neurons: int, seed: int = 42) -> 'BrainState':
        rng = np.random.RandomState(seed)
        pot = rng.uniform(0.0, 0.2, num_neurons).astype(np.float32)
        spikes = np.zeros(num_neurons, dtype=np.float32)
        ref = np.zeros(num_neurons, dtype=np.int32)
        act = np.zeros(num_neurons, dtype=np.float32)
        att = np.ones(num_neurons, dtype=np.float32) / num_neurons
        return cls(
            num_neurons=num_neurons,
            membrane_potentials=pot,
            spikes=spikes,
            refractory_steps=ref,
            activations=act,
            attention=att,
            goal_embedding=np.zeros(16, dtype=np.float32)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "total_spikes": self.total_spikes,
            "prediction_error": float(self.prediction_error),
            "predicted_reward": float(self.predicted_reward),
            "current_reward": float(self.current_reward),
            "active_goal": self.active_goal,
            "drives": {
                "energy": float(self.drives.energy),
                "curiosity": float(self.drives.curiosity),
                "social": float(self.drives.social),
                "integrity": float(self.drives.integrity)
            },
            "tool_associations": dict(self.tool_associations),
            "active_memory_refs": list(self.active_memory_refs),
            "mean_activation": float(np.mean(self.activations)),
            "max_activation": float(np.max(self.activations)),
            "active_spikes_count": int(np.sum(self.spikes > 0.5))
        }
