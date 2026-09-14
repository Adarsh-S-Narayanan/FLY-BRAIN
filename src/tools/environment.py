from typing import Dict, Any, Optional
from src.tools.base import ToolConnector
from src.brain.runtime import BrainRuntime

class ActInEnvironmentConnector(ToolConnector):
    def __init__(self):
        super().__init__(
            name="act_in_environment",
            description="Executes a physical or simulated locomotion/foraging action in the virtual ecosystem.",
            timeout_sec=5.0
        )
        self.agent_pos = [0.0, 0.0]
        self.heading_rad = 0.0

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "enum": ["move_forward", "turn_left", "turn_right", "forage"]},
                "magnitude": {"type": "number", "default": 1.0}
            },
            "required": ["action_type"]
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "new_position": {"type": "array"},
                "heading": {"type": "number"},
                "foraged_reward": {"type": "number"}
            },
            "required": ["new_position", "heading", "foraged_reward"]
        }

    def _execute(self, params: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        act = params["action_type"]
        mag = float(params.get("magnitude", 1.0))
        reward = 0.0

        if act == "move_forward":
            self.agent_pos[0] += mag * 0.1
            self.agent_pos[1] += mag * 0.1
        elif act == "turn_left":
            self.heading_rad += 0.1 * mag
        elif act == "turn_right":
            self.heading_rad -= 0.1 * mag
        elif act == "forage":
            reward = 0.5 * mag

        return {
            "new_position": [round(p, 3) for p in self.agent_pos],
            "heading": round(self.heading_rad, 3),
            "foraged_reward": reward
        }

class InspectSelfConnector(ToolConnector):
    def __init__(self, brain: BrainRuntime):
        super().__init__(
            name="inspect_self",
            description="Inspects internal computational state, drives, attention, and spike telemetry of the brain.",
            timeout_sec=5.0
        )
        self.brain = brain

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "step": {"type": "integer"},
                "total_spikes": {"type": "integer"},
                "energy": {"type": "number"},
                "curiosity": {"type": "number"},
                "social": {"type": "number"},
                "integrity": {"type": "number"},
                "num_neurons": {"type": "integer"},
                "num_synapses": {"type": "integer"}
            },
            "required": ["step", "energy", "num_neurons", "num_synapses"]
        }

    def _execute(self, params: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        return {
            "step": self.brain.state.step_count,
            "total_spikes": self.brain.state.total_spikes,
            "energy": round(self.brain.state.drives.energy, 4),
            "curiosity": round(self.brain.state.drives.curiosity, 4),
            "social": round(self.brain.state.drives.social, 4),
            "integrity": round(self.brain.state.drives.integrity, 4),
            "num_neurons": self.brain.graph.num_neurons,
            "num_synapses": self.brain.graph.num_synapses
        }

class SleepConnector(ToolConnector):
    def __init__(self, brain: BrainRuntime):
        super().__init__(
            name="sleep",
            description="Enters sleep/quiescent state to replenish homeostatic energy and repair neural integrity.",
            timeout_sec=5.0
        )
        self.brain = brain

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "duration_cycles": {"type": "integer", "default": 5}
            }
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "energy_restored": {"type": "number"},
                "current_energy": {"type": "number"}
            },
            "required": ["energy_restored", "current_energy"]
        }

    def _execute(self, params: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        cycles = int(params.get("duration_cycles", 5))
        before_e = self.brain.state.drives.energy
        restored = min(1.0 - before_e, 0.1 * cycles)
        self.brain.state.drives.energy = min(1.0, before_e + restored)
        self.brain.state.drives.integrity = min(1.0, self.brain.state.drives.integrity + 0.05 * cycles)
        return {
            "energy_restored": round(restored, 4),
            "current_energy": round(self.brain.state.drives.energy, 4)
        }
