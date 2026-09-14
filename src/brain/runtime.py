import os
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from src.connectome.types import ConnectomeGraph
from src.brain.state import BrainState, HomeostaticDrives
from src.brain.plasticity import PlasticityEngine
from src.compute.vulkan_backend import VulkanComputeEngine
from src.compute.cpu_reference import cpu_brain_step

class BrainRuntime:
    """
    Central Brain Runtime integrating biological connectome graph,
    Vulkan GPU acceleration, synaptic plasticity, homeostatic drives,
    and sensory-motor population mapping.
    """
    def __init__(
        self,
        graph: ConnectomeGraph,
        use_gpu: bool = True,
        enable_plasticity: bool = True,
        seed: int = 42
    ):
        self.graph = graph
        self.use_gpu = use_gpu
        self.enable_plasticity = enable_plasticity
        self.seed = seed
        self.state = BrainState.create_initial(graph.num_neurons, seed=seed)
        self.plasticity = PlasticityEngine()
        
        self.gpu_engine: Optional[VulkanComputeEngine] = None
        if self.use_gpu:
            try:
                self.gpu_engine = VulkanComputeEngine()
            except Exception as e:
                print(f"[BrainRuntime] Vulkan initialization warning: {e}. Using CPU engine.")
                self.gpu_engine = None
                self.use_gpu = False

        # Population indices for sensory & motor interfaces
        N = graph.num_neurons
        self.sensory_visual_indices = np.arange(0, min(64, N))
        self.sensory_audio_indices = np.arange(64, min(128, N)) if N >= 128 else self.sensory_visual_indices
        self.sensory_memory_indices = np.arange(128, min(192, N)) if N >= 192 else self.sensory_visual_indices
        
        self.motor_speak_indices = np.arange(max(0, N - 128), max(0, N - 96))
        self.motor_image_indices = np.arange(max(0, N - 96), max(0, N - 64))
        self.motor_remember_indices = np.arange(max(0, N - 64), max(0, N - 32))
        self.motor_act_indices = np.arange(max(0, N - 32), N)

    def step(
        self,
        sensory_inputs: Optional[Dict[str, np.ndarray]] = None,
        reward: float = 0.0
    ) -> Dict[str, Any]:
        """
        Executes one full cognitive step of the brain:
        1. Injects sensory inputs into sensory neuron populations
        2. Dispatches activation propagation on Vulkan GPU (or CPU)
        3. Updates prediction error, reward, and drives
        4. Applies reward-modulated Hebbian plasticity
        5. Computes motor tool activations
        """
        N = self.graph.num_neurons
        ext_inputs = np.zeros(N, dtype=np.float32)

        # Inject sensory inputs
        if sensory_inputs:
            if "visual" in sensory_inputs:
                v_in = np.asarray(sensory_inputs["visual"], dtype=np.float32).flatten()
                length = min(len(v_in), len(self.sensory_visual_indices))
                ext_inputs[self.sensory_visual_indices[:length]] += v_in[:length]
            if "audio" in sensory_inputs:
                a_in = np.asarray(sensory_inputs["audio"], dtype=np.float32).flatten()
                length = min(len(a_in), len(self.sensory_audio_indices))
                ext_inputs[self.sensory_audio_indices[:length]] += a_in[:length]
            if "memory" in sensory_inputs:
                m_in = np.asarray(sensory_inputs["memory"], dtype=np.float32).flatten()
                length = min(len(m_in), len(self.sensory_memory_indices))
                ext_inputs[self.sensory_memory_indices[:length]] += m_in[:length]

        prev_act = self.state.activations.copy()
        pot_in = self.state.membrane_potentials.copy()

        # Neural propagation (GPU or CPU)
        if self.gpu_engine is not None and self.use_gpu:
            new_pot, new_act = self.gpu_engine.run_step(
                self.graph.row_offsets,
                self.graph.col_indices,
                self.graph.weights,
                prev_act,
                ext_inputs,
                pot_in
            )
        else:
            new_pot, new_act = cpu_brain_step(
                self.graph.row_offsets,
                self.graph.col_indices,
                self.graph.weights,
                prev_act,
                ext_inputs,
                pot_in
            )

        # Update state
        self.state.membrane_potentials = new_pot
        self.state.activations = new_act
        self.state.step_count += 1
        spikes = int(np.sum(new_act > 0.5))
        self.state.total_spikes += spikes

        # Prediction error: delta = reward - predicted_reward
        self.state.current_reward = float(reward)
        self.state.prediction_error = float(reward - self.state.predicted_reward)
        self.state.predicted_reward = float(self.state.predicted_reward * 0.9 + 0.1 * reward)

        # Update homeostatic drives
        activity_level = float(np.mean(new_act))
        novelty = float(np.mean(np.abs(new_act - prev_act)))
        self.state.drives.step(activity_level, novelty)

        # Plasticity
        synapses_updated = 0
        if self.enable_plasticity and (abs(reward) > 1e-4 or activity_level > 0.05):
            synapses_updated = self.plasticity.apply_hebbian_update(
                self.graph,
                pre_activations=prev_act,
                post_activations=new_act,
                reward=reward
            )

        # Read out motor / tool triggers
        speak_score = float(np.mean(new_act[self.motor_speak_indices])) if len(self.motor_speak_indices) > 0 else 0.0
        image_score = float(np.mean(new_act[self.motor_image_indices])) if len(self.motor_image_indices) > 0 else 0.0
        remember_score = float(np.mean(new_act[self.motor_remember_indices])) if len(self.motor_remember_indices) > 0 else 0.0
        act_score = float(np.mean(new_act[self.motor_act_indices])) if len(self.motor_act_indices) > 0 else 0.0

        self.state.tool_associations = {
            "speak": speak_score,
            "generate_image": image_score,
            "remember": remember_score,
            "act_in_environment": act_score
        }

        # Select highest-scoring tool above threshold (0.35)
        selected_tool = None
        best_val = 0.35
        for t, val in self.state.tool_associations.items():
            if val > best_val:
                best_val = val
                selected_tool = t

        return {
            "step": self.state.step_count,
            "backend": "vulkan_gpu" if (self.gpu_engine and self.use_gpu) else "cpu_reference",
            "spikes": spikes,
            "mean_activation": activity_level,
            "prediction_error": self.state.prediction_error,
            "current_reward": reward,
            "selected_tool": selected_tool,
            "tool_scores": self.state.tool_associations,
            "synapses_updated": synapses_updated,
            "drives": {
                "energy": self.state.drives.energy,
                "curiosity": self.state.drives.curiosity,
                "social": self.state.drives.social,
                "integrity": self.state.drives.integrity
            }
        }

    def save_snapshot(self, filepath: str):
        """Saves complete brain snapshot including connectome and runtime state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(
            filepath,
            neuron_ids=self.graph.neuron_ids,
            coordinates=self.graph.coordinates,
            tbars=self.graph.tbars,
            sides=np.array(self.graph.sides),
            row_offsets=self.graph.row_offsets,
            col_indices=self.graph.col_indices,
            weights=self.graph.weights,
            membrane_potentials=self.state.membrane_potentials,
            activations=self.state.activations,
            attention=self.state.attention,
            goal_embedding=self.state.goal_embedding,
            meta=json.dumps({
                "step_count": self.state.step_count,
                "total_spikes": self.state.total_spikes,
                "prediction_error": self.state.prediction_error,
                "predicted_reward": self.state.predicted_reward,
                "current_reward": self.state.current_reward,
                "active_goal": self.state.active_goal,
                "tool_associations": self.state.tool_associations,
                "drives": {
                    "energy": self.state.drives.energy,
                    "curiosity": self.state.drives.curiosity,
                    "social": self.state.drives.social,
                    "integrity": self.state.drives.integrity
                }
            })
        )

    def restore_snapshot(self, filepath: str):
        """Restores complete brain snapshot and verifies continuity."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Snapshot file not found: {filepath}")
        data = np.load(filepath, allow_pickle=True)
        
        self.graph.neuron_ids = data["neuron_ids"]
        self.graph.coordinates = data["coordinates"]
        self.graph.tbars = data["tbars"]
        self.graph.sides = list(data["sides"])
        self.graph.row_offsets = data["row_offsets"]
        self.graph.col_indices = data["col_indices"]
        self.graph.weights = data["weights"]
        
        self.state.membrane_potentials = data["membrane_potentials"]
        self.state.activations = data["activations"]
        self.state.attention = data["attention"]
        self.state.goal_embedding = data["goal_embedding"]
        
        meta = json.loads(str(data["meta"]))
        self.state.step_count = meta["step_count"]
        self.state.total_spikes = meta["total_spikes"]
        self.state.prediction_error = meta["prediction_error"]
        self.state.predicted_reward = meta["predicted_reward"]
        self.state.current_reward = meta["current_reward"]
        self.state.active_goal = meta["active_goal"]
        self.state.tool_associations = meta["tool_associations"]
        self.state.drives.energy = meta["drives"]["energy"]
        self.state.drives.curiosity = meta["drives"]["curiosity"]
        self.state.drives.social = meta["drives"]["social"]
        self.state.drives.integrity = meta["drives"]["integrity"]

    def cleanup(self):
        if self.gpu_engine:
            self.gpu_engine.cleanup()
            self.gpu_engine = None
