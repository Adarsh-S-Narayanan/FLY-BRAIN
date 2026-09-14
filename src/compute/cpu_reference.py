import numpy as np
from typing import Tuple

def cpu_brain_step(
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    weights: np.ndarray,
    prev_activations: np.ndarray,
    external_inputs: np.ndarray,
    potentials_in: np.ndarray,
    decay: float = 0.85,
    threshold: float = 0.5,
    leak: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic CPU reference implementation of the connectome activation step.
    Matches the exact computation in shaders/brain_step.comp.
    """
    N = len(potentials_in)
    potentials_out = np.zeros(N, dtype=np.float32)
    next_activations = np.zeros(N, dtype=np.float32)

    for i in range(N):
        start_idx = row_offsets[i]
        end_idx = row_offsets[i + 1]
        
        synaptic_sum = 0.0
        for k in range(start_idx, end_idx):
            pre_idx = col_indices[k]
            synaptic_sum += weights[k] * prev_activations[pre_idx]
            
        v_old = potentials_in[i]
        v_new = v_old * decay + synaptic_sum + external_inputs[i] - leak
        
        # Sigmoid activation
        z = v_new - threshold
        a_new = 1.0 / (1.0 + np.exp(-z))
        
        potentials_out[i] = v_new
        next_activations[i] = a_new

    return potentials_out, next_activations

def cpu_plasticity_step(
    col_indices: np.ndarray,
    weights: np.ndarray,
    pre_activations: np.ndarray,
    learning_rate: float,
    reward: float,
    weight_decay: float = 0.01,
    min_weight: float = 0.01,
    max_weight: float = 1.0
) -> np.ndarray:
    """
    Deterministic CPU reference implementation of synaptic plasticity.
    Matches the computation in shaders/plasticity.comp.
    """
    M = len(weights)
    new_weights = np.copy(weights)
    
    for k in range(M):
        pre_idx = col_indices[k]
        a_pre = pre_activations[pre_idx]
        w = weights[k]
        delta_w = learning_rate * reward * (a_pre - weight_decay * w)
        new_weights[k] = np.clip(w + delta_w, min_weight, max_weight)
        
    return new_weights
