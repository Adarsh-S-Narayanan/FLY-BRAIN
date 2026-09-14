import numpy as np
from typing import Tuple, Optional

def cpu_lif_step(
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    weights: np.ndarray,
    prev_spikes: np.ndarray,
    external_inputs: np.ndarray,
    potentials_in: np.ndarray,
    refractory_in: Optional[np.ndarray] = None,
    decay: float = 0.85,
    threshold: float = 1.0,
    v_reset: float = 0.0,
    v_rest: float = 0.0,
    t_ref: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic CPU reference implementation of classical Leaky Integrate-and-Fire (LIF) dynamics.
    Implements:
      - Synaptic current summation: I_syn_i = sum_j W_ij * S_j(t)
      - Absolute refractory period handling (clamping to v_reset, spike suppression)
      - Leaky subthreshold integration: V_cand = v_rest + (V_old - v_rest)*decay + I_syn + I_ext
      - Action potential firing: S_i = 1.0, V_i = v_reset, R_i = t_ref when V_cand >= threshold
      - Hard reset and hyperpolarization floor
    
    Returns:
      (potentials_out, spikes_out, refractory_out)
    """
    N = len(potentials_in)
    if refractory_in is None:
        refractory_in = np.zeros(N, dtype=np.int32)
        
    potentials_out = np.zeros(N, dtype=np.float32)
    spikes_out = np.zeros(N, dtype=np.float32)
    refractory_out = np.zeros(N, dtype=np.int32)

    for i in range(N):
        ref_count = int(refractory_in[i])
        if ref_count > 0:
            # In absolute refractory period
            refractory_out[i] = ref_count - 1
            potentials_out[i] = v_reset
            spikes_out[i] = 0.0
            continue

        start_idx = row_offsets[i]
        end_idx = row_offsets[i + 1]

        synaptic_sum = 0.0
        for k in range(start_idx, end_idx):
            pre_idx = col_indices[k]
            synaptic_sum += float(weights[k]) * float(prev_spikes[pre_idx])

        v_old = float(potentials_in[i])
        v_cand = v_rest + (v_old - v_rest) * decay + synaptic_sum + float(external_inputs[i])

        if v_cand >= threshold:
            spikes_out[i] = 1.0
            potentials_out[i] = v_reset
            refractory_out[i] = t_ref
        else:
            spikes_out[i] = 0.0
            potentials_out[i] = max(v_cand, v_reset - 1.0)
            refractory_out[i] = 0

    return potentials_out, spikes_out, refractory_out

def cpu_brain_step(
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    weights: np.ndarray,
    prev_activations: np.ndarray,
    external_inputs: np.ndarray,
    potentials_in: np.ndarray,
    refractory_in: Optional[np.ndarray] = None,
    decay: float = 0.85,
    threshold: float = 1.0,
    v_reset: float = 0.0,
    v_rest: float = 0.0,
    t_ref: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backwards-compatible wrapper returning (potentials_out, spikes_out).
    """
    pot, spk, _ = cpu_lif_step(
        row_offsets=row_offsets,
        col_indices=col_indices,
        weights=weights,
        prev_spikes=prev_activations,
        external_inputs=external_inputs,
        potentials_in=potentials_in,
        refractory_in=refractory_in,
        decay=decay,
        threshold=threshold,
        v_reset=v_reset,
        v_rest=v_rest,
        t_ref=t_ref
    )
    return pot, spk

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
        a_pre = float(pre_activations[pre_idx])
        w = float(weights[k])
        delta_w = learning_rate * reward * (a_pre - weight_decay * w)
        new_w = float(np.clip(w + delta_w, min_weight, max_weight))
        new_weights[k] = new_w

    return new_weights
