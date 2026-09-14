"""
Deterministic CPU reference implementation of classical Leaky Integrate-and-Fire (LIF).

UNIT CONTRACT (architecture C): the implementation works in NORMALIZED simulation
units by default (decay=0.85, threshold=1.0, v_reset=0.0, v_rest=0.0, t_ref=2).
Physical millivolt semantics are obtained by passing explicit parameters, with the
reference mapping V_norm = (V_mV + 70.0) / 20.0, i.e. rest -70mV -> 0.0,
threshold -50mV -> 1.0, reset -70mV -> 0.0. The Vulkan shader implements the
identical parametrized equations; CPU/GPU parity is validated by
scripts + tests, not assumed.
"""
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
    post_activations: np.ndarray,
    row_offsets: np.ndarray,
    learning_rate: float = 0.05,
    reward: float = 1.0,
    weight_decay: float = 0.01,
    min_weight: float = 0.01,
    max_weight: float = 1.0
) -> np.ndarray:
    """
    Deterministic CPU reference implementation of the documented three-factor
    reward-modulated Hebbian rule. Matches shaders/plasticity.comp exactly:
        delta_w = lr * reward * (a_pre * a_post - decay * w)
    where a_post is the postsynaptic (CSR row owner) activation of each synapse.
    """
    M = len(weights)
    new_weights = np.copy(weights)
    pre_act = np.asarray(pre_activations, dtype=np.float64)
    post_act = np.asarray(post_activations, dtype=np.float64)
    # CSR row (postsynaptic neuron) owning each synapse k.
    post_idx = np.searchsorted(np.asarray(row_offsets), np.arange(M), side="right") - 1

    for k in range(M):
        pre_idx = int(col_indices[k])
        a_pre = float(pre_act[pre_idx])
        a_post = float(post_act[int(post_idx[k])])
        w = float(weights[k])
        delta_w = learning_rate * reward * (a_pre * a_post - weight_decay * w)
        new_w = float(np.clip(w + delta_w, min_weight, max_weight))
        new_weights[k] = new_w

    return new_weights
