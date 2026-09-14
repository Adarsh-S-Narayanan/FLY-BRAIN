import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import hashlib
import numpy as np
from src.connectome.loader import get_or_create_circuit
from src.brain.runtime import BrainRuntime

def file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_persistence_workflow():
    print("=== Testing Brain State Persistence and Deterministic Continuity ===")
    os.makedirs("diagnostics/snapshots", exist_ok=True)
    snapshot_path = "diagnostics/snapshots/brain_state_checkpoint.npz"

    circuit = get_or_create_circuit(512, cache_name="test_persistence_circuit_512.npz")
    brain_a = BrainRuntime(circuit, use_gpu=True, seed=42)
    
    # 1. Run 5 steps with mock sensory inputs
    print("Running initial 5 steps on Brain A...")
    for step_i in range(5):
        s_input = {
            "visual": np.full(64, 0.2 + 0.05 * step_i, dtype=np.float32),
            "audio": np.full(64, 0.1, dtype=np.float32)
        }
        res = brain_a.step(s_input, reward=0.5 if step_i % 2 == 0 else -0.1)
    
    print(f"Brain A step: {brain_a.state.step_count}, total spikes: {brain_a.state.total_spikes}")
    brain_a.save_snapshot(snapshot_path)
    snapshot_hash = file_sha256(snapshot_path)
    print(f"Saved snapshot to {snapshot_path} (SHA-256: {snapshot_hash})")
    
    # Record state after 5 steps
    pot_a_5 = brain_a.state.membrane_potentials.copy()
    act_a_5 = brain_a.state.activations.copy()
    weights_a_5 = brain_a.graph.weights.copy()
    step_a_5 = brain_a.state.step_count
    
    # Continue Brain A for 5 more steps
    print("Continuing Brain A for steps 6..10...")
    for step_i in range(5, 10):
        s_input = {
            "visual": np.full(64, 0.1 * step_i, dtype=np.float32),
            "audio": np.full(64, 0.15, dtype=np.float32)
        }
        brain_a.step(s_input, reward=0.8)
        
    pot_a_10 = brain_a.state.membrane_potentials.copy()
    act_a_10 = brain_a.state.activations.copy()
    weights_a_10 = brain_a.graph.weights.copy()
    step_a_10 = brain_a.state.step_count
    brain_a.cleanup()
    
    # 2. Spawn a fresh Brain B, restore from snapshot, and verify exact match with step 5
    print("\nSpawning fresh Brain B and restoring from snapshot...")
    fresh_circuit = get_or_create_circuit(512, cache_name="test_persistence_circuit_512.npz")
    brain_b = BrainRuntime(fresh_circuit, use_gpu=True, seed=999) # different initial seed
    brain_b.restore_snapshot(snapshot_path)
    
    # Verify state matches Brain A at step 5
    diff_pot_5 = float(np.max(np.abs(pot_a_5 - brain_b.state.membrane_potentials)))
    diff_act_5 = float(np.max(np.abs(act_a_5 - brain_b.state.activations)))
    diff_weights_5 = float(np.max(np.abs(weights_a_5 - brain_b.graph.weights)))
    print(f"Step 5 comparison after restore: pot diff={diff_pot_5:.2e}, act diff={diff_act_5:.2e}, weight diff={diff_weights_5:.2e}")
    assert diff_pot_5 == 0.0, "Potentials do not match after restore!"
    assert diff_act_5 == 0.0, "Activations do not match after restore!"
    assert diff_weights_5 == 0.0, "Weights do not match after restore!"
    assert brain_b.state.step_count == step_a_5, "Step counts do not match!"
    
    # 3. Execute identical inputs 6..10 on Brain B
    print("Executing identical steps 6..10 on Brain B...")
    for step_i in range(5, 10):
        s_input = {
            "visual": np.full(64, 0.1 * step_i, dtype=np.float32),
            "audio": np.full(64, 0.15, dtype=np.float32)
        }
        brain_b.step(s_input, reward=0.8)
        
    diff_pot_10 = float(np.max(np.abs(pot_a_10 - brain_b.state.membrane_potentials)))
    diff_act_10 = float(np.max(np.abs(act_a_10 - brain_b.state.activations)))
    diff_weights_10 = float(np.max(np.abs(weights_a_10 - brain_b.graph.weights)))
    print(f"Step 10 comparison after continued execution: pot diff={diff_pot_10:.2e}, act diff={diff_act_10:.2e}, weight diff={diff_weights_10:.2e}")
    assert diff_pot_10 < 1e-4, f"Divergence in potential at step 10: {diff_pot_10}"
    assert diff_act_10 < 1e-4, f"Divergence in activation at step 10: {diff_act_10}"
    assert diff_weights_10 < 1e-4, f"Divergence in weights at step 10: {diff_weights_10}"
    assert brain_b.state.step_count == step_a_10, "Step counts do not match at step 10!"
    brain_b.cleanup()
    
    manifest = {
        "snapshot_path": snapshot_path,
        "sha256": snapshot_hash,
        "num_neurons": int(circuit.num_neurons),
        "num_synapses": int(circuit.num_synapses),
        "step_at_snapshot": step_a_5,
        "step_restored": brain_b.state.step_count,
        "continuity_verified": True,
        "max_discrepancy_on_restore": 0.0,
        "max_discrepancy_after_replay": max(diff_pot_10, diff_act_10, diff_weights_10)
    }
    manifest_file = "diagnostics/brain_snapshot_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_file}")
    print("SUCCESS: State persistence and deterministic replay verified!")

if __name__ == "__main__":
    test_persistence_workflow()
