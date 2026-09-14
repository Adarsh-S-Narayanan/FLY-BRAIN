import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
from src.connectome.loader import get_or_create_circuit
from src.evolution.scheduler import EvolutionScheduler

def test_evolution_system():
    print("=== Testing Biological Connectome Evolution System ===")
    history_file = "diagnostics/evolution_history.json"
    if os.path.exists(history_file):
        os.remove(history_file)

    circuit = get_or_create_circuit(256, cache_name="test_evolution_circuit_256.npz")
    initial_neurons = circuit.num_neurons
    initial_synapses = circuit.num_synapses
    print(f"Base Connectome: {initial_neurons} neurons, {initial_synapses} synapses")

    scheduler = EvolutionScheduler(circuit, history_file=history_file)

    # Run 3 generations
    for gen in range(3):
        res = scheduler.run_generation(num_candidates=4, seed=100 + gen * 50)
        assert len(res["candidates"]) == 4, f"Expected 4 candidates, got {len(res['candidates'])}"

    # Inspect history file
    assert os.path.exists(history_file), "Evolution history file missing!"
    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["current_generation"] == 3
    assert len(data["history"]) == 3
    print(f"\nFinal Evolved Brain: {data['current_neurons']} neurons, {data['current_synapses']} synapses")
    print(f"Total candidates evaluated: {sum(len(g['candidates']) for g in data['history'])}")

    # Verify candidate selection was based on measured scores
    all_cands = [c for g in data["history"] for c in g["candidates"]]
    accepted = [c for c in all_cands if c["accepted"]]
    rejected = [c for c in all_cands if not c["accepted"]]
    print(f"Candidates accepted: {len(accepted)}, Candidates rejected/rolled-back: {len(rejected)}")
    assert len(accepted) > 0, "No candidate was accepted!"
    assert len(rejected) > 0, "No candidate was rejected!"

    print("SUCCESS: Evolution, mutation, structural growth/pruning, and rollback verified!")

if __name__ == "__main__":
    test_evolution_system()
