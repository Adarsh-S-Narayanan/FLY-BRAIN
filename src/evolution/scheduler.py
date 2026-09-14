import os
import json
import uuid
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional
from src.connectome.types import ConnectomeGraph
from src.brain.runtime import BrainRuntime
from src.evolution.mutation import StructuralMutator

class EvolutionScheduler:
    """
    Parallel candidate evolution scheduler.
    Evaluates candidate brain variants on benchmark battery, records comprehensive metadata
    (parent ID, generation, mutation set, seed, score, hashes), and supports rollback.
    """
    def __init__(self, base_graph: ConnectomeGraph, history_file: str = "diagnostics/evolution_history.json"):
        self.current_graph = base_graph
        self.history_file = history_file
        self.generation = 0
        self.current_brain_id = f"brain_gen0_{str(uuid.uuid4())[:8]}"
        self.history: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

    def _hash_graph(self, graph: ConnectomeGraph) -> str:
        h = hashlib.sha256()
        h.update(graph.weights.tobytes())
        h.update(graph.col_indices.tobytes())
        h.update(graph.row_offsets.tobytes())
        return h.hexdigest()

    def evaluate_candidate(self, graph: ConnectomeGraph, seed: int = 42) -> float:
        """
        Standardized fitness benchmark:
        Evaluates dynamic responsiveness, sensory sensitivity, and signal-to-noise ratio.
        """
        runtime = BrainRuntime(graph, use_gpu=False, enable_plasticity=False, seed=seed)
        scores = []

        # Trial 1: Sensory stimulation
        sensory = {"visual": np.full(min(64, graph.num_neurons), 0.5, dtype=np.float32)}
        step1 = runtime.step(sensory_inputs=sensory)
        mean_act = step1["mean_activation"]
        spikes = step1["spikes"]
        
        # Trial 2: Motor tool responsiveness
        speak_score = float(runtime.state.tool_associations.get("speak", 0.0))
        
        # Fitness combines responsiveness, selectivity, and energy-efficient firing
        fitness = (mean_act * 3.0) + (speak_score * 2.0) - (0.0005 * spikes)
        return float(round(fitness, 4))

    def run_generation(
        self,
        num_candidates: int = 4,
        seed: int = 100
    ) -> Dict[str, Any]:
        """
        Executes one evolutionary generation across multiple candidate variants.
        Evaluates all variants and selects the best candidate if it improves fitness.
        """
        self.generation += 1
        mutator = StructuralMutator(seed=seed + self.generation)
        baseline_score = self.evaluate_candidate(self.current_graph, seed=seed)
        
        print(f"\n--- Generation {self.generation} (Baseline Score: {baseline_score:.4f}) ---")
        candidates = []
        best_candidate = None
        best_score = baseline_score
        
        for c_idx in range(num_candidates):
            cand_seed = seed + self.generation * 10 + c_idx
            cand_graph = mutator.clone_graph(self.current_graph)
            
            # Apply distinct mutation combinations
            mutations = []
            if c_idx == 0:
                mutations.append(mutator.mutate_weights(cand_graph, rate=0.08, scale=0.15))
            elif c_idx == 1:
                mutations.append(mutator.prune_synapses(cand_graph, threshold=0.04))
                mutations.append(mutator.rewire_synapses(cand_graph, num_new=10))
            elif c_idx == 2:
                mutations.append(mutator.grow_population(cand_graph, new_neurons_count=4))
                mutations.append(mutator.mutate_weights(cand_graph, rate=0.05, scale=0.1))
            else:
                mutations.append(mutator.rewire_synapses(cand_graph, num_new=15))
                mutations.append(mutator.mutate_weights(cand_graph, rate=0.04, scale=0.08))

            cand_score = self.evaluate_candidate(cand_graph, seed=cand_seed)
            cand_id = f"brain_gen{self.generation}_c{c_idx}_{str(uuid.uuid4())[:6]}"
            cand_hash = self._hash_graph(cand_graph)

            accepted = cand_score > best_score
            decision_reason = "Improved fitness over baseline" if accepted else "Fitness did not exceed best candidate"

            cand_record = {
                "candidate_id": cand_id,
                "parent_id": self.current_brain_id,
                "generation": self.generation,
                "seed": cand_seed,
                "num_neurons": cand_graph.num_neurons,
                "num_synapses": cand_graph.num_synapses,
                "mutations": mutations,
                "score": cand_score,
                "baseline_score": baseline_score,
                "accepted": accepted,
                "reason": decision_reason,
                "sha256": cand_hash
            }
            candidates.append(cand_record)
            print(f"Candidate {c_idx}: Score={cand_score:.4f}, Neurons={cand_graph.num_neurons}, Synapses={cand_graph.num_synapses} "
                  f"| {'[NEW LEADER]' if accepted else '[REJECTED]'}")

            if accepted:
                best_score = cand_score
                best_candidate = (cand_id, cand_graph, cand_record)

        # Selection or Rollback
        if best_candidate is not None and best_score > baseline_score:
            self.current_brain_id, self.current_graph, best_rec = best_candidate
            best_rec["reason"] = "ACCEPTED: Highest fitness in generation"
            print(f"-> Selected Candidate {self.current_brain_id} (Score: {best_score:.4f})")
        else:
            print(f"-> Rollback: Retained parent baseline {self.current_brain_id} (Score: {baseline_score:.4f})")

        gen_summary = {
            "generation": self.generation,
            "parent_id": self.current_brain_id,
            "baseline_score": baseline_score,
            "best_score": best_score,
            "improved": best_score > baseline_score,
            "candidates": candidates
        }
        self.history.append(gen_summary)
        self.save_history()
        return gen_summary

    def get_lineage(self) -> Dict[str, Any]:
        """Returns the full evolutionary lineage tree with candidate decisions."""
        return {
            "current_generation": self.generation,
            "current_brain_id": self.current_brain_id,
            "current_neurons": self.current_graph.num_neurons,
            "current_synapses": self.current_graph.num_synapses,
            "history": self.history
        }

    def save_history(self):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.get_lineage(), f, indent=2)
