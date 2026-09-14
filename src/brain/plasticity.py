import numpy as np
from typing import Tuple, List, Dict, Optional
from src.connectome.types import ConnectomeGraph

class PlasticityEngine:
    def __init__(
        self,
        learning_rate: float = 0.05,
        weight_decay: float = 0.005,
        min_weight: float = 0.01,
        max_weight: float = 1.0,
        prune_threshold: float = 0.02,
        rewire_rate: float = 0.01
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.prune_threshold = prune_threshold
        self.rewire_rate = rewire_rate

    def apply_hebbian_update(
        self,
        graph: ConnectomeGraph,
        pre_activations: np.ndarray,
        post_activations: np.ndarray,
        reward: float
    ) -> int:
        """
        Applies reward-modulated Hebbian update directly to CSR weights.
        Returns the number of synapses updated.
        """
        M = len(graph.weights)
        for i in range(graph.num_neurons):
            start = graph.row_offsets[i]
            end = graph.row_offsets[i + 1]
            a_post = post_activations[i]
            for k in range(start, end):
                pre_idx = graph.col_indices[k]
                a_pre = pre_activations[pre_idx]
                w = graph.weights[k]
                
                # Three-factor rule: pre * post * reward - decay
                delta = self.learning_rate * reward * (a_pre * a_post - self.weight_decay * w)
                graph.weights[k] = float(np.clip(w + delta, self.min_weight, self.max_weight))
        return M

    def prune_weak_synapses(self, graph: ConnectomeGraph) -> int:
        """
        Prunes synapses with weight < prune_threshold.
        Rebuilds the CSR arrays in-place.
        Returns number of pruned synapses.
        """
        new_row_offsets = [0]
        new_col_indices = []
        new_weights = []
        pruned_count = 0

        for i in range(graph.num_neurons):
            start = graph.row_offsets[i]
            end = graph.row_offsets[i + 1]
            for k in range(start, end):
                w = graph.weights[k]
                if w >= self.prune_threshold:
                    new_col_indices.append(graph.col_indices[k])
                    new_weights.append(w)
                else:
                    pruned_count += 1
            new_row_offsets.append(len(new_col_indices))

        graph.row_offsets = np.array(new_row_offsets, dtype=np.int32)
        graph.col_indices = np.array(new_col_indices, dtype=np.int32)
        graph.weights = np.array(new_weights, dtype=np.float32)
        return pruned_count

    def rewire_coactive_synapses(
        self,
        graph: ConnectomeGraph,
        activations: np.ndarray,
        max_new_synapses: int = 20,
        seed: Optional[int] = None
    ) -> int:
        """
        Creates new synaptic connections between pairs of neurons that are simultaneously active
        but currently disconnected, respecting biological spatial proximity.
        """
        rng = np.random.RandomState(seed)
        active_indices = np.where(activations > 0.4)[0]
        if len(active_indices) < 2:
            return 0

        added_count = 0
        new_edges = {i: [] for i in range(graph.num_neurons)}
        
        # Populate existing edges
        for i in range(graph.num_neurons):
            start = graph.row_offsets[i]
            end = graph.row_offsets[i + 1]
            for k in range(start, end):
                new_edges[i].append((graph.col_indices[k], graph.weights[k]))

        # Search for candidates among active neurons
        shuffled = rng.permutation(active_indices)
        for idx_a in shuffled:
            if added_count >= max_new_synapses:
                break
            existing_targets = {t for t, _ in new_edges[idx_a]}
            coord_a = graph.coordinates[idx_a]
            
            for idx_b in shuffled:
                if idx_a == idx_b or idx_b in existing_targets:
                    continue
                coord_b = graph.coordinates[idx_b]
                dist = np.linalg.norm(coord_a - coord_b)
                # Within biological reach (e.g. 8000 nm)
                if dist < 8000.0:
                    init_weight = float(0.1 + 0.1 * rng.rand())
                    new_edges[idx_a].append((int(idx_b), init_weight))
                    added_count += 1
                    if added_count >= max_new_synapses:
                        break

        if added_count > 0:
            # Reconstruct CSR
            new_row_offsets = [0]
            new_col_indices = []
            new_weights = []
            for i in range(graph.num_neurons):
                # Sort targets by index
                new_edges[i].sort(key=lambda x: x[0])
                for target, w in new_edges[i]:
                    new_col_indices.append(target)
                    new_weights.append(w)
                new_row_offsets.append(len(new_col_indices))

            graph.row_offsets = np.array(new_row_offsets, dtype=np.int32)
            graph.col_indices = np.array(new_col_indices, dtype=np.int32)
            graph.weights = np.array(new_weights, dtype=np.float32)

        return added_count
