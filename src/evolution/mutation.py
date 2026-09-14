import copy
import hashlib
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from src.connectome.types import ConnectomeGraph

class StructuralMutator:
    """
    Applies measurable structural mutations to the biological connectome:
    - Weight perturbation (strengthening / weakening)
    - Synaptic pruning (removing low-efficacy connections)
    - Synaptic rewiring (forming new proximate connections)
    - Population growth (adding new interneurons with biological coordinates)
    All mutations produce an audit record and are rollback-capable.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def mutate_weights(self, graph: ConnectomeGraph, rate: float = 0.05, scale: float = 0.1) -> Dict[str, Any]:
        mask = self.rng.rand(len(graph.weights)) < rate
        count = int(np.sum(mask))
        if count > 0:
            deltas = self.rng.normal(0.0, scale, count).astype(np.float32)
            graph.weights[mask] = np.clip(graph.weights[mask] + deltas, 0.01, 1.0)
            graph.graph_hash = graph.compute_graph_hash()
        return {
            "type": "weight_perturbation",
            "synapses_modified": count,
            "rate": rate,
            "scale": scale
        }

    def prune_synapses(self, graph: ConnectomeGraph, threshold: float = 0.03) -> Dict[str, Any]:
        new_row_offsets = [0]
        new_col_indices = []
        new_weights = []
        pruned = 0

        for i in range(graph.num_neurons):
            start = graph.row_offsets[i]
            end = graph.row_offsets[i + 1]
            for k in range(start, end):
                if graph.weights[k] >= threshold:
                    new_col_indices.append(graph.col_indices[k])
                    new_weights.append(graph.weights[k])
                else:
                    pruned += 1
            new_row_offsets.append(len(new_col_indices))

        graph.row_offsets = np.array(new_row_offsets, dtype=np.int32)
        graph.col_indices = np.array(new_col_indices, dtype=np.int32)
        graph.weights = np.array(new_weights, dtype=np.float32)
        graph.validate_invariants()
        graph.graph_hash = graph.compute_graph_hash()
        return {
            "type": "synaptic_pruning",
            "pruned_synapses": pruned,
            "threshold": threshold,
            "remaining_synapses": len(graph.weights)
        }

    def rewire_synapses(self, graph: ConnectomeGraph, num_new: int = 15, max_distance: float = 6000.0) -> Dict[str, Any]:
        N = graph.num_neurons
        added = 0
        adj = {i: list(zip(graph.col_indices[graph.row_offsets[i]:graph.row_offsets[i+1]],
                           graph.weights[graph.row_offsets[i]:graph.row_offsets[i+1]]))
               for i in range(N)}

        attempts = 0
        while added < num_new and attempts < num_new * 10:
            attempts += 1
            src = self.rng.randint(0, N)
            tgt = self.rng.randint(0, N)
            if src == tgt:
                continue
            existing = [t for t, _ in adj[src]]
            if tgt in existing:
                continue

            dist = np.linalg.norm(graph.coordinates[src] - graph.coordinates[tgt])
            if dist < max_distance:
                w = float(self.rng.uniform(0.05, 0.25))
                adj[src].append((tgt, w))
                added += 1

        new_row_offsets = [0]
        new_col_indices = []
        new_weights = []
        for i in range(N):
            adj[i].sort(key=lambda x: x[0])
            for t, w in adj[i]:
                new_col_indices.append(t)
                new_weights.append(w)
            new_row_offsets.append(len(new_col_indices))

        graph.row_offsets = np.array(new_row_offsets, dtype=np.int32)
        graph.col_indices = np.array(new_col_indices, dtype=np.int32)
        graph.weights = np.array(new_weights, dtype=np.float32)
        graph.validate_invariants()
        graph.graph_hash = graph.compute_graph_hash()
        return {
            "type": "synaptic_rewiring",
            "new_synapses_added": added,
            "total_synapses": len(graph.weights)
        }

    def grow_population(self, graph: ConnectomeGraph, new_neurons_count: int = 4) -> Dict[str, Any]:
        """
        Adds new interneurons placed within the male CNS volume and connects them to local neighbors.
        """
        mean_coord = graph.coordinates.mean(axis=0)
        std_coord = graph.coordinates.std(axis=0) * 0.5
        new_coords = self.rng.normal(mean_coord, std_coord, (new_neurons_count, 3)).astype(np.float32)
        
        last_id = int(np.max(graph.neuron_ids))
        new_ids = np.array([last_id + 1 + i for i in range(new_neurons_count)], dtype=np.int64)
        new_tbars = np.full(new_neurons_count, 50, dtype=np.int32)
        new_sides = ["M" for _ in range(new_neurons_count)]

        old_N = graph.num_neurons
        updated_ids = np.concatenate([graph.neuron_ids, new_ids])
        updated_coords = np.vstack([graph.coordinates, new_coords])
        updated_tbars = np.concatenate([graph.tbars, new_tbars])
        updated_sides = graph.sides + new_sides

        adj = {i: list(zip(graph.col_indices[graph.row_offsets[i]:graph.row_offsets[i+1]],
                           graph.weights[graph.row_offsets[i]:graph.row_offsets[i+1]]))
               for i in range(old_N)}

        # Connect new neurons to closest existing neurons
        for n_i in range(new_neurons_count):
            curr_idx = old_N + n_i
            adj[curr_idx] = []
            dists = np.linalg.norm(graph.coordinates - new_coords[n_i], axis=1)
            nearest = np.argsort(dists)[:4]
            for near in nearest:
                w = float(self.rng.uniform(0.1, 0.3))
                adj[curr_idx].append((int(near), w))
                adj[int(near)].append((curr_idx, w))

        new_row_offsets = [0]
        new_col_indices = []
        new_weights = []
        total_N = old_N + new_neurons_count
        for i in range(total_N):
            adj[i].sort(key=lambda x: x[0])
            for t, w in adj[i]:
                new_col_indices.append(t)
                new_weights.append(w)
            new_row_offsets.append(len(new_col_indices))

        graph.neuron_ids = updated_ids
        graph.coordinates = updated_coords
        graph.tbars = updated_tbars
        graph.sides = updated_sides
        graph.row_offsets = np.array(new_row_offsets, dtype=np.int32)
        graph.col_indices = np.array(new_col_indices, dtype=np.int32)
        graph.weights = np.array(new_weights, dtype=np.float32)
        graph.validate_invariants()
        graph.graph_hash = graph.compute_graph_hash()

        return {
            "type": "population_growth",
            "neurons_added": new_neurons_count,
            "total_neurons": total_N,
            "total_synapses": len(graph.weights)
        }

    def clone_graph(self, graph: ConnectomeGraph) -> ConnectomeGraph:
        # P9: preserve graph subclass (mode/provenance), population registry,
        # and metadata so candidates remain the same kind of graph as the parent.
        cls = type(graph)
        clone = cls(
            neuron_ids=graph.neuron_ids.copy(),
            coordinates=graph.coordinates.copy(),
            tbars=graph.tbars.copy(),
            sides=list(graph.sides),
            row_offsets=graph.row_offsets.copy(),
            col_indices=graph.col_indices.copy(),
            weights=graph.weights.copy(),
            populations=graph.populations,
            provenance_metadata=dict(graph.provenance_metadata),
        )
        # Recompute: never propagate a possibly stale hash.
        clone.graph_hash = clone.compute_graph_hash()
        return clone
