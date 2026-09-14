"""Real developmental brain growth on the connectome graph (REAL, IMPLEMENTED).

Implements: progenitor division -> neurogenesis -> differentiation ->
migration -> axon/dendrite growth -> synaptogenesis -> stabilization/pruning
-> apoptosis. All deterministic given (development_seed, tick).
"""
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.connectome.types import ConnectomeGraph
from src.common.determinism import derive_subseed
from src.common.events import EventLog

CELL_TYPES = ("sensory", "interneuron", "motor", "modulatory", "memory")


@dataclass
class DevelopmentState:
    cell_types: List[str] = field(default_factory=list)
    birth_ticks: List[int] = field(default_factory=list)
    lineage_ids: List[str] = field(default_factory=list)
    developmental_states: List[str] = field(default_factory=list)
    alive: List[bool] = field(default_factory=list)
    activity_history: List[float] = field(default_factory=list)

    @classmethod
    def initialize(cls, n: int) -> "DevelopmentState":
        return cls(
            cell_types=["interneuron"] * n,
            birth_ticks=[0] * n,
            lineage_ids=[f"founder-{i}" for i in range(n)],
            developmental_states=["mature"] * n,
            alive=[True] * n,
            activity_history=[0.0] * n,
        )

    def extend(self, n_new: int, tick: int, types: List[str], lineages: List[str]):
        for i in range(n_new):
            self.cell_types.append(types[i])
            self.birth_ticks.append(tick)
            self.lineage_ids.append(lineages[i])
            self.developmental_states.append("newborn")
            self.alive.append(True)
            self.activity_history.append(0.0)


def _adjacency(graph: ConnectomeGraph) -> Dict[int, List]:
    nrows = len(graph.row_offsets) - 1
    N = min(nrows, graph.num_neurons)
    adj = {}
    for i in range(N):
        s, e = int(graph.row_offsets[i]), int(graph.row_offsets[i + 1])
        adj[i] = list(zip([int(x) for x in graph.col_indices[s:e]],
                          [float(x) for x in graph.weights[s:e]]))
    for i in range(N, graph.num_neurons):
        adj[i] = []
    return adj


def _rebuild_csr(graph: ConnectomeGraph, adj: Dict[int, List]):
    N = graph.num_neurons
    ro, ci, w = [0], [], []
    for i in range(N):
        seen = {}
        for t, wt in adj.get(i, []):
            if t == i:
                continue
            if not (0 <= t < N):
                raise ValueError(f"Invalid synapse target {t}")
            if wt != wt or wt in (float("inf"), float("-inf")):
                raise ValueError("Invalid weight NaN/Inf")
            if t not in seen:
                seen[t] = float(np.clip(wt, 0.01, 1.0))
        for t in sorted(seen):
            ci.append(t); w.append(seen[t])
        ro.append(len(ci))
    graph.row_offsets = np.array(ro, dtype=np.int32)
    graph.col_indices = np.array(ci, dtype=np.int32)
    graph.weights = np.array(w, dtype=np.float32)
    graph.validate_invariants()
    graph.graph_hash = graph.compute_graph_hash()


class DevelopmentEngine:
    """Deterministic structural development bound to a genome parameter set."""

    def __init__(self, genome_params: Dict[str, float], development_seed: int = 45):
        self.p = dict(genome_params)
        self.seed = int(development_seed)

    def _rng(self, tick: int, stream: str) -> np.random.RandomState:
        return np.random.RandomState(derive_subseed(self.seed, f"{stream}:{tick}"))

    def neurogenesis(self, graph: ConnectomeGraph, dev: DevelopmentState, tick: int,
                     events: Optional[EventLog] = None, organism_id: str = "",
                     generation: int = 0, max_new: int = 4) -> int:
        rng = self._rng(tick, "neurogenesis")
        rate = float(self.p.get("neurogenesis_rate", 0.3))
        n_new = min(max_new, int(rng.poisson(rate * 3.0)))
        if n_new <= 0:
            return 0
        mean = graph.coordinates.mean(axis=0)
        std = np.maximum(graph.coordinates.std(axis=0) * 0.3, 1.0)
        new_coords = (rng.normal(mean, std, (n_new, 3))).astype(np.float32)
        last_id = int(np.max(graph.neuron_ids)) if len(graph.neuron_ids) else 0
        graph.neuron_ids = np.concatenate([graph.neuron_ids,
            np.array([last_id + 1 + i for i in range(n_new)], dtype=np.int64)])
        graph.coordinates = np.vstack([graph.coordinates, new_coords])
        graph.tbars = np.concatenate([graph.tbars, np.full(n_new, 50, dtype=np.int32)])
        graph.sides = list(graph.sides) + ["M"] * n_new
        # extend CSR with empty rows BEFORE building adjacency
        old_n = len(graph.row_offsets) - 1
        if old_n < graph.num_neurons:
            graph.row_offsets = np.concatenate(
                [graph.row_offsets, np.full(graph.num_neurons - old_n,
                                            graph.row_offsets[-1], dtype=np.int32)])
        adj = _adjacency(graph)
        bias = float(self.p.get("differentiation_bias", 0.5))
        types, lineages = [], []
        for i in range(n_new):
            r = rng.rand()
            if r < 0.25 * (1 - bias) + 0.1:
                t = "sensory"
            elif r < 0.5:
                t = "interneuron"
            elif r < 0.75:
                t = "motor"
            else:
                t = rng.choice(["modulatory", "memory"])
            types.append(str(t))
            parent_idx = int(rng.randint(0, old_n)) if old_n > 0 else 0
            parent_lin = dev.lineage_ids[parent_idx] if parent_idx < len(dev.lineage_ids) else "founder"
            lineages.append(hashlib.sha256(f"{parent_lin}|{tick}|{i}".encode()).hexdigest()[:12])
        dev.extend(n_new, tick, types, lineages)
        graph.validate_invariants()
        graph.graph_hash = graph.compute_graph_hash()
        if events is not None:
            for i in range(n_new):
                idx = old_n + i
                events.log("NEURON_BORN", tick, organism_id, generation,
                           {"neuron_index": idx, "cell_type": types[i],
                            "lineage": lineages[i]})
        return n_new

    def differentiate(self, graph: ConnectomeGraph, dev: DevelopmentState, tick: int,
                      events: Optional[EventLog] = None, organism_id: str = "",
                      generation: int = 0) -> int:
        rng = self._rng(tick, "differentiation")
        changed = 0
        for i in range(graph.num_neurons):
            if not dev.alive[i] or dev.developmental_states[i] not in ("newborn", "migrating"):
                continue
            if rng.rand() < 0.5 + 0.5 * float(self.p.get("developmental_timing", 0.5)):
                dev.developmental_states[i] = "differentiated"
                changed += 1
                if events is not None:
                    events.log("NEURON_DIFFERENTIATED", tick, organism_id, generation,
                               {"neuron_index": i, "cell_type": dev.cell_types[i]})
        return changed

    def migrate(self, graph: ConnectomeGraph, dev: DevelopmentState, tick: int,
                events: Optional[EventLog] = None, organism_id: str = "",
                generation: int = 0) -> int:
        rng = self._rng(tick, "migration")
        rate = float(self.p.get("migration_rate", 0.4))
        moved = 0
        span = np.maximum(graph.coordinates.max(axis=0) - graph.coordinates.min(axis=0), 1.0)
        for i in range(graph.num_neurons):
            if not dev.alive[i] or dev.developmental_states[i] not in ("differentiated", "newborn"):
                continue
            step = (rng.rand(3).astype(np.float32) - 0.5) * 2.0 * span * 0.02 * rate
            graph.coordinates[i] = (graph.coordinates[i] + step).astype(np.float32)
            dev.developmental_states[i] = "migrating"
            moved += 1
            if events is not None:
                events.log("NEURON_MOVED", tick, organism_id, generation, {"neuron_index": i})
        return moved

    def grow_projections(self, graph: ConnectomeGraph, dev: DevelopmentState, tick: int,
                         events: Optional[EventLog] = None, organism_id: str = "",
                         generation: int = 0, max_candidates: int = 6) -> int:
        """Axon+dendrite exploration -> candidate contacts -> synaptogenesis.
        CSR CONVENTION v3: axon from i to j appends source i to row j."""
        rng = self._rng(tick, "growth")
        rate = float(self.p.get("synaptogenesis_rate", 0.5))
        adj = _adjacency(graph)
        created = 0
        N = graph.num_neurons
        for i in range(N):
            if not dev.alive[i]:
                continue
            if dev.developmental_states[i] not in ("migrating", "differentiated", "mature"):
                continue
            if rng.rand() > rate:
                continue
            dists = np.linalg.norm(graph.coordinates - graph.coordinates[i], axis=1)
            order = np.argsort(dists)
            added_here = 0
            for j in order[1:]:
                if added_here >= 2 or created >= max_candidates:
                    break
                j = int(j)
                if j == i or not dev.alive[j]:
                    continue
                existing = {s for s, _ in adj[j]}
                if i in existing:
                    continue
                w = float(rng.uniform(0.05, 0.2))
                adj[j].append((i, w))
                added_here += 1
                created += 1
                if events is not None:
                    events.log("AXON_GROWN", tick, organism_id, generation,
                               {"source": i, "target": j})
                    events.log("SYNAPSE_CREATED", tick, organism_id, generation,
                               {"source": i, "target": j, "weight": round(w, 4)})
            if added_here:
                dev.developmental_states[i] = "connected"
        if created:
            _rebuild_csr(graph, adj)
        return created

    def prune(self, graph: ConnectomeGraph, dev: DevelopmentState, tick: int,
              activity: Optional[np.ndarray] = None,
              events: Optional[EventLog] = None, organism_id: str = "",
              generation: int = 0) -> int:
        thr = float(self.p.get("pruning_threshold", 0.05))
        adj = _adjacency(graph)
        pruned = 0
        for i in range(graph.num_neurons):
            kept = []
            for (t, w) in adj[i]:
                inactive = activity is not None and float(activity[i]) < 0.01 and float(activity[t]) < 0.01
                if w < thr or (inactive and w < thr * 2.0):
                    pruned += 1
                    if events is not None:
                        events.log("SYNAPSE_PRUNED", tick, organism_id, generation,
                                   {"source": i, "target": t, "weight": round(w, 4)})
                else:
                    kept.append((t, w))
            adj[i] = kept
        if pruned:
            _rebuild_csr(graph, adj)
        return pruned

    def apoptosis(self, graph: ConnectomeGraph, dev: DevelopmentState, tick: int,
                  activity: Optional[np.ndarray] = None, age: int = 0,
                  events: Optional[EventLog] = None, organism_id: str = "",
                  generation: int = 0) -> int:
        rng = self._rng(tick, "apoptosis")
        adj = _adjacency(graph)
        died = 0
        for i in range(graph.num_neurons):
            if not dev.alive[i]:
                continue
            neuron_age = tick - dev.birth_ticks[i]
            inactive = activity is not None and float(activity[i]) < 0.005
            p = 0.0
            if inactive and neuron_age > 10:
                p = 0.05
            if neuron_age > 500:
                p = max(p, 0.02)
            if rng.rand() < p:
                dev.alive[i] = False
                dev.developmental_states[i] = "dead"
                adj[i] = []
                for k in adj:
                    adj[k] = [(t, w) for (t, w) in adj[k] if t != i]
                died += 1
                if events is not None:
                    events.log("NEURON_DIED", tick, organism_id, generation,
                               {"neuron_index": i, "age": neuron_age,
                                "cause": "inactivity" if inactive else "age"})
        if died:
            _rebuild_csr(graph, adj)
        return died

    def complexity_metrics(self, graph: ConnectomeGraph, dev: DevelopmentState) -> Dict[str, Any]:
        N = graph.num_neurons
        M = graph.num_synapses
        deg = np.diff(graph.row_offsets).astype(float) if N else np.array([0.0])
        alive_n = int(sum(dev.alive)) if dev.alive else N
        types = {}
        for i, t in enumerate(dev.cell_types):
            if i < len(dev.alive) and dev.alive[i]:
                types[t] = types.get(t, 0) + 1
        return {
            "neuron_count": N,
            "alive_neurons": alive_n,
            "synapse_count": M,
            "avg_degree": round(float(np.mean(deg)) if len(deg) else 0.0, 3),
            "cell_type_diversity": len(types),
            "cell_type_counts": types,
        }
