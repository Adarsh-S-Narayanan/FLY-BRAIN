"""LivingBrain: versioned living brain model (STAGE B/C).

Wraps a ConnectomeGraph + DevelopmentState with:
- persistent neuron identities (never reused) and persistent synapse identities;
- explicit provenance classes per element:
    BIOLOGICAL  empirical MaleCNS seed elements (GraphMode.REAL)
    DERIVED     transforms of biological data (GraphMode.SPATIAL_SURROGATE)
    EMERGENT    structures created during organism lifetime
    EVOLVED     architectures inherited across generations (assigned by lineage)
    SYNTHETIC   explicitly synthetic experimental structures (SYNTHETIC_TEST)
- a structural event log (reuses the 28 validated event types);
- resource-constrained development orchestration (growth must be paid for);
- versioned snapshot/restore + structural validation.

The biological seed graph remains immutable from the experiment's perspective:
any modification happens on the LivingBrain's own working copy and expansions
are never relabeled as empirical MaleCNS elements.
"""
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.determinism import derive_subseed
from src.common.events import EventLog
from src.connectome.types import ConnectomeGraph, GraphMode
from src.development.engine import DevelopmentEngine, DevelopmentState

SCHEMA_VERSION = "living_brain_v1"


class ProvenanceClass(str, Enum):
    BIOLOGICAL = "BIOLOGICAL"
    DERIVED = "DERIVED"
    EMERGENT = "EMERGENT"
    EVOLVED = "EVOLVED"
    SYNTHETIC = "SYNTHETIC"


SEED_CLASS_BY_MODE = {
    GraphMode.REAL: ProvenanceClass.BIOLOGICAL,
    GraphMode.SPATIAL_SURROGATE: ProvenanceClass.DERIVED,
    GraphMode.SYNTHETIC_TEST: ProvenanceClass.SYNTHETIC,
}


def _stable_id(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


@dataclass
class NeuronRecord:
    neuron_id: int            # persistent; never reused
    index: int                # current array index
    provenance_class: str
    birth_tick: int
    birth_op: str
    parent_lineage: str
    cell_type: str
    generation: int = 0
    death_tick: Optional[int] = None
    death_cause: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SynapseRecord:
    synapse_id: str           # persistent; never reused
    pre_id: int
    post_id: int
    provenance_class: str
    birth_tick: int
    birth_op: str
    weight_at_birth: float
    death_tick: Optional[int] = None
    death_cause: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class LivingBrain:
    """Versioned living brain: identity registry + provenance + development."""
    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(self, graph: ConnectomeGraph, dev: Optional[DevelopmentState] = None,
                 engine: Optional[DevelopmentEngine] = None,
                 experiment_seed: int = 42,
                 structural_events: Optional[EventLog] = None):
        self.graph = graph
        self.dev = dev if dev is not None else DevelopmentState.initialize(graph.num_neurons)
        self.engine = engine if engine is not None else DevelopmentEngine({}, experiment_seed)
        self.experiment_seed = int(experiment_seed)
        self.seed_class = SEED_CLASS_BY_MODE[graph.mode]
        self.structural_events = structural_events if structural_events is not None else EventLog()
        self._neurons: Dict[int, NeuronRecord] = {}
        self._synapses: Dict[str, SynapseRecord] = {}       # synapse_id -> record
        self._edge_index: Dict[Tuple[int, int], str] = {}   # (pre_id, post_id) -> synapse_id
        self._op_counter = 0
        self._reconcile_seed(tick=0)

    # ------------------------------------------------------------------ seed
    def _reconcile_seed(self, tick: int) -> None:
        for i in range(self.graph.num_neurons):
            nid = int(self.graph.neuron_ids[i])
            lineage = self.dev.lineage_ids[i] if i < len(self.dev.lineage_ids) else f"founder-{i}"
            self._neurons[nid] = NeuronRecord(
                neuron_id=nid, index=i, provenance_class=self.seed_class.value,
                birth_tick=tick, birth_op="seed", parent_lineage=lineage,
                cell_type=self.dev.cell_types[i] if i < len(self.dev.cell_types) else "interneuron")
        for (pre_i, post_i), w in self._edge_set().items():
            self._register_synapse(pre_i, post_i, w, tick, "seed", self.seed_class)

    # -------------------------------------------------------------- helpers
    def _edge_set(self) -> Dict[Tuple[int, int], float]:
        """Edges as {(pre_index, post_index): weight} from incoming-row CSR."""
        out: Dict[Tuple[int, int], float] = {}
        ro, ci, w = self.graph.row_offsets, self.graph.col_indices, self.graph.weights
        for row in range(self.graph.num_neurons):
            for k in range(int(ro[row]), int(ro[row + 1])):
                out[(int(ci[k]), row)] = float(w[k])
        return out

    def _register_synapse(self, pre_i: int, post_i: int, weight: float, tick: int,
                          op: str, pclass: ProvenanceClass) -> str:
        pre_id, post_id = int(self.graph.neuron_ids[pre_i]), int(self.graph.neuron_ids[post_i])
        self._op_counter += 1
        sid = _stable_id("syn", pre_id, post_id, tick, self._op_counter)
        self._synapses[sid] = SynapseRecord(
            synapse_id=sid, pre_id=pre_id, post_id=post_id,
            provenance_class=pclass.value, birth_tick=tick, birth_op=op,
            weight_at_birth=float(weight))
        self._edge_index[(pre_id, post_id)] = sid
        return sid

    def _reconcile(self, tick: int, op: str) -> Dict[str, int]:
        """Update registries after a development-engine mutation of the graph."""
        changes = {"neurons_born": 0, "neurons_died": 0,
                   "synapses_created": 0, "synapses_removed": 0}
        # new neurons appended -> EMERGENT (or EVOLVED when lineage says evolved)
        for i in range(self.graph.num_neurons):
            nid = int(self.graph.neuron_ids[i])
            if nid not in self._neurons:
                lineage = (self.dev.lineage_ids[i] if i < len(self.dev.lineage_ids)
                           else f"neuron-{nid}")
                pclass = (ProvenanceClass.EVOLVED
                          if lineage.startswith("evolved-") else ProvenanceClass.EMERGENT)
                self._neurons[nid] = NeuronRecord(
                    neuron_id=nid, index=i, provenance_class=pclass.value,
                    birth_tick=tick, birth_op=op, parent_lineage=lineage,
                    cell_type=(self.dev.cell_types[i] if i < len(self.dev.cell_types)
                               else "interneuron"),
                    generation=0)
                changes["neurons_born"] += 1
                self.structural_events.log("NEURON_BORN", tick, payload={
                    "neuron_id": nid, "provenance_class": pclass.value, "op": op})
            else:
                self._neurons[nid].index = i
        # deaths
        for i in range(min(len(self.dev.alive), self.graph.num_neurons)):
            if not self.dev.alive[i]:
                nid = int(self.graph.neuron_ids[i])
                rec = self._neurons.get(nid)
                if rec is not None and rec.death_tick is None:
                    rec.death_tick = tick
                    rec.death_cause = "development"
                    rec.cell_type = (self.dev.cell_types[i]
                                     if i < len(self.dev.cell_types) else rec.cell_type)
                    changes["neurons_died"] += 1
                    self.structural_events.log("NEURON_DIED", tick, payload={
                        "neuron_id": nid, "cause": "development", "op": op})
        # synapses
        current = self._edge_set()
        current_keys = {(int(self.graph.neuron_ids[p]), int(self.graph.neuron_ids[q]))
                        for (p, q) in current}
        # removals
        for (pre_id, post_id) in list(self._edge_index.keys()):
            if (pre_id, post_id) not in current_keys:
                sid = self._edge_index.pop((pre_id, post_id))
                rec = self._synapses.get(sid)
                if rec is not None and rec.death_tick is None:
                    rec.death_tick = tick
                    rec.death_cause = op
                    changes["synapses_removed"] += 1
                    self.structural_events.log("SYNAPSE_PRUNED", tick, payload={
                        "synapse_id": sid, "op": op})
        # additions
        for (pre_i, post_i), w in current.items():
            pre_id, post_id = int(self.graph.neuron_ids[pre_i]), int(self.graph.neuron_ids[post_i])
            if (pre_id, post_id) not in self._edge_index:
                self._register_synapse(pre_i, post_i, w, tick, op, ProvenanceClass.EMERGENT)
                changes["synapses_created"] += 1
                self.structural_events.log("SYNAPSE_CREATED", tick, payload={
                    "pre_id": pre_id, "post_id": post_id, "weight": round(float(w), 6),
                    "provenance_class": ProvenanceClass.EMERGENT.value, "op": op})
        return changes

    def _fit_activity(self, activity: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Pad/trim activity history to the CURRENT graph size (neurogenesis may
        have grown the graph earlier in this same cycle)."""
        if activity is None:
            return None
        n = self.graph.num_neurons
        a = np.asarray(activity, dtype=np.float32)
        if len(a) < n:
            a = np.concatenate([a, np.zeros(n - len(a), dtype=np.float32)])
        return a[:n]

    # ----------------------------------------------------- development API
    def run_development_cycle(self, tick: int, activity: Optional[np.ndarray] = None,
                              organism_id: str = "", generation: int = 0,
                              growth_budget: int = 2,
                              energy_cost_per_neuron: float = 0.02) -> Dict[str, Any]:
        """Resource-constrained structural development cycle.

        growth_budget is the maximum number of NEW neurons this cycle may create
        (paid by the caller in energy). Returns a summary of all changes.
        """
        summary: Dict[str, Any] = {"tick": tick, "ops": []}

        def _run(op_name: str, fn, *args, **kwargs):
            before_neurons = self.graph.num_neurons
            result = fn(*args, **kwargs)
            ch = self._reconcile(tick, op_name)
            ch["op"] = op_name
            ch["result"] = result
            summary["ops"].append(ch)
            return result

        _run("neurogenesis", self.engine.neurogenesis, self.graph, self.dev, tick,
             events=self.structural_events, organism_id=organism_id, generation=generation,
             max_new=max(0, int(growth_budget)))
        _run("differentiate", self.engine.differentiate, self.graph, self.dev, tick,
             events=self.structural_events, organism_id=organism_id, generation=generation)
        _run("migrate", self.engine.migrate, self.graph, self.dev, tick,
             events=self.structural_events, organism_id=organism_id, generation=generation)
        _run("grow_projections", self.engine.grow_projections, self.graph, self.dev, tick,
             events=self.structural_events, organism_id=organism_id, generation=generation)
        _run("prune", self.engine.prune, self.graph, self.dev, tick,
             activity=self._fit_activity(activity),
             events=self.structural_events, organism_id=organism_id, generation=generation)
        _run("apoptosis", self.engine.apoptosis, self.graph, self.dev, tick,
             activity=self._fit_activity(activity), age=tick, events=self.structural_events,
             organism_id=organism_id, generation=generation)
        summary["energy_spent"] = round(summary["ops"][0]["result"] * energy_cost_per_neuron, 6)
        summary["neurons"] = self.graph.num_neurons
        summary["synapses"] = self.graph.num_synapses
        self.validate()
        return summary

    # ------------------------------------------------------------- metrics
    @property
    def seed_size(self) -> int:
        return sum(1 for r in self._neurons.values() if r.birth_op == "seed")

    def counts_by_class(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self._neurons.values():
            out[r.provenance_class] = out.get(r.provenance_class, 0) + 1
        return out

    def synapse_counts_by_class(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self._synapses.values():
            out[r.provenance_class] = out.get(r.provenance_class, 0) + 1
        return out

    @property
    def expansion_ratio(self) -> float:
        return self.graph.num_neurons / max(1, self.seed_size)

    def provenance_chain(self) -> List[str]:
        chain = [f"{self.seed_class.value}_SEED"]
        counts = self.counts_by_class()
        for cls in (ProvenanceClass.DERIVED.value, ProvenanceClass.EMERGENT.value,
                    ProvenanceClass.EVOLVED.value):
            if counts.get(cls):
                chain.append(cls)
        return chain

    def structural_summary(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "neurons_total": self.graph.num_neurons,
            "neurons_alive": int(np.sum(self.dev.alive)) if len(self.dev.alive) else 0,
            "seed_size": self.seed_size,
            "expansion_ratio": round(self.expansion_ratio, 4),
            "exceeds_seed": self.graph.num_neurons > self.seed_size,
            "neuron_classes": self.counts_by_class(),
            "synapse_classes": self.synapse_counts_by_class(),
            "synapses_active": self.graph.num_synapses,
            "provenance_chain": self.provenance_chain(),
            "structural_events": len(self.structural_events),
        }

    # ------------------------------------------------------------ identity
    def brain_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.graph.graph_hash.encode())
        for nid in sorted(self._neurons):
            r = self._neurons[nid]
            h.update(f"{nid}|{r.provenance_class}|{r.birth_tick}|{r.death_tick}|".encode())
        for sid in sorted(self._synapses):
            r = self._synapses[sid]
            h.update(f"{sid}|{r.birth_tick}|{r.death_tick}|".encode())
        return h.hexdigest()

    def validate(self) -> None:
        if len(self._neurons) != self.graph.num_neurons:
            raise ValueError(f"registry size {len(self._neurons)} != graph size "
                             f"{self.graph.num_neurons}")
        ids = [int(x) for x in self.graph.neuron_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate persistent neuron ids in graph")
        # every edge must have exactly one synapse record; every record's edge must exist
        edges = self._edge_set()
        current = {(int(self.graph.neuron_ids[p]), int(self.graph.neuron_ids[q]))
                   for (p, q) in edges}
        mapped = set(self._edge_index.keys())
        if mapped != current:
            raise ValueError(f"synapse registry mismatch: {len(mapped ^ current)} edges differ")
        for r in self._synapses.values():
            if r.death_tick is not None and r.death_tick < r.birth_tick:
                raise ValueError(f"synapse {r.synapse_id}: death before birth")
        for nid, rec in self._neurons.items():
            if rec.index < 0 or rec.index >= self.graph.num_neurons:
                raise ValueError(f"neuron {nid}: index {rec.index} out of range")
            if int(self.graph.neuron_ids[rec.index]) != nid:
                raise ValueError(f"neuron {nid}: index misalignment at {rec.index}")

    # ---------------------------------------------------------- persistence
    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "experiment_seed": self.experiment_seed,
            "neurons": [r.to_dict() for r in self._neurons.values()],
            "synapses": [r.to_dict() for r in self._synapses.values()],
            "edge_index": {f"{p}|{q}": s for (p, q), s in self._edge_index.items()},
            "op_counter": self._op_counter,
            # NOTE: structural_events are intentionally NOT persisted here (same
            # policy as Organism snapshots); the registry is the provenance source
            # of truth and is fully persisted.
        }

    @classmethod
    def attach(cls, graph: ConnectomeGraph, dev: DevelopmentState, engine: DevelopmentEngine,
               experiment_seed: int, payload: Optional[Dict[str, Any]] = None,
               structural_events: Optional[EventLog] = None) -> "LivingBrain":
        """Attach to a graph; restore registries from payload, or migrate by
        rebuilding them (old checkpoints without living_brain payload)."""
        lb = cls(graph, dev=dev, engine=engine, experiment_seed=experiment_seed,
                 structural_events=structural_events)
        if payload is None:
            return lb  # migration path: seed reconcile only
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported living_brain schema: "
                             f"{payload.get('schema_version')}")
        lb._op_counter = int(payload.get("op_counter", 0))
        lb._neurons.clear()
        lb._synapses.clear()
        lb._edge_index.clear()
        for d in payload.get("neurons", []):
            rec = NeuronRecord(**d)
            lb._neurons[rec.neuron_id] = rec
        for d in payload.get("synapses", []):
            rec = SynapseRecord(**d)
            lb._synapses[rec.synapse_id] = rec
        for k, sid in payload.get("edge_index", {}).items():
            p, q = k.split("|")
            lb._edge_index[(int(p), int(q))] = sid
        lb.validate()
        return lb
