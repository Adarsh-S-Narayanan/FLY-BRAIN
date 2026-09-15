"""Provenance V4 (PHASE C): explicit provenance classes, immutable biological
baseline, stable synapse identities, layered full-brain identity.

NON-HALLUCINATION CONTRACT (mission §3/§9/§10/§12):
- The biological baseline (MaleCNS empirical measurements) is IMMUTABLE:
  plasticity, development, growth, evolution and rewiring may never rewrite it.
- Synapse identities are stable; new developmental/evolutionary synapses are
  EMERGENT or EVOLVED — never BIOLOGICAL unless they directly correspond to an
  immutable empirical source record.
- Full brain identity is a layered hash: same complete state == same identity;
  meaningfully different state == different identity (tested per component).
"""
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ProvenanceClassV4(str, Enum):
    BIOLOGICAL = "BIOLOGICAL"
    DERIVED = "DERIVED"
    HEURISTIC = "HEURISTIC"
    EMERGENT = "EMERGENT"
    EVOLVED = "EVOLVED"
    SYNTHETIC = "SYNTHETIC"
    UNKNOWN = "UNKNOWN"


# Graph state model (mission §11)
GRAPH_STATES = ("BIOLOGICAL_BASELINE", "BIOLOGICAL_DERIVED", "LIVING_EMERGENT",
                "EVOLVED", "SYNTHETIC")

STATE_BY_SEED = {
    "BIOLOGICAL": "BIOLOGICAL_BASELINE",
    "DERIVED": "BIOLOGICAL_DERIVED",
    "SYNTHETIC": "SYNTHETIC",
}


# ------------------------------------------------------------------ baseline
@dataclass(frozen=True)
class BiologicalSynapse:
    """Immutable empirical measurement (never rewritten by the simulation)."""
    source_id: int
    target_id: int
    original_synapse_count: int
    dataset_record_id: str
    biological_hash: str

    @staticmethod
    def make(source_id: int, target_id: int, count: int, dataset: str = "malecns",
             dataset_record_id: str = "") -> "BiologicalSynapse":
        payload = f"{dataset}|{source_id}|{target_id}|{count}"
        return BiologicalSynapse(
            source_id=source_id, target_id=target_id,
            original_synapse_count=count,
            dataset_record_id=dataset_record_id or f"{dataset}:{source_id}->{target_id}",
            biological_hash=hashlib.sha256(payload.encode()).hexdigest())


@dataclass
class BiologicalBaseline:
    """The immutable biological source of a living brain.

    snapshot() content is FROZEN at seed time; no simulation operation may
    modify it. `verify_untouched()` compares against the seed fingerprint.
    """
    dataset_name: str
    synapses: Tuple[BiologicalSynapse, ...]
    neuron_measurements: Dict[int, Dict[str, Any]]  # body_id -> empirical fields
    seed_fingerprint: str = ""

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.dataset_name.encode())
        for s in sorted(self.synapses, key=lambda x: (x.source_id, x.target_id)):
            h.update(s.biological_hash.encode())
        for nid in sorted(self.neuron_measurements):
            h.update(json.dumps({str(nid): self.neuron_measurements[nid]},
                                sort_keys=True).encode())
        return h.hexdigest()

    def seal(self) -> str:
        self.seed_fingerprint = self.fingerprint()
        return self.seed_fingerprint

    def verify_untouched(self) -> bool:
        if not self.seed_fingerprint:
            return False
        return self.fingerprint() == self.seed_fingerprint

    def snapshot(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "seed_fingerprint": self.seed_fingerprint,
            "synapses": [s.__dict__ for s in self.synapses],
            "neuron_measurements": self.neuron_measurements,
        }

    @classmethod
    def restore(cls, payload: Dict[str, Any]) -> "BiologicalBaseline":
        syns = tuple(BiologicalSynapse(**d) for d in payload["synapses"])
        bl = cls(payload["dataset_name"], syns, payload["neuron_measurements"],
                 payload.get("seed_fingerprint", ""))
        return bl


def build_biological_baseline(graph) -> BiologicalBaseline:
    """Freeze the current graph's edges as biological measurements.

    For GraphMode.REAL every edge corresponds to an empirical MaleCNS record;
    for derived/synthetic modes the baseline records what the seed provided
    (never labeled BIOLOGICAL unless the graph mode is REAL).
    """
    from src.connectome.types import GraphMode
    dataset = ("malecns" if graph.mode == GraphMode.REAL
               else f"seed:{graph.mode.value.lower()}")
    syns = []
    for row in range(graph.num_neurons):
        for k in range(int(graph.row_offsets[row]), int(graph.row_offsets[row + 1])):
            pre = int(graph.col_indices[k])
            src = int(graph.neuron_ids[pre])
            tgt = int(graph.neuron_ids[row])
            count = max(1, int(round(float(graph.weights[k]) * 20.0))) \
                if graph.mode == GraphMode.REAL else 1
            syns.append(BiologicalSynapse.make(
                src, tgt, count, dataset=dataset,
                dataset_record_id=f"{dataset}:{src}->{tgt}"))
    neuron_meas = {
        int(graph.neuron_ids[i]): {
            "position": [round(float(x), 3) for x in graph.coordinates[i]],
            "tbars": int(graph.tbars[i]),
            "side": graph.sides[i],
        }
        for i in range(graph.num_neurons)
    }
    bl = BiologicalBaseline(dataset_name=dataset, synapses=tuple(syns),
                            neuron_measurements=neuron_meas)
    bl.seal()
    return bl


# ----------------------------------------------------------------- identity
def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, default=str).encode()


class BrainIdentity:
    """Layered, canonical, deterministic full-brain identity (mission §12)."""

    LAYERS = ("topology", "geometry", "neuron_metadata", "synapse_provenance",
              "weights", "physiology", "development", "genome", "memory",
              "brain_state", "population", "event_stream")

    def __init__(self, layers: Dict[str, str]):
        self.layers = {k: str(v) for k, v in layers.items() if k in self.LAYERS}

    @classmethod
    def from_components(cls, graph, brain_state=None, genome=None,
                        memory=None, population_hash: str = "",
                        event_stream_hash: str = "",
                        synapse_provenance_hash: str = "",
                        development_hash: str = "") -> "BrainIdentity":
        layers: Dict[str, str] = {}
        layers["topology"] = cls._topology(graph)
        layers["geometry"] = cls._geometry(graph)
        layers["neuron_metadata"] = cls._neuron_metadata(graph)
        layers["synapse_provenance"] = synapse_provenance_hash or "none"
        layers["weights"] = hashlib.sha256(
            graph.weights.tobytes()).hexdigest()
        if brain_state is not None:
            layers["physiology"] = hashlib.sha256(_canon({
                "drives": {"energy": getattr(brain_state.drives, "energy", None),
                           "curiosity": getattr(brain_state.drives, "curiosity", None),
                           "social": getattr(brain_state.drives, "social", None),
                           "integrity": getattr(brain_state.drives, "integrity", None)},
                "predicted_reward": getattr(brain_state, "predicted_reward", None),
                "prediction_error": getattr(brain_state, "prediction_error", None),
                "step_count": getattr(brain_state, "step_count", None),
            })).hexdigest()
        else:
            layers["physiology"] = "none"
        layers["development"] = development_hash or "none"
        layers["genome"] = (genome.genome_hash() if genome is not None
                            and hasattr(genome, "genome_hash") else "none")
        layers["memory"] = (memory if isinstance(memory, str) else "none")
        layers["brain_state"] = hashlib.sha256(
            brain_state.membrane_potentials.tobytes()
            + brain_state.spikes.tobytes()
            + brain_state.refractory_steps.tobytes()).hexdigest() \
            if brain_state is not None else "none"
        layers["population"] = population_hash or "none"
        layers["event_stream"] = event_stream_hash or "none"
        return cls(layers)

    @staticmethod
    def _topology(graph) -> str:
        return hashlib.sha256(
            graph.row_offsets.tobytes() + graph.col_indices.tobytes()
            + graph.neuron_ids.tobytes()).hexdigest()

    @staticmethod
    def _geometry(graph) -> str:
        return hashlib.sha256(graph.coordinates.tobytes()).hexdigest()

    @staticmethod
    def _neuron_metadata(graph) -> str:
        return hashlib.sha256(_canon({
            "tbars": graph.tbars.tolist(),
            "sides": list(graph.sides),
        })).hexdigest()

    def full_identity(self) -> str:
        h = hashlib.sha256()
        for k in self.LAYERS:
            h.update(f"{k}={self.layers.get(k, 'none')}".encode())
        return h.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {"layers": dict(self.layers),
                "full_identity": self.full_identity()}
