"""Layered provenance hash architecture (REAL, IMPLEMENTED).

Separates hashes by semantic layer so a changed child can never hide behind an
unchanged parent: each layer hash is folded into the experiment fingerprint.
"""
import hashlib
import json
import os
from typing import Any, Dict, List, Optional


def _h(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode() if not isinstance(p, bytes) else p)
        h.update(b"|")
    return h.hexdigest()


def dataset_hash(soma_path: str, connections_path: str) -> str:
    from src.connectome.loader import _file_sha256
    return _h("DATASET", _file_sha256(soma_path), _file_sha256(connections_path))


def model_hash(model_path: Optional[str]) -> str:
    from src.llm.discovery import sha256_file
    if not model_path or not os.path.exists(model_path):
        return _h("MODEL", "unavailable")
    return _h("MODEL", sha256_file(model_path))


def graph_hash(graph) -> str:
    return _h("GRAPH", graph.graph_hash)


def brain_state_hash(state) -> str:
    import numpy as np
    return _h("BRAIN",
              np.ascontiguousarray(state.membrane_potentials).tobytes(),
              np.ascontiguousarray(state.spikes).tobytes(),
              np.ascontiguousarray(state.refractory_steps).tobytes(),
              state.step_count)


def world_state_hash(world) -> str:
    return _h("WORLD", world.world_hash())


def organism_hash(organism) -> str:
    return _h("ORGANISM", organism.organism_hash())


def population_hash(pop) -> str:
    return _h("POPULATION", pop.population_hash())


def event_log_hash(event_log) -> str:
    return _h("EVENTS", event_log.compute_hash())


def experiment_fingerprint(layers: Dict[str, str]) -> str:
    """Top-level provenance fingerprint over all provided layer hashes."""
    ordered: List[str] = [f"{k}={layers[k]}" for k in sorted(layers)]
    return _h("EXPERIMENT", *ordered)


def artifact_hash(path: str) -> str:
    from src.connectome.loader import _file_sha256
    return _h("ARTIFACT", os.path.basename(path), _file_sha256(path)
              if os.path.exists(path) else "missing")


def build_layers(graph=None, state=None, world=None, population=None,
                 event_log=None, dataset=None, model_path=None) -> Dict[str, str]:
    layers: Dict[str, Any] = {}
    if dataset:
        layers["dataset"] = dataset_hash(dataset[0], dataset[1])
    if model_path is not None:
        layers["model"] = model_hash(model_path)
    if graph is not None:
        layers["graph"] = graph_hash(graph)
    if state is not None:
        layers["brain_state"] = brain_state_hash(state)
    if world is not None:
        layers["world_state"] = world_state_hash(world)
    if population is not None:
        layers["population"] = population_hash(population)
    if event_log is not None:
        layers["event_log"] = event_log_hash(event_log)
    return layers
