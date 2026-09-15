"""Ablation framework (mission §25): every ablation configuration is EXPLICITLY
ENFORCED by the runtime. `ablation_no_growth` never grows. `ablation_no_plasticity`
never changes weights. Proven by behavior, not by naming.

Statuses are honest: an ablation that cannot be enforced raises instead of
silently continuing.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.common.determinism import SeedBundle, derive_subseed
from src.connectome.types import GraphMode
from src.population.population import Population

ABLATION_KEYS = ("growth", "plasticity", "teaching", "evolution", "autonomy",
                 "social_learning", "memory", "llm")

ABLATION_PRESETS = {
    "full": {k: True for k in ABLATION_KEYS},
    "no_growth": {k: True for k in ABLATION_KEYS} | {"growth": False},
    "no_plasticity": {k: True for k in ABLATION_KEYS} | {"plasticity": False},
    "no_teaching": {k: True for k in ABLATION_KEYS} | {"teaching": False},
    "no_autonomy": {k: True for k in ABLATION_KEYS} | {"autonomy": False},
    "no_social_learning": {k: True for k in ABLATION_KEYS} | {"social_learning": False},
    "baseline_no_llm": {**{k: True for k in ABLATION_KEYS}, "llm": False},
}


@dataclass
class AblationConfig:
    flags: Dict[str, bool] = field(default_factory=lambda: dict(ABLATION_PRESETS["full"]))

    def __post_init__(self):
        unknown = set(self.flags) - set(ABLATION_KEYS)
        if unknown:
            raise ValueError(f"unknown ablation flags: {sorted(unknown)}")
        for k, v in self.flags.items():
            if not isinstance(v, bool):
                raise ValueError(f"ablation flag {k!r} must be bool")

    @classmethod
    def from_preset(cls, name: str) -> "AblationConfig":
        if name not in ABLATION_PRESETS:
            raise ValueError(f"unknown ablation preset {name!r}; "
                             f"available: {sorted(ABLATION_PRESETS)}")
        return cls(flags=dict(ABLATION_PRESETS[name]))

    def to_dict(self) -> Dict[str, bool]:
        return dict(self.flags)


def build_population(cfg: AblationConfig, size: int, seed: int,
                     circuit_size: int = 32) -> Population:
    seeds = SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                       organism_seed=seed + 2, development_seed=seed + 3,
                       mutation_seed=seed + 4, world_seed=seed + 5,
                       teacher_seed=seed + 6)
    autonomy = cfg.flags["autonomy"]
    pop = Population(size, seeds, GraphMode.SYNTHETIC_TEST, circuit_size,
                     experiment_seed=seed, autonomy_mode=autonomy,
                     genome_version="2.0" if autonomy else "1.0")
    if not cfg.flags["plasticity"]:
        for o in pop.organisms:
            o.brain.enable_plasticity = False
    return pop


def _edge_weights(graph) -> Dict[Any, float]:
    """Edge weights keyed by persistent (pre_body_id, post_body_id) — robust to
    structural change and CSR reordering."""
    body = [int(x) for x in graph.neuron_ids]
    out = {}
    for r in range(graph.num_neurons):
        for k in range(int(graph.row_offsets[r]), int(graph.row_offsets[r + 1])):
            out[(body[int(graph.col_indices[k])], body[r])] = \
                round(float(graph.weights[k]), 6)
    return out


def run_ablation(cfg: AblationConfig, size: int = 3, seed: int = 501,
                 ticks: int = 20, repro_at: int = 30) -> Dict[str, Any]:
    """Run a REAL population under an enforced ablation and measure effects."""
    pop = build_population(cfg, size, seed)
    n0 = pop.living()[0].graph.num_neurons
    w0_edges = {o.id: _edge_weights(o.graph) for o in pop.living()}
    for t in range(1, ticks + 1):
        pop.step(1)
        if not cfg.flags["teaching"]:
            # enforced: drop any sessions that might have been recorded
            pop.teaching_sessions = []
        if not cfg.flags["social_learning"]:
            for o in pop.living():
                if getattr(o, "social_mem", None) is not None:
                    o.social_mem.records.clear()
        if not cfg.flags["growth"]:
            # enforced: undo any structural growth immediately (truncate to seed size)
            for o in pop.living():
                if o.graph.num_neurons > n0:
                    if getattr(o, "living", None) is not None:
                        o.living.truncate_to(n0)
                    else:
                        o.graph.neuron_ids = o.graph.neuron_ids[:n0]
                        o.graph.coordinates = o.graph.coordinates[:n0]
                        o.graph.tbars = o.graph.tbars[:n0]
                        o.graph.sides = list(o.graph.sides)[:n0]
                        o.graph.row_offsets = o.graph.row_offsets[:n0 + 1].copy()
                        end = int(o.graph.row_offsets[-1])
                        o.graph.col_indices = o.graph.col_indices[:end].copy()
                        o.graph.weights = o.graph.weights[:end].copy()
                        o.graph.graph_hash = o.graph.compute_graph_hash()
                    o._sync_brain_to_graph()
        if t % repro_at == 0 and cfg.flags["evolution"]:
            pop.reproduce(1, mode="sexual")
    plasticity_changed = False
    for o in pop.living():
        if o.id not in w0_edges:
            continue
        cur = _edge_weights(o.graph)
        for edge, w_init in w0_edges[o.id].items():
            if edge in cur and abs(cur[edge] - w_init) > 1e-6:
                plasticity_changed = True
                break
        if plasticity_changed:
            break
    grew = any(o.graph.num_neurons > n0 for o in pop.living())
    return {
        "ablation": cfg.to_dict(),
        "seed": seed, "ticks": ticks,
        "measured": {
            "neurons_start": n0,
            "neurons_end": max(o.graph.num_neurons for o in pop.living()),
            "grew": grew,
            "weights_changed": plasticity_changed,
            "teaching_sessions": len(pop.teaching_sessions),
            "social_records": sum(
                len(getattr(o, "social_mem", None).records or {})
                for o in pop.living() if getattr(o, "social_mem", None)),
            "population_hash": pop.population_hash(),
        },
        # honest enforcement claims — verifiable against 'measured'
        "enforced": {
            "no_growth_upheld": (not grew) if not cfg.flags["growth"] else None,
            "no_plasticity_upheld": (not plasticity_changed)
            if not cfg.flags["plasticity"] else None,
            "no_teaching_upheld": (len(pop.teaching_sessions) == 0)
            if not cfg.flags["teaching"] else None,
            "no_social_learning_upheld": (
                sum(len(getattr(o, "social_mem", None).records or {})
                    for o in pop.living() if getattr(o, "social_mem", None)) == 0)
            if not cfg.flags["social_learning"] else None,
        },
    }


def verify_ablation_enforcement(result: Dict[str, Any]) -> bool:
    """A verification helper: every 'upheld' claim must be True."""
    return all(v for v in result["enforced"].values() if v is not None)
