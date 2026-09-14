"""Deterministic genome mutation + crossover (REAL, IMPLEMENTED)."""
import numpy as np
from typing import Dict, Any, Tuple
from src.genome.schema import Genome, PARAM_BOUNDS
from src.common.determinism import derive_subseed

MUTATION_CLASSES = ("parameter", "developmental", "plasticity", "metabolic", "behavioral")


def _class_of_param(name: str) -> str:
    if name in ("neurogenesis_rate", "differentiation_bias", "migration_rate",
                "axon_growth_rate", "dendrite_growth_rate", "synaptogenesis_rate",
                "pruning_threshold", "developmental_timing", "growth_budget_fraction"):
        return "developmental"
    if name in ("plasticity_rate", "memory_retention", "eligibility_decay",
                "neuromod_novelty_weight", "neuromod_prediction_weight",
                "prediction_gain"):
        return "plasticity"
    if name in ("metabolism_rate", "reproduction_threshold"):
        return "metabolic"
    return "behavioral"


def mutate_genome(parent: Genome, mutation_seed: int, rate: float = 0.3,
                  scale: float = 0.1) -> Tuple[Genome, Dict[str, Any]]:
    """Deterministic per-parameter Gaussian mutation. Returns (child, record)."""
    parent.validate()
    parent_hash = parent.genome_hash()
    rng = np.random.RandomState(int(mutation_seed) % (2 ** 31))
    child_params = dict(parent.params)
    mutations = []
    for key in sorted(child_params):
        if rng.rand() < rate:
            lo, hi = PARAM_BOUNDS[key]
            old = float(child_params[key])
            new = float(np.clip(old + rng.normal(0.0, scale * (hi - lo)), lo, hi))
            child_params[key] = new
            mutations.append({"param": key, "class": _class_of_param(key),
                              "old": round(old, 6), "new": round(new, 6)})
    child = Genome(params=child_params, version=parent.version,
                   lineage={"parent_hash": parent_hash, "mutation_seed": int(mutation_seed)})
    child.validate()
    record = {
        "parent_genome_hash": parent_hash,
        "mutation_seed": int(mutation_seed),
        "mutation_type": "parameter_mutation",
        "mutations": mutations,
        "offspring_genome_hash": child.genome_hash(),
    }
    return child, record


def crossover_genomes(a: Genome, b: Genome, seed: int, mode: str = "sexual") -> Tuple[Genome, Dict[str, Any]]:
    """Deterministic uniform crossover + light mutation. Asexual returns clone+mutation."""
    a.validate(); b.validate()
    ha, hb = a.genome_hash(), b.genome_hash()
    if mode == "asexual":
        child, rec = mutate_genome(a, derive_subseed(seed, "asexual"), rate=0.2)
        rec.update({"mode": "asexual", "parent_a": ha, "parent_b": None})
        child.lineage.update({"crossover": "asexual", "parents": [ha]})
        return child, rec
    if mode != "sexual":
        raise ValueError(f"Unknown crossover mode: {mode}")
    rng = np.random.RandomState(int(seed) % (2 ** 31))
    # Version policy: same versions -> child keeps that version; mixed
    # v1 x v2 -> child is v2 (architecture genes present via the v2 parent).
    child_version = a.version if a.version == b.version else "2.0"
    keys = sorted(set(a.params) | set(b.params))
    child_params = {}
    choices = {}
    for key in keys:
        take_a = bool(rng.rand() < 0.5)
        primary, other = (a, b) if take_a else (b, a)
        # mixed-version safety: fall back to the parent that has the gene
        val = primary.params.get(key, other.params.get(key))
        if val is None:
            from src.genome.schema import ARCH_PARAMS
            val = ARCH_PARAMS[key]
        child_params[key] = float(val)
        choices[key] = "A" if take_a else "B"
    child = Genome(params=child_params, version=child_version,
                   lineage={"parents": [ha, hb], "crossover_seed": int(seed)})
    # post-crossover light mutation for variation
    child, mut_rec = mutate_genome(child, derive_subseed(seed, "post-xover"), rate=0.1)
    child.lineage.update({"parents": [ha, hb], "crossover_seed": int(seed)})
    rec = {"mode": "sexual", "parent_a": ha, "parent_b": hb,
           "choices": choices, "post_mutations": mut_rec["mutations"],
           "offspring_genome_hash": child.genome_hash(), "crossover_seed": int(seed)}
    return child, rec
