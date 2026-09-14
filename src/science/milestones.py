"""Scientific milestones (STAGE M): automatic detection of meaningful events
with evidence requirements. Every milestone record carries observable data,
state hashes and resolution — never a bare label."""
import hashlib
import time
from typing import Any, Dict, List, Optional


def _hash_state(obj_hash: str) -> str:
    return hashlib.sha256(str(obj_hash).encode()).hexdigest()[:16]


def detect_milestones(pop) -> List[Dict[str, Any]]:
    """Evaluate milestone predicates against REAL population state.
    Returns records for milestones whose evidence holds in this state."""
    out: List[Dict[str, Any]] = []
    living = pop.living()
    total = len(pop.organisms)

    # M1: structural brain expansion beyond biological seed
    for o in living:
        if hasattr(o, "living") and o.living is not None and \
                o.living.structural_summary()["exceeds_seed"]:
            s = o.living.structural_summary()
            out.append({
                "milestone": "STRUCTURAL_EXPANSION",
                "evidence": {"organism_id": o.id,
                             "seed_size": s["seed_size"],
                             "current_size": s["neurons_total"],
                             "expansion_ratio": s["expansion_ratio"],
                             "provenance_chain": s["provenance_chain"]},
                "brain_hash": _hash_state(o.living.brain_hash()),
            })
            break  # one record per capture

    # M2: cultural transmission with measured gain
    gains = [s.learning_gain for s in pop.teaching_sessions if s.learning_gain > 0]
    if gains:
        out.append({
            "milestone": "CULTURAL_TRANSMISSION",
            "evidence": {"sessions": len(pop.teaching_sessions),
                         "mean_gain": round(sum(gains) / len(gains), 4),
                         "max_gain": round(max(gains), 4)},
        })

    # M3: overlapping generations
    gens = sorted({o.generation for o in pop.organisms})
    if len(gens) >= 2:
        out.append({
            "milestone": "OVERLAPPING_GENERATIONS",
            "evidence": {"generations": gens, "total_organisms": total,
                         "living": len(living)},
        })

    # M4: emergent communication (grounded symbols actually produced)
    produced = []
    for o in living:
        log = getattr(o, "last_action_log", {}) or {}
        syms = log.get("produced_symbols") or []
        if syms:
            produced.append({"organism_id": o.id, "symbols": syms})
    if produced:
        out.append({
            "milestone": "EMERGENT_COMMUNICATION",
            "evidence": {"producers": produced[:5],
                         "vocabularies": {o.id: (o.language.vocabulary_size()
                                                 if getattr(o, "language", None) else 0)
                                          for o in living[:10]}},
        })

    # M5: social relationships with persistence evidence
    strong = []
    for o in living:
        mem = getattr(o, "social_mem", None)
        if mem is not None:
            n_strong = mem.summary()["strong_relationships"]
            if n_strong > 0:
                strong.append({"organism_id": o.id, "strong_relationships": n_strong,
                               "top_partner": (mem.top_partners(1) or ["none"])[0]})
    if strong:
        out.append({
            "milestone": "SOCIAL_RELATIONSHIP_PERSISTENCE",
            "evidence": {"organisms_with_strong_ties": strong[:5]},
        })

    # M6: speciation (divergent genomes measured)
    from src.evolution.speciation import assign_species
    species = assign_species([(o.id, o.genome) for o in living], threshold=0.12) \
        if living else []
    if len(species) >= 2:
        out.append({
            "milestone": "SPECIATION",
            "evidence": {"species_count": len(species),
                         "members_per_species": [len(s.member_ids) for s in species],
                         "criterion": "genome_distance>0.12"},
        })

    for rec in out:
        rec["population_hash"] = pop.population_hash()
        rec["world_hash"] = pop.world.world_hash()
        rec["tick"] = pop.tick
        rec["captured_ts"] = time.time()
        rec["resolution"] = "full"
    return out


def milestone_certificate(record: Dict[str, Any], experiment_seed: int) -> Dict[str, Any]:
    """Machine-readable certificate for a captured milestone."""
    return {
        "certificate_version": "milestone_cert_v1",
        "milestone": record.get("milestone"),
        "evidence": record.get("evidence"),
        "population_hash": record.get("population_hash"),
        "world_hash": record.get("world_hash"),
        "tick": record.get("tick"),
        "experiment_seed": experiment_seed,
        "captured_ts": record.get("captured_ts"),
        "valid": bool(record.get("evidence")),
    }
