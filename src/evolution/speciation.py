"""Speciation (STAGE J): genome-distance divergence with evidence.

Two organisms belong to the same species while their genome distance stays
below the threshold. A species split is RECORDED only when an actually
measurable divergence exists (param differences + lineage separation) —
never fabricated from labels alone.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.genome.schema import Genome, PARAM_BOUNDS


def genome_distance(a: Genome, b: Genome) -> float:
    """Mean normalized absolute param difference over the union of genes (0..1)."""
    a.validate()
    b.validate()
    keys = sorted(set(a.params) | set(b.params))
    if not keys:
        return 0.0
    total = 0.0
    for k in keys:
        lo, hi = PARAM_BOUNDS[k]
        span = (hi - lo) or 1.0
        va = float(a.params.get(k, lo))
        vb = float(b.params.get(k, lo))
        total += abs(va - vb) / span
    return total / len(keys)


@dataclass
class SpeciesRecord:
    species_id: str
    member_ids: List[str]
    founder_ids: List[str]
    divergence_tick: int
    parent_species: Optional[str]
    genome_distance_from_parent: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def assign_species(genomes: List[Tuple[str, Genome]], threshold: float = 0.12
                   ) -> List[SpeciesRecord]:
    """Single-linkage clustering by genome distance. Deterministic (sorted ids)."""
    if threshold <= 0 or threshold >= 1:
        raise ValueError("threshold must be in (0, 1)")
    items = sorted(genomes, key=lambda x: x[0])
    clusters: List[List[str]] = []
    for oid, g in items:
        placed = False
        for cl in clusters:
            rep = next(gg for oo, gg in items if oo == cl[0])
            if genome_distance(rep, g) <= threshold:
                cl.append(oid)
                placed = True
                break
        if not placed:
            clusters.append([oid])
    out: List[SpeciesRecord] = []
    for cl in clusters:
        sid = hashlib.sha256(("|".join(sorted(cl))).encode()).hexdigest()[:12]
        out.append(SpeciesRecord(species_id=sid, member_ids=sorted(cl),
                                 founder_ids=[cl[0]], divergence_tick=0,
                                 parent_species=None,
                                 evidence={"criterion": "genome_distance",
                                           "threshold": threshold}))
    return out


def detect_divergence(parents: List[Tuple[str, Genome]],
                      children: List[Tuple[str, Genome]], tick: int,
                      threshold: float = 0.12) -> List[SpeciesRecord]:
    """Evidence-backed divergence: children farther than threshold from ALL
    parents form a new species record with measured distance."""
    divergent: List[SpeciesRecord] = []
    for cid, cg in children:
        if not parents:
            break
        d_min, nearest = min((genome_distance(cg, pg), pid) for pid, pg in parents)
        if d_min > threshold:
            sid = hashlib.sha256(f"div|{cid}|{tick}".encode()).hexdigest()[:12]
            divergent.append(SpeciesRecord(
                species_id=sid, member_ids=[cid], founder_ids=[cid],
                divergence_tick=tick, parent_species=nearest,
                genome_distance_from_parent=round(d_min, 6),
                evidence={"criterion": "min_parent_distance_exceeded",
                          "threshold": threshold, "nearest_parent": nearest}))
    return divergent
