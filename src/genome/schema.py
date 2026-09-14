"""Versioned artificial genome schema (REAL, IMPLEMENTED)."""
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

GENOME_VERSION = "1.0"

DEFAULT_PARAMS: Dict[str, float] = {
    "neurogenesis_rate": 0.3,
    "differentiation_bias": 0.5,
    "migration_rate": 0.4,
    "axon_growth_rate": 0.5,
    "dendrite_growth_rate": 0.5,
    "synaptogenesis_rate": 0.5,
    "pruning_threshold": 0.05,
    "plasticity_rate": 0.05,
    "memory_retention": 0.8,
    "metabolism_rate": 0.01,
    "exploration": 0.5,
    "sociality": 0.5,
    "teaching_ability": 0.5,
    "reproduction_threshold": 0.6,
    "developmental_timing": 0.5,
}

PARAM_BOUNDS = {
    "neurogenesis_rate": (0.0, 1.0),
    "differentiation_bias": (0.0, 1.0),
    "migration_rate": (0.0, 1.0),
    "axon_growth_rate": (0.0, 1.0),
    "dendrite_growth_rate": (0.0, 1.0),
    "synaptogenesis_rate": (0.0, 1.0),
    "pruning_threshold": (0.0, 0.5),
    "plasticity_rate": (0.0, 0.5),
    "memory_retention": (0.0, 1.0),
    "metabolism_rate": (0.0, 0.1),
    "exploration": (0.0, 1.0),
    "sociality": (0.0, 1.0),
    "teaching_ability": (0.0, 1.0),
    "reproduction_threshold": (0.0, 1.0),
    "developmental_timing": (0.0, 1.0),
}


@dataclass
class Genome:
    params: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    version: str = GENOME_VERSION
    lineage: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        if self.version != GENOME_VERSION:
            raise ValueError(f"Unsupported genome version: {self.version}")
        for k, (lo, hi) in PARAM_BOUNDS.items():
            if k not in self.params:
                raise ValueError(f"Missing genome param: {k}")
            v = float(self.params[k])
            if not (lo <= v <= hi) or v != v:  # NaN check via v!=v
                raise ValueError(f"Param {k}={v} out of bounds [{lo},{hi}]")
        return True

    def genome_hash(self) -> str:
        self.validate()
        canon = json.dumps({"version": self.version,
                            "params": {k: round(float(self.params[k]), 6) for k in sorted(self.params)}},
                           sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "params": dict(self.params), "lineage": dict(self.lineage)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Genome":
        g = cls(params=dict(d["params"]), version=d.get("version", GENOME_VERSION),
                lineage=dict(d.get("lineage", {})))
        g.validate()
        return g

    @classmethod
    def founder(cls, seed: int = 42) -> "Genome":
        import numpy as np
        rng = np.random.RandomState(seed)
        params = {}
        for k in DEFAULT_PARAMS:
            lo, hi = PARAM_BOUNDS[k]
            params[k] = float(lo + (hi - lo) * 0.5 + rng.normal(0, 0.02 * (hi - lo)))
            params[k] = float(min(hi, max(lo, params[k])))
        g = cls(params=params, lineage={"origin": "founder", "seed": seed})
        g.validate()
        return g
