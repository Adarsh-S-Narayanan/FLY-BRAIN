"""Versioned artificial genome schema (REAL, IMPLEMENTED).

v1.0: 15 body/development/behavior params (compat baseline — legacy snapshots
      and canonical replays remain bit-exact and loadable).
v2.0: v1 params + 9 learning-architecture genes (STAGE I). Evolution can now
      alter eligibility decay, neuromodulation weights, growth budget,
      prediction influence, social-learning bias, sleep duration and
      communication tendency — i.e. the learning architecture itself.

Version policy: validate() accepts exactly "1.0" (15 params) or "2.0" (24
params). NO silent migration — a v1 genome stays v1 (consumers apply runtime
defaults for absent genes), so old replays reproduce exactly.
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict

GENOME_VERSION = "2.0"
LEGACY_VERSION = "1.0"
SUPPORTED_VERSIONS = (LEGACY_VERSION, GENOME_VERSION)

V1_PARAMS: Dict[str, float] = {
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

# STAGE I: learning-architecture genes (consumed by autonomy/v2 organisms)
ARCH_PARAMS: Dict[str, float] = {
    "eligibility_decay": 0.9,            # L2: trace persistence
    "neuromod_novelty_weight": 0.3,      # L2: novelty drives plasticity
    "neuromod_prediction_weight": 0.3,   # L2: surprise drives plasticity
    "growth_budget_fraction": 0.5,       # L6: metabolic allocation to growth
    "prediction_gain": 0.5,              # L5: attention steering by error
    "social_learning_bias": 0.5,         # L4: prefer trusted teachers
    "sleep_duration": 0.4,               # 0..1 -> 1..8 replay episodes
    "communication_tendency": 0.4,       # L4: produce grounded symbols
    "curiosity_drive": 0.5,              # L5: exploration scaling
}

DEFAULT_PARAMS: Dict[str, float] = dict(V1_PARAMS)
DEFAULT_PARAMS.update(ARCH_PARAMS)

V1_BOUNDS = {
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

ARCH_BOUNDS = {
    "eligibility_decay": (0.0, 0.99),
    "neuromod_novelty_weight": (0.0, 2.0),
    "neuromod_prediction_weight": (0.0, 2.0),
    "growth_budget_fraction": (0.0, 1.0),
    "prediction_gain": (0.0, 1.0),
    "social_learning_bias": (0.0, 1.0),
    "sleep_duration": (0.0, 1.0),
    "communication_tendency": (0.0, 1.0),
    "curiosity_drive": (0.0, 1.0),
}

PARAM_BOUNDS: Dict[str, Any] = dict(V1_BOUNDS)
PARAM_BOUNDS.update(ARCH_BOUNDS)

V1_PARAM_NAMES = frozenset(V1_PARAMS.keys())


@dataclass
class Genome:
    params: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    version: str = GENOME_VERSION
    lineage: Dict[str, Any] = field(default_factory=dict)

    def _required_names(self) -> frozenset:
        return V1_PARAM_NAMES if self.version == LEGACY_VERSION else \
            frozenset(PARAM_BOUNDS.keys())

    def validate(self) -> bool:
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported genome version: {self.version} "
                             f"(supported: {SUPPORTED_VERSIONS})")
        required = self._required_names()
        extra = set(self.params) - required
        if extra:
            raise ValueError(f"Genome version {self.version} does not allow "
                             f"params: {sorted(extra)}")
        for k in required:
            if k not in self.params:
                raise ValueError(f"Missing genome param: {k}")
        for k, v in self.params.items():
            lo, hi = PARAM_BOUNDS[k]
            v = float(v)
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
    def founder(cls, seed: int = 42, legacy: bool = False) -> "Genome":
        import numpy as np
        rng = np.random.RandomState(seed)
        # RNG consumption order MUST match the original implementation
        # (dict insertion order: v1 params first) so legacy founders stay
        # bit-identical with pre-v2 runs.
        ordered = V1_PARAMS if legacy else DEFAULT_PARAMS
        params = {}
        for k in ordered:
            lo, hi = PARAM_BOUNDS[k]
            params[k] = float(lo + (hi - lo) * 0.5 + rng.normal(0, 0.02 * (hi - lo)))
            params[k] = float(min(hi, max(lo, params[k])))
        g = cls(params=params, version=LEGACY_VERSION if legacy else GENOME_VERSION,
                lineage={"origin": "founder", "seed": seed})
        g.validate()
        return g

    # Runtime defaults for genes absent in v1 genomes (explicit, not silent):
    def get(self, name: str, default: float = 0.0) -> float:
        return float(self.params.get(name, ARCH_PARAMS.get(name, default)))
