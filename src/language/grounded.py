"""Grounded language (STAGE G): symbols linked to actual sensory/internal
representations — never free-floating text. Production comes from internal
state; comprehension maps symbols back to grounded concepts; information
transfer is measured by behavior change in the receiver.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.determinism import derive_subseed

GROUNDING_MODALITIES = ("sensory", "internal", "social")


@dataclass
class Concept:
    concept_id: str
    name: str
    modality: str                     # sensory | internal | social
    feature_vector: Tuple[float, ...]  # grounded features (e.g. sensory stats)
    grounding_evidence: Dict[str, Any] = field(default_factory=dict)
    use_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["feature_vector"] = list(self.feature_vector)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Concept":
        return cls(d["concept_id"], d["name"], d["modality"],
                   tuple(d["feature_vector"]), d.get("grounding_evidence", {}),
                   int(d.get("use_count", 0)))


class GroundedLanguageSystem:
    """Symbol <-> concept <-> grounded representation pipeline."""

    def __init__(self, seed: int = 42, feature_dim: int = 8):
        self.seed = int(seed)
        self.feature_dim = int(feature_dim)
        self.concepts: Dict[str, Concept] = {}       # concept_id -> Concept
        self.lexicon: Dict[str, str] = {}            # symbol -> concept_id
        self.symbol_usage: Dict[str, int] = {}       # symbol -> productions+comprehensions
        self._seq = 0

    # ------------------------------------------------------------ grounding
    def ground_concept(self, name: str, feature_vector: List[float],
                       modality: str, evidence: Optional[Dict[str, Any]] = None) -> Concept:
        if modality not in GROUNDING_MODALITIES:
            raise ValueError(f"unknown modality {modality!r}")
        v = tuple(round(float(x), 6) for x in feature_vector[:self.feature_dim])
        if len(v) < self.feature_dim:
            v = v + (0.0,) * (self.feature_dim - len(v))
        if any(not np.isfinite(x) for x in v):
            raise ValueError("concept features must be finite")
        self._seq += 1
        cid = hashlib.sha256(f"{self.seed}|{name}|{self._seq}".encode()).hexdigest()[:12]
        c = Concept(cid, name, modality, v,
                    dict(evidence or {"grounded_at": self._seq}), use_count=0)
        self.concepts[cid] = c
        return c

    def learn_symbol(self, symbol: str, concept_id: str, teacher: str = "self",
                     tick: int = 0) -> bool:
        """Bind a symbol to an ALREADY-GROUNDED concept (never the reverse)."""
        if concept_id not in self.concepts:
            raise ValueError(f"cannot bind symbol {symbol!r} to ungrounded concept")
        if not symbol or any(ch in symbol for ch in " \n\t"):
            raise ValueError("invalid symbol")
        prev = self.lexicon.get(symbol)
        self.lexicon[symbol] = concept_id
        self.symbol_usage.setdefault(symbol, 0)
        return prev is None or prev == concept_id

    # ------------------------------------------------------- communication
    def produce(self, internal_state: Dict[str, Any], max_symbols: int = 4) -> List[str]:
        """Symbol sequence FROM internal state (drive levels, sensed features,
        active concept salience). Empty when nothing is communicable."""
        out: List[str] = []
        salient = internal_state.get("salient_concepts", [])
        for cid in salient[:max_symbols]:
            sym = self._symbol_for_concept(cid)
            if sym and sym not in out:
                self.concepts[cid].use_count += 1
                self.symbol_usage[sym] = self.symbol_usage.get(sym, 0) + 1
                out.append(sym)
        return out

    def _symbol_for_concept(self, concept_id: str) -> Optional[str]:
        for sym, cid in self.lexicon.items():
            if cid == concept_id:
                return sym
        return None

    def comprehend(self, symbols: List[str]) -> List[str]:
        """Map received symbols to grounded concept ids."""
        cids: List[str] = []
        for s in symbols:
            cid = self.lexicon.get(s)
            if cid is not None:
                self.concepts[cid].use_count += 1
                self.symbol_usage[s] = self.symbol_usage.get(s, 0) + 1
                cids.append(cid)
        return cids

    # -------------------------------------------------------- measurement
    def grounding_strength(self, symbol: str) -> float:
        """0..1: how strongly a symbol is anchored to grounded experience."""
        cid = self.lexicon.get(symbol)
        if cid is None:
            return 0.0
        c = self.concepts[cid]
        usage = min(1.0, self.symbol_usage.get(symbol, 0) / 10.0)
        return round(0.5 * (1.0 if c.grounding_evidence else 0.0) + 0.5 * usage, 4)

    def vocabulary_size(self) -> int:
        return len(self.lexicon)

    def ungrounded_symbols(self) -> List[str]:
        return [s for s, cid in self.lexicon.items()
                if not self.concepts[cid].grounding_evidence]

    # -------------------------------------------------------- persistence
    def snapshot(self) -> Dict[str, Any]:
        return {"seed": self.seed, "feature_dim": self.feature_dim,
                "concepts": [c.to_dict() for c in self.concepts.values()],
                "lexicon": dict(self.lexicon),
                "symbol_usage": dict(self.symbol_usage), "seq": self._seq}

    @classmethod
    def restore(cls, payload: Dict[str, Any]) -> "GroundedLanguageSystem":
        sys_ = cls(payload["seed"], payload.get("feature_dim", 8))
        for d in payload.get("concepts", []):
            c = Concept.from_dict(d)
            sys_.concepts[c.concept_id] = c
        sys_.lexicon = dict(payload.get("lexicon", {}))
        sys_.symbol_usage = dict(payload.get("symbol_usage", {}))
        sys_._seq = int(payload.get("seq", 0))
        return sys_
