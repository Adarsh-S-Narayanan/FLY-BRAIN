"""Social model (STAGE H): identity recognition, interaction history, trust
that EMERGES from interaction outcomes (never hard-coded friendship),
persistent relationships."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TRUST_INIT = 0.3          # neutral prior, not friendship
TRUST_LEARNING_RATE = 0.2


@dataclass
class InteractionRecord:
    tick: int
    kind: str                 # taught_by | taught_to | communicated | cooperated | competed
    outcome: float            # -1..1 signed outcome for SELF
    other_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SocialRecord:
    other_id: str
    interactions: int = 0
    positive_outcomes: int = 0
    negative_outcomes: int = 0
    trust: float = TRUST_INIT
    first_tick: int = 0
    last_tick: int = 0
    history: List[InteractionRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"other_id": self.other_id, "interactions": self.interactions,
                "positive_outcomes": self.positive_outcomes,
                "negative_outcomes": self.negative_outcomes,
                "trust": self.trust, "first_tick": self.first_tick,
                "last_tick": self.last_tick,
                "history": [h.to_dict() for h in self.history[-20:]]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SocialRecord":
        return cls(d["other_id"], int(d.get("interactions", 0)),
                   int(d.get("positive_outcomes", 0)),
                   int(d.get("negative_outcomes", 0)), float(d.get("trust", TRUST_INIT)),
                   int(d.get("first_tick", 0)), int(d.get("last_tick", 0)),
                   [InteractionRecord(**h) for h in d.get("history", [])])


class SocialMemory:
    """Per-organism social memory: recognition + emergent trust."""

    def __init__(self, self_id: str, trust_lr: float = TRUST_LEARNING_RATE):
        self.self_id = self_id
        self.trust_lr = float(trust_lr)
        self.records: Dict[str, SocialRecord] = {}

    def knows(self, other_id: str) -> bool:
        return other_id in self.records

    def trust_of(self, other_id: str) -> float:
        rec = self.records.get(other_id)
        return rec.trust if rec is not None else TRUST_INIT

    def record_interaction(self, other_id: str, tick: int, kind: str,
                           outcome: float) -> SocialRecord:
        if not other_id or other_id == self.self_id:
            raise ValueError("invalid interaction partner")
        outcome = max(-1.0, min(1.0, float(outcome)))
        rec = self.records.get(other_id)
        if rec is None:
            rec = SocialRecord(other_id=other_id, first_tick=tick)
            self.records[other_id] = rec
        rec.interactions += 1
        rec.last_tick = tick
        rec.history.append(InteractionRecord(tick, kind, outcome, other_id))
        if outcome > 0.05:
            rec.positive_outcomes += 1
        elif outcome < -0.05:
            rec.negative_outcomes += 1
        # trust EMERGES from outcomes (delta rule toward observed outcome)
        rec.trust = float(min(1.0, max(0.0,
                       rec.trust + self.trust_lr * (outcome - rec.trust))))
        return rec

    def relationship_strength(self, other_id: str) -> float:
        """Persistent-relationship evidence: repeated interactions + trust."""
        rec = self.records.get(other_id)
        if rec is None:
            return 0.0
        frequency = min(1.0, rec.interactions / 10.0)
        return round(0.5 * rec.trust + 0.5 * frequency, 4)

    def top_partners(self, k: int = 3) -> List[str]:
        ranked = sorted(self.records.values(),
                        key=lambda r: (self.relationship_strength(r.other_id),
                                       r.interactions), reverse=True)
        return [r.other_id for r in ranked[:k]]

    def summary(self) -> Dict[str, Any]:
        return {"self_id": self.self_id, "known_others": len(self.records),
                "strong_relationships": sum(
                    1 for r in self.records.values()
                    if self.relationship_strength(r.other_id) >= 0.5),
                "total_interactions": sum(r.interactions for r in self.records.values())}

    def snapshot(self) -> Dict[str, Any]:
        return {"self_id": self.self_id, "trust_lr": self.trust_lr,
                "records": [r.to_dict() for r in self.records.values()]}

    @classmethod
    def restore(cls, payload: Dict[str, Any]) -> "SocialMemory":
        mem = cls(payload["self_id"], payload.get("trust_lr", TRUST_LEARNING_RATE))
        for d in payload.get("records", []):
            rec = SocialRecord.from_dict(d)
            mem.records[rec.other_id] = rec
        return mem
