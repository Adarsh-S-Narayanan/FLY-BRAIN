"""Embodiment (STAGE F): body state, damage/recovery, metabolic limits.

The body constrains what the brain can do: damage reduces speed capacity,
movement costs energy, rest recovers. Real state, no decorative counters.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class BodyState:
    energy: float = 1.0
    health: float = 1.0
    damage: float = 0.0            # cumulative unrecovered damage 0..inf
    position: Tuple[int, int] = (0, 0)
    heading: float = 0.0
    age: int = 0
    rest_ticks: int = 0
    damage_events: int = 0

    MAX_ENERGY: float = 1.5

    def speed_capacity(self) -> float:
        """Damage limits top speed (bodily constraint on autonomy)."""
        return max(0.15, 1.0 - 0.4 * min(1.0, self.damage))

    def metabolic_cost(self, activity: float, move: float, rate: float = 0.01) -> float:
        return rate * (0.5 + activity + 0.25 * move)

    def apply_damage(self, amount: float, cause: str = "") -> bool:
        """Returns True if damage was applied (health actually reduced)."""
        if amount <= 0:
            return False
        self.health = max(0.0, self.health - amount)
        self.damage = min(2.0, self.damage + amount)
        self.damage_events += 1
        self.last_damage_cause = cause
        return True

    def recover(self, dt: float = 0.02) -> float:
        """Rest recovers damage and some health (bounded)."""
        rec = min(self.damage, dt)
        self.damage -= rec
        self.health = min(1.0, self.health + 0.5 * rec)
        self.rest_ticks += 1
        return rec

    def consume_energy(self, amount: float) -> None:
        self.energy = max(0.0, self.energy - amount)

    def gain_energy(self, amount: float) -> None:
        self.energy = min(self.MAX_ENERGY, self.energy + amount)

    def is_mobile(self) -> bool:
        return self.health > 0.05 and self.energy > 0.02

    def to_dict(self) -> Dict[str, Any]:
        return {"energy": round(self.energy, 6), "health": round(self.health, 6),
                "damage": round(self.damage, 6), "position": list(self.position),
                "heading": round(self.heading, 4), "age": self.age,
                "rest_ticks": self.rest_ticks, "damage_events": self.damage_events,
                "last_damage_cause": getattr(self, "last_damage_cause", "")}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BodyState":
        b = cls(energy=float(d.get("energy", 1.0)), health=float(d.get("health", 1.0)),
                damage=float(d.get("damage", 0.0)),
                position=tuple(d.get("position", (0, 0))),
                heading=float(d.get("heading", 0.0)), age=int(d.get("age", 0)),
                rest_ticks=int(d.get("rest_ticks", 0)),
                damage_events=int(d.get("damage_events", 0)))
        b.last_damage_cause = d.get("last_damage_cause", "")
        return b
