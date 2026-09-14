"""Deterministic grid world with resources + hazards (REAL, IMPLEMENTED)."""
import hashlib
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from src.common.determinism import derive_subseed


@dataclass
class WorldConfig:
    width: int = 16
    height: int = 16
    n_resources: int = 12
    n_hazards: int = 4
    world_seed: int = 47


class GridWorld:
    def __init__(self, config: WorldConfig):
        self.config = config
        self.tick = 0
        self.resources: Dict[Tuple[int, int], float] = {}
        self.hazards: List[Tuple[int, int]] = []
        self.positions: Dict[str, Tuple[int, int]] = {}
        self.energy_injected = 0.0  # ecological influx via regrowth (tracked, not spontaneous)
        self.reset(config.world_seed)

    def reset(self, seed: int):
        rng = np.random.RandomState(int(seed) % (2 ** 31))
        self.tick = 0
        self.resources = {}
        cells = [(x, y) for x in range(self.config.width) for y in range(self.config.height)]
        idx = rng.choice(len(cells), size=min(self.config.n_resources, len(cells)), replace=False)
        for i in idx:
            self.resources[cells[int(i)]] = float(rng.uniform(0.5, 1.0))
        hidx = rng.choice(len(cells), size=min(self.config.n_hazards, len(cells)), replace=False)
        self.hazards = [cells[int(i)] for i in hidx]
        self.positions = {}

    def place(self, organism_id: str, pos: Tuple[int, int] = (0, 0)):
        self.positions[organism_id] = (int(pos[0]) % self.config.width,
                                       int(pos[1]) % self.config.height)

    def sense(self, organism_id: str, radius: int = 3) -> Dict[str, Any]:
        x, y = self.positions.get(organism_id, (0, 0))
        food, hazard, others = 0.0, 0.0, 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                c = ((x + dx) % self.config.width, (y + dy) % self.config.height)
                if c in self.resources:
                    food += self.resources[c] / (1.0 + abs(dx) + abs(dy))
                if c in self.hazards:
                    hazard += 1.0 / (1.0 + abs(dx) + abs(dy))
        for oid, p in self.positions.items():
            if oid != organism_id and abs(p[0] - x) + abs(p[1] - y) <= radius:
                others += 1
        return {"food_gradient": round(float(food), 4), "hazard_gradient": round(float(hazard), 4),
                "nearby_organisms": int(others), "position": [x, y]}

    def apply_move(self, organism_id: str, dx: int, dy: int):
        x, y = self.positions.get(organism_id, (0, 0))
        self.positions[organism_id] = ((x + int(dx)) % self.config.width,
                                       (y + int(dy)) % self.config.height)

    def regrow(self, tick: int, interval: int = 10, cap_mult: int = 2):
        """Deterministic ecological regrowth (tracked influx). Call once per tick."""
        if tick % interval != 0:
            return 0.0
        import numpy as np
        cap = self.config.n_resources * cap_mult
        if len(self.resources) >= cap:
            return 0.0
        rng = np.random.RandomState(derive_subseed(self.config.world_seed, f"regrow:{tick}"))
        for _ in range(3):  # few attempts to find empty cell
            c = (int(rng.randint(0, self.config.width)), int(rng.randint(0, self.config.height)))
            if c not in self.resources and c not in self.hazards:
                amt = round(float(rng.uniform(0.4, 0.8)), 4)
                self.resources[c] = amt
                self.energy_injected = round(self.energy_injected + amt, 6)
                return amt
        return 0.0

    def consume(self, organism_id: str, radius: int = 1) -> float:
        """Graze resources within Manhattan radius (transfers energy, conserves total)."""
        x, y = self.positions.get(organism_id, (0, 0))
        total = 0.0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) <= radius:
                    c = ((x + dx) % self.config.width, (y + dy) % self.config.height)
                    total += float(self.resources.pop(c, 0.0))
        return total

    def hazard_at(self, organism_id: str) -> bool:
        return self.positions.get(organism_id) in self.hazards

    def total_resource_energy(self) -> float:
        return round(float(sum(self.resources.values())), 6)

    def world_hash(self) -> str:
        canon = json.dumps({"tick": self.tick,
                            "resources": sorted([[k[0], k[1], v] for k, v in self.resources.items()]),
                            "hazards": sorted(self.hazards),
                            "positions": sorted([[k, list(v)] for k, v in self.positions.items()])},
                           sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()

    def snapshot(self) -> Dict[str, Any]:
        return {"tick": self.tick,
                "resources": [[k[0], k[1], v] for k, v in self.resources.items()],
                "hazards": [list(h) for h in self.hazards],
                "positions": {k: list(v) for k, v in self.positions.items()},
                "config": self.config.__dict__,
                "energy_injected": self.energy_injected}

    def restore(self, snap: Dict[str, Any]):
        cfg = snap.get("config", {})
        self.config = WorldConfig(width=int(cfg.get("width", 16)),
                                  height=int(cfg.get("height", 16)),
                                  n_resources=int(cfg.get("n_resources", 12)),
                                  n_hazards=int(cfg.get("n_hazards", 4)),
                                  world_seed=int(cfg.get("world_seed", 47)))
        self.tick = int(snap["tick"])
        self.resources = {(int(a), int(b)): float(c) for a, b, c in snap["resources"]}
        self.hazards = [(int(a), int(b)) for a, b in snap["hazards"]]
        self.positions = {k: (int(v[0]), int(v[1])) for k, v in snap["positions"].items()}
        self.energy_injected = float(snap.get("energy_injected", 0.0))
