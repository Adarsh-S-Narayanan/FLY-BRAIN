"""Autonomy engine (STAGE E): self-generated goals, compositional actions,
self-evaluation. The organism decides what to inspect, where to go, what to
repeat and what to avoid — humans do not issue per-task commands.

Brain-driven contract (mission rule 13): candidates are MODULATED by neural
state (motor readout activations, drives, prediction error); the engine never
bypasses the brain — it composes continuous control from it.
"""
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.determinism import derive_subseed


class GoalSource(str, Enum):
    NEED = "need"                 # unmet homeostatic need (energy, safety)
    CURIOSITY = "curiosity"       # internal drive for novelty
    PREDICTION_ERROR = "prediction_error"
    OPPORTUNITY = "opportunity"   # environmental (observed resource etc.)
    MEMORY = "memory"             # remembered success/failure
    SOCIAL = "social"


class GoalStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ACHIEVED = "ACHIEVED"
    ABANDONED = "ABANDONED"


@dataclass
class Goal:
    goal_id: str
    source: str
    kind: str            # forage | explore | investigate | rest | socialize | avoid
    target: Optional[Tuple[int, int]]
    priority: float
    created_tick: int
    status: str = GoalStatus.ACTIVE.value
    progress: float = 0.0
    ticks_active: int = 0
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["target"] = list(self.target) if self.target else None
        return d


@dataclass
class ActionCandidate:
    """Compositional, continuous body control (NOT a tiny action menu)."""
    kind: str                     # move | rest | investigate | communicate
    heading: float                # radians, continuous
    speed: float                  # 0..1 continuous
    duration: int                 # intended ticks
    intensity: float              # 0..1 e.g. communication/inspection strength
    target: Optional[Tuple[int, int]] = None
    goal_id: str = ""
    neural_evidence: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(kind=self.kind, heading=round(self.heading, 4), speed=round(self.speed, 4),
                 duration=int(self.duration), intensity=round(self.intensity, 4),
                 target=list(self.target) if self.target else None,
                 goal_id=self.goal_id)
        d["neural_evidence"] = {k: round(v, 5) for k, v in self.neural_evidence.items()}
        return d


class AutonomyEngine:
    """Self-generated goal management + compositional action synthesis."""

    def __init__(self, seed: int = 42, goal_patience: int = 12,
                 novelty_radius: int = 3):
        self.seed = int(seed)
        self.goal_patience = int(goal_patience)
        self.novelty_radius = int(novelty_radius)
        self.goals: List[Goal] = []
        self.completed: List[Goal] = []
        self._goal_seq = 0
        self._visited: set = set()
        self._rng_counter = 0

    # ------------------------------------------------------------ goals
    def _new_goal(self, tick: int, source: GoalSource, kind: str,
                  priority: float, target=None, evidence=None) -> Goal:
        self._goal_seq += 1
        gid = f"g{self.seed:04d}-{self._goal_seq:04d}"
        g = Goal(gid, source.value, kind, target, round(float(priority), 4),
                 tick, evidence=dict(evidence or {}))
        self.goals.append(g)
        return g

    def generate_goals(self, tick: int, energy: float, health: float,
                       drives: Dict[str, float], prediction_error: float,
                       world_sense: Dict[str, Any], position: Tuple[int, int],
                       skills: Dict[str, float]) -> List[Goal]:
        """Self-generated tasks from needs, drives, errors, opportunities, memory."""
        generated: List[Goal] = []
        # 1. Unmet needs dominate
        if energy < 0.45:
            generated.append(self._new_goal(
                tick, GoalSource.NEED, "forage", priority=1.0 + (0.45 - energy),
                evidence={"energy": round(energy, 3)}))
        if health < 0.5:
            generated.append(self._new_goal(
                tick, GoalSource.NEED, "rest", priority=0.9 + (0.5 - health),
                evidence={"health": round(health, 3)}))
        # 2. Curiosity / novelty (unvisited neighborhood)
        nearby = self._nearby_unvisited(position)
        if drives.get("curiosity", 0.0) > 0.75 and nearby:
            generated.append(self._new_goal(
                tick, GoalSource.CURIOSITY, "explore", priority=0.3 + drives["curiosity"],
                target=nearby[0], evidence={"unvisited_nearby": len(nearby)}))
        # 3. Prediction error -> investigate
        if prediction_error > 0.4:
            generated.append(self._new_goal(
                tick, GoalSource.PREDICTION_ERROR, "investigate",
                priority=0.4 + prediction_error, target=position,
                evidence={"prediction_error": round(prediction_error, 3)}))
        # 4. Environmental opportunity
        food = float(world_sense.get("food_gradient", 0.0))
        if food > 1.0 and energy < 0.85:
            generated.append(self._new_goal(
                tick, GoalSource.OPPORTUNITY, "forage", priority=0.5 + min(1.0, food / 3.0),
                evidence={"food_gradient": round(food, 3)}))
        hazard = float(world_sense.get("hazard_gradient", 0.0))
        if hazard > 0.6:
            generated.append(self._new_goal(
                tick, GoalSource.NEED, "avoid", priority=0.8 + hazard,
                evidence={"hazard_gradient": round(hazard, 3)}))
        # 5. Social drive
        if drives.get("social", 0.0) > 0.9 and int(world_sense.get("nearby_organisms", 0)) > 0:
            generated.append(self._new_goal(
                tick, GoalSource.SOCIAL, "socialize", priority=0.35 + drives["social"],
                evidence={"nearby": int(world_sense.get("nearby_organisms", 0))}))
        return generated

    def _nearby_unvisited(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = position
        r = self.novelty_radius
        cands = [(x + dx, y + dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                 if (dx, dy) != (0, 0)]
        return [c for c in cands if c not in self._visited]

    def update_goals(self, tick: int, energy_delta: float, novelty_gained: bool,
                     prediction_error: float) -> List[Goal]:
        """Self-evaluation: progress, achievement, abandonment, task switching."""
        finished: List[Goal] = []
        for g in self.goals:
            g.ticks_active += 1
            if g.kind == "forage":
                g.progress = min(1.0, g.progress + max(0.0, energy_delta) * 2.0)
            elif g.kind == "explore" and novelty_gained:
                g.progress = min(1.0, g.progress + 0.34)
            elif g.kind == "investigate":
                g.progress = min(1.0, g.progress + max(0.0, 0.5 - prediction_error))
            if g.progress >= 0.99:
                g.status = GoalStatus.ACHIEVED.value
            elif g.ticks_active > self.goal_patience and g.progress < 0.05:
                g.status = GoalStatus.ABANDONED.value
            if g.status != GoalStatus.ACTIVE.value:
                finished.append(g)
        if finished:
            self.completed.extend(finished)
            fin_ids = {g.goal_id for g in finished}
            self.goals = [g for g in self.goals if g.goal_id not in fin_ids]
        return finished

    def active_goal(self) -> Optional[Goal]:
        if not self.goals:
            return None
        return max(self.goals, key=lambda g: (g.priority, -g.created_tick))

    def note_position(self, position: Tuple[int, int]) -> bool:
        """Track visited cells for novelty. Returns True if novel."""
        if position in self._visited:
            return False
        self._visited.add(position)
        return True

    # ---------------------------------------------------------- actions
    def synthesize_action(self, tick: int, goal: Optional[Goal],
                          brain_out: Dict[str, Any], world_sense: Dict[str, Any],
                          energy: float, health: float,
                          exploration: float = 0.5) -> ActionCandidate:
        """Compose continuous control from the ACTIVE goal + neural state.

        Neural evidence (mission rule 13): motor_act asymmetry biases heading,
        drive levels scale speed/intensity, prediction error boosts inspection.
        """
        self._rng_counter += 1
        rng = np.random.RandomState(derive_subseed(self.seed, f"act:{tick}:{self._rng_counter}"))
        drives = brain_out.get("drives", {})
        assoc = brain_out.get("tool_associations", {})
        act_a = float(assoc.get("act_in_environment", 0.0))
        remember_a = float(assoc.get("remember", 0.0))
        pred_err = float(brain_out.get("prediction_error", 0.0))

        # heading: goal target pull + neural lateral bias + deterministic jitter
        base_heading = rng.uniform(0.0, 2.0 * math.pi)
        if goal is not None and goal.target is not None:
            dx, dy = goal.target[0] - world_sense["position"][0], goal.target[1] - world_sense["position"][1]
            if abs(dx) + abs(dy) > 0:
                base_heading = 0.4 * base_heading + 0.6 * math.atan2(dy, dx)
        neural_bias = (act_a - remember_a) * (math.pi / 4.0)
        heading = (base_heading + neural_bias) % (2.0 * math.pi)

        kind = "move"
        target = goal.target if goal is not None else None
        if goal is not None:
            if goal.kind == "rest":
                kind = "rest"
            elif goal.kind == "investigate":
                kind = "investigate"
            elif goal.kind == "socialize":
                kind = "communicate"
        elif health < 0.35:
            kind = "rest"

        curiosity = float(drives.get("curiosity", 0.5))
        energy_drive = float(drives.get("energy", 1.0))
        if kind == "rest":
            speed = 0.0
            duration = 3
            intensity = float(np.clip(1.0 - health, 0.0, 1.0))
        elif kind == "investigate":
            speed = float(np.clip(0.2 + pred_err, 0.0, 1.0))
            duration = 2
            intensity = float(np.clip(pred_err, 0.0, 1.0))
        elif kind == "communicate":
            speed = float(np.clip(0.1 + 0.2 * drives.get("social", 0.5), 0.0, 1.0))
            duration = 2
            intensity = float(np.clip(drives.get("social", 0.5), 0.0, 1.0))
        else:
            # hungry organisms sprint; satiated organisms wander curiously
            speed = float(np.clip(0.3 + (1.0 - energy) * 0.6 * energy_drive
                                  + 0.2 * curiosity * exploration, 0.05, 1.0))
            duration = 1 + int(rng.randint(0, 2))
            intensity = float(np.clip(act_a, 0.0, 1.0))
        return ActionCandidate(
            kind=kind, heading=float(heading), speed=float(np.clip(speed, 0.0, 1.0)),
            duration=int(duration), intensity=float(np.clip(intensity, 0.0, 1.0)),
            target=target, goal_id=goal.goal_id if goal is not None else "",
            neural_evidence={
                "motor_act": act_a, "motor_remember": remember_a,
                "curiosity": curiosity, "prediction_error": pred_err,
            })

    # -------------------------------------------------------- introspection
    def autonomy_summary(self) -> Dict[str, Any]:
        return {
            "active_goals": [g.to_dict() for g in self.goals],
            "completed_count": len(self.completed),
            "achieved": sum(1 for g in self.completed if g.status == GoalStatus.ACHIEVED.value),
            "abandoned": sum(1 for g in self.completed if g.status == GoalStatus.ABANDONED.value),
            "visited_cells": len(self._visited),
            "goals_generated": self._goal_seq,
            "sources": sorted({g.source for g in self.completed + self.goals}),
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "seed": self.seed, "goal_patience": self.goal_patience,
            "novelty_radius": self.novelty_radius,
            "goals": [g.to_dict() for g in self.goals],
            "completed": [g.to_dict() for g in self.completed],
            "goal_seq": self._goal_seq,
            "visited": sorted(list(self._visited)),
            "rng_counter": self._rng_counter,
        }

    @classmethod
    def restore(cls, payload: Dict[str, Any]) -> "AutonomyEngine":
        eng = cls(payload["seed"], payload.get("goal_patience", 12),
                  payload.get("novelty_radius", 3))
        def _goal(d):
            t = d.get("target")
            return Goal(d["goal_id"], d["source"], d["kind"],
                        tuple(t) if t else None, d["priority"], d["created_tick"],
                        d.get("status", "ACTIVE"), d.get("progress", 0.0),
                        d.get("ticks_active", 0), d.get("evidence", {}))
        eng.goals = [_goal(d) for d in payload.get("goals", [])]
        eng.completed = [_goal(d) for d in payload.get("completed", [])]
        eng._goal_seq = int(payload.get("goal_seq", 0))
        eng._visited = {tuple(c) for c in payload.get("visited", [])}
        eng._rng_counter = int(payload.get("rng_counter", 0))
        return eng
