"""Deep-time acceleration (STAGE L): event-driven coarse-grained evolution with
documented approximation, milestone escalation to full resolution, and
high-resolution replay. A coarse run NEVER claims equivalence to full neural
resolution — every record carries its resolution and approximation model.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

APPROXIMATION_MODEL = "aggregated_lifecycle_events_v1"


@dataclass
class DeepTimeConfig:
    coarse_ticks_per_step: int = 20       # behavioral ticks per coarse step
    offspring_per_generation: int = 2
    max_coarse_steps: int = 1000
    divergence_threshold: float = 0.12    # speciation genome distance
    mode: str = "ACCELERATED"             # EXACT | ACCELERATED (mission §36)
    milestone_hooks: List[Callable[[Dict[str, Any]], Optional[str]]] = field(
        default_factory=list)

    def validate(self) -> None:
        if self.coarse_ticks_per_step < 1:
            raise ValueError("coarse_ticks_per_step must be >= 1")
        if not (0 < self.divergence_threshold < 1):
            raise ValueError("divergence_threshold must be in (0, 1)")
        if self.mode not in ("EXACT", "ACCELERATED"):
            raise ValueError("mode must be EXACT or ACCELERATED")


@dataclass
class DeepTimeEvent:
    coarse_step: int
    sim_tick: int
    generation: int
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class DeepTimeRunner:
    """Accelerated or exact multi-generation evolution over a real Population.

    EXACT mode: tick-by-tick, full state, full event stream, exact replay.
    ACCELERATED mode: macro epochs with a documented approximation model.
    An accelerated run NEVER reports itself as exact (mission §36).
    """

    def __init__(self, population, config: Optional[DeepTimeConfig] = None):
        self.pop = population
        self.config = config or DeepTimeConfig()
        self.config.validate()
        self.ledger: Dict[str, Any] = {
            "mode": self.config.mode,
            "resolution": "full" if self.config.mode == "EXACT" else "coarse",
            "approximation_model": (None if self.config.mode == "EXACT"
                                    else APPROXIMATION_MODEL),
            "validation_method": ("exact_replay" if self.config.mode == "EXACT"
                                  else "checkpoint_escalation_replay"),
            "coarse_steps": 0, "sim_ticks": 0, "generations_seen": [],
            "births": 0, "deaths": 0, "teaching_sessions": 0,
            "species_snapshots": [], "milestones": [], "started_ts": time.time(),
        }
        self.events: List[DeepTimeEvent] = []
        self.escalation_checkpoints: List[Dict[str, Any]] = []
        self._prev_organism_ids: set = {o.id for o in self.pop.organisms}
        self._prev_species_count: int = 1
        self._prev_species: List[str] = []

    # ------------------------------------------------------------ stepping
    def fast_forward(self, n_coarse_steps: int,
                     milestone_fn: Optional[Callable[[Any], List[Dict[str, Any]]]] = None
                     ) -> Dict[str, Any]:
        for _ in range(min(n_coarse_steps, self.config.max_coarse_steps)):
            if self.config.mode == "EXACT":
                self.pop.step(1)  # tick-by-tick, no aggregation
                self.ledger["sim_ticks"] += 1
            else:
                self.pop.step(self.config.coarse_ticks_per_step)
                self.ledger["sim_ticks"] = self.pop.tick
            self.pop.reproduce(self.config.offspring_per_generation)
            self.ledger["coarse_steps"] += 1
            self.ledger["deaths"] = sum(
                1 for o in self.pop.organisms if not o.alive)
            self.ledger["teaching_sessions"] = len(self.pop.teaching_sessions)
            gens = sorted({o.generation for o in self.pop.organisms})
            for g in gens:
                if g not in self.ledger["generations_seen"]:
                    self.ledger["generations_seen"].append(g)
                    self.events.append(DeepTimeEvent(
                        self.ledger["coarse_steps"], self.pop.tick, g,
                        "GENERATION_REACHED", {"generation": g}))
            # speciation check (evidence-based)
            from src.evolution.speciation import assign_species
            species = assign_species(
                [(o.id, o.genome) for o in self.pop.living()],
                threshold=self.config.divergence_threshold)
            self.ledger["species_snapshots"].append(
                {"coarse_step": self.ledger["coarse_steps"], "species_count": len(species)})
            if len(species) > (self._prev_species_count or 1):
                self.events.append(DeepTimeEvent(
                    self.ledger["coarse_steps"], self.pop.tick,
                    max(gens) if gens else 0, "SPECIATION",
                    {"species_count": len(species),
                     "members": [s.member_ids for s in species]}))
            self._prev_species_count = len(species)
            # milestone detection + escalation hooks
            if milestone_fn is not None:
                for m in milestone_fn(self.pop):
                    self.ledger["milestones"].append(
                        {**m, "coarse_step": self.ledger["coarse_steps"],
                         "sim_tick": self.pop.tick, "resolution": self.ledger["resolution"]})
            for hook in self.config.milestone_hooks:
                name = hook(self.pop)
                if name:
                    self.ledger["milestones"].append(
                        {"milestone": name, "coarse_step": self.ledger["coarse_steps"],
                         "sim_tick": self.pop.tick, "resolution": self.ledger["resolution"]})
            if self._prev_organism_ids:
                new_ids = {o.id for o in self.pop.organisms} - self._prev_organism_ids
                self.ledger["births"] += len(new_ids)
                for nid in new_ids:
                    self.events.append(DeepTimeEvent(
                        self.ledger["coarse_steps"], self.pop.tick,
                        next(o.generation for o in self.pop.organisms if o.id == nid),
                        "BIRTH", {"organism_id": nid}))
            self._prev_organism_ids = {o.id for o in self.pop.organisms}
        return self.ledger_summary()

    def escalate(self, checkpoint_name: str) -> Dict[str, Any]:
        """Milestone escalation: full-resolution checkpoint for later replay."""
        snap = self.pop.snapshot()
        self.escalation_checkpoints.append({
            "checkpoint_name": checkpoint_name,
            "coarse_step": self.ledger["coarse_steps"],
            "sim_tick": self.pop.tick,
            "resolution": "RESOLVED",
            "population_hash": self.pop.population_hash(),
            "snapshot": snap,
        })
        return {"checkpoint_name": checkpoint_name,
                "population_hash": self.pop.population_hash(),
                "sim_tick": self.pop.tick}

    def ledger_summary(self) -> Dict[str, Any]:
        return {
            "mode": self.config.mode,
            "resolution": "full" if self.config.mode == "EXACT" else "coarse",
            "approximation_model": self.ledger["approximation_model"],
            "validation_method": self.ledger["validation_method"],
            "coarse_steps": self.ledger["coarse_steps"],
            "sim_ticks": self.ledger["sim_ticks"],
            "generations_seen": self.ledger["generations_seen"],
            "births": self.ledger["births"],
            "deaths": self.ledger["deaths"],
            "teaching_sessions": self.ledger["teaching_sessions"],
            "max_species_count": max(
                (s["species_count"] for s in self.ledger["species_snapshots"]), default=1),
            "milestones": len(self.ledger["milestones"]),
            "events": len(self.events),
        }

    # ------------------------------------------------------------- replay
    def replay_high_resolution(self, checkpoint_name: str, ticks: int) -> Dict[str, Any]:
        """Restore an escalated checkpoint and continue at full resolution.
        Verifies the restored state matches the checkpoint hash exactly."""
        ck = next((c for c in self.escalation_checkpoints
                   if c["checkpoint_name"] == checkpoint_name), None)
        if ck is None:
            return {"status": "FAILED", "reason": f"unknown checkpoint {checkpoint_name!r}"}
        from src.common.determinism import SeedBundle
        from src.population.population import Population
        seeds = SeedBundle(experiment_seed=self.pop.experiment_seed,
                           generation_seed=self.pop.seeds.generation_seed,
                           organism_seed=self.pop.seeds.organism_seed,
                           development_seed=self.pop.seeds.development_seed,
                           mutation_seed=self.pop.seeds.mutation_seed,
                           world_seed=self.pop.seeds.world_seed,
                           teacher_seed=self.pop.seeds.teacher_seed)
        restored = Population.restore(ck["snapshot"], seeds)
        hash_ok = (restored.population_hash() == ck["population_hash"])
        # full-resolution continuation (tick-level within the real simulation)
        restored.step(int(ticks))
        return {
            "status": "EXECUTED" if hash_ok else "FAILED",
            "checkpoint_hash_verified": hash_ok,
            "resolution": "full",
            "replay_ticks": int(ticks),
            "final_population_hash": restored.population_hash(),
            "checkpoint_population_hash": ck["population_hash"],
        }
