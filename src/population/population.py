"""Overlapping-generation population with selection + reproduction (REAL, IMPLEMENTED)."""
import hashlib
import json
from typing import Any, Dict, List, Optional
from src.connectome.types import GraphMode
from src.culture.transmission import teach, TeachingSession
from src.common.determinism import SeedBundle, derive_subseed
from src.common.events import EventLog
from src.genome.operators import mutate_genome, crossover_genomes
from src.genome.schema import Genome
from src.organism.organism import Organism, create_offspring_id
from src.world.environment import GridWorld, WorldConfig


class Population:
    def __init__(self, size: int, seeds: SeedBundle, graph_mode: GraphMode = GraphMode.SYNTHETIC_TEST,
                 circuit_size: int = 64, experiment_seed: int = 42,
                 autonomy_mode: bool = False, genome_version: str = "1.0"):
        self.seeds = seeds
        self.experiment_seed = int(experiment_seed)
        self.graph_mode = graph_mode
        self.circuit_size = int(circuit_size)
        self.autonomy_mode = bool(autonomy_mode)
        self.genome_version = genome_version
        self.tick = 0
        self.generation = 0
        self.repro_index = 0
        self.events = EventLog()
        self.organisms: List[Organism] = []
        self.genetic_lineage: Dict[str, List[str]] = {}   # child -> parents
        self.cultural_lineage: Dict[str, List[str]] = {}  # student -> teachers
        self.teaching_sessions: List[TeachingSession] = []
        self.events.log("GENERATION_STARTED", 0, "", 0, {"size": size})
        self.world = GridWorld(WorldConfig(width=16, height=16,
                                           n_resources=max(12, size * 4),
                                           n_hazards=max(1, min(4, size // 3)),
                                           world_seed=seeds.world_seed))
        for i in range(size):
            genome = Genome.founder(derive_subseed(seeds.mutation_seed, f"founder:{i}"),
                                    legacy=(genome_version == "1.0"))
            oid = create_offspring_id(experiment_seed, 0, ["founder"], i, genome.genome_hash())
            org = Organism(genome, oid, generation=0,
                           seeds={"organism_seed": derive_subseed(seeds.organism_seed, f"org:{i}"),
                                  "development_seed": derive_subseed(seeds.development_seed, f"dev:{i}")},
                           graph_mode=graph_mode, circuit_size=circuit_size,
                           parents=[], birth_tick=0, start_pos=(i % 16, (i * 3) % 16),
                           autonomy_mode=autonomy_mode)
            self.organisms.append(org)
            self.world.place(oid, org.position)
            self.genetic_lineage[oid] = []

    def living(self) -> List[Organism]:
        return [o for o in self.organisms if o.alive]

    def step(self, n_ticks: int = 1):
        for _ in range(n_ticks):
            self.tick += 1
            self.world.regrow(self.tick, interval=3)
            for org in list(self.living()):
                org.step(self.world)
            # social: nearby adults teach juveniles (deterministic pairing)
            living = self.living()
            for student in living:
                if student.stage.value in ("infancy", "juvenile"):
                    teachers = [t for t in living
                                if t.id != student.id and t.stage.value in ("adult", "elder")
                                and abs(t.position[0] - student.position[0])
                                + abs(t.position[1] - student.position[1]) <= 3]
                    if teachers:
                        import hashlib as _h
                        legacy_idx = int(_h.sha256(f"{student.id}:{self.tick}".encode()).hexdigest(), 16) \
                            % len(teachers)
                        # v2: social_learning_bias gene shifts teacher choice toward
                        # trusted partners (emergent trust, STAGE H). v1 students
                        # keep the exact legacy pairing (compat).
                        bias = 0.0
                        if student.social_mem is not None and student.genome.version == "2.0":
                            bias = float(student.genome.get("social_learning_bias", 0.0))
                        if bias > 0.05:
                            def _score(t):
                                jitter = int(_h.sha256(f"{t.id}:{student.id}:{self.tick}"
                                                       .encode()).hexdigest(), 16) / float(2 ** 256)
                                tr = student.social_mem.trust_of(t.id)
                                return tr * bias + jitter * (1.0 - bias)
                            pick = max(teachers, key=_score)
                        else:
                            pick = teachers[legacy_idx]
                        sess = teach(pick, student, "forage", self.tick, self.seeds.teacher_seed)
                        self.teaching_sessions.append(sess)
                        self.cultural_lineage.setdefault(student.id, []).append(pick.id)
                        # social memory records the interaction outcome (both sides)
                        if student.social_mem is not None:
                            student.social_mem.record_interaction(
                                pick.id, self.tick, "taught_by",
                                min(1.0, sess.learning_gain * 2.0))
                        if pick.social_mem is not None:
                            pick.social_mem.record_interaction(
                                student.id, self.tick, "taught_to",
                                min(1.0, sess.learning_gain * 1.5))
            self.events.log("WORLD_STEP", self.tick, "", self.generation,
                            {"living": len(self.living())})

    def eligible_parents(self) -> List[Organism]:
        out = []
        for o in self.living():
            if o.stage.value in ("adult", "elder") and o.energy > 0.4 and o.health > 0.4 \
                    and o.age > 30:
                out.append(o)
        return out

    def reproduce(self, n_offspring: int = 2, mode: str = "sexual",
                  selection: str = "random") -> List[Organism]:
        """Reproduction. selection: 'random' (legacy default) or 'pareto'
        (multi-objective non-dominated selection on age/learning/efficiency)."""
        parents = self.eligible_parents()
        if selection == "pareto":
            pool = self.select_parents_pareto(len(parents) or 0)
            if pool:
                parents = pool
        elif selection != "random":
            raise ValueError(f"unknown selection {selection!r}")
        newborns = []
        if len(parents) < (2 if mode == "sexual" else 1):
            return newborns
        import numpy as np
        rng = np.random.RandomState(derive_subseed(self.seeds.generation_seed, f"repro:{self.tick}"))
        for k in range(n_offspring):
            if mode == "sexual":
                i, j = rng.choice(len(parents), size=2, replace=False)
                pa, pb = parents[int(i)], parents[int(j)]
                child_genome, xrec = crossover_genomes(
                    pa.genome, pb.genome,
                    derive_subseed(self.seeds.mutation_seed, f"xover:{self.tick}:{k}"))
                self.events.log("CROSSOVER", self.tick, "", self.generation,
                                {"parents": [pa.id, pb.id]})
                parent_ids = [pa.id, pb.id]
                gen = max(pa.generation, pb.generation) + 1
            else:
                pa = parents[int(rng.randint(len(parents)))]
                child_genome, xrec = crossover_genomes(
                    pa.genome, pa.genome,
                    derive_subseed(self.seeds.mutation_seed, f"asex:{self.tick}:{k}"),
                    mode="asexual")
                parent_ids = [pa.id]
                gen = pa.generation + 1
            self.events.log("MUTATION", self.tick, "", gen,
                            {"mutations": xrec.get("post_mutations", xrec.get("mutations", []))})
            # reproduction cost (energy cannot appear spontaneously)
            for pid in parent_ids:
                p = next(o for o in self.organisms if o.id == pid)
                p.energy = max(0.0, p.energy - 0.25)
            oid = create_offspring_id(self.experiment_seed, gen, parent_ids,
                                      self.repro_index, child_genome.genome_hash())
            self.repro_index += 1
            child = Organism(child_genome, oid, generation=gen,
                             seeds={"organism_seed": derive_subseed(self.seeds.organism_seed, oid),
                                    "development_seed": derive_subseed(self.seeds.development_seed, oid)},
                             graph_mode=self.graph_mode, circuit_size=self.circuit_size,
                             parents=parent_ids, birth_tick=self.tick,
                             start_pos=(self.tick % 16, (self.tick * 5) % 16),
                             autonomy_mode=self.autonomy_mode)
            for pid in parent_ids:
                p = next(o for o in self.organisms if o.id == pid)
                p.children.append(oid)
            self.organisms.append(child)
            self.world.place(oid, child.position)
            self.genetic_lineage[oid] = list(parent_ids)
            self.events.log("REPRODUCTION", self.tick, oid, gen, {"parents": parent_ids})
            self.events.log("ORGANISM_BORN", self.tick, oid, gen,
                            {"genome_hash": child_genome.genome_hash()})
            newborns.append(child)
        return newborns

    def select_parents_pareto(self, k: int = 4) -> List[Organism]:
        """Pareto-nondominated sort on (age, learning, energy_efficiency); returns top-k."""
        living = self.living()
        if not living:
            return []
        vecs = [(o, o.fitness_vector()) for o in living]
        def dominates(a, b):
            return (a["age"] >= b["age"] and a["learning"] >= b["learning"]
                    and a["energy_efficiency"] >= b["energy_efficiency"]
                    and (a["age"] > b["age"] or a["learning"] > b["learning"]
                         or a["energy_efficiency"] > b["energy_efficiency"]))
        fronts, remaining = [], list(vecs)
        while remaining:
            front = [x for x in remaining if not any(dominates(y[1], x[1]) for y in remaining if y is not x)]
            fronts.append(front)
            remaining = [x for x in remaining if x not in front]
        out = []
        for front in fronts:
            front.sort(key=lambda x: x[0].id)
            for x in front:
                out.append(x[0])
                if len(out) >= k:
                    return out
        return out

    def population_hash(self) -> str:
        h = hashlib.sha256()
        for o in sorted(self.organisms, key=lambda x: x.id):
            h.update(o.organism_hash().encode())
        h.update(self.world.world_hash().encode())
        return h.hexdigest()

    def snapshot(self) -> Dict[str, Any]:
        return {"tick": self.tick, "generation": self.generation,
                "repro_index": self.repro_index, "experiment_seed": self.experiment_seed,
                "graph_mode": self.graph_mode.value, "circuit_size": self.circuit_size,
                "autonomy_mode": self.autonomy_mode, "genome_version": self.genome_version,
                "world": self.world.snapshot(),
                "organisms": [o.snapshot() for o in self.organisms],
                "genetic_lineage": self.genetic_lineage,
                "cultural_lineage": self.cultural_lineage}

    @classmethod
    def restore(cls, snap: Dict[str, Any], seeds: SeedBundle) -> "Population":
        pop = cls.__new__(cls)
        pop.seeds = seeds
        pop.experiment_seed = snap["experiment_seed"]
        pop.graph_mode = GraphMode(snap["graph_mode"])
        pop.circuit_size = snap["circuit_size"]
        pop.autonomy_mode = bool(snap.get("autonomy_mode", False))
        pop.genome_version = snap.get("genome_version", "1.0")
        pop.tick, pop.generation, pop.repro_index = snap["tick"], snap["generation"], snap["repro_index"]
        pop.world = GridWorld(WorldConfig())
        pop.world.restore(snap["world"])
        pop.organisms = [Organism.restore(s) for s in snap["organisms"]]
        pop.genetic_lineage = dict(snap["genetic_lineage"])
        pop.cultural_lineage = dict(snap["cultural_lineage"])
        pop.teaching_sessions = []
        pop.events = EventLog()
        return pop
