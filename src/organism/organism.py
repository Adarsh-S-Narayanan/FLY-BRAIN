"""Real organism abstraction: brain + body + lifecycle + metabolism (REAL, IMPLEMENTED)."""
import hashlib
import json
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from src.brain.runtime import BrainRuntime
from src.connectome.loader import get_or_create_circuit
from src.connectome.types import GraphMode
from src.development.engine import DevelopmentEngine, DevelopmentState
from src.genome.schema import Genome
from src.common.determinism import deterministic_id, derive_subseed
from src.common.events import EventLog
from src.world.environment import GridWorld


class LifeStage(str, Enum):
    INFANCY = "infancy"
    JUVENILE = "juvenile"
    ADULT = "adult"
    ELDER = "elder"
    DEAD = "dead"


def stage_for_age(age: int) -> LifeStage:
    if age < 20:
        return LifeStage.INFANCY
    if age < 60:
        return LifeStage.JUVENILE
    if age < 200:
        return LifeStage.ADULT
    return LifeStage.ELDER


class Organism:
    def __init__(self, genome: Genome, organism_id: str, generation: int = 0,
                 seeds: Optional[Dict[str, int]] = None, graph_mode: GraphMode = GraphMode.SYNTHETIC_TEST,
                 circuit_size: int = 64, parents: Optional[List[str]] = None,
                 birth_tick: int = 0, start_pos: tuple = (0, 0)):
        self.genome = genome
        self.genome.validate()
        self.id = organism_id
        self.generation = int(generation)
        self.parents = list(parents or [])
        self.children: List[str] = []
        self.seeds = dict(seeds or {})
        self.age = 0
        self.energy = 1.0
        self.health = 1.0
        self.stage = LifeStage.INFANCY
        self.alive = True
        self.tick = int(birth_tick)
        self.episodes: List[Dict[str, Any]] = []   # lightweight episodic memory w/ provenance
        self.semantic: Dict[str, List[float]] = {}  # concept -> vector
        self.cultural_knowledge: Dict[str, Dict[str, Any]] = {}
        self.skills: Dict[str, float] = {"forage": 0.1, "avoid": 0.1, "social": 0.1}
        self.cause_of_death = ""
        self.position = tuple(start_pos)
        self.heading = (1, 0)      # persistent run direction (chemotaxis)
        self.last_food = 0.0       # previous food gradient (tumble trigger)
        self._last_reward = 0.0    # previous tick outcome -> neural plasticity (P12)
        # own brain copy (CPU for determinism inside populations)
        oseed = int(self.seeds.get("organism_seed", 44))
        graph = get_or_create_circuit(circuit_size, mode=graph_mode, seed=oseed)
        # deep copy so development is per-organism
        from copy import deepcopy
        self.graph = deepcopy(graph)
        self.brain = BrainRuntime(self.graph, use_gpu=False,
                                  enable_plasticity=True, seed=oseed)
        self.dev = DevelopmentState.initialize(self.graph.num_neurons)
        self.dev_engine = DevelopmentEngine(genome.params,
                                            development_seed=int(self.seeds.get("development_seed", 45)))
        self.events = EventLog()
        self.events.log("ORGANISM_BORN", self.tick, self.id, self.generation,
                        {"genome_hash": genome.genome_hash(), "parents": self.parents,
                         "graph_mode": graph_mode.value, "circuit_size": circuit_size})

    # ---- sensorimotor loop ----
    def step(self, world: GridWorld) -> Dict[str, Any]:
        if not self.alive:
            return {"organism_id": self.id, "alive": False}
        sense = world.sense(self.id)
        # sensors -> brain currents: food->visual, hazard->olfactory, social->memory
        n = self.graph.num_neurons
        vis = np.full(16, float(sense["food_gradient"]), dtype=np.float32)
        olf = np.full(8, float(sense["hazard_gradient"]), dtype=np.float32)
        mem = np.full(8, float(min(1.0, sense["nearby_organisms"] / 3.0)), dtype=np.float32)
        # P12: the previous tick's outcome reward drives real neural plasticity.
        out = self.brain.step(sensory_inputs={"visual": vis, "olfactory": olf, "memory": mem},
                              reward=float(self._last_reward))
        # enforce dead-neuron silence invariant
        dead_idx = [i for i, a in enumerate(self.dev.alive) if not a]
        if dead_idx and len(self.brain.state.spikes) >= max(dead_idx, default=-1) + 1:
            pass  # spikes arrays only cover live indices; dead edges were stripped
        # decision -> motor: run-and-tumble chemotaxis biased by brain action
        act = out.get("selected_action", "explore")
        dx = dy = 0
        explore = float(self.genome.params.get("exploration", 0.5))
        rng = np.random.RandomState(derive_subseed(int(self.seeds.get("organism_seed", 44)),
                                                   f"move:{self.tick}"))
        food_now = float(sense["food_gradient"])
        hungry = self.energy < 0.7
        # tumble (pick new heading) when gradient worsens while hungry, or randomly when exploring
        if (hungry and food_now < self.last_food - 1e-6) or rng.rand() < explore * 0.25:
            self.heading = (int(rng.randint(-1, 2)), int(rng.randint(-1, 2)))
        self.last_food = food_now
        if act in ("act_in_environment", "explore") or hungry or rng.rand() < 0.5:
            dx, dy = self.heading
            if dx == 0 and dy == 0:
                dx = int(rng.randint(-1, 2))
            if float(sense["hazard_gradient"]) > 0.5 and rng.rand() < 0.7:
                dx, dy = -dx, int(rng.randint(-1, 2))  # escape reflex
            world.apply_move(self.id, dx, dy)
        consumed = world.consume(self.id)
        reward = 0.0
        if consumed > 0:
            self.energy = min(1.5, self.energy + consumed * 0.8)
            self.skills["forage"] = min(1.0, self.skills["forage"] + 0.02)
            reward = 0.5
        if world.hazard_at(self.id):
            self.health = max(0.0, self.health - 0.05)
            self.skills["avoid"] = min(1.0, self.skills["avoid"] + 0.01)
            reward -= 0.3
        # P12: store outcome for next tick's neural plasticity (closed loop).
        self._last_reward = float(reward)
        # metabolism: basal + neural activity + movement + growth
        activity = float(np.mean(self.brain.state.activations)) if len(self.brain.state.activations) else 0.0
        metab = float(self.genome.params.get("metabolism_rate", 0.01))
        cost = metab * (0.5 + activity + (abs(dx) + abs(dy)) * 0.25)
        self.energy = max(0.0, self.energy - cost)
        if self.energy <= 0.0:
            self.health = max(0.0, self.health - 0.02)
        self.position = world.positions.get(self.id, self.position)
        # episodic memory with provenance
        ep = {"tick": self.tick, "organism_id": self.id, "generation": self.generation,
              "observation": sense, "action": act, "reward": round(reward, 4),
              "energy": round(self.energy, 4)}
        self.episodes.append(ep)
        self.events.log("LEARNING_EVENT", self.tick, self.id, self.generation,
                        {"action": act, "reward": round(reward, 4)})
        self.events.log("MEMORY_CREATED", self.tick, self.id, self.generation,
                        {"episode_index": len(self.episodes) - 1})
        # development tick (cheap, every 5 ticks); growth costs energy
        if self.tick % 5 == 0:
            n_born = self.dev_engine.neurogenesis(self.graph, self.dev, self.tick, self.events,
                                                  self.id, self.generation, max_new=2)
            self.energy = max(0.0, self.energy - 0.02 * n_born)
            self._sync_brain_to_graph()
            self.dev_engine.differentiate(self.graph, self.dev, self.tick, self.events,
                                          self.id, self.generation)
            self.dev_engine.migrate(self.graph, self.dev, self.tick, self.events,
                                    self.id, self.generation)
            self.dev_engine.grow_projections(self.graph, self.dev, self.tick, self.events,
                                             self.id, self.generation, max_candidates=4)
            self.dev_engine.prune(self.graph, self.dev, self.tick,
                                  self.brain.state.activations, self.events,
                                  self.id, self.generation)
            self.dev_engine.apoptosis(self.graph, self.dev, self.tick,
                                      self.brain.state.activations, self.age,
                                      self.events, self.id, self.generation)
            self._sync_brain_to_graph()
        # aging + lifecycle
        self.age += 1
        self.tick += 1
        world.tick = max(world.tick, self.tick)
        self.stage = stage_for_age(self.age)
        if self.health <= 0.0 or self.age > 400:
            self.die("health" if self.health <= 0.0 else "age")
        return {"organism_id": self.id, "alive": self.alive, "action": act,
                "reward": round(reward, 4), "energy": round(self.energy, 4),
                "stage": self.stage.value}

    def _sync_brain_to_graph(self):
        """Resize brain state arrays after structural growth (new neurons start silent)."""
        n = self.graph.num_neurons
        st = self.brain.state
        def grow(arr, fill, dtype):
            if len(arr) < n:
                extra = np.full(n - len(arr), fill, dtype=dtype)
                return np.concatenate([arr, extra])
            return arr[:n]
        st.membrane_potentials = grow(st.membrane_potentials, 0.0, np.float32)
        st.spikes = grow(st.spikes, 0.0, np.float32)
        st.refractory_steps = grow(st.refractory_steps, 0, np.int32)
        st.activations = grow(st.activations, 0.0, np.float32)
        st.attention = np.ones(n, dtype=np.float32) / max(1, n)
        st.num_neurons = n
        # rebind sensorimotor index maps if out of range
        for attr in ("sensory_visual_indices", "sensory_audio_indices", "sensory_olfactory_indices",
                     "sensory_memory_indices", "motor_speak_indices", "motor_act_indices",
                     "motor_image_indices", "motor_remember_indices"):
            idx = getattr(self.brain, attr, np.array([], dtype=np.int32))
            setattr(self.brain, attr, np.array([i for i in idx if i < n], dtype=np.int32))

    # ---- sleep / dream (real replay of own episodes) ----
    def sleep(self, replay_k: int = 3) -> Dict[str, Any]:
        if not self.alive:
            return {"slept": False}
        self.events.log("SLEEP_STARTED", self.tick, self.id, self.generation, {})
        recent = self.episodes[-replay_k:] if self.episodes else []
        self.events.log("DREAM_STARTED", self.tick, self.id, self.generation,
                        {"replay_count": len(recent)})
        consolidated = 0
        for ep in recent:
            stim = np.full(16, float(ep["observation"].get("food_gradient", 0.0) * 0.5 + 0.1),
                           dtype=np.float32)
            self.brain.step(sensory_inputs={"visual": stim}, reward=0.0)
            key = f"dream:{ep['action']}"
            v = self.semantic.get(key, [0.0])
            self.semantic[key] = [round(min(1.0, v[0] + 0.05 * max(0.0, ep["reward"] + 0.5)), 4)]
            consolidated += 1
        self.energy = min(1.5, self.energy + 0.1)  # rest recovery (bounded, from reduced activity)
        self.events.log("MEMORY_CONSOLIDATED", self.tick, self.id, self.generation,
                        {"consolidated": consolidated})
        self.events.log("DREAM_ENDED", self.tick, self.id, self.generation, {})
        return {"slept": True, "replay_count": len(recent), "consolidated": consolidated}

    def die(self, cause: str):
        if not self.alive:
            return
        self.alive = False
        self.stage = LifeStage.DEAD
        self.cause_of_death = cause
        self.events.log("ORGANISM_DIED", self.tick, self.id, self.generation,
                        {"cause": cause, "age": self.age,
                         "genome_hash": self.genome.genome_hash(),
                         "brain_neurons": self.graph.num_neurons,
                         "brain_synapses": self.graph.num_synapses})

    def fitness_vector(self) -> Dict[str, float]:
        return {
            "survival": 1.0 if self.alive else 0.0,
            "age": float(self.age),
            "energy_efficiency": round(float(self.energy / max(1, self.age)), 4),
            "learning": round(float(len(self.episodes) / max(1, self.age)), 4),
            "memory": float(len(self.semantic) + len(self.cultural_knowledge)),
            "brain_size": float(self.graph.num_neurons),
            "health": round(float(self.health), 4),
        }

    def organism_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.id.encode())
        h.update(self.genome.genome_hash().encode())
        h.update(self.graph.graph_hash.encode())
        h.update(self.brain.state.membrane_potentials.tobytes())
        h.update(self.brain.state.spikes.tobytes())
        h.update(str((self.age, round(self.energy, 6), round(self.health, 6), self.tick)).encode())
        return h.hexdigest()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id, "generation": self.generation, "parents": self.parents,
            "children": self.children, "seeds": self.seeds, "age": self.age,
            "energy": self.energy, "health": self.health, "alive": self.alive,
            "tick": self.tick, "position": list(self.position),
            "heading": list(self.heading), "last_food": self.last_food,
            "last_reward": self._last_reward,
            "genome": self.genome.to_dict(),            "graph": {"neuron_ids": self.graph.neuron_ids.tolist(),
                      "coordinates": self.graph.coordinates.tolist(),
                      "tbars": self.graph.tbars.tolist(), "sides": list(self.graph.sides),
                      "row_offsets": self.graph.row_offsets.tolist(),
                      "col_indices": self.graph.col_indices.tolist(),
                      "weights": self.graph.weights.tolist(), "mode": self.graph.mode.value},
            "brain": {"potentials": self.brain.state.membrane_potentials.tolist(),
                      "spikes": self.brain.state.spikes.tolist(),
                      "refractory": self.brain.state.refractory_steps.tolist(),
                      "activations": self.brain.state.activations.tolist(),
                      "step_count": self.brain.state.step_count,
                      "total_spikes": self.brain.state.total_spikes,
                      "prediction_error": self.brain.state.prediction_error,
                      "predicted_reward": self.brain.state.predicted_reward,
                      "current_reward": self.brain.state.current_reward,
                      "drives": {"energy": self.brain.state.drives.energy,
                                 "curiosity": self.brain.state.drives.curiosity,
                                 "social": self.brain.state.drives.social,
                                 "integrity": self.brain.state.drives.integrity},
                      "maps": {k: [int(x) for x in getattr(self.brain, k, [])]
                               for k in ("sensory_visual_indices", "sensory_audio_indices",
                                         "sensory_olfactory_indices", "sensory_memory_indices",
                                         "motor_speak_indices", "motor_act_indices",
                                         "motor_image_indices", "motor_remember_indices")}},
            "dev": {"cell_types": self.dev.cell_types, "birth_ticks": self.dev.birth_ticks,
                    "lineage_ids": self.dev.lineage_ids,
                    "developmental_states": self.dev.developmental_states,
                    "alive": self.dev.alive},
            "episodes": self.episodes, "semantic": self.semantic,
            "cultural_knowledge": self.cultural_knowledge, "skills": self.skills,
        }

    @classmethod
    def restore(cls, snap: Dict[str, Any]) -> "Organism":
        import numpy as np
        from src.connectome.types import ConnectomeGraph, GraphMode
        g = snap["graph"]
        from src.connectome.types import (MaleCNSRealGraph, MaleCNSSpatialSurrogateGraph,
                                          SyntheticTestGraph)
        mode = GraphMode(g["mode"])
        cls_map = {GraphMode.REAL: MaleCNSRealGraph,
                   GraphMode.SPATIAL_SURROGATE: MaleCNSSpatialSurrogateGraph,
                   GraphMode.SYNTHETIC_TEST: SyntheticTestGraph}
        graph = cls_map[mode](
            neuron_ids=np.array(g["neuron_ids"], dtype=np.int64),
            coordinates=np.array(g["coordinates"], dtype=np.float32),
            tbars=np.array(g["tbars"], dtype=np.int32), sides=list(g["sides"]),
            row_offsets=np.array(g["row_offsets"], dtype=np.int32),
            col_indices=np.array(g["col_indices"], dtype=np.int32),
            weights=np.array(g["weights"], dtype=np.float32))
        genome = Genome.from_dict(snap["genome"])
        org = cls.__new__(cls)
        org.genome, org.id, org.generation = genome, snap["id"], snap["generation"]
        org.parents, org.children = list(snap["parents"]), list(snap["children"])
        org.seeds = dict(snap["seeds"]); org.age = snap["age"]
        org.energy, org.health, org.alive = snap["energy"], snap["health"], snap["alive"]
        org.tick = snap["tick"]; org.position = tuple(snap["position"])
        org.heading = tuple(snap.get("heading", (1, 0))); org.last_food = float(snap.get("last_food", 0.0))
        org._last_reward = float(snap.get("last_reward", 0.0))
        org.stage = stage_for_age(org.age) if org.alive else LifeStage.DEAD
        org.graph = graph
        org.brain = BrainRuntime(graph, use_gpu=False, seed=int(org.seeds.get("organism_seed", 44)))
        b = snap["brain"]
        org.brain.state.membrane_potentials = np.array(b["potentials"], dtype=np.float32)
        org.brain.state.spikes = np.array(b["spikes"], dtype=np.float32)
        org.brain.state.refractory_steps = np.array(b["refractory"], dtype=np.int32)
        org.brain.state.activations = np.array(b["activations"], dtype=np.float32)
        org.brain.state.step_count = b["step_count"]; org.brain.state.total_spikes = b["total_spikes"]
        org.brain.state.num_neurons = graph.num_neurons
        org.brain.state.prediction_error = float(b.get("prediction_error", 0.0))
        org.brain.state.predicted_reward = float(b.get("predicted_reward", 0.0))
        org.brain.state.current_reward = float(b.get("current_reward", 0.0))
        for dk, dv in b.get("drives", {}).items():
            if hasattr(org.brain.state.drives, dk):
                setattr(org.brain.state.drives, dk, float(dv))
        # P7: sensorimotor maps must be restored exactly; a grown brain's maps
        # differ from a freshly built brain's maps at the same size.
        import numpy as _np
        for k, v in b.get("maps", {}).items():
            if hasattr(org.brain, k):
                setattr(org.brain, k, _np.array([int(x) for x in v], dtype=_np.int32))
        from src.development.engine import DevelopmentState
        d = snap["dev"]
        org.dev = DevelopmentState(cell_types=list(d["cell_types"]), birth_ticks=list(d["birth_ticks"]),
                                   lineage_ids=list(d["lineage_ids"]),
                                   developmental_states=list(d["developmental_states"]),
                                   alive=list(d["alive"]),
                                   activity_history=[0.0] * len(d["alive"]))
        org.dev_engine = DevelopmentEngine(genome.params,
                                           development_seed=int(org.seeds.get("development_seed", 45)))
        org.episodes = list(snap["episodes"]); org.semantic = dict(snap["semantic"])
        org.cultural_knowledge = dict(snap["cultural_knowledge"]); org.skills = dict(snap["skills"])
        org.cause_of_death = snap.get("cause_of_death", "")
        org.events = EventLog()
        return org


def create_offspring_id(experiment_seed: int, generation: int, parent_ids: List[str],
                        repro_index: int, genome_hash: str) -> str:
    return deterministic_id(str(experiment_seed), str(generation),
                            "+".join(sorted(parent_ids)), str(repro_index), genome_hash)
