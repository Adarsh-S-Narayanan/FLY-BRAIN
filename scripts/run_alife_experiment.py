"""Canonical overlapping-generation ALife experiment (REAL, IMPLEMENTED).

Runs: founders -> sensorimotor life -> development -> sleep/dream cycles ->
social teaching -> periodic sexual reproduction -> selection tracking.
Produces a reproducible manifest with lineage trees and growth curves.

Usage:
    python scripts/run_alife_experiment.py --population 10 --ticks 150 --seed 7
    python scripts/run_alife_experiment.py --verify <manifest_path>
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from src.common.determinism import SeedBundle
from src.common.hashing import hash_dict
from src.connectome.types import GraphMode
from src.population.population import Population


def build_seeds(seed: int) -> SeedBundle:
    return SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                      organism_seed=seed + 2, development_seed=seed + 3,
                      mutation_seed=seed + 4, world_seed=seed + 5,
                      teacher_seed=seed + 6)


def run(population: int, ticks: int, seed: int, circuit_size: int,
        out_dir: str = "diagnostics/alife_experiments",
        substrate: str = "synthetic") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    seeds = build_seeds(seed)
    mode = {"synthetic": GraphMode.SYNTHETIC_TEST, "real": GraphMode.REAL,
            "surrogate": GraphMode.SPATIAL_SURROGATE}[substrate]
    pop = Population(population, seeds, mode, circuit_size,
                     experiment_seed=seed)
    repro_every, sleep_every = 15, 25
    births, deaths0 = 0, 0
    curves = {"tick": [], "living": [], "mean_neurons": [], "mean_synapses": [],
              "mean_energy": [], "teaching_sessions": [], "births_total": []}
    for t in range(0, ticks, 5):
        pop.step(5)
        if (t + 5) % sleep_every == 0:
            for o in pop.living()[:4]:
                o.sleep()
        if (t + 5) % repro_every == 0:
            births += len(pop.reproduce(2, mode="sexual"))
        living = pop.living()
        curves["tick"].append(t + 5)
        curves["living"].append(len(living))
        curves["mean_neurons"].append(
            round(sum(o.graph.num_neurons for o in living) / max(1, len(living)), 2))
        curves["mean_synapses"].append(
            round(sum(o.graph.num_synapses for o in living) / max(1, len(living)), 2))
        curves["mean_energy"].append(
            round(sum(o.energy for o in living) / max(1, len(living)), 4))
        curves["teaching_sessions"].append(len(pop.teaching_sessions))
        curves["births_total"].append(births)
    deaths = sum(1 for o in pop.organisms if not o.alive)
    generations = sorted({o.generation for o in pop.organisms})
    manifest = {
        "experiment": "canonical_alife_overlap",
        "seed": seed,
        "substrate": substrate,
        "population_start": population,
        "population_total": len(pop.organisms),
        "ticks": ticks,
        "circuit_size": circuit_size,
        "graph_mode": mode.value,
        "births": births,
        "deaths": deaths,
        "generations_present": generations,
        "overlap_demonstrated": len(generations) > 1 and any(o.alive for o in pop.organisms if o.generation == 0),
        "teaching_sessions": len(pop.teaching_sessions),
        "genetic_lineage": pop.genetic_lineage,
        "cultural_lineage": pop.cultural_lineage,
        "curves": curves,
        "population_hash": pop.population_hash(),
        "events_hash": pop.events.compute_hash(),
        "event_count": len(pop.events),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    exp_id = f"alife_p{population}_t{ticks}_s{seed}"
    path = os.path.join(out_dir, exp_id + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[alife] done: living={curves['living'][-1]} births={births} deaths={deaths} "
          f"gens={generations} teaching={len(pop.teaching_sessions)} hash={manifest['population_hash'][:16]}")
    print(f"[alife] manifest: {path}")
    return manifest


def verify(manifest_path: str) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    m2 = run(m["population_start"], m["ticks"], m["seed"], m["circuit_size"],
             out_dir=os.path.dirname(manifest_path) + "_reverify_tmp",
             substrate=m.get("substrate", "synthetic"))
    match = (m2["population_hash"] == m["population_hash"])
    print(f"[alife] verify: original={m['population_hash'][:16]} "
          f"reproduced={m2['population_hash'][:16]} match={match}")
    import shutil
    shutil.rmtree(os.path.dirname(manifest_path) + "_reverify_tmp", ignore_errors=True)
    return {"deterministic_match": match,
            "original": m["population_hash"], "reproduced": m2["population_hash"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--circuit-size", type=int, default=32)
    ap.add_argument("--substrate", choices=["synthetic", "real", "surrogate"],
                    default="synthetic")
    ap.add_argument("--verify", type=str, default="")
    args = ap.parse_args()
    if args.verify:
        ok = verify(args.verify)["deterministic_match"]
        sys.exit(0 if ok else 1)
    run(args.population, args.ticks, args.seed, args.circuit_size,
        substrate=args.substrate)
