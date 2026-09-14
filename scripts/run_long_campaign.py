"""Multi-generation ALife campaign with checkpoint/resume (REAL, IMPLEMENTED).

Usage:
  python scripts/run_long_campaign.py --generations 10 --population 6 --seed 11
  python scripts/run_long_campaign.py --resume diagnostics/campaigns/<id>/checkpoint.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from src.common.determinism import SeedBundle
from src.connectome.types import GraphMode
from src.population.population import Population

OUT_ROOT = "diagnostics/campaigns"
TICKS_PER_GEN = 20


def seeds_for(seed: int) -> SeedBundle:
    return SeedBundle(experiment_seed=seed, generation_seed=seed + 1,
                      organism_seed=seed + 2, development_seed=seed + 3,
                      mutation_seed=seed + 4, world_seed=seed + 5,
                      teacher_seed=seed + 6)


def _campaign_dir(seed, population) -> str:
    return os.path.join(OUT_ROOT, f"camp_{seed}_{population}")


def _checkpoint(pop: Population, path: str, generation_done: int, seed: int,
                population: int, curves: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "seed": seed, "population": population, "generation_done": generation_done,
        "population_snapshot": pop.snapshot(), "curves": curves,
        "population_hash": pop.population_hash(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def _curves_point(pop: Population, generation: int) -> dict:
    live = pop.living()
    return {
        "generation": generation,
        "tick": pop.tick,
        "living": len(live),
        "total": len(pop.organisms),
        "mean_energy": round(sum(o.energy for o in live) / max(1, len(live)), 4),
        "mean_neurons": round(sum(o.graph.num_neurons for o in live) / max(1, len(live)), 2),
        "teaching_sessions": len(pop.teaching_sessions),
        "population_hash": pop.population_hash(),
    }


def run_campaign(generations: int, population: int, seed: int,
                 checkpoint_every: int = 2, substrate: str = "synthetic"):
    mode = {"synthetic": GraphMode.SYNTHETIC_TEST, "real": GraphMode.REAL,
            "surrogate": GraphMode.SPATIAL_SURROGATE}[substrate]
    cdir = _campaign_dir(seed, population)
    os.makedirs(cdir, exist_ok=True)
    seeds = seeds_for(seed)
    pop = Population(population, seeds, mode, 32, experiment_seed=seed)
    curves = []
    t0 = time.time()
    gen_done = 0
    ckpt_path = os.path.join(cdir, "checkpoint.json")
    final_path = os.path.join(cdir, "final.json")
    for gen in range(generations):
        pop.step(TICKS_PER_GEN)
        pop.reproduce(2, mode="sexual")
        gen_done = gen + 1
        curves.append(_curves_point(pop, gen_done))
        print(f"[campaign] gen {gen_done}/{generations} living={curves[-1]['living']} "
              f"total={curves[-1]['total']} hash={curves[-1]['population_hash'][:12]}")
        if gen_done % checkpoint_every == 0 or gen_done == generations:
            _checkpoint(pop, ckpt_path, gen_done, seed, population, curves)
    manifest = {
        "campaign": "long_campaign", "seed": seed, "population_start": population,
        "generations": generations, "generations_done": gen_done,
        "substrate": substrate, "ticks_per_gen": TICKS_PER_GEN,
        "elapsed_sec": round(time.time() - t0, 2), "curves": curves,
        "final_population_hash": pop.population_hash(),
        "checkpoint": ckpt_path,
    }
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[campaign] manifest: {final_path} final_hash={manifest['final_population_hash'][:12]}")
    return manifest


def resume_campaign(checkpoint_path: str, extra_generations: int):
    with open(checkpoint_path, encoding="utf-8") as f:
        payload = json.load(f)
    seeds = seeds_for(payload["seed"])
    pop = Population.restore(payload["population_snapshot"], seeds)
    curves = list(payload["curves"])
    gen_done = payload["generation_done"]
    t0 = time.time()
    for _ in range(extra_generations):
        pop.step(TICKS_PER_GEN)
        pop.reproduce(2, mode="sexual")
        gen_done += 1
        curves.append(_curves_point(pop, gen_done))
        print(f"[campaign-resume] gen {gen_done} living={curves[-1]['living']} "
              f"hash={curves[-1]['population_hash'][:12]}")
    _checkpoint(pop, checkpoint_path, gen_done, payload["seed"], payload["population"], curves)
    out = dict(payload)
    out.update({"generations_done": gen_done, "curves": curves,
                "final_population_hash": pop.population_hash(),
                "elapsed_sec": round(time.time() - t0, 2)})
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[campaign-resume] done gen={gen_done} hash={out['final_population_hash'][:12]}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--population", type=int, default=6)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--substrate", choices=["synthetic", "real", "surrogate"],
                    default="synthetic")
    ap.add_argument("--checkpoint-every", type=int, default=2)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()
    if args.resume:
        resume_campaign(args.resume, args.generations)
    else:
        run_campaign(args.generations, args.population, args.seed,
                     checkpoint_every=args.checkpoint_every, substrate=args.substrate)