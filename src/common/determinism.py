"""Deterministic seeding + stable identity helpers (REAL, IMPLEMENTED)."""
import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedBundle:
    experiment_seed: int = 42
    generation_seed: int = 43
    organism_seed: int = 44
    development_seed: int = 45
    mutation_seed: int = 46
    world_seed: int = 47
    teacher_seed: int = 48

    def derive(self, stream: str, salt: str = "") -> int:
        h = hashlib.sha256()
        h.update(str(self.experiment_seed).encode())
        h.update(b"|" + stream.encode())
        h.update(b"|" + str(salt).encode())
        h.update(b"|" + str((self.generation_seed, self.organism_seed,
                              self.development_seed, self.mutation_seed,
                              self.world_seed, self.teacher_seed)).encode())
        return int(h.hexdigest()[:8], 16)


def deterministic_id(*parts: str) -> str:
    """Stable SHA-256 identity; prefer over uuid4 for scientific entities."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"|")
    return h.hexdigest()[:16]


def derive_subseed(base_seed: int, stream: str, salt: str = "") -> int:
    h = hashlib.sha256(f"{base_seed}|{stream}|{salt}".encode())
    return int(h.hexdigest()[:8], 16)
