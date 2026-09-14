"""State hashing helpers (REAL, IMPLEMENTED)."""
import hashlib
import json
import numpy as np
from typing import Any, Dict


def hash_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        arr = np.ascontiguousarray(a)
        h.update(str(arr.shape).encode() + b"|" + str(arr.dtype).encode() + b"|")
        h.update(arr.tobytes())
    return h.hexdigest()


def hash_dict(d: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def quantize_for_hash(weights: np.ndarray, scale: float = 1e6) -> bytes:
    wq = np.round(np.ascontiguousarray(weights, dtype=np.float64) * scale).astype(np.int64)
    return wq.tobytes()
