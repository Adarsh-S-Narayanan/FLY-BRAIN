"""Local GGUF model discovery (REAL, IMPLEMENTED).

Source of truth: *.gguf files physically present under llm/. Nothing is assumed
about architecture — all metadata is read from the GGUF KV store.
"""
import hashlib
import json
import os
import struct
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

MODEL_ROOTS = ("llm", "models")
MANIFEST_PATH = os.path.join("llm", "model_manifest.json")

FILE_TYPE_NAMES = {
    0: "ALL_F32", 1: "F32", 2: "MOSTLY_F16", 3: "MOSTLY_Q4_0", 4: "MOSTLY_Q4_1",
    5: "MOSTLY_Q4_1_SOME_F16", 6: "MOSTLY_Q4_2", 7: "MOSTLY_Q8_0",
    8: "MOSTLY_Q5_0", 9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K", 11: "MOSTLY_Q3_K_S",
    12: "MOSTLY_Q3_K_M", 13: "MOSTLY_Q3_K_L", 14: "MOSTLY_Q4_K_S",
    15: "MOSTLY_Q4_K_M", 16: "MOSTLY_Q5_K_S", 17: "MOSTLY_Q5_K_M",
    18: "MOSTLY_Q6_K",
}


def _read_exact(f, n: int, what: str) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError(f"truncated GGUF file while reading {what} "
                         f"(got {len(data)}/{n} bytes)")
    return data


def _read_gguf_kv(path: str, max_keys: int = 400) -> Dict[str, Any]:
    with open(path, "rb") as f:
        if _read_exact(f, 4, "magic") != b"GGUF":
            raise ValueError("not a GGUF file (bad magic)")
        _read_exact(f, 4, "version")  # version
        n_tensors = struct.unpack("<Q", _read_exact(f, 8, "tensor count"))[0]
        n_kv = struct.unpack("<Q", _read_exact(f, 8, "kv count"))[0]
        if n_kv > 100_000:
            raise ValueError(f"implausible GGUF kv count: {n_kv}")

        def read_str():
            n = struct.unpack("<Q", _read_exact(f, 8, "string length"))[0]
            if n > 10_000_000:
                raise ValueError("corrupt GGUF string length")
            return _read_exact(f, n, "string").decode("utf-8", "replace")

        def read_val(t):
            if t == 0:
                return struct.unpack("<B", _read_exact(f, 1, "u8"))[0]
            if t == 4:
                return struct.unpack("<i", _read_exact(f, 4, "i32"))[0]
            if t == 5:
                return struct.unpack("<I", _read_exact(f, 4, "u32"))[0]
            if t == 6:
                return struct.unpack("<f", _read_exact(f, 4, "f32"))[0]
            if t == 7:
                return bool(struct.unpack("<B", _read_exact(f, 1, "bool"))[0])
            if t == 8:
                return read_str()
            if t == 9:
                at = struct.unpack("<I", _read_exact(f, 4, "array type"))[0]
                n = struct.unpack("<Q", _read_exact(f, 8, "array length"))[0]
                if n > 1_000_000:
                    raise ValueError("corrupt GGUF array length")
                return [read_val(at) for _ in range(n)]
            if t == 11:
                return struct.unpack("<d", _read_exact(f, 8, "f64"))[0]
            raise ValueError(f"unsupported GGUF value type {t} (treat as corrupt/unknown)")

        keys = {}
        for _ in range(min(n_kv, max_keys)):
            _key = read_str()
            _type = struct.unpack("<I", _read_exact(f, 4, "type"))[0]
            keys[_key] = read_val(_type)
        keys["_tensor_count"] = n_tensors
        return keys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelInfo:
    path: str
    filename: str
    size_bytes: int
    sha256: str
    architecture: str = "unknown"
    model_name: str = "unknown"
    size_label: str = "unknown"
    quantization: str = "unknown"
    context_length: int = 0
    embedding_length: int = 0
    block_count: int = 0
    bos_token_id: int = -1
    eos_token_id: int = -1
    has_chat_template: bool = False
    status: str = "DISCOVERED"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def discover_models(roots: tuple = MODEL_ROOTS, compute_hash: bool = True) -> List[ModelInfo]:
    """Scans for *.gguf files. Missing dir -> empty list (UNAVAILABLE, not an error)."""
    found: List[ModelInfo] = []
    cached: Dict[str, Any] = {}
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, encoding="utf-8") as f:
                cached = {m["path"]: m for m in json.load(f).get("models", [])}
        except Exception:
            cached = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for fn in sorted(files):
                if not fn.lower().endswith(".gguf"):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(path)
                    mtime = os.path.getmtime(path)
                    prev = cached.get(path)
                    if prev and prev.get("size_bytes") == size and prev.get("mtime") == mtime \
                            and prev.get("sha256"):
                        found.append(ModelInfo(**{k: v for k, v in prev.items()
                                                   if k in ModelInfo.__dataclass_fields__}))
                        continue
                    kv = _read_gguf_kv(path)
                    arch = str(kv.get("general.architecture", "unknown"))
                    ft = kv.get("general.file_type", -1)
                    info = ModelInfo(
                        path=path, filename=fn, size_bytes=size,
                        sha256=sha256_file(path) if compute_hash else "not_computed",
                        architecture=arch,
                        model_name=str(kv.get("general.name", fn)),
                        size_label=str(kv.get("general.size_label", "unknown")),
                        quantization=FILE_TYPE_NAMES.get(int(ft), f"file_type_{ft}"),
                        context_length=int(kv.get(f"{arch}.context_length", 0)),
                        embedding_length=int(kv.get(f"{arch}.embedding_length", 0)),
                        block_count=int(kv.get(f"{arch}.block_count", 0)),
                        bos_token_id=int(kv.get("tokenizer.ggml.bos_token_id", -1)),
                        eos_token_id=int(kv.get("tokenizer.ggml.eos_token_id", -1)),
                        has_chat_template=bool(kv.get("tokenizer.chat_template", "")),
                        extra={"mtime": mtime,
                               "heads": kv.get(f"{arch}.attention.head_count"),
                               "heads_kv": kv.get(f"{arch}.attention.head_count_kv"),
                               "tensor_count": kv.get("_tensor_count")},
                    )
                    found.append(info)
                except Exception as e:  # noqa: BLE001 - corrupt files are reported, never loaded
                    found.append(ModelInfo(path=path, filename=fn,
                                           size_bytes=os.path.getsize(path),
                                           sha256="unreadable",
                                           status=f"CORRUPT: {type(e).__name__}: {e}"))
    try:
        os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump({"models": [m.to_dict() for m in found]}, f, indent=2)
    except Exception:
        pass
    return found
