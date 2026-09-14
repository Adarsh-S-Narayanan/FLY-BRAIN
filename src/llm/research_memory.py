"""Research memory (STAGE K): persistent, append-only, hash-chained records
of the scientist's experiments, hypotheses, outcomes and comparisons.
The scientist learns from the history of its own research."""
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

RESEARCH_MEMORY_VERSION = "research_memory_v1"


def _record_hash(rec: Dict[str, Any], prev_hash: str) -> str:
    payload = json.dumps(rec, sort_keys=True) + "|" + prev_hash
    return hashlib.sha256(payload.encode()).hexdigest()


class ResearchMemory:
    """Append-only JSONL store with a hash chain (tamper-evident)."""

    def __init__(self, path: str = "diagnostics/research_memory.jsonl"):
        self.path = path
        self.records: List[Dict[str, Any]] = []
        self._prev_hash = "0" * 64
        if os.path.exists(path):
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.records.append(rec)
                self._prev_hash = rec.get("record_hash", self._prev_hash)

    def append(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        rec = {
            "seq": len(self.records),
            "ts": time.time(),
            "kind": kind,
            "payload": payload,
            "prev_hash": self._prev_hash,
        }
        rec["record_hash"] = _record_hash({k: v for k, v in rec.items()
                                           if k != "record_hash"}, self._prev_hash)
        self.records.append(rec)
        self._prev_hash = rec["record_hash"]
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for rec in self.records:
            if rec.get("prev_hash") != prev:
                return False
            expected = _record_hash({k: v for k, v in rec.items()
                                     if k != "record_hash"}, prev)
            if rec.get("record_hash") != expected:
                return False
            prev = rec["record_hash"]
        return True

    # ---- scientist-facing queries ----
    def experiments(self) -> List[Dict[str, Any]]:
        return [r for r in self.records if r["kind"] == "experiment"]

    def hypotheses(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        out = [r for r in self.records if r["kind"] == "hypothesis"]
        if status is not None:
            out = [r for r in out if r["payload"].get("status") == status]
        return out

    def failed_experiments(self) -> List[Dict[str, Any]]:
        return [r for r in self.records
                if r["kind"] == "experiment" and r["payload"].get("status") == "FAILED"]

    def successful_protocols(self) -> List[Dict[str, Any]]:
        """Protocol shapes that produced measurable results before."""
        return [r["payload"] for r in self.records
                if r["kind"] == "experiment" and r["payload"].get("measured", True)
                and r["payload"].get("status") == "EXECUTED"]

    def summary(self) -> Dict[str, Any]:
        kinds: Dict[str, int] = {}
        for r in self.records:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        return {"records": len(self.records), "kinds": kinds,
                "chain_valid": self.verify_chain(),
                "version": RESEARCH_MEMORY_VERSION}
