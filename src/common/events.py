"""Deterministic biological event stream (REAL, IMPLEMENTED)."""
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

VALID_EVENTS = {
    "NEURON_BORN", "NEURON_DIFFERENTIATED", "NEURON_MOVED",
    "AXON_GROWN", "DENDRITE_GROWN", "SYNAPSE_CREATED",
    "SYNAPSE_STRENGTHENED", "SYNAPSE_WEAKENED", "SYNAPSE_PRUNED",
    "NEURON_DIED", "MEMORY_CREATED", "MEMORY_RECALLED",
    "MEMORY_CONSOLIDATED", "SLEEP_STARTED", "DREAM_STARTED",
    "DREAM_ENDED", "TEACHING_STARTED", "TEACHING_ENDED",
    "LEARNING_EVENT", "REPRODUCTION", "MUTATION", "CROSSOVER",
    "ORGANISM_BORN", "ORGANISM_DIED", "GENERATION_STARTED",
    "GENERATION_ENDED", "WORLD_STEP", "SNAPSHOT_SAVED",
}


class EventLog:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self._tick_counter = 0

    def log(self, event_type: str, tick: int, organism_id: str = "",
            generation: int = 0, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if event_type not in VALID_EVENTS:
            raise ValueError(f"Unknown event type: {event_type}")
        ev = {
            "seq": len(self.events),
            "type": event_type,
            "tick": int(tick),
            "organism_id": organism_id,
            "generation": int(generation),
            "payload": payload or {},
        }
        self.events.append(ev)
        return ev

    def __len__(self):
        return len(self.events)

    def filter(self, event_type: str = "", organism_id: str = "") -> List[Dict[str, Any]]:
        out = self.events
        if event_type:
            out = [e for e in out if e["type"] == event_type]
        if organism_id:
            out = [e for e in out if e["organism_id"] == organism_id]
        return out

    def compute_hash(self) -> str:
        h = hashlib.sha256()
        for e in self.events:
            h.update(json.dumps(e, sort_keys=True).encode())
        return h.hexdigest()

    def to_jsonl(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, sort_keys=True) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "EventLog":
        log = cls()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    log.events.append(json.loads(line))
        return log
