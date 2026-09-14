import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class WorkingMemoryItem:
    key: str
    content: Any
    salience: float = 1.0
    created_at: float = field(default_factory=time.time)
    access_count: int = 0

class WorkingMemory:
    """
    Capacity-limited working memory (Miller's Law: 7 +/- 2 slots).
    Decays over time unless refreshed by attention or access.
    """
    def __init__(self, capacity: int = 7, decay_rate: float = 0.05):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.items: Dict[str, WorkingMemoryItem] = {}

    def put(self, key: str, content: Any, salience: float = 1.0):
        if key in self.items:
            item = self.items[key]
            item.content = content
            item.salience = min(2.0, item.salience + salience)
            item.access_count += 1
            return

        # If full, evict lowest salience item
        if len(self.items) >= self.capacity:
            lowest_key = min(self.items.keys(), key=lambda k: self.items[k].salience)
            del self.items[lowest_key]

        self.items[key] = WorkingMemoryItem(key=key, content=content, salience=salience)

    def get(self, key: str) -> Optional[Any]:
        if key in self.items:
            item = self.items[key]
            item.salience = min(2.0, item.salience + 0.2)
            item.access_count += 1
            return item.content
        return None

    def decay_step(self):
        to_delete = []
        for key, item in self.items.items():
            item.salience -= self.decay_rate
            if item.salience <= 0.0:
                to_delete.append(key)
        for k in to_delete:
            del self.items[k]

    def get_all_active(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": item.key,
                "content": str(item.content),
                "salience": round(item.salience, 3),
                "access_count": item.access_count
            }
            for item in sorted(self.items.values(), key=lambda x: x.salience, reverse=True)
        ]
