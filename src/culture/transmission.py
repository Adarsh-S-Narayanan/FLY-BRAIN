"""Teachable cultural knowledge with provenance (REAL, IMPLEMENTED)."""
import hashlib
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List
from src.common.determinism import derive_subseed


@dataclass
class Knowledge:
    id: str
    domain: str
    vector: List[float]
    provenance: Dict[str, Any]  # origin_organism, tick, generation, teacher_chain
    confidence: float = 0.5

    def knowledge_hash(self) -> str:
        return hashlib.sha256(json.dumps(
            {"id": self.id, "domain": self.domain, "vector": self.vector,
             "provenance": self.provenance}, sort_keys=True).encode()).hexdigest()


@dataclass
class TeachingSession:
    teacher: str
    student: str
    domain: str
    start_tick: int
    end_tick: int
    demonstration: List[float]
    student_response: List[float]
    learning_gain: float
    retention: float


def teach(teacher, student, domain: str, tick: int, teacher_seed: int = 48) -> TeachingSession:
    """Real demonstration->imitation: teacher skill vector perturbs student skills/knowledge."""
    rng = np.random.RandomState(derive_subseed(int(teacher_seed), f"teach:{teacher.id}:{student.id}:{tick}"))
    demo_val = float(teacher.skills.get(domain, 0.2))
    demonstration = [round(demo_val + float(rng.normal(0, 0.02)), 4)]
    before = float(student.skills.get(domain, 0.1))
    t_ability = float(teacher.genome.params.get("teaching_ability", 0.5))
    s_plasticity = float(student.genome.params.get("plasticity_rate", 0.05)) * 10.0
    gain = round(max(0.0, (demo_val - before) * 0.3 * t_ability * (0.5 + s_plasticity)), 4)
    student.skills[domain] = round(min(1.0, before + gain), 4)
    retention = round(float(student.genome.params.get("memory_retention", 0.8)), 4)
    chain = list(student.cultural_knowledge.get(domain, {}).get("teacher_chain", [])) + [teacher.id]
    student.cultural_knowledge[domain] = {
        "vector": demonstration,
        "provenance": {"origin_organism": teacher.id, "tick": tick,
                       "generation": student.generation, "teacher_chain": chain},
        "confidence": round(min(1.0, 0.3 + gain * 5.0), 4),
    }
    sess = TeachingSession(teacher=teacher.id, student=student.id, domain=domain,
                           start_tick=tick, end_tick=tick + 1, demonstration=demonstration,
                           student_response=[student.skills[domain]],
                           learning_gain=gain, retention=retention)
    teacher.events.log("TEACHING_STARTED", tick, teacher.id, teacher.generation,
                       {"student": student.id, "domain": domain})
    student.events.log("LEARNING_EVENT", tick + 1, student.id, student.generation,
                       {"teacher": teacher.id, "domain": domain, "gain": gain})
    teacher.events.log("TEACHING_ENDED", tick + 1, teacher.id, teacher.generation,
                       {"student": student.id, "gain": gain})
    return sess


def teacher_effectiveness(sessions: List[TeachingSession], teacher_id: str) -> float:
    gains = [s.learning_gain for s in sessions if s.teacher == teacher_id]
    return round(float(sum(gains) / len(gains)), 4) if gains else 0.0
