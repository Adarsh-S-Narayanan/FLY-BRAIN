"""LLM/VLM + scripted teacher protocols. LLMs stay OUTSIDE the brain (REAL interface, HONEST status)."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class TeacherProtocol(ABC):
    @abstractmethod
    def observe(self, organism) -> Dict[str, Any]:
        ...

    @abstractmethod
    def propose_lesson(self, organism) -> Dict[str, Any]:
        ...

    @abstractmethod
    def demonstrate(self, domain: str):
        ...


class ScriptTeacher(TeacherProtocol):
    """Deterministic curriculum teacher (IMPLEMENTED). Lessons never write neural state
    directly; they go through the cultural teaching channel (culture.transmission.teach)."""
    def __init__(self, lessons: List[str]):
        self.lessons = list(lessons)
        self.id = "script-teacher"

    def observe(self, organism) -> Dict[str, Any]:
        return {"organism_id": organism.id, "skills": dict(organism.skills),
                "stage": organism.stage.value}

    def propose_lesson(self, organism) -> Dict[str, Any]:
        weak = min(organism.skills, key=lambda k: organism.skills[k])
        lesson = weak if weak in self.lessons else self.lessons[0]
        return {"domain": lesson, "method": "demonstration"}

    def demonstrate(self, domain: str):
        return [0.8]


class UnavailableLLMTeacher(TeacherProtocol):
    """Placeholder for an external LLM/VLM teacher (NOT IMPLEMENTED -- no weights configured).

    Raises instead of faking. Wire a real model backend here; it may only act via
    observe/propose/demonstrate -> cultural channel, never direct neural writes.
    """
    STATUS = "MODEL_UNAVAILABLE"

    def _fail(self):
        raise RuntimeError("LLM teacher unavailable: no model backend configured (honest status: MODEL_UNAVAILABLE)")

    def observe(self, organism):
        self._fail()

    def propose_lesson(self, organism):
        self._fail()

    def demonstrate(self, domain: str):
        self._fail()
