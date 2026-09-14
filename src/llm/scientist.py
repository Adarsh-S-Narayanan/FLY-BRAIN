"""Controlled LLM scientist loop (REAL, IMPLEMENTED).

The model may reason/propose/select tools. It NEVER mutates state directly:
only validated tool calls execute, every call and result is logged, and every
model assertion about simulation state is marked UNVERIFIED until a tool
measurement confirms it.
"""
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from src.llm.runtime import LocalLLM, GenerationConfig, RESEARCH_DETERMINISTIC

MAX_PROMPT_CHARS = 6000


@dataclass
class ToolSpec:
    name: str
    description: str
    params_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class LoopRecord:
    agent_id: str
    model_id: str
    iteration: int
    prompt_hash: str
    context_hash: str
    input_state_hash: str
    hypothesis: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    grounded_claims: List[Dict[str, Any]] = field(default_factory=list)
    experiment_id: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScientistLoop:
    def __init__(self, llm: LocalLLM, agent_id: str = "scientist-01"):
        self.llm = llm
        self.agent_id = agent_id
        self.tools: Dict[str, ToolSpec] = {}
        self.records: List[LoopRecord] = []

    def register(self, spec: ToolSpec):
        self.tools[spec.name] = spec

    def tool_names(self) -> List[str]:
        return sorted(self.tools.keys())

    def execute_tool_request(self, request: Any) -> Dict[str, Any]:
        """Validates and executes ONE tool request. Rejects everything else safely."""
        if not isinstance(request, dict):
            return {"status": "REJECTED", "reason": "tool request must be a JSON object"}
        name = request.get("tool")
        params = request.get("params", {})
        if not isinstance(name, str) or name not in self.tools:
            return {"status": "REJECTED", "reason": f"unknown tool: {name!r}",
                    "available": self.tool_names()}
        if not isinstance(params, dict):
            return {"status": "REJECTED", "reason": "params must be an object"}
        lowered = json.dumps(request).lower()
        for banned in ("os.system", "subprocess", "eval(", "exec(", "rm -rf", "powershell",
                       "drop table", "delete from", "__import__"):
            if banned in lowered:
                return {"status": "REJECTED", "reason": f"forbidden pattern in tool request: {banned}"}
        try:
            result = self.tools[name].handler(params)
            return {"status": "SUCCESS", "tool": name, "result": result}
        except Exception as e:  # noqa: BLE001
            return {"status": "TOOL_ERROR", "tool": name, "error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def ground_claims(text: str, measurements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Marks model assertions VERIFIED only when a measurement confirms them.

        A claim counts as grounded only if the literal 'key=value' rendering of a
        measurement appears in the text. Everything else stays UNVERIFIED.
        """
        out = []
        for k, v in measurements.items():
            token = f"{k}={v}"
            out.append({"measurement": token,
                        "status": "VERIFIED" if token in text else "UNVERIFIED"})
        return out

    def run_iteration(self, context: str, state_hash: str,
                      config: Optional[GenerationConfig] = None) -> LoopRecord:
        cfg = config or GenerationConfig(max_tokens=128)
        context = context[:MAX_PROMPT_CHARS]
        rec = LoopRecord(
            agent_id=self.agent_id,
            model_id=(self.llm.model.path if self.llm.model else "none"),
            iteration=len(self.records),
            prompt_hash=hashlib.sha256(context.encode()).hexdigest()[:16],
            context_hash=hashlib.sha256(context.encode()).hexdigest()[:16],
            input_state_hash=state_hash,
            hypothesis="",
            timestamp=time.time(),
        )
        if not self.llm.is_ready():
            rec.hypothesis = f"[{self.llm.status}] scientist iteration skipped: {self.llm.last_error}"
            self.records.append(rec)
            return rec
        prompt = (f"You are a neuroscience research assistant. Available tools: {self.tool_names()}. "
                  f"First state a hypothesis in one sentence. Then, on a new line, either a JSON tool "
                  f"request {{\"tool\": name, \"params\": {{...}}}} or the word DONE.\nCONTEXT:\n{context}")
        gen = self.llm.generate(prompt, cfg)
        if gen["status"] != "SUCCESS":
            rec.hypothesis = f"[{gen['status']}] {gen.get('error', '')}"
            self.records.append(rec)
            return rec
        text = gen["text"] or ""
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        rec.hypothesis = lines[0] if lines else ""
        for ln in lines[1:4]:
            if ln.startswith("{"):
                try:
                    req = json.loads(ln[ln.index("{"):ln.rindex("}") + 1])
                except Exception:
                    rec.tool_calls.append({"status": "REJECTED", "reason": "malformed JSON"})
                    continue
                res = self.execute_tool_request(req)
                rec.tool_calls.append(req if isinstance(req, dict) else {"raw": ln[:200]})
                rec.tool_results.append(res)
                if res["status"] == "SUCCESS":
                    rec.grounded_claims = self.ground_claims(
                        rec.hypothesis, _flatten(res["result"]))
                break
        self.records.append(rec)
        return rec


def _flatten(d: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {} if out is None else out
    if isinstance(d, dict):
        for k, v in d.items():
            _flatten(v, f"{prefix}{k}.", out)
    elif isinstance(d, (list, tuple)):
        out[prefix.rstrip(".")] = len(d)
    elif isinstance(d, (int, float, str, bool)) or d is None:
        out[prefix.rstrip(".")] = d
    return out
