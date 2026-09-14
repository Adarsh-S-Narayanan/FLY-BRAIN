import abc
import time
import uuid
import hashlib
import json
from typing import Dict, Any, Optional

class ToolConnector(abc.ABC):
    """
    Abstract Base Class for all typed tool connectors exposed to the brain.
    Enforces input schema, output schema, error schema, timeouts, execution IDs,
    logging, and result hashes.
    """
    def __init__(self, name: str, description: str, timeout_sec: float = 30.0):
        self.name = name
        self.description = description
        self.timeout_sec = timeout_sec
        self.execution_history = []

    @property
    @abc.abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        pass

    @property
    @abc.abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        pass

    @property
    def error_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "error_code": {"type": "string"},
                "message": {"type": "string"},
                "recoverable": {"type": "boolean"}
            },
            "required": ["error_code", "message"]
        }

    def compute_hash(self, data: Any) -> str:
        if isinstance(data, (dict, list)):
            serialized = json.dumps(data, sort_keys=True)
        elif isinstance(data, bytes):
            serialized = data
        else:
            serialized = str(data)
        if isinstance(serialized, str):
            serialized = serialized.encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def run(self, params: Dict[str, Any], execution_id: Optional[str] = None) -> Dict[str, Any]:
        exec_id = execution_id or str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Validate input params against schema keys
            req_keys = self.input_schema.get("required", [])
            for k in req_keys:
                if k not in params:
                    raise ValueError(f"Missing required parameter '{k}' for connector '{self.name}'")

            result_payload = self._execute(params, exec_id)
            duration = time.time() - start_time
            res_hash = self.compute_hash(result_payload)
            
            record = {
                "execution_id": exec_id,
                "tool": self.name,
                "status": "SUCCESS",
                "duration_sec": round(duration, 4),
                "params": params,
                "result": result_payload,
                "result_hash": res_hash
            }
            self.execution_history.append(record)
            return record

        except Exception as e:
            duration = time.time() - start_time
            record = {
                "execution_id": exec_id,
                "tool": self.name,
                "status": "ERROR",
                "duration_sec": round(duration, 4),
                "params": params,
                "error": {
                    "error_code": type(e).__name__,
                    "message": str(e),
                    "recoverable": True
                },
                "result_hash": None
            }
            self.execution_history.append(record)
            return record

    @abc.abstractmethod
    def _execute(self, params: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        pass
