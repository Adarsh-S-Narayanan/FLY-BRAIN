from typing import Dict, List, Any, Optional
from src.tools.base import ToolConnector

class ToolRegistry:
    def __init__(self):
        self._connectors: Dict[str, ToolConnector] = {}

    def register(self, connector: ToolConnector):
        self._connectors[connector.name] = connector

    def get(self, name: str) -> Optional[ToolConnector]:
        return self._connectors.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": c.name,
                "description": c.description,
                "input_schema": c.input_schema,
                "output_schema": c.output_schema,
                "execution_count": len(c.execution_history)
            }
            for c in self._connectors.values()
        ]

    def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        connector = self.get(tool_name)
        if not connector:
            raise KeyError(f"Tool connector '{tool_name}' not registered!")
        return connector.run(params)
