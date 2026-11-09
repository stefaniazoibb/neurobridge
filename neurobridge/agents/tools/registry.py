"""Tool registry. A tool = name + description + pydantic input schema + function(workspace, inputs) -> ToolOutput.

The manifest produced by `mcp_manifest()` follows the Model Context Protocol tool shape
({name, description, inputSchema}), so the same skills can later be exposed through an MCP server
or handed to an LLM as tool definitions without rewriting them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class ToolOutput(BaseModel):
    summary: str
    refs: list[str] = Field(default_factory=list, description="workspace artefact references produced")
    figures: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    degraded: bool = Field(False, description="True when the tool ran but with reduced scope (e.g. a dataset missing)")
    data: dict[str, Any] = Field(default_factory=dict, description="small JSON-serialisable payload for downstream agents")


class TransientToolError(RuntimeError):
    """Raise for failures worth retrying (network, rate limits)."""


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[Any, BaseModel], ToolOutput]
    category: str
    retryable: bool = False
    tags: list[str] = field(default_factory=list)

    def schema(self) -> dict:
        return self.input_model.model_json_schema()


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def mcp_manifest(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "inputSchema": t.schema(), "annotations": {"category": t.category, "retryable": t.retryable}} for t in self._tools.values()]


REGISTRY = Registry()


def register(name: str, description: str, input_model: type[BaseModel], category: str, retryable: bool = False, tags: Optional[list[str]] = None):
    def deco(fn):
        REGISTRY.add(Tool(name=name, description=description, input_model=input_model, fn=fn, category=category, retryable=retryable, tags=tags or []))
        return fn

    return deco
