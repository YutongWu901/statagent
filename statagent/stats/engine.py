"""Statistical tool registry and strict execution engine.

Every statistical procedure the agent can run is a *registered tool* with a
Pydantic parameter schema. The LLM never writes code: it selects tools and
fills parameters; the engine validates them against the schema and executes
the corresponding hard-coded, tested Python function. Numbers the LLM
invents are never used — only the engine's outputs count.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Type

import plotly.graph_objects as go
import plotly.io as pio
from pydantic import BaseModel, ConfigDict, ValidationError

from statagent.data.store import DatasetStore
from statagent.models.plan import AnalysisStep
from statagent.models.results import StepResult, ToolResult
from statagent.utils import to_jsonable


class Params(BaseModel):
    """Base class for tool parameter schemas: unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        param_model: Type[Params],
        fn: Callable[..., dict[str, Any]],
        output_description: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self.param_model = param_model
        self.fn = fn
        self.output_description = output_description

    def llm_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "output": self.output_description,
            "parameters": self.param_model.model_json_schema(),
        }


class ToolRegistry:
    _registry: dict[str, Tool] = {}

    @classmethod
    def register(
        cls, name: str, description: str, param_model: Type[Params], output_description: str = ""
    ) -> Callable[[Callable], Callable]:
        def decorator(fn: Callable) -> Callable:
            if name in cls._registry:
                raise ValueError(f"Tool '{name}' is already registered.")
            cls._registry[name] = Tool(name, description, param_model, fn, output_description)
            return fn

        return decorator

    @classmethod
    def get(cls, name: str) -> Tool | None:
        return cls._registry.get(name)

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def llm_tool_list(cls) -> list[dict[str, Any]]:
        return [t.llm_schema() for t in cls._registry.values()]


def execute_tool(store: DatasetStore, tool_name: str, parameters: dict[str, Any]) -> ToolResult:
    """Validate parameters and run a registered tool, catching all errors."""
    tool = ToolRegistry.get(tool_name)
    if tool is None:
        return ToolResult(
            tool=tool_name,
            parameters=parameters,
            ok=False,
            error=(
                f"Unknown tool '{tool_name}'. Registered tools: "
                f"{', '.join(ToolRegistry.names())}."
            ),
        )
    try:
        params = tool.param_model(**parameters)
    except ValidationError as exc:
        return ToolResult(
            tool=tool_name,
            parameters=parameters,
            ok=False,
            error=f"Invalid parameters for '{tool_name}': {exc}",
        )
    try:
        raw = tool.fn(store, params)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a clean ToolResult
        return ToolResult(
            tool=tool_name,
            parameters=parameters,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    figures = []
    for fig in raw.get("figures", []) or []:
        if isinstance(fig, go.Figure):
            figures.append(json.loads(pio.to_json(fig)))
    return ToolResult(
        tool=tool_name,
        parameters=to_jsonable(params.model_dump(mode="json")),
        ok=True,
        n=raw.get("n"),
        summary=to_jsonable(raw.get("summary", {})),
        details=to_jsonable(raw.get("details", {})),
        figures=figures,
        notes=[str(x) for x in raw.get("notes", [])],
        code=str(raw.get("code", "")),
    )


def execute_step(store: DatasetStore, step: AnalysisStep) -> StepResult:
    result = execute_tool(store, step.tool, step.parameters)
    return StepResult(step_id=step.step_id, tool=step.tool, result=result)
