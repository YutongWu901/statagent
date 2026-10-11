"""Statistical computation engine.

Importing this package registers every available tool in the
:class:`ToolRegistry`. The engine executes tools with strict parameter
validation; it never runs LLM-generated code.
"""
from statagent.stats import bootstrap, correlation, descriptive, hypothesis, regression  # noqa: F401
from statagent.stats.engine import (
    Params,
    Tool,
    ToolRegistry,
    execute_step,
    execute_tool,
)

__all__ = ["Params", "Tool", "ToolRegistry", "execute_step", "execute_tool"]
