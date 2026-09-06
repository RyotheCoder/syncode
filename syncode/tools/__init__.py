"""Bộ công cụ (tools) cho Agent — builtin tools + MCP servers."""

from syncode.tools.mcp import MCPManager
from syncode.tools.registry import BUILTIN_TOOLS, ChangeTracker, Tool, ToolRegistry

__all__ = ["BUILTIN_TOOLS", "ChangeTracker", "MCPManager", "Tool", "ToolRegistry", "build_tool_stack"]


def build_tool_stack(config):
    """Dung bo tool builtin + MCP tu config. Tra (registry, mcp).
    Loi MCP khong lam chet app (tinh nang tang cuong)."""
    registry = ToolRegistry()
    mcp = MCPManager(config)
    registry.mcp = mcp
    try:
        ok, _errs = mcp.connect_all()
        if ok:
            mcp.register_into(registry)
    except Exception:  # noqa: BLE001
        pass
    return registry, mcp