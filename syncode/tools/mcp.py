"""MCP (Model Context Protocol) client manager.

Kết nối tới các MCP server cấu hình trong config key `mcp_servers`:
{
  "mcp_servers": {
    "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}
  }
}

Gói `mcp` là PHỤ THUỘC TÙY CHỌN — nếu chưa cài (`pip install mcp`) thì
MCPManager vẫn hoạt động an toàn và chỉ báo cáo "chưa cài".
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict, List, Tuple

from syncode.config import Config
from syncode.tools.registry import Tool

# Timeout cho mỗi lượt gọi MCP (giây)
MCP_TIMEOUT = 30.0


def _clean_mcp_args(arguments: Dict[str, Any] | None) -> Dict[str, Any]:
    """Loai cac arg noi bo registry tu chen (_changes/_ask/_emit) truoc khi
    gui sang MCP server — server chi nhan tham so that cua tool."""
    return {k: v for k, v in (arguments or {}).items() if not k.startswith("_")}


class MCPError(RuntimeError):
    """Lỗi MCP (chưa cài gói mcp, server chết, tool sai...)."""


class _Session:
    """Session stdio tới 1 MCP server (async, chạy trong event loop riêng)."""

    def __init__(self, name: str, command: str, args: List[str], env: Dict[str, str]) -> None:
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self._session = None
        self._stack = None
        self._lock = threading.Lock()  # workers chay song song dung chung 1 session

    async def _start(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise MCPError("Chưa cài gói 'mcp' — chạy: pip install mcp") from exc
        params = StdioServerParameters(command=self.command, args=self.args, env=self.env or None)
        self._stack = __import__("contextlib").AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def _stop(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._stack = None
            self._session = None

    async def _ensure(self) -> None:
        if self._session is None:
            await self._start()

    def _run(self, coro):
        """Chạy coroutine trong event loop riêng (an toàn từ thread worker)."""
        return asyncio.run(coro)

    def list_tools(self) -> List[Tool]:
        async def _inner() -> List[Tool]:
            await self._ensure()
            assert self._session is not None
            result = await asyncio.wait_for(self._session.list_tools(), MCP_TIMEOUT)
            tools: List[Tool] = []
            for t in result.tools:
                schema = t.inputSchema if isinstance(t.inputSchema, dict) else {"type": "object"}
                tools.append(
                    Tool(
                        name=t.name,
                        description=t.description or f"MCP tool từ server '{self.name}'",
                        parameters=schema,
                        handler=lambda args, _s=self, _n=t.name: self.call(_n, args),
                        source=f"mcp:{self.name}",
                    )
                )
            return tools

        with self._lock:
            return self._run(_inner())

    def call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        async def _inner() -> str:
            await self._ensure()
            assert self._session is not None
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, _clean_mcp_args(arguments)), MCP_TIMEOUT
            )
            parts: List[str] = []
            for item in result.content or []:
                text = getattr(item, "text", None)
                if text is not None:
                    parts.append(text)
            return "\n".join(parts) or "(MCP trả về rỗng)"

        return self._run(_inner())

    def close(self) -> None:
        if self._stack is not None:
            try:
                self._run(self._stop())
            except Exception:  # noqa: BLE001
                pass


class MCPManager:
    """Quản lý các MCP server từ config; nối tool của chúng vào ToolRegistry."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.sessions: Dict[str, _Session] = {}
        self.errors: Dict[str, str] = {}

    # ------------------------------------------------------------- config
    def _server_specs(self) -> Dict[str, Dict[str, Any]]:
        raw = self.config.get("mcp_servers", {})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                raw = {}
        return raw if isinstance(raw, dict) else {}

    # --------------------------------------------------------------- list
    def connect_all(self) -> Tuple[List[str], List[str]]:
        """Kết nối mọi server trong config. Trả về (ok, lỗi)."""
        ok: List[str] = []
        errs: List[str] = []
        for name, spec in self._server_specs().items():
            if name in self.sessions:
                ok.append(name)
                continue
            try:
                session = _Session(
                    name,
                    str(spec.get("command", "")),
                    [str(a) for a in spec.get("args", [])],
                    dict(spec.get("env", {})),
                )
                session.list_tools()  # handshake + probe
                self.sessions[name] = session
                ok.append(name)
            except Exception as exc:  # noqa: BLE001
                self.errors[name] = str(exc)
                errs.append(f"{name}: {exc}")
        return ok, errs

    def register_into(self, registry) -> List[str]:
        """Đăng ký tool của mọi MCP server đã kết nối vào ToolRegistry."""
        names: List[str] = []
        for session in self.sessions.values():
            try:
                for tool in session.list_tools():
                    registry.register(tool)
                    names.append(tool.name)
            except Exception as exc:  # noqa: BLE001
                self.errors[session.name] = str(exc)
        return names

    def status_lines(self) -> List[str]:
        """Trạng thái cho /.mcp."""
        if not self._server_specs():
            return [
                "Chưa cấu hình MCP server nào.",
                'Thêm bằng: /.config set mcp_servers {"<tên>": {"command": "...", "args": [...]}}',
                "(yêu cầu pip install mcp)",
            ]
        rows = []
        for name in self._server_specs():
            if name in self.sessions:
                try:
                    count = len(self.sessions[name].list_tools())
                    rows.append(f"✓ {name} — {count} tool(s)")
                except Exception as exc:  # noqa: BLE001
                    rows.append(f"✗ {name} — {exc}")
            else:
                rows.append(f"✗ {name} — {self.errors.get(name, 'chưa kết nối')}")
        return rows

    def close(self) -> None:
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()