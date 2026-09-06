"""UI layer (rich): banner, panel, spinner, streaming markdown - phong cach Cline."""

from __future__ import annotations

from collections.abc import Iterable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

BANNER = r"""
 _  ___     _                _
| |/ / |   (_) __ _ _ __ ___| |__
| ' /| |   | |/ _` | '__/ __| '_ \
| . \| |___| | (_| | |  \__ \ | | |
|_|\_\_____/ |\__,_|_|  |___/_ |_|
          |_/                  |__|
"""


class UI:
    """Cac helper hien thi tren console."""

    def __init__(self) -> None:
        self.console = Console(highlight=False, soft_wrap=True)
        self.quick_mode: bool = False  # False = swarm (default), True = single-pass quick

    # ------------------------------------------------------------- basics
    def banner(self, version: str, model: str) -> None:
        self.console.print(Text(BANNER, style="bold cyan"))
        self.console.print(
            f"  [bold]Syncode[/] v{version}  •  model: [magenta]{model}[/]"
        )
        self.console.print(
            "  [bold cyan]/.help[/] for commands  •  [bold cyan]/.exit[/] to quit\n",
            style="dim",
        )

    def info(self, text: str) -> None:
        self.console.print(f"[dim]•[/] {text}")

    def warn(self, text: str) -> None:
        self.console.print(f"[yellow]⚠ {text}[/]")

    def error(self, text: str) -> None:
        self.console.print(f"[bold red]✗ {text}[/]")

    def success(self, text: str) -> None:
        self.console.print(f"[bold green]✓ {text}[/]")

    def toggle_swarm_mode(self) -> None:
        """Toggle swarm/quick (called from /.swarm in plain REPL)."""
        self.quick_mode = not self.quick_mode
        if self.quick_mode:
            self.info("mode -> quick (single pass, /.swarm for swarm)")
        else:
            self.info("mode -> swarm (/.swarm for quick)")

    def user_prompt(self) -> str:
        self.console.print()
        return self.console.input("[bold green]❯ [/]")

    def agent_badge(self, name: str, icon: str, style: str, note: str = "") -> None:
        label = Text(f" {icon} {name} ", style=style)
        line = Text.assemble(label, ("  " + note, "dim")) if note else label
        self.console.print(line)

    def agent_panel(self, name: str, icon: str, style: str, content: str, subtitle: str = "") -> None:
        self.agent_badge(name, icon, style)
        body = Markdown(content) if content.strip() else Text("(empty)", style="dim")
        self.console.print(
            Panel(body, border_style=style, subtitle=subtitle or None, subtitle_align="right")
        )

    # ---------------------------------------------------------- streaming
    def stream_markdown(self, chunks: Iterable[str], title: str, style: str) -> str:
        """Render markdown live trong khi nhan token tu Judge."""
        buffer = ""
        with Live(
            Panel(Markdown(""), title=title, border_style=style),
            console=self.console,
            refresh_per_second=12,
            vertical_overflow="visible",
        ) as live:
            for piece in chunks:
                buffer += piece
                live.update(Panel(Markdown(buffer), title=title, border_style=style))
        self.console.print()
        return buffer

    # ------------------------------------------------------------ spinner
    class _Spin:
        def __init__(self, console: Console, text: str) -> None:
            self._ctx = console.status(f"[bold]{text}[/]", spinner="dots")

        def __enter__(self):
            self._ctx.start()
            return self

        def update(self, text: str) -> None:
            self._ctx.update(f"[bold]{text}[/]")

        def __exit__(self, *exc) -> None:
            self._ctx.stop()

    def spinner(self, text: str) -> UI._Spin:
        return UI._Spin(self.console, text)

    # -------------------------------------------------------------- misc
    def key_value_table(self, title: str, rows: dict, style: str = "cyan") -> None:
        table = Table(title=title, border_style=style, show_lines=False)
        table.add_column("Key", style="bold")
        table.add_column("Value")
        for key, value in rows.items():
            table.add_row(str(key), str(value))
        self.console.print(table)

    def list_rows(self, title: str, items, style: str = "cyan") -> None:
        table = Table(title=title, border_style=style)
        table.add_column("#", style="dim")
        table.add_column("Item")
        for i, item in enumerate(items, start=1):
            table.add_row(str(i), str(item))
        self.console.print(table)

    def rule(self, text: str = "", style: str = "dim") -> None:
        self.console.rule(Text(text) if text else None, style=style)

    def clear(self) -> None:
        self.console.clear()
