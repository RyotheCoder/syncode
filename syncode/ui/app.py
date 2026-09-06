"""Syncode TUI — design language cua opencode (theme Catppuccin Mocha).

Editor box (border solid + meta line) · footer segments · dialog noi.
Cau hinh chinh sua qua modal panel, khong sua inline.
Palette lay tu: opencode packages/tui/src/theme/assets/catppuccin.json
"""

from __future__ import annotations

import re
import threading
from typing import List, Optional, Tuple

from rich.align import Align
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.color import Color
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Input, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from syncode import __version__
from syncode.commands.slash import (
    COMMANDS,
    ROLE_MODEL_KEYS,
    SlashCommandContext,
    execute,
    is_slash_command,
)
from syncode.config import DEFAULTS, Config
from syncode.history import (
    Session,
    add_exchange_to_session,
    compacted_history,
    load_sessions,
    open_todos,
    save_sessions,
)
from syncode.llm.client import LLMClient, LLMError
from syncode.swarm.orchestrator import SwarmOrchestrator
from syncode.tools import ChangeTracker, build_tool_stack

POPULAR_MODELS = [
    # === NVIDIA Nemotron / NIM ===
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-nano-3-30b-a3b",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-51b-instruct",
    "nvidia/llama3-chatqa-1.5-70b",
    "nvidia/mistral-nemo-minitron-8b-8k-instruct",
    "nvidia/nemotron-4-340b-instruct",
    "nvidia/cosmos-reason2-8b",
    "nvidia/neva-22b",
    "nvidia/vila",
    "nvidia/nvclip",
    "nvidia/nemotron-parse",
    "nvidia/ai-synthetic-video-detector",
    "nvidia/ising-calibration-1.5-31b",
    # === NVIDIA guard / embed / reward (khong phai chat thuan) ===
    "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "nvidia/llama-3.1-nemoguard-8b-topic-control",
    "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
    "nvidia/nemotron-3.5-content-safety",
    "nvidia/nemotron-4-340b-reward",
    "nvidia/embed-qa-4",
    "nvidia/llama-3.2-nv-embedqa-1b-v1",
    "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
    "nvidia/llama-nemotron-embed-vl-1b-v2",
    "nvidia/nemotron-3-embed-1b",
    "nvidia/nv-embedqa-mistral-7b-v2",
    "nvidia/riva-translate-4b-instruct",
    "nvidia/riva-translate-4b-instruct-v1.1",
    "nvidia/riva-translate-4b-instruct-v2",
    # === OpenAI ===
    "openai/gpt-oss-20b",
    # === Meta ===
    "meta/codellama-70b",
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "meta/llama-guard-4-12b",
    "meta/llama2-70b",
    "meta/muse-glimmer-30b",
    # === Mistral ===
    "nv-mistralai/mistral-nemo-12b-instruct",
    "mistralai/codestral-22b-instruct-v0.1",
    "mistralai/mistral-7b-instruct-v0.3",
    "mistralai/mistral-large",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mistral-nemotron",
    "mistralai/mixtral-8x22b-v0.1",
    # === Google ===
    "google/codegemma-1.1-7b",
    "google/codegemma-7b",
    "google/deplot",
    "google/diffusiongemma-26b-a4b-it",
    "google/gemma-2b",
    "google/gemma-3-12b-it",
    "google/gemma-3-4b-it",
    "google/gemma-4-31b-it",
    "google/recurrentgemma-2b",
    # === DeepSeek ===
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-ai/deepseek-v4-flash-0731",
    "deepseek-ai/deepseek-v4-pro-0813",
    # === Moonshot AI ===
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k3",
    # === Microsoft ===
    "microsoft/kosmos-2",
    "microsoft/phi-3-vision-128k-instruct",
    "microsoft/phi-3.5-moe-instruct",
    # === IBM Granite ===
    "ibm/granite-3.0-3b-a800m-instruct",
    "ibm/granite-3.0-8b-instruct",
    "ibm/granite-34b-code-instruct",
    "ibm/granite-8b-code-instruct",
    # === Writer ===
    "writer/palmyra-creative-122b",
    "writer/palmyra-fin-70b-32k",
    "writer/palmyra-med-70b",
    "writer/palmyra-med-70b-32k",
    # === Cac vendor khac ===
    "01-ai/yi-large",
    "adept/fuyu-8b",
    "ai21labs/jamba-1.5-large-instruct",
    "aisingapore/sea-lion-7b-instruct",
    "bigcode/starcoder2-15b",
    "databricks/dbrx-instruct",
    "minimaxai/minimax-m3",
    "poolside/laguna-xs-2.1",
    "snowflake/arctic-embed-l",
    "zyphra/zamba2-7b-instruct",
]

# ============================================ Palette: Vercel black + Vesper gold
C = {
    "base": "#000000",      # den tuyen (Vercel background200)
    "mantle": "#0A0A0A",    # nen phu
    "crust": "#111111",     # panel/editor fill
    "text": "#EDEDED",      # chu chinh (gray1000)
    "text2": "#C9C9C9",
    "muted": "#878787",     # gray600
    "primary": "#FFC799",   # gold — cinematic highlight (Vesper)
    "secondary": "#99FFE4", # mint — phu kinematik (Vesper)
    "accent": "#FFC799",
    "success": "#46A758",   # green700
    "warning": "#FFB224",   # amber700
    "error": "#E5484D",     # red700
    "border": "#1F1F1F",    # gray200 — vien lanh
    "border_hi": "#454545", # gray500 — vien active
}

SYNCODE_THEME = Theme(
    name="syncode",
    primary=C["primary"],
    secondary=C["secondary"],
    accent=C["accent"],
    foreground=C["text"],
    background=C["base"],
    surface=C["mantle"],
    panel=C["crust"],
    boost=C["mantle"],
    success=C["success"],
    warning=C["warning"],
    error=C["error"],
    dark=True,
    variables={
        "text-muted": C["muted"],
        "text-disabled": "#6c7086",
    },
)


# ============================================ spinner + thinking + collapse
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
_THINK_TAG_RE = re.compile(r"</?think(ing)?>", re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Bo sot tag think con lai (khi stream khong dong duoc cap the)."""
    return _THINK_TAG_RE.sub("", text)


def split_thinking(buffer: str) -> tuple:
    """Tach buffer thanh (answer_before, thinking, think_done, answer_after).
    Model (vd Nemotron/DeepSeek-R1) viet suy nghi trong <think>...</think>."""
    import re as _re
    m = _re.search(r'(?i)(.*?)(<think(ing)?>)(.*?)(</think(ing)?>)(.*)', buffer, _re.DOTALL)
    if m:
        before = m.group(1)
        thinking = m.group(3)
        after = m.group(5)
        return before, thinking, True, after
    return buffer, "", False, ""


def _clean_stream_text(text: str) -> str:
    """Loc dong rac realtime khi streaming.

    Uy thac ve orchestrator.clean_answer (NGUON DUY NHAT) de stream va dap an
    cuoi loc giong het nhau — tranh truong hop stream sach ma cuoi ban, hoac nguoc lai.
    """
    if not text:
        return text
    from syncode.swarm.orchestrator import clean_answer as _orch_clean
    result = _orch_clean(text)
    # gom lai, bo dong trong thua
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def clean_final_answer(text: str) -> str:
    """Loc cau tra loi cuoi cung — dung chung bo loc voi clean_answer
    (k, thinking-process, echo Q&A, heading bao cao, loi mo dau thua...).
    Dam bao chi con lai noi dung thuc su danh cho nguoi dung.

    Thu tu quan trong: loc block TRUOC (khi tag con nguyen),
    roi moi strip tag le sot lai — lam nguoc se mat cau truc block va rot noi dung
    thinking vao dap an.
    """
    cleaned = _clean_stream_text(text or "")
    return strip_think_tags(cleaned).strip()


def _load_history_file() -> List[Session]:
    """Doc lich su session tu file."""
    return load_sessions()


def _save_history_file(sessions: List[Session]) -> None:
    """Luu lich su session vao file."""
    save_sessions(sessions)


def _render_footer_right() -> Text:
    """Right segment of the footer — version only."""
    line = Text()
    line.append(f"v{__version__}", style=C["muted"])
    return line


class CollapsePanel(Vertical):
    """Panel thu gon duoc (hoat dong / thinking): header click = mo - dong."""

    def __init__(self, title: str, *, accent: str, classes: str = "") -> None:
        super().__init__(classes=classes)
        self._title = title
        self._accent = accent
        self._collapsed = False
        self._steps = 0
        self._header = Static("", classes="panel-header")
        self._body = Vertical(classes="panel-body")
        self._stream: Static = Static(Text(), classes="panel-line")

    def compose(self):
        yield self._header
        with self._body:
            yield self._stream

    def on_mount(self) -> None:
        self._update_header()

    # ------------------------------------------------------------ header
    def _update_header(self) -> None:
        arrow = "▸" if self._collapsed else "▾"
        line = Text()
        line.append(f" {arrow} ", style=self._accent)
        line.append(self._title, style=f"bold {C['text']}")
        self._header.update(line)

    def set_title(self, title: str, accent: str | None = None) -> None:
        self._title = title
        if accent:
            self._accent = accent
        self._update_header()

    # ---------------------------------------------------------- collapse
    def action_toggle(self) -> None:
        self._collapsed = not self._collapsed
        self._body.display = not self._collapsed
        self._update_header()

    def collapse(self) -> None:
        self._collapsed = True
        self._body.display = False
        self._update_header()

    def expand(self) -> None:
        self._collapsed = False
        self._body.display = True
        self._update_header()

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    # -------------------------------------------------------------- click
    def on_click(self, event: events.Click) -> None:
        # chi toggle khi bam dung header, khong tinh click vao noi dung
        if event.widget is self._header or event.widget.has_class("panel-header"):
            event.stop()
            self.action_toggle()

    # -------------------------------------------------------------- body
    def add_line(self, renderable) -> None:
        st = Static(renderable, classes="panel-line")
        self._steps += 1
        if self._body.is_mounted:
            self._body.mount(st)
        else:  # body chua kip mount trong tick nay -> defer 1 vong refresh
            self.call_after_refresh(self._body.mount, st)
        body = self._body
        body.scroll_end(animate=False, force=True)

    def add_text(self, text: str) -> None:
        """Cap nhat mot khoi text thuan (dung cho thinking stream)."""
        self._body.mount(Static(Text(text, style=C["muted"]), classes="panel-line"))

    def set_stream(self, text: str, cursor: str = "▌") -> None:
        """Cap nhat khoi stream duoc tao san trong body (dung cho thinking)."""
        self._stream.update(Text(text + cursor, style=C["muted"]))

    @property
    def steps(self) -> int:
        return self._steps


def _get_available_models() -> List[str]:
    """Kiem tra va tra ve danh sach model co san voi API key hien tai.
    Ket qua duoc cache de khong phai goi API moi lan."""
    global _AVAILABLE_MODELS_CACHE
    if _AVAILABLE_MODELS_CACHE is not None:
        return _AVAILABLE_MODELS_CACHE
    available = []
    try:
        from syncode.config import Config
        from syncode.llm.client import LLMClient
        config = Config()
        if not config.has_api_key:
            return list(POPULAR_MODELS)
        client = LLMClient(config)
        messages = [{"role": "user", "content": "hi"}]
        for model in POPULAR_MODELS:
            try:
                client.client.chat.completions.create(
                    model=model, messages=messages, max_tokens=5,
                )
                available.append(model)
            except Exception:
                pass
    except Exception:
        pass
    _AVAILABLE_MODELS_CACHE = available if available else list(POPULAR_MODELS)
    return _AVAILABLE_MODELS_CACHE


_AVAILABLE_MODELS_CACHE: Optional[List[str]] = None


# Routing quick/swarm dung chung dat trong orchestrator.
# Giu 2 wrapper tai day de code cu van import duoc tu syncode.ui.app.
from syncode.swarm.orchestrator import is_simple_chat as _orch_is_simple_chat
from syncode.swarm.orchestrator import needs_swarm as _orch_needs_swarm


def is_simple_chat(text: str) -> bool:
    """Cau chao/hoi don gian -> chi can hoi dap nhanh, khong chay swarm
    (tranh in ra 'Kế hoạch', 'Issues to fix', 'Fixes'... cho 1 loi chao)."""
    return _orch_is_simple_chat(text)


def needs_swarm(text: str) -> bool:
    """True nếu cần full swarm; False nếu quick 1 lượt là đủ."""
    return _orch_needs_swarm(text)


def build_suggestions(text: str) -> List[Tuple[str, str, str]]:
    """Sinh danh sach goi y (hien_thi, mo_ta, gia_tri_chen). Ham thuan de test."""
    text = text.strip()
    if not text.startswith("/"):
        return []
    cfg_set = "/.config set "
    if text.startswith(cfg_set):
        prefix = text[len(cfg_set):]
        return [
            (f"/.config set {k}", "config key", f"/.config set {k} ")
            for k in DEFAULTS if k.startswith(prefix)
        ]
    if text.rstrip() == "/.config set":
        return [(f"/.config set {k}", "config key", f"/.config set {k} ") for k in DEFAULTS]
    if text.startswith("/.model") and text != "/.model":
        raw = text[len("/.model"):]
        token = raw.strip().lower()
        roles = ["main", "planner", "worker", "w1", "w2", "critic", "refiner", "judge"]
        first = token.split()[0] if token else ""
        if first in roles and len(token.split()) > 1:
            return []  # da chon role — de nguoi dung tu go ten model
        suggestions: List[Tuple[str, str, str]] = []
        if first and " " not in token:
            # goi y role (w1/w2 co the co model rieng) truoc, roi cac model NIM
            suggestions += [
                (f"/.model {r} ", "role model" if r != "main" else "main model", f"/.model {r} ")
                for r in roles if r.startswith(first)
            ]
        # Chi goi y model co san voi API key hien tai (tim substring cho de)
        available = _get_available_models()
        suggestions += [
            (f"/.model {m}", "NIM model", f"/.model {m}")
            for m in available if token in m.lower()
        ]
        return suggestions
    word = text[1:]
    word = word.removeprefix(".")
    matches: List[Tuple[str, str, str]] = []
    for name, (_handler, usage, desc) in COMMANDS.items():
        if name.startswith(word.lower()):
            display = f"/.{name} {usage}".strip()
            matches.append((display, desc, f"/.{name}"))
    return matches


# =================================================================== adapter
class TUIAdapter:
    """Adapter interface giong ui.console — palette Catppuccin Mocha."""

    def __init__(self, app: SyncodeApp) -> None:
        self._app = app

    def info(self, text: str) -> None:
        self._app.post_chat(Text(f"  · {text}", style=C["muted"]))

    def success(self, text: str) -> None:
        self._app.post_chat(Text(f"  ✓ {text}", style=C["success"]))

    def warn(self, text: str) -> None:
        self._app.post_chat(Text(f"  ⚠ {text}", style=C["warning"]))

    def error(self, text: str) -> None:
        self._app.post_chat(Text(f"  ✗ {text}", style=f"bold {C['error']}"))

    def key_value_table(self, title: str, rows: dict, style: str = "cyan") -> None:
        self._app.post_chat(Text(f"▌ {title}", style=f"bold {C['secondary']}"))
        table = Table(box=None, show_header=False, padding=(0, 4))
        table.add_column(style="bold")
        table.add_column(style=C["text2"])
        for key, value in rows.items():
            table.add_row(str(key), str(value))
        self._app.post_chat(table)

    def list_rows(self, title: str, items, style: str = "cyan") -> None:
        self._app.post_chat(Text(f"▌ {title}", style=f"bold {C['secondary']}"))
        table = Table(box=None, show_header=False, padding=(0, 4))
        table.add_column(style=C["muted"])
        table.add_column(style=C["text2"])
        for i, item in enumerate(items, start=1):
            table.add_row(f"{i}.", str(item))
        self._app.post_chat(table)

    def agent_badge(self, name: str, icon: str, style: str, note: str = "") -> None:
        line = Text()
        line.append("▌ ", style=C["secondary"])
        line.append(f"{icon} {name}", style=f"bold {C['text']}")
        if note:
            line.append(f" — {note}", style=C["muted"])
        self._app.post_chat(line)

    def agent_panel(self, name: str, icon: str, style: str, content: str, subtitle: str = "") -> None:
        self.agent_badge(name, icon, style)
        for ln in (content or "(empty)").splitlines() or [""]:
            self._app.post_chat(Text(f"      {ln}", style=C["text2"]))

    def clear(self) -> None:
        self._app.clear_chat()

    def toggle_swarm_mode(self) -> None:
        """Chuyen doi Swarm/Quick mode (goi tu lenh /.swarm)."""
        self._app.action_toggle_swarm_mode()

    def get_plan_mode(self) -> str:
        """Mode plan/build hien tai cua app (cho lenh /.mode doc)."""
        return self._app._mode

    def set_plan_mode(self, mode: str) -> bool:
        """Dat plan/build truc tiep tu lenh /.mode. Tra False neu mode la."""
        return self._app.set_plan_mode(mode)


class PromptInput(Input):
    """Input chan phim Up/Down/Tab/Esc cho popup goi y; sang vien khi focus."""

    # Tab = chuyen mode plan/build, Shift+Tab = chuyen swarm/quick —
    # binding cua widget focus uu tien cao hon binding focus_next/focus_previous
    # cua Screen nen khong bi an mat. (Ctrl+W bo vi terminal hay an truoc.)
    BINDINGS = [
        ("tab", "toggle_mode_tab", "Plan/Build"),
        ("shift+tab", "toggle_swarm_tab", "Swarm/Quick"),
    ]

    async def action_toggle_mode_tab(self) -> None:
        app: SyncodeApp = self.app  # type: ignore[assignment]
        app.action_toggle_mode()

    async def action_toggle_swarm_tab(self) -> None:
        app: SyncodeApp = self.app  # type: ignore[assignment]
        app.action_toggle_swarm_mode()

    def on_focus(self) -> None:
        editor = self.parent
        if editor is not None:
            editor.add_class("editing")

    def on_blur(self) -> None:
        editor = self.parent
        if editor is not None:
            editor.remove_class("editing")

    async def _on_key(self, event) -> None:
        app: SyncodeApp = self.app  # type: ignore[assignment]
        if app.suggest_visible:
            if event.key == "down":
                event.stop()
                event.prevent_default()
                app.suggest_move(1)
                return
            if event.key == "up":
                event.stop()
                event.prevent_default()
                app.suggest_move(-1)
                return
            if event.key == "tab":
                event.stop()
                event.prevent_default()
                app.suggest_accept()
                return
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                app.hide_suggest()
                return
        # Input._on_key la coroutine trong textual — phai await
        await super()._on_key(event)


# =================================================================== banner
# Figlet-style block font — chu SYNCODE
_ASCII_FONT = {
    "S": ["████", "█   ", "████", "   █", "████"],
    "Y": ["█  █", "█  █", " ██ ", " █  ", " █  "],
    "N": ["█  █", "██ █", "█ ██", "█  █", "█  █"],
    "C": ["████", "█   ", "█   ", "█   ", "████"],
    "O": [" ██ ", "█  █", "█  █", "█  █", " ██ "],
    "D": ["███ ", "█  █", "█  █", "█  █", "███ "],
    "E": ["████", "█   ", "███ ", "█   ", "████"],
}

# Cac bo mau gradient (cinematic tren nen den) — banner xoay vong cac bo nay
_BANNER_SCHEMES = [
    ("gold", ["#FFE8CC", "#FFC799", "#FFB86B", "#8A6A4D"]),
    ("mint", ["#E0FFF5", "#99FFE4", "#5AE8C8", "#2E8C77"]),
    ("ice", ["#FFFFFF", "#C9C9C9", "#878787", "#454545"]),
    ("ember", ["#FFD9A8", "#FF9E64", "#E5484D", "#7A2E2E"]),
]


def _ascii_rows(word: str) -> List[str]:
    """Sinh 5 dong block-letter cho mot chu."""
    rows = [""] * 5
    for ch in word.upper():
        glyph = _ASCII_FONT[ch]
        for i in range(5):
            rows[i] += glyph[i] + "  "
    return rows


class Banner(Static):
    """Banner 'SYNCODE' ASCII tuong tac: click / space / enter doi mau gradient."""

    can_focus = True
    BINDINGS = [("space", "cycle", "recolor"), ("enter", "cycle", "recolor")]

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._scheme_idx = 0

    def on_mount(self) -> None:
        self._paint()

    def action_cycle(self) -> None:
        self._scheme_idx = (self._scheme_idx + 1) % len(_BANNER_SCHEMES)
        self._paint()

    def on_click(self) -> None:
        self.action_cycle()

    def _paint(self) -> None:
        name, scheme = _BANNER_SCHEMES[self._scheme_idx]
        kid = _ascii_rows("SYN")
        agent = _ascii_rows("CODE")
        width = max(max(len(r) for r in kid), max(len(r) for r in agent))

        def centered(rows: List[str]) -> List[str]:
            pad = (width - len(rows[0])) // 2
            return [" " * pad + r for r in rows]

        lines = centered(kid) + [""] + centered(agent)
        art = Text()
        total = len(lines)
        for i, ln in enumerate(lines):
            if ln:
                # gradient doc: gan dong dau vao mau sang, cuoi vao mau dam
                color = scheme[min(3, (i * (len(scheme) - 1)) // max(1, total - 1))]
                art.append(ln, style=f"bold {color}")
            art.append("\n")
        art.append("▀" * width + "\n", style=f"bold {scheme[3]}")
        # can giua ngang trong widget
        self.update(Align(art, "center"))
        self.tooltip = f"banner {name} — click / ctrl+b to recolor"

# =================================================================== dialogs
class DialogScreen(ModalScreen):
    """Dialog noi giua man hinh — opencode style (panel crust, vien surface)."""

    CSS = f"""
    /* overlay chi dim nhe 55% — TUI (chat · editor · footer) van hien ro
       phia sau, panel thuc su NOI TREN cung 1 man hinh */
    DialogScreen {{ align: center middle; background: {C['base']} 55%; }}
    #dialog {{
        width: 58; max-height: 72%;
        border: heavy {C["primary"]};
        background: {C['base']};
        padding: 0 1;
    }}
    #dialog-title {{
        text-style: bold; color: {C['primary']};
        background: {C['mantle']}; padding: 0 1;
    }}
    #dialog-hint {{ color: {C['muted']}; }}
    .dialog-list {{ height: auto; max-height: 14; border: none; background: transparent; }}
    .dialog-input {{ height: 1; border: none; background: transparent; }}
    .dialog-note {{ color: {C['muted']}; }}
    """
    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(self, title: str, hint: str = "esc close") -> None:
        super().__init__()
        # Inline styles — bắt buộc: textual 8.x type-selector chi match dung
        # ten class, nen rule `DialogScreen {...}` trong CSS khong ap dung len
        # cac subclass (ConfigModal...), cung nhu rule `Screen` cua App.CSS
        # (nen #000000 dac) khong the de nos overlay bang CSS.
        self.styles.background = Color.parse(C["base"]).with_alpha(0.55)
        self.styles.align = ("center", "middle")
        self._title = title
        self._hint = hint

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self._title, id="dialog-title")
            yield Vertical(id="dialog-body")
            yield Static(self._hint, id="dialog-hint")

    def set_body(self, *widgets) -> None:
        body = self.query_one("#dialog-body", Vertical)
        body.remove_children()
        body.mount(*widgets)


class HelpModal(DialogScreen):
    """Danh sach lenh /. — compact."""

    def __init__(self) -> None:

        super().__init__("commands", "esc close")

    def on_mount(self) -> None:
        lines = Text()
        for name, (_h, usage, desc) in COMMANDS.items():
            lines.append(f"  /.{name}", style=f"bold {C['primary']}")
            if usage:
                lines.append(f" {usage}", style=C["warning"])
            lines.append(f"  {desc}\n", style=C["muted"])
        self.set_body(Static(lines, classes="dialog-note"))


class ConfigModal(DialogScreen):
    """Panel cau hinh: chon key -> sua gia tri -> enter luu."""

    def __init__(self, config: Config) -> None:
        super().__init__("config", "up/down select · enter edit · esc close")
        self._config = config
        self._edit_key: Optional[str] = None

    def on_mount(self) -> None:
        self._show_list()

    def _show_list(self) -> None:
        ol = OptionList(classes="dialog-list")
        for key, value in self._config.as_display().items():
            line = Text()
            line.append(f"{key:<12}", style=f"bold {C['primary']}")
            line.append(f" {value}", style=C["text2"])
            ol.add_option(Option(line, id=key))
        self.set_body(ol)
        self._edit_key = None

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        key = event.option_id
        if not key:
            return
        self._edit_key = key
        inp = Input(
            value="" if key == "api_key" and not self._config.get(key) else str(self._config.get(key)),
            password=(key == "api_key"),
            classes="dialog-input",
        )
        self.set_body(Static(f"edit: {key}", classes="dialog-note"), inp)
        inp.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not self._edit_key:
            return
        try:
            self._config.set(self._edit_key, event.value)
        except (ValueError, TypeError):
            self.query_one("#dialog-hint", Static).update(
                f"! invalid value for {self._edit_key}"
            )
            return
        self._show_list()


class ModelModal(DialogScreen):
    """Panel chọn model RIÊNG CHO TỪNG ROLE (Main/Planner/Worker-1/...).

    Hai cột: trái = danh sách role (+ model hiện tại), phải = danh sách
    model để chọn. Không cần gõ lệnh — điều hướng bằng mũi tên + enter.
    """

    def __init__(self, config: Config, orchestrator: SwarmOrchestrator) -> None:
        super().__init__(
            "models",
            "up/down pick role · enter to models · enter select · esc close",
        )
        self._config = config
        self._orchestrator = orchestrator
        self._roles: List[Tuple[str, str, str]] = []  # (label, role_key, cfg_key)

    # ------------------------------------------------------------- helpers
    def _build_roles(self) -> None:

        roles: List[Tuple[str, str, str]] = [("Main", "main", "model")]
        roles.append(("Planner", "planner", ROLE_MODEL_KEYS["planner"]))
        for i, w in enumerate(self._orchestrator.workers, start=1):
            rk = "w1" if i == 1 else ("w2" if i == 2 else "worker")
            roles.append((w.name, rk, ROLE_MODEL_KEYS[rk]))
        for name in ("Critic", "Refiner", "Judge"):
            roles.append((name, name.lower(), ROLE_MODEL_KEYS[name.lower()]))
        self._roles = roles

    def _current_model(self, cfg_key: str) -> str:
        if cfg_key == "model":
            return str(self._config.get("model"))
        custom = str(self._config.get(cfg_key) or "")
        return custom or str(self._config.get("model"))

    def _current_role(self) -> Tuple[str, str, str]:
        """Role dang chon (label, role_key, cfg_key) — doc truc tiep tu
        role-list.highlighted de khong le vao event highlighted."""
        try:
            ol = self.query_one("#role-list", OptionList)
            idx = ol.highlighted if ol.highlighted is not None else 0
        except Exception:  # noqa: BLE001
            idx = 0
        if 0 <= idx < len(self._roles):
            return self._roles[idx]
        return self._roles[0]

    def _role_row(self, label: str, role_key: str, cfg_key: str) -> Text:
        line = Text()
        mark = "▸ " if role_key == self._current_role()[1] else "  "
        line.append(mark, style=C["border_hi"])
        line.append(f"{label:<12}", style=f"bold {C['text']}")
        model = self._current_model(cfg_key)
        is_custom = cfg_key != "model" and bool(self._config.get(cfg_key))
        line.append(self._short(model, 28), style=C["primary"] if is_custom else C["muted"])
        return line

    @staticmethod
    def _short(model: str, n: int) -> str:
        return model if len(model) <= n else model[: n - 1] + "…"

    # -------------------------------------------------------------- mount
    def on_mount(self) -> None:
        self._build_roles()
        self._filter = ""
        # rong panel 2 cot du cho ten model dai
        self.query_one("#dialog").styles.width = 72
        self.query_one("#dialog").styles.max_height = "80%"
        filt = Input(placeholder="filter models...", classes="dialog-input")
        roles = OptionList(classes="dialog-list", id="role-list")
        models = OptionList(classes="dialog-list", id="model-list")
        from textual.containers import Horizontal

        self.set_body(filt, Horizontal(roles, models))
        roles.styles.width = 34
        models.styles.width = "1fr"
        # defer fill: OptionList can mount xong moi nhan duoc options
        self.call_after_refresh(self._post_mount_fill)

    def _post_mount_fill(self) -> None:
        self._fill_roles()
        self._fill_models()
        self.query_one("#role-list", OptionList).focus()

    # --------------------------------------------------------------- fill
    def _fill_roles(self) -> None:
        ol = self.query_one("#role-list", OptionList)
        old = ol.highlighted
        ol.clear_options()
        for label, role_key, cfg_key in self._roles:
            ol.add_option(Option(self._role_row(label, role_key, cfg_key), id=role_key))
        ol.highlighted = old

    def _fill_models(self) -> None:
        ol = self.query_one("#model-list", OptionList)
        _label, role_key, cfg_key = self._current_role()
        current = self._current_model(cfg_key)
        ol.clear_options()
        if role_key != "main":
            ol.add_option(Option(
                Text("use main model", style=f"italic {C['muted']}"), id="@main"
            ))
        models = list(POPULAR_MODELS)
        if current not in models:
            models.insert(0, current)
        for m in models:
            if self._filter and not self._filter.lower() in m.lower():
                continue
            mark = " ✓" if m == current else ""
            ol.add_option(Option(
                Text(self._short(m + mark, 34), style=C["primary"] if mark else C["text2"]),
                id=m,
            ))
        ol.highlighted = 0 if ol.option_count else None

    # -------------------------------------------------------------- events
    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.has_class("dialog-input"):
            self._filter = event.value.strip()
            self._fill_models()

    def on_option_list_option_highlighted(self, event) -> None:
        if event.option_list.id == "role-list" and event.option_id:
            self._fill_roles()
            self._fill_models()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        ol_id = event.option_list.id
        if ol_id == "role-list":
            # enter tren role -> focus sang danh sach model
            self.query_one("#model-list", OptionList).focus()
            return
        option_id = event.option_id
        if not option_id:
            return
        _label, role_key, cfg_key = self._current_role()
        if str(option_id) == "@main":
            self._config.set(cfg_key, "")
        else:
            self._config.set(cfg_key, str(option_id))
        self._fill_roles()
        self._fill_models()
        # ve lai danh sach role de tiep tuc chon role khac
        self.query_one("#role-list", OptionList).focus()


class KeyModal(DialogScreen):
    """Nhap API key (masked)."""

    def __init__(self) -> None:
        super().__init__("api key", "enter save · esc cancel")

    def on_mount(self) -> None:
        inp = Input(placeholder="nvapi-...", password=True, classes="dialog-input")
        self.set_body(inp)
        inp.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())


class HistoryModal(DialogScreen):
    """Lich su session — chon session de mo lai cuoc tro chuyen."""

    def __init__(self, sessions: List[Session]) -> None:
        super().__init__("history", "up/down select · enter open · esc close")
        self._sessions = sessions
        self._session_by_id: dict = {}

    def on_mount(self) -> None:
        if not self._sessions:
            self.set_body(Static("(empty)", classes="dialog-note"))
            return
        ol = OptionList(classes="dialog-list")
        # Hien thi session moi nhat dau tien
        for i, session in enumerate(reversed(self._sessions), start=1):
            opt_id = f"s{i}"
            self._session_by_id[opt_id] = session.id
            line = Text()
            line.append(f"{i}. ", style=C["muted"])
            line.append(session.title, style=f"bold {C['text']}")
            line.append("\n   ", style="")
            line.append(session.summary, style=C["text2"])
            ol.add_option(Option(line, id=opt_id))
        self.set_body(ol)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        session_id = self._session_by_id.get(str(event.option_id))
        self.dismiss(session_id)


class AskUserModal(DialogScreen):
    """Form hỏi đáp: agent hỏi 1..5 câu trong MỘT form, user trả lời xong
    thì gửi toàn bộ về agent để xử lý (không cần chat nhiều lượt)."""

    def __init__(self, questions: List[str], on_done) -> None:
        title = "input needed"
        hint = "enter next field · enter on last sends · esc cancel"
        super().__init__(title, hint)
        self._questions = list(questions)
        self._on_done = on_done
        self._inputs: List[Input] = []
        self._answered = False

    def on_mount(self) -> None:
        box = Vertical(id="qa-box")
        self.set_body(box)  # mount box len dialog truoc, roi moi nhanh con
        for i, q in enumerate(self._questions):
            box.mount(Static(Text(f" {i + 1}. {q}", style=f"bold {C['text']}"), classes="dialog-note"))
            inp = Input(placeholder="your answer...", classes="dialog-input")
            self._inputs.append(inp)
            box.mount(inp)
        if self._inputs:
            self._inputs[0].focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            idx = self._inputs.index(event.input)
        except ValueError:
            return
        if idx < len(self._inputs) - 1:
            self._inputs[idx + 1].focus()
            return
        # cau cuoi -> gom cau tra loi va gui ve agent
        answers = [inp.value.strip() or "(không trả lời)" for inp in self._inputs]
        self._answered = True
        try:
            self._on_done(answers)
        finally:
            self.dismiss()

    def dismiss(self, result=None) -> None:
        # ESC (hoac dong modal kieu khac) van phai tra loi agent ngay,
        # khong de worker thread treo doi den timeout.
        if not self._answered:
            self._answered = True
            try:
                self._on_done(["(không trả lời)"] * max(1, len(self._questions)))
            except Exception:  # noqa: BLE001
                pass
        super().dismiss(result)


class ConfirmModal(DialogScreen):
    """Modal Yes/No cho lenh nguy hiem: chon xong tra bool ve agent ngay.
    ESC = deny (an toan). Mac dinh highlight No."""

    def __init__(self, prompt_text: str, on_done) -> None:
        super().__init__("confirm", "up/down select · enter ok · esc deny")
        self._prompt_text = prompt_text
        self._on_done = on_done
        self._answered = False

    def on_mount(self) -> None:
        box = Vertical(id="confirm-box")
        self.set_body(box)
        box.mount(Static(
            Text(f" {self._prompt_text}", style=f"bold {C['text']}"),
            classes="dialog-note",
        ))
        ol = OptionList(classes="dialog-list", id="confirm-list")
        ol.add_option(Option(Text("Yes, run it", style=f"bold {C['success']}"), id="yes"))
        ol.add_option(Option(Text("No, skip", style=f"bold {C['error']}"), id="no"))
        box.mount(ol)
        ol.highlighted = 1  # mac dinh ve No
        ol.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._answered = True
        try:
            self._on_done(str(event.option_id) == "yes")
        finally:
            self.dismiss()

    def dismiss(self, result=None) -> None:
        if not self._answered:
            self._answered = True
            try:
                self._on_done(False)
            except Exception:  # noqa: BLE001
                pass
        super().dismiss(result)


class ThinkingModal(DialogScreen):
    """Chon che do thinking rieng cua model: off / low / auto.
    ESC = giu nguyen, khong doi."""

    MODES = (
        ("off", "concise — tat reasoning, nhanh + re"),
        ("low", "light reasoning — can bang toc do/chat luong"),
        ("auto", "de model tu quyet dinh (mac dinh)"),
    )

    def __init__(self, config: Config) -> None:
        super().__init__("thinking", "up/down select · enter save · esc cancel")
        self._config = config

    def on_mount(self) -> None:
        cur = str(self._config.get("thinking", "auto") or "auto").strip().lower()
        ol = OptionList(classes="dialog-list", id="thinking-list")
        highlight = 0
        for i, (mode, desc) in enumerate(self.MODES):
            if mode == cur:
                highlight = i
            line = Text()
            line.append("✓ " if mode == cur else "  ", style=f"bold {C['success']}")
            line.append(mode, style=f"bold {C['text']}")
            line.append(f"  {desc}", style=C["muted"])
            ol.add_option(Option(line, id=mode))
        self.set_body(ol)
        ol.highlighted = highlight
        ol.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        mode = str(event.option_id or "")
        if mode in ("off", "low", "auto"):
            self._config.set("thinking", mode)
        self.dismiss(mode)


# =================================================================== app
class SyncodeApp(App):
    """TUI theo design language opencode: editor box · footer · dialog noi."""

    TITLE = "Syncode"
    CSS = f"""
    Screen {{ background: {C['base']}; }}
    /* chat: cung mot mau nen den tuyen; padding ngang 2 = dung cot viền
       cua editor box — mau chu khong tran ra ngoai khung */
    #chat {{ padding: 1 2; background: {C['base']}; }}
    .msg {{ margin: 0; min-height: 1; background: {C['base']}; }}
    .assistant {{ margin: 1 0 1 1; }}
    #chat Markdown {{ background: {C['base']}; }}
    #chat > .banner {{ background: {C['base']}; }}
    /* banner ASCII tuong tac: click / ctrl+b doi mau gradient (khong hover fx) */
    .banner {{ background: transparent; text-style: bold; }}
    .banner:focus {{ background: {C['mantle']}; }}

    /* ---- collapse panel (hoat dong / thinking) ---- */
    .collapse-panel {{
        height: auto; background: {C['base']}; margin: 0;
        border-left: thick {C['border']};
    }}
    .think-panel {{ border-left: thick {C['secondary']}; }}
    .panel-header {{ height: 1; background: {C['mantle']}; }}
    .panel-body {{ height: auto; padding: 0 0 0 1; display: block; }}
    .panel-line {{ height: auto; background: {C['base']}; }}

    /* ---- editor box (opencode) ---- */
    #editor {{
        dock: bottom; height: auto; margin: 0 2 1 2;
        background: {C['crust']};
        border: solid {C["border"]};
    }}
    #editor.editing {{ border: solid {C["primary"]}; }}
    #prompt {{
        height: 1; border: none; padding: 0 1; background: transparent;
        color: {C['text']};
    }}
    #prompt:focus {{ border: none; text-style: bold; }}
    #editor-meta {{
        height: 1; padding: 0 1; color: {C['muted']}; background: transparent;
    }}

    /* ---- suggest popup (xuat hien tren editor) ---- */
    #suggest {{
        display: none; height: auto; max-height: 7;
        margin: 0 2 1 2;
        background: {C['mantle']};
        border: solid {C['border_hi']};
    }}
    #suggest.visible {{ display: block; }}

    /* ---- footer 1 dong (opencode segments: trai status · phai author) ---- */
    #footer {{
        dock: bottom; height: 1; padding: 0 2;
        background: {C['mantle']};
    }}
    #footer-left {{ width: 1fr; color: {C['muted']}; }}
    #footer-right {{ width: auto; color: {C['muted']}; }}
    """
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+b", "cycle_banner", "Banner color"),
        ("ctrl+t", "toggle_mode", "Plan/Build"),
        ("shift+tab", "toggle_swarm_mode", "Swarm/Quick"),
    ]

    def __init__(self, config: Config, client: LLMClient, orchestrator: SwarmOrchestrator) -> None:
        super().__init__()
        self._config = config
        self._client = client
        self._orchestrator = orchestrator
        # tools builtin + MCP (mcp_servers doc tu config; yeu cau pip install mcp)
        self._tools, self._mcp = build_tool_stack(config)
        # gan tool registry vao orchestrator de Worker run co duoc tool-calling
        self._orchestrator.tools = self._tools
        # lich su: doc tu ~/.syncode/history.json de /.history hoat dong
        self._sessions: List[Session] = _load_history_file()
        self._current_session_id: str | None = None
        self._ctx = SlashCommandContext(config, TUIAdapter(self), client, orchestrator, [])
        self._ctx.sessions = self._sessions
        self._ctx.current_session_id = self._current_session_id
        self._ctx.tools = self._tools
        self._ctx.mcp = self._mcp
        self._mode = "build"  # plan | build
        self._quick_mode = False  # False = swarm (mac dinh), True = quick 1 luot chat
        self._spinner_idx = 0
        self._agent_state: dict = {}
        self._suggest_items: List[Tuple[str, str, str]] = []
        self._suggest_navigated = False
        self._stream_buffer = ""
        self._think_buffer = ""
        self._stream_widget: Optional[Markdown] = None
        self._activity: Optional[CollapsePanel] = None
        self._think_panel: Optional[CollapsePanel] = None
        self._changes: Optional[ChangeTracker] = None
        self._changes_panel: Optional[CollapsePanel] = None
        self._busy = False
        self._question: Optional[str] = None

    # ------------------------------------------------------------ compose
    def compose(self) -> ComposeResult:
        # suggest popup dock bottom nam tren editor, chi hien khi co goi y
        yield OptionList(id="suggest")
        with Container(id="editor"):
            yield PromptInput(
                placeholder="Ask anything — type / for commands",
                id="prompt",
            )
            yield Static(Text(), id="editor-meta")
        with Horizontal(id="footer"):
            yield Static(Text(), id="footer-left")
            yield Static(_render_footer_right(), id="footer-right")
        yield VerticalScroll(id="chat")

    def on_mount(self) -> None:
        self.register_theme(SYNCODE_THEME)
        self.theme = SYNCODE_THEME.name
        # dong ho xoay vong cho footer khi model dang chay
        self.set_interval(0.12, self._tick_spinner)
        for agent in self._orchestrator.agents():
            self._agent_state[agent.name] = ("idle", "")
        self._render_editor_meta()
        self.refresh_topbar()
        self._render_footer()
        self._post_banner()
        self.query_one("#prompt", Input).focus()
        if not self._config.has_api_key:
            self.post_chat(Text(
                "  ! No API key — run /.key to open the key panel (build.nvidia.com)",
                style=C["warning"],
            ))

    # ------------------------------------------------------------ banner
    def action_cycle_banner(self) -> None:
        """Phim tat toan cuc ctrl+b — doi mau gradient banner moi noi."""
        self.query_one(Banner).action_cycle()

    # ------------------------------------------------------------ spinner
    def _tick_spinner(self) -> None:
        """Xoay vong hoat anh trong footer khi model dang chay (busy)."""
        if not self._busy:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(SPINNER_FRAMES)
        self._render_footer()

    def _spinner(self) -> str:
        return SPINNER_FRAMES[self._spinner_idx]

    # ------------------------------------------------------------ mode
    def action_toggle_swarm_mode(self) -> None:
        """Doi qua lai swarm / quick (shift+tab hoac /.swarm). Quick = 1 luot chat
        truc tiep, Swarm = chay day du Planner > Workers > Critic > Judge."""
        self._quick_mode = not self._quick_mode
        self._render_editor_meta()
        self.query_one("#editor-meta", Static).update(self._render_editor_meta())
        self._render_footer()
        if self._quick_mode:
            note = Text(
                "  mode -> quick (single pass, shift+tab for swarm)",
                style=f"bold {C['warning']}",
            )
        else:
            note = Text(
                "  mode -> swarm (shift+tab for quick)",
                style=f"bold {C['success']}",
            )
        self.post_chat(note)

    def set_plan_mode(self, mode: str) -> bool:
        """Dat che do plan/build truc tiep (dung cho lenh /.mode).
        Tra False neu mode khong hop le. Khong post chat (caller tu thong bao)."""
        mode = (mode or "").lower()
        if mode not in ("plan", "build"):
            return False
        if mode != self._mode:
            self._mode = mode
            self._tools.set_allow_dangerous(mode == "build")
            self._render_editor_meta()
            self.query_one("#editor-meta", Static).update(self._render_editor_meta())
            self._render_footer()
        return True

    def action_toggle_mode(self) -> None:
        """Doi qua lai plan / build (tab hoac ctrl+t). Swarm van chay day du
        o ca 2 mode — khac biet la quyen dung tool:
        Plan  = chi tool doc/tinh (doc file, xem thu muc, tinh toan, thoi gian)
        Build = full tools (doc/ghi file, chay lenh, MCP dangerous)."""
        self._mode = "plan" if self._mode == "build" else "build"
        self._tools.set_allow_dangerous(self._mode == "build")
        self._render_editor_meta()
        self.query_one("#editor-meta", Static).update(self._render_editor_meta())
        self._render_footer()
        if self._mode == "plan":
            note = Text(
                "  mode -> plan (read-only, tab for build)",
                style=f"bold {C['warning']}",
            )
        else:
            note = Text(
                "  mode -> build (full tools, tab for plan)",
                style=f"bold {C['success']}",
            )
        self.post_chat(note)

    def _post_banner(self) -> None:
        """Welcome — chu 'SYNCODE' ASCII block-letter, gradient cinematic,
        TUONG TAC: click / space / enter de doi mau (gold → mint → ice → ember).
        Kem dong credit + pipeline, tat ca can giua, nen den tuyen."""
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(Banner(classes="msg banner"))
        chat.scroll_end(animate=False, force=True)

        sub = Text()
        sub.append("S W A R M   A G E N T", style=f"bold {C['text']}")
        sub.append("   ·   ", style=C["border_hi"])
        sub.append("NVIDIA NIM", style=f"bold {C['secondary']}")
        sub.append("   ·   ", style=C["border_hi"])
        sub.append(f"v{__version__}", style=C["muted"])
        hint = Text(
            "click banner or ctrl+b to change color", style="italic " + C["muted"]
        )
        self.post_chat(Align.center(sub))
        self.post_chat(Align.center(Text("\n", style="")))
        self.post_chat(Align.center(hint))
        self.post_chat(Align.center(Text("\n", style="")))


    # ------------------------------------------------------------ renderers
    def _render_editor_meta(self) -> Text:
        """Meta line in the editor box — opencode style."""
        line = Text()
        if self._mode == "plan":
            line.append(" [plan]", style=f"bold {C['warning']}")
        else:
            line.append(" [build]", style=f"bold {C['success']}")
        if self._quick_mode:
            line.append(" [quick]", style=f"bold {C['warning']}")
        else:
            line.append(" [swarm]", style=f"bold {C['primary']}")
        key_ok = self._config.has_api_key
        line.append(" · ", style=C["muted"])
        line.append(str(self._config.get("model")), style=C["text2"])
        line.append(" · ", style=C["muted"])
        line.append("key ok" if key_ok else "no key", style=C["success"] if key_ok else C["warning"])
        line.append(" · ", style=C["muted"])
        line.append(f"{len(self._tools)} tools", style=C["muted"])
        line.append(" · ", style=C["muted"])
        line.append("tab mode · shift+tab agent · enter send", style=C["muted"])
        return line

    def refresh_topbar(self) -> None:
        self._render_editor_meta()
        self.query_one("#editor-meta", Static).update(self._render_editor_meta())

    def _render_footer(self) -> None:
        """Footer segments kieu opencode: status · agents · keybind chips."""
        line = Text()
        if self._busy:
            line.append(f"{self._spinner()} busy", style=f"bold {C['warning']}")
        else:
            line.append("○ ready", style=C["muted"])
        line.append("  │  ", style=C["border_hi"])
        if self._mode == "plan":
            line.append("(plan) ", style=f"bold {C['warning']}")
        for agent in self._orchestrator.agents():
            status, _note = self._agent_state.get(agent.name, ("idle", ""))
            icon = {"working": self._spinner(), "done": "✓", "idle": "·"}[status]
            color = {"working": C["warning"], "done": C["success"], "idle": C["muted"]}[status]
            short = agent.name.replace("Worker-", "w")
            line.append(f"{icon} {short} ", style=color)
        line.append("  │  ", style=C["border_hi"])
        # keybind chips (neu nen crust nhu opencode)
        chip_bg = C["crust"]
        for key, desc in (("tab", "mode"), ("shift+tab", "agent"), ("enter", "send"), ("ctrl+c", "quit")):
            line.append(f" {key} ", style=f"{C['text']} on {chip_bg}")
            line.append(f" {desc} ", style=C["muted"])
        self.query_one("#footer-left", Static).update(line)

    def refresh_sidebar(self) -> None:
        for agent in self._orchestrator.agents():
            self._agent_state.setdefault(agent.name, ("idle", ""))
        self._render_footer()

    # ------------------------------------------------------------ chat helpers
    def post_chat(self, renderable) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(Static(renderable, classes="msg"))
        chat.scroll_end(animate=False, force=True)

    def clear_chat(self) -> None:
        self.query_one("#chat", VerticalScroll).remove_children()
        self._stream_widget = None

    def set_agent_status(self, name: str, status: str, note: str = "") -> None:
        self._agent_state[name] = (status, note)
        self._render_footer()

    # ------------------------------------------------------------ suggestions
    @property
    def suggest_visible(self) -> bool:
        return self.query_one("#suggest", OptionList).has_class("visible")

    def hide_suggest(self) -> None:
        popup = self.query_one("#suggest", OptionList)
        popup.remove_class("visible")
        popup.clear_options()
        self._suggest_items = []
        self._suggest_navigated = False

    def _update_suggestions(self, text: str) -> None:
        popup = self.query_one("#suggest", OptionList)
        items = build_suggestions(text)
        self._suggest_items = items
        self._suggest_navigated = False
        popup.clear_options()
        if not items:
            popup.remove_class("visible")  # an popup khi khong co goi y
            return
        for display, desc, _value in items:
            line = Text()
            line.append(display, style=f"bold {C['primary']}")
            line.append(f"  {desc}", style=C["muted"])
            popup.add_option(Option(line))
        popup.highlighted = 0
        popup.add_class("visible")

    def suggest_move(self, delta: int) -> None:
        popup = self.query_one("#suggest", OptionList)
        if not self._suggest_items:
            return
        current = popup.highlighted if popup.highlighted is not None else 0
        popup.highlighted = max(0, min(popup.option_count - 1, current + delta))
        self._suggest_navigated = True

    def suggest_accept(self) -> str:
        """Chon goi y dang highlight -> dien vao prompt. Tra ve gia tri da dien
        ("" neu khong co goi y) — de nguoi goi (ENTER) quyet dinh chay hay khong."""
        if not self._suggest_items:
            return ""
        idx = popup = self.query_one("#suggest", OptionList).highlighted
        if popup is None or idx is None:
            return ""
        _display, _desc, value = self._suggest_items[min(idx, len(self._suggest_items) - 1)]
        prompt = self.query_one("#prompt", Input)
        prompt.value = value
        prompt.cursor_position = len(value)
        self.hide_suggest()
        return value

    # ------------------------------------------------------------ modal routing
    def _open_modal(self, name: str) -> None:
        if name == "config":
            self.push_screen(ConfigModal(self._config), self._on_config_close)
        elif name == "model":
            self.push_screen(
                ModelModal(self._config, self._orchestrator), self._on_modal_closed
            )
        elif name == "key":
            self.push_screen(KeyModal(), self._on_key_pick)
        elif name == "history":
            self.push_screen(HistoryModal(self._sessions), self._on_history_pick)
        elif name == "thinking":
            self.push_screen(ThinkingModal(self._config), self._on_modal_closed)
        else:
            self.push_screen(HelpModal())

    def _on_config_close(self, _result) -> None:
        self.refresh_topbar()
        self._orchestrator.set_num_workers(int(self._config.get("num_workers")))
        self.refresh_sidebar()

    def _on_modal_closed(self, _result=None) -> None:
        """Sau khi dong panel model — áp dụng model riêng từng role + refresh."""
        try:
            self._orchestrator.apply_models(self._config)
        except Exception:  # noqa: BLE001
            pass
        self.refresh_topbar()

    def _on_key_pick(self, key) -> None:
        if key:
            self._config.set("api_key", key)
            self._client.reset()
            self.refresh_topbar()
            self.post_chat(Text("  ✓ api key saved", style=C["success"]))

    def _on_history_pick(self, session_id) -> None:
        """Mo lai session da chon — hien thi toan bo cuoc tro chuyen."""
        if not session_id:
            return
        for session in self._sessions:
            if session.id == session_id:
                self._restore_session(session)
                break

    def _restore_session(self, session: Session) -> None:
        """Khoi phuc session — hien thi lai toan bo exchange."""
        self.clear_chat()
        self._current_session_id = session.id
        # Hien thi banner lai
        self._post_banner()
        # Hien thi toan bo exchange trong session
        chat = self.query_one("#chat", VerticalScroll)
        for exchange in session.exchanges:
            # Hien thi cau hoi
            line = Text()
            line.append("❯ ", style=f"bold {C['secondary']}")
            line.append(exchange.question, style=C["text"])
            chat.mount(Static(line, classes="msg"))
            # Hien thi cau tra loi
            if exchange.answer.strip():
                md = Markdown(exchange.answer, classes="assistant")
                chat.mount(md)
        chat.scroll_end(animate=False, force=True)
        self.post_chat(Text(
            f"  restored session ({len(session.exchanges)} turns)",
            style=f"italic {C['muted']}",
        ))

    # ------------------------------------------------------------ input events
    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "prompt":
            self._update_suggestions(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt":
            return
        self._submit_prompt(event.input)

    def _submit_prompt(self, prompt: Input) -> None:
        """Xu ly cau hoi nguoi dung (dung chung cho ENTER va TAB).

        Popup goi y dang mo:
        - mui ten ↑↓ = di chuyen chon;
        - enter = dung lenh dang chon ngay lap tuc (khong gui lam tin nhan),
          tru khi goi y dang "dien tiep" (vi du /.config set <key>, /.model w1
          — vi tri con tro cho user go tiep) thi chi dien vao o nhap.
        """
        if self.suggest_visible and self._suggest_items:
            value = self.suggest_accept()
            # gia tri khong co khoang trang duoi = lenh hoan chinh -> chay luon
            if value and not value.endswith(" "):
                prompt.value = ""
                self._handle_user_text(value)
            return
        text = prompt.value.strip()
        prompt.value = ""
        self.hide_suggest()
        if not text or self._busy:
            return
        self._handle_user_text(text)

    # ------------------------------------------------------------ dispatch
    def _handle_user_text(self, text: str) -> None:
        line = Text()
        line.append("❯ ", style=f"bold {C['secondary']}")
        line.append(text, style=C["text"])
        self.post_chat(line)
        if is_slash_command(text):
            self._run_slash(text)
            return
        if not self._config.has_api_key:
            self.post_chat(Text("  ✗ run /.key <key> first", style=f"bold {C['error']}"))
            return
        if self._quick_mode:
            self._start_quick_reply(text)
        else:
            # Bắt buộc swarm mọi câu hỏi khi ở mode Swarm.
            # Chỉ quick khi user bật mode Quick (shift+tab / /.swarm).
            self._start_swarm_task(text)

    def _run_slash(self, text: str) -> None:
        parts = text.split()
        name = parts[0][2:].lower() if parts[0].startswith("/.") else parts[0][1:].lower()
        has_args = len(parts) > 1
        # Lenh mo panel (chi khi go dung lenh, khong co tham so)
        if not has_args and name in ("config", "model", "key", "help", "history", "thinking"):
            self._open_modal(name)
            return
        # Con lai chay inline (/.config set ..., /.save, /.exit ...)
        try:
            should_exit = execute(text, self._ctx)
        except Exception as exc:  # noqa: BLE001
            self.post_chat(Text(f"  ✗ command error: {exc}", style=f"bold {C['error']}"))
            should_exit = False
        self.refresh_topbar()
        self._orchestrator.set_num_workers(int(self._config.get("num_workers"))
        )
        self.refresh_sidebar()
        if should_exit:
            self.exit()

    # ------------------------------------------------------------ swarm run
    def _start_swarm_task(self, question: str) -> None:
        self._busy = True
        self._question = question
        self.query_one("#prompt", Input).disabled = True
        self._stream_buffer = ""
        self._think_buffer = ""
        self._stream_widget = None
        self._answer_buffer = ""
        self._think_panel = None
        self._activity = None
        # theo doi thay doi file cua luot nay de ve diff (do/them) phia tren dap an
        self._changes = ChangeTracker()
        self._changes_panel = None
        for name in self._agent_state:
            self.set_agent_status(name, "idle")
        self._render_footer()

        def on_event(event: str, data) -> None:
            self.call_from_thread(self._handle_event, event, data)

        def worker() -> None:
            orchestrator = self._orchestrator
            orchestrator.on_event = on_event
            orchestrator.changes = self._changes
            orchestrator.ask_callback = self._ask_user_handler
            orchestrator.confirm_callback = self._confirm_handler
            try:
                # Swarm chay day du o ca 2 mode. Plan = chi tool doc/tinh,
                # Build = full tools (doc/ghi file, chay lenh).
                self._tools.set_allow_dangerous(self._mode == "build")
                self._tools.confirm_dangerous = bool(self._config.get("confirm_dangerous", True))
                hist = compacted_history(self._sessions, self._current_session_id)
                todo_list = open_todos(self._sessions, self._current_session_id)
                result = orchestrator.run(
                    question,
                    max_rounds=int(self._config.get("max_rounds")),
                    history=hist,
                    todos=todo_list,
                )
                # truyen answer ve UI: fallback khi model khong stream
                self.call_from_thread(self._handle_event, "done", result.answer)
            except LLMError as exc:
                self.call_from_thread(self._handle_event, "error", str(exc))
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._handle_event, "error", f"{type(exc).__name__}: {exc}")
            finally:
                orchestrator.on_event = None
                orchestrator.changes = None
                orchestrator.ask_callback = None
                orchestrator.confirm_callback = None

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------ quick reply
    def _start_quick_reply(self, question: str) -> None:
        """Tra loi nhanh cho cau chao/hoi don gian: 1 luot chat truc tiep,
        KHONG chay swarm, KHONG tool, KHONG panel hoat dong — chi in loi dap."""
        self._busy = True
        self._question = question
        self.query_one("#prompt", Input).disabled = True
        self._stream_buffer = ""
        self._think_buffer = ""
        self._stream_widget = None
        self._think_panel = None
        self._activity = None
        self._changes = ChangeTracker()
        self._changes_panel = None
        self._render_footer()

        def on_event(event: str, data) -> None:
            self.call_from_thread(self._handle_event, event, data)

        def worker() -> None:
            orchestrator = self._orchestrator
            orchestrator.on_event = on_event
            orchestrator.changes = self._changes
            try:
                hist = compacted_history(self._sessions, self._current_session_id)
                answer = orchestrator.quick_reply(
                    question,
                    on_delta=lambda piece: orchestrator._emit("synth_delta", piece),
                    history=hist,
                    on_thinking=lambda piece: orchestrator._emit("think_delta", piece),
                )
                self.call_from_thread(self._handle_event, "done", answer)
            except LLMError as exc:
                self.call_from_thread(self._handle_event, "error", str(exc))
            except Exception as exc:  # noqa: BLE001
                self.call_from_thread(self._handle_event, "error", f"{type(exc).__name__}: {exc}")
            finally:
                orchestrator.on_event = None
                orchestrator.changes = None

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    # ------------------------------------------------------- panel helpers
    def _ensure_activity(self, title: str = "activity") -> CollapsePanel:
        """Panel thu gon chua toan bo buoc swarm cua luot hien tai.
        Mac dinh COLLAPSED — user chi thay dap an stream, khong bi phat
        cac buoc trung gian (Kế hoạch, Worker, Critic...) lam xon xao chat."""
        chat = self.query_one("#chat", VerticalScroll)
        if self._activity is None:
            panel = CollapsePanel(
                title, accent=C["primary"], classes="msg collapse-panel activity"
            )
            panel.collapse()  # bat dau de o trang thai thu gon
            chat.mount(panel)
            self._activity = panel
        return self._activity

    def _ensure_think(self) -> CollapsePanel:
        """Panel thinking — mo khi model bat dau suy nghi, dong khi xong."""
        chat = self.query_one("#chat", VerticalScroll)
        if self._think_panel is None:
            panel = CollapsePanel(
                "thinking", accent=C["secondary"], classes="msg collapse-panel think-panel"
            )
            chat.mount(panel)
            self._think_panel = panel
        return self._think_panel

    # ------------------------------------------------------- ask-user (q&a)
    def _ask_user_handler(self, questions) -> List[str]:
        """Chay trong thread worker: mo modal hoi tren main thread, cho den
        khi user tra loi xong roi tra ket qua ve tool ask_user/qa_user."""
        box: dict = {}
        done = threading.Event()

        def on_answers(answers) -> None:
            box["answers"] = list(answers)
            done.set()

        try:
            self.call_from_thread(self._open_ask_modal, list(questions), on_answers)
        except Exception:  # noqa: BLE001
            return []
        done.wait(timeout=900)  # user co the can nhieu thoi gian
        return box.get("answers", [])

    def _open_ask_modal(self, questions: List[str], on_answers) -> None:
        self.push_screen(AskUserModal(questions, on_answers))

    # ------------------------------------------------------- confirm (y/n)
    def _confirm_handler(self, prompt_text) -> bool:
        """Chay trong thread worker: mo modal confirm tren main thread, cho
        user chon Yes/No. Mac dinh deny khi timeout/exception."""
        box: dict = {}
        done = threading.Event()

        def on_answer(ok) -> None:
            box["ok"] = bool(ok)
            done.set()

        try:
            self.call_from_thread(self._open_confirm_modal, str(prompt_text), on_answer)
        except Exception:  # noqa: BLE001
            return False
        done.wait(timeout=300)
        return bool(box.get("ok", False))

    def _open_confirm_modal(self, prompt_text: str, on_answer) -> None:
        self.push_screen(ConfirmModal(prompt_text, on_answer))

    # ------------------------------------------------- changes panel (diff)
    def _mount_changes_panel(self) -> None:
        """Panel 'Thay doi' dat PHIA TREN cau tra loi chinh: moi file + diff
        (dòng xóa = nền đỏ, dòng thêm = nền xanh lá), thu gon/phong to duoc."""
        tracker = self._changes
        if tracker is None or not tracker.files:
            return
        chat = self.query_one("#chat", VerticalScroll)
        add_t, rem_t = tracker.added_removed()
        title = f"changes · {len(tracker.files)} files"
        panel = CollapsePanel(title, accent=C["success"], classes="msg collapse-panel changes-panel")
        chat.mount(panel)
        self._changes_panel = panel
        # dòng xóa / thêm với highlight nền
        del_style = f"bold {C['error']} on #3a0d0d"
        add_style = f"bold {C['success']} on #0d3a0d"
        for path, e in tracker.files.items():
            if e["deleted"] and not e["after"]:
                head = Text(f"  - {path}  (deleted)", style=f"bold {C['error']}")
            elif not e["before"] and e["after"]:
                head = Text(f"  + {path}  (new)", style=f"bold {C['success']}")
            else:
                head = Text(f"  ~ {path}", style=f"bold {C['text']}")
            blocks = tracker.diff_blocks(path)
            if e["deleted"] and not e["after"] and not blocks:
                panel.add_line(head)
                continue
            head_txt = head
            tail = Text()
            for kind, block in blocks:
                sign = kind if kind in ("+", "-") else " "
                for ln in block.splitlines():
                    styl = add_style if kind == "+" else (del_style if kind == "-" else C["muted"])
                    tail.append(f"      {sign} {ln}\n", style=styl)
            merged = Text()
            merged.append_text(head_txt)
            merged.append("\n")
            merged.append_text(tail)
            panel.add_line(merged)
        panel.set_title(
            f"{title}  (+{add_t} −{rem_t})",
            C["success"] if add_t else C["warning"],
        )
        chat.scroll_end(animate=False, force=True)

    def _finish_ui_turn(self) -> None:
        self._busy = False
        self._question = None
        # giu tracker de /.undo co the hoan tac sau khi luot ket thuc
        self._ctx.last_changes = self._changes
        self.query_one("#prompt", Input).disabled = False
        self.query_one("#prompt", Input).focus()
        self._render_footer()

    def _final_answer(self) -> str:
        """Tach thinking khoi buffer stream de lay cau tra loi cuoi cung."""
        before, thinking, done, after = split_thinking(self._stream_buffer)
        answer = after if (thinking or done) else before
        if not answer.strip() and (thinking or done):
            answer = self._stream_buffer.replace(THINK_OPEN, " ").replace(THINK_CLOSE, " ")
        # don dep tag think sot lai + khoang trang du thua o dau/cuoi
        return strip_think_tags(answer).strip()

    def _is_echo_dup_empty(self) -> bool:
        """True neu luot vua roi provider nhan doi trace vao content (echo-dup)
        va khong con dap an that (repair that bai hoac khong co). Dung de
        `done` khong dump trace 7k chars ra chat."""
        try:
            orch = getattr(self, "_orchestrator", None)
            info = getattr(getattr(orch, "client", None), "last_call", None) or {}
            if info.get("answer_was_echo"):
                return True
            return bool(getattr(orch, "last_echo_repair", False))
        except Exception:  # noqa: BLE001
            return False

    def _handle_event(self, event: str, data) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        if event == "phase":
            name, note = data
            self.set_agent_status(name, "working", note)
            self._ensure_activity().set_title(f"{name}: {note}", C["warning"])
        elif event == "plan":
            self.set_agent_status("Planner", "done", "")
            head = Text()
            head.append("▌ ", style=C["secondary"])
            head.append("plan", style=f"bold {C['text']}")
            task_lines: List[Text] = []
            for task in data:  # SubTask dataclass hoac dict
                tid = task["id"] if isinstance(task, dict) else task.id
                title = task["title"] if isinstance(task, dict) else task.title
                tline = Text()
                tline.append(f"{tid} ", style=f"bold {C['primary']}")
                tline.append(title, style=C["text2"])
                task_lines.append(tline)
            # luon gom vao panel (collapse) — khong post chat truc tien
            panel = self._ensure_activity("plan")
            panel.add_line(head)
            for t in task_lines:
                panel.add_line(Text("  " + t.plain, style=t.style))
        elif event == "worker_done":
            name, subtask_id, output = data
            self.set_agent_status(name, "done", "")
            preview = output.strip().splitlines()[0] if output.strip() else ""
            # bo tien to "### Subtask t1:" / markdown dau dong de preview gon
            preview = preview.lstrip("#").strip()
            if preview.lower().startswith("subtask"):
                preview = preview.split(None, 1)[-1] if " " in preview else ""
            if len(preview) > 72:
                preview = preview[:72] + "…"
            line = Text()
            line.append("> ", style=C["secondary"])
            line.append(f"{name} ", style=f"bold {C['text']}")
            line.append(f"[{subtask_id}]", style=C["muted"])
            if preview:
                line.append(f"  {preview}", style=C["muted"])
            # luon gom vao panel (collapse) — khong post chat truc tien
            self._ensure_activity().add_line(line)
        elif event == "tool":
            # Agent dung tool (builtin hoac MCP): (agent, tool_name, output)
            agent_name, tool_name, output = data
            preview = output.strip().splitlines()[0] if output.strip() else ""
            if len(preview) > 72:
                preview = preview[:72] + "…"
            line = Text()
            line.append("* ", style=C["primary"])
            line.append(f"{agent_name} → {tool_name}", style=f"bold {C['text']}")
            if preview:
                line.append(f"  {preview}", style=C["muted"])
            self._ensure_activity().add_line(line)
        elif event == "critique":
            verdict = str(data.get("verdict", ""))
            ok = verdict.upper().startswith("APPROVE")
            self.set_agent_status("Critic", "done", "")
            color = C["success"] if ok else C["warning"]
            line = Text()
            line.append("▌ ", style=color)
            line.append("review ", style=f"bold {C['text']}")
            line.append(verdict, style=f"bold {color}")
            panel = self._ensure_activity()
            panel.add_line(line)
            for ln in str(data.get("guidance", "")).splitlines():
                if ln.strip():
                    panel.add_line(Text("  " + ln.strip(), style=C["muted"]))
        elif event == "synth_delta":
            # Stream nhe bang Static; ho tro tag  cua model thinking:
            # hien panel "Thinking" trong khi suy nghi, dong lai khi xong.
            # Loc realtime cac dong rac (Here's think progress, Issues to fix, Fixes...)
            # de user chi thay noi dung thuc su.
            self._stream_buffer += data
            before, thinking, done, after = split_thinking(self._stream_buffer)
            if thinking or done:
                panel = self._ensure_think()
                if done:
                    panel.set_stream(thinking.strip(), cursor="")
                    panel.set_title("thinking · done", C["success"])
                    panel.collapse()
                    answer_text = after
                else:
                    panel.set_stream(thinking, cursor="▌")
                    panel.set_title("thinking...", C["secondary"])
                    answer_text = ""
            else:
                answer_text = before
            # loc dong rac truoc khi hien thi
            answer_text = _clean_stream_text(answer_text)
            if answer_text:
                if self._stream_widget is None:
                    st = Static(Text(), classes="assistant assistant-stream")
                    chat.mount(st)
                    self._stream_widget = st
                    self._answer_buffer = ""
                self._answer_buffer = answer_text
                self._stream_widget.update(Text(answer_text + "▌", style=C["text2"]))
            chat.scroll_end(animate=False, force=True)
        elif event == "think_delta":
            # Reasoning rieng kenh tu provider: hien panel thinking,
            # KHONG tron vao dap an.
            self._think_buffer = (self._think_buffer + str(data or ""))[-8000:]
            panel = self._ensure_think()
            panel.set_stream(self._think_buffer, cursor="▌")
            panel.set_title("thinking...", C["secondary"])
            chat.scroll_end(animate=False, force=True)
        elif event == "synth_done":
            self.set_agent_status("Judge", "done", "")
        elif event == "done":
            # Chot cau tra loi: thay Static stream bang Markdown hoan chinh.
            # full tinh tu ket qua orchestrator (fallback khi model khong stream)
            # roi den _final_answer (tach thinking + don dep tag sot).
            full = data.strip() if isinstance(data, str) and data.strip() else ""
            full = clean_final_answer(full) if full else ""
            if not full:
                full = self._final_answer()
            if self._stream_widget is not None:
                try:
                    self._stream_widget.remove()
                except Exception:  # noqa: BLE001
                    pass
                self._stream_widget = None
            # Panel "Thay doi file" dat PHIA TREN cau tra loi chinh (nhu
            # code-agent: diff do/them cls tren, dap an noi dung ben duoi).
            if self._changes is not None and self._changes.files:
                self._mount_changes_panel()
            if full:
                md = Markdown(classes="assistant")
                chat.mount(md)
                self.call_after_refresh(md.update, full)
                chat.scroll_end(animate=False, force=True)
            elif self._is_echo_dup_empty():
                # Echo-dup + repair that bai: hien bao that GAN, giu panel
                # thinking mo de user xem trace, KHONG luu rac vao history.
                self.post_chat(Text(
                    "  (model chi tra thinking trace, khong co dap an — "
                    "thu /.thinking off roi hoi lai)",
                    style=f"bold {C['warning']}",
                ))
                if self._think_panel is not None:
                    self._think_panel.expand()
                chat.scroll_end(animate=False, force=True)
            # Thu gon panel hoat dong sau khi luot hoan tat
            if self._activity is not None:
                self._activity.set_title(f"activity · {self._activity.steps} steps", C["muted"])
                self._activity.collapse()
            # Chot panel thinking neu co reasoning rieng kenh (tag <think>
            # tu xu ly rieng o tren; day la cho kenh reasoning_content).
            if self._think_buffer and self._think_panel is not None:
                self._think_panel.set_stream(self._think_buffer.strip(), cursor="")
                self._think_panel.set_title("thinking · done", C["success"])
                self._think_panel.collapse()
            # Luu lich su theo session de /.history hoat dong ca sau khi dong app
            if self._question and full:
                self._sessions, self._current_session_id = add_exchange_to_session(
                    self._sessions, self._question, full, self._current_session_id
                )
                self._ctx.sessions = self._sessions
                self._ctx.current_session_id = self._current_session_id
                _save_history_file(self._sessions)
            self.set_agent_status("Judge", "done", "")
            for name in self._agent_state:
                self._agent_state[name] = ("done", "")
            self._render_footer()
            self._finish_ui_turn()
        elif event == "error":
            self.post_chat(Text(f"  ✗ {data}", style=f"bold {C['error']}"))
            if self._think_panel is not None:
                self._think_panel.set_title("thinking · stopped", C["error"])
                self._think_panel.expand()
            if self._activity is not None:
                self._activity.set_title(f"activity · {self._activity.steps} steps — error", C["error"])
                self._activity.expand()  # mo panel de xay chi tiet loi
            self._busy = False
            question = self._question  # giu lai truoc khi xoa de luu history loi
            self._question = None
            self.query_one("#prompt", Input).disabled = False
            self.query_one("#prompt", Input).focus()
            self._render_footer()
            # Luon luu lich su khi gap loi — ke ca khi loi xay ra truoc khi model stream
            if question:
                err_ans = f"(error) {data}" if data else f"(error) {self._final_answer()}"
                self._sessions, self._current_session_id = add_exchange_to_session(
                    self._sessions, question, err_ans, self._current_session_id
                )
                self._ctx.sessions = self._sessions
                self._ctx.current_session_id = self._current_session_id
                _save_history_file(self._sessions)
            self._finish_ui_turn()

    # ------------------------------------------------------------ cli entry
    def exit(self, result=None, message: str | None = None, **kwargs) -> None:
        """Luu lich su truoc khi thoat — ke ca khi thoai giua luot dang chay."""
        if getattr(self, "_sessions", None):
            _save_history_file(self._sessions)
        try:  # dong MCP servers, tranh tien trinh npx zombie
            mcp = getattr(self, "_mcp", None)
            if mcp is not None:
                mcp.close()
        except Exception:  # noqa: BLE001
            pass
        super().exit(result=result, message=message, **kwargs)

    @classmethod
    def make_and_run(cls, config: Config) -> None:
        client = LLMClient(config)
        orchestrator = SwarmOrchestrator(
            client,
            num_workers=int(config.get("num_workers")),
        )
        # __init__ tu gan tool stack (builtin + MCP) vao orchestrator
        app = cls(config, client, orchestrator)
        app.run()
