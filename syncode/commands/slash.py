"""Cac lenh tien to `/.` khi dang chay (slash commands)."""

from __future__ import annotations

import shlex
from typing import Callable, Dict, List, Tuple

from syncode import __version__
from syncode.config import DEFAULTS, Config
from syncode.history import (
    COMPACT_KEEP_DEFAULT,
    COMPACT_MIN_SUMMARIZE,
    COMPACT_SYSTEM,
    apply_compact,
    build_compact_prompt,
    save_sessions,
)
from syncode.llm.client import LLMClient
from syncode.swarm.orchestrator import SwarmOrchestrator
from syncode.ui.console import UI


class SlashCommandContext:
    """Bundle truyen vao moi handler."""

    def __init__(self, config: Config, ui: UI, client: LLMClient, orchestrator: SwarmOrchestrator, history: List[Tuple[str, str]]) -> None:
        self.config = config
        self.ui = ui
        self.client = client
        self.orchestrator = orchestrator
        self.history = history          # List[Tuple[str, str]] — tu truoc
        self.sessions: list = []        # List[Session] — session-based history
        self.current_session_id: str | None = None
        self.tools = None          # ToolRegistry (neu co)
        self.mcp = None            # MCPManager (neu co)
        self.mode = "build"        # plan | build
        self.last_changes = None   # ChangeTracker cua luot gan nhat (cho /.undo)


Handler = Callable[[SlashCommandContext, List[str]], bool]


def _cmd_help(ctx: SlashCommandContext, args: List[str]) -> bool:
    rows = {f"/. {name} {cmd[1]}": cmd[2] for name, cmd in COMMANDS.items() if name != "help"}
    ctx.ui.key_value_table(f"Syncode v{__version__} — commands", rows)
    return False


def _cmd_config(ctx: SlashCommandContext, args: List[str]) -> bool:
    if not args or args[0] == "show":
        ctx.ui.key_value_table("Config (~/.syncode/config.json)", ctx.config.as_display())
    elif args[0] == "set" and len(args) >= 3:
        key, value = args[1], " ".join(args[2:])
        try:
            saved = ctx.config.set(key, value)
        except (ValueError, TypeError) as exc:
            ctx.ui.error(f"Invalid value for {key}: {exc}")
            return False
        if saved:
            if key in ("api_key", "base_url"):
                ctx.client.reset()
            ctx.ui.success(f"saved {key} = {ctx.config.as_display().get(key, value)}")
        else:
            ctx.ui.error(f"Invalid key. Valid keys: {', '.join(DEFAULTS)}")
    elif args[0] == "reset":
        ctx.config.reset()
        ctx.client.reset()
        ctx.ui.success("Reset to defaults.")
    else:
        ctx.ui.warn("Usage: /.config [show] | /.config set <key> <value> | /.config reset")
    return False


ROLE_MODEL_KEYS = {
    "planner": "planner_model",
    "worker": "worker_model",
    "workers": "worker_model",
    "w1": "worker1_model",
    "worker1": "worker1_model",
    "w2": "worker2_model",
    "worker2": "worker2_model",
    "critic": "critic_model",
    "refiner": "refiner_model",
    "judge": "judge_model",
    "synthesizer": "synthesizer_model",  # alias cu, van doc lam fallback cho judge
}


def _cmd_model(ctx: SlashCommandContext, args: List[str]) -> bool:
    if not args:
        ctx.ui.info(f"Main model: [magenta]{ctx.config.get('model')}[/]")
        for line in ctx.orchestrator.models_summary():
            ctx.ui.info(line)
        ctx.ui.info(
            "Set: /.model <name>  |  per-role: "
            "/.model <planner|w1|w2|worker|critic|refiner|judge> <name> "
            "('main' resets to the main model)"
        )
        return False
    role_key = ROLE_MODEL_KEYS.get(args[0].lower())
    if role_key is not None:
        if len(args) < 2:
            current = ctx.config.get(role_key)
            ctx.ui.info(
                f"Model {args[0].lower()}: [magenta]{current or ctx.config.get('model') + ' (main)'}[/]"
            )
            return False
        value = " ".join(args[1:])
        if value.lower() in ("main", "-", "reset"):
            ctx.config.set(role_key, "")
            ctx.ui.success(f"Role {args[0].lower()} reset to the main model.")
        else:
            ctx.config.set(role_key, value)
            ctx.ui.success(f"Role {args[0].lower()} uses model: [magenta]{value}[/]")
        return False
    ctx.config.set("model", " ".join(args))
    ctx.ui.success(f"Main model: [magenta]{ctx.config.get('model')}[/]")
    return False


def _cmd_key(ctx: SlashCommandContext, args: List[str]) -> bool:
    if not args:
        ctx.ui.warn("Usage: /.key <NVIDIA_API_KEY>  (get one at https://build.nvidia.com)")
    else:
        ctx.config.set("api_key", args[0])
        ctx.client.reset()
        ctx.ui.success("API key saved.")
    return False


def _cmd_history(ctx: SlashCommandContext, args: List[str]) -> bool:
    sessions = ctx.sessions if ctx.sessions else []
    if not sessions:
        ctx.ui.info("No history yet.")
        return False
    for i, session in enumerate(reversed(sessions), start=1):
        ctx.ui.info(f"[bold]{i}.[/] {session.title} — {session.summary}")
    return False


def _cmd_clear(ctx: SlashCommandContext, args: List[str]) -> bool:
    ctx.ui.clear()
    return False


def _compact_one_session(ctx: SlashCommandContext, sess, keep: int):
    """Compact 1 session: tra (ok, thong diep). That bai -> (False, ly do),
    session giu nguyen. Dung chung cho /.compact va /.compact --all."""
    exchanges = list(getattr(sess, "exchanges", None) or [])
    if len(exchanges) <= keep + COMPACT_MIN_SUMMARIZE - 1:
        return False, f"moi co {len(exchanges)} turns — chua can"
    older = [
        (getattr(e, "question", ""), getattr(e, "answer", ""))
        for e in exchanges[: len(exchanges) - keep]
    ]
    old_summary = getattr(sess, "compact_summary", "") or ""
    prompt = build_compact_prompt(older, old_summary)
    try:
        summary = ctx.client.chat(
            [
                {"role": "system", "content": COMPACT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"that bai (giu nguyen): {exc}"
    summary = (summary or "").strip()
    if not summary:
        return False, "that bai: model tra ve rong (giu nguyen)"
    stats = apply_compact(sess, summary, keep)
    title = getattr(sess, "title", sess.id)
    return True, (
        f"{stats['dropped']} turns -> tom tat {len(summary)} chars, "
        f"giu {stats['kept']} turns gan nhat "
        f"(giai phong ~{stats['freed_chars']} chars) [{title}]"
    )


def _cmd_compact(ctx: SlashCommandContext, args: List[str]) -> bool:
    """Nen session hien tai de nhe context (/.compact, khong tham so).

    Giu 2 luot gan nhat, cac luot con lai gom thanh tom tat 1 lan bang LLM.
    That bai o bat cu buoc nao -> session giu nguyen.
    """
    if args:
        ctx.ui.error("Usage: /.compact (khong tham so)")
        return False
    sessions = ctx.sessions if ctx.sessions else []
    sess = next((s for s in sessions if s.id == ctx.current_session_id), None)
    if sess is None or not getattr(sess, "exchanges", None):
        ctx.ui.warn("No active session yet — ask something first.")
        return False
    ok, msg = _compact_one_session(ctx, sess, COMPACT_KEEP_DEFAULT)
    if not ok:
        if "chua can" in msg:
            ctx.ui.info(f"Session {msg}.")
        else:
            ctx.ui.error(f"Compact {msg}.")
        return False
    save_sessions(sessions)
    ctx.ui.success(f"Compacted: {msg}.")
    return False


def _cmd_save(ctx: SlashCommandContext, args: List[str]) -> bool:
    import time
    from pathlib import Path

    sessions = ctx.sessions if ctx.sessions else []
    if not sessions:
        ctx.ui.warn("Nothing to save yet.")
        return False
    path = Path(args[0]) if args else Path("syncode_history.md")
    lines = ["# Syncode history\n"]
    for session in sessions:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(session.created_at))
        lines.append(f"## Session ({ts}) — {len(session.exchanges)} turns\n")
        for exchange in session.exchanges:
            lines += [f"### ❯ {exchange.question}\n", exchange.answer, "\n"]
    path.write_text("\n".join(lines), encoding="utf-8")
    ctx.ui.success(f"Saved {len(sessions)} sessions to {path}")
    return False


def _cmd_exit(ctx: SlashCommandContext, args: List[str]) -> bool:
    ctx.ui.info("bye")
    return True  # True = thoat chuong trinh


def _cmd_tools(ctx: SlashCommandContext, args: List[str]) -> bool:
    if ctx.tools is None:
        ctx.ui.warn("Tools are available in the TUI only.")
        return False
    ctx.ui.list_rows("Tools", ctx.tools.summary(), style="cyan")
    return False


def _cmd_mcp(ctx: SlashCommandContext, args: List[str]) -> bool:
    if ctx.mcp is None:
        ctx.ui.warn("MCP is not configured.")
        return False
    ok, errs = ctx.mcp.connect_all()
    if ok:
        names = ctx.mcp.register_into(ctx.tools) if ctx.tools else []
        if names:
            ctx.ui.success(f"Loaded {len(names)} MCP tools: {', '.join(names)}")
    for err in errs:
        ctx.ui.error(f"MCP {err}")
    for line in ctx.mcp.status_lines():
        ctx.ui.info(line)
    return False


def _cmd_mode(ctx: SlashCommandContext, args: List[str]) -> bool:
    # TUI co mode that nam tren app (qua adapter); CLI plain dung ctx.mode.
    get_mode = getattr(ctx.ui, "get_plan_mode", None)
    cur = get_mode() if callable(get_mode) else ctx.mode
    if not args:
        ctx.ui.info(f"Current mode: [bold]{cur}[/] (tab to switch)")
        ctx.ui.info(
            "plan = read-only tools (no writes, no shell)"
        )
        ctx.ui.info("build = full tools: write files, run commands")
    elif args[0].lower() in ("plan", "build"):
        setter = getattr(ctx.ui, "set_plan_mode", None)
        if callable(setter):
            ok = setter(args[0].lower())
            if not ok:
                ctx.ui.warn("Usage: /.mode [plan|build]")
                return False
        else:
            ctx.mode = args[0].lower()
        ctx.ui.success(f"Switched to [bold]{args[0].lower()}[/] mode")
    else:
        ctx.ui.warn("Usage: /.mode [plan|build]")
    return False


def _cmd_swarm(ctx: SlashCommandContext, args: List[str]) -> bool:
    """Chuyen doi Swarm/Quick mode. Swarm = chay day du pipeline,
    Quick = 1 luot chat truc tiep. Mac dinh la Swarm."""
    ctx.ui.toggle_swarm_mode()
    return False


def _cmd_thinking(ctx: SlashCommandContext, args: List[str]) -> bool:
    """Xem/doi che do thinking rieng cua model (off/low/auto).

    off = tat reasoning (dap an gon, re hon); low = reasoning nhe;
    auto = de model tu quyet dinh (mac dinh).
    """
    if not args:
        ctx.ui.info(
            f"Thinking mode: [bold]{ctx.config.get('thinking', 'auto')}[/] "
            "(off = concise, low = light reasoning, auto = model default)"
        )
        return False
    mode = args[0].lower()
    if mode not in ("auto", "off", "low"):
        ctx.ui.warn("Usage: /.thinking [off|low|auto]")
        return False
    ctx.config.set("thinking", mode)
    ctx.ui.success(f"Thinking mode: [bold]{mode}[/]")
    return False


def _cmd_debug(ctx: SlashCommandContext, args: List[str]) -> bool:
    """Rua goi LLM gan nhat: model, thinking, extra_body, reasoning, loc."""
    info = getattr(ctx.client, "last_call", None) or {}
    if not info:
        ctx.ui.info("No LLM call yet.")
        return False
    extra = info.get("extra_body")
    rows = {
        "model": info.get("model", "?"),
        "thinking mode": info.get("thinking", "?"),
        "extra_body sent": str(extra) if extra else "-",
        "reasoning seen": "yes" if info.get("reasoning_seen") else "no",
        "reasoning chars": info.get("reasoning_chars", 0),
        "answer chars": info.get("answer_chars", 0),
    }
    if info.get("answer_was_echo"):
        rows["echo dup"] = f"yes ({info.get('echo_dropped_chars', 0)} chars dropped)"
    cleaned = getattr(getattr(ctx, "orchestrator", None), "last_clean", None) or {}
    if cleaned:
        rows["answer raw chars"] = cleaned.get("raw_chars", 0)
        rows["answer final chars"] = cleaned.get("final_chars", 0)
        if cleaned.get("echo_repair"):
            rows["echo repair"] = "yes"
    ctx.ui.key_value_table("Last LLM call", rows)
    return False


def _cmd_usage(ctx: SlashCommandContext, args: List[str]) -> bool:
    """Hien thi token da dung trong process hien tai."""
    u = getattr(ctx.client, "usage", None) or {}
    last = getattr(ctx.client, "last_usage", None)
    rows = {
        "prompt tokens": u.get("prompt_tokens", 0),
        "completion tokens": u.get("completion_tokens", 0),
        "total tokens": u.get("total_tokens", 0),
        "measured calls": u.get("measured_calls", 0),
        "streamed calls (tokens n/a)": u.get("streamed_calls", 0),
    }
    if last:
        rows["last call"] = (
            f"{last.get('prompt_tokens', 0)}p + "
            f"{last.get('completion_tokens', 0)}c = "
            f"{last.get('total_tokens', 0)}"
        )
    ctx.ui.key_value_table("Token usage (this process)", rows)
    return False


def _cmd_undo(ctx: SlashCommandContext, args: List[str]) -> bool:
    """Hoan tac thay doi file cua luot gan nhat (ghi de/sua/xoa -> tra lai)."""
    tracker = getattr(ctx, "last_changes", None)
    files = getattr(tracker, "files", None) if tracker is not None else None
    if not files:
        ctx.ui.warn("Nothing to undo (no file changes in the last turn).")
        return False
    notes = tracker.undo_all()
    ctx.ui.list_rows("Undo", notes)
    failed = sum(1 for n in notes if n.startswith("FAILED"))
    if failed:
        ctx.ui.warn(f"{failed} change(s) could not be undone.")
    else:
        ctx.ui.success(f"Undid {len(notes)} change(s).")
    return False


def _cmd_init(ctx: SlashCommandContext, args: List[str]) -> bool:
    """Tao file AGENTS.md mau (neu chua co) de luu quy uoc repo cho agent."""
    from pathlib import Path

    from syncode.conventions import AGENTS_TEMPLATE

    target = Path(args[0]) if args else Path("AGENTS.md")
    if target.exists():
        ctx.ui.warn(f"{target} already exists — edit it directly.")
        return False
    try:
        target.write_text(AGENTS_TEMPLATE, encoding="utf-8")
    except OSError as exc:
        ctx.ui.error(f"Cannot write {target}: {exc}")
        return False
    ctx.ui.success(f"Created {target} — edit it with your repo conventions.")
    return False


def _cmd_todo(ctx: SlashCommandContext, args: List[str]) -> bool:
    """Checklist theo session: /.todo | add <text> | done <n> | clear."""
    from syncode.history import open_todos

    sessions = ctx.sessions or []
    sess = next((s for s in sessions if s.id == ctx.current_session_id), None)
    if not args:
        items = open_todos(sessions, ctx.current_session_id)
        if not items:
            ctx.ui.info("No open todos. Add one: /.todo add <text>")
        else:
            ctx.ui.list_rows(
                "Todos (open)",
                [t.get("text", "") for t in items],
            )
        return False
    if sess is None:
        ctx.ui.warn("No active session yet — ask something first.")
        return False
    sub = args[0].lower()
    if sub == "add" and len(args) >= 2:
        text = " ".join(args[1:]).strip()
        if not text:
            ctx.ui.warn("Usage: /.todo add <text>")
        else:
            sess.todos.append({"text": text, "done": False})
            _save_todo_sessions(ctx)
            ctx.ui.success(f"Added todo #{len(open_todos(sessions, sess.id))}: {text}")
    elif sub == "done" and len(args) >= 2:
        items = open_todos(sessions, sess.id)
        try:
            idx = int(args[1]) - 1
            target = items[idx] if 0 <= idx < len(items) else None
        except (TypeError, ValueError):
            target = None
        if target is None:
            ctx.ui.warn(f"Usage: /.todo done <1..{len(items)}>")
        else:
            target["done"] = True
            _save_todo_sessions(ctx)
            ctx.ui.success(f"Done: {target.get('text', '')}")
    elif sub == "clear":
        before = len(sess.todos)
        sess.todos = [t for t in sess.todos if not t.get("done")]
        _save_todo_sessions(ctx)
        ctx.ui.success(f"Cleared {before - len(sess.todos)} done todo(s).")
    else:
        ctx.ui.warn("Usage: /.todo [add <text> | done <n> | clear]")
    return False


def _save_todo_sessions(ctx: SlashCommandContext) -> None:
    try:
        save_sessions(ctx.sessions or [])
    except Exception:  # noqa: BLE001
        pass


def _cmd_skills(ctx: SlashCommandContext, args: List[str]) -> bool:
    """Liet ke skills markdown dang nap (.skills/ + ~/.syncode/skills/)."""
    from syncode.conventions import find_skill_files

    files = find_skill_files()
    if not files:
        ctx.ui.info("No skills. Add markdown files to .skills/ or ~/.syncode/skills/.")
        return False
    by_name: Dict[str, list] = {}
    for p in files:
        by_name.setdefault(p.name, []).append(p)
    rows = []
    for name, ps in by_name.items():
        # ten trung nhau moi hien full path de phan biet
        rows.extend([name] if len(ps) == 1 else [str(p) for p in ps])
    ctx.ui.list_rows("Skills", rows)
    return False


COMMANDS: Dict[str, Tuple[Handler, str, str]] = {
    "help": (_cmd_help, "", "Show commands"),
    "config": (_cmd_config, "[show|set|reset]", "Show/edit config (/.config set <key> <value>)"),
    "model": (_cmd_model, "[<role> <model>|<model>]", "Show/change model (per-role models supported)"),
    "key": (_cmd_key, "<NVIDIA_API_KEY>", "Set the API key"),
    "tools": (_cmd_tools, "", "List tools (builtin + MCP)"),
    "skills": (_cmd_skills, "", "List loaded skills"),
    "mcp": (_cmd_mcp, "", "Connect/show MCP servers"),
    "mode": (_cmd_mode, "[plan|build]", "Show/switch mode (plan = read-only)"),
    "swarm": (_cmd_swarm, "", "Toggle swarm/quick mode"),
    "thinking": (_cmd_thinking, "[off|low|auto]", "Model thinking mode"),
    "debug": (_cmd_debug, "", "Show last LLM call diagnostics"),
    "usage": (_cmd_usage, "", "Show token usage"),
    "undo": (_cmd_undo, "", "Undo file changes from the last turn"),
    "init": (_cmd_init, "[<file>]", "Create AGENTS.md template"),
    "todo": (_cmd_todo, "[add <text>|done <n>|clear]", "Session checklist"),
    "history": (_cmd_history, "", "Show history"),
    "save": (_cmd_save, "[<file>]", "Save history to a markdown file"),
    "compact": (_cmd_compact, "", "Summarize + trim session to free context"),
    "clear": (_cmd_clear, "", "Clear the screen"),
    "exit": (_cmd_exit, "", "Quit (alias: /.quit)"),
}
COMMANDS["quit"] = COMMANDS["exit"]


def is_slash_command(text: str) -> bool:
    return text.strip().startswith("/.")


def parse_slash(text: str) -> Tuple[str, List[str]]:
    parts = shlex.split(text.strip(), posix=True)
    if not parts:
        return "", []
    return parts[0][2:].lower(), parts[1:]


def execute(text: str, ctx: SlashCommandContext) -> bool:
    """Thuc thi lenh /. Tra ve True neu can thoat chuong trinh."""
    name, args = parse_slash(text)
    entry = COMMANDS.get(name)
    if entry is None:
        ctx.ui.error(
            f"Unknown command: /.{name} (lenh khong ton tai). Run /.help for the list."
        )
        return False
    return entry[0](ctx, args)
