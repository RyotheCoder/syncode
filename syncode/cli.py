"""Syncode CLI - entry point va REPL chinh."""

from __future__ import annotations

import argparse
import sys
from typing import List

from syncode import __version__
from syncode.commands.slash import SlashCommandContext, execute, is_slash_command
from syncode.config import Config
from syncode.history import (
    Session,
    add_exchange_to_session,
    compacted_history,
    load_sessions,
    open_todos,
    save_sessions,
)
from syncode.llm.client import LLMClient, LLMError
from syncode.swarm.orchestrator import SubTask, SwarmOrchestrator
from syncode.tools import ChangeTracker, build_tool_stack
from syncode.ui.console import UI

ONBOARDING = (
    "[bold cyan]First-time setup:[/]\n"
    "  1. Get a free API key at [link=https://build.nvidia.com]https://build.nvidia.com[/]\n"
    "  2. Run: [bold]/.key <NVIDIA_API_KEY>[/]  (or export NVIDIA_API_KEY=...)\n"
    "  3. Switch model anytime: [bold]/.model mistralai/mistral-7b-instruct-v0.3[/]"
)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="syncode",
        description="Syncode - CLI Swarm Agent (Planner > Workers > Critic > Judge) tren NVIDIA NIM.",
    )
    parser.add_argument("--api-key", help="NVIDIA NIM API key (hoac dung env NVIDIA_API_KEY)")
    parser.add_argument("--model", help="Model NIM, vd: nvidia/nemotron-3.5-lightning-30b-a3b")
    parser.add_argument("--base-url", help="Endpoint OpenAI-compatible (mac dinh: NVIDIA NIM)")
    parser.add_argument("--task", help="Chay 1 yeu cau roi thoat (one-shot)")
    parser.add_argument(
        "-p", "--print", action="store_true",
        help="Non-interactive: chay 1 luot (tu --task/stdin), in dap an text ra stdout",
    )
    parser.add_argument(
        "--plain", action="store_true",
        help="Dung giao dien dong lenh (REPL) thay vi full-screen TUI",
    )
    parser.add_argument("--version", action="version", version=f"syncode {__version__}")
    return parser.parse_args(argv)


def read_piped_stdin(max_chars: int = 20000) -> str:
    """Doc stdin neu duoc pipe (vd `cat err.log | syncode --task "fix"`).
    TTY (go truc tiep) -> ''. Khong bao gio block/raise."""
    try:
        if sys.stdin.isatty():
            return ""
        data = sys.stdin.read(max_chars + 1)
    except (OSError, ValueError):
        return ""
    data = (data or "").strip()
    if len(data) > max_chars:
        data = data[:max_chars] + "\n… (stdin truncated)"
    return data


def with_stdin_context(task: str, stdin_text: str) -> str:
    """Ghep du lieu pipe vao cau hoi. stdin rong -> tra task nguyen ven."""
    stdin_text = (stdin_text or "").strip()
    if not stdin_text:
        return task
    return f"{task}\n\n---\nDU LIEU TU STDIN (pipe):\n{stdin_text}"


def attach_ui_events(orchestrator: SwarmOrchestrator, ui: UI) -> None:
    """Noi event cua swarm voi UI (badge, ke hoach, phan hoi critic...)."""

    def on_event(event: str, data) -> None:
        if event == "phase":
            name, note = data
            agent = orchestrator.find_agent(name)
            if agent:
                ui.agent_badge(name, agent.icon, agent.style, note)
        elif event == "plan":
            subtasks: List[SubTask] = data
            ui.list_rows(
                "plan",
                [f"{s.id}: {s.title} — {s.goal}" for s in subtasks],
                style="yellow",
            )
        elif event == "worker_done":
            name, sid, output = data
            agent = orchestrator.find_agent(name)
            if agent and name == "Refiner":
                ui.agent_panel(name, agent.icon, agent.style, output, subtitle="(revised)")
        elif event == "critique":
            verdict = str(data.get("verdict", "?"))
            ui.info(f"Critic verdict: [bold]{verdict}[/]")
            for issue in data.get("issues", []):
                ui.warn(f"issue: {issue}")

    orchestrator.on_event = on_event


def run_swarm_task(orchestrator: SwarmOrchestrator, ui: UI, config: Config, question: str, quick_mode: bool = False, history=None, changes=None, todos=None) -> str:
    """Chay task cho 1 cau hoi; stream truc tiep ra panel markdown.

    - Mặc định (quick_mode=False): BẮT BUỘC full swarm mọi câu hỏi
      (Planner→Workers→Critic→Judge), KHÔNG giới hạn token gắt
      (giữ max_tokens cao để code đầy đủ).
    - Chỉ khi quick_mode=True (bật qua /.swarm) mới chạy 1 lượt quick.
    - history: [(q, a), ...] gan nhat de giu ngu canh hoi thoai.
    - changes: ChangeTracker caller cung cap (de /.undo); None = tu tao.
    """
    try:
        if changes is None:
            changes = ChangeTracker()
        prev_changes = orchestrator.changes
        orchestrator.changes = changes
        try:
            if quick_mode:
                answer = _run_quick_with_stream(orchestrator, ui, question, history)
                ui.agent_panel(
                    "Answer", "*", "bold green", answer,
                    subtitle="(quick)",
                )
                return answer
            result = _run_with_stream(orchestrator, ui, config, question, history, todos)
        finally:
            orchestrator.changes = prev_changes
        for row in changes.diff_summary():
            ui.info(f"changed: {row}")
        ui.agent_panel(
            "Answer", "*", "bold green", result.answer,
            subtitle=f"({result.rounds} revisions)" if result.rounds else "(approved)",
        )
        return result.answer
    except LLMError as exc:
        ui.error(f"LLM error: {exc}")
        return ""


def _run_quick_with_stream(orchestrator: SwarmOrchestrator, ui: UI, question: str, history=None) -> str:
    """Chay 1 lượt quick ở thread nền; stream live qua queue (giống swarm)."""
    import queue
    import threading

    q: queue.Queue = queue.Queue()
    holder: dict = {}
    base_event = orchestrator.on_event

    def on_delta(piece: str) -> None:
        q.put(piece)

    def worker() -> None:
        try:
            holder["answer"] = orchestrator.quick_reply(question, on_delta=on_delta, history=history)
        except LLMError as exc:
            holder["error"] = exc
        finally:
            q.put(None)  # tin hieu ket thuc stream

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    ui.stream_markdown(iter(q.get, None), "answer", "bold white")
    thread.join()
    orchestrator.on_event = base_event
    if "error" in holder:
        raise holder["error"]
    return holder.get("answer", "")


def _run_with_stream(orchestrator: SwarmOrchestrator, ui: UI, config: Config, question: str, history=None, todos=None):
    """Chay swarm o thread nen; stream phan Judge live qua queue."""
    import queue
    import threading

    q: queue.Queue = queue.Queue()
    holder: dict = {}
    base_event = orchestrator.on_event

    def on_event(event: str, data) -> None:
        if event == "synth_delta":
            q.put(data)
        elif base_event:
            base_event(event, data)

    def worker() -> None:
        try:
            holder["result"] = orchestrator.run(
                question, max_rounds=int(config.get("max_rounds")),
                history=history, todos=todos,
            )
        except LLMError as exc:
            holder["error"] = exc
        finally:
            q.put(None)  # tin hieu ket thuc stream

    orchestrator.on_event = on_event
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    ui.stream_markdown(iter(q.get, None), "answer", "bold white")
    thread.join()
    orchestrator.on_event = base_event
    if "error" in holder:
        raise holder["error"]
    return holder["result"]


def _cli_confirm(prompt_text: str) -> bool:
    """Hoi y/n tren stdin cho lenh nguy hiem (CLI plain). EOF/Ctrl+C = deny."""
    try:
        ans = input(f"{prompt_text} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("y", "yes")


def repl(config: Config, client: LLMClient, ui: UI) -> None:
    # Orchestrator tao ngay tu dau (khong can mang) de /.agents luon dung duoc
    tools, mcp = build_tool_stack(config)
    orchestrator = SwarmOrchestrator(
        client, num_workers=int(config.get("num_workers")), tools=tools
    )
    orchestrator.confirm_callback = _cli_confirm
    attach_ui_events(orchestrator, ui)
    # Lich su persistent: doc lai tu ~/.syncode/history.json de /.history hoat dong
    sessions: List[Session] = load_sessions()
    current_session_id: str | None = None
    ctx = SlashCommandContext(config, ui, client, orchestrator, [])
    ctx.tools = tools
    ctx.mcp = mcp

    if not config.has_api_key:
        ui.warn("No NVIDIA API key — you can still configure first.")
        ui.console.print(ONBOARDING)
    else:
        ui.info(
            f"ready — {config.get('num_workers')} workers · "
            f"model {config.get('model')} · {len(tools)} tools"
        )

    while True:
        try:
            text = ui.user_prompt().strip()
        except (EOFError, KeyboardInterrupt):
            save_sessions(sessions)  # luu truoc khi thoat
            try:
                mcp.close()
            except Exception:  # noqa: BLE001
                pass
            ui.info("\nbye")
            return
        if not text:
            continue
        if is_slash_command(text):
            if execute(text, ctx):
                save_sessions(sessions)  # luu truoc khi thoat
                try:
                    mcp.close()
                except Exception:  # noqa: BLE001
                    pass
                return
            # dong bo so worker neu nguoi doi config trong phien
            orchestrator.set_num_workers(int(config.get("num_workers")))
            # /.mode chi tac dung khi co tools (allow_dangerous)
            tools.set_allow_dangerous(ctx.mode == "build")
            tools.confirm_dangerous = bool(config.get("confirm_dangerous", True))
            continue
        if not config.has_api_key:
            ui.error("No API key. Run /.key <NVIDIA_API_KEY> first.")
            continue
        try:
            hist = compacted_history(sessions, current_session_id)
            todo_list = open_todos(sessions, current_session_id)
            tracker = ChangeTracker()
            answer = run_swarm_task(
                orchestrator, ui, config, text,
                quick_mode=getattr(ui, "quick_mode", False),
                history=hist,
                changes=tracker,
                todos=todo_list,
            )
            ctx.last_changes = tracker
            if answer:
                sessions, current_session_id = add_exchange_to_session(
                    sessions, text, answer, current_session_id
                )
                ctx.sessions = sessions
                ctx.current_session_id = current_session_id
                save_sessions(sessions)  # luu ngay sau moi luot thanh cong
        except KeyboardInterrupt:
            ui.warn("Cancelled.")


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    config = Config()
    if args.api_key:
        config.set("api_key", args.api_key)
    if args.model:
        config.set("model", args.model)
    if args.base_url:
        config.set("base_url", args.base_url)

    ui = UI()
    client = LLMClient(config)

    stdin_text = read_piped_stdin()

    if args.print:
        # Non-interactive: cau hoi tu --task va/hoac stdin, in dap an text thuần.
        task_text = (args.task or "").strip()
        if task_text and stdin_text:
            task_text = with_stdin_context(task_text, stdin_text)
        elif stdin_text:
            task_text = stdin_text
        if not task_text:
            print('Usage: syncode -p --task "..." (or pipe stdin)', file=sys.stderr)
            return 2
        if not config.has_api_key:
            print("Missing API key (--api-key or env NVIDIA_API_KEY).", file=sys.stderr)
            return 1
        tools, mcp = build_tool_stack(config)
        orchestrator = SwarmOrchestrator(
            client, num_workers=int(config.get("num_workers")), tools=tools
        )
        try:
            result = orchestrator.run(task_text, max_rounds=int(config.get("max_rounds")))
        except LLMError as exc:
            print(f"LLM error: {exc}", file=sys.stderr)
            return 1
        finally:
            try:
                mcp.close()
            except Exception:  # noqa: BLE001
                pass
        print(result.answer)
        return 0 if result.answer else 1

    if args.plain or args.task:
        if args.task and not config.has_api_key:
            ui.error("Missing API key (--api-key or env NVIDIA_API_KEY).")
            return 1
        ui.banner(__version__, config.get("model"))
        if args.task:
            if stdin_text:
                args.task = with_stdin_context(args.task, stdin_text)
            tools, mcp = build_tool_stack(config)
            orchestrator = SwarmOrchestrator(
                client, num_workers=int(config.get("num_workers")), tools=tools
            )
            attach_ui_events(orchestrator, ui)
            try:
                answer = run_swarm_task(orchestrator, ui, config, args.task)
            finally:
                try:
                    mcp.close()
                except Exception:  # noqa: BLE001
                    pass
            return 0 if answer else 1
        repl(config, client, ui)
        return 0

    # Mac dinh: full-screen TUI (giao dien desktop trong terminal)
    from syncode.ui.app import SyncodeApp

    SyncodeApp.make_and_run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
