from syncode.commands.slash import SlashCommandContext, execute, parse_slash
from syncode.config import Config
from syncode.swarm.orchestrator import SwarmOrchestrator, extract_json


def _ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNCODE_HOME", str(tmp_path))
    config = Config()
    from syncode.llm.client import LLMClient
    from syncode.ui.console import UI

    client = LLMClient(config)
    orchestrator = SwarmOrchestrator(client, num_workers=2)
    ui = UI()
    return SlashCommandContext(config, ui, client, orchestrator, [])


def test_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNCODE_HOME", str(tmp_path))
    config = Config()
    assert config.set("model", "test/model-1")
    config2 = Config()
    assert config2.get("model") == "test/model-1"


def test_config_rejects_unknown_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNCODE_HOME", str(tmp_path))
    config = Config()
    assert not config.set("khong_ton_tai", "x")


def test_config_masks_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNCODE_HOME", str(tmp_path))
    config = Config()
    config.set("api_key", "nvapi-abcdefgh12345678")
    shown = config.as_display()["api_key"]
    assert shown.startswith("nvapi-") and "abcdefgh12345678" not in shown


def test_parse_slash():
    name, args = parse_slash('/.config set model "mistralai/mistral-7b-instruct-v0.3"')
    assert name == "config" and args[0] == "set"


def test_build_tool_stack_has_builtins():
    from syncode.tools import ToolRegistry, build_tool_stack

    config = Config()
    tools, mcp = build_tool_stack(config)
    assert isinstance(tools, ToolRegistry)
    names = {t.name for t in tools.all_tools()}
    assert {"calculator", "read_file", "write_file", "list_dir",
            "run_command", "ask_user"} <= names
    assert mcp.sessions == {}  # mac dinh khong cau hinh server nao


def test_cli_run_swarm_task_with_tools():
    """CLI chay duoc tool-calling (calculator) qua ToolRegistry that."""
    from test_orchestrator import FakeClient, _role

    from syncode.cli import run_swarm_task
    from syncode.tools import ToolRegistry
    from syncode.ui.console import UI

    class CalcClient(FakeClient):
        def __init__(self):
            super().__init__()
            self._called = False

        def chat_with_tools(self, messages, schemas, temperature=None, max_tokens=None, model=None):
            self.calls.append({
                "role": _role(messages[0]["content"]),
                "user": messages[1]["content"],
                "model": model,
            })
            if not self._called:
                self._called = True
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "c1", "name": "calculator",
                        "arguments": {"expression": "2+3"},
                    }],
                }
            return {"content": "ket qua la 5", "tool_calls": []}

        def _reply(self, system, user):
            role = _role(system)
            if "PLANNER" in role:
                return '[{"id": "t1", "title": "Tinh", "goal": "tinh"}]'
            if "CRITIC" in role:
                return '{"verdict": "APPROVED"}'
            if "JUDGE" in role:
                return "CALC DONE"
            return super()._reply(system, user)

    class RecRegistry(ToolRegistry):
        def __init__(self):
            super().__init__()
            self.ran = []

        def execute(self, name, arguments, **kwargs):
            self.ran.append(name)
            return super().execute(name, arguments, **kwargs)

    reg = RecRegistry()
    orch = SwarmOrchestrator(CalcClient(), num_workers=1, tools=reg)
    answer = run_swarm_task(orch, UI(), Config(), "tinh 2+3")
    assert answer == "CALC DONE"
    assert "calculator" in reg.ran


def test_config_set_bad_value_no_crash(tmp_path, monkeypatch, capsys):
    """Gia tri sai kieu (vd temperature=abc) phai bao loi, khong vang traceback."""
    ctx = _ctx(tmp_path, monkeypatch)
    assert execute("/.config set temperature abc", ctx) is False
    assert "invalid value" in capsys.readouterr().out.lower()
    # config cu van giu nguyen
    assert ctx.config.get("temperature") == 0.7


def test_clean_mcp_args_strips_internals():
    """Arg noi bo registry (_changes/_ask/_emit) khong duoc gui sang MCP server."""
    from syncode.tools.mcp import _clean_mcp_args

    cleaned = _clean_mcp_args({
        "path": "/tmp",
        "max_lines": 10,
        "_changes": object(),
        "_ask": object(),
        "_emit": object(),
    })
    assert cleaned == {"path": "/tmp", "max_lines": 10}
    assert _clean_mcp_args(None) == {}


def test_execute_unknown_command(tmp_path, monkeypatch, capsys):
    ctx = _ctx(tmp_path, monkeypatch)
    assert execute("/.khongco", ctx) is False
    assert "khong ton tai" in capsys.readouterr().out.lower()


def test_usage_command_shows_totals(tmp_path, monkeypatch, capsys):
    ctx = _ctx(tmp_path, monkeypatch)
    ctx.client.usage["total_tokens"] = 123
    ctx.client.usage["measured_calls"] = 2
    assert execute("/.usage", ctx) is False
    out = capsys.readouterr().out.lower()
    assert "total tokens" in out and "123" in out


def test_undo_command_restores_file(tmp_path, monkeypatch, capsys):
    from syncode.tools import ChangeTracker

    ctx = _ctx(tmp_path, monkeypatch)
    f = tmp_path / "u.txt"
    f.write_text("orig", encoding="utf-8")
    tr = ChangeTracker()
    tr.record_file(str(f), "orig", "mod")
    f.write_text("mod", encoding="utf-8")
    ctx.last_changes = tr
    assert execute("/.undo", ctx) is False
    assert f.read_text(encoding="utf-8") == "orig"
    assert "reverted" in capsys.readouterr().out.lower()


def test_undo_command_empty_warns(tmp_path, monkeypatch, capsys):
    ctx = _ctx(tmp_path, monkeypatch)
    assert execute("/.undo", ctx) is False
    assert "nothing to undo" in capsys.readouterr().out.lower()


def test_thinking_command(tmp_path, monkeypatch, capsys):
    from syncode.config import Config as _Config

    monkeypatch.setattr(_Config, "save", lambda self: None)
    ctx = _ctx(tmp_path, monkeypatch)
    assert execute("/.thinking", ctx) is False
    assert "auto" in capsys.readouterr().out.lower()  # default la auto
    assert execute("/.thinking off", ctx) is False
    assert ctx.config.get("thinking") == "off"
    assert "off" in capsys.readouterr().out.lower()
    assert execute("/.thinking bogus", ctx) is False
    assert "usage" in capsys.readouterr().out.lower()
    assert ctx.config.get("thinking") == "off"


def test_todo_add_done_clear(tmp_path, monkeypatch, capsys):
    import time

    from syncode.history import Session

    ctx = _ctx(tmp_path, monkeypatch)
    # chua co session -> bao chua co phien
    assert execute("/.todo add viet test", ctx) is False
    assert "no active session" in capsys.readouterr().out.lower()
    # tao session roi lam viec
    sess = Session(id="s9", created_at=time.time(), exchanges=[])
    ctx.sessions = [sess]
    ctx.current_session_id = "s9"
    assert execute("/.todo add viet test", ctx) is False
    assert execute("/.todo add viet docs", ctx) is False
    assert len(sess.todos) == 2
    assert execute("/.todo", ctx) is False
    out = capsys.readouterr().out.lower()
    assert "viet test" in out and "viet docs" in out
    assert execute("/.todo done 1", ctx) is False
    assert sess.todos[0]["done"] is True
    assert execute("/.todo done 9", ctx) is False
    assert "usage" in capsys.readouterr().out.lower()
    assert execute("/.todo clear", ctx) is False
    assert len(sess.todos) == 1 and sess.todos[0]["text"] == "viet docs"


def test_execute_model_and_quit(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, monkeypatch)
    assert execute("/.model meta/test-model", ctx) is False
    assert ctx.config.get("model") == "meta/test-model"
    assert execute("/.exit", ctx) is True


def test_extract_json_variants():
    assert extract_json('[{"id": "t1", "title": "a", "goal": "b"}]') == [
        {"id": "t1", "title": "a", "goal": "b"}
    ]
    fenced = 'day la ket qua:\n```json\n{"verdict": "APPROVED"}\n```'
    assert extract_json(fenced) == {"verdict": "APPROVED"}
    noisy = 'Tôi trả về: {"verdict": "REVISE", "issues": ["x"]}'
    assert extract_json(noisy)["verdict"] == "REVISE"
    assert extract_json("khong co json") is None


def test_orchestrator_worker_scaling(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNCODE_HOME", str(tmp_path))
    from syncode.llm.client import LLMClient

    client = LLMClient(Config())
    orchestrator = SwarmOrchestrator(client, num_workers=2)
    assert len(orchestrator.workers) == 2
    orchestrator.set_num_workers(4)
    assert len(orchestrator.workers) == 4
    assert [a.name for a in orchestrator.agents()][0] == "Planner"


def test_skills_command_lists_files(tmp_path, monkeypatch, capsys):
    from syncode.commands.slash import execute

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("SYNCODE_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    ctx = _ctx(tmp_path, monkeypatch)
    assert execute("/.skills", ctx) is False
    assert "no skills" in capsys.readouterr().out.lower()
    (tmp_path / ".skills").mkdir()
    (tmp_path / ".skills" / "a.md").write_text("Skill A.", encoding="utf-8")
    assert execute("/.skills", ctx) is False
    assert "a.md" in capsys.readouterr().out


def test_debug_command(tmp_path, monkeypatch, capsys):
    ctx = _ctx(tmp_path, monkeypatch)
    assert execute("/.debug", ctx) is False
    assert "no llm call" in capsys.readouterr().out.lower()
    ctx.client.last_call = {
        "model": "m-test",
        "thinking": "off",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "reasoning_seen": True,
        "reasoning_chars": 500,
        "answer_chars": 42,
    }
    assert execute("/.debug", ctx) is False
    out = capsys.readouterr().out.lower()
    assert "m-test" in out and "reasoning seen" in out and "yes" in out
    assert "reasoning chars" in out and "500" in out


def test_debug_command_shows_echo_dup(tmp_path, monkeypatch, capsys):
    ctx = _ctx(tmp_path, monkeypatch)
    ctx.client.last_call = {
        "model": "m-test",
        "thinking": "auto",
        "extra_body": None,
        "reasoning_seen": True,
        "reasoning_chars": 7281,
        "answer_chars": 7281,
        "echo_dropped_chars": 7281,
        "answer_was_echo": True,
    }
    ctx.orchestrator.last_clean = {"raw_chars": 0, "final_chars": 0, "echo_repair": True}
    assert execute("/.debug", ctx) is False
    out = capsys.readouterr().out.lower()
    assert "echo dup" in out and "7281" in out
    assert "echo repair" in out
