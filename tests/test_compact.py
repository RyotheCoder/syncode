"""Kiem thu /.compact: tom tat + tia session de nhe context."""

from syncode.commands.slash import COMMANDS, execute, SlashCommandContext
from syncode.config import Config
from syncode.history import (
    COMPACT_SUMMARY_QUESTION,
    apply_compact,
    build_compact_prompt,
    compacted_history,
    Exchange,
    load_sessions,
    recent_exchanges,
    Session,
)


def _sess(n=5, summary=""):
    return Session(
        id="s1",
        created_at=1.0,
        exchanges=[Exchange(question=f"q{i}", answer=f"a{i}") for i in range(n)],
        compact_summary=summary,
    )


def _ctx(tmp_path, monkeypatch, client):
    monkeypatch.setenv("SYNCODE_HOME", str(tmp_path))
    import syncode.history as hist_mod

    monkeypatch.setattr(hist_mod, "CONFIG_DIR", tmp_path)
    config = Config()
    from syncode.llm.client import LLMClient
    from syncode.swarm.orchestrator import SwarmOrchestrator
    from syncode.ui.console import UI

    real = LLMClient(config)
    orch = SwarmOrchestrator(real, num_workers=1)
    ui = UI()
    ctx = SlashCommandContext(config, ui, client, orch, [])
    ctx.sessions = [_sess()]
    ctx.current_session_id = "s1"
    return ctx


class _SumClient:
    """Gia lap LLM tom tat: tra chuoi co dinh, ghi nhan prompt."""

    def __init__(self, text="TOM TAT: lam X, sua Y.", fail=False):
        self.text = text
        self.fail = fail
        self.prompts = []

    def chat(self, messages, temperature=None, **kwargs):
        self.prompts.append(messages)
        if self.fail:
            raise RuntimeError("mang loi")
        return self.text


def test_compact_summary_field_roundtrip(tmp_path, monkeypatch):
    import json
    import syncode.history as hist_mod
    from syncode.history import save_sessions

    monkeypatch.setattr(hist_mod, "CONFIG_DIR", tmp_path)
    save_sessions([_sess(summary="nho X")])
    back = load_sessions()
    assert back[0].compact_summary == "nho X"
    # file cu thieu key -> "" (khong vo)
    p = tmp_path / "history.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    del raw["sessions"][0]["compact_summary"]
    p.write_text(json.dumps(raw), encoding="utf-8")
    assert load_sessions()[0].compact_summary == ""


def test_build_compact_prompt_truncates_and_merges():
    long_a = "x" * 2000
    prompt = build_compact_prompt([("q0", long_a)], old_summary="cu: nho Z")
    assert "cu: nho Z" in prompt
    assert "q0" in prompt
    assert "x" * 2000 not in prompt  # dap an dai bi cat
    assert "cắt ngắn" in prompt


def test_apply_compact_trims_and_stats():
    s = _sess(n=5)
    stats = apply_compact(s, "TOM TAT", keep=2)
    assert stats == {"dropped": 3, "kept": 2, "freed_chars": len("q0a0q1a1q2a2")}
    assert [e.question for e in s.exchanges] == ["q3", "q4"]
    assert s.compact_summary == "TOM TAT"


def test_apply_compact_caps_runaway_summary():
    s = _sess(n=5)
    apply_compact(s, "z" * 5000, keep=2)
    assert len(s.compact_summary) <= 2001


def test_compacted_history_without_summary_equals_recent():
    sessions = [_sess()]
    assert compacted_history(sessions, "s1") == recent_exchanges(sessions, "s1")


def test_compacted_history_prepends_summary():
    sessions = [_sess(summary="nho X")]
    hist = compacted_history(sessions, "s1")
    assert hist[0] == (COMPACT_SUMMARY_QUESTION, "nho X")
    assert hist[1:] == recent_exchanges(sessions, "s1")


def test_compact_command_registered():
    assert "compact" in COMMANDS


def test_cmd_compact_happy_path(tmp_path, monkeypatch, capsys):
    client = _SumClient()
    ctx = _ctx(tmp_path, monkeypatch, client)
    assert execute("/.compact", ctx) is False
    out = capsys.readouterr().out.lower()
    assert "compacted" in out
    sess = ctx.sessions[0]
    assert sess.compact_summary == "TOM TAT: lam X, sua Y."
    assert [e.question for e in sess.exchanges] == ["q3", "q4"]
    # da luu file (nap lai van con)
    assert load_sessions()[0].compact_summary.startswith("TOM TAT")
    # system prompt dung COMPACT (khong phai prompt swarm)
    system = client.prompts[0][0]["content"]
    assert "tóm tắt phiên trò chuyện" in system.lower()


def test_cmd_compact_merges_old_summary(tmp_path, monkeypatch, capsys):
    client = _SumClient()
    ctx = _ctx(tmp_path, monkeypatch, client)
    ctx.sessions[0].compact_summary = "cu: nho Z"
    # can du luot de compact (5 turns, keep 2 -> 3 cu)
    assert execute("/.compact", ctx) is False
    capsys.readouterr()
    user_prompt = client.prompts[0][1]["content"]
    assert "cu: nho Z" in user_prompt


def test_cmd_compact_no_session_warns(tmp_path, monkeypatch, capsys):
    client = _SumClient()
    ctx = _ctx(tmp_path, monkeypatch, client)
    ctx.sessions = []
    ctx.current_session_id = None
    assert execute("/.compact", ctx) is False
    assert "no active session" in capsys.readouterr().out.lower()


def test_cmd_compact_too_short_informs(tmp_path, monkeypatch, capsys):
    client = _SumClient()
    ctx = _ctx(tmp_path, monkeypatch, client)
    ctx.sessions = [_sess(n=2)]
    ctx.current_session_id = "s1"
    assert execute("/.compact", ctx) is False
    assert "chua can" in capsys.readouterr().out.lower()
    assert client.prompts == []  # khong ton call LLM


def test_cmd_compact_failure_keeps_session(tmp_path, monkeypatch, capsys):
    client = _SumClient(fail=True)
    ctx = _ctx(tmp_path, monkeypatch, client)
    before = [(e.question, e.answer) for e in ctx.sessions[0].exchanges]
    assert execute("/.compact", ctx) is False
    assert "that bai" in capsys.readouterr().out.lower()
    assert [(e.question, e.answer) for e in ctx.sessions[0].exchanges] == before
    assert ctx.sessions[0].compact_summary == ""


def test_cmd_compact_empty_summary_keeps_session(tmp_path, monkeypatch, capsys):
    client = _SumClient(text="   ")
    ctx = _ctx(tmp_path, monkeypatch, client)
    assert execute("/.compact", ctx) is False
    assert "rong" in capsys.readouterr().out.lower()
    assert len(ctx.sessions[0].exchanges) == 5


def test_cmd_compact_bad_args_usage(tmp_path, monkeypatch, capsys):
    client = _SumClient()
    ctx = _ctx(tmp_path, monkeypatch, client)
    assert execute("/.compact xyz", ctx) is False
    assert "usage" in capsys.readouterr().out.lower()
    assert client.prompts == []  # co tham so -> bao loi, khong goi LLM
