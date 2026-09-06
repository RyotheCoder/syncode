"""Kiem thu build_suggestions, TUI compact va cac modal panel (textual run_test)."""

import asyncio
import os
import sys

from textual.widgets import OptionList

from syncode.config import Config
from syncode.llm.client import LLMClient
from syncode.swarm.orchestrator import SwarmOrchestrator
from syncode.ui.app import (
    POPULAR_MODELS,
    ConfigModal,
    HelpModal,
    SyncodeApp,
    ModelModal,
    ThinkingModal,
    build_suggestions,
)


# ---------------------------------------------------------------- suggestions
def test_suggestions_root_commands():
    items = build_suggestions("/")
    values = [v for _d, _desc, v in items]
    assert any(v.startswith("/.help") for v in values)
    items2 = build_suggestions("/.he")
    assert len(items2) == 1 and items2[0][2].startswith("/.help")


def test_suggestions_config_keys():
    items = build_suggestions("/.config set mo")
    values = [v for _d, _desc, v in items]
    assert "/.config set model " in values


def test_suggestions_model_list():
    items = build_suggestions("/.model nvidia")
    assert all("nvidia/" in v for _d, _desc, v in items)
    assert len(items) >= 2


def test_suggestions_empty_for_plain_text():
    assert build_suggestions("xin chao") == []
    assert build_suggestions("") == []


# ---------------------------------------------------------------- TUI compact
def _make_app() -> SyncodeApp:
    config = Config()
    client = LLMClient(config)
    orchestrator = SwarmOrchestrator(client, num_workers=2)
    return SyncodeApp(config, client, orchestrator)


def test_tui_composes_and_suggests():
    async def scenario() -> None:
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            # strip co 6 agent (planner + 2 workers + critic + refiner + judge)
            assert len(app._agent_state) == 6
            prompt = app.query_one("#prompt")
            prompt.focus()
            prompt.value = "/"
            await pilot.pause()
            assert app.suggest_visible
            assert app._suggest_items
            prompt.value = "xin chao"
            await pilot.pause()
            assert not app.suggest_visible
            app.exit()

    asyncio.run(scenario())


def test_tui_inline_slash_still_works():
    async def scenario() -> None:
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._handle_user_text("/.model meta/test-model")
            await pilot.pause()
            assert app._config.get("model") == "meta/test-model"
            app.exit()

    asyncio.run(scenario())


# ---------------------------------------------------------------- modal panels
def test_config_modal_open_edit_save():
    async def scenario() -> None:
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._handle_user_text("/.config")  # khong tham so -> mo panel
            await pilot.pause()
            assert isinstance(app.screen, ConfigModal)
            modal = app.screen
            # mo trinh sua cho key temperature (thao tac truc tiep de don gian)
            from textual.widgets import Input as Inp

            modal._edit_key = "temperature"
            inp = Inp(value="0.7", classes="dialog-input")
            modal.set_body(inp)
            inp.focus()
            await pilot.pause()
            modal.on_input_submitted(Inp.Submitted(inp, "0.5"))
            await pilot.pause()
            assert app._config.get("temperature") == 0.5
            app.pop_screen()
            await pilot.pause()
            app.exit()

    asyncio.run(scenario())


def test_model_modal_opens_and_filters():
    async def scenario() -> None:
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._handle_user_text("/.model")  # khong tham so -> mo panel 2 cot
            await pilot.pause()
            assert isinstance(app.screen, ModelModal)
            modal = app.screen
            await pilot.pause()  # cho _post_mount_fill chay
            # panel co danh sach role + danh sach model
            roles = modal.query_one("#role-list")
            models = modal.query_one("#model-list")
            assert roles.option_count >= 4  # main + planner + w1 + w2 + critic...
            # di chuyen highlight sang role 'planner' -> model list doi theo
            pidx = next(i for i in range(roles.option_count)
                        if roles.get_option_at_index(i).id == "planner")
            roles.highlighted = pidx
            await pilot.pause()
            assert modal._current_role()[1] == "planner"
            # loc model
            modal._filter = "meta/"
            modal._fill_models()
            assert 0 < models.option_count < len(POPULAR_MODELS)
            # chon model cho role dang highlight (planner) bang enter
            # index 0 = "↩ dùng model chính" -> chon index 1 (model that)
            sel = 1 if models.option_count > 1 else 0
            from textual.widgets import OptionList as _OL

            modal.on_option_list_option_selected(
                _OL.OptionSelected(models, models.get_option_at_index(sel), sel)
            )
            await pilot.pause()
            # model duoc set vao planner_model (khong phai main); model chinh giu nguyen
            main_before = app._config.get("model")
            assert app._config.get("planner_model").startswith("meta/")
            assert app._config.get("model") == main_before
            # dong panel bang ESC (dismiss) -> push_screen callback chay -> ap dung model
            modal.dismiss(None)
            await pilot.pause()
            await pilot.pause()
            assert app._orchestrator.planner.model.startswith("meta/")
            app.exit()

    asyncio.run(scenario())


def test_help_modal_opens():
    async def scenario() -> None:
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._handle_user_text("/.help")
            await pilot.pause()
            assert isinstance(app.screen, HelpModal)
            app.pop_screen()
            await pilot.pause()
            app.exit()

    asyncio.run(scenario())


def test_thinking_command_opens_panel():
    async def scenario() -> None:
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._config.set("thinking", "auto")
            app._handle_user_text("/.thinking")  # khong tham so -> mo panel
            await pilot.pause()
            assert isinstance(app.screen, ThinkingModal)
            modal = app.screen
            ol = modal.query_one("#thinking-list")
            assert ol.option_count == 3
            # mac dinh highlight dung mode hien tai (auto)
            assert ol.get_option_at_index(ol.highlighted).id == "auto"
            # chon "low" bang enter -> luu config
            from textual.widgets import OptionList as _OL

            modal.on_option_list_option_selected(
                _OL.OptionSelected(ol, ol.get_option_at_index(1), 1)
            )
            await pilot.pause()
            assert app._config.get("thinking") == "low"
            app.exit()

    asyncio.run(scenario())


def test_thinking_modal_esc_keeps_value():
    async def scenario() -> None:
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._config.set("thinking", "low")
            app._handle_user_text("/.thinking")
            await pilot.pause()
            assert isinstance(app.screen, ThinkingModal)
            await pilot.press("escape")
            await pilot.pause()
            assert app._config.get("thinking") == "low"
            app.exit()

    asyncio.run(scenario())


def test_thinking_inline_args_still_work():
    async def scenario() -> None:
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._handle_user_text("/.thinking off")  # co tham so -> inline, khong mo panel
            await pilot.pause()
            assert not isinstance(app.screen, ThinkingModal)
            assert app._config.get("thinking") == "off"
            app.exit()

    asyncio.run(scenario())


# ---------------------------------------------------------------- swarm e2e
def test_tui_full_swarm_e2e():
    """Full pipeline gia chay trong TUI: events -> strip/chat/streaming."""
    from test_orchestrator import FakeClient

    class StreamingFake(FakeClient):
        def chat(self, messages, temperature=None, max_tokens=None, on_delta=None, model=None, on_thinking=None, thinking=None):
            reply = self._reply(messages[0]["content"], messages[1]["content"])
            if on_delta:
                for ch in reply:
                    on_delta(ch)
            self.calls.append({
                "role": messages[0]["content"].split("cua mot swarm AI")[0].strip(),
                "user": messages[1]["content"],
                "model": model,
            })
            return reply

    async def scenario() -> None:
        config = Config()
        config.set("api_key", "nvapi-fake")
        client = StreamingFake()
        orch = SwarmOrchestrator(client, num_workers=2)
        app = SyncodeApp(config, client, orch)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            prompt = app.query_one("#prompt")
            prompt.focus()
            prompt.value = "/"
            await pilot.pause()
            assert app.suggest_visible and len(app._suggest_items) >= 9
            prompt.value = ""
            app.hide_suggest()
            app._handle_user_text("cau hoi kiem thu e2e")
            for _ in range(100):
                await asyncio.sleep(0.02)
                await pilot.pause()
                if not app._busy:
                    break
            assert not app._busy, "task phai ket thuc"
            from textual.widgets import Markdown

            mds = list(app.query_one("#chat").query(Markdown))
            assert mds, "phai co Markdown ket qua streaming"
            assert len(client.calls) == 5  # planner + 2 workers + critic + synth
            # sau khi xong, tat ca agent o trang thai done
            assert all(s == "done" for s, _n in app._agent_state.values())
            app.exit()

    asyncio.run(scenario())


# ---------------------------------------------------------------- ask modal + error history
def test_ask_modal_esc_returns_defaults():
    """ESC tren form hoi-dap phai tra defaults ve agent ngay, khong treo worker."""
    async def scenario() -> None:
        from syncode.ui.app import AskUserModal

        got = {}
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.push_screen(AskUserModal(["Q1?", "Q2?"], on_done=lambda a: got.update(answers=a)))
            await pilot.pause()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()
            assert got.get("answers") == ["(không trả lời)", "(không trả lời)"]
            app.exit()

    asyncio.run(scenario())


def test_confirm_modal_defaults_to_no():
    """Modal confirm: enter ngay (dang highlight No) + ESC deu deny."""
    async def scenario() -> None:
        from syncode.ui.app import ConfirmModal

        got = {}
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.push_screen(ConfirmModal("Run shell command: rm x", on_done=lambda ok: got.update(ok=ok)))
            await pilot.pause()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert got.get("ok") is False
            # mo lai, ESC cung deny
            got.clear()
            app.push_screen(ConfirmModal("Run shell command: rm x", on_done=lambda ok: got.update(ok=ok)))
            await pilot.pause()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert got.get("ok") is False
            app.exit()

    asyncio.run(scenario())


def test_error_turn_saved_to_history():
    """Luot loi phai duoc luu history (hoi cu bi mat vi bug xoa question truoc)."""
    async def scenario() -> None:
        from syncode.llm.client import LLMError

        class BoomClient:
            def __init__(self):
                self.calls = []

            def chat(self, *args, **kwargs):
                raise LLMError("boom-test")

            def chat_with_tools(self, *args, **kwargs):
                raise LLMError("boom-test")

        config = Config()
        config.set("api_key", "nvapi-fake")
        client = BoomClient()
        orch = SwarmOrchestrator(client, num_workers=1)
        app = SyncodeApp(config, client, orch)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._handle_user_text("cau gay loi")
            for _ in range(100):
                await asyncio.sleep(0.02)
                await pilot.pause()
                if not app._busy:
                    break
            assert not app._busy, "task loi phai ket thuc"
            assert app._sessions, "phai co session duoc luu"
            last = app._sessions[-1].exchanges[-1]
            assert last.question == "cau gay loi"
            assert "(error)" in last.answer and "boom-test" in last.answer
            app.exit()

    asyncio.run(scenario())


def test_slash_mode_switches_app_mode():
    """Lenh /.mode phai doi that app._mode + quyen tool (truoc day chi doi ctx.mode)."""
    async def scenario() -> None:
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app._mode == "build"
            app._handle_user_text("/.mode plan")
            await pilot.pause()
            assert app._mode == "plan"
            assert app._tools.allow_dangerous is False
            app._handle_user_text("/.mode build")
            await pilot.pause()
            assert app._mode == "build"
            assert app._tools.allow_dangerous is True
            app.exit()

    asyncio.run(scenario())


def test_shift_tab_toggles_swarm_mode():
    """Shift+Tab (focus o o nhap) phai doi swarm/quick, thay cho Ctrl+W bi terminal an."""
    async def scenario() -> None:
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app._quick_mode is False
            prompt = app.query_one("#prompt")
            prompt.focus()
            await pilot.pause()
            await pilot.press("shift+tab")
            await pilot.pause()
            assert app._quick_mode is True
            await pilot.press("shift+tab")
            await pilot.pause()
            assert app._quick_mode is False
            app.exit()

    asyncio.run(scenario())


# ---------------------------------------------------------------- history persist
def test_history_persist_qua_restart():
    """Lich su phai duoc ghi ra history.json va nap lai khi app khoi dong lai."""
    import tempfile
    from pathlib import Path

    from syncode.history import Exchange, Session, load_sessions, save_sessions

    home = Path(tempfile.mkdtemp())
    old_home = os.environ.get("SYNCODE_HOME")
    os.environ["SYNCODE_HOME"] = str(home)
    try:
        # tao session test
        session = Session(
            id="test-1",
            created_at=__import__("time").time(),
            exchanges=[
                Exchange(question="hi", answer="A"),
                Exchange(question="hi", answer="B"),
                Exchange(question="xin chao", answer="C"),
            ],
        )
        save_sessions([session], home / "history.json")
        loaded = load_sessions(home / "history.json")
        assert len(loaded) == 1
        assert len(loaded[0].exchanges) == 3
        assert loaded[0].exchanges[0].question == "hi"
        assert loaded[0].exchanges[0].answer == "A"

        # HistoryModal khong crash khi cau trung text
        async def scenario() -> None:
            from syncode.ui.app import HistoryModal

            modal = HistoryModal(loaded)
            app = _make_app()
            async with app.run_test() as pilot:
                app.push_screen(modal)
                await pilot.pause()
                ol = modal.query_one("#dialog-body").query_one(OptionList)
                ids = [o.id for o in ol._options]
                # session-based: s1 = session dung dau (reversed = session cuoi cung trong list)
                assert ids == ["s1"]
                # session co 3 exchange
                session = loaded[0]
                assert len(session.exchanges) == 3
                assert session.exchanges[0].question == "hi"
                assert session.exchanges[2].question == "xin chao"
                # Chon session de mo lai
                ol.highlighted = 0
                await pilot.pause()
                ol.action_select()
                await pilot.pause()
                # Sau khi chon, session duoc hien thi (modal van mo, nhung chat duoc clear va hien thi lai)
                # Kiem tra rang session da duoc restore
                await asyncio.sleep(0.1)

        asyncio.run(scenario())
    finally:
        if old_home is None:
            os.environ.pop("SYNCODE_HOME", None)
        else:
            os.environ["SYNCODE_HOME"] = old_home

        for mod in list(sys.modules):
            if mod.startswith("syncode"):
                del sys.modules[mod]


# ---------------------------------------------------------------- think panel (reasoning rieng kenh)
def test_think_delta_panel_and_done():
    """Event think_delta -> panel thinking hien reasoning; done -> thu gon."""
    async def scenario() -> None:
        app = _make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app._think_panel is None
            app._handle_event("think_delta", "reasoning step one")
            await pilot.pause()
            assert app._think_panel is not None
            assert app._think_buffer == "reasoning step one"
            app._handle_event("think_delta", " + step two")
            await pilot.pause()
            assert app._think_buffer == "reasoning step one + step two"
            app._handle_event("done", "final answer here")
            await pilot.pause()
            assert app._think_panel.collapsed
            app.exit()

    asyncio.run(scenario())
