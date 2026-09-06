"""Kiem thu pipeline swarm Planner -> Workers -> Critic -> Judge (LLM gia)."""

from syncode.swarm.orchestrator import SwarmOrchestrator, clean_answer


def _role(system: str) -> str:
    """Doan 'Ban la X' trong system prompt -> ten vai tro."""
    return system.split("cua mot swarm AI")[0].strip()


def test_clean_answer_cat_bao_cao():
    """Cau tra loi cuoi khong duoc con sot 'Ma trien khai' / 'Tong ket' / 'Yeu cau:'."""
    msg = (
        "Phản hồi chào hỏi\n\n"
        "• Yêu cầu: Nhận diện từ \"hi\" và trả lời chào hỏi.\n"
        "• Câu trả lời: Chào bạn! Tôi có thể giúp gì cho bạn hôm nay?\n\n"
        "Mã triển khai (Python)\n\n"
        "```python\nimport re\n```\n\n"
        "Tổng kết\nYêu cầu chào hỏi được xử lý thành công."
    )
    out = clean_answer(msg)
    assert "Mã triển khai" not in out
    assert "Tổng kết" not in out
    assert "yêu cầu:" not in out.lower().replace("• ", "")
    assert "Chào bạn! Tôi có thể giúp gì cho bạn hôm nay?" in out
    # khong con xot code khong ai yeu cau
    assert "import re" not in out


class FakeClient:
    """Gia lap LLMClient. Lop con chi override _reply, chat tu dong ghi log."""

    def __init__(self) -> None:
        self.calls = []

    def chat(self, messages, temperature=None, max_tokens=None, on_delta=None, model=None, on_thinking=None, thinking=None):
        self.calls.append({"role": _role(messages[0]["content"]), "user": messages[1]["content"], "model": model})
        return self._reply(messages[0]["content"], messages[1]["content"])

    def chat_with_tools(self, messages, schemas, temperature=None, max_tokens=None, model=None):
        """Gia lap tool-calling: model tu tra loi, khong goi tool nao."""
        self.calls.append({"role": _role(messages[0]["content"]), "user": messages[1]["content"], "model": model})
        return {"content": self._reply(messages[0]["content"], messages[1]["content"]), "tool_calls": []}

    def _reply(self, system: str, user: str) -> str:
        role = _role(system)
        if "PLANNER" in role:
            return ('```json\n[{"id": "t1", "title": "Phan tich", "goal": "g1"},'
                    '{"id": "t2", "title": "Tong hop", "goal": "g2"}]\n```')
        if "CRITIC" in role:
            return '{"verdict": "APPROVED"}'
        if "WORKER" in role:
            return f"ket qua cho: {user[:40]}"
        return "CAU TRA LOI HOAN CHINH"


def test_swarm_pipeline_happy_path():
    client = FakeClient()
    orchestrator = SwarmOrchestrator(client, num_workers=2)
    result = orchestrator.run("cau hoi test", max_rounds=1)

    assert result.answer == "CAU TRA LOI HOAN CHINH"
    assert [s.id for s in result.subtasks] == ["t1", "t2"]
    assert result.rounds == 0  # APPROVED ngay -> khong can refiner
    roles = [c["role"] for c in client.calls]
    # Planner(1) + Worker(2) + Critic(1) + Judge(1) = 5 lan goi LLM
    assert len(client.calls) == 5
    assert any("PLANNER" in r for r in roles)
    assert sum("WORKER" in r for r in roles) == 2
    assert any("CRITIC" in r for r in roles)
    assert any("JUDGE" in r for r in roles)
    # Judge nhan duoc ngu canh cua ca 2 worker
    synth_user = client.calls[-1]["user"]
    assert "[t1]" in synth_user and "[t2]" in synth_user


def test_clean_answer_strips_thinking_process_variants():
    """Cac bien the 'thinking process' khong tag phai bi loc sach."""
    msg = (
        "Here's a thinking process:\n"
        "1. Chon Django.\n"
        "2. Tra loi gon.\n\n"
        "Here is my thinking:\nblabla\n\n"
        "My thinking\nblabla\n\n"
        "Kết quả: framework là Django."
    )
    out = clean_answer(msg)
    low = out.lower()
    assert "here's a thinking process" not in low
    assert "here is my thinking" not in low
    assert "my thinking" not in low
    assert "framework là django" in low


def test_clean_answer_strips_qa_echo_block():
    """Khoi echo Q&A cua tool (header + dong 'Q -> A') bi xoa, dong '->' o noi khac giu lai."""
    msg = (
        "Framework là Django.\n\n"
        "Câu trả lời của người dùng:\n"
        "Framework gì? -> Django\n"
        "Cần test không? -> Có\n\n"
        "Code minh hoa: x -> y."
    )
    out = clean_answer(msg)
    assert "Câu trả lời của người dùng" not in out
    assert "Framework gì? -> Django" not in out
    assert "Cần test không? -> Có" not in out
    assert "Framework là Django." in out
    assert "x -> y" in out  # dong '->' binh thuong khong bi xoa


def test_clean_answer_preamble_keeps_rest():
    """Loi mo dau thua bi boc, phan noi dung that giu lai."""
    msg = "Dựa trên câu trả lời của bạn, framework là Django."
    assert clean_answer(msg) == "framework là Django."
    msg2 = "Cảm ơn bạn đã cung cấp thông tin. Kết quả: dùng Django."
    assert clean_answer(msg2) == "dùng Django."
    # Loi chao don thuan van giu (khong mat cau tra loi)
    assert clean_answer("Cảm ơn bạn!") == "Cảm ơn bạn!"


def test_clean_answer_never_empty():
    """Loc qua tay cung khong duoc tra ve rong khi dau vao con chu."""
    msg = "Tổng kết\nKết luận\nSummary"
    out = clean_answer(msg)
    assert out.strip() != ""


def test_clean_answer_strips_think_blocks():
    """Khoi <think>/<thinking> bi xoa ca tag + noi dung, giu dap an."""
    assert clean_answer("<think>reasoning here</think>Done.") == "Done."
    assert clean_answer("<THINKING>abc</THINKING>Kết quả: ok.") == "ok."
    assert "fib" in clean_answer("Code:\n```python\ndef fib(n): pass\n```")


def test_ask_user_result_has_no_echo_guard():
    """Tool-result hoi user phai kem chi dan cam echo + giu Q&A."""
    from syncode.tools.registry import ToolRegistry

    reg = ToolRegistry()
    out = reg.execute("ask_user", {"questions": ["Framework gì?"]}, ask=lambda qs: ["Django"])
    assert "Câu trả lời của người dùng:" in out
    assert "Framework gì? -> Django" in out
    assert "KHÔNG trích dẫn" in out  # chi dan anti-echo di kem


def test_swarm_run_with_tools_no_verbose_leak():
    """Mo phong worker dung ask_user roi tra loi verbose: dap an cuoi phai sach."""

    class QAVerboseClient(FakeClient):
        def __init__(self):
            super().__init__()
            self._tool_round_done = False

        def chat_with_tools(self, messages, schemas, temperature=None, max_tokens=None, model=None):
            self.calls.append({
                "role": _role(messages[0]["content"]),
                "user": messages[1]["content"],
                "model": model,
            })
            if not self._tool_round_done:
                self._tool_round_done = True
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "c1", "name": "ask_user",
                        "arguments": {"questions": ["Framework gì?"]},
                    }],
                }
            return {
                "content": (
                    "Here's a thinking process:\n"
                    "1. User chon Django.\n"
                    "Cảm ơn bạn đã cung cấp thông tin!\n"
                    "Câu trả lời của người dùng:\nFramework gì? -> Django\n\n"
                    "Dựa trên câu trả lời của bạn, framework là Django."
                ),
                "tool_calls": [],
            }

        def _reply(self, system, user):
            role = _role(system)
            if "PLANNER" in role:
                return '[{"id": "t1", "title": "Lam", "goal": "lam"}]'
            if "CRITIC" in role:
                return '{"verdict": "APPROVED"}'
            if "JUDGE" in role:
                return (
                    "Cảm ơn bạn đã cung cấp thông tin. "
                    "Dựa trên câu trả lời của bạn, framework là Django.\n\n"
                    "Câu trả lời của người dùng:\nFramework gì? -> Django\n\n"
                    "Tổng kết\nXong."
                )
            return super()._reply(system, user)

    from syncode.tools.registry import ToolRegistry

    client = QAVerboseClient()
    orch = SwarmOrchestrator(client, num_workers=1, tools=ToolRegistry())
    orch.ask_callback = lambda qs: ["Django"]
    result = orch.run("chon framework", max_rounds=1)

    low = result.answer.lower()
    for junk in ("here's a thinking process", "câu trả lời của người dùng",
                 "framework gì? -> django", "tổng kết", "cảm ơn bạn"):
        assert junk not in low, f"leak: {junk!r} trong {result.answer!r}"
    assert "framework là django" in low


def test_workers_run_in_parallel():
    """2 subtask phai chay DONG THOI (barrier): Worker-2 khong doi Worker-1 xong."""
    import threading

    class ParallelClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.barrier = threading.Barrier(2, timeout=20)

        def chat(self, messages, temperature=None, max_tokens=None, on_delta=None, model=None, on_thinking=None, thinking=None):
            role = _role(messages[0]["content"])
            self.calls.append({"role": role, "user": messages[1]["content"], "model": model})
            if "WORKER" in role:
                self.barrier.wait()  # chi pass khi ca 2 worker cung dang chay
                return "worker-done"
            return self._reply(messages[0]["content"], messages[1]["content"])

    client = ParallelClient()
    orch = SwarmOrchestrator(client, num_workers=2)
    result = orch.run("cau hoi test", max_rounds=1)
    assert result.answer == "CAU TRA LOI HOAN CHINH"
    assert sum("WORKER" in c["role"] for c in client.calls) == 2
    assert [s.output for s in result.subtasks] == ["worker-done", "worker-done"]


def test_ask_user_without_callback_reports_error():
    """Khong co ask_callback: tool phai bao loi ro rang cho model,
    khong duoc gia vo '(không trả lời)' roi lam tiep."""
    from syncode.tools.registry import ToolRegistry

    class AskOnceClient(FakeClient):
        def __init__(self):
            super().__init__()
            self._asked = False
            self.second_messages = None

        def chat_with_tools(self, messages, schemas, temperature=None, max_tokens=None, model=None):
            self.calls.append({
                "role": _role(messages[0]["content"]),
                "user": messages[1]["content"],
                "model": model,
            })
            if not self._asked:
                self._asked = True
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "c1", "name": "ask_user",
                        "arguments": {"questions": ["Q?"]},
                    }],
                }
            self.second_messages = messages
            return {"content": "xong", "tool_calls": []}

        def _reply(self, system, user):
            role = _role(system)
            if "PLANNER" in role:
                return '[{"id": "t1", "title": "Lam", "goal": "lam"}]'
            if "CRITIC" in role:
                return '{"verdict": "APPROVED"}'
            if "JUDGE" in role:
                return "OK"
            return super()._reply(system, user)

    client = AskOnceClient()
    orch = SwarmOrchestrator(client, num_workers=1, tools=ToolRegistry())
    # co tinh KHONG gan ask_callback
    orch.run("task can hoi", max_rounds=1)
    tool_texts = [
        m.get("content", "") for m in (client.second_messages or [])
        if m.get("role") == "tool"
    ]
    assert any("không hỏi được" in t for t in tool_texts)


def test_swarm_pipeline_revise_round():
    class ReviseClient(FakeClient):
        def _reply(self, system: str, user: str) -> str:
            role = _role(system)
            if "CRITIC" in role:
                return '{"verdict": "REVISE", "issues": ["thieu vi du"], "guidance": "bo sung"}'
            if "REFINER" in role:
                return "BAN DA REFINE"
            return super()._reply(system, user)

    client = ReviseClient()
    orchestrator = SwarmOrchestrator(client, num_workers=2)
    result = orchestrator.run("cau hoi test", max_rounds=1)

    assert result.rounds == 1
    assert result.answer == "CAU TRA LOI HOAN CHINH"
    # Planner(1)+Worker(2)+Critic(1)+Refiner(1)+Judge(1) = 6
    assert len(client.calls) == 6
    refiner_calls = [c for c in client.calls if "REFINER" in c["role"]]
    assert len(refiner_calls) == 1
    assert "thieu vi du" in refiner_calls[0]["user"]  # refiner nhan issue tu critic


def test_format_history_caps_and_empty():
    from syncode.swarm.orchestrator import (
        HISTORY_MAX_CHARS,
        format_history,
    )

    assert format_history(None) == ""
    assert format_history([]) == ""
    out = format_history([("q1", "a1"), ("q2", "a2")])
    assert "Q: q1" in out and "A: a2" in out
    out2 = format_history([("q", "x" * 5000)])
    assert len(out2) <= HISTORY_MAX_CHARS + 200
    assert "rut gon" in out2


def test_recent_exchanges_limit():
    import time

    from syncode.history import Exchange, Session, recent_exchanges

    s = Session(
        id="s1",
        created_at=time.time(),
        exchanges=[Exchange(question=f"q{i}", answer=f"a{i}") for i in range(10)],
    )
    assert recent_exchanges([s], "s1", limit=3) == [("q7", "a7"), ("q8", "a8"), ("q9", "a9")]
    assert recent_exchanges([s], "nope") == []
    assert recent_exchanges([], None) == []


def test_swarm_run_includes_history_in_prompts():
    """Planner + worker + synth deu nhan khoi LICH SU de hieu 'cho do'."""

    class HistClient(FakeClient):
        def _reply(self, system, user):
            role = _role(system)
            if "PLANNER" in role:
                assert "LICH SU HOI DAP" in user
                return '[{"id": "t1", "title": "Lam", "goal": "lam"}]'
            if "CRITIC" in role:
                return '{"verdict": "APPROVED"}'
            if "WORKER" in role:
                assert "LICH SU HOI DAP" in user
                return "worker-ok"
            if "JUDGE" in role:
                assert "LICH SU HOI DAP" in user
                return "SYNTH OK"
            return super()._reply(system, user)

    client = HistClient()
    orch = SwarmOrchestrator(client, num_workers=1)
    result = orch.run("sua lai cho do", max_rounds=1, history=[("viet ham fib", "def fib...")])
    assert result.answer == "SYNTH OK"


def test_quick_reply_includes_history_messages():
    seen = {}

    class QClient(FakeClient):
        def chat(self, messages, temperature=None, max_tokens=None, on_delta=None, model=None, on_thinking=None, thinking=None):
            seen["messages"] = messages
            return "HIST OK"

    client = QClient()
    orch = SwarmOrchestrator(client, num_workers=1)
    ans = orch.quick_reply("con cach khac?", history=[("q0", "a0")])
    assert ans == "HIST OK"
    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert seen["messages"][1]["content"] == "q0"
    assert seen["messages"][-1]["content"] == "con cach khac?"


def test_tool_output_chars_cap():
    """Ket qua tool dai bi cat theo cap (mac dinh 8000, tuy chinh duoc)."""
    from syncode.swarm.agent import TOOL_OUTPUT_CHARS, Agent
    from syncode.swarm.roles import WORKER_SYSTEM

    seen = {}

    class BigRegistry:
        def openai_schemas(self):
            return []

        def execute(self, name, arguments, **kwargs):
            return "z" * 20000

    class OneCallClient:
        def __init__(self):
            self.n = 0

        def chat_with_tools(self, messages, schemas, temperature=None, max_tokens=None, model=None):
            # lan 1 goi tool, lan 2 het tool de lay output
            self.n += 1
            if self.n == 1:
                return {"content": "", "tool_calls": [{"id": "c1", "name": "big", "arguments": {}}]}
            seen["tool_msgs"] = [m for m in messages if m.get("role") == "tool"]
            return {"content": "xong", "tool_calls": []}

    agent = Agent("W", "*", "bold", WORKER_SYSTEM)
    agent.run_with_tools(OneCallClient(), "task", BigRegistry(), tool_output_chars=100)
    assert len(seen["tool_msgs"]) == 1
    assert len(seen["tool_msgs"][0]["content"]) == 100
    assert TOOL_OUTPUT_CHARS == 8000


def test_max_tool_rounds_respected():
    """max_tool_rounds gioi han so vong goi tool (mac dinh 8)."""
    from syncode.swarm.agent import MAX_TOOL_ROUNDS, Agent
    from syncode.swarm.roles import WORKER_SYSTEM

    class AlwaysToolClient:
        def __init__(self):
            self.tool_calls = 0
            self.final_calls = 0

        def chat_with_tools(self, messages, schemas, temperature=None, max_tokens=None, model=None):
            self.tool_calls += 1
            return {"content": "", "tool_calls": [{"id": "c1", "name": "t", "arguments": {}}]}

        def chat(self, messages, temperature=None, max_tokens=None, model=None, on_thinking=None, thinking=None):
            self.final_calls += 1
            return "final"

    class OkRegistry:
        def openai_schemas(self):
            return []

        def execute(self, name, arguments, **kwargs):
            return "ok"

    assert MAX_TOOL_ROUNDS == 8
    agent = Agent("W", "*", "bold", WORKER_SYSTEM)
    client = AlwaysToolClient()
    assert agent.run_with_tools(client, "task", OkRegistry(), max_tool_rounds=2) == "final"
    assert client.tool_calls == 2
    assert client.final_calls == 1


def test_session_todos_roundtrip():
    import time

    from syncode.history import Session, format_todos, open_todos

    s = Session(
        id="s1",
        created_at=time.time(),
        exchanges=[],
        todos=[{"text": "viet test", "done": False}, {"text": "xong", "done": True}],
    )
    d = s.to_dict()
    s2 = Session.from_dict(d)
    assert len(s2.todos) == 2
    assert open_todos([s2], "s1") == [{"text": "viet test", "done": False}]
    out = format_todos(s2.todos)
    assert "CHECKLIST" in out and "viet test" in out and "xong" not in out
    assert format_todos([]) == "" and format_todos(None) == ""
    # du lieu cu khong co todos van doc duoc
    s3 = Session.from_dict({"id": "x", "created_at": 0.0, "exchanges": []})
    assert s3.todos == []


def test_swarm_run_includes_todos():
    """Worker nhan checklist dang do, viec xong bi loai."""

    class TodoClient(FakeClient):
        def _reply(self, system, user):
            role = _role(system)
            if "PLANNER" in role:
                return '[{"id": "t1", "title": "Lam", "goal": "lam"}]'
            if "CRITIC" in role:
                return '{"verdict": "APPROVED"}'
            if "WORKER" in role:
                assert "CHECKLIST" in user and "viet test" in user
                assert "viec xong" not in user
                return "worker-ok"
            if "JUDGE" in role:
                return "TODO OK"
            return super()._reply(system, user)

    orch = SwarmOrchestrator(
        TodoClient(),
        num_workers=1,
    )
    result = orch.run(
        "tiep tuc",
        max_rounds=1,
        todos=[{"text": "viet test", "done": False}, {"text": "viec xong", "done": True}],
    )
    assert result.answer == "TODO OK"


def test_context_overflow_retries_without_history():
    """Tran context -> tu thu lai 1 lan, bo lich su; loi khac nem ngay."""
    from syncode.llm.client import LLMError

    planner_inputs = []

    class OverflowClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.plan_calls = 0

        def chat(self, messages, temperature=None, max_tokens=None, on_delta=None, model=None, on_thinking=None, thinking=None):
            role = _role(messages[0]["content"])
            user = messages[1]["content"]
            self.calls.append({"role": role, "user": user, "model": model})
            if "PLANNER" in role:
                self.plan_calls += 1
                planner_inputs.append(user)
                if self.plan_calls == 1:
                    raise LLMError("This model's maximum context length is 8000 tokens")
                return '[{"id": "t1", "title": "Lam", "goal": "lam"}]'
            return self._reply(messages[0]["content"], user)

    client = OverflowClient()
    orch = SwarmOrchestrator(client, num_workers=1)
    result = orch.run("cau dai", max_rounds=1, history=[("q0", "a0" * 500)])
    assert client.plan_calls == 2
    assert "LICH SU HOI DAP" in planner_inputs[0]
    assert "LICH SU HOI DAP" not in planner_inputs[1]
    assert result.answer == "CAU TRA LOI HOAN CHINH"

    class AuthFailClient(FakeClient):
        def chat(self, messages, temperature=None, max_tokens=None, on_delta=None, model=None, on_thinking=None, thinking=None):
            raise LLMError("auth failed 401")

    orch2 = SwarmOrchestrator(AuthFailClient(), num_workers=1)
    try:
        orch2.run("hi", max_rounds=1, history=[("q", "a")])
        raise AssertionError("phai nem loi auth")
    except LLMError as exc:
        assert "auth failed" in str(exc)


def test_clean_answer_strips_analysis_narration():
    """Cac muc tu phan tich kieu 'Analyze user input' / 'identify core task'."""
    msg = (
        "**Analyze user input:**\n"
        "The user wants a login form.\n\n"
        "### Identify core task\n"
        "Build the form.\n\n"
        "1. Analyze the request\n"
        "2. Break down the request\n\n"
        "Step 1: Analyze user input\n\n"
        "Let me analyze the request first.\n\n"
        "Here is the login form code:"
    )
    out = clean_answer(msg)
    low = out.lower()
    for junk in ("analyze user input", "identify core task",
                 "break down the request", "let me analyze", "step 1"):
        assert junk not in low, f"leak: {junk!r} trong {out!r}"
    assert "Here is the login form code:" in out


def test_clean_answer_analysis_inline_and_legit():
    """Inline restatement -> bo; noi dung that sau ':' -> giu; huong dan that giu nguyen."""
    msg = "Identify core task: build login form\n\nHere is the code:"
    out = clean_answer(msg)
    assert "identify core task" not in out.lower()
    assert "Here is the code:" in out
    # noi dung that sau "Analyze the X:" duoc giu
    assert clean_answer("Analyze the login function: it returns None") == "it returns None"
    assert clean_answer("My analysis: the bug is a race") == "the bug is a race"
    # cac buoc huong dan that khong bi cat
    legit = "1. Install deps\n2. Run tests\n\nAnalyze data carefully."
    out = clean_answer(legit)
    assert "Install deps" in out and "Run tests" in out
    assert "Analyze data carefully." in out


def test_clean_answer_strips_midline_narration():
    """Cau narration mo dau dong bi cat, phan con lai + code fence duoc giu."""
    msg = "Let me analyze user input. Here is the fix:\n```python\nx = 1\n```"
    out = clean_answer(msg)
    assert "Let me analyze" not in out
    assert "Here is the fix:" in out
    assert "x = 1" in out
    # comment trong code fence khong bi dong toi
    code = "```python\n# First, I will analyze the data\nx = 1\n```"
    assert "First, I will analyze the data" in clean_answer(code)


def test_judge_output_strips_analysis_sections():
    """Judge tra ve verbose co muc phan tich: dap an cuoi phai sach."""

    class AnalysisClient(FakeClient):
        def _reply(self, system, user):
            role = _role(system)
            if "PLANNER" in role:
                return '[{"id": "t1", "title": "Lam", "goal": "lam"}]'
            if "CRITIC" in role:
                return '{"verdict": "APPROVED"}'
            if "WORKER" in role:
                return "worker-ok"
            if "JUDGE" in role:
                return (
                    "**Analyze user input:**\n"
                    "User hoi ve Django.\n\n"
                    "### Identify core task\n"
                    "Chon framework.\n\n"
                    "1. Analyze the request\n"
                    "2. Break down the request\n\n"
                    "Framework là Django."
                )
            return super()._reply(system, user)

    orch = SwarmOrchestrator(AnalysisClient(), num_workers=1)
    result = orch.run("chon framework", max_rounds=1)
    low = result.answer.lower()
    for junk in ("analyze user input", "identify core task",
                 "break down the request"):
        assert junk not in low, f"leak: {junk!r} trong {result.answer!r}"
    assert "framework là django" in low


def test_judge_thinking_emitted_as_event():
    """Reasoning rieng kenh tu Judge -> event think_delta, khong vao dap an."""

    class ThinkClient(FakeClient):
        def chat(self, messages, temperature=None, max_tokens=None, on_delta=None, model=None, on_thinking=None, thinking=None):
            role = _role(messages[0]["content"])
            self.calls.append({"role": role, "user": messages[1]["content"], "model": model})
            if "JUDGE" in role and on_thinking is not None:
                on_thinking("analyzing options")
            return self._reply(messages[0]["content"], messages[1]["content"])

    events = []
    orch = SwarmOrchestrator(ThinkClient(), num_workers=1)
    orch.on_event = lambda e, d: events.append((e, d))
    result = orch.run("hi", max_rounds=1)
    assert ("think_delta", "analyzing options") in events
    assert result.answer == "CAU TRA LOI HOAN CHINH"


def test_clean_answer_strips_bare_narration():
    """Cau throat-clearing ngan ve yeu cau dung mot minh -> bo ca dong."""
    msg = "Intro.\n\nI need to analyze the request.\n\nReal content."
    out = clean_answer(msg)
    assert "I need to analyze" not in out
    assert "Intro." in out and "Real content." in out
    # cung cau do trong code fence -> bao ve, khong dong toi
    code = "```\nFirst, I will analyze user input\nprint(x)\n```"
    out = clean_answer(code)
    assert "First, I will analyze user input" in out
    assert "print(x)" in out
    # cau dai/khong chi yeu cau -> giu
    assert "I need to understand quantum computing to explain it." in clean_answer(
        "I need to understand quantum computing to explain it."
    )


def test_judge_model_fallback_synthesizer_key():
    """Config cu synthesizer_model van dung duoc cho Judge."""

    class Cfg:
        def __init__(self, data):
            self._d = data

        def get(self, key, default=None):
            return self._d.get(key, default)

    class FakeCfgClient(FakeClient):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg

    orch = SwarmOrchestrator(FakeCfgClient(Cfg({"synthesizer_model": "m-old"})), num_workers=1)
    orch.apply_models()
    assert orch.judge.model == "m-old"
    orch2 = SwarmOrchestrator(
        FakeCfgClient(Cfg({"judge_model": "m-new", "synthesizer_model": "m-old"})),
        num_workers=1,
    )
    orch2.apply_models()
    assert orch2.judge.model == "m-new"


def test_specialist_routing_and_fallback():
    """Planner gan specialist -> dung agent chuyen mon; sai/thieu -> worker thuong."""

    class SpecClient(FakeClient):
        def _reply(self, system, user):
            role = _role(system)
            if "PLANNER" in role:
                return (
                    '[{"id": "t1", "title": "Viet", "goal": "g1", "specialist": "reviewer"},'
                    ' {"id": "t2", "title": "Doc", "goal": "g2", "specialist": "khong-co"},'
                    ' {"id": "t3", "title": "Tim", "goal": "g3"}]'
                )
            if "CRITIC" in role:
                return '{"verdict": "APPROVED"}'
            if "JUDGE" in role:
                return "SPEC OK"
            return super()._reply(system, user)

    client = SpecClient()
    orch = SwarmOrchestrator(client, num_workers=2)
    result = orch.run("lam 3 viec", max_rounds=1)
    assert [s.specialist for s in result.subtasks] == ["reviewer", "", ""]
    systems = [c["user"] for c in client.calls]
    assert result.answer == "SPEC OK"
    # subtask reviewer chay bang prompt REVIEWER, con lai bang WORKER thuong
    worker_roles = [c["role"] for c in client.calls if "WORKER" in c["role"] or "REVIEWER" in c["role"]]
    assert sum("REVIEWER" in r for r in worker_roles) == 1
    assert sum("WORKER" in r and "REVIEWER" not in r for r in worker_roles) == 2


def test_sibling_overview_shared_board():
    """Nhieu subtask -> worker nao cung thay toan cuc de khoi chong lan."""

    class BoardClient(FakeClient):
        def _reply(self, system, user):
            role = _role(system)
            if "PLANNER" in role:
                return ('[{"id": "t1", "title": "Viet A", "goal": "g1"},'
                        '{"id": "t2", "title": "Viet B", "goal": "g2"}]')
            if "CRITIC" in role:
                return '{"verdict": "APPROVED"}'
            if "WORKER" in role:
                assert "TOAN CUC" in user
                assert "t1 Viet A" in user and "t2 Viet B" in user
                return "board-ok"
            if "JUDGE" in role:
                return "BOARD OK"
            return super()._reply(system, user)

    orch = SwarmOrchestrator(BoardClient(), num_workers=2)
    assert orch.run("2 viec", max_rounds=1).answer == "BOARD OK"


def test_find_agent_covers_pool_and_specialists():
    orch = SwarmOrchestrator(FakeClient(), num_workers=2)
    assert orch.find_agent("Worker-1").name == "Worker-1"
    assert orch.find_agent("Coder").name == "Coder"
    assert orch.find_agent("reviewer").name == "Reviewer"
    assert orch.find_agent("Judge").name == "Judge"
    assert orch.find_agent("khong-co") is None


def test_strip_reasoning_echo_verbatim_only():
    """Chi xoa ban sao verbatim DAI; cau ngan/legit giu nguyen."""
    from syncode.swarm.orchestrator import _strip_reasoning_echo

    trace = "x" * 150
    assert _strip_reasoning_echo(f"Intro {trace} outro.", trace) == "Intro  outro."
    assert _strip_reasoning_echo("Hello.", "short") == "Hello."
    assert _strip_reasoning_echo("Hello.", "") == "Hello."
    assert _strip_reasoning_echo("No echo here.", "y" * 200) == "No echo here."


def test_judge_dedups_duplicated_thinking():
    """Provider paste trace verbatim vao content -> dap an cuoi sach."""

    class EchoClient(FakeClient):
        def chat(self, messages, temperature=None, max_tokens=None, on_delta=None, model=None, on_thinking=None, thinking=None):
            role = _role(messages[0]["content"])
            self.calls.append({"role": role, "user": messages[1]["content"], "model": model})
            if "JUDGE" in role:
                trace = "t" * 120
                if on_thinking is not None:
                    on_thinking(trace)
                return f"Intro {trace} FINAL."
            return self._reply(messages[0]["content"], messages[1]["content"])

    orch = SwarmOrchestrator(EchoClient(), num_workers=1)
    result = orch.run("hi", max_rounds=1)
    assert "t" * 120 not in result.answer
    assert "FINAL." in result.answer
    assert "Intro" in result.answer


def test_last_clean_stats_recorded():
    orch = SwarmOrchestrator(FakeClient(), num_workers=1)
    result = orch.run("hi", max_rounds=1)
    assert result.answer == "CAU TRA LOI HOAN CHINH"
    assert orch.last_clean["raw_chars"] >= orch.last_clean["final_chars"] > 0


def test_worker_output_dedups_thinking():
    """Worker echo trace vao output -> subtask.output sach truoc khi toi Judge."""
    from syncode.tools.registry import ToolRegistry

    class WorkerEchoClient(FakeClient):
        def chat_with_tools(self, messages, schemas, temperature=None, max_tokens=None, model=None):
            self.calls.append({
                "role": _role(messages[0]["content"]),
                "user": messages[1]["content"],
                "model": model,
            })
            trace = "w" * 120
            return {"content": f"draft {trace} done.", "reasoning": trace, "tool_calls": []}

        def _reply(self, system, user):
            role = _role(system)
            if "PLANNER" in role:
                return '[{"id": "t1", "title": "Lam", "goal": "lam"}]'
            if "CRITIC" in role:
                return '{"verdict": "APPROVED"}'
            if "JUDGE" in role:
                return "JUDGED."
            return super()._reply(system, user)

    orch = SwarmOrchestrator(WorkerEchoClient(), num_workers=1, tools=ToolRegistry())
    result = orch.run("task", max_rounds=1)
    assert "w" * 120 not in result.subtasks[0].output
    assert "draft" in result.subtasks[0].output
    assert result.answer == "JUDGED."


_ECHO_DUP_TRACE = "Here's a thinking process: " + "y" * 300


class _EchoDupClient(FakeClient):
    """Judge lan 1 tra echo-dup (content == reasoning), lan repair (off) tra sach."""

    def __init__(self):
        super().__init__()
        self.thinkings = []

    def chat(self, messages, temperature=None, max_tokens=None, on_delta=None, model=None, on_thinking=None, thinking=None):
        self.calls.append({"role": _role(messages[0]["content"]), "user": messages[1]["content"], "model": model})
        self.thinkings.append(thinking)
        if len(self.thinkings) == 1:
            if on_thinking is not None:
                on_thinking(_ECHO_DUP_TRACE)
            return _ECHO_DUP_TRACE
        return "Ba câu hỏi: 1) ... 2) ... 3) ...?"


def test_synthesize_repairs_full_echo_dup():
    """Echo nuot het dap an Judge -> goi lai 1 lan thinking=off, co dap an that."""
    from syncode.swarm.orchestrator import SubTask

    client = _EchoDupClient()
    orch = SwarmOrchestrator(client, num_workers=1)
    out = orch._synthesize("đặt 3 câu hỏi", [SubTask(id="t1", title="T", goal="g", output="w")])
    assert out == "Ba câu hỏi: 1) ... 2) ... 3) ...?"
    assert client.thinkings == [None, "off"]  # repair ep off, khong doi config
    assert orch.last_echo_repair is True


def test_quick_reply_repairs_full_echo_dup():
    client = _EchoDupClient()
    orch = SwarmOrchestrator(client, num_workers=1)
    out = orch.quick_reply("đặt 3 câu hỏi")
    assert "Ba câu hỏi" in out
    assert client.thinkings == [None, "off"]
    assert orch.last_clean.get("echo_repair") is True


def test_no_repair_when_answer_healthy():
    """Dap an binh thuong -> khong repair, khong ton them luot goi."""
    client = FakeClient()
    orch = SwarmOrchestrator(client, num_workers=2)
    result = orch.run("cau hoi test", max_rounds=1)
    assert result.answer == "CAU TRA LOI HOAN CHINH"
    assert orch.last_echo_repair is False
    assert orch.last_clean.get("echo_repair") is False


def test_parse_tool_qa_real_format():
    """Trich Q&A tu ket qua that cua tool ask_user/qa_user."""
    from syncode.swarm.agent import parse_tool_qa

    out = (
        "[NỘI BỘ — KHÔNG trích dẫn, KHÔNG nhắc lại: dùng câu trả lời dưới đây LẶNG LẼ "
        "để làm tiếp nhiệm vụ gốc.]\n"
        "Câu trả lời của người dùng:\n"
        "Số của bạn lớn hơn 5 không? -> Không\n"
        "Số của bạn là số chẵn không? -> Có\n"
        "[Hết phần nội bộ — trả lời TRỰC TIẾP nhiệm vụ gốc, gọn.]"
    )
    assert parse_tool_qa(out) == [
        ("Số của bạn lớn hơn 5 không?", "Không"),
        ("Số của bạn là số chẵn không?", "Có"),
    ]
    assert parse_tool_qa("ket qua grep binh thuong, khong co Q&A") == []
    assert parse_tool_qa("A -> B trong code khong co marker") == []
    assert parse_tool_qa("") == []


class _QARegistry:
    """Registry gia: tool qa_user tra dap an co dinh."""

    def openai_schemas(self):
        return [{"type": "function", "function": {"name": "qa_user"}}]

    def execute(self, name, arguments, **kwargs):
        assert name == "qa_user"
        return (
            "[NỘI BỘ]\nCâu trả lời của người dùng:\n"
            "Số của bạn lớn hơn 5 không? -> Không\n"
            "Số của bạn là số chẵn không? -> Có\n"
            "[Hết phần nội bộ.]"
        )


class _QAToolClient(FakeClient):
    """Lan 1 goi tool qa_user, lan 2 chot dap an."""

    def chat_with_tools(self, messages, schemas, temperature=None, max_tokens=None, model=None):
        self.calls.append({"role": "tools", "user": "", "model": model})
        if len(self.calls) == 1:
            return {
                "content": "",
                "reasoning": "",
                "tool_calls": [{"id": "c1", "name": "qa_user", "arguments": {"questions": ["q1?", "q2?"]}}],
            }
        return {"content": "Đoán số của bạn là 4.", "reasoning": "", "tool_calls": []}


def test_worker_tracks_completed_tool_qa():
    """Worker goi qa_user -> agent ghi nhan cap Q&A de judge ket luan."""
    from syncode.swarm.agent import Agent
    from syncode.swarm.roles import WORKER_SYSTEM

    client = _QAToolClient()
    worker = Agent("Worker-1", "*", "bold cyan", WORKER_SYSTEM)
    out = worker.run_with_tools(client, "đoán số 1-10", _QARegistry(), "")
    assert out == "Đoán số của bạn là 4."
    assert worker.tool_qa == [
        ("Số của bạn lớn hơn 5 không?", "Không"),
        ("Số của bạn là số chẵn không?", "Có"),
    ]


class _JudgeRecorder(FakeClient):
    """Ghi lai input cua Judge de kiem tra block DA-HOI."""

    def __init__(self):
        super().__init__()
        self.judge_inputs = []

    def chat(self, messages, temperature=None, max_tokens=None, on_delta=None, model=None, on_thinking=None, thinking=None):
        self.calls.append({"role": _role(messages[0]["content"]), "user": messages[1]["content"], "model": model})
        if "JUDGE" in _role(messages[0]["content"]):
            self.judge_inputs.append(messages[1]["content"])
        return "KẾT LUẬN: số 4."


def test_synthesize_includes_completed_tool_qa():
    """Co Q&A da xong -> judge nhan block DA-HOI + lenh ket luan, khong hoi lai."""
    from syncode.swarm.orchestrator import SubTask

    client = _JudgeRecorder()
    orch = SwarmOrchestrator(client, num_workers=1)
    orch.last_tool_qa = [("Số của bạn lớn hơn 5 không?", "Không")]
    out = orch._synthesize("đoán số", [SubTask(id="t1", title="T", goal="g", output="w")])
    assert out == "KẾT LUẬN: số 4."
    assert len(client.judge_inputs) == 1
    assert "ĐÃ HỎI QUA TOOL" in client.judge_inputs[0]
    assert "Số của bạn lớn hơn 5 không? -> Không" in client.judge_inputs[0]
    assert "KHÔNG hỏi lại dưới mọi dạng" in client.judge_inputs[0]


def test_synthesize_no_qa_block_when_no_tool_qa():
    """Khong co Q&A -> khong chen block thua."""
    from syncode.swarm.orchestrator import SubTask

    client = _JudgeRecorder()
    orch = SwarmOrchestrator(client, num_workers=1)
    orch.last_tool_qa = []
    orch._synthesize("cau hoi thuong", [SubTask(id="t1", title="T", goal="g", output="w")])
    assert "ĐÃ HỎI QUA TOOL" not in client.judge_inputs[0]


def test_repair_keeps_completed_tool_qa_block():
    """Luot repair cung phai co block DA-HOI (khong hoi lai bang Q1/A1 rong)."""
    from syncode.swarm.orchestrator import SubTask

    class _EchoDupRecorder(_JudgeRecorder):
        def __init__(self):
            super().__init__()
            self.thinkings = []

        def chat(self, messages, temperature=None, max_tokens=None, on_delta=None, model=None, on_thinking=None, thinking=None):
            self.thinkings.append(thinking)
            if len(self.thinkings) == 1:
                if on_thinking is not None:
                    on_thinking(_ECHO_DUP_TRACE)
                self.calls.append({"role": "judge", "user": messages[1]["content"], "model": model})
                self.judge_inputs.append(messages[1]["content"])
                return _ECHO_DUP_TRACE
            return super().chat(messages, temperature, max_tokens, on_delta, model, on_thinking, thinking)

    client = _EchoDupRecorder()
    orch = SwarmOrchestrator(client, num_workers=1)
    orch.last_tool_qa = [("Số của bạn lớn hơn 5 không?", "Không")]
    out = orch._synthesize("đoán số", [SubTask(id="t1", title="T", goal="g", output="w")])
    assert out == "KẾT LUẬN: số 4."
    assert client.thinkings == [None, "off"]
    assert len(client.judge_inputs) == 2  # judge chinh + repair
    for inp in client.judge_inputs:
        assert "ĐÃ HỎI QUA TOOL" in inp
        assert "Số của bạn lớn hơn 5 không? -> Không" in inp
    assert orch.last_echo_repair is True


def test_run_with_tools_caps_user_questions_to_one():
    """Model doi hoi lan 2 qua tool -> bi chan cung, registry chi chay 1 lan."""
    from syncode.swarm.agent import Agent
    from syncode.swarm.roles import WORKER_SYSTEM

    executed = []

    class _SpammyClient:
        def __init__(self):
            self.n = 0

        def chat_with_tools(self, messages, schemas, temperature=None, max_tokens=None, model=None):
            self.n += 1
            if self.n <= 2:
                return {
                    "content": "",
                    "reasoning": "",
                    "tool_calls": [{"id": f"c{self.n}", "name": "ask_user", "arguments": {"questions": ["q?"]}}],
                }
            return {"content": "Tự quyết: xong.", "reasoning": "", "tool_calls": []}

        def chat(self, *args, **kwargs):
            raise AssertionError("khong duoc rot xuong final chat khi model da dung")

    class _CountRegistry:
        def openai_schemas(self):
            return []

        def execute(self, name, arguments, **kwargs):
            executed.append(name)
            return "Câu trả lời của người dùng:\nq? -> Có"

    worker = Agent("Worker-1", "*", "bold cyan", WORKER_SYSTEM)
    out = worker.run_with_tools(_SpammyClient(), "viec can hoi", _CountRegistry(), "")
    assert out == "Tự quyết: xong."
    assert executed == ["ask_user"]  # lan 2 bi chan, khong thuc thi
    assert worker.tool_qa == [("q?", "Có")]
