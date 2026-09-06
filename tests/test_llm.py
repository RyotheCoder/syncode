"""Kiem thu retry/backoff/timeout cua LLMClient voi stub (khong mang that)."""

import types

from syncode.config import Config
from syncode.llm.client import LLMClient, LLMError


def _retryable_exc():
    from openai import APIConnectionError

    return APIConnectionError(message="conn reset", request=None)


def _ok_resp(text="OK"):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=text, tool_calls=None)
            )
        ]
    )


def _stub_client(behaviors):
    class StubCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            b = behaviors[min(self.calls, len(behaviors) - 1)]
            self.calls += 1
            if isinstance(b, Exception):
                raise b
            return b

    comp = StubCompletions()
    stub = types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp))
    return stub, comp


def test_retry_defaults_in_config():
    config = Config()
    assert config.get("timeout") == 120.0
    assert config.get("max_retries") == 3


def test_chat_retries_then_succeeds(monkeypatch):
    import time

    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    client = LLMClient(Config())
    stub, comp = _stub_client([_retryable_exc(), _retryable_exc(), _ok_resp("DONE")])
    client._client = stub
    assert client.chat([{"role": "user", "content": "hi"}]) == "DONE"
    assert comp.calls == 3
    assert slept == [1.0, 2.0]


def test_chat_gives_up_after_max_retries(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda s: None)
    config = Config()
    config.set("max_retries", 1)
    client = LLMClient(config)
    stub, comp = _stub_client([_retryable_exc()])
    client._client = stub
    try:
        client.chat([{"role": "user", "content": "hi"}])
        raise AssertionError("phai nem LLMError")
    except LLMError:
        pass
    assert comp.calls == 2  # 1 lan dau + 1 retry


def test_chat_no_retry_on_fatal(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda s: (_ for _ in ()).throw(Exception("sleep called")))
    client = LLMClient(Config())
    stub, comp = _stub_client([ValueError("bad request")])
    client._client = stub
    try:
        client.chat([{"role": "user", "content": "hi"}])
        raise AssertionError("phai nem LLMError")
    except LLMError as exc:
        assert "bad request" in str(exc)
    assert comp.calls == 1


def test_stream_mid_failure_does_not_retry():
    """Rot giua stream: nhan loi luon, khong thu lai (tranh lap delta)."""
    got = []

    class StreamCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            assert kwargs.get("stream") is True

            def gen():
                yield types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content="A")
                        )
                    ]
                )
                raise _retryable_exc()

            return gen()

    comp = StreamCompletions()
    stub = types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp))
    client = LLMClient(Config())
    client._client = stub
    try:
        client.chat([{"role": "user", "content": "hi"}], on_delta=got.append)
        raise AssertionError("phai nem LLMError")
    except LLMError:
        pass
    assert got == ["A"]
    assert comp.calls == 1


def test_usage_recorded_non_stream():
    client = LLMClient(Config())
    resp = _ok_resp("hi")
    resp.usage = types.SimpleNamespace(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    stub, _comp = _stub_client([resp])
    client._client = stub
    assert client.chat([{"role": "user", "content": "hi"}]) == "hi"
    assert client.usage["prompt_tokens"] == 10
    assert client.usage["completion_tokens"] == 5
    assert client.usage["total_tokens"] == 15
    assert client.usage["measured_calls"] == 1
    assert client.last_usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_usage_stream_counts_without_tokens():
    def gen():
        yield types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(delta=types.SimpleNamespace(content="B"))
            ]
        )

    class StreamCompletions:
        def create(self, **kwargs):
            assert kwargs.get("stream") is True
            return gen()

    stub = types.SimpleNamespace(chat=types.SimpleNamespace(completions=StreamCompletions()))
    client = LLMClient(Config())
    client._client = stub
    got = []
    assert client.chat([{"role": "user", "content": "hi"}], on_delta=got.append) == "B"
    assert got == ["B"]
    assert client.usage["streamed_calls"] == 1
    assert client.usage["total_tokens"] == 0
    assert client.last_usage is None


def test_get_reasoning_forms():
    from syncode.llm.client import _get_reasoning

    assert _get_reasoning(types.SimpleNamespace()) == ""
    assert _get_reasoning({}) == ""
    assert _get_reasoning(types.SimpleNamespace(reasoning_content="r1")) == "r1"
    assert _get_reasoning({"reasoning": "r2"}) == "r2"
    assert _get_reasoning(types.SimpleNamespace(reasoning_content=["a", "b"])) == ""
    assert _get_reasoning(
        types.SimpleNamespace(reasoning=[{"text": "x"}, {"text": "y"}])
    ) == "xy"


def test_stream_reasoning_separated():
    """reasoning_content di kenh rieng: khong tron vao dap an."""

    def gen():
        yield types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content="", reasoning_content="thinking about X")
                )
            ]
        )
        yield types.SimpleNamespace(
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="Hi"))]
        )

    class StreamCompletions:
        def create(self, **kwargs):
            assert kwargs.get("stream") is True
            return gen()

    stub = types.SimpleNamespace(chat=types.SimpleNamespace(completions=StreamCompletions()))
    client = LLMClient(Config())
    client._client = stub
    got, thought = [], []
    out = client.chat(
        [{"role": "user", "content": "hi"}],
        on_delta=got.append,
        on_thinking=thought.append,
    )
    assert out == "Hi"
    assert got == ["Hi"]
    assert thought == ["thinking about X"]


def test_nonstream_reasoning_separated():
    resp = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="Answer", reasoning_content="plan: first do this"
                )
            )
        ]
    )
    stub, _comp = _stub_client([resp])
    client = LLMClient(Config())
    client._client = stub
    thought = []
    assert client.chat([{"role": "user", "content": "hi"}], on_thinking=thought.append) == "Answer"
    assert thought == ["plan: first do this"]


def _cfg_nosave(monkeypatch, **kv):
    """Config khong ghi dia (tranh lam ban HOME dung chung cua suite)."""
    from syncode.config import Config as _Config

    config = _Config()
    monkeypatch.setattr(_Config, "save", lambda self: None)
    for k, v in kv.items():
        config.set(k, v)
    return config


def test_thinking_off_sends_extra_body(monkeypatch):
    seen = {}

    class Cap:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _ok_resp("ok")

    config = _cfg_nosave(monkeypatch, thinking="off")
    client = LLMClient(config)
    client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Cap()))
    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert seen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "reasoning_effort" not in seen


def test_thinking_off_ultra_adds_reasoning_effort(monkeypatch):
    seen = {}

    class Cap:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _ok_resp("ok")

    config = _cfg_nosave(monkeypatch, thinking="off")
    client = LLMClient(config)
    client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Cap()))
    client.chat([{"role": "user", "content": "hi"}], model="nvidia/nemotron-3-ultra-550b")
    assert seen["reasoning_effort"] == "none"
    assert seen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_thinking_auto_sends_nothing_extra(monkeypatch):
    seen = {}

    class Cap:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _ok_resp("ok")

    client = LLMClient(_cfg_nosave(monkeypatch, thinking="auto"))  # auto ghi ro, khong phu thuoc default
    client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Cap()))
    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert "extra_body" not in seen
    assert "reasoning_effort" not in seen


def test_thinking_default_is_auto(monkeypatch):
    seen = {}

    class Cap:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _ok_resp("ok")

    client = LLMClient(_cfg_nosave(monkeypatch))  # khong set -> default auto
    client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Cap()))
    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert "extra_body" not in seen  # auto = giu hanh vi model, khong gui gi them
    assert client.last_call["thinking"] == "auto"


def test_thinking_low_sends_low_effort(monkeypatch):
    seen = {}

    class Cap:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _ok_resp("ok")

    config = _cfg_nosave(monkeypatch, thinking="low")
    client = LLMClient(config)
    client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Cap()))
    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert seen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}


def test_last_call_records_reasoning_and_params(monkeypatch):
    def gen():
        yield types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content="Hi", reasoning_content="r-think")
                )
            ]
        )

    class StreamCompletions:
        def create(self, **kwargs):
            assert kwargs.get("stream") is True
            return gen()

    client = LLMClient(_cfg_nosave(monkeypatch, thinking="auto"))
    client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=StreamCompletions()))
    got = []
    assert client.chat([{"role": "user", "content": "hi"}], on_delta=got.append) == "Hi"
    assert got == ["Hi"]
    info = client.last_call
    assert info["reasoning_seen"] is True
    assert info["answer_chars"] == 2
    assert info["thinking"] == "auto"
    assert info["extra_body"] is None


def test_last_call_no_reasoning():
    client = LLMClient(Config())
    stub, _comp = _stub_client([_ok_resp("ok")])
    client._client = stub
    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert client.last_call["reasoning_seen"] is False
    assert client.last_call["answer_chars"] == 2


_ECHO_TRACE = "Here's a thinking process: " + "x" * 300  # >= 100 ky tu


def _stream_cap(spec):
    """Gia lap SSE stream: spec = [(content, reasoning), ...]."""
    seen = {}

    class Cap:
        def create(self, **kwargs):
            seen.update(kwargs)
            assert kwargs.get("stream") is True
            for content, reasoning in spec:
                yield types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            delta=types.SimpleNamespace(content=content, reasoning_content=reasoning)
                        )
                    ]
                )

    return Cap(), seen


def test_stream_suppresses_verbatim_content_echo(monkeypatch):
    """Provider nhan doi trace vao content: echo khong lot ra live/panel dap an."""
    deltas, thinks = [], []
    client = LLMClient(_cfg_nosave(monkeypatch, thinking="auto"))
    cap, _seen = _stream_cap([(None, _ECHO_TRACE), (_ECHO_TRACE, None)])
    client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=cap))
    out = client.chat(
        [{"role": "user", "content": "hi"}],
        on_delta=deltas.append, on_thinking=thinks.append,
    )
    assert out == ""
    assert deltas == []
    assert "".join(thinks) == _ECHO_TRACE  # reasoning van vao panel thinking
    assert client.last_call["reasoning_seen"] is True
    assert client.last_call["answer_was_echo"] is True
    assert client.last_call["echo_dropped_chars"] == len(_ECHO_TRACE)


def test_stream_suppresses_split_echo(monkeypatch):
    """Echo bi cat le bien chunk giua 2 kenh van bi chan (withhold-and-release)."""
    parts = [_ECHO_TRACE[i:i + 37] for i in range(0, len(_ECHO_TRACE), 37)]
    assert len(parts) > 3
    deltas = []
    client = LLMClient(_cfg_nosave(monkeypatch, thinking="auto"))
    cap, _seen = _stream_cap([(None, _ECHO_TRACE)] + [(p, None) for p in parts])
    client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=cap))
    assert client.chat([{"role": "user", "content": "hi"}], on_delta=deltas.append) == ""
    assert deltas == []
    assert client.last_call["answer_was_echo"] is True


def test_stream_keeps_real_answer(monkeypatch):
    """Content that thi khong bi cat nham du reasoning co truoc."""
    think = "private reasoning " * 20
    deltas = []
    client = LLMClient(_cfg_nosave(monkeypatch, thinking="auto"))
    cap, _seen = _stream_cap([(None, think), ("Xin chào!", None)])
    client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=cap))
    assert client.chat([{"role": "user", "content": "hi"}], on_delta=deltas.append) == "Xin chào!"
    assert deltas == ["Xin chào!"]
    assert client.last_call["answer_was_echo"] is False


def test_stream_drops_echo_keeps_following_answer(monkeypatch):
    """Echo xong dap an that den ngay sau: bo echo, giu dap an."""
    deltas = []
    client = LLMClient(_cfg_nosave(monkeypatch, thinking="auto"))
    cap, _seen = _stream_cap([(None, _ECHO_TRACE), (_ECHO_TRACE, None), ("Câu trả lời thật.", None)])
    client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=cap))
    out = client.chat([{"role": "user", "content": "hi"}], on_delta=deltas.append)
    assert out == "Câu trả lời thật."
    assert deltas == ["Câu trả lời thật."]
    assert client.last_call["echo_dropped_chars"] == len(_ECHO_TRACE)
    assert client.last_call["answer_was_echo"] is False


def test_once_flags_full_echo(monkeypatch):
    """Non-stream: content == reasoning verbatim -> danh dau answer_was_echo."""
    client = LLMClient(_cfg_nosave(monkeypatch, thinking="auto"))
    resp = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=_ECHO_TRACE, reasoning_content=_ECHO_TRACE)
            )
        ]
    )
    stub, _comp = _stub_client([resp])
    client._client = stub
    assert client.chat([{"role": "user", "content": "hi"}]) == _ECHO_TRACE
    assert client.last_call["answer_was_echo"] is True


def test_thinking_override_off_for_one_call(monkeypatch):
    """thinking='off' ep 1 luot goi ma khong doi config (dung cho repair)."""
    client = LLMClient(_cfg_nosave(monkeypatch, thinking="auto"))
    seen = {}

    class Cap:
        def create(self, **kwargs):
            seen.update(kwargs)
            return _ok_resp("ok")

    client._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Cap()))
    assert client.chat([{"role": "user", "content": "hi"}], thinking="off") == "ok"
    assert seen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert client.last_call["thinking"] == "off"
    assert client.config.get("thinking") == "auto"  # config khong doi
