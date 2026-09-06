"""NVIDIA NIM client (OpenAI-compatible endpoint) with streaming support."""

from __future__ import annotations

import json
import time
from typing import Callable, Dict, List, Optional

from openai import OpenAI

from syncode.config import Config

Message = Dict[str, str]
DeltaCallback = Optional[Callable[[str], None]]
ThinkCallback = Optional[Callable[[str], None]]

# Doan reasoning gan nhat giu lai de nhan dien content-echo khi stream
# (provider paste trace verbatim vao content). Cap de `in` luon re.
_ECHO_WINDOW_CHARS = 20000
# Doan withheld ngan hon nguong nay thi KHONG ket luan echo (tranh cat nham
# cau tra loi ngan trung hop voi reasoning) — chi giu lai cho den khi du bang
# chung hoac stream ket thuc. Cung nguong voi MIN_ECHO_CHARS cua orchestrator.
_MIN_ECHO_SUPPRESS = 100


class LLMError(RuntimeError):
    """Loi goi LLM (auth, network, model...)."""


def _get_reasoning(obj) -> str:
    """Lay reasoning rieng kenh tu delta/message (reasoning_content/reasoning).

    Mot so provider (DeepSeek-R1/Nemotron tren NIM) tra thinking o field
    rieng thay vi tron vao content. Tra '' neu khong co. Khong bao gio raise.
    """
    for key in ("reasoning_content", "reasoning"):
        try:
            val = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(val, str) and val:
            return val
        if isinstance(val, list):  # dang content-block list
            parts = []
            for p in val:
                try:
                    t = p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "") or ""
                except Exception:  # noqa: BLE001
                    t = ""
                if isinstance(t, str) and t:
                    parts.append(t)
            if parts:
                return "".join(parts)
    return ""


def _echo_eaten(text: str, reasoning: str) -> bool:
    """True neu text KHONG con gi sau khi tru echo verbatim.

    Cung quy tac voi _strip_reasoning_echo cua orchestrator (>=100 ky tu
    verbatim moi tinh) nhung viet doc lap o day de tranh import vong tron
    (orchestrator import client). Dung de danh dau `answer_was_echo`.
    """
    body = (text or "").strip()
    needle = (reasoning or "").strip()
    if not body or len(needle) < _MIN_ECHO_SUPPRESS:
        return False
    return not body.replace(needle, "").strip()


def _is_retryable(exc: Exception) -> bool:
    """Loi mang/429/5xx/timeout thi dang thu lai; loi auth/tham so thi khong."""
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        )
    except ImportError:
        return False
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    try:
        from openai import APIStatusError

        if isinstance(exc, APIStatusError):
            code = exc.status_code
            return code is None or code == 429 or code >= 500
    except ImportError:
        pass
    return False


def _retry_after(exc: Exception, attempt: int) -> float:
    """Delay exponential (1s, 2s, 4s, ...) + ton trong header Retry-After neu co."""
    delay = min(8.0, 1.0 * (2 ** attempt))
    try:
        raw = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
        if raw is not None:
            delay = max(delay, min(60.0, float(raw)))
    except (TypeError, ValueError, AttributeError):
        pass
    return delay


class LLMClient:
    """Wrapper mo rong quanh OpenAI SDK tro vao NVIDIA NIM.

    NIM dung giao thuc OpenAI-compatible nen chi can doi base_url.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client: Optional[OpenAI] = None
        # Chan doan goi gan nhat (cho /.debug): model, che do thinking,
        # extra_body da gui, co thay reasoning rieng kenh khong.
        self.last_call: Dict[str, object] = {}
        # Cong don token su dung trong process (cho /.usage).
        # Streamed turns khong co usage -> dem rieng, khong uoc luong.
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "measured_calls": 0,
            "streamed_calls": 0,
        }
        self.last_usage = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            api_key = self.config.get("api_key")
            if not api_key:
                raise LLMError(
                    "Chua cau hinh API key. Chay /.key <NVIDIA_API_KEY> hoac "
                    "export NVIDIA_API_KEY=..."
                )
            try:
                timeout = float(self.config.get("timeout", 120.0) or 120.0)
            except (TypeError, ValueError):
                timeout = 120.0
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.config.get("base_url"),
                timeout=timeout,
            )
        return self._client

    def reset(self) -> None:
        """Goi lai sau khi doi api_key/base_url."""
        self._client = None

    def _thinking_mode(self, override: Optional[str] = None) -> str:
        """Che do thinking hien tai (auto/off/low), chuan hoa chu thuong.

        override: ep che do cho 1 luot goi (vd repair voi "off") ma khong
        doi config.
        """
        raw = override if override is not None else self.config.get("thinking", "auto")
        return str(raw or "auto").strip().lower()

    def _thinking_params(self, model_name: str, override: Optional[str] = None):
        """Tham so dieu khien thinking theo config `thinking` (auto/off/low).

        Tra ve (extra_body, top_level): chi gui khi user chon off/low;
        auto = giu nguyen hanh vi cu de khong lam vo model la.
        - off: tat reasoning (Nemotron 3/DeepSeek-v4 hieu enable_thinking;
          Ultra mac dinh reasoning high nen them reasoning_effort=none).
        - low: reasoning nhe, re (low_effort; Ultra dung medium).
        """
        mode = self._thinking_mode(override)
        name = str(model_name or "").lower()
        if mode == "off":
            extra: dict = {"chat_template_kwargs": {"enable_thinking": False}}
            top: dict = {}
            if "ultra" in name:
                top["reasoning_effort"] = "none"
            return extra, top
        if mode == "low":
            extra = {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}
            top = {}
            if "ultra" in name:
                top["reasoning_effort"] = "medium"
            return extra, top
        return {}, {}

    def _record_usage(self, resp) -> None:
        """Cong don usage tu response (neu co). Stream khong co usage
        thi chi dem so lượt (resp=None). Khong bao gio raise."""
        if resp is None:
            self.usage["streamed_calls"] += 1
            return
        try:
            u = getattr(resp, "usage", None)
            if u is None:
                return
            pt = int(getattr(u, "prompt_tokens", 0) or 0)
            ct = int(getattr(u, "completion_tokens", 0) or 0)
            tt = int(getattr(u, "total_tokens", 0) or (pt + ct))
        except (TypeError, ValueError, AttributeError):
            return
        self.last_usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
        self.usage["prompt_tokens"] += pt
        self.usage["completion_tokens"] += ct
        self.usage["total_tokens"] += tt
        self.usage["measured_calls"] += 1

    def _call_with_retry(self, fn):
        """Goi fn(), tu thu lai khi gap loi mang/429/5xx. Loi khac nem ngay."""
        try:
            max_retries = int(self.config.get("max_retries", 3) or 0)
        except (TypeError, ValueError):
            max_retries = 3
        last: Optional[Exception] = None
        for attempt in range(max(0, max_retries) + 1):
            try:
                return fn()
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt >= max(0, max_retries) or not _is_retryable(exc):
                    break
                time.sleep(_retry_after(exc, attempt))
        raise LLMError(str(last)) from last

    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        on_delta: DeltaCallback = None,
        model: Optional[str] = None,
        on_thinking: ThinkCallback = None,
        thinking: Optional[str] = None,
    ) -> str:
        """Goi chat completion. Neu on_delta duoc truyen -> stream token.

        model: de trong thi dung model chinh trong config (cho phep moi
        role trong swarm dung model rieng).
        on_thinking: nhan reasoning rieng kenh (neu provider tach ra);
        reasoning KHONG bao gio tron vao dap an tra ve.
        thinking: ep che do thinking cho rieng luot goi nay (auto/off/low);
        None = theo config. Dung cho repair khi echo nuot het dap an.
        """
        req_model = model or self.config.get("model")
        mode = self._thinking_mode(thinking)
        t_body, t_top = self._thinking_params(req_model, thinking)
        kwargs = dict(
            model=req_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.config.get("temperature") if temperature is None else temperature,
            max_tokens=self.config.get("max_tokens") if max_tokens is None else max_tokens,
            **t_top,
        )
        if t_body:
            kwargs["extra_body"] = t_body
        try:
            if on_delta is not None:
                chunks: List[str] = []
                got_any = False
                got_think = False
                think_chars = 0
                think_buf = ""
                pending = ""
                echo_dropped = 0

                def _emit_piece(text: str) -> None:
                    nonlocal got_any
                    if not text:
                        return
                    got_any = True
                    chunks.append(text)
                    on_delta(text)

                def do_stream():
                    nonlocal got_any, got_think, think_chars, think_buf, pending, echo_dropped
                    try:
                        stream = self.client.chat.completions.create(**kwargs, stream=True)  # type: ignore[call-overload]
                        for event in stream:
                            if event.choices:
                                delta = event.choices[0].delta
                                rpiece = _get_reasoning(delta)
                                if rpiece:
                                    got_think = True
                                    think_chars += len(rpiece)
                                    think_buf = (think_buf + rpiece)[-_ECHO_WINDOW_CHARS:]
                                    if on_thinking is not None:
                                        on_thinking(rpiece)
                                    # Reasoning den sau: pending truoc do khong khop
                                    # thi khong phai echo -> xa ra ngay.
                                    if pending and pending not in think_buf:
                                        _emit_piece(pending)
                                        pending = ""
                                piece = delta.content or ""
                                if not piece:
                                    continue
                                old = pending
                                pending = ""
                                buf = old + piece
                                if buf in think_buf:
                                    # Trung verbatim voi reasoning da thay:
                                    # giu lai (khong emit) cho den khi du bang
                                    # chung hoac stream ket thuc — chong lech
                                    # bien chunk giua 2 kenh.
                                    pending = buf
                                    continue
                                if len(old) >= _MIN_ECHO_SUPPRESS and old in think_buf:
                                    # Phan giu truoc do da du dai + khop verbatim
                                    # reasoning -> echo xac nhan: bo han, roi xu
                                    # ly piece hien tai nhu doan moi (echo xong
                                    # dap an that thuong den ngay sau).
                                    echo_dropped += len(old)
                                    if piece in think_buf:
                                        pending = piece
                                        continue
                                    _emit_piece(piece)
                                else:
                                    _emit_piece(buf)
                        if pending:
                            if len(pending) >= _MIN_ECHO_SUPPRESS and pending in think_buf:
                                echo_dropped = len(pending)
                                pending = ""
                            else:
                                _emit_piece(pending)
                                pending = ""
                    except LLMError:
                        raise
                    except Exception as exc:
                        if got_any:
                            # Rot giua chung: nhan loi luon, KHONG retry
                            # vi thu lai se lap delta tren UI.
                            raise LLMError(str(exc)) from exc
                        raise
                    return "".join(chunks)

                result = self._call_with_retry(do_stream)
                self._record_usage(None)  # stream: khong co usage
                was_echo = got_think and not result.strip() and think_chars > 0
                self.last_call = {
                    "model": req_model,
                    "thinking": mode,
                    "extra_body": dict(t_body) if t_body else None,
                    "reasoning_seen": got_think,
                    "reasoning_chars": think_chars,
                    "answer_chars": len(result),
                    "echo_dropped_chars": echo_dropped,
                    "answer_was_echo": was_echo,
                }
                return result

            def do_once():
                response = self.client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
                self._record_usage(response)
                message = response.choices[0].message
                r = _get_reasoning(message)
                if r and on_thinking is not None:
                    on_thinking(r)
                content = message.content or ""
                self.last_call = {
                    "model": req_model,
                    "thinking": mode,
                    "extra_body": dict(t_body) if t_body else None,
                    "reasoning_seen": bool(r),
                    "reasoning_chars": len(r),
                    "answer_chars": len(content),
                    "echo_dropped_chars": 0,
                    "answer_was_echo": _echo_eaten(content, r),
                }
                return content
            return self._call_with_retry(do_once)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(str(exc)) from exc

    def chat_with_tools(
        self,
        messages: List[Message],
        tools: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        thinking: Optional[str] = None,
    ) -> dict:
        """Mot luat chat co function calling.

        Tra ve {"content": str, "reasoning": str,
                "tool_calls": [{"id", "name", "arguments": dict}]}.
        Neu model khong goi tool nao -> tool_calls rong.
        "reasoning" la thinking rieng kenh (neu co), KHONG tron vao content.
        thinking: ep che do cho rieng luot goi (None = theo config).
        """
        req_model = model or self.config.get("model")
        mode = self._thinking_mode(thinking)
        t_body, t_top = self._thinking_params(req_model, thinking)
        kwargs: dict = dict(
            model=req_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.config.get("temperature") if temperature is None else temperature,
            max_tokens=self.config.get("max_tokens") if max_tokens is None else max_tokens,
            tools=tools,
            tool_choice="auto",
            **t_top,
        )
        if t_body:
            kwargs["extra_body"] = t_body
        try:
            def do_tools():
                response = self.client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
                self._record_usage(response)
                message = response.choices[0].message
                calls = []
                for call in getattr(message, "tool_calls", None) or []:
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except (json.JSONDecodeError, ValueError):
                        arguments = {}
                    calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})
                content = message.content or ""
                reason_text = _get_reasoning(message)
                self.last_call = {
                    "model": req_model,
                    "thinking": mode,
                    "extra_body": dict(t_body) if t_body else None,
                    "reasoning_seen": bool(reason_text),
                    "reasoning_chars": len(reason_text),
                    "answer_chars": len(content),
                    "echo_dropped_chars": 0,
                    "answer_was_echo": _echo_eaten(content, reason_text),
                }
                return {"content": content, "reasoning": reason_text, "tool_calls": calls}
            return self._call_with_retry(do_tools)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(str(exc)) from exc
