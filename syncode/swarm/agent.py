"""Agent trong swarm: mot vai tro + system prompt rieng."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from syncode.llm.client import LLMClient, Message

# Gioi han vong lap tool-call de tranh lap vo han
MAX_TOOL_ROUNDS = 8

# Gioi han ky tu ket qua tool dua vao context (chong tran context file lon)
TOOL_OUTPUT_CHARS = 8000

# Tool hoi-dap user: chi nhan cac dong "Q -> A" SAU marker nay (format cua
# tool ask_user/qa_user) de tranh nhan nham mui ten trong code/text thuong.
_QA_ANSWERS_MARKER = "câu trả lời của người dùng"
_QA_LINE_RE = re.compile(r"^(.+?)\s*->\s*(.+)$")
_QA_MAX_PAIRS = 5


def parse_tool_qa(output: str) -> List[Tuple[str, str]]:
    """Trich cac cap (cau hoi, dap an) tu ket qua tool ask_user/qa_user.

    Chi tin cac dong "Q -> A" sau marker dap an (format co dinh cua tool).
    Tra [] neu khong phai ket qua hoi-dap. Khong bao gio raise.
    """
    try:
        lines = (output or "").splitlines()
    except Exception:  # noqa: BLE001
        return []
    start = 0
    for i, ln in enumerate(lines):
        try:
            if _QA_ANSWERS_MARKER in ln.lower():
                start = i + 1
                break
        except Exception:  # noqa: BLE001
            continue
    if start == 0:
        return []
    pairs: List[Tuple[str, str]] = []
    for ln in lines[start:]:
        if len(pairs) >= _QA_MAX_PAIRS:
            break
        try:
            m = _QA_LINE_RE.match(ln.strip())
        except Exception:  # noqa: BLE001
            continue
        if not m:
            continue
        q, a = m.group(1).strip(), m.group(2).strip()
        if q and a:
            pairs.append((q, a))
    return pairs


@dataclass
class Agent:
    name: str
    icon: str
    style: str          # mau rich, vi du "bold cyan"
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int | None = None
    model: str = ""     # model rieng cua role; de trong = model chinh
    transcript: List[Message] = field(default_factory=list, repr=False)
    tool_qa: List[Tuple[str, str]] = field(default_factory=list, repr=False)  # Q&A da hoi xong qua ask_user/qa_user

    def run(
        self,
        client: LLMClient,
        task: str,
        context: str = "",
        on_delta=None,
        on_thinking=None,
        thinking=None,
    ) -> str:
        """Thuc hien mot nhiem vu, tra ve ket qua van ban.
        on_thinking: nhan reasoning rieng kenh (khong tron vao ket qua).
        thinking: ep che do thinking rieng luot goi (None = theo config)."""
        user_content = task if not context else f"{task}\n\n---\nNGU CANH:\n{context}"
        messages: List[Message] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        output = client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            on_delta=on_delta,
            model=self.model or None,
            on_thinking=on_thinking,
            thinking=thinking,
        )
        self.transcript = messages + [{"role": "assistant", "content": output}]
        return output

    def run_with_tools(
        self,
        client: LLMClient,
        task: str,
        registry,
        context: str = "",
        on_event=None,
        *,
        changes=None,
        ask_user=None,
        confirm=None,
        tool_output_chars: int | None = None,
        max_tool_rounds: int | None = None,
        on_thinking=None,
    ) -> str:
        """Chay vong tool-calling: model goi tool -> thuc thi -> dua ket qua
        -> lap lai toi da max_tool_rounds vong (mac dinh MAX_TOOL_ROUNDS),
        hoac tra loi van ban cuoi."""
        cap = tool_output_chars if isinstance(tool_output_chars, int) and tool_output_chars > 0 else TOOL_OUTPUT_CHARS
        rounds = max_tool_rounds if isinstance(max_tool_rounds, int) and max_tool_rounds > 0 else MAX_TOOL_ROUNDS
        user_content = task if not context else f"{task}\n\n---\nNGU CANH:\n{context}"
        messages: List[Message] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        schemas = registry.openai_schemas()
        ask_used = False  # moi luot run chi hoi user qua tool TOI DA 1 lan (chong hoi don)
        for _ in range(rounds):
            result = client.chat_with_tools(
                messages, schemas, temperature=self.temperature, max_tokens=self.max_tokens,
                model=self.model or None,
            )
            calls = result.get("tool_calls", [])
            content = result.get("content", "")
            think = result.get("reasoning", "")
            if think and on_thinking is not None:
                on_thinking(think)
            if not calls:
                self.transcript = messages + [{"role": "assistant", "content": content}]
                return content
            # bo sung message assistant voi tool_calls de OpenAI SDK Chap nhan
            messages.append({
                "role": "assistant",
                "content": content or None,  # type: ignore[dict-item]
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)},
                    }
                    for c in calls
                ],
            })
            for call in calls:
                if call["name"] in ("ask_user", "qa_user"):
                    if ask_used:
                        # Chan cung: hoi don thi khong thuc thi, ep model tu quyet.
                        output = (
                            "Hệ thống: lượt này đã hỏi user 1 lần qua tool — "
                            "KHÔNG hỏi thêm. Tự quyết từ đáp án đã có và trả lời luôn."
                        )
                    else:
                        ask_used = True
                        output = registry.execute(
                            call["name"],
                            call["arguments"],
                            changes=changes,
                            ask=ask_user,
                            emit=on_event,
                            confirm=confirm,
                        )
                        # Ghi nhan hoi-dap DA XONG de judge ket luan thay vi hoi lai.
                        try:
                            self.tool_qa.extend(parse_tool_qa(output))
                        except Exception:  # noqa: BLE001
                            pass
                else:
                    output = registry.execute(
                        call["name"],
                        call["arguments"],
                        changes=changes,
                        ask=ask_user,
                        emit=on_event,
                        confirm=confirm,
                    )
                if on_event:
                    on_event("tool", (self.name, call["name"], output))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": output[:cap],  # gioi han de khong tran context
                })
        # het vong -> chot bang mot luot khong tool
        final = client.chat(
            messages, temperature=self.temperature, max_tokens=self.max_tokens,
            model=self.model or None, on_thinking=on_thinking,
        )
        self.transcript = messages + [{"role": "assistant", "content": final}]
        return final

    def reset(self) -> None:
        self.transcript.clear()
