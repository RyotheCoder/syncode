"""Orchestrator: dieu phoi swarm Planner -> Workers -> Critic -> (Refiner) -> Judge."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from syncode.conventions import load_conventions, load_skills
from syncode.history import format_todos
from syncode.llm.client import LLMClient, LLMError
from syncode.swarm.agent import Agent
from syncode.swarm.roles import (
    CODER_SYSTEM,
    CRITIC_SYSTEM,
    JUDGE_SYSTEM,
    PLANNER_SYSTEM,
    QUICK_SYSTEM,
    REFINER_SYSTEM,
    RESEARCHER_SYSTEM,
    REVIEWER_SYSTEM,
    WORKER_SYSTEM,
)

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# Khoi thinking co tag (<think>...</think>, <thinking>...</thinking>): xoa CA KHOI
# (tag + noi dung), khong chi xoa tag. Dat o clean_answer de ap dung moi noi.
_THINK_BLOCK_RE = re.compile(r"<think(ing)?>.*?</think(ing)?>", re.IGNORECASE | re.DOTALL)

# Câu chào / hỏi thăm / cảm ơn / tạm biệt ngắn — trả lời nhanh 1 lượt chat,
# KHÔNG chạy swarm (tránh tách 2 subtask + tổng hợp dài cho 1 lời chào).
# Regex dùng chung cho TUI (app.py) và CLI (cli.py).
_SIMPLE_CHAT_RE = re.compile(
    r"^\s*("
    r"xin chào|chào bạn|chào|hello|hi|hey|halo|alo|helo|hi there|hello there|"
    r"kem chao|chao ban|chao|"
    r"cảm ơn|cám ơn|thanks|thank you|thankyou|thx|thank|"
    r"tạm biệt|bye|goodbye|bái bai|good night|chúc ngủ ngon|"
    r"bạn khỏe không|khỏe không|khỏe ko|how are you|how r u|you ok|r u ok|"
    r"bạn là ai|mày là ai|who are you|bạn tên gì|tên gì|"
    r"bạn làm được gì|bạn giúp được gì|bạn ok không|what can you do|"
    r"ok|okay|được|yes|no|um|uhm|"
    r"[!?….\s]+"
    r")\s*[!?.…]*\s*$",
    re.IGNORECASE,
)


def is_simple_chat(text: str) -> bool:
    """True nếu chỉ là chào/cảm ơn/tạm biệt ngắn → đủ trả lời 1 lượt quick."""
    t = (text or "").strip()
    if not t or len(t) > 40:
        return False
    return bool(_SIMPLE_CHAT_RE.match(t))


def needs_swarm(text: str) -> bool:
    """True nếu cần chạy full swarm; False nếu 1 lượt quick là đủ.

    Quy tắc hiện tại (bảo thủ để không làm cụt task code):
    - chào/cảm ơn/tạm biệt ngắn → False (quick).
    - mọi thứ còn lại (kể cả hỏi kiến thức ngắn) → True (swarm, nhưng swarm
      đã được dặn nén + đúng trọng tâm qua roles.py).
    """
    return not is_simple_chat(text)

# Cac dau hieu "bao cao" / tuong thuat qua trinh / echo hoi-dap — clean_answer cat bo.
# (Mo rong sau su co verbose "Here's a thinking process" + echo Q&A sau tool ask_user/qa_user.)
_VERBOSE_HEADINGS = (
    "tổng kết", "kết luận", "conclusion", "summary", "mã triển khai",
    "implementation", "subtask", "kế hoạch", "báo cáo", "quá trình",
    "here's think", "here's thinking", "here's a thinking", "here is a thinking",
    "here's my thinking", "here is my thinking", "here's the thinking",
    "here's what i did", "here is what i did", "my thinking",
    "thinking process", "thinking progress", "think progress", "thought process", "reasoning:",
    "issues to fix", "issues found", "fixes:", "fixes",
    "câu trả lời của người dùng",
    "[nội bộ", "[hết phần nội bộ",
)
# Note: "kết quả" KHONG nam trong list tren — no duoc xu ly rieng o duoi
# (giu phan sau dau ":" thay vi xoa ca dong).

# Loi mo dau thua ("Dựa trên câu trả lời của bạn, ...", "Cảm ơn bạn...! ...",
# "Here's what I did: ..."): GIU LAI phan con lai cua dong thay vi xoa ca dong,
# de khong bao gio lam mat noi dung that.
_PREAMBLE_KEEP_REST = (
    r"dựa (?:trên|vào) (?:câu trả lời của bạn|thông tin bạn cung cấp|những thông tin trên)",
    r"cảm ơn bạn[^.!?]*",
    r"thank you[^.!?]*",
    r"here'?s what i did",
    r"here is what i did",
    r"here'?s my analysis",
    r"here is my analysis",
    r"đây là những gì (?:tôi|mình) đã làm",
    r"analy[sz](?:e|ing)?\s+the\s+\w+",
    r"analy[sz](?:e|ing)?\s+this\s+\w+",
    r"analy[sz](?:e|ing)?\s+your\s+\w+",
    r"my\s+analysis",
    r"analysis",
)
_PREAMBLE_RE = re.compile(
    r"^\s*(?:" + "|".join(_PREAMBLE_KEEP_REST) + r")\s*[,.:]\s*(.*?)\s*$",
    re.IGNORECASE,
)

# Tuong thuat tu phan tich ("Analyze user input", "identify core task"...):
# model reasoning tu narrate qua trinh lam viec. Xu ly theo 3 muc:
# - _ANALYSIS_OPENERS: loi mo dau thuan tuy, khong mang thong tin -> bo ca dong.
# - _ANALYSIS_RESTATED: cau viet lai yeu cau (khong them thong tin) -> bo ca dong.
# - _ANALYSIS_LEADS + _PREAMBLE bo sung: dang "Cau: phan con lai" noi ve
#   noi dung that -> giu phan con lai; dang heading -> bo ca dong.
_ANALYSIS_OPENERS = (
    "let me analyze", "let me identify", "let me understand", "let me break",
)
_ANALYSIS_RESTATED = (
    "analyze user input",
    "analyze the request", "analyze this request", "analyze the query",
    "analyze the question", "analyze your request",
    "analyzing user input", "analyzing the request",
    "analysis of the request", "analysis of the query",
    "identify core task", "identify the core task", "identify main task",
    "identifying core task",
    "understand the request", "understanding the request",
    "break down the request", "breaking down the request",
)
_ANALYSIS_LEADS = (
    "analyze ", "analyzing ", "analysing ",
    "identify ", "identifying ",
    "understand ", "understanding ",
    "break down", "breaking down", "breakdown",
    "analysis of", "analysis",
    "step ",
)
_ANALYSIS_LIST_RE = re.compile(r"^(?:\d+[.)]?\s+|step\s*\d+\s*[:.)\-–]*\s*|[-*•·]\s+)+")
_ANALYSIS_REQUEST_WORDS = ("input", "request", "query", "question", "task", "requirement", "requirements")


def _strip_list_marker(s: str) -> str:
    """Bo danh so/bullet dau dong ("1.", "Step 2:", "- ") de nhan dien noi dung."""
    return _ANALYSIS_LIST_RE.sub("", s).strip()


def _match_restated(core: str) -> bool:
    """Dong viet lai yeu cau (khong them thong tin) -> bo ca dong."""
    for d in _ANALYSIS_RESTATED:
        if core == d or core.startswith(d + ":") or core.startswith(d + ".") or core.startswith(d + ","):
            return True
    return False


def _is_analysis_heading(line: str, head: str) -> bool:
    """Dong tuong thuat phan tich dang heading -> bo ca dong."""
    for lead in _ANALYSIS_OPENERS:
        if head.startswith(lead):
            return True
    core = _strip_list_marker(head)
    if not any(core.startswith(lead) for lead in _ANALYSIS_LEADS):
        return False
    raw = line.strip()
    if raw.endswith(":"):
        return True
    s = raw.lstrip()
    if s.startswith("#") or s.startswith("**"):
        return True
    if _ANALYSIS_LIST_RE.match(raw):
        # Danh so/bullet: chi cat khi noi ve chinh yeu cau ("Step 1: ...",
        # "1. Analyze user input"), giu cac buoc huong dan that ("1. Cai dat X").
        words = core.split()[:6]
        if core.startswith("step ") or any(w in words for w in _ANALYSIS_REQUEST_WORDS):
            return True
        if core.startswith(("break down", "breaking down", "breakdown", "analysis of", "analysis")):
            return True
        return False
    return False


_NARRATION_OPENER_CORE = (
    r"let me (?:analyze|identify|understand|break down)"
    r"|first,?\s+i(?:'ll| will)\s+(?:analyze|identify|understand)"
    r"|i need to\s+(?:first\s+)?(?:analyze|identify|understand)"
    r"|i(?:'ll| will)\s+(?:start\s+by\s+)?analy[sz](?:e|ing)?"
    r"|to\s+(?:better\s+)?understand\s+the\s+request"
)
_NARRATION_OPENERS_RE = re.compile(
    r"^\s*(?:" + _NARRATION_OPENER_CORE + r")\b[^.!?\n]*[.!?]",
    re.IGNORECASE,
)

# Cau narration ngan dung mot minh ("I need to analyze the request."):
# mo dau bang opener + chua tu chi yeu cau + ≤15 tu -> bo ca dong.
_REQUEST_WORD_RE = re.compile(
    r"\b(input|request|query|question|task|requirements?)\b", re.IGNORECASE
)
_BARE_NARRATION_RE = re.compile(
    r"^\s*(?:" + _NARRATION_OPENER_CORE + r")\b[^.!?\n]{0,120}[.!?]?\s*$",
    re.IGNORECASE,
)


def _is_bare_narration(line: str) -> bool:
    """Cau throat-clearing ngan ve chinh yeu cau -> bo ca dong."""
    s = line.strip()
    if not s or len(s.split()) > 15:
        return False
    if not _BARE_NARRATION_RE.match(s):
        return False
    return bool(_REQUEST_WORD_RE.search(s))

# Nhan phan tich nhieu tu + dau ":" ("Analyze the login function: X" -> "X").
# Chi ap dung dau ":" (khong dau "," de tranh cat vun cau menh lenh).
# Cac cau viet lai yeu cau ("Analyze the request: ...") da bi cat o buoc 0.
_ANALYSIS_INLINE_RE = re.compile(
    r"^\s*analy[sz](?:e|ing)?\s+(?:the|this|your)\s+.+?\s*:\s*(.*?)\s*$",
    re.IGNORECASE,
)


def _strip_leading_narration(line: str) -> str:
    """Cat cac cau throat-clearing mo dau dong, giu phan con lai."""
    while True:
        m = _NARRATION_OPENERS_RE.match(line)
        if not m:
            return line
        rest = line[m.end():].strip()
        if not rest:
            return line
        line = rest


def _strip_narration_lines(text: str) -> str:
    """Cat cau narration mo dau moi dong (ngoai code fence), giu cau truc dong."""
    out: List[str] = []
    in_fence = False
    for raw in text.splitlines():
        if raw.strip().startswith("```"):
            in_fence = not in_fence
            out.append(raw)
            continue
        if not in_fence:
            raw = _strip_leading_narration(raw)
        out.append(raw)
    return "\n".join(out)


# Doan reasoning provider gui kem (neu nemotron tra ca trace lan dap an
# trong content): xoa ban sao verbatim DAI de khoi nhan doi suy nghi.
# Nguong 100 ky tu: cau ngan trung hop khong bao gio bi cat nham.
MIN_ECHO_CHARS = 100


def _strip_reasoning_echo(text: str, reasoning: str = "") -> str:
    """Xoa trace reasoning neu provider paste verbatim vao content."""
    needle = (reasoning or "").strip()
    if len(needle) >= MIN_ECHO_CHARS and needle in text:
        text = text.replace(needle, "")
    return text


def clean_answer(text: str, reasoning: str = "") -> str:
    """Loai sach phan verbose con sot de cau tra loi cuoi giong LLM pho thong:
    chi con noi dung tra loi thoi.

    - reasoning: trace rieng kenh provider gui kem; neu no bi paste verbatim
      vao content thi xoa ban sao (chi ap dung doan >= 100 ky tu). Dong heading bao cao / tuong thuat ("Tong ket", "Here's a thinking process",
      "What I did", "Qua trinh...", "Issues to fix"...) -> bo di.
    - Tuong thuat tu phan tich ("Analyze user input", "identify core task",
      "Step 1: ...", "Let me analyze..."): cau viet lai yeu cau -> bo ca dong;
      dang heading (markdown/bold/danh so/ket thuc ":") -> bo ca dong;
      dang "Cau: phan con lai" noi ve noi dung that -> giu phan con lai.
    - Loi mo dau thua ("Dua tren cau tra loi cua ban, ...", "Cam on ban...! ..."):
      GIU LAI phan con lai cua dong, khong xoa ca dong.
    - Khoi echo Q&A cua tool ask_user ("Cau tra loi cua nguoi dung:" + cac dong
      "Q -> A" ke sau) -> bo ca khoi. Dong "->" o noi khac duoc giu nguyen.
    - Code fence (```...```) đi ngay sau heading "Ma trien khai / Implementation"
      là code thừa không ai yêu cầu -> bỏ luôn cả khối code đó. Code hợp lệ
      (không có heading báo cáo đi trước) được giữ nguyên đầy đủ.
    - Dong "Yeu cau: ..." -> bo di; dong "Cau tra loi: X" -> giu X.
    - Bao dam: khong bao gio tra ve rong khi dau vao con chu (chong loc qua tay).
    """
    if not text:
        return text
    text = _strip_reasoning_echo(text, reasoning)
    text = _THINK_BLOCK_RE.sub("", text)
    text = _strip_narration_lines(text)
    lines = text.splitlines()
    keep: List[str] = []
    i = 0
    in_fence = False
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            in_fence = not in_fence
        low = line.lower().strip()
        head = low.lstrip("#*-•· \t").rstrip(":")
        # 0. Cau viet lai yeu cau ("Analyze user input", "identify core task"...):
        #    khong mang thong tin moi -> bo ca dong truoc khi xu ly tiep.
        if _match_restated(_strip_list_marker(head)):
            i += 1
            continue
        # 1. Boc loi mo dau thua (co the long nhau) nhung GIU phan con lai.
        #    Moi vong boc ngan hon vong truoc nen luon dung.
        while True:
            m = _PREAMBLE_RE.match(line)
            if not m:
                break
            line = m.group(1).strip()
            if not line:
                break
        # 1b. Nhan phan tich nhieu tu + ":" -> giu phan sau ":".
        m = _ANALYSIS_INLINE_RE.match(line)
        if m:
            line = m.group(1).strip()
        if not line.strip():
            i += 1
            continue
        low = line.lower().strip()
        head = low.lstrip("#*-•· \t").rstrip(":")
        # 2. Khoi echo Q&A cua tool: header + cac dong "Q -> A" lien ke ngay sau.
        if head.startswith("câu trả lời của người dùng"):
            i += 1
            while i < len(lines) and lines[i].strip() and " -> " in lines[i]:
                i += 1
            continue
        if any(head.startswith(m) for m in _VERBOSE_HEADINGS):
            is_code_heading = head.startswith("mã triển khai") or head.startswith("implementation")
            i += 1
            if is_code_heading:
                # Bỏ dòng trống + cả khối fence đi kèm (nếu có)
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines) and lines[i].strip().startswith("```"):
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("```"):
                        i += 1
                    if i < len(lines):
                        i += 1  # dòng ``` đóng
            continue
        # 3b. Tường thuật tự phân tích dạng heading -> bỏ cả dòng.
        if _is_analysis_heading(line, head):
            i += 1
            continue
        # 3c. Câu narration ngắn về chính yêu cầu ("I need to analyze the
        #     request.") -> bỏ cả dòng. Bỏ qua trong code fence.
        if not in_fence and _is_bare_narration(line):
            i += 1
            continue
        if head.startswith("câu trả lời") or head.startswith("trả lời"):
            keep.append(line.split(":", 1)[1].strip() if ":" in line else line)
            i += 1
            continue
        if head.startswith("kết quả"):
            # "Kết quả: X" -> giu X; "Kết quả" dung mot minh -> heading bao cao, bo.
            if ":" in line:
                rest = line.split(":", 1)[1].strip()
                if rest:
                    keep.append(rest)
            i += 1
            continue
        if head.startswith("yêu cầu"):
            i += 1
            continue
        keep.append(line)
        i += 1
    out = "\n".join(keep).strip()
    # Chong loc qua tay: neu loc xong khong con gi thi tra ve ban goc (strip).
    return out if out else text.strip()


@dataclass
class SubTask:
    id: str
    title: str
    goal: str
    specialist: str = ""  # coder | reviewer | researcher (rong = worker thuong)
    output: str = ""


@dataclass
class SwarmResult:
    answer: str
    subtasks: List[SubTask] = field(default_factory=list)
    critique: str = ""
    rounds: int = 0


# Tri nho hoi thoai: so exchange + gioi han ky tu de khong pha context.
HISTORY_EXCHANGES = 6
HISTORY_MAX_CHARS = 4000
HISTORY_ANSWER_CHARS = 800


def format_history(history) -> str:
    """Goi lich su [(q, a), ...] thanh khoi text gan vao prompt.
    Rut gon dap an dai, cat tong theo gioi han. Tra '' neu khong co lich su."""
    if not history:
        return ""
    parts: List[str] = []
    for q, a in history[-HISTORY_EXCHANGES:]:
        q = (q or "").strip()
        a = (a or "").strip()
        if not q and not a:
            continue
        if len(a) > HISTORY_ANSWER_CHARS:
            a = a[:HISTORY_ANSWER_CHARS] + "… (rut gon)"
        parts.append(f"Q: {q}\nA: {a}")
    block = "\n\n".join(parts).strip()
    if len(block) > HISTORY_MAX_CHARS:
        block = block[-HISTORY_MAX_CHARS:]
    if not block:
        return ""
    return (
        "LICH SU HOI DAP GAN DAY (de hieu cac tu nhu \"cho do\", \"cach khac\", "
        "\"sua lai\"):\n" + block
    )


_CONTEXT_ERROR_HINTS = (
    "context length",
    "maximum context",
    "context_length",
    "too many tokens",
    "input too large",
    "prompt too long",
)


def _is_context_error(exc: Exception) -> bool:
    """Nhan dien loi tran context cua LLM de tu thu lai gon hon."""
    s = str(exc).lower()
    return any(h in s for h in _CONTEXT_ERROR_HINTS)


def extract_json(text: str):
    """Trich JSON tu phan hoi LLM (xu ly ca truong hop boc trong code fence)."""
    candidates = [text.strip()]
    candidates += [m.group(1).strip() for m in JSON_FENCE_RE.finditer(text)]
    for match in re.finditer(r"[\[{]", text):
        candidates.append(text[match.start():].strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


class SwarmOrchestrator:
    """Lien ket chat: Planner tach viec, Worker lam, Critic gop y, Judge chot."""

    WORKER_STYLES = ["bold cyan", "bold magenta", "bold blue", "bold green"]

    def __init__(self, client: LLMClient, num_workers: int = 2, tools=None) -> None:
        self.client = client
        self.tools = tools  # ToolRegistry hoac None (khong dung tool)
        self.planner = Agent("Planner", ">", "bold yellow", PLANNER_SYSTEM, temperature=0.3)
        self.critic = Agent("Critic", "?", "bold red", CRITIC_SYSTEM, temperature=0.2)
        self.refiner = Agent("Refiner", "~", "bold green", REFINER_SYSTEM, temperature=0.5)
        self.judge = Agent("Judge", "+", "bold white", JUDGE_SYSTEM, temperature=0.6)
        self.workers = self._make_workers(num_workers)
        # Chuyen gia dung chung: planner gan subtask theo chuyen mon.
        # Ngoai pool workers (giu agents() on dinh cho footer), nen UI cu van dung.
        self.specialists = {
            "coder": Agent("Coder", "#", "bold cyan", CODER_SYSTEM, temperature=0.7),
            "reviewer": Agent("Reviewer", "=", "bold magenta", REVIEWER_SYSTEM, temperature=0.3),
            "researcher": Agent("Researcher", "@", "bold blue", RESEARCHER_SYSTEM, temperature=0.5),
        }
        self.on_event = None  # callback(event: str, data) de UI hien thi
        self.changes = None          # ChangeTracker cua luot chay (UI dua vao)
        self.ask_callback = None     # callable(questions) -> answers (tool ask_user)
        self.confirm_callback = None  # callable(prompt) -> bool (duyet lenh nguy hiem)
        self.last_clean: dict = {}   # thong ke loc lan cuoi: raw/final chars
        self.last_echo_repair: bool = False  # True neu luot cuoi phai repair echo-dup
        self.last_tool_qa: List[Tuple[str, str]] = []  # hoi-dap da xong qua tool trong luot cuoi

    # --------------------------------------------------------------- helpers
    def _emit(self, event: str, data=None) -> None:
        if self.on_event:
            self.on_event(event, data)

    def _make_workers(self, num_workers: int) -> List[Agent]:
        return [
            Agent(
                f"Worker-{i + 1}",
                "*",
                self.WORKER_STYLES[i % len(self.WORKER_STYLES)],
                WORKER_SYSTEM,
                temperature=0.7,
            )
            for i in range(max(1, num_workers))
        ]

    def set_num_workers(self, n: int) -> None:
        self.workers = self._make_workers(n)
        self.apply_models()

    def apply_models(self, cfg=None) -> None:
        """Doc config va gan model rieng cho tung agent (ke ca w1/w2 rieng).

        Key config: planner_model / worker_model / worker1_model /
        worker2_model / critic_model / refiner_model / judge_model.
        De trong = dung model chinh; w1/w2 uu hon worker_model chung.
        Key cu synthesizer_model van duoc doc lam fallback cho judge.
        cfg tùy chọn: neu truyen thi dung cfg do (khong can client.config).
        """
        if cfg is None:
            cfg = getattr(self.client, "config", None)
            if cfg is None:
                return
        self.planner.model = str(cfg.get("planner_model") or "")
        self.critic.model = str(cfg.get("critic_model") or "")
        self.refiner.model = str(cfg.get("refiner_model") or "")
        self.judge.model = str(cfg.get("judge_model") or cfg.get("synthesizer_model") or "")
        worker_model = str(cfg.get("worker_model") or "")
        w1 = str(cfg.get("worker1_model") or "") or worker_model
        w2 = str(cfg.get("worker2_model") or "") or worker_model
        for i, w in enumerate(self.workers):
            if i == 0:
                w.model = w1
            elif i == 1:
                w.model = w2
            else:
                w.model = worker_model
        for spec in self.specialists.values():
            spec.model = worker_model  # chuyen gia dung chung worker_model

    def models_summary(self) -> List[str]:
        """Mo ta model dang dung cua tung agent (de /.model hien thi)."""
        cfg = getattr(self.client, "config", None)
        main = str(cfg.get("model")) if cfg else "?"
        rows = [f"main: {main}"]
        pairs = [("Planner", self.planner.model)]
        for i, w in enumerate(self.workers[:2], start=1):
            pairs.append((f"Worker-{i} (w{i})", w.model))
        pairs += [
            ("Critic", self.critic.model),
            ("Refiner", self.refiner.model),
            ("Judge", self.judge.model),
        ]
        for name, model in pairs:
            rows.append(f"{name}: {model or main}{' (custom)' if model else ''}")
        for key in ("coder", "reviewer", "researcher"):
            spec = self.specialists[key]
            rows.append(f"{spec.name} (specialist): {spec.model or main}{' (custom)' if spec.model else ''}")
        return rows

    def find_agent(self, name: str) -> Optional[Agent]:
        """Tim agent theo ten trong pool + chuyen gia (cho UI lookup)."""
        for a in self.agents():
            if a.name == name:
                return a
        return self.specialists.get((name or "").lower())

    def agents(self) -> List[Agent]:
        return [self.planner, *self.workers, self.critic, self.refiner, self.judge]

    def _cfg_int(self, key: str, default: int) -> int:
        """Doc so nguyen tu config, an toan khi client gia khong co config."""
        cfg = getattr(self.client, "config", None)
        try:
            v = int(cfg.get(key, default)) if cfg is not None else default
            return v if v > 0 else default
        except (TypeError, ValueError, AttributeError):
            return default

    def _plan(self, user_input: str, history=None, conventions="", todos=None, skills="") -> List[SubTask]:
        self.apply_models()  # nap model moi nhat cho tung role truoc khi chay
        self._emit("phase", ("Planner", "planning"))
        plan_input = user_input
        hist_block = format_history(history)
        if hist_block:
            plan_input = f"{user_input}\n\n---\n{hist_block}"
        if conventions:
            plan_input = f"{plan_input}\n\n---\n{conventions}"
        if skills:
            plan_input = f"{plan_input}\n\n---\n{skills}"
        todo_block = format_todos(todos)
        if todo_block:
            plan_input = f"{plan_input}\n\n---\n{todo_block}"
        raw = self.planner.run(self.client, plan_input)
        data = extract_json(raw)
        subtasks: List[SubTask] = []
        if isinstance(data, list):
            for i, item in enumerate(data[:5], start=1):
                if isinstance(item, dict):
                    spec = str(item.get("specialist", "") or "").strip().lower()
                    subtasks.append(
                        SubTask(
                            id=str(item.get("id", f"t{i}")),
                            title=str(item.get("title", f"Subtask {i}")),
                            goal=str(item.get("goal", "")),
                            specialist=spec if spec in self.specialists else "",
                        )
                    )
                elif isinstance(item, str):
                    subtasks.append(SubTask(id=f"t{i}", title=item, goal=item))
        if not subtasks:  # fallback: lam truc tiep 1 subtask
            subtasks = [SubTask(id="t1", title="Answer directly", goal=user_input)]
        self._emit("plan", subtasks)
        return subtasks

    def _execute_workers(self, user_input: str, subtasks: List[SubTask], history=None, conventions="", todos=None, skills="") -> None:
        """Workers chay SONG SONG (ThreadPoolExecutor): moi subtask doc lap.
        Khong con context chuyen tay nhu chay tuan tu — moi worker chi nhan
        yeu cau goc + subtask cua minh (Planner da tach thanh viec doc lap)."""
        import concurrent.futures
        import threading

        ask_cb = self.ask_callback
        ask_lock = threading.Lock()
        hist_block = format_history(history)
        todo_block = format_todos(todos)
        confirm_cb = self.confirm_callback
        confirm_lock = threading.Lock()
        # Hoi-dap DA XONG qua tool trong luot nay (de judge ket luan, khong hoi lai).
        self.last_tool_qa = []
        for _a in list(self.workers) + list(self.specialists.values()):
            try:
                _a.tool_qa = []
            except Exception:  # noqa: BLE001
                pass
        qa_lock = threading.Lock()

        def locked_ask(questions):
            if ask_cb is None:
                return []
            with ask_lock:  # 1 modal hoi-dap tai 1 thoi diem
                return ask_cb(questions)

        ask_arg = locked_ask if ask_cb is not None else None

        def locked_confirm(prompt_text):
            if confirm_cb is None:
                return True  # non-interactive: mac dinh cho chay
            with confirm_lock:  # 1 modal confirm tai 1 thoi diem
                try:
                    return bool(confirm_cb(prompt_text))
                except Exception:  # noqa: BLE001
                    return False

        confirm_arg = locked_confirm
        tool_cap = self._cfg_int("tool_output_max_chars", 8000)
        tool_rounds = self._cfg_int("max_tool_rounds", 8)

        # Bang chung cho moi worker (shared board mini): moi nguoi biet
        # toan cuc de khong chong lan viec cua nhau.
        board = ""
        if len(subtasks) > 1:
            rows = "; ".join(
                f"{s.id} {s.title}" + (f" [{s.specialist}]" if s.specialist else "")
                for s in subtasks
            )
            board = f"\n\n---\nTOAN CUC ({len(subtasks)} viec song song, moi nguoi 1 viec khac nhau):\n{rows}"

        def do_one(pair) -> None:
            i, subtask = pair
            worker = self.specialists.get(subtask.specialist) or self.workers[i % len(self.workers)]
            self._emit("phase", (worker.name, subtask.title))
            task = (
                f"Yeu cau goc: {user_input}\n"
                f"Subtask {subtask.id} - {subtask.title}: {subtask.goal}"
            )
            if board:
                task += board
            if hist_block:
                task += f"\n\n---\n{hist_block}"
            if conventions:
                task += f"\n\n---\n{conventions}"
            if skills:
                task += f"\n\n---\n{skills}"
            if todo_block:
                task += f"\n\n---\n{todo_block}"
            tparts: List[str] = []

            def _wt(piece: str) -> None:
                tparts.append(piece)
                self._emit("think_delta", piece)

            if self.tools is not None:
                subtask.output = worker.run_with_tools(
                    self.client, task, self.tools, "", on_event=self._emit,
                    changes=self.changes, ask_user=ask_arg, confirm=confirm_arg,
                    tool_output_chars=tool_cap, max_tool_rounds=tool_rounds,
                    on_thinking=_wt,
                )
            else:
                subtask.output = worker.run(
                    self.client, task, "",
                    on_thinking=_wt,
                )
            subtask.output = _strip_reasoning_echo(subtask.output, "".join(tparts))
            try:
                got_qa = list(getattr(worker, "tool_qa", None) or [])
            except Exception:  # noqa: BLE001
                got_qa = []
            if got_qa:
                with qa_lock:  # giu thu tu, khong trung cap
                    for pair in got_qa:
                        if pair not in self.last_tool_qa:
                            self.last_tool_qa.append(pair)
            self._emit("worker_done", (worker.name, subtask.id, subtask.output))

        if len(subtasks) <= 1:
            for pair in enumerate(subtasks):
                do_one(pair)
            return
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(self.workers), len(subtasks))
        ) as pool:
            list(pool.map(do_one, enumerate(subtasks)))

    def _critique(self, user_input: str, subtasks: List[SubTask]) -> dict:
        self._emit("phase", ("Critic", "reviewing"))
        draft = "\n\n".join(f"[{s.id}] {s.title}:\n{s.output}" for s in subtasks)
        task = f"Yeu cau goc: {user_input}\n\nBan nhap:\n{draft}"
        raw = self.critic.run(self.client, task)
        data = extract_json(raw)
        if not isinstance(data, dict):
            data = {"verdict": "APPROVED"}  # khong parse duoc -> coi nhu dat
        self._emit("critique", data)
        return data

    def _refine(self, critique: dict, subtasks: List[SubTask]) -> None:
        self._emit("phase", ("Refiner", "revising"))
        draft = "\n\n".join(f"[{s.id}] {s.title}:\n{s.output}" for s in subtasks)
        issues = "\n".join(f"- {i}" for i in critique.get("issues", []))
        task = (
            f"Ban nhap cu:\n{draft}\n\nCac issue can khac phuc:\n{issues}\n\n"
            f"Huong dan: {critique.get('guidance', '')}"
        )
        refined = self.refiner.run(self.client, task)
        subtasks[0].output = refined  # judge dung ban da sua
        self._emit("worker_done", (self.refiner.name, "refined", refined))

    def _tool_qa_suffix(self) -> str:
        """Block DA-HOI cho judge: hoi-dap da xong qua tool thi chi ket luan.

        Tra '' neu luot nay khong co Q&A. Dung chung cho ca luot judge chinh
        va luot repair (repair truoc day thieu block nay nen model hoi lai
        bang template Q1/A1 rong).
        """
        if not self.last_tool_qa:
            return ""
        try:
            qa_lines = "\n".join(
                f"- {q} -> {a}" for q, a in self.last_tool_qa[:10]
            )
        except Exception:  # noqa: BLE001
            return ""
        if not qa_lines:
            return ""
        return (
            f"\n\n---\nĐÃ HỎI QUA TOOL (xong, KHÔNG hỏi lại dưới mọi dạng, "
            f"kể cả Q1/A1):\n{qa_lines}\n"
            f"Dùng đáp án trên để KẾT LUẬN (đoán/kết quả/quyết định), chỉ in kết luận."
        )

    def _synthesize(self, user_input: str, subtasks: List[SubTask], history=None, conventions="", todos=None, skills="") -> str:
        self._emit("phase", ("Judge", "judging answer"))
        inputs = "\n\n".join(f"[{s.id}] {s.title}:\n{s.output}" for s in subtasks)
        task = (
            f"Yeu cau goc: {user_input}\n\nCac buoc nhap:\n{inputs}\n\n"
            f"Yeu cau: tra loi TRUC TIEP cau hoi goc, dung trong tam. "
            f"Nhap dai lan man/lap y -> NEN lai giu y chinh. "
            f"Bo cam on mo dau/tuong thuat/echo Q&A; chi giu KET QUA."
        )
        if self.last_tool_qa:
            # Worker DA HOI user qua tool va co dap an: judge chi KET LUAN,
            # tuyet doi khong liet ke lai cau hoi (loi hoi-lai-bang-text).
            task += self._tool_qa_suffix()
        hist_block = format_history(history)
        if hist_block:
            task += f"\n\n---\n{hist_block}"
        if conventions:
            task += f"\n\n---\n{conventions}"
        if skills:
            task += f"\n\n---\n{skills}"
        todo_block = format_todos(todos)
        if todo_block:
            task += f"\n\n---\n{todo_block}"
        think_parts: List[str] = []

        def _think(piece: str) -> None:
            think_parts.append(piece)
            self._emit("think_delta", piece)

        def _synth_delta(piece: str) -> None:
            self._emit("synth_delta", piece)

        raw = self.judge.run(
            self.client, task,
            on_delta=_synth_delta,
            on_thinking=_think,
        )
        stripped = _strip_reasoning_echo(raw, "".join(think_parts))
        self.last_echo_repair = False
        if not stripped.strip() and (raw.strip() or "".join(think_parts).strip()):
            # Provider nhan doi trace vao content: echo nuot HET dap an
            # (case that 7281 chars). Repair DUY NHAT 1 lan voi thinking=off
            # de co dap an that thay vi tra ve rong.
            repair_task = (
                f"Yeu cau goc: {user_input}\n\nCac buoc nhap:\n{inputs}\n\n"
                f"Yeu cau: tra loi TRUC TIEP cau hoi goc, chi in DAP AN cuoi, "
                f"tuyet doi khong tuong thuat qua trinh suy nghi."
                f"{self._tool_qa_suffix()}"
            )
            rparts: List[str] = []

            def _rthink(piece: str) -> None:
                rparts.append(piece)
                self._emit("think_delta", piece)

            rraw = self.judge.run(
                self.client, repair_task,
                on_delta=_synth_delta,
                on_thinking=_rthink,
                thinking="off",
            )
            stripped = _strip_reasoning_echo(rraw, "".join(rparts))
            self.last_echo_repair = True
        return stripped

# -------------------------------------------------------- quick reply
    def quick_reply(self, user_input: str, on_delta=None, history=None, on_thinking=None) -> str:
        """Hoi dap nhanh cho cau chao/hoi don gian: MOT luot chat truc tiep,
        khong qua swarm, khong tool, stream qua on_delta. Dau ra chi la loi dap."""
        messages = [
            {"role": "system", "content": QUICK_SYSTEM},
        ]
        for q, a in (history or [])[-4:]:
            q, a = (q or "").strip(), (a or "").strip()
            if not q and not a:
                continue
            messages.append({"role": "user", "content": q})
            messages.append({
                "role": "assistant",
                "content": a[:HISTORY_ANSWER_CHARS] if len(a) > HISTORY_ANSWER_CHARS else a,
            })
        messages.append({"role": "user", "content": user_input})
        think_parts: List[str] = []

        def _qt(piece: str) -> None:
            think_parts.append(piece)
            if on_thinking is not None:
                on_thinking(piece)

        answer = self.client.chat(
            messages,
            temperature=0.8,
            max_tokens=self.judge.max_tokens,
            on_delta=on_delta,
            on_thinking=_qt,
        )
        think_all = "".join(think_parts)
        stripped = _strip_reasoning_echo(answer, think_all)
        final_reasoning = think_all
        self.last_echo_repair = False
        if not stripped.strip() and (answer.strip() or think_all.strip()):
            # Echo nuot het (giong case swarm) -> repair 1 lan thinking off.
            rparts: List[str] = []

            def _rt(piece: str) -> None:
                rparts.append(piece)
                if on_thinking is not None:
                    on_thinking(piece)

            answer = self.client.chat(
                messages,
                temperature=0.8,
                max_tokens=self.judge.max_tokens,
                on_delta=on_delta,
                on_thinking=_rt,
                thinking="off",
            )
            final_reasoning = "".join(rparts)
            answer = _strip_reasoning_echo(answer, final_reasoning)
            self.last_echo_repair = True
        self.last_clean = {"raw_chars": len(answer)}
        answer = clean_answer(answer, reasoning=final_reasoning)
        self.last_clean["final_chars"] = len(answer)
        self.last_clean["echo_repair"] = self.last_echo_repair
        return answer

    # ------------------------------------------------------------------ main
    # ------------------------------------------------------------------ main
    def run(self, user_input: str, max_rounds: int = 1, history=None, conventions=None, todos=None, skills=None) -> SwarmResult:
        """Chay full swarm. history: [(cau hoi, tra loi), ...] gan nhat de giu
        ngu canh hoi thoai (vd "sua lai cho do"). None = khong co lich su.
        conventions: khoi QUY UOC REPO gan vao prompt; None = tu nap AGENTS.md.
        todos: checklist dang do [{text, done}] gan vao prompt.
        skills: khoi KY NANG gan vao prompt; None = tu nap .skills/.
        Tran context -> tu thu lai 1 lan, bo lich su + conventions + todos."""
        if conventions is None:
            try:
                conventions = load_conventions()
            except Exception:  # noqa: BLE001
                conventions = ""
        if skills is None:
            try:
                skills = load_skills()
            except Exception:  # noqa: BLE001
                skills = ""
        try:
            return self._run_once(user_input, max_rounds, history, conventions, todos, skills)
        except LLMError as exc:
            if (history or conventions or todos) and _is_context_error(exc):
                return self._run_once(user_input, max_rounds, None, "", None, "")
            raise

    def _run_once(self, user_input: str, max_rounds: int = 1, history=None, conventions="", todos=None, skills="") -> SwarmResult:
        subtasks = self._plan(user_input, history, conventions, todos, skills)
        self._execute_workers(user_input, subtasks, history, conventions, todos, skills)

        rounds = 0
        critique: dict = {}
        for _ in range(max(0, max_rounds)):
            critique = self._critique(user_input, subtasks)
            if str(critique.get("verdict", "")).upper() == "APPROVED":
                break
            rounds += 1
            self._refine(critique, subtasks)

        answer = self._synthesize(user_input, subtasks, history, conventions, todos, skills)
        self.last_clean = {"raw_chars": len(answer)}
        answer = clean_answer(answer)
        self.last_clean["final_chars"] = len(answer)
        self.last_clean["echo_repair"] = self.last_echo_repair
        return SwarmResult(answer=answer, subtasks=subtasks, critique=str(critique), rounds=rounds)

