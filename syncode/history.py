"""Luu/doc lich su hoi dap theo SESSION (cuoc tro chuyen) ra ~/.syncode/history.json."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from syncode.config import CONFIG_DIR

MAX_SESSIONS = 50
MAX_EXCHANGES_PER_SESSION = 100

# Compact: so lượt gần nhất giữ nguyên, còn lại gom thành tóm tắt.
COMPACT_KEEP_DEFAULT = 2
COMPACT_MIN_SUMMARIZE = 2  # ít hơn số này lượt cũ thì không đáng compact
COMPACT_SUMMARY_MAX_CHARS = 2000  # trần cứng để tóm tắt không tự phình
COMPACT_EXCHANGE_CHARS = 600  # cắt mỗi đáp án cũ khi đưa vào prompt tóm tắt

# Marker cau hoi cho tuple tom tat trong compacted_history (hien nhu 1 cap
# Q/A dau tien de ca format_history lan quick_reply deu hieu, khong doi
# chu ky ham orchestrator).
COMPACT_SUMMARY_QUESTION = "[Tóm tắt phiên trước — đã compact, chỉ giữ ý chính]"


@dataclass
class Exchange:
    """Mot cap question/answer trong session."""
    question: str
    answer: str

    def to_dict(self) -> dict:
        return {"q": self.question, "a": self.answer}

    @classmethod
    def from_dict(cls, d: dict) -> Exchange:
        return cls(question=str(d.get("q", "")), answer=str(d.get("a", "")))


@dataclass
class Session:
    """Mot cuoc tro chuyen (session) — bao gom nhieu exchange."""
    id: str
    created_at: float
    exchanges: List[Exchange] = field(default_factory=list)
    todos: List[Dict[str, Any]] = field(default_factory=list)  # [{text, done}]
    compact_summary: str = ""  # tom tat cac luot da compact (file cu thieu key -> "")

    @property
    def title(self) -> str:
        """Tieu de hien thi — cau hoi dau tien cua session."""
        if self.exchanges:
            q = self.exchanges[0].question.strip()
            return q if len(q) <= 50 else q[:50] + "..."
        return "(empty)"

    @property
    def summary(self) -> str:
        """Tom tat session: so luot, thoi gian."""
        n = len(self.exchanges)
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created_at))
        return f"{n} turns · {ts}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "exchanges": [e.to_dict() for e in self.exchanges],
            "todos": [
                {"text": str(t.get("text", "")), "done": bool(t.get("done", False))}
                for t in self.todos
                if isinstance(t, dict)
            ],
            "compact_summary": self.compact_summary or "",
        }

    @classmethod
    def from_dict(cls, d: dict) -> Session:
        exchanges = [Exchange.from_dict(e) for e in d.get("exchanges", [])]
        todos = []
        for t in d.get("todos", []) or []:
            if isinstance(t, dict) and str(t.get("text", "")).strip():
                todos.append({"text": str(t["text"]), "done": bool(t.get("done", False))})
        return cls(
            id=str(d.get("id", "")),
            created_at=float(d.get("created_at", 0)),
            exchanges=exchanges,
            todos=todos,
            compact_summary=str(d.get("compact_summary", "") or ""),
        )


def load_sessions(path: Path | None = None) -> List[Session]:
    """Doc danh sach session tu file JSON."""
    p = path or CONFIG_DIR / "history.json"
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            # format moi: {"sessions": [...]}
            sessions_raw = raw.get("sessions", [])
        elif isinstance(raw, list):
            # format cu: list cac (q, a) — chuyen doi sang 1 session
            if raw and isinstance(raw[0], (list, tuple)):
                session = Session(
                    id=str(int(time.time())),
                    created_at=time.time(),
                    exchanges=[Exchange(question=q, answer=a) for q, a in raw],
                )
                return [session]
            sessions_raw = raw
        else:
            return []
        return [Session.from_dict(s) for s in sessions_raw if isinstance(s, dict)]
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return []


def save_sessions(sessions: List[Session], path: Path | None = None) -> None:
    """Ghi danh sach session (toi da MAX_SESSIONS) vao file JSON."""
    p = path or CONFIG_DIR / "history.json"
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        # Chi giu session co it nhat 1 exchange
        valid = [s for s in sessions if s.exchanges][-MAX_SESSIONS:]
        data = {"sessions": [s.to_dict() for s in valid]}
        p.write_text(
            json.dumps(data, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    except OSError:
        pass


def add_exchange_to_session(
    sessions: List[Session],
    question: str,
    answer: str,
    current_session_id: str | None = None,
) -> Tuple[List[Session], str]:
    """Them exchange vao session hien tai, hoac tao session moi."""
    exchange = Exchange(question=question, answer=answer)

    if current_session_id:
        for s in sessions:
            if s.id == current_session_id:
                s.exchanges.append(exchange)
                # gioi han so exchange trong 1 session
                if len(s.exchanges) > MAX_EXCHANGES_PER_SESSION:
                    s.exchanges = s.exchanges[-MAX_EXCHANGES_PER_SESSION:]
                return sessions, current_session_id

    # tao session moi
    new_session = Session(
        id=str(int(time.time() * 1000)),
        created_at=time.time(),
        exchanges=[exchange],
    )
    sessions.append(new_session)
    return sessions, new_session.id


def recent_exchanges(
    sessions: List[Session],
    session_id: str | None,
    limit: int = 6,
) -> List[Tuple[str, str]]:
    """Lay N cap (cau hoi, tra loi) gan nhat cua session hien tai lam tri nho
    hoi thoai. Tra [] neu khong co session/khong co lich su."""
    if not session_id or limit <= 0:
        return []
    for s in sessions:
        if s.id == session_id:
            return [
                (e.question, e.answer)
                for e in s.exchanges[-limit:]
                if e.question.strip() or e.answer.strip()
            ]
    return []


def open_todos(sessions: List[Session], session_id: str | None) -> List[Dict[str, Any]]:
    """Todo chua xong cua session hien tai (checkpoint cho luot sau)."""
    if not session_id:
        return []
    for s in sessions:
        if s.id == session_id:
            return [t for t in s.todos if isinstance(t, dict) and not t.get("done")]
    return []


def get_session_summary(sessions: List[Session], session_id: str | None) -> str:
    """Tom tat compact cua session hien tai ('' neu chua compact bao gio)."""
    if not session_id:
        return ""
    for s in sessions:
        if s.id == session_id:
            try:
                return str(getattr(s, "compact_summary", "") or "")
            except Exception:  # noqa: BLE001
                return ""
    return ""


def compacted_history(
    sessions: List[Session],
    session_id: str | None,
    limit: int = 6,
) -> List[Tuple[str, str]]:
    """Tri nho hoi thoai TIET KIEM context: tom tat compact (neu co) + N luot
    gan nhat. Thay the truc tiep recent_exchanges o app/cli — orchestrator
    khong can doi gi (ca format_history lan quick_reply deu an tuple Q/A)."""
    hist = recent_exchanges(sessions, session_id, limit)
    summary = get_session_summary(sessions, session_id).strip()
    if summary:
        return [(COMPACT_SUMMARY_QUESTION, summary)] + hist
    return hist


COMPACT_SYSTEM = """Bạn chỉ tóm tắt phiên trò chuyện, không trò chuyện tiếp.
Đầu ra CHỈ là tóm tắt tiếng Việt, gọn (<= 1200 ký tự), gạch đầu dòng theo mục:
- Việc đã làm & quyết định đã chốt
- File đã đọc/sửa (đường dẫn + ý chính thay đổi)
- Lỗi đã gặp & cách đã sửa
- Việc còn dở / cần làm tiếp
Cấm mở bài, cấm tường thuật quá trình suy nghĩ, không thêm gì ngoài tóm tắt."""


def build_compact_prompt(
    older: List[Tuple[str, str]],
    old_summary: str = "",
) -> str:
    """Dung prompt tóm tắt cho /.compact tu cac luot cu (+ tom tat cu de gop).

    Thuan tuy (khong goi LLM) de test duoc. Dap an dai bi cat ngan truoc khi
    dua vao — tom tat dau vao chu khong nem ca session vao model.
    """
    parts: List[str] = []
    if (old_summary or "").strip():
        parts.append("TÓM TẮT CŨ (gộp vào, không bỏ ý):\n" + old_summary.strip())
    rows = []
    for q, a in older or []:
        q = (q or "").strip()
        a = (a or "").strip()
        if len(a) > COMPACT_EXCHANGE_CHARS:
            a = a[:COMPACT_EXCHANGE_CHARS] + "… (cắt ngắn)"
        if q or a:
            rows.append(f"Q: {q}\nA: {a}")
    parts.append("CÁC LƯỢT CŨ CẦN TÓM TẮT:\n" + "\n\n".join(rows))
    return "\n\n---\n".join(p for p in parts if p)


def apply_compact(session: Session, summary: str, keep: int = COMPACT_KEEP_DEFAULT) -> dict:
    """Cat session chi giu `keep` luot gan nhat + luu tom tat. Thuan tuy
    (truoc khi save file) de test duoc. Tra stats {dropped, kept, freed_chars}."""
    summary = (summary or "").strip()
    if len(summary) > COMPACT_SUMMARY_MAX_CHARS:
        summary = summary[:COMPACT_SUMMARY_MAX_CHARS] + "…"
    exchanges = list(getattr(session, "exchanges", None) or [])
    keep = max(1, int(keep or COMPACT_KEEP_DEFAULT))
    dropped = exchanges[: max(0, len(exchanges) - keep)]
    freed = sum(len((e.question or "")) + len((e.answer or "")) for e in dropped)
    try:
        session.compact_summary = summary
        session.exchanges = exchanges[len(dropped):]
    except Exception:  # noqa: BLE001
        pass
    return {"dropped": len(dropped), "kept": len(session.exchanges), "freed_chars": freed}


def format_todos(todos) -> str:
    """Goi checklist dang do thanh khoi text gan vao prompt. '' neu het viec."""
    items = [t for t in (todos or []) if isinstance(t, dict) and not t.get("done")]
    if not items:
        return ""
    lines = "\n".join(f"- [ ] {t.get('text', '')}" for t in items[:10])
    return "CHECKLIST CON DANG DO (uu tien hoan thanh, dung mo viec moi):\n" + lines