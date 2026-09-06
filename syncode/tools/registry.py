"""Tool registry: builtin tools dạng JSON-schema (OpenAI function calling)."""

from __future__ import annotations

import ast
import difflib
import operator
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Tool:
    """Một tool: tên, mô tả, JSON schema tham số, hàm thực thi."""

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], str]
    source: str = "builtin"  # builtin | mcp:<server>
    dangerous: bool = False  # cần ghi file / chạy lệnh
    need_confirm: bool = False  # hoi user truoc khi chay (lenh shell, xoa file...)

    def openai_schema(self) -> Dict[str, Any]:
        """Chuyển sang dạng 'tools' của OpenAI chat completion."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ----------------------------------------------------------------- operators
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node: ast.AST) -> float:
    """Tính biểu thức số học an toàn (chỉ + - * / // % ** và ngoặc)."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Biểu thức không được phép")


def _tool_calculator(args: Dict[str, Any]) -> str:
    expression = str(args.get("expression", "")).strip()
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"{expression} = {result}"
    except Exception as exc:  # noqa: BLE001
        return f"Lỗi tính toán: {exc}"


def _tool_current_time(args: Dict[str, Any]) -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


def _tool_read_file(args: Dict[str, Any]) -> str:
    path = Path(args.get("path", "")).expanduser()
    limit = int(args.get("max_lines", 200))
    try:
        offset = int(args.get("offset", 0) or 0)
    except (TypeError, ValueError):
        return "Lỗi: offset phải là số"
    offset = max(0, offset)
    if not path.exists():
        return f"File không tồn tại: {path}"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Lỗi đọc file: {exc}"
    total = len(lines)
    if offset >= total:
        return f"(hết file: tổng {total} dòng, offset {offset})"
    head = lines[offset:offset + limit]
    body = "\n".join(head)
    rest = total - offset - len(head)
    if rest > 0:
        body += f"\n... (còn {rest} dòng nữa, tổng {total} dòng)"
    return body or "(file trống)"


def _tool_write_file(args: Dict[str, Any]) -> str:
    args = dict(args)
    tracker = args.pop("_changes", None)
    path = Path(args.get("path", "")).expanduser()
    content = str(args.get("content", ""))
    try:
        existed = path.exists()
        before = path.read_text(encoding="utf-8", errors="replace") if existed else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if tracker is not None:
            tracker.record_file(str(path), before, content)
        return f"Đã ghi {len(content)} ký tự vào {path}"
    except OSError as exc:
        return f"Lỗi ghi file: {exc}"


def _tool_list_dir(args: Dict[str, Any]) -> str:
    path = Path(args.get("path", ".")).expanduser()
    if not path.exists():
        return f"Thư mục không tồn tại: {path}"
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    rows = [f"{'📁' if e.is_dir() else '📄'} {e.name}" for e in entries[:100]]
    if not rows:
        return "(thư mục trống)"
    return "\n".join(rows)


def _tool_edit_file(args: Dict[str, Any]) -> str:
    """Sua 1 doan trong file (khop chinh xac). Mac dinh doi hoi duy nhat."""
    args = dict(args)
    tracker = args.pop("_changes", None)
    path = Path(args.get("path", "")).expanduser()
    old_text = str(args.get("old_text", ""))
    new_text = str(args.get("new_text", ""))
    replace_all = bool(args.get("replace_all", False))
    if not old_text:
        return "Lỗi: thiếu old_text"
    try:
        if not path.is_file():
            return f"File không tồn tại: {path}"
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Lỗi đọc file: {exc}"
    count = content.count(old_text)
    if count == 0:
        return "Lỗi: không tìm thấy old_text trong file (khớp chính xác, kể cả khoảng trắng)"
    if count > 1 and not replace_all:
        return (
            f"Lỗi: old_text xuất hiện {count} lần — hãy cho đoạn dài/duy nhất hơn, "
            f"hoặc đặt replace_all=true"
        )
    new_content = content.replace(old_text, new_text) if replace_all else content.replace(old_text, new_text, 1)
    n = count if replace_all else 1
    try:
        path.write_text(new_content, encoding="utf-8")
        if tracker is not None:
            tracker.record_file(str(path), content, new_content)
        return f"Đã thay thế {n} vị trí trong {path}"
    except OSError as exc:
        return f"Lỗi ghi file: {exc}"


def _tool_delete_file(args: Dict[str, Any]) -> str:
    args = dict(args)
    tracker = args.pop("_changes", None)
    path = Path(args.get("path", "")).expanduser()
    try:
        if not path.exists():
            return f"File không tồn tại: {path}"
        if path.is_dir():
            return f"Lỗi: {path} là thư mục (chỉ xóa file)"
        before = path.read_text(encoding="utf-8", errors="replace")
        path.unlink()
        if tracker is not None:
            tracker.record_file(str(path), before, "", deleted=True)
        return f"Đã xóa {path}"
    except OSError as exc:
        return f"Lỗi xóa file: {exc}"


_GREP_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".hg", ".svn"}
_GREP_MAX_FILES = 200
_GREP_MAX_HITS = 50
_GREP_MAX_BYTES = 1_000_000


def _tool_grep(args: Dict[str, Any]) -> str:
    pattern = str(args.get("pattern", ""))
    root = Path(args.get("path", ".")).expanduser()
    include = str(args.get("include", "") or "").strip()
    if not pattern:
        return "Lỗi: thiếu pattern"
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"Lỗi regex: {exc}"
    if not root.exists():
        return f"Đường dẫn không tồn tại: {root}"
    if root.is_file():
        files = [root]
    else:
        files = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in _GREP_SKIP_DIRS or part.startswith(".") for part in p.parts):
                continue
            if include and not Path(p.name).match(include):
                continue
            try:
                if p.stat().st_size > _GREP_MAX_BYTES:
                    continue
            except OSError:
                continue
            files.append(p)
            if len(files) >= _GREP_MAX_FILES:
                break
    hits: List[str] = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for no, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                hits.append(f"{p}:{no}: {line.strip()[:200]}")
                if len(hits) >= _GREP_MAX_HITS:
                    return "\n".join(hits) + f"\n... (đã cắt ở {_GREP_MAX_HITS} kết quả)"
    if not hits:
        return "(không tìm thấy)"
    return "\n".join(hits)


def _tool_glob(args: Dict[str, Any]) -> str:
    pattern = str(args.get("pattern", "") or "*").strip() or "*"
    root = Path(args.get("path", ".")).expanduser()
    if not root.exists():
        return f"Đường dẫn không tồn tại: {root}"
    base = root if root.is_dir() else root.parent
    try:
        hits = sorted(base.glob(pattern), key=lambda p: str(p).lower())[:100]
    except (OSError, ValueError) as exc:
        return f"Lỗi glob: {exc}"
    if not hits:
        return "(không tìm thấy)"
    rows = []
    for p in hits:
        try:
            rel = p.relative_to(base)
        except ValueError:
            rel = p
        rows.append(f"{rel}/" if p.is_dir() else str(rel))
    return "\n".join(rows)


def _tool_fetch_url(args: Dict[str, Any]) -> str:
    import html as _html
    import urllib.request

    url = str(args.get("url", "")).strip()
    try:
        max_chars = min(int(args.get("max_chars", 6000) or 6000), 20000)
    except (TypeError, ValueError):
        return "Lỗi: max_chars phải là số"
    if not url.startswith(("http://", "https://")):
        return "Lỗi: chỉ hỗ trợ http(s) URL"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "syncode/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(25000)
    except Exception as exc:  # noqa: BLE001
        return f"Lỗi tải URL: {exc}"
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", _html.unescape(text)).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "… (đã cắt)"
    return text or "(trang trống)"


def _parse_ddg(html: str, limit: int):
    """Tach (tieu de, url, mo ta) tu HTML DuckDuckGo. Don gian, khong phu thuoc."""
    import html as _html
    import urllib.parse

    def clean(s: str) -> str:
        s = re.sub(r"(?s)<[^>]+>", " ", s)
        return re.sub(r"\s+", " ", _html.unescape(s)).strip()

    links = re.findall(r'result__a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
    snips = re.findall(r'result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
    results = []
    for i, (href, title) in enumerate(links):
        if len(results) >= limit:
            break
        title = clean(title)
        if not title:
            continue
        link = href.strip()
        m = re.search(r"[?&]uddg=([^&]+)", link)  # link redirect cua DDG
        if m:
            try:
                link = urllib.parse.unquote(m.group(1))
            except Exception:  # noqa: BLE001
                pass
        if link.startswith("//"):
            link = "https:" + link
        snippet = clean(snips[i]) if i < len(snips) else ""
        results.append((title, link, snippet[:300]))
    return results


def _tool_web_search(args: Dict[str, Any]) -> str:
    """Tim kiem web qua DuckDuckGo (khong can API key)."""
    import urllib.parse
    import urllib.request

    query = str(args.get("query", "")).strip()
    try:
        raw_max = args.get("max_results", 5)
        max_results = min(int(5 if raw_max is None else raw_max), 10)
    except (TypeError, ValueError):
        return "Lỗi: max_results phải là số"
    if not query:
        return "Lỗi: thiếu query"
    if max_results <= 0:
        return "Lỗi: max_results phải > 0"
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read(200000).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"Lỗi tìm kiếm: {exc}"
    results = _parse_ddg(html, max_results)
    if not results:
        return "(không tìm thấy kết quả)"
    out = []
    for i, (title, link, snippet) in enumerate(results, start=1):
        row = f"{i}. {title}\n   {link}"
        if snippet:
            row += f"\n   {snippet}"
        out.append(row)
    return "\n".join(out)


# Phong bao chi dan kem theo ket qua hoi user: model phai dung dap an LANG LE,
# cam echo/tuong thuat — day la nguyen nhan chinh gay output verbose sau tool q&a-user.
# Dat chi dan QUAN TRONG NHAT len dau de song sot qua cat ngan output[:4000] o agent.py.
_QA_INTERNAL_HEAD = (
    "[NỘI BỘ — KHÔNG trích dẫn, KHÔNG nhắc lại: dùng câu trả lời dưới đây LẶNG LẼ "
    "để làm tiếp nhiệm vụ gốc. CẤM echo hỏi/đáp, CẤM cảm ơn, CẤM tường thuật quá trình "
    "hay suy nghĩ.]"
)
_QA_INTERNAL_TAIL = "[Hết phần nội bộ — trả lời TRỰC TIẾP nhiệm vụ gốc, gọn.]"


def _tool_ask_user(args: Dict[str, Any]) -> str:
    """Hỏi người dùng nhiều câu trong MỘT form, chờ trả lời rồi trả về agent.

    App gắn callables qua registry.execute(..., on_ask=..., emit=...):
    - _ask: callable(questions) -> list[str] answers (chặn đến khi user xong)
    - _emit: callback event để UI biết đang hỏi (vd mở modal)
    """
    ask = args.pop("_ask", None)
    emit = args.pop("_emit", None)
    questions = args.get("questions", [])
    if isinstance(questions, str):
        questions = [questions]
    questions = [str(q) for q in questions][:5]
    if not questions:
        return "Lỗi: không có câu hỏi nào"
    if emit is not None:
        try:
            emit("ask_user", questions)
        except Exception:  # noqa: BLE001
            pass
    if ask is None:
        return "Lỗi: hiện không hỏi được người dùng (giao diện không khả dụng)."
    try:
        answers = ask(questions)
    except Exception as exc:  # noqa: BLE001
        return f"Lỗi khi hỏi người dùng: {exc}"
    if not answers:
        answers = ["(không trả lời)"] * len(questions)
    answers = list(answers) + ["(không trả lời)"] * (len(questions) - len(answers))
    lines = [f"{q} -> {a}" for q, a in zip(questions, answers)]
    return f"{_QA_INTERNAL_HEAD}\nCâu trả lời của người dùng:\n" + "\n".join(lines) + f"\n{_QA_INTERNAL_TAIL}"


class ChangeTracker:
    """Ghi lại thay đổi file của một lượt chạy để UI hiển thị diff kiểu code-agent."""

    def __init__(self) -> None:
        self.files: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()  # workers chay song song

    def record_file(self, path: str, before: str, after: str, *, deleted: bool = False) -> None:
        with self._lock:
            if path not in self.files:
                # lan ghi dau tien = anh chup GOC cua file trong luot chay nay
                self.files[path] = {"before": before, "after": after, "deleted": False, "writes": 0}
            entry = self.files[path]
            entry["after"] = after
            if deleted:
                entry["deleted"] = True
                entry["after"] = ""
            entry["writes"] += 1

    def record_deleted_files(self, command: str) -> None:
        """Heuristic: lệnh rm — đánh dấu các file được nhắc trong lệnh là đã xóa."""
        if not re.search(r"\brm\b", command):
            return
        with self._lock:
            for raw in re.findall(r"(?<![\w./-])[\w./~-]+\.[A-Za-z0-9]{1,6}", command):
                p = str(Path(raw).expanduser())
                entry = self.files.setdefault(
                    p, {"before": "", "after": "", "deleted": False, "writes": 0}
                )
                entry["deleted"] = True
                entry["after"] = ""

    def added_removed(self) -> Tuple[int, int]:
        add = rem = 0
        for e in self.files.values():
            b, a = e["before"].splitlines(), e["after"].splitlines()
            add += max(0, len(a) - len(b))
            rem += max(0, len(b) - len(a))
        return add, rem

    def diff_summary(self) -> List[str]:
        rows = []
        for path, e in self.files.items():
            b, a = e["before"].splitlines(), e["after"].splitlines()
            plus = sum(1 for ln in a if ln not in b)
            minus = sum(1 for ln in b if ln not in a)
            if e["deleted"] and not e["after"]:
                tag = "xóa"
            elif not e["before"] and e["after"]:
                tag = "tạo mới"
            else:
                tag = "sửa"
            rows.append(f"{path} ({tag}, +{plus}/-{minus})")
        return rows

    def undo_all(self) -> List[str]:
        """Hoan tac moi thay doi file cua luot: tra noi dung goc, xoa file
        tao moi, dung lai file da xoa. Xoa dau vet sau khi undo de undo
        lan 2 la no-op. Khong bao gio raise."""
        with self._lock:
            items = list(self.files.items())
            self.files.clear()
        notes: List[str] = []
        for path, e in items:
            p = Path(path).expanduser()
            try:
                if e["deleted"]:
                    if e["before"]:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(e["before"], encoding="utf-8")
                        notes.append(f"restored {path}")
                    # before rong = chi la heuristic rm, khong co gi de dung
                elif not e["before"] and e["after"]:
                    if p.exists():
                        if p.is_dir():
                            notes.append(f"skip {path} (is a directory)")
                        else:
                            p.unlink()
                            notes.append(f"removed new file {path}")
                    else:
                        notes.append(f"already gone {path}")
                elif e["before"] == e["after"]:
                    notes.append(f"unchanged {path}")
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(e["before"], encoding="utf-8")
                    notes.append(f"reverted {path}")
            except OSError as exc:
                notes.append(f"FAILED {path}: {exc}")
        return notes or ["(nothing to undo)"]

    def diff_lines(self, path: str) -> List[Tuple[str, str]]:
        """Unified diff (ngắn) cho một file -> [(ký_hiệu, dòng)]."""
        e = self.files[path]
        b, a = e["before"].splitlines(), e["after"].splitlines()
        out: List[Tuple[str, str]] = []
        for ln in difflib.unified_diff(b, a, lineterm="", n=0):
            if ln.startswith(("+++", "---", "@@")):
                continue
            out.append((ln[:1], ln[1:]))
        return out

    def diff_blocks(self, path: str) -> List[Tuple[str, str]]:
        """Unified diff (gom cụm liền kề) -> [(ký_hiệu, khối_dòng)]."""
        groups: List[Tuple[str, str]] = []
        for kind, ln in self.diff_lines(path):
            if groups and groups[-1][0] == kind:
                groups[-1] = (kind, groups[-1][1] + "\n" + ln)
            else:
                groups.append((kind, ln))
        return groups


BUILTIN_TOOLS: List[Tool] = [
    Tool(
        name="calculator",
        description="Tính biểu thức số học an toàn (chỉ + - * / // % **).",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Biểu thức, vd: (2+3)*7/2"},
            },
            "required": ["expression"],
        },
        handler=_tool_calculator,
    ),
    Tool(
        name="current_time",
        description="Lấy ngày giờ hiện tại của máy.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_tool_current_time,
    ),
    Tool(
        name="read_file",
        description="Đọc nội dung một file văn bản (hỗ trợ offset để lật trang file lớn).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Đường dẫn file"},
                "max_lines": {"type": "integer", "description": "Số dòng tối đa (mặc định 200)"},
                "offset": {"type": "integer", "description": "Bỏ qua N dòng đầu (mặc định 0)"},
            },
            "required": ["path"],
        },
        handler=_tool_read_file,
    ),
    Tool(
        name="write_file",
        description="Ghi (hoặc ghi đè) nội dung vào một file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Đường dẫn file"},
                "content": {"type": "string", "description": "Nội dung cần ghi"},
            },
            "required": ["path", "content"],
        },
        handler=_tool_write_file,
        dangerous=True,
    ),
    Tool(
        name="list_dir",
        description="Liệt kê file/thư mục trong một thư mục.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Đường dẫn thư mục (mặc định '.')"},
            },
            "required": [],
        },
        handler=_tool_list_dir,
    ),
    Tool(
        name="edit_file",
        description="Sửa 1 đoạn trong file (khớp chính xác old_text). Mặc định yêu cầu đoạn duy nhất; đặt replace_all=true để thay hết. (chỉ khả dụng ở mode Build).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Đường dẫn file"},
                "old_text": {"type": "string", "description": "Đoạn cần thay (khớp chính xác)"},
                "new_text": {"type": "string", "description": "Đoạn thay thế"},
                "replace_all": {"type": "boolean", "description": "Thay tất cả vị trí (mặc định false)"},
            },
            "required": ["path", "old_text", "new_text"],
        },
        handler=_tool_edit_file,
        dangerous=True,
    ),
    Tool(
        name="delete_file",
        description="Xóa một file (hỏi xác nhận trước, chỉ khả dụng ở mode Build).",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Đường dẫn file cần xóa"},
            },
            "required": ["path"],
        },
        handler=_tool_delete_file,
        dangerous=True,
        need_confirm=True,
    ),
    Tool(
        name="grep",
        description="Tìm regex trong file/thư mục (bỏ qua .git/node_modules, tối đa 50 kết quả).",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex cần tìm"},
                "path": {"type": "string", "description": "File/thư mục (mặc định '.')"},
                "include": {"type": "string", "description": "Lọc tên file kiểu glob, vd '*.py'"},
            },
            "required": ["pattern"],
        },
        handler=_tool_grep,
    ),
    Tool(
        name="glob",
        description="Liệt kê file khớp glob pattern (vd '**/*.py'), tối đa 100.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (mặc định '*')"},
                "path": {"type": "string", "description": "Thư mục gốc (mặc định '.')"},
            },
            "required": [],
        },
        handler=_tool_glob,
    ),
    Tool(
        name="fetch_url",
        description="Tải URL http(s) về text (bỏ HTML tags, tối đa max_chars ký tự).",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL http(s)"},
                "max_chars": {"type": "integer", "description": "Số ký tự tối đa (mặc định 6000)"},
            },
            "required": ["url"],
        },
        handler=_tool_fetch_url,
    ),
    Tool(
        name="web_search",
        description="Tìm kiếm web (DuckDuckGo, không cần key). Trả về tiêu đề + URL + mô tả. Dùng trước fetch_url khi cần thông tin mới.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Từ khóa tìm kiếm"},
                "max_results": {"type": "integer", "description": "Số kết quả tối đa (mặc định 5)"},
            },
            "required": ["query"],
        },
        handler=_tool_web_search,
    ),
    Tool(
        name="ask_user",
        description=(
            "Hỏi user 1-5 câu trong MỘT form và NHẬN đáp án về để làm tiếp. "
            "BẮT BUỘC dùng khi cần đáp án từ user (đố, khảo sát, chọn, thiếu "
            "info bắt buộc) — viết câu hỏi bằng text thì KHÔNG nhận được trả "
            "lời. User nêu đích danh tool thì bắt buộc gọi. Đủ info thì tự "
            "quyết, không gọi."
        ),
        parameters={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Danh sách câu hỏi, vd: [\"Dự án dùng framework gì?\", \"Chủ đề là gì?\"]",
                },
            },
            "required": ["questions"],
        },
        handler=_tool_ask_user,
    ),
]


def _tool_run_command(args: Dict[str, Any]) -> str:
    import subprocess

    args = dict(args)
    changes = args.pop("_changes", None)
    command = str(args.get("command", "")).strip()
    timeout = int(args.get("timeout", 30) or 30)
    if not command:
        return "Lỗi: thiếu lệnh"
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        if changes is not None:
            changes.record_deleted_files(command)
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        parts = [f"$ {command}", f"exit: {proc.returncode}"]
        if out:
            parts.append(out[:3000])
        if err:
            parts.append("stderr: " + err[:1500])
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Lệnh quá {timeout}s — đã hủy"
    except OSError as exc:
        return f"Lỗi chạy lệnh: {exc}"


BUILTIN_TOOLS.append(
    Tool(
        name="run_command",
        description="Chạy một lệnh shell (hỏi xác nhận trước, chỉ khả dụng ở mode Build).",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Lệnh cần chạy"},
                "timeout": {"type": "integer", "description": "Giây tối đa (mặc định 30)"},
            },
            "required": ["command"],
        },
        handler=_tool_run_command,
        dangerous=True,
        need_confirm=True,
    )
)

# Biệt danh q&a-user — cùng handler với ask_user: hỏi người dùng nhiều câu
# trong MỘT form rồi nhận câu trả lời để tiếp tục xử lý.
BUILTIN_TOOLS.append(
    Tool(
        name="qa_user",
        description=(
            "Bí danh của ask_user: hỏi user 1-5 câu trong MỘT form, nhận đáp án "
            "để xử lý tiếp. Dùng cho nhiệm vụ tương tác (đố, khảo sát, chọn). "
            "User nêu tên tool này thì bắt buộc gọi, cấm hỏi bằng text."
        ),
        parameters={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Danh sách câu hỏi, vd: [\"Dự án dùng framework gì?\", \"Chủ đề là gì?\"]",
                },
            },
            "required": ["questions"],
        },
        handler=_tool_ask_user,
    )
)


class ToolRegistry:
    """Tập hợp tool: builtin + MCP; map tên -> Tool để agent gọi.

    Mode an toàn: tool nguy hiểm (ghi file, chạy lệnh) chỉ dùng được khi
    allow_dangerous = True (mode Build). Mode Plan = chỉ tool đọc/tính.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {t.name: t for t in BUILTIN_TOOLS}
        self.mcp: MCPManager | None = None
        self.allow_dangerous = True
        self.confirm_dangerous = True  # hoi user truoc tool need_confirm

    def set_allow_dangerous(self, allowed: bool) -> None:
        self.allow_dangerous = allowed

    def _usable(self, tool: Tool) -> bool:
        return self.allow_dangerous or not tool.dangerous

    # ---------------------------------------------------------------- CRUD
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    # ------------------------------------------------------------- schemas
    def openai_schemas(self) -> List[Dict[str, Any]]:
        """Schema cho tham số 'tools' của OpenAI SDK (lọc theo mode)."""
        return [t.openai_schema() for t in self._tools.values() if self._usable(t)]

    # ------------------------------------------------------------- execute
    @staticmethod
    def _confirm_prompt(tool: Tool, arguments: Dict[str, Any]) -> str:
        """Mo ta ngan de user duyet: lenh cu the thay vi ten tool chung chung."""
        if tool.name == "run_command":
            cmd = str(arguments.get("command", "")).strip()
            return f"Run shell command: {cmd[:150]}" if cmd else "Run shell command?"
        if tool.name == "delete_file":
            return f"Delete file: {arguments.get('path', '')}"
        detail = ", ".join(f"{k}={v}" for k, v in list(arguments.items())[:3])
        return f"Run tool '{tool.name}'? {detail}".strip()

    def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        *,
        changes=None,
        ask=None,
        emit=None,
        confirm=None,
    ) -> str:
        """Chạy tool theo tên; trả về kết quả văn bản (an toàn, không raise).

        - changes: ChangeTracker của lượt chạy (ghi file đã sửa để UI vẽ diff).
        - ask:     callback(questions)->answers cho tool ask_user/qa_user.
        - emit:    callback(event, data) để UI biết tool đang hỏi user.
        - confirm: callback(prompt)->bool duyet tool need_confirm (lenh shell,
          xoa file...). None = khong hoi (non-interactive), cho chay luôn.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Lỗi: không có tool nào tên '{name}'"
        if not self._usable(tool):
            return (
                f"Lỗi: tool '{name}' chỉ dùng được ở mode Build "
                "(tab để chuyển mode)"
            )
        if tool.need_confirm and self.confirm_dangerous and confirm is not None:
            try:
                ok = confirm(self._confirm_prompt(tool, arguments or {}))
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                return f"Cancelled by user: tool '{name}' không được duyệt."
        args = dict(arguments or {})
        if changes is not None:
            args["_changes"] = changes
        if ask is not None:
            args["_ask"] = ask
        if emit is not None:
            args["_emit"] = emit
        try:
            return str(tool.handler(args))
        except Exception as exc:  # noqa: BLE001
            return f"Lỗi khi chạy tool {name}: {type(exc).__name__}: {exc}"

    # ------------------------------------------------------------- summary
    def summary(self) -> List[str]:
        rows = []
        for t in self._tools.values():
            mark = "  [build-only]" if (t.dangerous and not self.allow_dangerous) else ""
            rows.append(f"{t.name} [{t.source}] — {t.description}{mark}")
        return rows