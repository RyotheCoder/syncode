"""Quy uoc repo (AGENTS.md): tu nap de agent nhat quan voi codebase.

Tim AGENTS.md tu thu muc hien tai nguoc len tren, doc toi da
CONVENTION_MAX_CHARS ky tu. Khong bao gio raise."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

CONVENTION_FILENAMES = ("AGENTS.md",)
CONVENTION_MAX_CHARS = 4000
CONVENTION_MAX_DEPTH = 4
CONVENTION_MAX_BYTES = 32_000

SKILLS_DIRNAME = ".skills"
SKILLS_HOME_SUBDIR = "skills"
SKILLS_MAX_FILES = 10
SKILLS_MAX_CHARS = 6000  # giu de tuong thich; load_skills gio chi tra muc luc
SKILL_INDEX_DESC_CHARS = 160


def _home_dir():
    """Thu muc home cua Syncode (ton trong SYNCODE_HOME de test cach ly)."""
    try:
        return Path(os.environ.get("SYNCODE_HOME", Path.home() / ".syncode"))
    except OSError:
        return Path.home() / ".syncode"


def find_conventions(start: Optional[str | Path] = None, max_depth: int = CONVENTION_MAX_DEPTH):
    """Tim file AGENTS.md gan nhat tu start nguoc len. Tra Path hoac None."""
    try:
        d = Path(start or os.getcwd()).resolve()
    except OSError:
        return None
    for _ in range(max(0, max_depth) + 1):
        for name in CONVENTION_FILENAMES:
            p = d / name
            try:
                if p.is_file():
                    return p
            except OSError:
                continue
        parent = d.parent
        if parent == d:
            break
        d = parent
    return None


def load_conventions(start: Optional[str | Path] = None) -> str:
    """Doc quy uoc repo thanh khoi text gan vao prompt. Tra '' neu khong co."""
    p = find_conventions(start)
    if p is None:
        return ""
    try:
        if p.stat().st_size > CONVENTION_MAX_BYTES:
            return ""
        text = p.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not text:
        return ""
    if len(text) > CONVENTION_MAX_CHARS:
        text = text[:CONVENTION_MAX_CHARS] + "… (rut gon)"
    return f"QUY UOC REPO ({p.name}, de nhat quan voi codebase):\n{text}"


AGENTS_TEMPLATE = """# AGENTS.md — repo conventions (auto-loaded by Syncode)

## Build / test
- Build: ...
- Test: ...

## Code style
- ...

## Notes for AI agents
- Keep answers focused; full code only when asked.
- Prefer small edits over full-file rewrites.
- Never commit unless asked.
"""


def find_skill_files(start: Optional[str | Path] = None):
    """Tim file skill (*.md): thu muc .skills GAN NHAT tu start nguoc len
    + ~/.syncode/skills/. Tra list[Path] (toi da SKILLS_MAX_FILES)."""
    found = []
    seen = set()
    try:
        d = Path(start or os.getcwd()).resolve()
    except OSError:
        d = None
    depth = 0
    while d is not None and depth <= CONVENTION_MAX_DEPTH:
        sdir = d / SKILLS_DIRNAME
        try:
            is_dir = sdir.is_dir()
        except OSError:
            is_dir = False
        if is_dir:
            try:
                names = sorted(p.name for p in sdir.glob("*.md") if p.is_file())
            except OSError:
                names = []
            if names:  # lay thu muc gan nhat co skill, khong gom lan
                for n in names:
                    p = sdir / n
                    if str(p) not in seen:
                        seen.add(str(p))
                        found.append(p)
                break
        parent = d.parent
        if parent == d:
            break
        d = parent
        depth += 1
    try:
        home_skills = _home_dir() / SKILLS_HOME_SUBDIR
        if home_skills.is_dir():
            for p in sorted(home_skills.glob("*.md")):
                if p.is_file() and str(p) not in seen:
                    seen.add(str(p))
                    found.append(p)
    except OSError:
        pass
    return found[:SKILLS_MAX_FILES]


def _skill_index_desc(text: str) -> str:
    """Dong mo ta ngan cho muc luc: dong noi dung dau tien (bo dau #),
    cat 160 ky tu. Tra '' neu file khong co dong nao."""
    for ln in (text or "").splitlines():
        s = ln.strip().lstrip("#").strip()
        if s:
            if len(s) > SKILL_INDEX_DESC_CHARS:
                return s[:SKILL_INDEX_DESC_CHARS] + "…"
            return s
    return ""


def load_skills(start: Optional[str | Path] = None) -> str:
    """Muc luc skills gan vao prompt (progressive disclosure).

    Chi tra ten + 1 dong mo ta + duong dan moi skill; agent tu doc file bang
    tool read_file khi subtask can roi ap dung. Nho hon full-text rat nhieu
    nen thinking khong con phan tich ca thu vien skill. Tra '' neu khong co.
    """
    files = find_skill_files(start)
    if not files:
        return ""
    rows = []
    for p in files:
        try:
            if p.stat().st_size > CONVENTION_MAX_BYTES:
                continue
            text = p.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        desc = _skill_index_desc(text)
        if not desc:
            continue
        rows.append(f"- {p.stem} ({p}): {desc}")
    if not rows:
        return ""
    return (
        "KỸ NĂNG KHẢ DỤNG (mục lục — KHÔNG phân tích ở đây; subtask nào cần "
        "skill nào thì đọc file bằng tool read_file rồi áp dụng):\n"
        + "\n".join(rows)
    )
