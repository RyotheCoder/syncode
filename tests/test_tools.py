"""Kiem thu cac tool file/search/fetch (stdlib, tmp_path, khong mang that)."""

from syncode.tools.registry import (
    ChangeTracker,
    ToolRegistry,
    _tool_delete_file,
    _tool_edit_file,
    _tool_fetch_url,
    _tool_glob,
    _tool_grep,
)


def test_edit_file_unique_and_counts(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("foo bar foo\n", encoding="utf-8")
    out = _tool_edit_file({"path": str(f), "old_text": "bar", "new_text": "BAZ"})
    assert "1 vị trí" in out
    assert f.read_text(encoding="utf-8") == "foo BAZ foo\n"
    out2 = _tool_edit_file({"path": str(f), "old_text": "foo", "new_text": "x"})
    assert "2 lần" in out2  # mo ho khong replace_all
    out3 = _tool_edit_file(
        {"path": str(f), "old_text": "foo", "new_text": "x", "replace_all": True}
    )
    assert "2 vị trí" in out3
    assert f.read_text(encoding="utf-8") == "x BAZ x\n"
    assert "không tìm thấy" in _tool_edit_file(
        {"path": str(f), "old_text": "zzz", "new_text": "x"}
    )
    assert "không tồn tại" in _tool_edit_file(
        {"path": str(tmp_path / "nope.txt"), "old_text": "a", "new_text": "b"}
    )


def test_edit_file_tracks_changes(tmp_path):
    reg = ToolRegistry()
    tr = ChangeTracker()
    f = tmp_path / "b.txt"
    f.write_text("hello", encoding="utf-8")
    out = reg.execute(
        "edit_file",
        {"path": str(f), "old_text": "hello", "new_text": "hi"},
        changes=tr,
    )
    assert "1 vị trí" in out
    assert str(f) in tr.files
    assert tr.files[str(f)]["before"] == "hello"


def test_delete_file(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("bye", encoding="utf-8")
    tr = ChangeTracker()
    out = _tool_delete_file({"path": str(f), "_changes": tr})
    assert "Đã xóa" in out
    assert not f.exists()
    assert tr.files[str(f)]["deleted"] is True
    assert "không tồn tại" in _tool_delete_file({"path": str(f)})
    d = tmp_path / "sub"
    d.mkdir()
    assert "thư mục" in _tool_delete_file({"path": str(d)})


def test_grep_basic_skip_and_include(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("foo bar\n", encoding="utf-8")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "x.py").write_text("foo\n", encoding="utf-8")
    out = _tool_grep({"pattern": "foo", "path": str(tmp_path)})
    assert "a.py:1" in out and "b.txt:1" in out
    assert ".git" not in out
    out2 = _tool_grep({"pattern": "foo", "path": str(tmp_path), "include": "*.py"})
    assert "a.py" in out2 and "b.txt" not in out2
    assert "Lỗi regex" in _tool_grep({"pattern": "([", "path": str(tmp_path)})
    assert "thiếu pattern" in _tool_grep({"path": str(tmp_path)})
    assert "không tìm thấy" in _tool_grep({"pattern": "zzzqqq", "path": str(tmp_path)})


def test_glob_basic(tmp_path):
    (tmp_path / "x.py").write_text("1", encoding="utf-8")
    (tmp_path / "y.md").write_text("2", encoding="utf-8")
    out = _tool_glob({"pattern": "*.py", "path": str(tmp_path)})
    assert "x.py" in out and "y.md" not in out
    assert "không tồn tại" in _tool_glob({"pattern": "*", "path": str(tmp_path / "nope")})


def test_fetch_url_validation_and_strip(monkeypatch):
    assert "http(s)" in _tool_fetch_url({"url": "ftp://x"})

    class FakeResp:
        def __init__(self, data):
            self._data = data

        def read(self, n=-1):
            return self._data if n is None or n < 0 else self._data[:n]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    html = (
        b"<html><head><style>.a{color:red}</style></head>"
        b"<body><h1>Hello</h1><script>evil()</script><p>World</p></body></html>"
    )

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=20: FakeResp(html),
    )
    out = _tool_fetch_url({"url": "https://example.test/"})
    assert "Hello" in out and "World" in out
    assert "evil" not in out
    assert "color:red" not in out
    assert "max_chars" in _tool_fetch_url({"url": "https://example.test/", "max_chars": "abc"})


def test_web_search_validation():
    from syncode.tools.registry import _tool_web_search

    assert "thiếu query" in _tool_web_search({"query": "   "})
    assert "max_results" in _tool_web_search({"query": "x", "max_results": "abc"})
    assert "max_results" in _tool_web_search({"query": "x", "max_results": 0})


def test_web_search_parses_ddg(monkeypatch):
    from syncode.tools.registry import _tool_web_search

    html = """
    <div class="result">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=x">First <b>Result</b></a>
      <a class="result__snippet" href="x">Snippet one here</a>
    </div>
    <div class="result">
      <a rel="nofollow" class="result__a" href="https://example.org/b">Second</a>
    </div>
    """.encode()

    class FakeResp:
        def read(self, n=-1):
            return html if n is None or n < 0 else html[:n]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=20: FakeResp(),
    )
    out = _tool_web_search({"query": "test"})
    assert "1. First Result" in out
    assert "https://example.com/a" in out
    assert "Snippet one here" in out
    assert "2. Second" in out
    assert "https://example.org/b" in out
    out1 = _tool_web_search({"query": "test", "max_results": 1})
    assert "First Result" in out1 and "Second" not in out1


def test_web_search_no_results(monkeypatch):
    from syncode.tools.registry import _tool_web_search

    class FakeResp:
        def read(self, n=-1):
            return b"<html><body>nothing here</body></html>"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=20: FakeResp(),
    )
    assert "không tìm thấy" in _tool_web_search({"query": "zzzqqq"})


def test_web_search_registered():
    from syncode.tools import ToolRegistry

    names = {t.name for t in ToolRegistry().all_tools()}
    assert "web_search" in names and "fetch_url" in names


def test_undo_all_restores_modified_and_new(tmp_path):
    tr = ChangeTracker()
    f1 = tmp_path / "a.txt"
    f1.write_text("orig", encoding="utf-8")
    f2 = tmp_path / "b.txt"
    tr.record_file(str(f1), "orig", "mod")
    tr.record_file(str(f2), "", "new")
    f1.write_text("mod", encoding="utf-8")
    f2.write_text("new", encoding="utf-8")
    notes = tr.undo_all()
    assert f1.read_text(encoding="utf-8") == "orig"
    assert not f2.exists()
    assert any("reverted" in n for n in notes)
    assert any("removed new file" in n for n in notes)
    assert tr.files == {}  # undo xong het dau vet
    assert tr.undo_all() == ["(nothing to undo)"]  # undo lan 2 la no-op


def test_undo_all_restores_deleted(tmp_path):
    tr = ChangeTracker()
    f = tmp_path / "d.txt"
    f.write_text("keep", encoding="utf-8")
    tr.record_file(str(f), "keep", "", deleted=True)
    f.unlink()
    notes = tr.undo_all()
    assert f.read_text(encoding="utf-8") == "keep"
    assert any("restored" in n for n in notes)


def test_read_file_offset(tmp_path):
    from syncode.tools.registry import _tool_read_file

    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")
    out = _tool_read_file({"path": str(f), "max_lines": 4, "offset": 3})
    assert "line4" in out and "line7" in out
    assert "line3" not in out.splitlines()[0] and "line8" not in out
    assert "còn 3 dòng nữa" in out
    assert "hết file" in _tool_read_file({"path": str(f), "offset": 50})
    assert "offset phải là số" in _tool_read_file({"path": str(f), "offset": "abc"})


def test_confirm_allow_deny_and_bypass():
    reg = ToolRegistry()
    ok = reg.execute("run_command", {"command": "echo hi"}, confirm=lambda p: True)
    assert "hi" in ok
    denied = reg.execute("run_command", {"command": "echo hi"}, confirm=lambda p: False)
    assert "Cancelled by user" in denied
    # tat confirm -> chay luon, khong goi callback
    reg.confirm_dangerous = False

    def _boom(prompt_text):
        raise AssertionError("khong duoc hoi")

    assert "hi" in reg.execute("run_command", {"command": "echo hi"}, confirm=_boom)
    # khong callback (non-interactive) -> cho chay (tuong thich cu)
    reg.confirm_dangerous = True
    assert "hi" in reg.execute("run_command", {"command": "echo hi"})
    # tool thuong khong bi hoi
    assert "4" in reg.execute("calculator", {"expression": "2+2"}, confirm=_boom)


def test_confirm_prompt_shows_command():
    reg = ToolRegistry()
    seen = {}
    reg.execute("run_command", {"command": "rm -rf /tmp/x"}, confirm=lambda p: seen.setdefault("p", p) or False)
    assert "rm -rf /tmp/x" in seen["p"]
    reg.execute("delete_file", {"path": "/tmp/y"}, confirm=lambda p: seen.setdefault("p2", p) or False)
    assert "/tmp/y" in seen["p2"]
