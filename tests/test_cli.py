"""Kiem thu stdin pipe + flag -p (khong mang, khong LLM that)."""

import sys

from syncode.cli import parse_args, read_piped_stdin, with_stdin_context


def test_parse_print_flag():
    args = parse_args(["-p", "--task", "hi"])
    assert args.print is True and args.task == "hi"
    args2 = parse_args([])
    assert args2.print is False


def test_with_stdin_context_passthrough():
    assert with_stdin_context("task", "") == "task"
    assert with_stdin_context("task", "   ") == "task"
    out = with_stdin_context("task", "log line")
    assert "task" in out and "log line" in out and "STDIN" in out


def test_read_piped_stdin_tty(monkeypatch):
    class FakeTTY:
        def isatty(self):
            return True

        def read(self, n=-1):
            raise AssertionError("khong duoc doc stdin cua tty")

    monkeypatch.setattr(sys, "stdin", FakeTTY())
    assert read_piped_stdin() == ""


def test_read_piped_stdin_pipe_and_cap(monkeypatch):
    class FakePipe:
        def isatty(self):
            return False

        def read(self, n=-1):
            return "line1\nline2\n"

    monkeypatch.setattr(sys, "stdin", FakePipe())
    assert read_piped_stdin() == "line1\nline2"

    class FakeBigPipe(FakePipe):
        def read(self, n=-1):
            return "z" * (n + 100)

    monkeypatch.setattr(sys, "stdin", FakeBigPipe())
    out = read_piped_stdin(max_chars=100)
    assert len(out) <= 130 and "truncated" in out


def test_read_piped_stdin_broken(monkeypatch):
    class FakeBroken:
        def isatty(self):
            return False

        def read(self, n=-1):
            raise OSError("closed")

    monkeypatch.setattr(sys, "stdin", FakeBroken())
    assert read_piped_stdin() == ""


def test_print_without_task_errors(capsys):
    from syncode.cli import main

    # stdin trong pytest khong phai tty va rong -> thieu task -> return 2
    assert main(["-p"]) == 2
    assert "usage" in capsys.readouterr().err.lower()
