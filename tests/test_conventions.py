"""Kiem thu AGENTS.md (tim/nap/tiem vao swarm) + lenh /.init."""

from syncode.conventions import find_conventions, load_conventions


def test_find_conventions_upward(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("Quy uoc A.", encoding="utf-8")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    found = find_conventions()
    assert found is not None and found.name == "AGENTS.md"
    assert "Quy uoc A." in load_conventions()


def test_load_conventions_missing_or_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert find_conventions() is None
    assert load_conventions() == ""
    (tmp_path / "AGENTS.md").write_text("   \n", encoding="utf-8")
    assert load_conventions() == ""


def test_swarm_run_includes_conventions():
    from test_orchestrator import FakeClient, _role

    from syncode.swarm.orchestrator import SwarmOrchestrator

    class ConvClient(FakeClient):
        def _reply(self, system, user):
            role = _role(system)
            if "PLANNER" in role:
                assert "QUY UOC REPO" in user
                return '[{"id": "t1", "title": "Lam", "goal": "lam"}]'
            if "CRITIC" in role:
                return '{"verdict": "APPROVED"}'
            if "WORKER" in role:
                assert "QUY UOC REPO" in user
                return "worker-ok"
            if "JUDGE" in role:
                assert "QUY UOC REPO" in user
                return "CONV OK"
            return super()._reply(system, user)

    orch = SwarmOrchestrator(ConvClient(), num_workers=1)
    result = orch.run("lam gi do", max_rounds=1, conventions="QUY UOC REPO (test): tabs.")
    assert result.answer == "CONV OK"


def test_init_command_creates_template(tmp_path, monkeypatch, capsys):
    from test_smoke import _ctx

    from syncode.commands.slash import execute

    monkeypatch.chdir(tmp_path)
    ctx = _ctx(tmp_path, monkeypatch)
    assert execute("/.init", ctx) is False
    target = tmp_path / "AGENTS.md"
    assert target.exists()
    assert "AGENTS.md" in target.read_text(encoding="utf-8")
    # chay lai -> tu choi ghi de
    assert execute("/.init", ctx) is False
    assert "already exists" in capsys.readouterr().out.lower()


def test_find_skill_files_repo_and_home(tmp_path, monkeypatch):
    from syncode.conventions import find_skill_files, load_skills

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("SYNCODE_HOME", str(home))
    (home / "skills").mkdir()
    (home / "skills" / "greet.md").write_text("Chao hoi ngan gon.", encoding="utf-8")
    repo = tmp_path / "repo"
    sub = repo / "pkg"
    sub.mkdir(parents=True)
    (repo / ".skills").mkdir()
    (repo / ".skills" / "tests.md").write_text("Chay pytest -q.", encoding="utf-8")
    monkeypatch.chdir(sub)
    found = [p.name for p in find_skill_files()]
    assert found == ["tests.md", "greet.md"]  # repo truoc, home sau
    out = load_skills()
    # muc luc: ten + dong mo ta dau + path de read_file, KHONG full-text
    assert "tests" in out and "pytest" in out
    assert "greet" in out and "Chao hoi" in out
    assert "read_file" in out
    assert "###" not in out


def test_load_skills_index_hides_body(tmp_path, monkeypatch):
    from syncode.conventions import load_skills

    home = tmp_path / "home3"
    home.mkdir()
    monkeypatch.setenv("SYNCODE_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".skills").mkdir()
    (tmp_path / ".skills" / "secret.md").write_text(
        "# Mo ta ngan\nNOI DUNG CHI TIET KHONG DUOC LO.\n", encoding="utf-8"
    )
    out = load_skills()
    assert "secret" in out and "Mo ta ngan" in out
    assert "secret.md" in out  # path de agent read_file khi can
    assert "KHONG DUOC LO" not in out


def test_load_skills_missing_or_empty(tmp_path, monkeypatch):
    from syncode.conventions import find_skill_files, load_skills

    home = tmp_path / "home2"
    home.mkdir()
    monkeypatch.setenv("SYNCODE_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    assert find_skill_files() == []
    assert load_skills() == ""
    (tmp_path / ".skills").mkdir()
    (tmp_path / ".skills" / "trong.md").write_text("   \n", encoding="utf-8")
    assert load_skills() == ""


def test_swarm_run_includes_skills():
    from test_orchestrator import FakeClient, _role

    from syncode.swarm.orchestrator import SwarmOrchestrator

    class SkillClient(FakeClient):
        def _reply(self, system, user):
            role = _role(system)
            if "PLANNER" in role:
                return '[{"id": "t1", "title": "Lam", "goal": "lam"}]'
            if "CRITIC" in role:
                return '{"verdict": "APPROVED"}'
            if "WORKER" in role:
                assert "KY NANG" in user
                return "worker-ok"
            if "JUDGE" in role:
                return "SKILL OK"
            return super()._reply(system, user)

    orch = SwarmOrchestrator(SkillClient(), num_workers=1)
    result = orch.run("lam gi do", max_rounds=1, skills="KY NANG (test): xyz.")
    assert result.answer == "SKILL OK"
