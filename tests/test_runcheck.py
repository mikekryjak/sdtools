"""runcheck.py, the checks runq.py and slotd.py share, plus the slotlib and
sdrun.py changes it relies on. Run against throwaway cases and slot config."""
import json
import os
import shutil
import subprocess
import sys

import pytest

CLI = os.path.join(os.path.dirname(__file__), "..", "cli")
sys.path.insert(0, CLI)
import runcheck  # noqa: E402
import sdrun  # noqa: E402
import slotlib  # noqa: E402

USER = {"kind": "user"}
AGENT = {"kind": "agent", "session": "s1", "tmux": None}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Slots 1 and 2 in queue mode; slot 1 owned by campaign camp, whose
    cases are cases/camp and whose clone holds build-x/hermes-3."""
    cases = tmp_path / "cases"
    camp = cases / "camp"
    camp.mkdir(parents=True)
    clone = tmp_path / "clone"
    (clone / "build-x").mkdir(parents=True)
    exe = clone / "build-x" / "hermes-3"
    shutil.copy("/bin/true", exe)
    (tmp_path / "camp.toml").write_text(
        f'name = "camp"\ncases = "{camp}"\nclone = "{clone}"\n')
    state = tmp_path / "st"
    cfg = {"enforce": True, "cases": str(cases), "cores": {"1": "0", "2": "1"},
           "state_dir": str(state), "campaigns": str(tmp_path / "*.toml")}
    (tmp_path / "slots.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("SDTOOLS_SLOT_CONFIG", str(tmp_path / "slots.json"))
    monkeypatch.delenv("SDTOOLS_SLOT_ENFORCE", raising=False)
    cfg = slotlib.config()
    for n in ("1", "2"):
        open(slotlib.path(cfg, n, "mode"), "w").write("queue\n")
    open(slotlib.path(cfg, "1", "owner"), "w").write(
        json.dumps({"campaign": "camp"}))
    return {"cfg": cfg, "cases": cases, "camp": camp, "clone": clone,
            "exe": str(exe), "state": state}


def case(env, name="c1", inp="error_on_unused_options = true\n"):
    d = env["camp"] / name
    d.mkdir(exist_ok=True)
    (d / "BOUT.inp").write_text(inp)
    return d


def job(env, *args, slot="1", who=USER, cwd=None):
    return runcheck.make_job(env["cfg"], slot, list(args),
                             cwd=str(cwd or env["camp"]), who=who)


def rules(env, j, when="add", slot=None):
    return sorted(r.rule for r in runcheck.problems(env["cfg"], j, when, slot)
                  if not r.warn)


def store(env, where, slot, j):
    d = env["state"] / where / slot
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{j['id']}.json").write_text(json.dumps(j))


def test_a_clean_case_passes(env):
    c = case(env)
    j = job(env, "c1", f"-p={env['exe']}")
    assert rules(env, j) == []
    assert j["case"] == str(c) and j["cwd"] == str(env["camp"])
    assert j["argv"] == [f"-p={env['exe']}"] and j["exe_hash"]
    assert j["campaign"] == "camp"


def test_case_by_d_flag_and_runq_options(env):
    case(env)
    j = job(env, "-d", "c1", "-p", env["exe"], "--timeout", "2h",
            "--study", "S03", "nout=5")
    assert j["case"].endswith("/c1") and j["timeout_s"] == 7200
    assert j["study"] == "S03" and j["argv"][-1] == "nout=5"


def test_spent_by_log_or_by_dump_alone(env):
    c = case(env)
    (c / "BOUT.log.0").write_text("no Run time line: killed at a timeout\n")
    r = runcheck.problems(env["cfg"], job(env, "c1", f"-p={env['exe']}"),
                          "add")
    assert [x.rule for x in r] == ["spent-case"]
    assert "-restart -append" in r[0].fix and "--rerun" in r[0].fix
    (c / "BOUT.log.0").unlink()
    (c / "BOUT.dmp.0.nc").write_text("")
    assert rules(env, job(env, "c1", f"-p={env['exe']}")) == ["spent-case"]


def test_restart_flags_on_a_spent_case(env):
    c = case(env)
    (c / "BOUT.dmp.0.nc").write_text("")
    p = f"-p={env['exe']}"
    assert rules(env, job(env, "c1", p, "-restart")) == ["restart-only"]
    assert rules(env, job(env, "c1", p, "-append")) == []
    assert rules(env, job(env, "c1", p, "-restart", "-append")) == []
    assert rules(env, job(env, "c1", p, "--rerun")) == []
    assert rules(env, job(env, "c1", p, "--rerun", "-restart")) == \
        ["rerun-restart"]


def test_validate_leaves_a_spent_case_alone(env):
    c = case(env)
    (c / "BOUT.log.0").write_text("")
    j = job(env, "c1", f"-p={env['exe']}", "--validate")
    assert rules(env, j) == [] and j["timeout_s"] == 900


def test_relative_grid_resolves_from_the_parent(env):
    case(env, inp='grid = "grid.nc"\n')
    (env["camp"] / "grid.nc").write_text("")
    assert rules(env, job(env, "c1", f"-p={env['exe']}")) == []
    (env["camp"] / "grid.nc").rename(env["camp"] / "c1" / "grid.nc")
    r = runcheck.problems(env["cfg"], job(env, "c1", f"-p={env['exe']}"),
                          "add")
    assert [x.rule for x in r] == ["file-unresolved"]
    assert f"--cwd {env['camp'] / 'c1'}" in r[0].fix
    j = job(env, "c1", f"-p={env['exe']}", "--cwd", "c1")
    assert j["cwd"] == str(env["camp"] / "c1")
    assert j["case"] == str(env["camp"] / "c1" / "c1")


def test_mesh_file_and_nc_values(env):
    case(env, inp='[mesh]\nfile = "g.nc"   # grid\n'
                  '[x]\nprofile = "p.nc"\nvalue = "${data}/q.nc"\n')
    j = job(env, "c1", f"-p={env['exe']}")
    r = runcheck.problems(env["cfg"], j, "add")
    assert r[0].rule == "file-unresolved"
    assert "g.nc, p.nc" in r[0].message and "q.nc" not in r[0].message


def test_binary_must_be_named_and_exist(env, monkeypatch):
    case(env)
    r = runcheck.problems(env["cfg"], job(env, "c1"), "add")
    assert [x.rule for x in r] == ["binary-unnamed"]
    assert "-p=<path to hermes-3>" in r[0].fix
    assert rules(env, job(env, "c1", "-p=/nowhere/hermes-3")) == \
        ["binary-missing"]
    monkeypatch.setenv("hermes", str(env["clone"]))
    j = job(env, "c1", "-b", "build-x")
    assert rules(env, j) == [] and j["exe"] == env["exe"]
    assert j["argv"] == [f"-p={env['exe']}"]


def test_binary_change_refuses_at_start(env):
    case(env)
    j = job(env, "c1", f"-p={env['exe']}")
    assert rules(env, j, "start") == []
    with open(env["exe"], "ab") as f:
        f.write(b"rebuilt")
    assert rules(env, j, "start") == ["binary-changed"]


def test_a_build_local_library_counts_in_the_hash(env, monkeypatch):
    case(env)
    lib = env["clone"] / "build-x" / "libhermes-3-lib.so"
    lib.write_text("v1")
    monkeypatch.setattr(runcheck, "linked_libs", lambda exe: [str(lib)])
    j = job(env, "c1", f"-p={env['exe']}")
    lib.write_text("v2")
    assert rules(env, j, "start") == ["binary-changed"]


def test_linked_libs_keeps_only_the_build_directory(env):
    assert runcheck.linked_libs(env["exe"]) == []   # /bin/true: system libs


def test_duplicate_in_queue_or_runplan(env):
    case(env)
    first = job(env, "c1", f"-p={env['exe']}")
    store(env, "queue", "2", first)
    assert rules(env, job(env, "c1", f"-p={env['exe']}")) == ["duplicate"]
    chained = job(env, "c1", f"-p={env['exe']}", "-append",
                  "--after", first["id"])
    assert rules(env, chained) == []
    (env["state"] / "queue" / "2" / f"{first['id']}.json").unlink()
    (env["cases"] / "runplan2.sh").write_text(
        f"cd {env['camp']}\nsdrun.py -s=2 -d c1 2>&1 | tee c1/BOUT.log.console\n")
    assert rules(env, job(env, "c1", f"-p={env['exe']}")) == ["duplicate"]


def test_caller(monkeypatch):
    assert runcheck.caller({}, tty=True) == {"kind": "user"}
    for mark in ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "AI_AGENT",
                 "CODEX_HOME"):
        assert runcheck.caller({mark: "1"}, tty=True)["kind"] == "agent"
    assert runcheck.caller({}, tty=False)["kind"] == "agent"
    assert runcheck.caller({"CLAUDE_CODE_SESSION_ID": "s9"}, tty=False)[
        "session"] == "s9"
    assert runcheck.caller({"RUNQ_JOB": "j1"}, tty=True) == \
        {"kind": "job", "job": "j1"}


def test_unused_options_off_only_warns(env):
    case(env, inp="error_on_unused_options = false\n")
    r = runcheck.problems(env["cfg"], job(env, "c1", f"-p={env['exe']}"),
                          "add")
    assert [(x.rule, x.warn) for x in r] == [("unused-options", True)]


def test_slot_mode(env):
    case(env)
    cfg = env["cfg"]
    os.remove(slotlib.path(cfg, "1", "mode"))
    assert rules(env, job(env, "c1", f"-p={env['exe']}")) == ["slot-mode"]
    open(slotlib.path(cfg, "1", "mode"), "w").write("queu\n")
    r = runcheck.problems(cfg, job(env, "c1", f"-p={env['exe']}"), "add")
    assert r[0].rule == "slot-mode" and "slot1.mode" in r[0].message
    os.remove(slotlib.path(cfg, "1", "mode"))
    os.remove(slotlib.path(cfg, "2", "mode"))
    assert rules(env, job(env, "c1", f"-p={env['exe']}", slot="any")) == \
        ["slot-mode"]
    assert rules(env, job(env, "c1", f"-p={env['exe']}", slot="7")) == \
        ["slot-unknown"]


def test_slot_and_core_flags(env):
    case(env)
    p = f"-p={env['exe']}"
    assert rules(env, job(env, "c1", p, "-s=2")) == ["slot-flag"]
    assert rules(env, job(env, "c1", p, "-s=1")) == []
    assert rules(env, job(env, "c1", p, "-c=1")) == ["slot-flag"]
    assert rules(env, job(env, "c1", p, "-c=0")) == []
    assert rules(env, job(env, "c1", p, "-c=0", slot="any")) == ["slot-flag"]
    assert rules(env, job(env, "c1", p, "HERMES_SLOT=2")) == ["escape"]


def test_agent_rules(env):
    case(env)
    p = f"-p={env['exe']}"
    assert rules(env, job(env, "c1", p, who=AGENT)) == ["agent-timeout"]
    assert rules(env, job(env, "c1", p, "--timeout", "1h", who=AGENT)) == []
    assert rules(env, job(env, "c1", p, "--timeout", "1h", who=AGENT,
                          slot="2")) == ["agent-owner"]
    assert rules(env, job(env, "c1", p, "--timeout", "1h", who=AGENT,
                          slot="any")) == []
    other = env["cases"] / "other" / "c2"
    other.mkdir(parents=True)
    (other / "BOUT.inp").write_text("")
    assert rules(env, job(env, str(other), p, "--timeout", "1h",
                          who=AGENT)) == ["agent-campaign"]
    assert rules(env, job(env, str(other), p)) == []   # the user: no owner


def test_dash_dash_jobs(env):
    clone = env["clone"]
    tests = ["--cwd", str(clone), "--timeout", "30m", "--",
             "./build-x/hermes_unit_tests"]
    j = job(env, *tests, who=AGENT)
    assert rules(env, j) == [] and j["campaign"] == "camp"
    assert j["cwd"] == str(clone) and j["case"] is None
    for cmd in (["sdrun.py", "-d", "c1"], ["bash", "-c", "mpirun -np 2 x"],
                ["./build-x/hermes-3", "-d", "c1"]):
        j = job(env, "--cwd", str(clone), "--timeout", "1h", "--", *cmd,
                who=AGENT)
        assert rules(env, j) == ["agent-command"], cmd
    j = job(env, "--cwd", str(clone), "--", "sdrun.py", "-d", "c1")
    assert rules(env, j) == []    # the user's -- jobs may name anything
    j = job(env, "--cwd", str(clone), "--", "python3",
            "/home/u/hermes-3/tools/plot.py", who=AGENT)
    assert rules(env, j) == ["agent-timeout"]   # a hermes-3/ directory is fine
    with pytest.raises(ValueError):
        job(env, "c1", "--", "ls")
    with pytest.raises(ValueError):
        job(env, "--cwd", str(clone), "--")


def test_after(env):
    case(env)
    p = f"-p={env['exe']}"
    j = job(env, "c1", p, "--after", "nope")
    assert rules(env, j) == ["after-unknown"]
    prior = job(env, "c1", p)
    store(env, "queue", "1", prior)
    j = job(env, "c1", p, "-append", "--after", prior["id"])
    assert rules(env, j) == []
    assert rules(env, j, "start") == ["after-pending"]
    (env["state"] / "queue" / "1" / f"{prior['id']}.json").unlink()
    store(env, "done", "1", dict(prior, outcome="failed"))
    assert rules(env, j, "start") == ["after-failed"]
    store(env, "done", "1", dict(prior, outcome="ok"))
    assert rules(env, j, "start") == []


def test_after_defers_case_checks_to_start(env):
    p = f"-p={env['exe']}"
    prior = job(env, "--", "true")
    store(env, "queue", "1", prior)
    j = job(env, "later", p, "--after", prior["id"])
    assert rules(env, j) == []
    (env["state"] / "queue" / "1" / f"{prior['id']}.json").unlink()
    store(env, "done", "1", dict(prior, outcome="ok"))
    assert rules(env, j, "start") == ["case-missing"]


def test_a_job_queues_no_jobs(env):
    case(env)
    j = job(env, "c1", f"-p={env['exe']}", who={"kind": "job", "job": "j0"})
    assert rules(env, j) == ["nested-job"]


def test_mandate_ended(env, tmp_path):
    case(env)
    j = job(env, "c1", f"-p={env['exe']}")
    j["mandate"] = {"state": str(tmp_path / "gone.json"),
                    "expires": "2999-01-01T00:00"}
    assert rules(env, j, "start") == ["mandate-ended"]
    (tmp_path / "gone.json").write_text("{}")
    assert rules(env, j, "start") == []


def test_config_stops_on_a_bad_file(tmp_path, monkeypatch):
    bad = tmp_path / "slots.json"
    monkeypatch.setenv("SDTOOLS_SLOT_CONFIG", str(bad))
    assert slotlib.config()["enforce"] is False     # no file: no slots here
    bad.write_text('{"enforce": true, "cores": ')
    with pytest.raises(SystemExit, match="unreadable"):
        slotlib.config()
    bad.write_text('{"enforce": true}')
    with pytest.raises(SystemExit, match="names no cores"):
        slotlib.config()


def test_owner_by_mode(env):
    cfg = env["cfg"]
    assert slotlib.owner(cfg, "1") == "camp"
    assert slotlib.owner(cfg, "2") is None
    os.remove(slotlib.path(cfg, "1", "mode"))
    (env["cases"] / "runplan1.sh").write_text("# owner: other\n")
    assert slotlib.owner(cfg, "1") == "other"
    (env["cases"] / "runplan1.sh").write_text("# owner: none\n")
    assert slotlib.owner(cfg, "1") is None
    assert slotlib.owner_clone(cfg, "camp") == str(env["clone"])


def test_sdrun_parse_args_raises():
    with pytest.raises(sdrun.UsageError, match="-p requires a value"):
        sdrun.parse_args(["-p"])
    assert sdrun.parse_args(["-h", "-p"])["help"] is True
    r = subprocess.run([sys.executable, os.path.join(CLI, "sdrun.py"), "-h"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("Usage: sdrun.py")
    r = subprocess.run([sys.executable, os.path.join(CLI, "sdrun.py"), "-p"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "-p requires a value" in r.stderr
    assert "Usage: sdrun.py" in r.stderr
