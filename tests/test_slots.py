"""slotd.py, runlaunch.py and the slot checks in sdrun.py and runwait.py,
run against a throwaway cases directory and slot config."""
import json
import os
import subprocess
import sys
import time

import pytest

CLI = os.path.join(os.path.dirname(__file__), "..", "cli")
sys.path.insert(0, CLI)
import runwait  # noqa: E402
import slotlib  # noqa: E402

PY = sys.executable


@pytest.fixture
def env(tmp_path, monkeypatch):
    cases = tmp_path / "cases"
    (cases / "camp").mkdir(parents=True)
    cfg = {"enforce": True, "cases": str(cases),
           "cores": {"1": "0", "2": "1"}, "state_dir": str(tmp_path / "st")}
    cfg_path = tmp_path / "slots.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setenv("SDTOOLS_SLOT_CONFIG", str(cfg_path))
    monkeypatch.delenv("SDTOOLS_SLOT_ENFORCE", raising=False)
    monkeypatch.delenv("HERMES_SLOT", raising=False)
    for n in ("1", "2"):
        (cases / f"runplan{n}.sh").write_text(
            f"# owner: none\nset -u\ncd {cases}/camp\n")
    return cases


def plan(cases, body, owner="camp", slot="1"):
    (cases / f"runplan{slot}.sh").write_text(
        f"# owner: {owner}\nset -u\ncd {cases}/camp\n{body}\n")


def launch(*args):
    return subprocess.run([PY, os.path.join(CLI, "runlaunch.py"), *args],
                          capture_output=True, text=True)


@pytest.fixture
def slotd(env):
    p = subprocess.Popen([PY, os.path.join(CLI, "slotd.py"), "1",
                          "--no-screen-check"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cfg = slotlib.config()
    for _ in range(50):
        if slotlib.read_state(cfg, "1")["alive"]:
            break
        time.sleep(0.1)
    yield p
    p.terminate()
    p.wait()


def sdrun_line(case, slot="1", extra=""):
    return (f"timeout 1m sdrun.py -y -s={slot} -p=/bin/true -d {case} {extra} "
            f"2>&1 | tee {case}/BOUT.log.console")


def test_refuses_without_owner(env, slotd):
    plan(env, "echo hi", owner="none")
    r = launch("1", "--check")
    assert r.returncode == 1 and "no owner" in r.stdout


def test_refuses_without_slotd(env):
    plan(env, "echo hi")
    r = launch("1", "--check")
    assert r.returncode == 1 and "slotd 1 is not running" in r.stdout


def test_refuses_extra_runplan(env, slotd):
    plan(env, "echo hi")
    (env / "camp" / "runplan1.sh").write_text("echo\n")
    r = launch("1", "--check")
    assert r.returncode == 1 and "extra runplan" in r.stdout


def test_refuses_case_outside_owner(env, slotd):
    other = env / "other" / "c1"
    other.mkdir(parents=True)
    plan(env, sdrun_line(other))
    r = launch("1", "--check")
    assert r.returncode == 1 and "not inside 'camp'" in r.stdout


def test_refuses_wrong_slot_and_missing_tee(env, slotd):
    case = env / "camp" / "c1"
    case.mkdir()
    plan(env, f"sdrun.py -y -s=2 -d c1")
    r = launch("1", "--check")
    assert "not on slot 1" in r.stdout and "BOUT.log.console" in r.stdout


def test_refuses_cores_outside_slot(env, slotd):
    case = env / "camp" / "c1"
    case.mkdir()
    plan(env, sdrun_line("c1").replace("-s=1", "-c=0-1"))
    r = launch("1", "--check")
    assert r.returncode == 1 and "leaves slot 1" in r.stdout


def test_refuses_finished_scratch_case(env, slotd):
    case = env / "camp" / "c1"
    case.mkdir()
    (case / "BOUT.log.0").write_text("Run time : 1 m\n")
    plan(env, sdrun_line("c1"))
    r = launch("1", "--check")
    assert r.returncode == 1 and "already finished" in r.stdout
    plan(env, sdrun_line("c1", extra="-restart"))
    assert launch("1", "--check").returncode == 0


def test_refuses_empty_runplan(env, slotd):
    plan(env, "# nothing queued")
    r = launch("1", "--check")
    assert r.returncode == 1 and "queues no command" in r.stdout


def test_launch_runs_pinned_with_slot_env(env, slotd):
    out = env / "camp" / "out.txt"
    plan(env, f"env > {out}; grep Cpus_allowed_list /proc/self/status >> {out}; "
              "sleep 2")
    r = launch("1")
    assert r.returncode == 0, r.stdout
    assert "started" in r.stdout
    cfg = slotlib.config()
    assert not slotlib.lock_free(cfg, "1")
    r2 = launch("1", "--check")
    assert r2.returncode == 1 and "busy" in r2.stdout
    for _ in range(100):
        if slotlib.read_state(cfg, "1").get("state") == "idle":
            break
        time.sleep(0.1)
    st = slotlib.read_state(cfg, "1")
    assert st["state"] == "idle" and st["last_exit"] == 0
    text = out.read_text()
    assert "HERMES_SLOT=1" in text
    assert text.rstrip().split()[-1] == "0"
    assert slotlib.lock_free(cfg, "1")


def test_slotd_refuses_outside_its_screen(env, monkeypatch):
    monkeypatch.setenv("STY", "123.h2")
    r = subprocess.run([PY, os.path.join(CLI, "slotd.py"), "1"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "only inside screen h1" in r.stderr


def sdrun(*args, **env):
    e = dict(os.environ, **env)
    return subprocess.run([PY, os.path.join(CLI, "sdrun.py"), *args],
                          capture_output=True, text=True, env=e)


def test_sdrun_refuses_outside_slotd(env):
    r = sdrun("-y", "-s=1", "-d", "nowhere")
    assert r.returncode == 1 and "only through slotd" in r.stderr


def test_sdrun_refuses_other_slot_and_cores(env):
    r = sdrun("-y", "-s=2", "-d", "nowhere", HERMES_SLOT="1")
    assert r.returncode == 1 and "belongs to slot 1" in r.stderr
    r = sdrun("-y", "-c=0-1", "-d", "nowhere", HERMES_SLOT="1")
    assert r.returncode == 1 and "leave slot 1" in r.stderr


def test_sdrun_passes_slot_check_inside_slotd(env):
    r = sdrun("-y", "-s=1", "-d", "nowhere", HERMES_SLOT="1")
    assert r.returncode == 1 and "case directory not found" in r.stderr


def test_sdrun_unenforced(env):
    r = sdrun("-y", "-s=1", "-d", "nowhere", SDTOOLS_SLOT_ENFORCE="0")
    assert "case directory not found" in r.stderr


def test_runwait_counts_a_busy_slot(env):
    calls = iter([True, True, False, False, False])
    why, states = runwait.wait_cases(lambda: [], 60, time.time() + 30,
                                     idle_s=0, busy_fn=lambda: next(calls))
    assert why == "done"


def test_refuses_chained_and_escaping_lines(env, slotd):
    case = env / "camp" / "c1"
    case.mkdir()
    plan(env, "cd c1 && sdrun.py -y -s=2 -d . 2>&1 | tee BOUT.log.console")
    r = launch("1", "--check")
    assert r.returncode == 1 and "not on slot 1" in r.stdout
    plan(env, "HERMES_SLOT=2 " + sdrun_line("c1"))
    r = launch("1", "--check")
    assert r.returncode == 1 and "sets the slot" in r.stdout


def test_runplan_keeps_lock_after_slotd_dies(env, slotd):
    plan(env, "sleep 3")
    assert launch("1").returncode == 0
    slotd.kill()
    slotd.wait()
    cfg = slotlib.config()
    assert not slotlib.lock_free(cfg, "1")
    p = subprocess.Popen([PY, os.path.join(CLI, "slotd.py"), "1",
                          "--no-screen-check"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(50):
            if slotlib.read_state(cfg, "1").get("state") == "orphan":
                break
            time.sleep(0.1)
        assert slotlib.read_state(cfg, "1")["state"] == "orphan"
        assert launch("1", "--check").returncode == 1
    finally:
        p.terminate()
        p.wait()


def test_slotd_ignores_stale_trigger(env, slotd):
    out = env / "camp" / "ran.txt"
    plan(env, f"touch {out}")
    cfg = slotlib.config()
    with open(slotlib.path(cfg, "1", "go"), "w") as f:
        json.dump({"id": "old", "time": time.time() - 60}, f)
    time.sleep(2.5)
    assert not out.exists()
