"""slotd.py in queue mode, with runq.py, runlaunch.py, runwait.py and
hcompile.sh around it. slotd runs against a throwaway config; the real
sdrun.py starts a fake mpirun, which runs a fake hermes-3 script."""
import json
import os
import subprocess
import sys
import time

import pytest

CLI = os.path.join(os.path.dirname(__file__), "..", "cli")
HCOMPILE = os.path.join(os.path.dirname(__file__), "..", "install_scripts",
                        "hcompile.sh")
sys.path.insert(0, CLI)
import runq  # noqa: E402
import runwait  # noqa: E402
import slotlib  # noqa: E402

PY = sys.executable
USER = {"kind": "user"}

MPIRUN = """#!/bin/bash
while [ $# -gt 0 ]; do
  case "$1" in
    --bind-to|-np) shift 2 ;;
    --oversubscribe) shift ;;
    --mca) shift 3 ;;
    *) break ;;
  esac
done
exec "$@"
"""

HERMES = """#!/bin/bash
case="$2"
env > "$case/run.env"
grep Cpus_allowed_list /proc/self/status >> "$case/run.env"
echo "$@" > "$case/run.args"
[ -f "$case/sleep" ] && sleep "$(cat "$case/sleep")"
printf 'fake run\\nRun time : 1 s\\n' > "$case/BOUT.log.0"
exit "$(cat "$case/exit" 2>/dev/null || echo 0)"
"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Slot 1 (cores 0-1) and slot 2 (core 2) in queue mode, both owned by
    campaign camp, whose cases are cases/camp and whose clone holds the fake
    build-x/hermes-3."""
    cases = tmp_path / "cases"
    camp = cases / "camp"
    camp.mkdir(parents=True)
    clone = tmp_path / "clone"
    (clone / "build-x").mkdir(parents=True)
    exe = clone / "build-x" / "hermes-3"
    exe.write_text(HERMES)
    exe.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "mpirun").write_text(MPIRUN)
    (bin_dir / "mpirun").chmod(0o755)
    (tmp_path / "camp.toml").write_text(
        f'name = "camp"\ncases = "{camp}"\nclone = "{clone}"\n')
    cfg = {"enforce": True, "cases": str(cases),
           "cores": {"1": "0-1", "2": "2"}, "state_dir": str(tmp_path / "st"),
           "campaigns": str(tmp_path / "*.toml")}
    (tmp_path / "slots.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("SDTOOLS_SLOT_CONFIG", str(tmp_path / "slots.json"))
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("CLAUDECODE", "1")   # slotd must drop it from runs
    for k in ("SDTOOLS_SLOT_ENFORCE", "HERMES_SLOT", "RUNQ_JOB"):
        monkeypatch.delenv(k, raising=False)
    cfg = slotlib.config()
    for n in ("1", "2"):
        open(slotlib.path(cfg, n, "mode"), "w").write("queue\n")
        open(slotlib.path(cfg, n, "owner"), "w").write(
            json.dumps({"campaign": "camp"}))
    procs = []

    def slotd(slot="1"):
        p = subprocess.Popen([PY, os.path.join(CLI, "slotd.py"), slot,
                              "--no-screen-check"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append(p)
        until(lambda: slotlib.read_state(cfg, slot)["alive"])
        return p

    yield {"cfg": cfg, "cases": cases, "camp": camp, "clone": clone,
           "exe": str(exe), "slotd": slotd, "tmp": tmp_path}
    for p in procs:
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()


def until(cond, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = cond()
        if v:
            return v
        time.sleep(0.1)
    raise AssertionError("condition not met in time")


def case(env, name="c1", **files):
    d = env["camp"] / name
    d.mkdir(exist_ok=True)
    (d / "BOUT.inp").write_text("error_on_unused_options = true\n")
    for k, v in files.items():
        (d / k.replace("_", ".")).write_text(v)
    return d


def add(env, *args, slot="1"):
    assert runq.main(["add", slot, *args], who=USER,
                     cwd=str(env["camp"])) == 0


def done(env, n=1, slot="1"):
    return until(lambda: len(slotlib.read_jobs(env["cfg"], "done", slot)) >= n
                 and slotlib.read_jobs(env["cfg"], "done", slot))


def test_a_case_job_runs_pinned_with_a_clean_environment(env):
    c = case(env)
    env["slotd"]()
    add(env, "c1", f"-p={env['exe']}")
    [j] = done(env)
    assert j["outcome"] == "ok" and j["exit"] == 0 and j["slot_ran"] == "1"
    run_env = (c / "run.env").read_text()
    assert "HERMES_SLOT=1\n" in run_env and f"RUNQ_JOB={j['id']}\n" in run_env
    assert "CLAUDECODE" not in run_env
    assert "Cpus_allowed_list:\t0-1" in run_env
    assert "Running:" in (c / "BOUT.log.console").read_text()   # sdrun banner


def test_a_continuation_appends_to_the_console(env):
    c = case(env)
    env["slotd"]()
    add(env, "c1", f"-p={env['exe']}")
    done(env)
    (c / "BOUT.log.console").write_text("first run\n")
    add(env, "c1", f"-p={env['exe']}", "-restart", "-append")
    [_, j] = done(env, 2)
    assert j["outcome"] == "ok"
    console = (c / "BOUT.log.console").read_text()
    assert console.startswith("first run\n") and "Running:" in console


def test_pipefail_keeps_the_exit_code(env):
    case(env, exit="3")
    env["slotd"]()
    add(env, "c1", f"-p={env['exe']}")
    [j] = done(env)
    assert j["outcome"] == "failed" and j["exit"] == 3


def test_a_command_job_logs_to_done(env):
    env["slotd"]()
    add(env, "--", "sh", "-c", "echo hi; pwd")
    [j] = done(env)
    log = slotlib.job_path(env["cfg"], "done", "1", j["id"])[:-5] + ".log"
    assert j["outcome"] == "ok"
    assert open(log).read() == f"hi\n{env['camp']}\n"


def test_start_checks_refuse_a_changed_binary(env):
    case(env)
    assert runq.main(["hold", "1"], who=USER) == 0
    env["slotd"]()
    add(env, "c1", f"-p={env['exe']}")
    with open(env["exe"], "a") as f:
        f.write("# rebuilt\n")
    assert runq.main(["go", "1"], who=USER) == 0
    [j] = done(env)
    assert j["outcome"] == "refused" and "changed" in j["why"][0]


def test_after_a_failed_job_the_next_is_skipped(env):
    case(env, exit="1")
    (env["camp"] / "batch.txt").write_text(
        f"c1 -p={env['exe']}\n--after prev -- true\n")
    add(env, "-f", "batch.txt")
    env["slotd"]()
    first, second = done(env, 2)
    assert first["outcome"] == "failed" and second["outcome"] == "skipped"


def test_a_timeout_stops_the_job(env):
    case(env, sleep="60")
    env["slotd"]()
    add(env, "c1", f"-p={env['exe']}", "--timeout", "2s")
    [j] = done(env)
    assert j["outcome"] == "timeout" and j["wall_s"] < 15


def test_cancel_stops_the_job_and_holds_the_slot(env):
    case(env, sleep="60")
    env["slotd"]()
    add(env, "c1", f"-p={env['exe']}")
    until(lambda: slotlib.read_state(env["cfg"], "1").get("state") ==
          "running")
    assert runq.main(["cancel", "1"], who=USER) == 0
    [j] = done(env)
    assert j["outcome"] == "cancelled"
    until(lambda: slotlib.read_state(env["cfg"], "1").get("state") == "held")
    assert runq.main(["go", "1"], who=USER) == 0
    until(lambda: slotlib.read_state(env["cfg"], "1").get("state") == "idle")


def test_ctrl_c_stops_the_job_and_holds_the_slot(env):
    case(env, sleep="60")
    p = env["slotd"]()
    add(env, "c1", f"-p={env['exe']}")
    until(lambda: slotlib.read_state(env["cfg"], "1").get("state") ==
          "running")
    p.send_signal(2)
    [j] = done(env)
    assert j["outcome"] == "cancelled"
    assert os.path.exists(slotlib.path(env["cfg"], "1", "hold"))
    assert p.poll() is None      # slotd itself keeps running


def test_a_job_left_running_ends_lost(env):
    j = {"id": "20260101-000000-000000-abcd", "slot": "1", "by": USER}
    slotlib.write_job(env["cfg"], "running", "1", j)
    env["slotd"]()
    [j] = done(env)
    assert j["outcome"] == "lost"


def test_two_slotds_share_the_any_queue(env):
    for k in range(6):
        add(env, "--", "true", slot="any")
    env["slotd"]("1")
    env["slotd"]("2")
    until(lambda: len(slotlib.read_jobs(env["cfg"], "done", "1")) +
          len(slotlib.read_jobs(env["cfg"], "done", "2")) == 6)
    ids = [j["id"] for n in ("1", "2")
           for j in slotlib.read_jobs(env["cfg"], "done", n)]
    assert len(set(ids)) == 6
    assert slotlib.read_jobs(env["cfg"], "queue", "any") == []


def test_jobs_on_disjoint_cores_run_together(env):
    for name in ("c1", "c2"):
        case(env, name, sleep="3")
    add(env, "c1", f"-p={env['exe']}", "-c=0")
    add(env, "c2", f"-p={env['exe']}", "-c=1")
    env["slotd"]()
    until(lambda: len(slotlib.read_state(env["cfg"], "1").get("jobs") or [])
          == 2)
    assert [j["outcome"] for j in done(env, 2)] == ["ok", "ok"]


def test_validate_runs_in_a_copy(env):
    c = case(env)
    env["slotd"]()
    add(env, "c1", f"-p={env['exe']}", "--validate")
    [j] = done(env)
    assert j["outcome"] == "ok"
    assert (c / "validate.log.0").exists() and (c / "validate.console").exists()
    assert not (c / "BOUT.log.0").exists()
    copy = env["tmp"] / "st" / "validate" / j["id"] / "c1"
    assert "input:validate=true" in (copy / "run.args").read_text()


def test_rerun_moves_the_old_output(env):
    c = case(env, BOUT_log_0="old\n", BOUT_dmp_0_nc="x")
    env["slotd"]()
    add(env, "c1", f"-p={env['exe']}", "--rerun")
    [j] = done(env)
    [old] = list(c.glob("overwritten-*"))
    assert (old / "BOUT.log.0").read_text() == "old\n"
    assert (old / "BOUT.dmp.0.nc").exists()
    assert "Run time" in (c / "BOUT.log.0").read_text()


def test_slotd_restarts_itself_on_a_mode_change(env):
    p = env["slotd"]()
    first = slotlib.read_state(env["cfg"], "1")
    assert first["mode"] == "queue"
    open(slotlib.path(env["cfg"], "1", "mode"), "w").write("runplan\n")
    st = until(lambda: (lambda s: s.get("mode") == "runplan" and s["alive"]
                        and s)(slotlib.read_state(env["cfg"], "1")))
    assert st["slotd_pid"] == p.pid and st["started"] > first["started"]


def test_a_bad_mode_file_stops_the_slot(env):
    case(env)
    open(slotlib.path(env["cfg"], "1", "mode"), "w").write("quuee\n")
    env["slotd"]()
    st = until(lambda: slotlib.read_state(env["cfg"], "1"))
    assert st["state"] == "stopped" and "quuee" in st["mode_error"]


def test_runlaunch_on_a_queue_slot_prints_runq_lines(env):
    case(env)
    (env["cases"] / "runplan1.sh").write_text(
        f"# owner: camp\nset -u\ncd {env['camp']}\n"
        f"timeout 1h sdrun.py -y -s=1 -p={env['exe']} -d c1 -restart "
        "2>&1 | tee c1/BOUT.log.console\n"
        f"cd {env['clone']} && ctest -j 4\n")
    r = subprocess.run([PY, os.path.join(CLI, "runlaunch.py"), "1"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "runs its queue" in r.stdout
    assert f"runq add 1 {env['camp']}/c1 -p={env['exe']} -restart " \
           "--timeout 1h" in r.stdout
    assert f"runq add 1 --cwd {env['clone']} -- ctest -j 4" in r.stdout


def test_runwait_sees_a_queue_slot_busy_or_held(env):
    cfg = env["cfg"]
    assert not runwait.slot_busy(cfg, "1")
    add(env, "--", "true")
    assert runwait.slot_busy(cfg, "1") and not runwait.slot_held(cfg, "1")
    assert runq.main(["hold", "1"], who=USER) == 0
    assert not runwait.slot_busy(cfg, "1") and runwait.slot_held(cfg, "1")


def hcompile(env, **extra):
    e = dict(os.environ, **extra)
    return subprocess.run(["bash", HCOMPILE, "-S", "1", "-y", "-r",
                           str(env["tmp"] / "no-hermes")],
                          capture_output=True, text=True, env=e, timeout=60)


def test_hcompile_on_a_running_slot_refuses(env):
    lock = open(slotlib.path(env["cfg"], "1", "lock"), "a")
    import fcntl
    fcntl.flock(lock, fcntl.LOCK_EX)
    r = hcompile(env)
    assert r.returncode == 1 and "runq hold 1" in r.stdout
    # Inside slot 1's own run, the build already holds the lock.
    r = hcompile(env, HERMES_SLOT="1")
    assert r.returncode == 2 and "Hermes directory not found" in r.stdout
    lock.close()
    assert "Hermes directory not found" in hcompile(env).stdout


def test_restart_refuses_a_busy_slot_or_a_missing_screen(env, monkeypatch,
                                                         capsys):
    # Never reach the live screens h1-h3 from a test.
    monkeypatch.setattr(runq, "screen_of", lambda slot: None)
    monkeypatch.setattr(runq.os, "kill", lambda *a: pytest.fail("kill"))
    j = {"id": "20260101-000000-000000-abcd", "slot": "1", "by": USER}
    slotlib.write_job(env["cfg"], "running", "1", j)
    assert runq.main(["restart", "1"], who=USER) == 1
    out = capsys.readouterr().out
    assert "slot 1 runs a job" in out and "no screen session h1" in out
    assert runq.main(["restart", "2"], who=USER) == 1
    assert "no screen session h2" in capsys.readouterr().out


def test_a_mandate_job_runs_only_on_the_mandates_slots(env):
    add(env, "--", "true", slot="any")
    [j] = slotlib.read_jobs(env["cfg"], "queue", "any")
    j["only_slots"] = ["1"]
    slotlib.write_job(env["cfg"], "queue", "any", j)
    env["slotd"]("2")
    time.sleep(3)
    assert slotlib.read_jobs(env["cfg"], "queue", "any") != []
    env["slotd"]("1")
    [j] = done(env)
    assert j["outcome"] == "ok" and j["slot_ran"] == "1"


def test_slotd_restarts_for_a_commit_not_for_an_edit(env, tmp_path):
    """A copy of cli/ in its own git repository: an edit to slotd.py leaves
    the running slotd alone until the edit is committed."""
    import shutil
    repo = tmp_path / "repo"
    shutil.copytree(CLI, repo / "cli",
                    ignore=shutil.ignore_patterns("__pycache__"))
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c",
           "user.email=t@t"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "a"], check=True)
    p = subprocess.Popen([PY, str(repo / "cli" / "slotd.py"), "1",
                          "--no-screen-check"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        cfg = env["cfg"]
        first = until(lambda: (lambda s: s["alive"] and s)(
            slotlib.read_state(cfg, "1")))
        with open(repo / "cli" / "slotd.py", "a") as f:
            f.write("# an edit\n")
        time.sleep(3)
        assert slotlib.read_state(cfg, "1")["started"] == first["started"]
        subprocess.run(git + ["commit", "-qam", "b"], check=True)
        until(lambda: slotlib.read_state(cfg, "1")["started"] >
              first["started"] and slotlib.read_state(cfg, "1")["alive"])
        assert slotlib.read_state(cfg, "1")["slotd_pid"] == p.pid
    finally:
        p.terminate()
        p.wait(10)
