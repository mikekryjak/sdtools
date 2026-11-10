"""runq's watch flags (stalled, idle) in `runq ls`, and `runq wait`. Each
test calls runq.main() in-process against the throwaway slot config of
test_runcheck.py; no slotd runs, so the tests move jobs by hand."""
import json
import os
import sys
import time

CLI = os.path.join(os.path.dirname(__file__), "..", "cli")
sys.path.insert(0, CLI)
sys.path.insert(0, os.path.dirname(__file__))
import runq  # noqa: E402
import slotlib  # noqa: E402
from test_runcheck import AGENT, USER, case, env  # noqa: E402,F401

OLD = "2026-01-01T00:00:00"   # long before any test runs


def run(env, *argv, who=USER, cwd=None):
    return runq.main(list(argv), who=who, cwd=str(cwd or env["camp"]))


def add(env, name, slot="1", who=USER, extra=()):
    case(env, name)
    args = ["add", slot, name, f"-p={env['exe']}", *extra]
    if who["kind"] == "agent":
        args += ["--timeout", "1h"]
    assert run(env, *args, who=who) == 0
    return max(runq.store_jobs(env["cfg"], "queue", slot),
               key=lambda j: j["id"])


def edit(env, job, where, slot="1", **fields):
    """Rewrite a job's fields and put it in <where>/<slot>/."""
    for w in ("queue", "running", "done"):
        p = slotlib.job_path(env["cfg"], w, job["slot"], job["id"])
        if os.path.exists(p):
            os.remove(p)
    job.update(fields)
    slotlib.write_job(env["cfg"], where, slot, job)
    return job


def ls_json(env, capsys):
    capsys.readouterr()
    assert run(env, "ls", "--json") == 0
    return json.loads(capsys.readouterr().out)


def test_a_quiet_running_job_is_flagged_stalled(env, capsys):
    j = add(env, "c1")
    console = env["camp"] / "c1" / "BOUT.log.console"
    console.write_text("x")
    edit(env, j, "running", started=OLD, output=str(console))
    os.utime(console, (time.time() - 3 * 3600,) * 2)
    [f] = ls_json(env, capsys)["flags"]
    assert f["flag"] == "stalled" and f["job"] == j["id"]
    assert run(env, "ls") == 0
    assert "FLAG stalled: job " in capsys.readouterr().out
    os.utime(console)            # it writes again: no flag
    assert ls_json(env, capsys)["flags"] == []


def test_a_just_started_job_is_not_stalled(env, capsys):
    j = add(env, "c1")
    edit(env, j, "running", started=runq.now())
    assert ls_json(env, capsys)["flags"] == []


def test_a_free_slot_with_old_work_queued_is_flagged_idle(env, capsys):
    j = add(env, "c1")
    edit(env, j, "queue", added=OLD)
    [f] = ls_json(env, capsys)["flags"]
    assert f["flag"] == "idle" and f["slot"] == "1"
    assert "slotd is not running" in f["message"]


def test_no_idle_flag_when_held_new_or_waiting_on_after(env, capsys):
    j = add(env, "c1")
    assert ls_json(env, capsys)["flags"] == []          # added just now
    edit(env, j, "queue", added=OLD)
    assert run(env, "hold", "1") == 0
    assert ls_json(env, capsys)["flags"] == []          # held
    assert run(env, "go", "1") == 0
    first = add(env, "c2", slot="2")
    edit(env, first, "running", slot="2", started=runq.now())
    edit(env, j, "queue", after=first["id"])
    assert ls_json(env, capsys)["flags"] == []          # its --after runs


def test_an_any_job_no_slot_may_take_is_flagged(env, capsys):
    case(env, "c1")
    assert run(env, "add", "any", "c1", f"-p={env['exe']}", "--timeout",
               "1h", who=AGENT) == 0
    [j] = runq.store_jobs(env["cfg"], "queue", "any")
    edit(env, j, "queue", slot="any", added=OLD)
    os.remove(slotlib.path(env["cfg"], "1", "owner"))
    [f] = ls_json(env, capsys)["flags"]
    assert f["slot"] == "any" and f["job"] == j["id"]


def test_wait_returns_done_when_nothing_is_watched(env, capsys):
    assert run(env, "wait", "--max-s", "5") == 0
    assert capsys.readouterr().out == "done\n"


def test_wait_mine_watches_only_its_sessions_jobs(env, capsys, monkeypatch):
    monkeypatch.setattr(runq, "WAIT_POLL_S", 0)
    add(env, "c1")                                    # the user's job
    capsys.readouterr()
    assert run(env, "wait", "--mine", "--max-s", "0", who=AGENT) == 0
    assert capsys.readouterr().out == "done\n"
    mine = add(env, "c2", who=AGENT)
    capsys.readouterr()
    assert run(env, "wait", "--mine", "--max-s", "0", who=AGENT) == 0
    out = capsys.readouterr().out
    assert out.startswith("deadline\n") and f"{mine['id']}\tqueue" in out
    assert run(env, "wait", "--mine") == 2            # the user has no session


def test_wait_wakes_on_a_stall(env, capsys, monkeypatch):
    monkeypatch.setattr(runq, "WAIT_POLL_S", 0)
    j = add(env, "c1", who=AGENT)
    edit(env, j, "running", started=OLD)
    capsys.readouterr()
    assert run(env, "wait", "1", "--stall-min", "1") == 0
    out = capsys.readouterr().out
    assert out.startswith("stalled\nFLAG stalled: job ")
    assert f"{j['id']}\trunning" in out


def test_wait_reports_a_held_slot_and_each_ended_job(env, capsys,
                                                     monkeypatch):
    monkeypatch.setattr(runq, "WAIT_POLL_S", 0)
    j = add(env, "c1", who=AGENT)
    assert run(env, "hold", "1", "checking") == 0
    capsys.readouterr()
    assert run(env, "wait", "--mine", who=AGENT) == 0
    out = capsys.readouterr().out
    assert out.startswith("held\nHELD slot 1: user: checking\n")
    assert run(env, "go", "1") == 0
    edit(env, j, "done", outcome="ok", ended=runq.now())
    capsys.readouterr()
    assert run(env, "wait", "1") == 0
    assert capsys.readouterr().out == "done\n"


def test_wait_bad_usage_exits_2(env, capsys):
    assert run(env, "wait", "7") == 2
    assert run(env, "wait", "--max-s") == 2
    assert run(env, "wait", "--stall-min", "x") == 2
