"""runq.py, the run queue command, and the slotlib job store. Each test calls
runq.main() in-process with a chosen caller, against throwaway cases and slot
config (the env fixture of test_runcheck.py)."""
import datetime as dt
import json
import os
import sys

CLI = os.path.join(os.path.dirname(__file__), "..", "cli")
sys.path.insert(0, CLI)
sys.path.insert(0, os.path.dirname(__file__))
import runcheck  # noqa: E402
import runq  # noqa: E402
import slotlib  # noqa: E402
from test_runcheck import AGENT, USER, case, env  # noqa: E402,F401


def run(env, *argv, who=USER, cwd=None):
    return runq.main(list(argv), who=who, cwd=str(cwd or env["camp"]))


def stored(env, where="queue", slot="1"):
    return runq.store_jobs(env["cfg"], where, slot)


def test_add_then_ls(env, capsys):
    case(env)
    assert run(env, "add", "1", "c1", f"-p={env['exe']}") == 0
    [j] = stored(env)
    assert j["case"] == str(env["camp"] / "c1") and j["by"] == USER
    assert f"queued {j['id']} on slot 1" in capsys.readouterr().out
    assert run(env, "ls") == 0
    out = capsys.readouterr().out
    assert "slot 1  camp  IDLE" in out and f"queued  {j['id']}" in out
    assert run(env, "ls", "--json") == 0
    snap = json.loads(capsys.readouterr().out)
    assert snap["slots"]["1"]["queued"][0]["id"] == j["id"]
    assert snap["slots"]["2"]["owner"] is None


def test_dry_run_queues_nothing(env, capsys):
    case(env)
    assert run(env, "add", "1", "c1", f"-p={env['exe']}", "--dry-run") == 0
    assert "dry run passes" in capsys.readouterr().out
    assert stored(env) == []
    assert run(env, "add", "1", "c1", "--dry-run") == 1
    assert stored(env) == []


def test_a_refusal_prints_its_fix(env, capsys):
    case(env)
    assert run(env, "add", "1", "c1") == 1
    out = capsys.readouterr().out
    assert out.startswith("runq add: refused, nothing queued\n- name the "
                          "binary")
    assert "\n  fix: runq add 1 c1 '-p=<path to hermes-3>'" in out


def test_warnings_are_kept_in_the_job(env, capsys):
    case(env, inp="error_on_unused_options = false\n")
    assert run(env, "add", "1", "c1", f"-p={env['exe']}") == 0
    assert "- warning: " in capsys.readouterr().out
    assert "error_on_unused_options" in stored(env)[0]["warnings"][0]


def test_bad_usage_exits_2(env, capsys):
    assert run(env, "add", "1") == 2
    assert run(env, "add", "9", "c1") == 2
    assert run(env, "add", "1", "c1", "-np") == 2      # sdrun.py UsageError
    assert run(env, "frob") == 2
    assert run(env, "rm") == 2
    assert stored(env) == []


def test_a_runplan_slot_takes_no_job(env, capsys):
    case(env)
    os.remove(slotlib.path(env["cfg"], "2", "mode"))
    assert run(env, "add", "2", "c1", f"-p={env['exe']}") == 1
    assert "runs runplans" in capsys.readouterr().out


def test_batch_is_all_or_nothing(env, capsys):
    for c in ("c1", "c2"):
        case(env, c)
    batch = env["camp"] / "batch.txt"
    batch.write_text(f"# two good lines and one bad\nc1 -p={env['exe']}\n\n"
                     f"c2 -p={env['exe']}\nc3 -p={env['exe']}\n")
    assert run(env, "add", "1", "-f", "batch.txt") == 1
    assert "- line 5: " in capsys.readouterr().out
    assert stored(env) == []
    batch.write_text(f"c1 -p={env['exe']}\nc2 -p={env['exe']}\n")
    assert run(env, "add", "1", "-f", "batch.txt") == 0
    assert [j["case"][-2:] for j in stored(env)] == ["c1", "c2"]


def test_batch_after_prev_chains_the_lines(env, capsys):
    case(env)
    batch = env["camp"] / "batch.txt"
    batch.write_text(f"c1 -p={env['exe']}\n"
                     f"c1 -p={env['exe']} -restart -append --after prev\n")
    assert run(env, "add", "1", "-f", "batch.txt") == 0
    first, second = sorted(stored(env), key=lambda j: j["after"] is not None)
    assert second["after"] == first["id"] and first["after"] is None


def test_batch_refuses_a_case_twice_unchained(env, capsys):
    case(env)
    (env["camp"] / "batch.txt").write_text(
        f"c1 -p={env['exe']}\nc1 -p={env['exe']}\n")
    assert run(env, "add", "1", "-f", "batch.txt") == 1
    assert "line 2: " in capsys.readouterr().out and stored(env) == []


def test_rm(env, capsys):
    case(env)
    run(env, "add", "1", "c1", f"-p={env['exe']}")
    [j] = stored(env)
    run(env, "add", "1", "--timeout", "1m", "--after", j["id"], "--", "true")
    [k] = [x for x in stored(env) if x["id"] != j["id"]]
    capsys.readouterr()
    assert run(env, "rm", j["id"]) == 1
    assert f"fix: runq rm {k['id']}" in capsys.readouterr().out
    assert run(env, "rm", k["id"]) == 0 and run(env, "rm", j["id"]) == 0
    assert stored(env) == []
    assert run(env, "rm", "nosuchjob") == 1


def test_rm_refuses_a_running_job(env, capsys):
    j = runcheck.make_job(env["cfg"], "1", ["--", "true"],
                          cwd=str(env["camp"]), who=USER)
    slotlib.write_job(env["cfg"], "running", "1", j)
    assert run(env, "rm", j["id"]) == 1
    assert "fix: runq cancel 1" in capsys.readouterr().out


def test_hold_and_go(env, capsys):
    assert run(env, "hold", "1", "rebuilding", "build-x") == 0
    hold = json.load(open(slotlib.path(env["cfg"], "1", "hold")))
    assert hold["why"] == "rebuilding build-x" and hold["by"] == USER
    run(env, "ls")
    assert "slot 1  camp  HELD" in capsys.readouterr().out
    assert run(env, "go", "1") == 0
    assert not os.path.exists(slotlib.path(env["cfg"], "1", "hold"))
    assert run(env, "go", "1") == 0


def test_cancel(env, capsys):
    assert run(env, "cancel", "1") == 1
    assert "runs no job" in capsys.readouterr().out
    j = runcheck.make_job(env["cfg"], "1", ["--", "sleep", "9"],
                          cwd=str(env["camp"]), who=USER)
    slotlib.write_job(env["cfg"], "running", "1", j)
    assert run(env, "cancel", "1") == 0
    req = json.load(open(slotlib.path(env["cfg"], "1", "cancel")))
    assert req["jobs"] == [j["id"]]


def test_own(env, capsys):
    assert run(env, "own", "2", "camp", "use slot 2 for camp") == 0
    assert slotlib.owner(env["cfg"], "2") == "camp"
    assert "was none" in capsys.readouterr().out
    assert run(env, "own", "2", "nosuch", "x") == 1
    assert run(env, "own", "2", "camp", who=AGENT) == 1
    assert "the user's own words" in capsys.readouterr().out
    assert run(env, "own", "2", "none", "release it", who=AGENT) == 0
    assert slotlib.owner(env["cfg"], "2") is None
    os.remove(slotlib.path(env["cfg"], "2", "mode"))
    assert run(env, "own", "2", "camp", "x") == 1


def test_agent_permissions(env, capsys):
    case(env)
    exe = f"-p={env['exe']}"
    assert run(env, "add", "2", "c1", exe, "--timeout", "1h", who=AGENT) == 1
    assert "slot 2 belongs to no campaign" in capsys.readouterr().out
    assert run(env, "add", "1", "c1", exe, who=AGENT) == 1      # no timeout
    assert run(env, "add", "1", "c1", exe, "--timeout", "1h",
               who=AGENT) == 0
    assert run(env, "hold", "2", who=AGENT) == 1
    assert run(env, "hold", "1", who=AGENT) == 0
    assert run(env, "go", "1", who=AGENT, cwd=env["cases"]) == 1
    assert run(env, "go", "1", who=AGENT) == 0
    # A user's job outside every campaign is not the agent's to remove.
    other = env["cases"] / "elsewhere"
    other.mkdir()
    assert run(env, "add", "1", "--cwd", str(other), "--", "true") == 0
    [mine] = [j for j in stored(env) if j.get("cmd")]
    capsys.readouterr()
    assert run(env, "rm", mine["id"], who=AGENT) == 1
    assert "belongs to no campaign" in capsys.readouterr().out
    # An agent removes its own session's job from anywhere.
    [own] = [j for j in stored(env) if j.get("case")]
    assert run(env, "rm", own["id"], who=AGENT, cwd=env["cases"]) == 0


def test_agent_command_job_is_refused(env, capsys):
    assert run(env, "add", "1", "--timeout", "1h", "--", "sdrun.py", "-d",
               "c1", who=AGENT) == 1
    assert "case job" in capsys.readouterr().out


def test_any_queues_for_an_owned_slot(env, capsys):
    case(env)
    assert run(env, "add", "any", "c1", f"-p={env['exe']}", "--timeout",
               "1h", who=AGENT) == 0
    assert [j["slot"] for j in stored(env, slot="any")] == ["any"]
    run(env, "ls")
    assert "any     queued" in capsys.readouterr().out


def test_mode_back_to_runplan_moves_the_queue(env, capsys):
    case(env)
    run(env, "add", "1", "c1", f"-p={env['exe']}", "-c=0")
    run(env, "add", "1", "--cwd", str(env["clone"]), "--", "ctest", "-j", "2")
    assert run(env, "mode", "1", "runplan") == 0
    plan = (env["cases"] / "runplan1.sh").read_text()
    assert plan.startswith("#!/usr/bin/env bash\n# owner: camp ")
    assert f"sdrun.py -y -d {env['camp']}/c1 -c=0 -p={env['exe']} 2>&1 | " \
           f"tee {env['camp']}/c1/BOUT.log.console" in plan
    assert f"cd {env['clone']}\nctest -j 2\n" in plan
    assert stored(env) == [] and slotlib.mode(env["cfg"], "1") == "runplan"
    assert slotlib.owner(env["cfg"], "1") == "camp"
    assert runcheck.runplan_cases(env["cfg"], "1") == [str(env["camp"] / "c1")]
    assert run(env, "mode", "1", "queue") == 0
    assert "still names 1 case" in capsys.readouterr().out
    assert slotlib.mode(env["cfg"], "1") == "queue"


def test_mode_keeps_an_owner_line_and_comments_a_validate_job(env, capsys):
    case(env)
    (env["cases"] / "runplan1.sh").write_text(
        "#!/usr/bin/env bash\n# owner: none (old)\nset -u\n")
    run(env, "add", "1", "c1", f"-p={env['exe']}", "--validate")
    assert run(env, "mode", "1", "runplan") == 0
    plan = (env["cases"] / "runplan1.sh").read_text()
    assert plan.count("# owner:") == 1 and "# owner: camp" in plan
    assert "# not converted (--validate has no runplan form): runq add 1" \
        in plan


def test_mode_refuses_while_a_job_runs(env, capsys):
    j = runcheck.make_job(env["cfg"], "1", ["--", "true"],
                          cwd=str(env["camp"]), who=USER)
    slotlib.write_job(env["cfg"], "running", "1", j)
    assert run(env, "mode", "1", "runplan") == 1
    assert "slot 1 runs a job" in capsys.readouterr().out
    assert slotlib.mode(env["cfg"], "1") == "queue"


def test_move_job_lets_one_mover_win(env):
    j = runcheck.make_job(env["cfg"], "any", ["--", "true"],
                          cwd=str(env["camp"]), who=USER)
    slotlib.write_job(env["cfg"], "queue", "any", j)
    assert slotlib.move_job(env["cfg"], j["id"], ("queue", "any"),
                            ("running", "1"))
    assert slotlib.move_job(env["cfg"], j["id"], ("queue", "any"),
                            ("running", "2")) is None
    assert [x["id"] for x in stored(env, "running", "1")] == [j["id"]]


def test_agent_cancels_only_its_campaigns_job(env, capsys):
    other = env["cases"] / "elsewhere"
    other.mkdir()
    j = runcheck.make_job(env["cfg"], "1", ["--", "sleep", "9"],
                          cwd=str(other), who=USER)
    slotlib.write_job(env["cfg"], "running", "1", j)
    assert run(env, "cancel", "1", who=AGENT) == 1
    os.remove(slotlib.job_path(env["cfg"], "running", "1", j["id"]))
    j = runcheck.make_job(env["cfg"], "1", ["--", "sleep", "9"],
                          cwd=str(env["camp"]), who=USER)
    slotlib.write_job(env["cfg"], "running", "1", j)
    assert run(env, "cancel", "1", who=AGENT) == 0


def test_ids_keep_the_order_they_were_made_in():
    ids = [runcheck.new_id() for _ in range(200)]
    assert ids == sorted(ids) and len(set(ids)) == 200


FAKE_MANDATE = '''
import datetime as dt, json, os
STATE = {state!r}

def load(p):
    return json.load(open(p))

def expiry(v):
    return dt.datetime.fromisoformat(v)

def state_path(sid):
    return os.path.join(STATE, sid + ".json")

def check_jobs(state, jobs, dry_run=False):
    out = [{{"rule": "mandate-slot", "target": j["slot"], "job": j["id"],
             "message": "slot " + j["slot"] + " is not the mandate's"}}
           for j in jobs if j["slot"] not in ("1", "any")]
    if not out and not dry_run:
        with open(os.path.join(STATE, "charged"), "a") as f:
            f.write(" ".join(j["id"] for j in jobs) + "\\n")
    return out

def record_refusal(path, rule, target):
    with open(os.path.join(STATE, "refusals"), "a") as f:
        f.write(rule + " " + target + "\\n")
    return " counted"
'''


def mandated(env, tmp_path, live=True):
    """A fake mandate module in the slot config, and a mandate for agent
    session s1 on slot 1 (expired unless `live`)."""
    st = tmp_path / "mandates"
    st.mkdir()
    mod = tmp_path / "mandate.py"
    mod.write_text(FAKE_MANDATE.format(state=str(st)))
    env["cfg"]["mandate"] = str(mod)
    when = dt.datetime.now() + dt.timedelta(hours=1 if live else -1)
    (st / "s1.json").write_text(json.dumps(
        {"bounds": {"slots": [1], "expires": when.isoformat()}}))
    return st


def test_runq_adds_the_mandates_checks(env, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(slotlib, "config", lambda: env["cfg"])
    st = mandated(env, tmp_path)
    case(env)
    exe = f"-p={env['exe']}"
    open(slotlib.path(env["cfg"], "2", "owner"), "w").write(
        json.dumps({"campaign": "camp"}))
    assert run(env, "add", "2", "c1", exe, "--timeout", "1h",
               who=AGENT) == 1
    assert "mandate: slot 2 is not the mandate's" in capsys.readouterr().out
    assert (st / "refusals").read_text() == "mandate-slot 2\n"
    assert run(env, "add", "2", "c1", exe, "--timeout", "1h", "--dry-run",
               who=AGENT) == 1
    assert (st / "refusals").read_text() == "mandate-slot 2\n"   # not counted
    assert run(env, "add", "1", "c1", exe, "--timeout", "1h", who=AGENT) == 0
    [j] = stored(env)
    assert j["mandate"]["state"] == str(st / "s1.json")
    assert j["only_slots"] == ["1"]
    assert (st / "charged").read_text().split() == [j["id"]]
    # The user's jobs, and other sessions' agents, meet no mandate.
    assert run(env, "add", "2", "--", "true") == 0
    other = dict(AGENT, session="s2")
    assert run(env, "add", "2", "--cwd", str(env["camp"]), "--timeout", "1m",
               "--", "true", who=other) == 0


def test_under_a_mandate_an_agent_owns_modes_and_restarts_nothing(
        env, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(slotlib, "config", lambda: env["cfg"])
    monkeypatch.setattr(runq, "screen_of", lambda slot: None)
    mandated(env, tmp_path)
    assert run(env, "own", "1", "camp", "words", who=AGENT) == 1
    assert run(env, "mode", "1", "runplan", who=AGENT) == 1
    assert run(env, "restart", "1", who=AGENT) == 1
    assert capsys.readouterr().out.count("under a mandate an agent") == 3
    assert run(env, "own", "1", "camp", "words") == 0       # the user may


def test_an_agent_without_a_session_is_refused_on_a_mandate_slot(
        env, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(slotlib, "config", lambda: env["cfg"])
    mandated(env, tmp_path)
    case(env)
    codex = dict(AGENT, session=None)
    assert run(env, "add", "1", "c1", f"-p={env['exe']}", "--timeout", "1h",
               who=codex) == 1
    assert "names no session" in capsys.readouterr().out


def test_an_expired_mandate_checks_nothing(env, tmp_path, monkeypatch):
    monkeypatch.setattr(slotlib, "config", lambda: env["cfg"])
    mandated(env, tmp_path, live=False)
    case(env)
    assert run(env, "add", "1", "c1", f"-p={env['exe']}", "--timeout", "1h",
               who=AGENT) == 0
    assert stored(env)[0]["mandate"] is None


def test_a_mandate_module_that_will_not_load_stops_agents_only(
        env, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(slotlib, "config", lambda: env["cfg"])
    env["cfg"]["mandate"] = str(tmp_path / "missing.py")
    case(env)
    assert run(env, "add", "1", "c1", f"-p={env['exe']}", "--timeout", "1h",
               who=AGENT) == 1
    assert "will not load" in capsys.readouterr().out
    assert run(env, "add", "1", "c1", f"-p={env['exe']}") == 0


def test_add_json_names_each_rule(env, capsys):
    case(env)
    assert run(env, "add", "1", "c1", "--dry-run", "--json") == 1
    out = json.loads(capsys.readouterr().out)
    assert out["result"] == "refused"
    assert [r["rule"] for r in out["refusals"]] == ["binary-unnamed"]
    assert run(env, "add", "1", "c1", f"-p={env['exe']}", "--json") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["result"] == "queued" and out["jobs"] == [stored(env)[0]["id"]]
