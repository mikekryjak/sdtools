#!/usr/bin/env python3
"""Run a Hermes-3 case on a fixed core slot, with a confirmation summary.

Python port of the old sdrun.sh. The main reason for Python is that any
argument (the case path, -p/-b values, and extra BOUT options) is passed
through os.path.expandvars/expanduser, so shell-style references such as
$hermes, ${cases}, or ~ are expanded by the script itself even when they
arrive quoted/unexpanded.
"""

import os
import shutil
import sys

USAGE = """\
Usage: sdrun.py -s=1|2|3 -d CASE [-restart] [-append] [extra BOUT options...]

  -s=1      Run on cores 2-11
  -s=2      Run on cores 12-21
  -s=3      Run on cores 22-31
  -c=SPEC   Run on an explicit core list (e.g. 2-11 or 2,4,6-9); overrides -s
            and sets mpirun -np to the number of cores in the list
  -b=NAME   Hermes-3 build folder to use (default: build-mc-master)
  -p=PATH   Path to the Hermes-3 executable to use (overrides -b)
  -d=CASE   Case directory: a path relative to your current directory, or an absolute path
  -restart  Pass restart to Hermes/BOUT++
  -append   Pass append to Hermes/BOUT++ (implies restart)

Shell variables like $hermes or ${cases} and ~ are expanded in any argument.
"""

CORES = {"1": "2-11", "2": "12-21", "3": "22-31"}


def count_cores(spec):
    """Count cores in a taskset-style list, e.g. '2-11' -> 10, '2,4,6-9' -> 6."""
    total = 0
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                lo, hi = int(lo), int(hi)
            except ValueError:
                die(f"invalid core spec: {spec}", with_usage=True)
            if hi < lo:
                die(f"invalid core range: {part}", with_usage=True)
            total += hi - lo + 1
        else:
            try:
                int(part)
            except ValueError:
                die(f"invalid core spec: {spec}", with_usage=True)
            total += 1
    if total < 1:
        die(f"invalid core spec: {spec}", with_usage=True)
    return total


def die(msg, with_usage=False):
    print(f"error: {msg}", file=sys.stderr)
    if with_usage:
        print(USAGE, file=sys.stderr, end="")
    sys.exit(1)


def expand(value):
    """Expand $VAR / ${VAR} / ~ in a single argument."""
    return os.path.expanduser(os.path.expandvars(value))


def parse_args(argv):
    slot = ""
    core_spec = ""
    build = "build-mc-master"
    exe_path = ""
    case_arg = ""
    pass_restart = False
    pass_append = False
    extra_args = []

    def need_value(flag, rest):
        if not rest:
            die(f"{flag} requires a value", with_usage=True)
        return rest[0]

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-s=1", "-s=2", "-s=3"):
            slot = arg.split("=", 1)[1]
        elif arg == "-s":
            slot = need_value("-s", argv[i + 1:])
            i += 1
        elif arg.startswith("-c="):
            core_spec = arg.split("=", 1)[1]
        elif arg == "-c":
            core_spec = need_value("-c", argv[i + 1:])
            i += 1
        elif arg.startswith("-b="):
            build = arg.split("=", 1)[1]
        elif arg == "-b":
            build = need_value("-b", argv[i + 1:])
            i += 1
        elif arg.startswith("-p=") or arg.startswith("--path="):
            exe_path = arg.split("=", 1)[1]
        elif arg in ("-p", "--path"):
            exe_path = need_value(arg, argv[i + 1:])
            i += 1
        elif arg.startswith("-d="):
            case_arg = arg.split("=", 1)[1]
        elif arg == "-d":
            case_arg = need_value("-d", argv[i + 1:])
            i += 1
        elif arg in ("-restart", "restart"):
            pass_restart = True
        elif arg in ("-append", "append"):
            pass_append = True
        elif arg in ("-h", "--help"):
            print(USAGE, end="")
            sys.exit(0)
        else:
            extra_args.append(arg)
        i += 1

    return {
        "slot": slot,
        "core_spec": core_spec,
        "build": build,
        "exe_path": exe_path,
        "case_arg": case_arg,
        "pass_restart": pass_restart,
        "pass_append": pass_append,
        "extra_args": extra_args,
    }


def read_git_data(build_dir):
    """Report what the build was compiled against, not the repo's current
    checkout. CMake's git-data mechanism regenerates these on every build."""
    git_data = os.path.join(build_dir, "CMakeFiles", "git-data")
    commit = "?"
    branch = "?"
    head_ref = os.path.join(git_data, "head-ref")
    head = os.path.join(git_data, "HEAD")
    try:
        with open(head_ref) as fh:
            commit = fh.read().strip()[:7] or "?"
    except OSError:
        pass
    try:
        with open(head) as fh:
            line = fh.read().strip()
        if line.startswith("ref: refs/heads/"):
            branch = line[len("ref: refs/heads/"):] or "detached"
        else:
            branch = "detached"
    except OSError:
        pass
    return commit, branch


class Style:
    def __init__(self, enabled):
        if enabled:
            self.b = "\033[1m"
            self.d = "\033[2m"
            self.c = "\033[38;5;208m"
            self.g = "\033[38;5;154m"
            self.y = "\033[33m"
            self.r = "\033[0m"
        else:
            self.b = self.d = self.c = self.g = self.y = self.r = ""


def main():
    opts = parse_args(sys.argv[1:])

    slot = opts["slot"]
    core_spec = opts["core_spec"]
    if core_spec:
        # Explicit core list overrides the slot.
        cores = core_spec
    elif slot in CORES:
        cores = CORES[slot]
    else:
        die("choose -s=1, -s=2, -s=3, or -c=SPEC (e.g. -c=2-11)", with_usage=True)
    np = count_cores(cores)

    if not opts["case_arg"]:
        die("missing case directory: pass it with -d=CASE", with_usage=True)

    pass_restart = opts["pass_restart"] or opts["pass_append"]
    pass_append = opts["pass_append"]

    # Expand $hermes, ~, etc. in the case path, then resolve.
    case_arg = expand(opts["case_arg"])
    if os.path.isabs(case_arg):
        case_dir = case_arg
    else:
        case_dir = os.path.join(os.getcwd(), case_arg)

    if not os.path.isdir(case_dir):
        die(f"case directory not found: {case_dir}")

    case_dir = os.path.realpath(case_dir)

    if not os.path.isfile(os.path.join(case_dir, "BOUT.inp")):
        die(f"BOUT.inp not found in {case_dir}")

    case_name = os.path.basename(case_dir)
    hermes_root = os.environ.get("hermes", "/home/mike/work/hermes-3")

    exe_path = expand(opts["exe_path"])
    build = expand(opts["build"])
    if exe_path:
        # Explicit executable path overrides the build folder.
        if os.path.isabs(exe_path):
            exe = exe_path
        else:
            exe = os.path.join(os.getcwd(), exe_path)
        build_dir = os.path.dirname(exe)
        build_label = exe
    else:
        build_dir = os.path.join(hermes_root, build)
        exe = os.path.join(build_dir, "hermes-3")
        build_label = build

    commit, branch = read_git_data(build_dir)

    if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
        die(f"Hermes executable not found at {exe}")

    extra_args = [expand(a) for a in opts["extra_args"]]

    cmd = [
        "taskset", "-c", cores,
        "mpirun", "--bind-to", "none", "-np", str(np),
        exe, "-d", case_dir,
    ]
    if pass_restart:
        cmd.append("restart")
    if pass_append:
        cmd.append("append")
    cmd += extra_args

    # --- Pretty confirmation summary --------------------------------------
    s = Style(sys.stdout.isatty())
    term_width = shutil.get_terminal_size((80, 24)).columns

    def row(label, value):
        print(f"   {s.d}{label:<8}{s.r} {value}")

    def pathrow(label, val):
        """label + value, wrapping long paths on '/' under the value column."""
        pad = " " * 12  # 3 indent + 8 label + 1 gap
        avail = max(term_width - 13, 24)
        parts = val.split("/")
        cur = ""
        out = []
        for i, part in enumerate(parts):
            seg = ("/" + part) if i > 0 else part
            if not cur:
                cur = seg
            elif len(cur) + len(seg) <= avail:
                cur += seg
            else:
                out.append(cur)
                cur = seg
        out.append(cur)
        row(label, out[0])
        for line in out[1:]:
            print(f"{pad}{line}")

    flags_parts = []
    if pass_restart:
        flags_parts.append("restart")
    if pass_append:
        flags_parts.append("append")
    if flags_parts:
        flags = f"{s.g}{' + '.join(flags_parts)}{s.r}"
    else:
        flags = f"{s.d}none{s.r}"

    print()
    print(f"  {s.c}┌─ {s.r}{s.d}Running: {s.r}{s.c}{s.b}{case_name}{s.r}")
    pathrow("Path", case_dir)
    cores_note = f"slot {slot}, {np} procs" if slot and not core_spec else f"{np} procs"
    row("Cores", f"{cores} {s.d}({cores_note}){s.r}")
    row("Build", f"{build_label} {s.d}@{s.r} {s.y}{branch}{s.r}, {s.y}{commit}{s.r}")
    row("Flags", flags)
    print(f"  {s.c}└─{s.r}")
    print()

    try:
        reply = input(f"  Proceed? {s.b}[y/N]{s.r} ")
    except (EOFError, KeyboardInterrupt):
        print("\n  Aborted.")
        sys.exit(1)
    if reply.strip().lower() not in ("y", "yes"):
        print("  Aborted.")
        sys.exit(1)
    print()

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
