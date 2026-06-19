#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./run.sh -s=1|2|3 CASE [-restart] [-append] [extra BOUT options...]

  -s=1      Run on cores 2-11
  -s=2      Run on cores 12-21
  -s=3      Run on cores 22-31
  -b=NAME   Hermes-3 build folder to use (default: build-mc-master)
  -p=PATH   Path to the Hermes-3 executable to use (overrides -b)
  CASE      Case directory: a path relative to your current directory, or an absolute path
  -restart  Pass restart to Hermes/BOUT++
  -append   Pass append to Hermes/BOUT++ (implies restart)
EOF
}

slot=""
build="build-mc-master"
exe_path=""
case_arg=""
pass_restart=false
pass_append=false
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s=1|-s=2|-s=3)
      slot="${1#*=}"
      shift
      ;;
    -s)
      if [[ $# -lt 2 ]]; then
        echo "error: -s requires a value" >&2
        usage >&2
        exit 1
      fi
      slot="$2"
      shift 2
      ;;
    -b=*)
      build="${1#*=}"
      shift
      ;;
    -b)
      if [[ $# -lt 2 ]]; then
        echo "error: -b requires a value" >&2
        usage >&2
        exit 1
      fi
      build="$2"
      shift 2
      ;;
    -p=*|--path=*)
      exe_path="${1#*=}"
      shift
      ;;
    -p|--path)
      if [[ $# -lt 2 ]]; then
        echo "error: $1 requires a value" >&2
        usage >&2
        exit 1
      fi
      exe_path="$2"
      shift 2
      ;;
    -restart|restart)
      pass_restart=true
      shift
      ;;
    -append|append)
      pass_append=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$case_arg" ]]; then
        case_arg="$1"
      else
        extra_args+=("$1")
      fi
      shift
      ;;
  esac
done

case "$slot" in
  1)
    cores="2-11"
    ;;
  2)
    cores="12-21"
    ;;
  3)
    cores="22-31"
    ;;
  *)
    echo "error: choose -s=1, -s=2, or -s=3" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ -z "$case_arg" ]]; then
  echo "error: missing case name" >&2
  usage >&2
  exit 1
fi

if [[ "$pass_append" == true ]]; then
  pass_restart=true
fi

if [[ "$case_arg" = /* ]]; then
  case_dir="$case_arg"
else
  case_dir="$PWD/$case_arg"
fi

if [[ ! -d "$case_dir" ]]; then
  echo "error: case directory not found: $case_dir" >&2
  exit 1
fi

# Normalise to an absolute, canonical path
case_dir="$(cd -- "$case_dir" && pwd)"

if [[ ! -f "$case_dir/BOUT.inp" ]]; then
  echo "error: BOUT.inp not found in $case_dir" >&2
  exit 1
fi

case_name="$(basename "$case_dir")"
hermes_root="${hermes:-/home/mike/work/hermes-3}"

if [[ -n "$exe_path" ]]; then
  # Explicit executable path overrides the build folder.
  if [[ "$exe_path" = /* ]]; then
    exe="$exe_path"
  else
    exe="$PWD/$exe_path"
  fi
  build_dir="$(dirname "$exe")"
  build_label="$exe"
else
  build_dir="$hermes_root/$build"
  exe="$build_dir/hermes-3"
  build_label="$build"
fi

# Report what the build was actually compiled against, not the repo's current
# checkout. CMake's git-data mechanism regenerates these on every build.
git_data="$build_dir/CMakeFiles/git-data"
if [[ -f "$git_data/head-ref" ]]; then
  commit="$(cut -c1-7 "$git_data/head-ref" 2>/dev/null || echo '?')"
else
  commit="?"
fi
if [[ -f "$git_data/HEAD" ]]; then
  branch="$(sed -n 's#^ref: refs/heads/##p' "$git_data/HEAD" 2>/dev/null)"
  branch="${branch:-detached}"
else
  branch="?"
fi

if [[ ! -x "$exe" ]]; then
  echo "error: Hermes executable not found at $exe" >&2
  exit 1
fi

cmd=(taskset -c "$cores" mpirun --bind-to none -np 10 "$exe" -d "$case_dir")

if [[ "$pass_restart" == true ]]; then
  cmd+=(restart)
fi

if [[ "$pass_append" == true ]]; then
  cmd+=(append)
fi

cmd+=("${extra_args[@]}")

# --- Pretty confirmation summary -------------------------------------------
if [[ -t 1 ]]; then
  b=$'\033[1m'; d=$'\033[2m'; c=$'\033[38;5;208m'; g=$'\033[38;5;154m'; y=$'\033[33m'; r=$'\033[0m'
else
  b=''; d=''; c=''; g=''; y=''; r=''
fi

term_width=$(tput cols 2>/dev/null || echo 80)

row() { printf '   %s%-8s%s %s\n' "$d" "$1" "$r" "$2"; }

# Print a label + value, wrapping long paths on '/' aligned under the value column.
pathrow() {
  local label="$1" val="$2"
  local pad='            '          # 12 spaces: 3 indent + 8 label + 1 gap
  local avail=$(( term_width - 13 )); (( avail < 24 )) && avail=24
  local IFS='/'
  local parts; read -ra parts <<< "$val"
  local cur="" out=() i seg
  for i in "${!parts[@]}"; do
    seg="${parts[$i]}"; (( i > 0 )) && seg="/$seg"
    if [[ -z "$cur" ]]; then cur="$seg"
    elif (( ${#cur} + ${#seg} <= avail )); then cur+="$seg"
    else out+=("$cur"); cur="$seg"; fi
  done
  out+=("$cur")
  row "$label" "${out[0]}"
  for (( i=1; i<${#out[@]}; i++ )); do printf '%s%s\n' "$pad" "${out[$i]}"; done
}

flags=""
[[ "$pass_restart" == true ]] && flags+="restart"
[[ "$pass_append"  == true ]] && flags+="${flags:+ + }append"
[[ -z "$flags" ]] && flags="${d}none${r}" || flags="${g}${flags}${r}"

printf '\n'
printf '  %s┌─ %sRunning: %s%s%s\n' "$c" "${r}${d}" "${r}${c}${b}" "${case_name}" "$r"
pathrow "Path" "${case_dir}"
row "Cores"  "${cores} ${d}(slot ${slot})${r}"
row "Build"  "${build_label} ${d}@${r} ${y}${branch}${r}, ${y}${commit}${r}"
row "Flags"  "${flags}"
printf '  %s└─%s\n' "$c" "$r"
printf '\n'

read -r -p "  Proceed? ${b}[y/N]${r} " reply
case "$reply" in
  [yY]|[yY][eE][sS]) ;;
  *)
    echo "  Aborted."
    exit 1
    ;;
esac
printf '\n'

exec "${cmd[@]}"