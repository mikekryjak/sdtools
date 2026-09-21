#!/bin/bash
#
# Configure and build Hermes-3.
#
# Every setting that used to be a hand-edited variable at the top of this file
# is now a flag. The defaults reproduce the previous hardcoded values, so an
# invocation with no flags builds exactly what it built before.
#
# The build directory name is composed, not given:
# build-<limiter>[-<conduction>]-<name>, where <limiter> is va or mc and the
# conduction method appears only when it is not Original. So -s VanAlbada
# -b master gives build-va-master, and adding -m Harmonic gives
# build-va-harmonic-master. This keeps two builds that differ only in a
# compile-time physics option from silently overwriting each other.

set -euo pipefail


# Defaults for the settings that used to be hardcoded
BUILD_NAME="separate-neutral-fluxlim"
LIMITER="VanAlbada"
CONDMETHOD="Original"
HERMES_DIR="hermes-3"
PETSC_DIR_SET="${PETSC_DIR:-/home/mike/work/petsc-3.23.3}"
PETSC_ARCH_SET="arch-linux-c-opt"
JOBS="1"
JOBS_EXPLICIT=false
ASSUME_YES=false

# Core slots, matching sdrun.py exactly. Cores 0-1 are left for the system.
SLOT=""
CORE_SPEC=""
slot_cores() {
  case "$1" in
    1) echo "2-11" ;;
    2) echo "12-21" ;;
    3) echo "22-31" ;;
    *) echo "" ;;
  esac
}

# Count cores in a taskset-style list: '2-11' -> 10, '2,4,6-9' -> 6
count_cores() {
  local spec="$1" total=0 part lo hi
  local IFS=,
  for part in $spec; do
    case "$part" in
      *-*) lo="${part%%-*}"; hi="${part##*-}"
           case "$lo$hi" in ''|*[!0-9]*) echo 0; return ;; esac
           [ "$hi" -lt "$lo" ] && { echo 0; return ; }
           total=$((total + hi - lo + 1)) ;;
      *)   case "$part" in ''|*[!0-9]*) echo 0; return ;; esac
           total=$((total + 1)) ;;
    esac
  done
  echo "$total"
}

# Defaults for the existing flags
FAST_BUILD=false
DISABLE_TESTS=false
NO_BOUT=false
EXPRESS=false
CUSTOM_BOUT_SRC=""
CHECK_LEVEL="2"
BUILD_TYPE="RelWithDebInfo"
NO_SUBMODULE_UPDATE=false
OVERRIDE_WARNINGS=false
BOUTPP=false
WRITE_LOG=false

# Flags that change the CMake configuration, and so cannot be used with -e.
# Collected as they are parsed rather than inferred by comparing values.
CONFIG_OPTS_GIVEN=""

usage() {
  cat <<EOF
Usage: $0 [-b <name>] [-s <limiter>] [-m <method>] [-r <dir>] [-P <dir>]
          [-j <n>] [-k <n>] [-y] [-f] [-t] [-n] [-e] [-c <path>] [-d] [-u] [-w] [-p] [-l]

Settings (defaults in brackets):
  -b <name>   Build name; build dir becomes
              build-<va|mc>[-<conduction>]-<name>                  [$BUILD_NAME]
  -s <lim>    Slope limiter: MC or VanAlbada                       [$LIMITER]
  -m <meth>   Conduction method: Original, ProductJK or Harmonic   [$CONDMETHOD]
  -r <dir>    Hermes repo directory; relative paths resolve
              against this script's directory                      [$HERMES_DIR]
  -P <dir>    PETSC_DIR                                            [$PETSC_DIR_SET]
  -j <n>      Parallel build jobs                                  [$JOBS]
  -k <n>      CHECK level                                          [$CHECK_LEVEL]
  -S <slot>   Pin the build to a core slot, as sdrun.py numbers them:
              1 = cores 2-11, 2 = 12-21, 3 = 22-31. Building on the same
              slot the test will run on keeps a build off another slot's
              timed run. Sets -j to the core count unless -j is given.
  -C <cores>  Pin to an explicit core list (e.g. 2-11 or 2,4,6-9);
              overrides -S
  -y          Skip the confirmation prompt (required when not on a terminal,
              so this script can be driven from another script)

Build options:
  -f          Fast incremental build: keep build dir, reconfigure, and build
  -t          Disable Hermes test building (HERMES_TESTS=OFF)
  -n          Do not rebuild BOUT++ (this is WIP)
  -e          Express build: skip all configuration and only run cmake --build
  -c <path>   Path to custom BOUT++ source directory (sets DHERMES_BOUT_SRC)
  -d          Use debug build (CHECK=4, Debug)
  -u          Do not update git submodules
  -w          Override compilation error on warnings
  -p          Compile with the BOUT++ python interface
  -l          Write build log to hermes-buildlog.out
EOF
  exit 1
}

# Parse flags
while getopts "b:s:m:r:P:j:k:S:C:yftnec:duwpl" opt; do
  case $opt in
    b) BUILD_NAME="$OPTARG" ;;
    s) LIMITER="$OPTARG" ;;
    m) CONDMETHOD="$OPTARG"; CONFIG_OPTS_GIVEN="$CONFIG_OPTS_GIVEN -m" ;;
    r) HERMES_DIR="$OPTARG" ;;
    P) PETSC_DIR_SET="$OPTARG" ;;
    j) JOBS="$OPTARG"; JOBS_EXPLICIT=true ;;
    S) SLOT="$OPTARG" ;;
    C) CORE_SPEC="$OPTARG" ;;
    y) ASSUME_YES=true ;;
    k) CHECK_LEVEL="$OPTARG"; CONFIG_OPTS_GIVEN="$CONFIG_OPTS_GIVEN -k" ;;
    f) FAST_BUILD=true; CONFIG_OPTS_GIVEN="$CONFIG_OPTS_GIVEN -f" ;;
    t) DISABLE_TESTS=true; CONFIG_OPTS_GIVEN="$CONFIG_OPTS_GIVEN -t" ;;
    n) NO_BOUT=true; CONFIG_OPTS_GIVEN="$CONFIG_OPTS_GIVEN -n" ;;
    e) EXPRESS=true ;;
    c) CUSTOM_BOUT_SRC="$OPTARG"; CONFIG_OPTS_GIVEN="$CONFIG_OPTS_GIVEN -c" ;;
    d) CHECK_LEVEL="4"
       BUILD_TYPE="Debug"
       CONFIG_OPTS_GIVEN="$CONFIG_OPTS_GIVEN -d" ;;
    u) NO_SUBMODULE_UPDATE=true; CONFIG_OPTS_GIVEN="$CONFIG_OPTS_GIVEN -u" ;;
    w) OVERRIDE_WARNINGS=true; CONFIG_OPTS_GIVEN="$CONFIG_OPTS_GIVEN -w" ;;
    p) BOUTPP=true; CONFIG_OPTS_GIVEN="$CONFIG_OPTS_GIVEN -p" ;;
    l) WRITE_LOG=true ;;
    *) usage ;;
  esac
done

# -e skips configure entirely, so any configure-affecting option is meaningless
# with it. -b, -s, -r and -j are allowed: they only locate the build directory
# or control the build step itself.
if [ "$EXPRESS" = true ] && [ -n "$CONFIG_OPTS_GIVEN" ]; then
  echo "Error: -e (express build) cannot be combined with configure options:$CONFIG_OPTS_GIVEN"
  echo "Use -f and/or -t for configure+build, or drop -e."
  exit 2
fi

# Validate the settings, so a typo fails here rather than deep inside CMake
case "$LIMITER" in
  MC)        LIMITER_CODE="mc" ;;
  VanAlbada) LIMITER_CODE="va" ;;
  *) echo "Error: unknown slope limiter '$LIMITER'. Use MC or VanAlbada."; exit 2 ;;
esac

# Original is the default and is left out of the name, matching the existing
# build directories; anything else is named so it cannot collide with it.
case "$CONDMETHOD" in
  Original)  COND_CODE="" ;;
  ProductJK) COND_CODE="productjk-" ;;
  Harmonic)  COND_CODE="harmonic-" ;;
  *) echo "Error: unknown conduction method '$CONDMETHOD'. Use Original, ProductJK or Harmonic."; exit 2 ;;
esac

case "$CHECK_LEVEL" in
  ''|*[!0-9]*) echo "Error: CHECK level must be a number, got '$CHECK_LEVEL'."; exit 2 ;;
esac

case "$JOBS" in
  ''|*[!0-9]*) echo "Error: job count must be a number, got '$JOBS'."; exit 2 ;;
esac

# Resolve core pinning. An explicit list overrides a slot, as in sdrun.py.
if [ -n "$CORE_SPEC" ]; then
  CORES="$CORE_SPEC"
  PIN_LABEL="cores $CORES"
elif [ -n "$SLOT" ]; then
  CORES="$(slot_cores "$SLOT")"
  if [ -z "$CORES" ]; then
    echo "Error: unknown slot '$SLOT'. Use 1 (cores 2-11), 2 (12-21) or 3 (22-31)."
    exit 2
  fi
  PIN_LABEL="slot $SLOT, cores $CORES"
else
  CORES=""
  PIN_LABEL=""
fi

TASKSET=""
if [ -n "$CORES" ]; then
  NCORES="$(count_cores "$CORES")"
  if [ "$NCORES" -lt 1 ]; then
    echo "Error: invalid core list '$CORES'. Use e.g. 2-11 or 2,4,6-9."
    exit 2
  fi
  TASKSET="taskset -c $CORES"
  # Match the job count to the cores the build is allowed to use, unless the
  # user asked for a specific number.
  if [ "$JOBS_EXPLICIT" = false ]; then
    JOBS="$NCORES"
  fi
fi

if [ -z "$BUILD_NAME" ]; then
  echo "Error: build name (-b) must not be empty."
  exit 2
fi

# Compose the build directory from the compile-time physics options and the name
BUILD_DIR="build-$LIMITER_CODE-$COND_CODE$BUILD_NAME"

# Resolve the Hermes directory. Relative paths resolve against this script's
# own directory, so the script works from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$HERMES_DIR" in
  /*) HERMES_PATH="$HERMES_DIR" ;;
  *)  HERMES_PATH="$SCRIPT_DIR/$HERMES_DIR" ;;
esac

if [ ! -d "$HERMES_PATH" ]; then
  echo "Error: Hermes directory not found: $HERMES_PATH"
  exit 2
fi

# CMake settings
export PETSC_DIR="$PETSC_DIR_SET"
export PETSC_ARCH="$PETSC_ARCH_SET"

# The build log belongs where the script was called from, not in the repo
LOG_PATH="$PWD/hermes-buildlog.out"

# Enter the Hermes directory before anything that reads the repo
cd "$HERMES_PATH"

# Record what is being built. Never fatal: the directory may not be a git repo.
HERMES_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
HERMES_REF="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

# --- Confirmation summary, in the style of sdrun.py ----------------------
if [ -t 1 ]; then
  S_B=$'\033[1m'; S_D=$'\033[2m'; S_C=$'\033[38;5;208m'
  S_G=$'\033[38;5;154m'; S_Y=$'\033[33m'; S_R=$'\033[0m'
else
  S_B=""; S_D=""; S_C=""; S_G=""; S_Y=""; S_R=""
fi

row() { printf "   %s%-11s%s %s\n" "$S_D" "$1" "$S_R" "$2"; }

if [ "$EXPRESS" = true ]; then
  MODE="express ${S_D}(build only, no configure)${S_R}"
elif [ "$FAST_BUILD" = true ]; then
  MODE="fast ${S_D}(reconfigure in place)${S_R}"
else
  MODE="full ${S_D}(build dir deleted and reconfigured)${S_R}"
fi

EXTRAS=""
[ "$DISABLE_TESTS" = true ]       && EXTRAS="$EXTRAS no-tests"
[ "$NO_BOUT" = true ]             && EXTRAS="$EXTRAS no-bout"
[ "$NO_SUBMODULE_UPDATE" = true ] && EXTRAS="$EXTRAS no-submodule-update"
[ "$OVERRIDE_WARNINGS" = true ]   && EXTRAS="$EXTRAS warnings-ok"
[ "$BOUTPP" = true ]              && EXTRAS="$EXTRAS boutpp"
[ -n "$CUSTOM_BOUT_SRC" ]         && EXTRAS="$EXTRAS custom-bout-src"
[ "$WRITE_LOG" = true ]           && EXTRAS="$EXTRAS logged"
if [ -n "$EXTRAS" ]; then
  EXTRAS="${S_G}${EXTRAS# }${S_R}"
else
  EXTRAS="${S_D}none${S_R}"
fi

echo
printf "  %s┌─ %s%sBuilding: %s%s%s%s%s\n" \
  "$S_C" "$S_R" "$S_D" "$S_R" "$S_C" "$S_B" "$BUILD_DIR" "$S_R"
row "Repo" "$HERMES_PATH"
row "Commit" "${S_Y}${HERMES_REF}${S_R}, ${S_Y}${HERMES_SHA}${S_R}"
row "Physics" "$LIMITER ${S_D}limiter${S_R}, $CONDMETHOD ${S_D}conduction${S_R}"
row "Build" "$BUILD_TYPE ${S_D}(CHECK=$CHECK_LEVEL)${S_R}"
row "PETSc" "$PETSC_DIR"
if [ -n "$CORES" ]; then
  row "Cores" "$CORES ${S_D}($PIN_LABEL)${S_R}"
else
  row "Cores" "${S_D}unpinned (any core)${S_R}"
fi
row "Jobs" "$JOBS"
row "Mode" "$MODE"
row "Flags" "$EXTRAS"
printf "  %s└─%s\n\n" "$S_C" "$S_R"

if [ "$ASSUME_YES" = false ]; then
  if [ ! -t 0 ]; then
    echo "  Not running on a terminal and -y was not given. Aborted."
    exit 1
  fi
  read -r -p "  Proceed? ${S_B}[y/N]${S_R} " REPLY_YN || { echo; echo "  Aborted."; exit 1; }
  case "$REPLY_YN" in
    y|Y|yes|YES) ;;
    *) echo "  Aborted."; exit 1 ;;
  esac
  echo
fi

if [ "$WRITE_LOG" = true ]; then
  # Log outcome. Set up after the prompt, so the summary stays on screen.
  rm -f "$LOG_PATH" # Remove if already exists
  exec 3>&1 4>&2 # Trap stdout, stderr etc all at the same time.
  trap 'exec 2>&4 1>&3' 0 1 2 3
  exec 1>"$LOG_PATH" 2>&1
  echo "Building $BUILD_DIR from $HERMES_REF, $HERMES_SHA"
fi

if [ "$EXPRESS" = false ]; then

  # Prepare CMake arguments
  CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCHECK=$CHECK_LEVEL -DBOUT_DOWNLOAD_SUNDIALS=ON -DBOUT_USE_PETSC=ON -DHERMES_SLOPE_LIMITER=$LIMITER -DHERMES_CONDUCTION_METHOD=$CONDMETHOD -DBOUT_USE_NLS=OFF"

  if [ "$NO_BOUT" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DHERMES_BUILD_BOUT=OFF"
    echo "Disabling BOUT++ build"
  fi

  # -n makes sure string length is not zero
  if [ -n "$CUSTOM_BOUT_SRC" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DHERMES_BOUT_SRC=$CUSTOM_BOUT_SRC"
    echo "Using custom BOUT++ source: $CUSTOM_BOUT_SRC"
  fi

  if [ "$NO_SUBMODULE_UPDATE" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DHERMES_UPDATE_GIT_SUBMODULE=OFF"
    echo "Disabling git submodule update"
  fi

  if [ "$BOUTPP" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DHERMES_BUILD_BOUT_PYTHON=ON"
    echo "Enabling BOUT++ python interface"
  fi

  if [ "$OVERRIDE_WARNINGS" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DHERMES_ERROR_ON_WARNINGS=OFF"
    echo "Overriding error on warnings during compilation"
  else
    CMAKE_ARGS="$CMAKE_ARGS -DHERMES_ERROR_ON_WARNINGS=ON"
  fi

  if [ "$DISABLE_TESTS" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DHERMES_TESTS=OFF -DHERMES_COPY_TESTS_TO_BUILD=OFF"
    echo "Disabling Hermes tests"
  fi

  # Make fresh build directory if required
  if [ "$FAST_BUILD" = false ]; then
    echo "Forcing rebuild by removing existing build directory..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
  else
    echo "Fast mode: reconfiguring existing build directory"
  fi

  # Compile
  echo "CMake command: $TASKSET cmake -S . -B $BUILD_DIR $CMAKE_ARGS"
  $TASKSET cmake -S . -B "$BUILD_DIR" $CMAKE_ARGS

else

  # Express install, just build
  echo "Express mode: skipping configure and building existing CMake cache"
  if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: no build directory to build: $HERMES_PATH/$BUILD_DIR"
    exit 2
  fi
fi

$TASKSET cmake --build "$BUILD_DIR" -j "$JOBS" #--verbose
