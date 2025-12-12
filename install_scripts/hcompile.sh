#!/bin/bash

# BUILD_DIR="build-mc-master"


# Slope limiter setting
LIMITER="MC"

# Directory
HERMES_DIR="hermes-3"

# Defaults
FAST_BUILD=false
NO_BOUT=false
EXPRESS=false
CUSTOM_BOUT_SRC=""
CHECK_LEVEL=0
BUILD_TYPE="Release"
NO_SUBMODULE_UPDATE=false
OVERRIDE_WARNINGS=false

usage() {
  cat <<EOF
Usage: $0 [-f] [-n] [e] [-c <path>] [-d]
  -f          Skip removing existing build directory (i.e. “fast” build)
  -n          Do not rebuild BOUT++ (this is WIP)
  -e          Skip BOUT++ rebuild and Hermes-3 configuration (go to build directory and run cmake)
  -c <path>   Path to custom BOUT++ source directory (sets DHERMES_BOUT_SRC)
  -d          Use debug build
  -u          Do not update git submodules
  -w          Override compilation error on warnings
EOF
  exit 1
}

# Parse flags
while getopts "fnec:du" opt; do
  case $opt in
    f) FAST_BUILD=true ;;
    n) NO_BOUT=true      ;;
    e) EXPRESS=true     ;;
    c) CUSTOM_BOUT_SRC="$OPTARG" ;;
    d) CHECK_LEVEL="3"
       BUILD_TYPE="Debug" ;;
    u) NO_SUBMODULE_UPDATE=true ;;
    w) OVERRIDE_WARNINGS=true ;;
    *) usage              ;;
  esac
done


# CMake settings
export PETSC_DIR=/home/mike/work/petsc-3.23.3
export PETSC_ARCH=arch-linux-c-opt

if [ "$EXPRESS" = false ]; then

  # Prepare CMake arguments
  CMAKE_ARGS="-B $BUILD_DIR -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCHECK=$CHECK_LEVEL -DBOUT_DOWNLOAD_SUNDIALS=ON -DBOUT_USE_PETSC=ON -DHERMES_SLOPE_LIMITER=$LIMITER"

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

  if [ "$OVERRIDE_WARNINGS" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DHERMES_ERROR_ON_WARNINGS=OFF"
    echo "Overriding error on warnings during compilation"
  else
    CMAKE_ARGS="$CMAKE_ARGS -DHERMES_ERROR_ON_WARNINGS=ON"
  fi

  # Enter the directories and compile
  cd $HERMES_DIR

  # Make fresh build directory if required
  if [ "$FAST_BUILD" = false ]; then
    echo "Forcing rebuild by removing existing build directory..."
    rm -rf "$BUILD_DIR"
    mkdir "$BUILD_DIR"
  fi

  # Compile
  cmake . -B $BUILD_DIR $CMAKE_ARGS
else

  # Express install, just build
  cd $HERMES_DIR
fi

cmake --build $BUILD_DIR -j 32 #--verbose