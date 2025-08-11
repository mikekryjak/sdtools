#!/bin/bash

# BUILD_DIR="build-mc-master"
BUILD_DIR="build-mc-ena"   #  E-AFN-neutral-advection
# BUILD_DIR="build-mc-master-fix-heatflux-diags"
# BUILD_DIR="build-mc-master-old-remkit-comparison"
# BUILD_DIR="build-mc-master-next"
# BUILD_DIR="build-mc-selective-collisions"
# BUILD_DIR="build-mc-eafn"
# BUILD_DIR="build-mc-neutral-fluxlims"

# Defaults
FAST_BUILD=false
NO_BOUT=false
EXPRESS=false

usage() {
  cat <<EOF
Usage: $0 [-f] [-n]
  -f    Skip removing existing build directory (i.e. “fast” build)
  -n    Do not rebuild BOUT++ (this is WIP)
  -e    Skip BOUT++ rebuild and Hermes-3 configuration (go to build directory and run cmake)
EOF
  exit 1
}

# Parse flags
while getopts "fne" opt; do
  case $opt in
    f) FAST_BUILD=true ;;
    n) NO_BOUT=true      ;;
    e) EXPRESS=true     ;;
    *) usage              ;;
  esac
done


# CMake settings
export PETSC_DIR=/home/mike/work/petsc-3.23.3
export PETSC_ARCH=arch-linux-c-opt

if [ "$EXPRESS" = false ]; then

  echo "IN STATEMENT"
  # Prepare CMake arguments
  CMAKE_ARGS="-B $BUILD_DIR -DBOUT_DOWNLOAD_SUNDIALS=ON -DBOUT_USE_PETSC=ON -DHERMES_SLOPE_LIMITER=MC"

  if [ "$NO_BOUT" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DHERMES_BUILD_BOUT=OFF"
    echo "Disabling BOUT++ build"
  fi

  # Enter the directories and compile
  cd hermes-3

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
  cd hermes-3
fi

cmake --build $BUILD_DIR -j 32 #--verbose