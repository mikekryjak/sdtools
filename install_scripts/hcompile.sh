#!/bin/bash

BUILD_DIR="build-mc-master"

FORCE_REBUILD=true
NO_BOUT=false


# Check for force rebuild flag
while getopts "s" opt; do
  case $opt in
    f)
      FORCE_REBUILD=false
      ;;
    *)
      echo "Usage: $0 [-f]"
      echo "  -s  Speedy build, do not remove existing build directory"
      exit 1
      ;;
  esac
done


# Check for build BOUT++ flag
while getopts "nobout" opt; do
  case $opt in
    f)
      NO_BOUT=true
      ;;
    *)
      echo "Usage: $0 [-f]"
      echo "  -nobout do not rebuild BOUT++"
      exit 1
      ;;
  esac
done


# CMake settings
export PETSC_DIR=/home/mike/work/petsc-3.22.1
export PETSC_ARCH=arch-linux-c-opt

### WARNING: SUNDIALS IS OFF
# Prepare CMake arguments
CMAKE_ARGS="-B $BUILD_DIR -DBOUT_DOWNLOAD_SUNDIALS=ON -DBOUT_USE_PETSC=ON -DHERMES_SLOPE_LIMITER=MC"

if [ "$NO_BOUT" = true ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DHERMES_BUILD_BOUT=OFF"
  echo "Disabling BOUT++ build"
fi

# Enter the directories and compile
cd hermes-3

# Make fresh build directory if required
if [ "$FORCE_REBUILD" = true ]; then
  echo "Forcing rebuild by removing existing build directory..."
  rm -rf "$BUILD_DIR"
fi

# Compile
cmake . -B $BUILD_DIR $CMAKE_ARGS
cmake --build $BUILD_DIR -j 32 #--verbose