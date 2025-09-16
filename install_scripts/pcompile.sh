# The script will perform a full clean install of PETSc
# Make sure you edit the number of cores in the with-make-np flag if needed!

PETSC_VERSION="3.23.3"

rm -rf "petsc-$PETSC_VERSION.tar.gz"
wget "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-$PETSC_VERSION.tar.gz"
rm -rf "petsc-$PETSC_VERSION"
tar -xf "petsc-$PETSC_VERSION.tar.gz"
cd "petsc-$PETSC_VERSION"
./configure \
    COPTFLAGS="-O3" \
    CXXOPTFLAGS="-O3" \
    FOPTFLAGS="-O3"\
    --with-fortran-bindings=0 \
    --with-debugging=0 \
    --with-mpi=yes \
    --download-hypre \
    --download-make \
    --download-openblas=1 \
    --download-metis \
    --download-parmetis \
    --download-zfp \
    --download-strumpack \
    --download-scalapack \
    --download-ptscotch \
    --download-mumps \
    --download-superlu \
    --download-suitesparse \
    --download-superlu_dist \
    --download-slepc \
    --download-hpddm \
    --with-make-np=32    # Edit if needed

make PETSC_DIR=$PWD PETSC_ARCH=arch-linux-c-opt all

