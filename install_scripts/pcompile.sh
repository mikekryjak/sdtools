wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.22.1.tar.gz
tar -xf petsc-3.22.1.tar.gz
cd petsc-3.22.1/
./configure --with-mpi=yes --download-hypre --download-make --with-fortran-bindings=0 --with-debugging=0 --download-fblaslapack=1
make PETSC_DIR=/home/mike/work/petsc-3.22.1 PETSC_ARCH=arch-linux-c-opt all -j 32
