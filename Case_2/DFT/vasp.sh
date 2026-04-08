#!/bin/bash
#JSUB -J CASE1_FDM
#JSUB -n 28
#JSUB -q normal
##JSUB -e _err.%J
#JSUB -o out.%J

module purge
module load tbb/latest
module load compiler-rt/latest
module load mkl/2025.3
module load intelmpi/mpi
module load vasp/6.3.2

ulimit -s unlimited

dir=`pwd`


mpirun vasp_std > output


echo "Job %J done at " `date`
