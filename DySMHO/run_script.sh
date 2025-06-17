#!/bin/bash
#PBS -J 4-10
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -l walltime=12:00:00
#PBS -N DySMHO_0x
#PBS -k oe

module load tools/prod
module load Python/3.8.6-GCCcore-10.2.0
module load OpenBLAS/0.3.12-GCC-10.2.0

cd $PBS_O_WORKDIR

# Source the DySMHO environment
source $HOME/ODR-BINDy-DySMHO-Benchmark/DySMHO_Env/bin/activate

export PATH=$PATH:$HOME/ODR-BINDy-DySMHO-Benchmark/bindir/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/ODR-BINDy-DySMHO-Benchmark/bindir/lib
export PKG_CONFIG_PATH=$HOME/ODR-BINDy-DySMHO-Benchmark/bindir/lib/pkgconfig

export ii=$PBS_ARRAY_INDEX

# Call the Python script
python3 L_heatmap.py 
