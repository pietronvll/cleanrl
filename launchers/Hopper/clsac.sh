#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Use an optional experiment argument passed via qsub -v EXPERIMENT=...
SEED=${SEED:-1}
# Load environment
source ~/.bashrc
conda activate cleanrl

python cleanrl/RepL/cl_sac_continuous_action.py 