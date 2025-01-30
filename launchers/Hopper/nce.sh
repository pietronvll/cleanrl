#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR



# Load environment
source ~/.bashrc
conda activate cleanrl

python cleanrl/RepL/NCE/main.py --env "Hopper-v3"