#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

SEED=${SEED:-1}

# Load environment
source ~/.bashrc
mamba activate cleanrl

python cleanrl/sac_continuous_action.py \
    --exp_name "sac" \
    --env-id "HalfCheetah-v4" \
    --seed $SEED \