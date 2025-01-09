#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

SEED=${SEED:-1}

# Load environment
source ~/.bashrc
mamba activate cleanrl

python cleanrl/ppo_continuous_action.py \
    --exp_name "ppo" \
    --env-id "HalfCheetah-v4" \
    --learning-rate 0.0003 \
    --seed $SEED \
    --total-timesteps 1000000 \
    --num-envs 1 \
    --num-steps 2048 \
    --anneal-lr \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --num-minibatches 32 \
    --update-epochs 10 \
    --norm-adv \
    --clip-coef 0.2 \
    --clip-vloss \
    --ent-coef 0.0 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5
