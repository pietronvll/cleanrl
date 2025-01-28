#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=04:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# # Use an optional experiment argument passed via qsub -v EXPERIMENT=...
REP_LOSS=${REP_LOSS:-supervised}
Q_FEATURE_TRAIN="${Q_FEATURE_TRAIN:-True}"
LEARNING_STARTS=${LEARNING_STARTS:-1000}
REWARD_WEIGHT=${REWARD_WEIGHT:-1.0}
CRITIC_LAYERS=${CRITIC_LAYERS:-1}
# SEED=${SEED:0}

# if [ "$Q_FEATURE_TRAIN" == "true" ]; then
#   q_feature_arg="--q-feature-train"
# else
#   q_feature_arg="--no-q-feature-train"
# fi

# Load environment
source ~/.bashrc
conda activate cleanrl

python cleanrl/RepL/replearn_sac_continuous_action.py \
  --env_id "HalfCeetah-v4" \
  --rep-loss "$REP_LOSS" \
  --critic-layers $CRITIC_LAYERS \
  --learning-starts $LEARNING_STARTS \
  --reward-weight $REWARD_WEIGHT \
  --track
