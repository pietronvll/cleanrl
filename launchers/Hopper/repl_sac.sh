#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# # Use an optional experiment argument passed via qsub -v EXPERIMENT=...
REP_LOSS=${REP_LOSS:-supervised}
Q_FEATURE_TRAIN="${Q_FEATURE_TRAIN:-True}"
LEARNING_STARTS=${LEARNING_STARTS:-1000}
REWARD_WEIGHT=${REWARD_WEIGHT:-1.0}
CRITIC_LAYERS=${CRITIC_LAYERS:-1}
# SEED=${SEED:0}

if [ "$CRITIC_TRAINING" == "True" ]; then
  critic_training_arg="--critic_feat_training"
else
  critic_training_arg="--no-critic_feat_training"
fi

if [ "$REP_LOSS" == "nce" ]; then
  additional_args="--cont_batch_size 512"
else
  additional_args=""
fi

# Load environment
source ~/.bashrc
conda activate cleanrl

python cleanrl/RepL/replearn_sac_continuous_action.py \
  --env_id "Hopper-v4" \
  --rep-loss "$REP_LOSS" \
  --critic-layers $CRITIC_LAYERS \
  --learning-starts $LEARNING_STARTS \
  --reward-weight $REWARD_WEIGHT \
  $critic_training_arg \
  --track \
  $additional_args
