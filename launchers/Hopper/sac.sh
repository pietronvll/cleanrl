#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Use an optional experiment argument passed via qsub -v EXPERIMENT=...

ALPHA=${ALPHA:-0.2}
AUTOTUNE=${AUTOTUNE:-True}
BATCH_SIZE=${BATCH_SIZE:-256}
BUFFER_SIZE=${BUFFER_SIZE:-1000000}
CAPTURE_VIDEO=${CAPTURE_VIDEO:-False}
CONT_BATCH_SIZE=${CONT_BATCH_SIZE:-2048}
CRITIC_FEAT_TRAINING=${CRITIC_FEAT_TRAINING:-True}
CRITIC_HIDDEN_DIM=${CRITIC_HIDDEN_DIM:-256}
CRITIC_LAYERS=${CRITIC_LAYERS:-1}
CUDA=${CUDA:-True}
ENV_ID=${ENV_ID:-Hopper-v4}
EXP_NAME=${EXP_NAME:-repsac}
EXTRA_FEATURE_STEPS=${EXTRA_FEATURE_STEPS:-4}
FEATURE_DIM=${FEATURE_DIM:-256}
FEATURE_HIDDEN_DIM=${FEATURE_HIDDEN_DIM:-256}
FEAT_LR=${FEAT_LR:-0.001}
FEATURE_TAU=${FEATURE_TAU:-0.005}
GAMMA=${GAMMA:-0.99}
LEARNING_STARTS=${LEARNING_STARTS:-5000}
NOISE_CLIP=${NOISE_CLIP:-0.5}
POLICY_FREQUENCY=${POLICY_FREQUENCY:-2}
POLICY_LR=${POLICY_LR:-0.0003}
Q_LR=${Q_LR:-0.001}
REP_LOSS=${REP_LOSS:-nce}
SEED=${SEED:-1}
TARGET_NETWORK_FREQUENCY=${TARGET_NETWORK_FREQUENCY:-1}
TAU=${TAU:-0.005}
TORCH_DETERMINISTIC=${TORCH_DETERMINISTIC:-True}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000}
TRACK=${TRACK:-True}
USE_FEATURE_TARGET=${USE_FEATURE_TARGET:-False}
WANDB_ENTITY=${WANDB_ENTITY:-None}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-cleanRL}


# # Load environment
source ~/.bashrc
conda activate cleanrl

echo python cleanrl/sac_continuous_action_multi_env.py \
  --exp_name "${EXP_NAME}" \
  --seed "${SEED}" \
  $( [ "${TORCH_DETERMINISTIC}" == "True" ] && echo "--torch_deterministic" || echo "--no-torch_deterministic" ) \
  $( [ "${CUDA}" == "True" ] && echo "--cuda" || echo "--no-cuda" ) \
  $( [ "${TRACK}" == "True" ] && echo "--track" || echo "--no-track" ) \
  --wandb_project_name "${WANDB_PROJECT_NAME}" \
  $( [ "${CAPTURE_VIDEO}" == "True" ] && echo "--capture_video" || echo "--no-capture_video" ) \
  --env_id "${ENV_ID}" \
  --n_envs 16 \
  --total_timesteps $TOTAL_TIMESTEPS \
  --buffer_size $BUFFER_SIZE \
  --gamma $GAMMA \
  --tau $TAU \
  --batch_size $BATCH_SIZE \
  --learning_starts $LEARNING_STARTS \
  --policy_lr $POLICY_LR \
  --q_lr $Q_LR \
  --policy_frequency $POLICY_FREQUENCY \
  --target_network_frequency $TARGET_NETWORK_FREQUENCY \
  --alpha $ALPHA \
  $( [ "${AUTOTUNE}" == "True" ] && echo "--autotune" || echo "--no-autotune" ) \
  