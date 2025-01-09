#!/bin/bash
python cleanrl/powr/ppo_continuous_action.py \
    --env_id "HalfCheetah-v4" \
    --num_envs 1 \
    --num_steps 2048 \
    --total_timesteps 1000000 \
    --learning_rate 3e-4 \
    --num_minibatches 32 \
    --update_epochs 10 \
    --ent_coef 0.0 \
    --clip_coef 0.2 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --max_grad_norm 0.5 \
    --norm_adv \
    --no-dump_buffer \
    --capture_video