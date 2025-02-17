#!/bin/bash

rep_loss_arr="infonce" #supervised spectral nce infonce
reward_weight_arr="0.5"
critic_layers_arr="1 2"
critic_feat_training_arr="True False"
policy_feat_training_arr="True False"
reward_prediction_loss_arr="True False"
seed_arr="1"


for seed in $seed_arr; do
  for critic_feat_training in $critic_feat_training_arr; do
    for policy_feat_training in $policy_feat_training_arr; do
      for reward_prediction_loss in $reward_prediction_loss_arr; do
        for rep_loss in $rep_loss_arr; do
          for critic_layers in $critic_layers_arr; do
            for reward_weight in $reward_weight_arr; do
              qsub -v REP_LOSS="$rep_loss",CRITIC_LAYERS="$critic_layers",REWARD_WEIGHT="$reward_weight",CRITIC_FEAT_TRAINING="$critic_feat_training",POLICY_FEAT_TRAINING="$policy_feat_training",REWARD_PREDICTION_LOSS="$reward_prediction_loss",SEED="$seed" launchers/Hopper/repl_sac.sh
            done
          done
        done
      done
    done
  done
done