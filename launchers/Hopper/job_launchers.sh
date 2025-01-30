#!/bin/bash

rep_loss_arr="supervised spectral nce" #supervised spectral 
learning_starts_arr="1000"
reward_weight_arr="0.5"
critic_layers_arr="1 2"
critic_traying_arr="True"

for critic_training in $critic_traying_arr; do
  for rep_loss in $rep_loss_arr; do
    for critic_layers in $critic_layers_arr; do
      for learning_starts in $learning_starts_arr; do
        for reward_weight in $reward_weight_arr; do
          qsub -v REP_LOSS="$rep_loss",CRITIC_LAYERS="$critic_layers",LEARNING_STARTS="$learning_starts",REWARD_WEIGHT="$reward_weight",CRITIC_TRAINING="$critic_training" launchers/Hopper/repl_sac.sh
        done
      done
    done
  done
done