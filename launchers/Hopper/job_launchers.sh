#!/bin/bash

rep_loss_arr="supervised contrastive nce"
q_feature_train_arr="false"
learning_starts_arr="1000 25000"
reward_weight_arr="1.0 0.1"

for rep_loss in $rep_loss_arr; do
  for q_feature_train in $q_feature_train_arr; do
    for learning_starts in $learning_starts_arr; do
      for reward_weight in $reward_weight_arr; do
        qsub -v REP_LOSS="$rep_loss",Q_FEATURE_TRAIN="$q_feature_train",LEARNING_STARTS="$learning_starts",REWARD_WEIGHT="$reward_weight" launchers/Hopper/repl_sac.sh
      done
    done
  done
done