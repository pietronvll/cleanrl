#!/bin/bash

env_arr="Hopper-v4" #HalfCheetah-v4 Ant-v4
rep_loss_arr="spectral" # nce infonce supervised
seed_arr="1 2 3 4 5"
critic_layers_arr="1 2"

for env in $env_arr; do
  for rep_loss in $rep_loss_arr; do
    for seed in $seed_arr; do
      for critic_layers in $critic_layers_arr; do        
          qsub -v REP_LOSS="$rep_loss",CRITIC_LAYERS="$critic_layers",SEED="$seed",ENV_ID="$env" launchers/benchmark_single_run.sh
      done
    done
  done
done