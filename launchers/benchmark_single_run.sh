#!/bin/bash
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# # Use an optional experiment argument passed via qsub -v EXPERIMENT=...
REP_LOSS=${REP_LOSS:-supervised}
onal_args=""
fi

# Load environment
source ~/.bashrc
conda activate cleanrl

python cleanrl/RepL/replearn_sac_continuous_action.py \
  --env_id $ENV_ID \
  --rep-loss "$REP_LOSS" \
  --track \
  --wandb_project_name "BenchmarkCleanRL" \ 
  --seed $SEED \