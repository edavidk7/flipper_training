#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24 --nodes=1 --tasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpu # partition name
#SBATCH --job-name=RL_optuna

module restore basic_python
source ~/.venv/bin/activate
cd /mnt/personal/korcadav/flipper_training
srun -n1 -N1 python flipper_training/experiments/ppo/optuna_train.py --train_config test_configs/random_trunk_random_start_goal.yaml --optuna_config test_configs/optuna_for_random_trunk.yaml use_gpus=[0]
