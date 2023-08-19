#!/bin/bash 
#SBATCH --nodes=1                        # requests 1 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:rtx8000:1  ## To request specific GPU (v100 or rtx8000)
#SBATCH --time=05:30:00
#SBATCH --mem-per-cpu=20GB
#SBATCH --job-name=baseline_train
#SBATCH --output=baseline_train5.out


python3 Baseline_train.py --config_path='base.2urban' --test_model_path='' \

