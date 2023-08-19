#!/bin/bash 
#SBATCH --nodes=1                        # requests 1 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:v100:1  ## To request specific GPU (v100 or rtx8000)
#SBATCH --time=5:00:00
#SBATCH --mem-per-cpu=20GB
#SBATCH --job-name=train
#SBATCH --output=train.out


python3 train.py


#--resume 'checkpoint_0017.pth.tar' \
