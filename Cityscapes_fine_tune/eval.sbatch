#!/bin/bash 
#SBATCH --nodes=1                        # requests 1 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=4                # uses 1 compute core per task
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:rtx8000:1  ## To request specific GPU (v100 or rtx8000)
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=20GB
#SBATCH --job-name=eval
#SBATCH --output=eval.out


python3 evall.py


#--resume 'checkpoint_0017.pth.tar' \
