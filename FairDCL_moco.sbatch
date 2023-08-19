#!/bin/bash 
#SBATCH --nodes=1                        # requests 1 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=4                # uses 1 compute core per task
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:rtx8000:1  ## To request specific GPU (v100 or rtx8000)
#SBATCH --time=40:00:00
#SBATCH --mem-per-cpu=20GB
#SBATCH --job-name=main_moco
#SBATCH --output=main_moco.out


python3 FairDCL_moco.py \
-a resnet50 \
--lr 0.001 \
--batch-size 32 \
--moco-dim 512 \
--mlp --moco-t 0.2 --aug-plus --cos \
--resume 'checkpoint10_0018.pth.tar' \
  /scratch/mz2466/LoveDA_mixedD_moco/Train
  #/scratch/mz2466/Face/train2017


#--resume 'checkpoint_0017.pth.tar' \
