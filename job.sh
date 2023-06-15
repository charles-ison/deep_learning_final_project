#!/bin/bash
#SBATCH -w cn-m-2
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH --job-name=stemgen
#SBATCH -t 3-00:00:00
#SBATCH -c 1
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --export=ALL


#SBATCH -o logs/train_model.out
#SBATCH -e logs/train_model.err

# load any software environment module required for app (e.g. matlab, gcc, cuda)
module load python/3.10 cuda/11.7 sox

# load env
source env/bin/activate

# run model
python train_model.py