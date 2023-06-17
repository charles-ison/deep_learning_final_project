#!/bin/bash
#SBATCH -w dgx2-6
#SBATCH -p dgx2
#SBATCH --job-name=stemgen
#SBATCH -t 1-00:00:00
#SBATCH -c 4
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --export=ALL

#SBATCH -o logs/train_model.out
#SBATCH -e logs/train_model.err

# load any software environment module required for app (e.g. matlab, gcc, cuda)
module load python/3.10 cuda/11.7 sox

# load env
source env/bin/activate

# run model
python train_model.py