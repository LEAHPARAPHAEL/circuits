#!/bin/bash
#SBATCH -p mesonet 
#SBATCH --account=m25146
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Activate virtual environment
source env_circuit/bin/activate

# Run the training script
python minimality.py \
    -t 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
