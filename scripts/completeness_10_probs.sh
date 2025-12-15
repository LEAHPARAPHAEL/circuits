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
python completeness.py \
    --probabilities 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --num_sets 20
