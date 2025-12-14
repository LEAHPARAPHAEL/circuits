#!/bin/bash
#SBATCH -p mesonet 
#SBATCH --account=m25146
#SBATCH --job-name=minimality_paper_circuit
#SBATCH --output=logs/minimality.out
#SBATCH --error=logs/minimality.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Activate virtual environment
source env_circuit/bin/activate

# Run the training script
python minimality.py
