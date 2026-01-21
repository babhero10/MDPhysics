#!/bin/bash
#SBATCH --job-name=MDPhysics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/project/c2229/multi_model/image_deblurring/MDPhysics/logs/log_test.out
#SBATCH --error=/ibex/project/c2229/multi_model/image_deblurring/MDPhysics/logs/log_test.err

# Load Conda
source ~/.bashrc
conda activate fundus_conda
module load cuda/12.1

# Move to working directory
cd /ibex/project/c2229/multi_model/image_deblurring/MDPhysics

HYDRA_FULL_ERROR=1 python src/test.py
