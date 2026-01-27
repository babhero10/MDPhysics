#!/bin/bash
#SBATCH --job-name=MDPhysics_web
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/project/c2229/multi_model/image_deblurring/MDPhysics/logs/log_web.out
#SBATCH --error=/ibex/project/c2229/multi_model/image_deblurring/MDPhysics/logs/log_web.err

# Load Conda
source ~/.bashrc
conda activate fundus_conda
module load cuda/12.1

# Move to working directory
cd /ibex/project/c2229/multi_model/image_deblurring/MDPhysics

export MODEL_CONFIG=configs/model/mdt_edited.yaml
export DEVICE=cuda
export ENABLE_DEPTH=true

uvicorn src.web.app:app --host 0.0.0.0 --port 8000
