#!/bin/bash
#SBATCH --job-name=deep_steg
#SBATCH --partition=ALL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=output/slurm-%j.out
#SBATCH --error=output/slurm-%j-err.out
#SBATCH --mail-type=END,FAIL

# Activate conda environment
source activate deep_steg

# Set LD_LIBRARY_PATH for GPU support
export LD_LIBRARY_PATH=$(python -c "
import nvidia, os
base = os.path.join(os.path.dirname(nvidia.__path__[0]), 'nvidia')
dirs = ['cudnn','cuda_runtime','cublas','cufft','cuda_cupti','cusolver','cusparse','nccl','nvtx']
print(':'.join(os.path.join(base, d, 'lib') for d in dirs))
")

# Create output directory
mkdir -p output

# Run training
python train.py \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --train_per_class 10 \
    --test_images 500 \
    --data_dir ./tiny-imagenet-200 \
    --output_dir ./output
