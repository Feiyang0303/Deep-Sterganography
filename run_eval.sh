#!/bin/bash
#SBATCH --job-name=steg_eval
#SBATCH --partition=ALL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=output/eval-%j.out
#SBATCH --error=output/eval-%j-err.out

# Activate conda environment
source activate deep_steg

# Set LD_LIBRARY_PATH for GPU support
export LD_LIBRARY_PATH=$(python -c "
import nvidia, os
base = os.path.join(os.path.dirname(nvidia.__path__[0]), 'nvidia')
dirs = ['cudnn','cuda_runtime','cublas','cufft','cuda_cupti','cusolver','cusparse','nccl','nvtx']
print(':'.join(os.path.join(base, d, 'lib') for d in dirs))
")

mkdir -p output

python -u evaluate.py \
    --weights ./output/model_weights.weights.h5 \
    --data_dir ./tiny-imagenet-200 \
    --output_dir ./output
