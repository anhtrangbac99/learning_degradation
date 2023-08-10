#!/bin/sh 
#SBATCH -o outputs/GOPRO/train_image_conditioning/job_%j.output
#SBATCH -e errors/GOPRO/train_image_conditioning/job_%j.error
#SBATCH -p RTXA6Kq
#SBATCH --gres=gpu:1
#SBATCH -n 2
#SBATCH -c 2

module load cuda11.1/toolkit 
module load cuda11.1/blas/11.1.1 
source activate science
python train_image_conditioning.py --batchsize 1 --modch 64 --data_path datasets/GOPRO/train --dataset gopro  