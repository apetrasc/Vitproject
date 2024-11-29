#!/bin/bash -l
#SBATCH -A NAISS2023-3-13 -p alvis
#SBATCH -J eval
#SBATCH -t 00:29:00
#SBATCH -N 1 --gpus-per-node=A100:4
# Number of MPI processes per node (the following is actually the default)
# SBATCH --ntasks-per-node=4
#SBATCH -e sb_error.e
#SBATCH -o sb_output.o
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source sadanori/bin/activate
python3 ViT-predict.py > eval/eval_vit.txt
