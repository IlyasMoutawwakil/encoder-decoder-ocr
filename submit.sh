#!/bin/bash

#SBATCH --partition=gpu_inter
#SBATCH --time=2:00:00
#SBATCH --signal=SIGUSR1@90

srun python3 main.py train_ocr