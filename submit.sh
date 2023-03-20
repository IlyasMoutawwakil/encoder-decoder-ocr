# (submit.sh)
#!/bin/bash -l

# SLURM SUBMIT SCRIPT
SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
SBATCH --gres=gpu:1
SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
SBATCH --mem=0
SBATCH --time=0-02:00:00
SBATCH --signal=SIGUSR1@90
SBATCH --output=slurm-%j.out
SBATCH --error=slurm-%j.err

# activate conda env
# source activate env

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# run script
srun python3 main.py train_ocr