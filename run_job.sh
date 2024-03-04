#!/bin/bash
#SBATCH --job-name=test_job             # Name your job
#SBATCH --partition=gpu                 # Setting the partition
#SBATCH --gres=gpu:1                    # You need this for gpu, remove if not using the gpu
#SBATCH --account=PT_CLOUD-MUHQXQC6FXO-DEFAULT-GPU         # GPU account: you can find it by running the command "mybalance"
#SBATCH --output=output%j.log  # Standard output/error log nta ou zehrek (%j to add date info to the log file name)

pwd; hostname; date # Log the current working directory, hostname and date at the strat of the execution

user="anass.grini" # <-- Add your HPC username here

# If your job will need to use the gpu ressource, add this line also
# module load cuDNN and CUDA
# module load cuDNN/8.2.1.32-CUDA-11.3.1
# module load cuda11/toolkit/11.7.0
# module load Python/3.8.2-GCCcore-9.3.0

# Uncomment if you work with conda environments
# load modules or conda environments here (you can use your own conda environment)
# module load Anaconda3
source /home/${user}/.bashrc

conda activate ML

nvidia-smi

# Run your python script (You copy-paste the .py file path)
python3.10 /home/anass.grini/anass.grini/aim_HPC_handson/train_wandb.py

# Launch Sweep
# wandb agent anass-gr/wandb_test/c6z4lg1y

date






