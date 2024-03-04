# Ai movement Seminars: Hands-on training on High Performance Computing

The workshop aims to introduce new users to working with HPC and get them to level
with the necessary technologies.

## Workshop highlights

### 1. Explain Toubkal HPC and its structure
Demystify what HPC is and explain its capabilities and partitions.

### 2. Show how to connect to the HPC
- The workshop aims to introduce new users to working with HPC and get them to level with the necessary technologies.
- Enumerate the required steps needed to connect to HPC
- Applications to use to ease the connection process across multiple OSs

### 3. SLURM: Workload manager & job scheduler
- Explaining the general idea behind SLURM
- How to properly interact with HPC through SLURM

### 4. What can be done and can’t be done in HPC
- Commands we are used to but are intentionally blocked or restricted for security
or other reasons.
- Alternative commands and solutions

### 5. Hands-on applications and examples: (Tips and tricks)
- How to set your development environment
- Load the necessary modules (SLURM terminology)
- Set up your Python environment (e.g., conda environment)
- Introduction to bash scripting (needed for interacting with SLURM)
- Running Python scripts in SLURM
- Train a neural network

### 6. Monitoring training with Weights and Biases (wandb)
- Setting wandb to work with your Python scripts
- Introducing the wandb platform
- Optimize your networks’ hyperparameters automatically with wandb (sweeps)

***
***

## Connect to HPC through a terminal
For Simlab:
```bash
$ ssh <login>@simlab-cluster.um6p.ma
```

For Toubkal:
```bash
$ ssh <login>@toubkal.hpc.um6p.ma
```

## Check Available Modules
```bash
$ module avail <module name>
```
**OR**
```bash
$ module spider <module name>
```
This command gives you detailed description of available modules with the specified name.
This command looks for the ```<module name>``` in the available modules or in their descriptions and returns the matches.
### eg.
```bash
$ module avail cuda

------------------------- /srv/software/easybuild/modules/all ---------------------------
   Anaconda2/2019.10    Anaconda3/2020.11
```
## Load Module
```bash
$ module load Anaconda3/2020.11
```

## SLURM commands

### To monitor jobs which are waiting or in execution
```bash
$ squeue -u $USER 

JOBID  PARTITION  NAME  USER  ST   TIME  NODES  NODELIST(REASON)   
  235  part_name  test   abc   R  00:02      1  node01 
```
Where *JOBID*: Job identifier *PARTITION*: Partition used *NAME*: Job name *USER*: User name of job owner *ST*: Status of job execution ( R=running, PD=pending, CG=completing ) *TIME*: Elapsed time *NODES*: Number of nodes used *NODELIST*: List of nodes used.

### Cancel your job
```sh
$ scancel <JOBID> 
```

### pt_cloud-

### Obtaining a terminal on a GPU compute node

It is possible to open a terminal directly on a compute node on which the resources have been reserved for you (here 4 cores) by using the following command:
```sh
$ srun --pty --partition=gpu --nodes=1  --gres=gpu:1 [--other-options] bash
```

## Make your Bash script
```sh
#!/bin/bash

#SBATCH --partition=gpu # partition name
#SBATCH --account=<lab_account> # ONLY for Toubkal users
#SBATCH --nodes=1  # number of nodes to reserve
#SBATCH --gres=gpu:1 # use 1 gpus (On toubkal each node has 4 gpus)

module load 

```

## Run interactive node