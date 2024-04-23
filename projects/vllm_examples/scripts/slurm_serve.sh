#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH -A gts-wliao60
#SBATCH --ntasks=1                 # Number of tasks (CPU cores) to request
#SBATCH -N1 --gres=gpu:RTX_6000:4                           # Number of nodes and GPUs required
#SBATCH --gres-flags=enforce-binding                # Map CPUs to GPUs
#SBATCH -t 8:00:00                                        # Duration of the job (Ex: 15 mins)
#SBATCH -o /storage/coda1/p-wliao60/0/ahavrilla3/alex/repos/slurm_utils/slurm_logs/Report-%j.out
#SBATCH -q embers

MODEL_PATH=$1
HOSTNAME_PATH=$2

# Load any necessary modules (optional)
module load git
module load gcc

# Change to the directory where you want to run your job
source ~/.envs/vllm2/bin/activate

# Get hostname and store in HOSTNAME_PATH
hostname > $HOSTNAME_PATH

# Your job commands go here
# For example:
srun python -m vllm.entrypoints.openai.api_server --model $MODEL_PATH

# Remember to exit with an appropriate exit code (0 for success, non-zero for failure)
exit 0