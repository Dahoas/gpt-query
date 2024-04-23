import argparse


template = """\
#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH -A gts-wliao60
#SBATCH --ntasks=1                 # Number of tasks (CPU cores) to request
#SBATCH -N1 --gres=gpu:RTX_6000:4                           # Number of nodes and GPUs required
#SBATCH --gres-flags=enforce-binding                # Map CPUs to GPUs
#SBATCH -t 8:00:00                                        # Duration of the job (Ex: 15 mins)
#SBATCH -o /storage/coda1/p-wliao60/0/ahavrilla3/alex/repos/slurm_utils/slurm_logs/Report-%j.out
#SBATCH -q embers

HOSTNAME_PATH={hostname_path}

# Load any necessary modules (optional)
module load git
module load gcc

# Change to the directory where you want to run your job
source ~/.envs/vllm2/bin/activate

# Get hostname and store in HOSTNAME_PATH
hostname > $HOSTNAME_PATH

# Your job commands go here
# For example:
srun python -m vllm.entrypoints.openai.api_server {vllm_flags}

# Remember to exit with an appropriate exit code (0 for success, non-zero for failure)
exit 0
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--hostname_path", type=str)
    parser.add_argument("--script_save_path", type=str)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--awq", action="store_true")
    args = parser.parse_args()

    vllm_flags = f"--model {args.model_path} "
    if args.fp16: vllm_flags += "--dtype half "
    if args.tensor_parallel_size: vllm_flags += f"--tensor-parallel-size {args.tensor_parallel_size} "
    if args.awq: vllm_flags += f"--awq "

    script = template.format(hostname_path=args.hostname_path, vllm_flags=vllm_flags)
    with open(args.script_save_path, "w") as f:
        f.write(script)