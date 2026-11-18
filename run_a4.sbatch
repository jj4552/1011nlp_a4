#!/bin/bash
#SBATCH --job-name=a4-v6
#SBATCH --account=torch_pr_932_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --requeue
#SBATCH --output=/scratch/jj4552/nlp1011/logs/slurm-%j.log

set -u
cd /scratch/jj4552/nlp1011 || exit 2
mkdir -p logs .cache/huggingface tmp
export HF_HOME="$PWD/.cache/huggingface"
export XDG_CACHE_HOME="$PWD/.cache"
export TMPDIR="$PWD/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python3 -u run_pipeline.py