#!/usr/bin/env bash
#SBATCH --job-name=spectral-massive
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-18
#SBATCH --output=slurm_logs/spectral_massive_%A_%a.out
#SBATCH --error=slurm_logs/spectral_massive_%A_%a.err

set -euo pipefail

REPO_DIR="/n/home06/drooryck/spectral-profling-gnns"
UV_BIN="/n/home06/drooryck/.local/bin/uv"
OUT_ROOT="${OUT_ROOT:-spectral_massive_spatial_masks_slurm}"
MAX_TASK_BATCH="${MAX_TASK_BATCH:-90}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASETS=(
  Cora
  PubMed
  Photo
  Texas
  Chameleon
  CiteSeer
  Computers
  CS
  Physics
  Cornell
  Wisconsin
  Actor
  WikiCS
  Squirrel
  RomanEmpire
  AmazonRatings
  Minesweeper
  Tolokers
  Questions
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATASET_DIR="${DATASET//[^A-Za-z0-9_.-]/_}"
RUN_OUT_DIR="${OUT_ROOT}/${DATASET_DIR}/array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "$REPO_DIR"
mkdir -p "$RUN_OUT_DIR"

echo "job_id=${SLURM_JOB_ID}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "dataset=${DATASET}"
echo "host=$(hostname)"
echo "out_dir=${RUN_OUT_DIR}"
date
nvidia-smi --query-gpu=name,index,memory.total --format=csv,noheader

PYTHONUNBUFFERED=1 "$UV_BIN" run python spectral/train_spectral_massive.py \
  --datasets "$DATASET" \
  --models GPRGNN ChebGNN BernNet JacobiConv \
  --k 4 10 \
  --hidden 64 128 \
  --runs 10 \
  --epochs 500 \
  --patience 50 \
  --lr 0.01 0.005 0.001 \
  --weight-decay 0.0005 0.0001 0.0 \
  --mask-dir spatial/masks \
  --device cuda \
  --max-task-batch "$MAX_TASK_BATCH" \
  --out-dir "$RUN_OUT_DIR"
