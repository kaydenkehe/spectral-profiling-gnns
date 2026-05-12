#!/usr/bin/env bash
#SBATCH --job-name=jacobi-massive
#SBATCH --partition=seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=5:59:00
#SBATCH --array=0-18
#SBATCH --output=slurm_logs/jacobi_massive_%A_%a.out
#SBATCH --error=slurm_logs/jacobi_massive_%A_%a.err

set -euo pipefail

REPO_DIR="/n/home06/drooryck/spectral-profling-gnns"
UV_BIN="/n/home06/drooryck/.local/bin/uv"
OUT_ROOT="${OUT_ROOT:-jacobi_ab_sweep_spatial_masks_massive}"
K_VALUES="${K_VALUES:-4 10}"

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

PYTHONUNBUFFERED=1 "$UV_BIN" run python spectral/jacobi_ab_sweep_massive.py \
  --datasets "$DATASET" \
  --K $K_VALUES \
  --a-min -0.99 \
  --a-max 4.0 \
  --b-min -0.99 \
  --b-max 4.0 \
  --step 0.1 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --epochs 500 \
  --patience 50 \
  --mask-dir spatial/masks \
  --device cuda \
  --out-dir "$RUN_OUT_DIR"
