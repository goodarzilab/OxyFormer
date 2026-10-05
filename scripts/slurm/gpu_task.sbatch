#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=root
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gpus=1
# Invoked by an already allocated worker; this script never submits work.
set -euo pipefail
: "${SWARM_UNIT_DIR:?}"
: "${STAGE:?}"
git clone --depth 1 --branch dev https://github.com/goodarzilab/OxyFormer.git "$SWARM_UNIT_DIR/src"
git -C "$SWARM_UNIT_DIR/src" rev-parse HEAD > "$SWARM_UNIT_DIR/code_commit.txt"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SWARM_UNIT_DIR/src/src"
if [[ -n "${TASK_MANIFEST:-}" && "$TASK_MANIFEST" != /* ]]; then TASK_MANIFEST="$PWD/$TASK_MANIFEST"; fi
cd "$SWARM_UNIT_DIR/src"
args=(--stage "$STAGE" --repo "$SWARM_UNIT_DIR/src" --out "$SWARM_UNIT_DIR" --deps-env
      --approvals "$SWARM_UNIT_DIR/src/configs/approvals.yaml")
if [[ -n "${TASK_MANIFEST:-}" ]]; then args+=(--task "$TASK_MANIFEST"); fi
if [[ -n "${TASK_ID:-}" ]]; then args+=(--task-id "$TASK_ID"); fi
exec /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python -m oxyformer.cli run-stage "${args[@]}" > "$SWARM_UNIT_DIR/run.log" 2>&1
