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
export GIT_NO_REPLACE_OBJECTS=1
: "${SWARM_UNIT_DIR:?}"
: "${STAGE:?}"
git clone --depth 1 --branch dev https://github.com/goodarzilab/OxyFormer.git "$SWARM_UNIT_DIR/src"
git -C "$SWARM_UNIT_DIR/src" rev-parse HEAD > "$SWARM_UNIT_DIR/code_commit.txt"
exec bash "$SWARM_UNIT_DIR/src/scripts/slurm/run_stage.sh" --prepared
