#!/bin/bash
# Invoked by an already allocated worker; this script never submits work.
set -euo pipefail
export GIT_NO_REPLACE_OBJECTS=1
: "${SWARM_UNIT_DIR:?}"
: "${STAGE:?}"
if [[ "${1:-}" != --prepared ]]; then
    git clone --depth 1 --branch dev https://github.com/goodarzilab/OxyFormer.git "$SWARM_UNIT_DIR/src"
    git -C "$SWARM_UNIT_DIR/src" rev-parse HEAD > "$SWARM_UNIT_DIR/code_commit.txt"
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SWARM_UNIT_DIR/src/src"
cd "$SWARM_UNIT_DIR/src"
if [[ -n "${TASK_MANIFEST:-}" && "$TASK_MANIFEST" != /* ]]; then TASK_MANIFEST="$PWD/$TASK_MANIFEST"; fi
args=(--stage "$STAGE" --repo "$SWARM_UNIT_DIR/src" --out "$SWARM_UNIT_DIR" --deps-env
      --approvals "$SWARM_UNIT_DIR/src/configs/approvals.yaml")
if [[ -n "${TASK_MANIFEST:-}" ]]; then args+=(--task "$TASK_MANIFEST"); fi
if [[ -n "${TASK_ID:-}" ]]; then args+=(--task-id "$TASK_ID"); fi
exec /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python -B -m oxyformer.cli run-stage "${args[@]}" > "$SWARM_UNIT_DIR/run.log" 2>&1
