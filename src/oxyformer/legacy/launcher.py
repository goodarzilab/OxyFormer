"""Compatibility CLI; all execution and output authority stays in oxyformer.cli.

Example (a task must carry matching parameters.mode and parameters.entrypoint):
  python -B phase3_causal_demo.py --mode repaired-benchmark --task /abs/task.json \
      --repo /abs/repo --out /abs/attempt --deps-env
v2 additionally requires --stage, one of the registered owning stages. Missing
owning modules are reported by the common worker as blocked prerequisites.
"""
import argparse
import json
from pathlib import Path

import yaml


def configuration():
    return yaml.safe_load((Path(__file__).resolve().parents[3] / "configs/legacy_entrypoints.yaml").read_text())


def main(entrypoint, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("initial-release", "repaired-benchmark", "v2"))
    parser.add_argument("--stage")
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-id")
    args, remaining = parser.parse_known_args(argv)
    config = configuration()
    task = yaml.safe_load(Path(args.task).read_text())
    if "tasks" in task:
        matches = [t for t in task["tasks"] if t.get("id") == args.task_id]
        if len(matches) != 1:
            parser.error("--task-id must select exactly one task")
        task = matches[0]
    parameters = task.get("parameters", {})
    if parameters.get("mode") != args.mode or parameters.get("entrypoint") != entrypoint:
        parser.error("task parameters.mode/entrypoint must match the compatibility launcher")
    stage = args.stage or ("legacy-reproduction" if args.mode != "v2" else None)
    if args.mode == "v2":
        if stage not in config["entrypoints"][entrypoint]["v2_stages"]:
            parser.error("v2 requires an explicit owning stage: " + json.dumps(config["entrypoints"][entrypoint]["v2_stages"]))
    elif stage != "legacy-reproduction":
        parser.error("historical/repaired benchmarks use the registered legacy-reproduction stage")
    from oxyformer.cli import main as common_main
    forwarded = ["run-stage", "--stage", stage, "--task", args.task, *remaining]
    if args.task_id:
        forwarded += ["--task-id", args.task_id]
    return common_main(forwarded)
