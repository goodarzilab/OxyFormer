"""Expand a reviewed finite campaign spec; output only, never dispatch."""
import argparse
from pathlib import Path

from oxyformer.execution.campaign import expand_campaign
from oxyformer.execution.paths import atomic_json
from oxyformer.execution.runner import read_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--approvals', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    plan = expand_campaign(read_mapping(args.spec), read_mapping(args.approvals))
    out = Path(args.out).resolve(strict=True)
    atomic_json(out, 'task_manifest.json', {'schema_version': 1, 'tasks': plan['tasks']})
    atomic_json(out, 'expanded_units.json', plan)


if __name__ == '__main__':
    main()
