"""Expand a reviewed finite campaign spec; output only, never dispatch."""
import sys
if not sys.dont_write_bytecode:
    print('blocked: bytecode-disabled startup required; use python -B with PYTHONDONTWRITEBYTECODE=1', file=sys.stderr)
    raise SystemExit(2)

import argparse
import os
from pathlib import Path

from oxyformer.execution.campaign import expand_campaign
from oxyformer.execution.paths import atomic_json, output_path
from oxyformer.execution.integrity import read_regular
from oxyformer.execution.identity import git_bytes
from oxyformer.execution.runner import read_mapping
from oxyformer.provenance import require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--approvals', help='must match this checkout owner record')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    # Authority comes from the checkout containing this production entry point,
    # not from an alternate mapping selected alongside a campaign specification.
    authoritative = Path(__file__).resolve().parents[1] / 'configs/approvals.yaml'
    committed_approvals = git_bytes(authoritative.parents[1], 'show', 'HEAD:configs/approvals.yaml')
    approvals = read_mapping(authoritative, expected_bytes=committed_approvals)
    if args.approvals is not None:
        require(read_regular(args.approvals) == committed_approvals,
                'supplied approvals differ from authoritative owner approvals')
    plan = expand_campaign(read_mapping(args.spec), approvals)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out = out.resolve(strict=True)
    outputs = {'task_manifest.json': {'schema_version': 1, 'tasks': plan['tasks']},
               'expanded_units.json': plan}
    # Admit the complete reserved set before exposing either final manifest.
    # Individual publications remain create-once; this is not a transaction.
    for name in outputs:
        require(not os.path.lexists(output_path(out, name)), f'output already exists: {name}')
    for name, value in outputs.items():
        atomic_json(out, name, value)


if __name__ == '__main__':
    main()
