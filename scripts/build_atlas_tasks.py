"""Generate reviewed atlas tasks using only local acquisition headers."""
import argparse
from pathlib import Path
import sys

if not sys.dont_write_bytecode:
    raise SystemExit('Use python -B')

from oxyformer.execution.atlas_tasks import build_tasks, inspect_dem, write_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dem-acquisition', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    write_tasks(args.out, build_tasks(inspect_dem(args.dem_acquisition)))


if __name__ == '__main__':
    main()
