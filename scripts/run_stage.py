"""Compatibility entry point; use provisioned Python -B, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=src."""
import sys
if not sys.dont_write_bytecode:
    print('blocked: bytecode-disabled startup required; use python -B with PYTHONDONTWRITEBYTECODE=1', file=sys.stderr)
    raise SystemExit(2)
from oxyformer.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['run-stage', *sys.argv[1:]]))
