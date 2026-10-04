"""Compatibility entry point; use the provisioned Python with PYTHONPATH=src."""
import sys
from oxyformer.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['run-stage', *sys.argv[1:]]))
