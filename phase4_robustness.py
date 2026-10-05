"""phase4 compatibility entry point; select an explicit mode and isolated output.

Initial-release algorithms are executed from the verified pinned source archive.
Repaired PLR is a B0 association benchmark. v2 delegates to oxyformer.cli.
"""
from oxyformer.legacy.launcher import main as launch


def main(argv=None):
    return launch("phase4", argv)


if __name__ == "__main__":
    raise SystemExit(main())


from oxyformer.legacy.benchmark import run_leave_one_state_out
