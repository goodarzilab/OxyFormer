"""phase3 compatibility entry point; select an explicit mode and isolated output.

Initial-release algorithms are executed from the verified pinned source archive.
Repaired PLR is a B0 association benchmark. v2 delegates to oxyformer.cli.
"""
from oxyformer.legacy.launcher import main as launch


def main(argv=None):
    return launch("phase3", argv)


if __name__ == "__main__":
    raise SystemExit(main())


from oxyformer.legacy.benchmark import cross_fitted_partial_linear_dml
