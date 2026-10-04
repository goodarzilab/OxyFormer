"""Common stage CLI. Import scientific modules only after validating a request."""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog='oxyformer')
    commands = parser.add_subparsers(dest='command', required=True)
    stage = commands.add_parser('run-stage')
    stage.add_argument('--stage', required=True)
    stage.add_argument('--out', required=True)
    stage.add_argument('--repo', required=True)
    stage.add_argument('--deps-env', action='store_true')
    stage.add_argument('--task', dest='task_file')
    stage.add_argument('--task-id')
    stage.add_argument('--approvals')
    args = vars(parser.parse_args(argv))
    args.pop('command')
    from oxyformer.execution.runner import run
    try:
        result = run(**args)
    except (ValueError, OSError, KeyError) as exc:
        print(f'blocked: {exc}', file=sys.stderr)
        return 2
    print(result.to_json())
    return {'pass': 0, 'fail': 1, 'blocked': 2}[result.status]


if __name__ == '__main__':
    raise SystemExit(main())
