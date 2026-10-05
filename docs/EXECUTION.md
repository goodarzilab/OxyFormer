# Execution and verification

The [frozen science plan](plan/OXYFORMER_V2_PLAN.md), [owner mandate](orchestrator-mandate.md)
and read-only [approvals](../configs/approvals.yaml) govern this run. The merged
[interfaces](INTERFACES.md) define artifact names. A passing unit test is not a
clinical release, a compute allocation, or permission to access a source.

## Current admission status

Production campaign admission is **blocked** at base
`7eeef3f1b50fb49e2a9cdb9e21785199fa74e6aa`. Direct scientific APIs can be tested
offline; the common dispatcher is not yet an executable route through every
stage. The integration tests exercise those direct APIs, without a replacement
dispatcher or a fabricated production adapter.

Reproduced on Slurm in both the oxyformer and CI CPU environments:

| Boundary | Reproduction / required owning-unit correction |
| --- | --- |
| Dispatcher configuration | `execution.runner.read_mapping` loads the actual `approved_on` value as a date. `runner.run` includes approvals in `atomic_json`, which raises `TypeError: Object of type date is not JSON serializable` before publishing the request. Reconcile serialization without editing owner decisions. |
| Reporting task | Dispatcher requires `stage` and supplies `needs`/`outputs`; reporting accepts exactly `bundle`, `manifest`, `receipts`, `approvals`. Adding the required dispatcher field to a valid direct reporting request produces `invalid reporting task fields`. |
| Reporting configuration | Dispatcher passes `_execution/config.json`; reporting requires schema version 1 and the repository's `configs/reporting.yaml`. The dispatcher mapping produces `unsupported reporting config`; changing the version alone cannot satisfy its path requirement. |
| Stage names and design configuration | Registry names `tract-support-gate`, `anchor-review`, `audit-collect`, `tract-release` differ from direct APIs `tract_design`, `anchor_review`, `audit_collection`, `tract_release`. Design also requires its own config and dependency task shape. A name translation alone does not reconcile the schemas. |

The owning implementation units must reconcile these prerequisites and provide
merged receipts before production dispatch. Do not patch these files from a
documentation/integration unit. Admission must be reassessed on that merged
commit; this document does not certify subsequent commits.

## Coordinator contract

A code unit is **DONE only with its merged receipt** for the approved repository
and target branch `dev`. A pushed branch, open PR, passing review or local
`completion.json` is insufficient. Code dependencies therefore provide merge
barriers. Code-attempt directories are never acquisition inputs.

Every repository-running Slurm unit, including collectors and continuations,
starts with the fixed prefix emitted by `execution.campaign.stage_command`:

```sh
set -euo pipefail
: "${SWARM_UNIT_DIR:?}"
export GIT_NO_REPLACE_OBJECTS=1
git clone --depth 1 --branch dev https://github.com/goodarzilab/OxyFormer.git "$SWARM_UNIT_DIR/src"
git -C "$SWARM_UNIT_DIR/src" rev-parse HEAD > "$SWARM_UNIT_DIR/code_commit.txt"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SWARM_UNIT_DIR/src/src"
cd "$SWARM_UNIT_DIR/src"
```

The registered command then uses that clone's CLI with `--repo`, `--out`,
`--deps-env`, explicit task selection and the clone's owner registry. The
`--prepared` shell path is only for a worker that already performed the same
clone/identity preparation; it is not permission to reuse an unreviewed tree.

Only normalized dependency variables resolve upstream work: for example,
`atlas-mid-atlantic` becomes `SWARM_DEP_ATLAS_MID_ATLANTIC`. IDs must not collide
after normalization. Never search run directories for the newest output or
guess an upstream path. Verify producer IDs, passing receipts, declared files,
hashes and immutable attempt fingerprints. A stage receipt's `pass` status is
required in addition to `StageResult.verify(request)`.

Acquisition executes merged source manifests and reviewed adapters. It retains
source receipts and seals its own attempt; consumers never seal an upstream
attempt. The early fetch → nine-division atlas → atlas collection → tract
support path must not depend on `release-docs`, `campaign-lock` or final release.
Check the external coordinator DAG as well as the repository registry; local
registry tests cannot certify coordinator state.

## Scientific handoffs

`StageRequest` uses `dependency_paths`, `dependency_hashes`, `output_dir` and
`code_identity`. These merged names supersede older prompt shorthand. Each
request binds absolute input paths and hashes; artifacts use confined relative
paths. `pass`, `blocked` and `fail` are distinct. Never consume a failed or
blocked stage as a satisfied prerequisite.

The test chain computes pressure before population aggregation, runs the sealed
design/support gate, loads its frozen target and splits, fits the complete
five-fold × three-seed schedule, resumes one interrupted fit from a portable
checkpoint, assembles OOF predictions, joins outcomes for one-step/CV-TMLE,
aligns covariance by original ID, and emits a diagnostic report. It also tests
missing atlas coverage, a missing OOF row and a changed target. All data are
synthetic. One-epoch test fits cannot establish production calibration.

`PreparedEndpoint` is the fitting boundary supplied by reviewed adapters and
design. It binds data, sources, entity graph, geography, policy, treatment
knots, outer/inner splits and permitted feature families. Coordinates remain
in split/inference metadata, never predictor views. The same original ID,
fold, seed, origin weight and estimand must survive prediction assembly.
Seed averaging does not create independent observations.

A successful nested slice may contain only `continuation.tar`, `progress.json`
and `artifact_manifest.json`. Require `progress.complete == true` before
consuming terminal `nuisances.parquet`, `model_bundle.tar` and `metrics.json`.
Resume validates model/optimizer/scheduler/RNG/sampler, preprocessing,
calibration, fold/reference identities, code/environment/source/recipe hashes
and progress. New slices get new attempt roots and consecutive predecessor
links; they may not shorten tuning or restart from an approximate state.

## Offline verification

Run from this checkout on a compute node. The login node is for editing, Git,
GitHub and review-gate calls only. No tests, mutations, probes or acquisition
reads belong there. This command preserves the required inner test command:

```sh
srun --partition=standard --account=root --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=16G --time=00:15:00 \
  timeout 600s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python -m pytest -q tests
```

Before every review round repeat with
`/mnt/weka/home/hgoodarzi/oxyformer-swarm/envs/ci-cpu/bin/python`. Keep each
complete suite below ten minutes. Record failures and build-specific skips;
do not relax tolerances to hide CPU float32 differences. The focused files are
`tests/test_end_to_end.py` and `tests/test_plan_integrity.py`.

For the required mutation, remove a leaf from the expanded final collector's
`needs` in `test_final_collector_dependencies`, run that test, and require the
`collector omitted required leaf dependency` failure. Revert the injection,
rerun the test and complete suites, and record both results outside Git.

Review the complete delta from the coordinator-recorded base with the author
excluded. Commit/push the anchored attempt branch, open a PR into `dev` only
after REVIEW_PASS, and stop; the worker never merges or enables auto-merge.
Write the declared evidence into `$SWARM_UNIT_DIR`, with `completion.json`
last. That code receipt does not admit a scientific campaign.
