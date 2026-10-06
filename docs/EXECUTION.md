# Execution and verification

The [frozen science plan](plan/OXYFORMER_V2_PLAN.md), [owner mandate](orchestrator-mandate.md)
and read-only [approvals](../configs/approvals.yaml) govern this run. The merged
[interfaces](INTERFACES.md) define artifact names. A passing unit test is not a
clinical release, a compute allocation, or permission to access a source.

## Current admission status

Production campaign admission is **blocked** at the audited merge base
`00c340adcf961068257d3f17749f2623683d533f`. The runner's owner-approval YAML date
serialization and reporting task/config handoff have been corrected upstream.
They must not be carried forward as current defects. Reporting accepts the
registered hyphenated stage names and binds the clone's reporting config and
owner registry; missing scientific evidence or scoped approvals still blocks it.

The remaining boundaries below must be resolved before admitting an executable
campaign. The integration fixture's direct scientific APIs do not certify the
registered runner route. Reproductions use synthetic inputs only; their
receipts, script and exact results belong in the attempt's `SUMMARY.md` and
`pytest.log`, outside Git.

| Boundary | Current contract and reproduction |
| --- | --- |
| Runner → design | The unchanged `tract-support-gate` registration reaches `design.gate.run_stage` with an execution envelope. It returns `invalid design stage configuration`: design requires `config.schema_version == 1` and `config.stage == request.stage == "tract_design"`. The direct design control passes; translating only the name still fails. |
| Design input roles and config | The registered needs supply raw acquisition and atlas files. Design requires canonical `DataManifest`, `CovariateView`, `GeographyTable`, `CollectedAtlas`, `EntityGraph` and a hash-bound owner file in `task.dependencies`. Isolated probes return `task must bind approvals...`, `task dependency is not hash-bound by StageRequest`, and, after adding just the direct schema/name, `missing or contradictory fixed approval: support_design_fraction`. No approval is actually missing from the owner registry; the adapter does not read the runner envelope. |
| Generic expansion → nested slices | `expand_campaign` emits scheduler resources but no `task.slice` execution limit. A 120-second leaf therefore reaches a stage whose default is 14,400 seconds with a 120-second checkpoint margin. Extra per-slice limits/outputs in the spec are ignored; all slices share `work.outputs`. Declaring terminal nuisances on a partial slice produces `stage omitted declared outputs`. Common checkpoint-only outputs require a separate terminal-completeness check. |
| Registered comparisons → nested stage | A5, A6, A7, F0 and F1 each return `blocked`: the density-ratio checkpoint interface cannot represent their signed corrections or foundation state. Their direct model interfaces do not provide an exact campaign continuation path. |
| Generic expansion → report collector | The generated `audit-collect` task has `expected_leaves`, but lacks `bundle`, `manifest`, `receipts` and `approvals`. The registered reporting worker returns `invalid reporting task fields`. A separately constructed complete reporting task is supported; the generic expander does not construct it. |
| Coverage collector → reporting evidence | `validation.campaign` publishes `result.json`, `gate.json` and an artifact manifest. Reporting requires canonical artifacts of kind `coverage_scenario`. A passing synthetic final collection still yields `coverage task has no CoverageScenario artifact` at that boundary. Copying metric values into a report cannot replace verified producer evidence. |

Source inspection also finds no registered producer of the typed design inputs
or `PreparedEndpoint`, and no registered score/targeting/covariance stage that
assembles a `ReportBundle`. These APIs remain callable directly; the synthetic
fixture explicitly assembles their records. That fixture is not a reviewed
production adapter. The nested and coverage runners are CPU-only; coverage
batches complete repetitions and has no cross-leaf interrupted-repetition
continuation. Neither limitation authorizes shortened tuning or assumed GPU
performance.

Production admission requires the design implementation/registration to accept
the merged runner contract and bind the real typed inputs. Direct API tests do
not establish an executable coordinator path. Continue offline verification of
the scientific APIs and plan-integrity checks while retaining these production
blocks. Reassess admission after merged corrections; the early atlas/support
path must remain independent of this documentation/integration unit.

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

Cloning `dev` selects the launch revision; it does not override the locked
scientific identity. Before invoking a locked stage, `execution.runner` calls
`verify_recipe` and requires the clone's `tracked-science-v2` fingerprint to
match the recipe. It also checks the scientific fingerprints of locked upstream
attempts, including transitive dependencies. The fingerprint covers tracked
files except `*.md` outside `src/`, `scripts/` and `docs/plan/`. Each clone's
full commit is still recorded and bound to its stage request.

If an ordinary merge advances `dev` with changed source, config or frozen-plan
content, a later leaf refuses with `recipe scientific code/config drift` before
producing scientific output. Do not combine results from different scientific
fingerprints or rewrite the old lock. A changed scientific recipe requires a
new prospective lock, timing and authorizations. A documentation-only merge
can preserve the scientific fingerprint and pass the generic lock check;
portable nested continuation additionally binds its full request code identity
and may refuse that change. A clone prefix alone is not evidence that either
identity check passed.

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

The commands below use this lab's installed interpreters and Slurm settings.
On another host, provision and verify the equivalent environments and use its
site-approved scheduler settings; these absolute interpreter paths are local
to this run. Run from this checkout on a compute node. The login node is for
editing, Git, GitHub and review-gate calls only. No tests, mutations, probes or acquisition
reads belong there. This command preserves the required inner test command:

```sh
srun --partition=standard --account=root --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=32G --time=01:30:00 \
  timeout 3600s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python -B -m pytest -q tests
```

Before every review round repeat the complete suite with
`/mnt/weka/home/hgoodarzi/oxyformer-swarm/envs/ci-cpu/bin/python`. The owner
replaced the obsolete ten-minute complete-suite limit with a 3,600-second
timeout; the measured eight-CPU full runs in this attempt took approximately
32–33 minutes. Record failures and build-specific skips; do not relax tolerances to hide CPU float32 differences.
Also run `tests/test_end_to_end.py tests/test_plan_integrity.py` **without `-B`**
in both environments before review, matching CI's ordinary interpreter mode.
Do not make tests depend on bytecode flags. The integration additions must add
well under a minute; shortened synthetic timing cannot admit production work.

For the required mutation, use an isolated copy of
`test_final_coverage_collector_dependencies` with its repository root preserved.
Remove a leaf from the expanded final collector's `needs`, run that test, and require the
`collector omitted required leaf dependency` failure. Revert the injection,
rerun the test and complete suites, and record both results outside Git.

Review the complete delta from the coordinator-recorded base with the author
excluded. Commit/push the anchored attempt branch, open a PR into `dev` only
after REVIEW_PASS, and stop; the worker never merges or enables auto-merge.
Write the declared evidence into `$SWARM_UNIT_DIR`, with `completion.json`
last. That code receipt does not admit a scientific campaign.
