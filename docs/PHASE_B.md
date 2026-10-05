# Bounded campaign expansion

[phase_b_templates.json](../configs/execution/phase_b_templates.json) is an
admission checklist/catalogue, **not an executable plan**. It contains no
approved timing or allocation and must not be submitted as `--spec`.
The [execution blockers](EXECUTION.md#current-admission-status) and missing
campaign allocations currently block production dispatch.

The mandate allows at most 2,500 H100-hours, eight concurrent GPU units and six
CPU units on partition `standard`, account `root`. Each campaign instance has
at most **40 work leaves**, counting continuation slices; a collector is an
additional CPU unit. Every GPU leaf has at most **four GPU-hours**:
`gpus * ceil(wall_seconds / 60) / 60 <= 4`. Two GPUs for two hours consume the
whole cap. These are binding limits, not a prediction that a fit will finish.

## Admission procedure

1. Inventory the complete endpoint/target × variant × fold × seed × scenario ×
   repetition × deletion work before budgeting. Retain explicit IDs and output
   declarations. A0 has 15 outer-fold/seed jobs; its full schedule has 390
   nuisance fits and approximately 60 SSL fits. A1–A7 remain required in their
   registered scope; A8 is a separately approved months 2–3 extension. B0 is
   calibration only and D0 is diagnostic only.
2. Profile complete production procedures, including loading, setup, nested
   tuning, calibration, publication and verification. Record the code,
   environment, data/split/recipe hashes, device and wall/GPU time. Test or
   one-epoch smoke timings cannot admit production. CPU timings cannot justify
   GPU leaves: the merged nested runner currently has no GPU device interface.
3. Choose repetitions per leaf and slice counts from those measurements, with
   the declared safety factor. For N repetitions and batch size B, each
   scenario needs `ceil(N/B)` leaves. Sum over all scenarios. One thousand
   repetitions at 25 per leaf occupy all 40 leaves for one scenario; this is
   arithmetic, not an admitted batch size. If measured cost cannot fit, block
   or request prospective additional allocation/instances. Never drop draws,
   seeds, folds, comparators or tuning choices to satisfy the limit.
4. Resolve every input to a passed producer and every code prerequisite to a
   merged receipt. Build a concrete `execution.campaign` spec: `schema_version`,
   `id`, `kind`, `prerequisites`, `inputs`, `recipe_lock`, `work`, `collector`.
   Each work item has its own concrete `id`, `stage`, `parameters`, relative
   `outputs`, and nonempty `slices` of integer `gpus`/`wall_seconds`. IDs are at
   most 32 characters and unique after dependency-variable normalization.
5. Expand with `expand_campaign(spec, approvals)` and independently call
   `validate_plan(expansion, approvals)`. The latter rederives commands,
   resources, tasks, ownership and dependencies; modifying an emitted plan
   invalidates it. The generic planner checks structure and allocations, not
   scientific breadth, measured feasibility or stage adapter compatibility.
6. Bind final collector dependency roles to **all** concrete collector IDs
   across instances and preserve the complete expected request-hash manifest.
   Check scopes, attempt confinement, clone prefix, runtime, outputs, limits,
   normalized dependencies, acyclicity, predecessor ownership and fan-in.
   Review and ratify the expansion before dispatch; merely creating manifests
   does not submit jobs or authorize scientific work.

For an independently admitted concrete spec, the existing expansion-only CLI
is (run through Slurm, from the verified clone, with paths inside the attempt):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SWARM_UNIT_DIR/src/src" \
  /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python -B scripts/build_tasks.py \
  --spec "$SWARM_UNIT_DIR/admitted_spec.json" \
  --approvals "$SWARM_UNIT_DIR/src/configs/approvals.yaml" \
  --out "$SWARM_UNIT_DIR/expansion"
```

It writes `task_manifest.json` and `expanded_units.json`, and never submits.
An admission operator must supply the real reviewed spec; the command above
does not create approvals or measure runtime.

## Exact continuation and complete fan-in

`work.slices` expands consecutive steps with the same owner hash, parameters
and recipe, each depending on its predecessor's task/request/result and
declared outputs. The collector depends on every slice, including intermediate
slices. A passing checkpoint slice is not a completed fit. Inspect the terminal
progress and require all five folds × all three seeds before scoring.

The generic expander currently copies a common work-output list to every
slice and does not translate scheduler time into `nested_cv`'s `task.slice`
limits. Do not declare final-only nuisances on a partial slice, or assume
scheduler termination produces a portable checkpoint. Admission requires a
reviewed owning-unit bridge that sets safe execution limits and distinguishes
partial from terminal outputs. The direct continuation APIs are testable;
that does not make the generic multi-slice task an admitted production chain.

Long nested fits must preserve optimizer/scheduler/RNG/sampler, models,
preprocessing, calibration, fold/reference/source/config identities and frozen
prediction shards. Reducing 150 epochs, patience, the four tuning choices or
three inner folds is not continuation. The current coverage runner batches
**complete** repetitions; interruption records an incomplete repetition and
does not carry it to another leaf. Until exact cross-leaf repetition continuation
exists, admission must require that each full repetition fits or remain blocked.

Screening follows `campaign-lock` and cannot modify the recipe. The lock stage
binds production profiles, endpoint/frame artifacts, independently allocated
draws, fixed retry rules and screening/final expansions. Final leaves and final
collection depend on the screening collector for the same lock. All locked
batches must be present, with identical draws and identities; collector output
cannot silently certify a subset. Runtime overruns must be reported, not used
to replace the prospective budget with a retrospective one.

The final scientific report also requires all primary, coverage, anchor and
refit/audit collectors, the full expected task manifest and the gates in
[SCIENTIFIC_GATES.md](SCIENTIFIC_GATES.md). A single passed leaf or collector
cannot authorize release. If multiple instances are prospectively authorized,
the terminal collector must cover their complete union with no repeated seeds
or repeated repetition IDs; the current lock implementation does not itself
provide arbitrary multi-instance aggregate coverage admission.

## Budget and permission boundaries

The plan's initial 300-hour primary, 300-hour comparator and 1,700-hour
screening figures are planning ceilings. The 9,000-hour simulation discussion
is not an allocation under this 2,500-hour mandate. Full comparator/refit
breadth may exceed the initial budgets; admission must block or obtain
prospective authorization, without weakening the fixed science.

Final-coverage, anchor and refit-audit instances each require an explicit
`owner_decisions.campaign_allocations[instance_id]` with the matching kind and
sufficient GPU-hours, even for a CPU instance with zero GPU-hours. None is
implied by source permission or recipe lock. The current registry has no such
allocations. The owner alone supplies them. Final reporting additionally
requires scoped release approvals. No shared-root promotion is planned; it
requires the PI if later requested.
