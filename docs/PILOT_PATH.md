# Paired diagnostic pilot

This path implements OD-6 for one owner-approved cell d and five sign-changing
pairs from collected m2-diag-A chains. It certifies no release criterion.

The owner adds a named `owner_decisions` block with purpose `diagnostic_pilot`,
`cells: {d: {reference_cap: 32, frozen_epochs: 60, grid_points: 2, seed_count: 1}}`,
`partitions`, `repetitions_per_scenario: 5`, `scenarios: [sign_changing]`,
`certifies_release_criteria: false`, `gate_se` (`unclustered_influence` or
`spatial_primary`), `paired_from: m2-diag-A`, and a recorded `wandb_waiver`.
This tree has no W&B hook: it refuses a pilot without that explicit owner waiver.
This code does not write approvals or grant waivers.

Call `real_frame.diagnostic_profile_tasks` with `campaign_mode` set to the block
name and `paired_manifest` set to the collected-chains manifest path. The builder
snapshots the declaration in the recipe; every consumer revalidates it against
the owner block. It emits profile, estimate and lock tasks without submission.
The estimator fits the approved study, retaining the existing nested recipe.

The lock selects the five lowest canonical source draw indices in the declared
scenario before inspecting results. Missing or corrupt selected chains stop it
with the chain name; no other draw substitutes. Every pin includes the source
chain, draw index, repetition ID, seed, scenario, source lock/tree, observation,
frame, truth, archive and result hashes. The source artifacts remain read only.
`draws_continue_from` applies only to the existing diagnostic campaign, never to
these pinned pairs. The pilot has its own campaign label and evaluation namespace.

Measured pilot profiles must pass the unchanged diagnostic sizing arithmetic:
at most four GPU-hours per leaf and the month-2 maximum leaves per chain, with
its approved safety factor and reserves. The public admission stage rechecks
sizing and regenerates every pinned draw with the saved source SCM, frame and
policy. Each chain's first leaf repeats this falsification check before fitting;
its actual generation inputs must agree with the source. Regenerated observation
and structural-truth content hashes must match. A failure stops the chain.
Leaves use the declared partition, eight CPUs, CUDA, and `--no-requeue`.

The collector uses the declared gate SE and publishes `paired_readout.json` and
`paired_readout.md`. Each draw retains its truth, pilot estimate and named SE
definitions, source estimate, replayed oracle cells, and both paired contrasts.
Summary contrasts use paired t intervals; five pairs are exploratory. Local
collectors describe their local pair; the first chain's collector receives all
five chains. Every numerical readout value has a basis label.

## Supported envelope and accepted limits

one approved cell per block; scenario sign_changing; five pinned draws from m2-diag-A; the
 current tree's generator and SCM (a different tree is a new campaign); preemptible and standard partitions with the owner's
 concurrency caps enforced by the coordinator, not by this code. Limits: cost is uncalibrated (109.1 H100-h per repetition projected;
 the screening factor 0.414 does not transfer); on that projection a cell-d chain does NOT fit leaves_per_chain_max 40 at safety factor
 1.25 (43 leaves at zero overhead, 55 with the source profile's overheads: attempt 1c66e36b2d62913e's bound probe), so whether the
 pilot runs is decided by the harness's sizing of the pilot's MEASURED profile after the owner commits the block, not by this unit;
 five pairs certify nothing; the collector's gate on five repetitions is descriptive.
