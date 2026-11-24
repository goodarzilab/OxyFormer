# Q screen (exploratory)

This CPU-only harness evaluates saved simulation predictions. It fits no model,
uses truth only for evaluation, and certifies no release criterion. Only Q changes;
ratios, targets, policy, targeting and SE routines remain those of the saved chain.
The supported endpoint is continuous with an identity link, one state, one seed,
and one saved nuisance part per fold/seed. Use at most 25 chains per screen.

## Prediction schema

One JSON file per candidate per terminal chain, including the matched baseline:

```json
{
  "schema": "oxyformer.q_parts.v1",
  "chain_dir": "/absolute/resolved/terminal/leaf",
  "frame_hash": "the state's binding frame_hash",
  "variant": "A4-Q",
  "study": {"frozen_epochs": 60},
  "code_commit": "producer commit",
  "parts": [
    {"fold": 0, "seed": 1103, "original_ids": ["target-id"],
     "mu_a": [50.0], "mu_d": [50.1]}
  ]
}
```

These are exactly the permitted keys at each level. Each saved part needs exactly
one candidate part; part order may differ, row order must match exactly. Fold and
seed are integers. Prediction arrays must have the saved length and finite numeric
values. Ratios cannot be supplied. Refusals identify the chain, fold and field.
Variant, study and producer commit must agree across chains for each named cell.
A SHA-256 binds the non-Q fields (including saved lineage/spec), policy, target
split and bound data before and after each splice. Candidate file hashes and
producer metadata are also recorded. Loading verifies archive and dependency
hashes with the same loader as attribution_audit; its public load still accepts
only diagnostic mode. The screen additionally accepts m2_pilot_d, and refuses
other modes. Loading writes nothing into the saved leaves.

## Command

Write a manifest JSON list, with all paths relative to the working directory or
absolute (chain_dir inside each prediction file must always be resolved/absolute):

```json
[
  {"chain_dir": "/saved/chain1", "baseline": "/q/chain1-A0.json",
   "candidates": {"A4-Q": "/q/chain1-A4.json"}},
  {"chain_dir": "/saved/chain2", "baseline": "/q/chain2-A0.json",
   "candidates": {"A4-Q": "/q/chain2-A4.json"}}
]
```

Run on a Slurm CPU allocation, excluding idle nodes under local scheduling rules:

```sh
PYTHONPATH=src python -m oxyformer.validation.q_screen manifest.json \
  --output-dir screen-output --interval-level 98.75 --bias-floor 0.003 \
  --bias-ratio 0.5 --contrast-reduction 0.25 --iid-coverage-min 16
```

Use separate screens for the primary diagnostic chains and supplementary pilot
chains. Repetition IDs, chain paths and named cells must pair uniquely. The matched
baseline is an explicit prediction file, never silently the saved fitted Q.
`fitted`, `oracle_Q` and `baseline` are reserved cell names. All thresholds are
echoed in the output. `q_screen.json` and `q_screen.md` contain the same complete
report, labelled exploratory. Invalid input raises an error before writing a
report. The destination should be a fresh directory for each invocation.

## Readouts and decisions

Every cell reports one-step and CV-TMLE estimates and replay's spatial (50/100/200
km), county, iid and residual frame-conditional SEs. Error and iid/primary-spatial
95% normal-interval coverage use truth recomputed by the simulation generator.
The primary bandwidth is taken from the saved inference recipe. Saved estimates
and spatial SEs must replay to 1e-12 before candidate results are computed.

Factual OOF R² and RMSE are computed on targets. The weighted factual MSE is
`sum(w * residual²) / sum(w)` with `w = q² (r_a - 1)²`; it is null when all these
weights vanish. `q` is the saved target mass normalized over targets. Oracle
comparisons report factual mean RMSE and contrast bias, RMSE, Pearson correlation,
slope of predicted contrast on true contrast, and sign agreement for movers,
all targets and nonmovers. These descriptive fit metrics use equal target rows.
Undefined correlations/slopes, zero-variance R² and empty-subset metrics are null.
Sign agreement includes zeros. The remainder is `sum(q*(r0_a-r_a)*(mu_a-mu0_a))`,
a descriptive quantity, not an identity.

Plug-in and correction diagnostics use `mu_d-mu_a` and
`(r_a-1)*(Y-mu_a)`, both before targeting (one_step) and after targeting (cv_tmle).
The overall values use normalized saved masses. Each fold value is conditional
on that fold, with its target mass reported so contributions can be reconstructed.
CV-TMLE's returned estimate is computed by the original targeting code; finite
numerical targeting residuals can separate it slightly from a reconstructed score.

For CV-TMLE, `B_0` and `B_c` are mean errors. `s = sign(-B_0)` is fixed per screen;
`d_i = s*(estimate_c,i - estimate_0,i)`. Positive improvement points toward truth.
The two-sided paired t interval defaults to 98.75%, accounting for up to four
candidates; use 95% for the single-winner confirmation. Empirical SD, mean SEs and
their ratios are descriptive. Iid and spatial coverage counts have exact
Clopper-Pearson 95% bounds. Spatial coverage never gates a decision.

The defaults and their command flags are:

| Flag | Default | Passing rule |
|---|---:|---|
| `--interval-level` | 98.75 | paired t lower bound strictly positive |
| `--bias-floor` | 0.003 | absolute baseline mean error at least floor |
| `--bias-ratio` | 0.5 | absolute candidate bias / absolute baseline bias at most ratio |
| `--contrast-reduction` | 0.25 | `1 - mean(mover RMSE_c)/mean(mover RMSE_0)` at least threshold |
| `--iid-coverage-min` | 16 | iid coverage count at least this explicit count |

`floor_not_met` takes precedence when there is insufficient baseline attenuation;
otherwise `pass` requires every criterion above, and `fail` names every failed
criterion. When the baseline mover RMSE is zero, or a mover subset is empty, the relative
reduction is undefined (null) and cannot satisfy its passing criterion. The
attenuation floor still takes precedence. A singleton screen has a null paired
interval and empirical SD and cannot satisfy the paired-interval criterion. The supplementary
`gross_futility` flag is true only when mean paired improvement is nonpositive
and mover RMSE is worse. These flags do not initiate training or later campaigns.
Use `--bias-floor 0.002 --interval-level 95 --iid-coverage-min 10` for a
12-chain d1 confirmation. Counts never scale automatically with chain count.

A paired t interval over 20 chains assumes approximately normal paired differences.
Twenty chains cannot establish nominal coverage: 20/20 has an exact 95% lower
bound near 0.83. All results are exploratory.
