# Declared gate standard errors (OD-5)

Repetition estimates retain `value`, `se`, and `spatial_se`. The legacy `se` is
still the registered primary-bandwidth spatial standard error, with variance
target `geographic_process`. Each estimate also carries `se_definitions`:

| Definition | Variance target | Computation |
| --- | --- | --- |
| `spatial_<bandwidth>_km` | `geographic_process` | Existing spatial sensitivity covariance at every registered bandwidth |
| `spatial_primary` | `geographic_process` | Registered primary spatial bandwidth |
| `county_cluster` | `geographic_process` | Existing county cluster covariance, including its finite-cluster correction |
| `unclustered_influence` | `fixed_frame_repetition` | Square root of n/(n-1) times the sum of squared normalized influence contributions |
| `residual_frame_conditional` | `frame_conditional_residual` | Square root of the sum of squared weighted, seed-averaged residual corrections |

Every entry contains `se`, `variance_target`, and a one-sentence `assumptions`
statement. The residual calculation uses the fitted one-step means or the
seed-specific targeted CV-TMLE means, following the attribution audit.

`coverage.summarize(..., gate_se=None)` preserves legacy output, including its
schema. Explicit `gate_se="unclustered_influence"` or `"spatial_primary"` uses
that definition for coverage, null rejection bounds, mean SE, SE/SD, and their
dependent pass flags. It adds top-level `gate_se` and `gate_variance_target`,
and per-method `reported_se_mean` and `reported_se_variance_target` for the
registered SE. Missing, nonfinite, negative, or mislabelled selected definitions
raise `ContractError` identifying the method and definition. Failed repetitions
remain in unconditional denominators. The minimum repetition count stays fixed.

An owner campaign block may optionally name `gate_se` only for
`purpose: diagnostic_pilot`. Its declaration must contain exactly the same
selection. Six-field blocks retain their original behavior. This validator
admission does not add a new campaign executor. Screening and diagnostic
collectors retain their sealed calls without a gate selection.

The synthetic golden fixtures were captured from launch-base code before
editing it; the full collector golden was captured using saved copies of the
launch-base coverage and campaign modules. They include failed and rejected
repetitions, and are compared as canonical JSON bytes. No training is performed
by this unit beyond synthetic unit-test fits.

Supported envelope and accepted limits: both estimators (one_step, cv_tmle); the registered spatial bandwidths of the recipe; records
 produced by the current leaf code. Limits: the unclustered influence SE is a fixed-frame descriptive estimator, not a certification;
 five-repetition pilots never certify coverage (MIN_REPETITIONS unchanged); no claim about which SE is "correct" for real data.
