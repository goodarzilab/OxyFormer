# Screening cost reconciliation

Run on a CPU compute node with the pinned environment:

```sh
PYTHONPATH=src python -m oxyformer.validation.cost_reconcile \
  --manifest /path/to/collected_chains.json \
  --report /path/to/recipe_cost_m2.json --out /path/to/evidence
```

The manifest has the coordinator's `chains` list with `chain`,
`terminal_leaf_attempt_dir` and `terminal_leaf_state`. Relative attempt paths
resolve against the manifest directory. Inputs are read only; no fit runs.
Outputs contain input hashes, every retained leaf's compute and overhead,
repetition totals, the harness formula, receipt, overhead and candidate causes.
Numeric quantities carry `[measured]`, `[derived]` or `[extrapolated]` labels.
Identifiers and hashes are strings, not numerical quantities.

The supported terminal archives retain OOFNuisances blobs and coverage state.
They discard controller task timing history. The module therefore uses the
contract's fallback: real operation counts/durations, excess shares and term
correction factors are null. It never divides cumulative fold time by fit
counts. An unexpected controller archive refuses with an inspection diagnostic;
its timing schema needs review before any decomposition can be published.

The corrected screening projection uses the mean of complete repetitions;
the observed range, sample SD and standard error describe its uncertainty.
The aggregate correction factor is descriptive and does not transfer to other
cells. The sole scenario adjustment is the month-one sign-changing/null-effect
profile ratio; the adjusted receipt and chain mean are extrapolations with
unmeasured scenario-transfer uncertainty. No causal attribution of the excess
is established by matching totals. No final cell or repetition count is chosen.

Supported envelope and accepted limits: m2-diag-A artifacts on tree 626d4dd and the recipe-cost-m2-report-r2 output; CPU only; the
scenario differs between the receipt (null_effect) and the chains (sign_changing): report the month-1 profile ratio 31.8/29.0 as the
only available scenario adjustment and label it [derived]. Limits: no new GPU timing; no claim about hardware other than the H100s
the chains ran on.

Leaf wall time is checked against the sum of its recorded phases. The sum of
leaf compute includes wrapper work, synchronisation and already-complete calls;
the profile repetition timer encloses only draw execution. Both totals and their
difference are reported separately, with the profile checked against the
archived cumulative repetition timer. No equality between these scopes is assumed.
