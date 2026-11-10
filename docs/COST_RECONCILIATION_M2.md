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
Numeric quantities preserve all source bases, including projected and uncalibrated.
Copied counts add `[derived]`; computed quantities use `[derived]` or `[extrapolated]`.
Identifiers and hashes are strings, not numerical quantities.

The supported terminal archives retain OOFNuisances blobs and coverage state.
They discard controller task timing history. The module therefore uses the
contract's fallback: real operation counts/durations, excess shares and term
correction factors are null. It never divides cumulative fold time by fit
counts. Inert extra archive members are ignored. A referenced part of an unexpected
type refuses with its recorded type in the inspection diagnostic;
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

Leaf phases must conserve wall time. Leaf compute includes wrapper work and
already-complete calls; repetition compute encloses draw execution only.
Both scopes and their difference are reported; the repetition timer must
match the archive.

Stored cell hours may be rounded: formula consistency uses relative tolerance
1e-6. JSON preserves source calibration verbatim; Markdown summarizes it beside the
descriptive factor. Calibration and transfer limits still apply.

Output destinations must be distinct. Aliases between outputs, or from outputs
to source artifacts, refuse before any write.
