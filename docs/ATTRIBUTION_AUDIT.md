# Saved-repetition attribution audit

Run on CPU against terminal diagnostic attempts in place:

```sh
PYTHONPATH=src python -m oxyformer.validation.attribution_audit ATTEMPT --output-dir AUDIT
```

The audit reads archived observations, truth configuration and OOF nuisances;
it fits nothing and changes no estimator or gate. Every reported number has a
basis. A single repetition cannot provide paired intervals or SE/SD ratios.
The A3 rules retain “inconclusive (partial recovery)” when neither branch holds.

Fixed-exposure training-only rows (zero estimand weight and empty frozen support)
have no oracle kernel. Each repetition reports their excluded count.

Dependencies are authenticated by the saved request's content hashes. Absolute
pointers remain authoritative. When an original file is missing, a pointer
inside the request's original output directory can be resolved under the moved
attempt directory and is still hash checked. External dependencies are not
discovered by basename or guessed from the host directory layout. A moved
attempt without its external dependency is reported unreadable, including the
missing path; restore that authenticated file or audit the original attempt.
Other readable attempts remain reportable. These are diagnostic outputs only.
