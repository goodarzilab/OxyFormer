# Orchestrator mandate: OxyFormer v2, month-1 run

Confirmed by the owner, Hani Goodarzi, on 2026-10-04 ("all recommendations; I confirm the mandate; file the 44 issues.",
in the orchestrating Claude Code session, after the three enumerated sections below were shown to him verbatim).

- **Owner:** Hani Goodarzi.
- **Run:** plan `~/oxyformer-swarm/.swarm/plan.json` (`oxyformer-v2-m1`); Linear project (team Arc, created at filing);
  repo goodarzilab/OxyFormer, pull requests into `dev`.
- **Valid until:** the tract-release decision or 2026-11-15, whichever comes first.
- **Revocation:** the owner can revoke or alter this at any time, effective immediately.

## Priority
Build the OxyFormer v2 transformer engine and reach the week-4 tract feasibility verdict under
`docs/plan/OXYFORMER_V2_PLAN.md`. Infrastructure and gating get the minimum needed for scientifically valid results.
Anything beyond that is a follow-up issue, never a blocker.

## Granted, without asking
1. Run `swarm.py` advance, status, outbox and merge-recording commands, and keep the 5-minute advance loop alive.
2. Dispatch units of the ratified plan. Send bounded corrections or continuations to code agents. Start new code attempts
   seeded from preserved attempt or recovery refs.
3. Fix planning and declaration errors that leave the science unchanged: output declarations, prompt wording that
   contradicts the coordinator protocol, retry contracts, review bounds, and the runtime environment. Ratify each with
   `--accept-plan-change` and record the reason in `.swarm/PLAN.md`.
4. At `campaign-lock`, expand the registered campaign templates into leaf units within the Bounded limits (at most 4
   GPU-hours per leaf, at most 40 leaves per campaign instance) and file their issues in the project.
5. Merge a code unit's PR into `dev` when all of these hold: the PR's tests pass with the oxyformer environment and CI is
   green; hanig-review-gate returns REVIEW_PASS with the author excluded from the panel; the diff has been read; the scope
   check is clean. Use `merge_unit.py`.
6. Apply tracker intents to the run's Linear project under the evidence rules; create, update, comment on and close its
   issues, including follow-ups.
7. Build or repair `~/envs/oxyformer`.
8. Report hourly while units are live. Reports inform; they do not ask.
9. Open and merge non-unit PRs into `dev` that the run needs (setup, CI, amendments to this mandate), under the checks of
   item 5 where they apply.
10. Decide the disposition of review findings under Bounded item 6, without the owner.

## Bounded by
1. A total of 2,500 H100-hours for this run; at most 8 concurrent GPU units, 6 CPU units and 8 code agents; partition
   `standard` with account `root` only.
2. Writes only to `~/oxyformer-swarm`, the coordinator state directory, `~/envs/oxyformer`, and the repository through PRs
   into `dev`. `main`, the root data CSVs, `outputs/` and `report/` are read-only.
3. Code agents are Codex gpt-6-astra (thinking high) in `auto-review` mode.
4. At most 3 substantive review rounds per change, then the step-back committee for an exit criterion. The committee's
   converged criterion, or its tie-break ruling, is binding.
5. The scientific protocol in `docs/plan/OXYFORMER_V2_PLAN.md` and `configs/approvals.yaml` stays fixed: estimands, the
   shift policy, support and eligibility rules, endpoints, multiplicity, gates and thresholds, folds, comparators and the
   public-data-only access rule.
6. Finding disposition:
   - A reproduced defect that threatens scientific validity is always fixed and never waived. This covers outcome or label
     leakage across folds, fold-external preprocessing or pretraining, location, terrain or satellite inputs reaching a
     nuisance model, a wrong pushforward, score, targeting or variance computation, a wrong exposure computation, and any
     plan violation.
   - Hardening against malformed or adversarial inputs that honest producers in this pipeline cannot emit is filed as a
     follow-up issue and does not block.
   - An allegation that does not reproduce is recorded and does not block.

## Always stop and ask
1. Any change to the science plan, `docs/plan/OXYFORMER_V2_PLAN.md` or `configs/approvals.yaml`.
2. A failed gate: the tract support gate, a coverage gate, or a physiological or birth anchor. Report it and do not
   continue past it.
3. Promotion to any shared path.
4. Exceeding the budget or concurrency caps, a leaf estimated above 4 GPU-hours, any other partition or account, or
   launching the final-coverage, anchor or refit-audit campaigns (each needs its own approved allocation).
5. A step-back committee that routes a question to the owner because it would change the science, or a scientific-validity
   defect that cannot be fixed within the round bound.
6. Anything outward-facing beyond PRs and merges into `dev` of goodarzilab/OxyFormer and issues inside the run's Linear
   project. This includes merging into `main`, creating repos, projects or public artifacts, force-pushing, deleting
   branches that hold unmerged work, and any data access that needs an application.
7. Starting the months 2-6 follow-on work.

## Narrow mode (before confirmation)
Read-only status and diagnosis; keep the advance loop alive; drain start and close intents that carry evidence; report.
No merges, no plan changes, and no dispatch beyond what `advance` does on its own.

## Concurrence party
The step-back committee of hanig-review-gate: two members on contrasting providers, the author excluded. Hani Goodarzi
remains the owner and can take any decision back.
