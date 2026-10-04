# OxyFormer v2 causal AI implementation plan

> Frozen science plan for the OxyFormer v2 run. Drafted by gpt-6-astra from 26 research subagent reports and reviewed by Claude; adopted by Hani Goodarzi on 2026-10-04. PI decisions recorded after drafting: the `src/oxyformer/` package is approved, and TabPFN (all versions) is admitted alongside TabICLv2 because Arc is a nonprofit (section 8.4 amended to match). Owner approvals of fixed parameters live in `configs/approvals.yaml`. Changes to this file require the owner (see `docs/orchestrator-mandate.md`).


**Decision date: October 4, 2026**

**Status:** implementation specification, not a report of completed training or validated causal estimates. Architecture choices and numerical gates below are **proposed**. Dataset payloads, cluster compatibility, and measured runtimes remain **UNVERIFIED** until the corresponding tests pass.

---

## 1. Executive summary

**OxyFormer v2 will be a hierarchical, treatment-query transformer system for estimating support-qualified changes in disease outcomes under continuous exposure-shift policies.** Transformers—not ridge regression on embeddings—will estimate both the outcome response and the exposure-distribution correction.

The primary system combines:

1. Fold-nested masked self-supervision on approved ACS covariates.
2. Feature-token encoding and training-reference county-context attention.
3. A continuous-treatment-query cross-attention outcome model.
4. A separate transformer classifier estimating the policy-induced density ratio.
5. Cross-fitted one-step estimation, confirmed with cross-validated targeted estimation.
6. Geographic/survey uncertainty, known-truth calibration, and the existing falsification ladder.

The first fully gated application is **USALEEP within-county tract life expectancy**, targeting a supported **2-mmHg increase in inspired-oxygen deficit**. Approximately 72,000 candidate ACS tracts provide a substantial covariate-pretraining pool, even if only a few thousand tracts have fully observed mortality inputs. Actual eligible counts must be measured. USALEEP explicitly distinguishes observed, predicted, and mixed mortality inputs. ([cdc.gov](https://www.cdc.gov/nchs/nvss/usaleep/usaleep.html))

The transformer’s scientific contribution is flexible estimation, efficient use of unlabeled covariates, and controlled transfer across related tasks. It does **not** establish exchangeability, recover lifetime residence, or isolate oxygen from other characteristics of altitude.

The program will retain separate residential, genetic, and randomized-drug evidence streams. It will not interpret a geographic association as an oxygen-treatment recommendation.

**First delivery:** a reproducible transformer pipeline, an audited tract-support assessment, and calibrated estimates—or an explicit finding that the proposed contrast is not identifiable or estimable in the available data.

---

## 2. System architecture

### Primary architecture: **OxyFormer v2 — Hierarchical Treatment-Query Transformer**

This is a new composition, drawing on feature tokenization, Set Transformer pooling, and TransTEE-style treatment queries. None of those components supplies the complete proposed causal estimator; OxyFormer must implement and validate their integration. ([github.com](https://github.com/hlzhang109/TransTEE))

```text
PUBLIC SOURCE FILES + LICENCES + IMMUTABLE PROVENANCE
             |
             +---------------- EXPOSURE PHYSICS SERVICE ----------------+
             |  DEM + 2010 block populations + placement scenarios     |
             |  fixed physical transformation and population weighting |
             |                  -> A in mmHg + measurement diagnostics  |
             |                  NO outcomes, NO learned outcome weights |
             |                                                          |
             +--------------------- DATA CONTRACTS <--------------------+
                endpoint, population, approved X, county/group,
                outcome scale, weights, source lineage, policy
                                  |
                        FROZEN DESIGN/SUPPORT STAGE
                        within-county comparison rules
                                  |
                         OUTER GEOGRAPHIC SPLITS
                                  |
                   training-only ACS covariate records
                                  |
                    MASKED FEATURE-TOKEN PRETRAINING
                         no A, Y, disease auxiliaries
                                  |
                      copy initialization; separate weights
                           /                         \
                          v                           v
             OUTCOME TRANSFORMER             ORIGIN TRANSFORMER
             feature encoder                 feature encoder
             county PMA context              county PMA context
             treatment-query attention       treatment-query attention
             raw-X bypass                    raw-X bypass
             family-specific likelihood      calibrated binary classifier
                |             |                        |
             mu(A,X)       mu(d(A,X),X)                r_d(A,X)
                           \             |             /
                            CROSS-FITTED MTP SCORE
                                      |
                         ONE-STEP PRIMARY ESTIMATE
                         CV-TMLE CONFIRMATION
                                      |
                      CLUSTER / SPATIAL / SURVEY INFERENCE
                                      |
                    SUPPORT + CALIBRATION + FALSIFICATION REPORT

Separate, access-controlled diagnostic path:
coordinates / terrain / SatCLIP / GeoCLIP / AlphaEarth -> red-team probes
NEVER -> primary nuisance-model inputs

Comparators:
transformer Riesz correction; TabICLv2 nuisances; fixed architecture ablations
Conventional learners: calibration benchmarks only
```

### Component contract

All dimensions below are starting specifications, not optimized findings.

| Component | Role and inputs | Outputs | Training objective | Starting size | Status |
|---|---|---|---|---|---|
| Exposure service | DEM, population, boundaries | Deficit \(A\), distributions, uncertainty flags | None; deterministic physics | CPU/geospatial | Primary |
| Feature tokenizer/encoder | Approved \(X\), missingness | Feature states \(H\), pooled \(Z\) | Masked reconstruction, then nuisance loss | Width 64; 3 layers; 4 heads | Primary |
| County context | Training-reference tract \(X\) only | Four context tokens per county | Trained through the relevant nuisance objective | PMA, four seeds | Primary |
| Outcome transformer | \(A,X,Z,\) county context; raw-\(X\) bypass | \(\mu(A,X)\), \(\mu(d(A,X),X)\) | Endpoint-specific proper loss | One treatment cross-attention block | Primary |
| Origin transformer | Original/shifted \(A\), same approved \(X\) and context | Class logit and \(r_d\) | Weighted binary cross-entropy | Independent copy of backbone | Primary |
| Riesz transformer | Same admissible inputs | Signed \(\alpha_d=r_d-1\) | Functional-specific Riesz loss | Independent backbone | Sensitivity |
| One-step estimator | Held-out means, ratio, outcomes, target weights | MTP contrast, influence contributions | No neural optimization | FP64 arithmetic | Primary |
| CV-TMLE | Frozen held-out nuisances | Targeted contrast and influence contributions | Low-dimensional fluctuation | One scalar per endpoint/seed | Confirmation |
| TabICLv2 | Fold-specific training context | Alternative outcome/ratio predictions | Frozen in-context inference | Pinned checkpoint | Foundation comparator |
| Country/task adapters | Approved shared/private features | Task-specific nuisance functions | Balanced multitask objectives | Bottleneck 16 | Months 2–3 extension |

**Architectural invariant:** both nuisance heads receive the complete approved \(X\) alongside learned representations. A low-dimensional embedding is never assumed to preserve sufficient adjustment information.

---

## 3. Model specifications

### 3.1 Exposure and feature separation

The exposure service constructs:

\[
A_t=
\frac{\sum_{b,u} n_{bu}
\left[P_{IO_2,\mathrm{ref}}-P_{IO_2}(z_{bu})\right]}
{\sum_{b,u}n_{bu}}.
\]

Transform location-level elevation into pressure **before** population aggregation. Keep fixed population weights, physical constants, DEM versions, and allocation scenarios in the exposure manifest. USGS provides the ground-elevation products; individual tile provenance still requires inspection. ([usgs.gov](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services))

No outcome-trained attention weight may alter \(A_t\). No gradients cross into the exposure service.

The nuisance input registry prohibits:

- Elevation, alternative pressure transforms, or exposure-model embeddings in \(X\).
- COPD/poor-health prevalence and other unapproved downstream health measurements.
- Post-outcome PLACES variables presented as historical confounders.
- Coordinates, terrain descriptors, satellite/location embeddings, and learned tract-ID embeddings.
- Outcome availability, USALEEP SEs, or mortality-derived flags as ordinary predictors.

County membership is an explicitly approved **coarse comparison stratum**, implemented through context routing and scalar offsets—not a general geographic embedding.

### 3.2 Tokenization and masked pretraining

Start with **30–60 endpoint-approved concepts**, selected through causal and measurement review rather than predictive importance.

For numerical feature \(j\):

\[
e_{ij}=e_j^{\mathrm{feature}}
+W_j\widetilde X_{ij}
+e_j^{\mathrm{missing}}M_{ij}.
\]

Use categorical embeddings for observed categories, with explicit missing/unknown states. Real missingness, padding, and artificial masking are distinct.

The encoder uses:

- Width \(d=64\), four attention heads.
- Three pre-LayerNorm transformer blocks.
- Feed-forward width 128; GELU activation.
- One CLS token.
- No positional encoding based on feature order or geography.
- Batch size 256 initially.
- A **one-million-parameter cap per nuisance network**, checked programmatically.

Feature order may change only with its feature identifiers; predictions must remain unchanged under a joint permutation.

**SSL objective:** reconstruct masked approved covariates. Mask semantic families together where totals/complements would otherwise reveal the answer. Use Huber reconstruction for standardized numerical variables and cross-entropy for categorical variables. Give feature families comparable total weight.

Starting pretraining settings: mask rate 0.30, AdamW learning rate \(3\times10^{-4}\), weight decay \(10^{-4}\), dropout 0.10, maximum 30 epochs, patience five.

The potential pretraining pool is the available ACS tract frame—not 72,000 outcome-bearing tracts. Blocks remain exposure units; copying tract features onto blocks does not create millions of independent socioeconomic observations. ACS’s documented tables supply the actual covariate definitions. ([api.census.gov](https://api.census.gov/data/2010/acs/acs5/groups.html))

### 3.3 County context

For county \(c\), construct a fold-specific reference set \(\mathcal R_{c,-k}\) containing only training-permitted tract covariates:

\[
K_{c,-k}
=
\operatorname{PMA}_{4}
\left(
\{Z_j:j\in\mathcal R_{c,-k}\}
\right)
\in\mathbb R^{4\times64}.
\]

Pooling by attention supplies permutation-invariant fixed-size context from a set. ([proceedings.mlr.press](https://proceedings.mlr.press/v97/lee19d.html))

Rules:

- References contain **no exposure values, outcomes, residuals, or fitted county effects**.
- Exclude the training query itself from its reference set.
- Held-out queries cannot update reference states.
- Cache keys include fold, reference IDs, preprocessing hash, and checkpoint hash.
- Reference selection is outcome-blind.
- All allowable references are used initially; a fixed, outcome-blind cap is introduced only if profiling requires it.

These context tokens summarize approved county-level covariate information. They do not constitute a spatial-confounding cure.

### 3.4 Treatment-query outcome head

Encode continuous \(a\) using its standardized scalar value plus six continuous spline-basis values. Knots are frozen from the design stage. **No dose categorization or monotonic protective constraint.**

For \(R=2\) queries—observed and shifted:

```text
X tokens                     [B, P+1, 64]
county context               [B, 4, 64]
combined keys/values         [B, P+5, 64]

A_query                      [B, 2, 1]
continuous treatment basis   [B, 2, 7]
treatment queries            [B, 2, 64]
cross-attention output       [B, 2, 64]

readout:
[query output, CLS, approved raw-X bypass, treatment basis]
                            -> 64 -> 32 -> endpoint parameters
```

There is **no query–query attention**: prediction at \(a\) must not depend on which other doses happen to be queried.

For continuous outcomes:

\[
\widehat\mu(a,x,c)
=
\widehat\gamma_{c,-k}
+
f_\theta(a,x,K_{c,-k}).
\]

Estimate \(\gamma_c\) using training labels only. Under squared loss, profile it as the weighted training mean of \(Y-f_\theta\), updating during fitting. For other likelihoods, optimize the corresponding scalar link-scale intercept jointly.

County offsets cancel in the plug-in within-county contrast, but still matter for residual correction and prediction. They are part of the transformer nuisance model—not a separate conventional estimation engine.

### 3.5 Origin classifier and Riesz comparator

The origin classifier has **separate trainable parameters** from the outcome model. It may start from the same fold-specific SSL checkpoint.

Create paired records:

\[
(A_i,X_i,C=0),\qquad(d(A_i,X_i),X_i,C=1).
\]

Both copies inherit the original target weight and remain together through every split. County context and a scalar county classifier offset permit county-specific exposure-distribution differences. The readout retains raw \(X\).

Calibrate logits using **inner out-of-fold predictions from outer-training data**, never outer-held-out records. Use a prespecified affine-logit calibration; audit its transfer to the final refitted classifier.

The Riesz comparator replaces classification with:

\[
\mathcal L_R(v)=
E_T\!\left[
v(A,X)^2
-2\{v(d(A,X),X)-v(A,X)\}
\right].
\]

Its output is signed. Do not constrain it to be positive. This is our specialization of neural Riesz learning to the policy contrast, not a claim that published RieszNet already implements OxyFormer. ([proceedings.mlr.press](https://proceedings.mlr.press/v162/chernozhukov22a.html))

### 3.6 Endpoint likelihoods

| Endpoint | Primary outcome head/loss | Estimation scale |
|---|---|---|
| USALEEP life expectancy | Identity mean; squared loss | Years |
| Published standardized cancer rate | Identity mean; squared loss; supplied-SE measurement sensitivity separately | Cases/deaths per 100,000 |
| Ecuador birth weight | Identity mean; squared loss | Grams |
| Colombia LBW | Bernoulli head; categorical bands secondary | Risk difference |
| ENDES raw Hb | Survey-weighted identity mean | Verified Hb units |
| Municipal mortality | Log-rate head, population offset, normalized Poisson loss | Standardized rate difference |

For mortality cell \(i\), \(Y_i^*=D_i/N_i\), \(\lambda_i=\exp f_\theta\), and target mass \(w_i\):

\[
\mathcal L_\mu
=
\sum_i \frac{w_i}{N_i}
\ell_{\mathrm{Poisson}}(D_i,N_i\lambda_i).
\]

This makes the objective compatible with the declared rate-scale target. Population weights, age-standardization weights, and likelihood exposure are not interchangeable. The endpoint specification must define them separately.

**A death-only file is not a population cohort.** Build complete age–sex–geography–year cells, including zero deaths, against validated denominators. Do not train a cause classifier among decedents and call its predictions population disease risks.

---

## 4. Estimation and inference

### 4.1 Fixed target and within-county policy

Let \(T\) be the frozen eligible population. For the first application, it comprises supported **flag-1 USALEEP tracts** in eligible counties, with equal-tract weighting. Population-weighted results are a separate estimand.

Initial geographic eligibility remains:

- At least four usable flag-1 tracts.
- Inhabited P90–P10 tract-elevation contrast at least 300 m.
- Plausible local comparisons within 25 km.

These are preliminary screens. Cross-fitting additionally requires enough training observations per county; four total tracts will often be insufficient.

Reserve **20% of outcome-blinded geographic subblocks** for support design, with their outcomes sealed. Freeze conservative conditional intervals using design-set \(A,X,\) and county membership, supplemented by physical/local-support checks. Restrict inference to the resulting fixed design. Do not learn a different policy in every outcome fold.

For a hole-free supported interval:

\[
S_x=[L(x),U(x)-\delta],
\]

with \(S_x=\varnothing\) if \(U-L<\delta\), define

\[
d_\delta(a,x)=a+\delta\,1\{a\in S_x\}.
\]

County membership is included in \(x\). The primary tract shift is \(\delta=2\) mmHg. A five-mmHg shift elsewhere requires a separately approved support assessment.

The target is

\[
\boxed{
\Delta_d=
E_T[\mu(d(A,X),X)-\mu(A,X)].
}
\]

Report the shifted fraction and achieved average shift. A policy that moves almost nobody is not an informative test of oxygen biology.

For grouped individual records, the policy is defined at the **exposure-assignment geography**, so all records sharing that exposure receive the same shift. Individual covariates still enter confounding adjustment and conditional-support diagnostics.

### 4.2 Pushforward and classifier ratio

Under a continuous target exposure density \(g_T\):

\[
g_{T,d}(b\mid x)
=
g_T(b-\delta\mid x)1\{b-\delta\in S_x\}
+
g_T(b\mid x)1\{b\notin S_x\}.
\]

Therefore,

\[
\boxed{
r_d(b,x)=
\frac{g_{T,d}(b\mid x)}{g_T(b\mid x)},\qquad
\alpha_d=r_d-1.
}
\]

The unchanged branch is essential. One destination can receive both shifted and unchanged mass. This construction follows the inverse-branch formulation of modified-treatment policies. ([arxiv.org](https://arxiv.org/pdf/2006.01366))

For classifier probability \(\eta=P(C=1\mid a,x)\) and class prior \(\pi\):

\[
\widehat r_d(a,x)
=
\frac{1-\pi}{\pi}
\frac{\widehat\eta(a,x)}{1-\widehat\eta(a,x)}.
\]

Balanced weighted pairs give \(\pi=1/2\).

**Ratio fitting uses the endpoint’s target distribution—not every unlabeled ACS tract.** Unlabeled records support SSL; they cannot silently change the exposure law for selected flag-1 outcomes.

Weights depending on exposure change the target conditional density. All transformed records retain their **origin weight**. Policies involving genuine atoms require a mixed/discrete-measure derivation; no undocumented jitter or boundary clipping.

### 4.3 Primary cross-fitted one-step estimator

With fold-held-out nuisance predictions:

\[
H_i=
\widehat\mu_i^d-\widehat\mu_i
+
(\widehat r_i-1)(Y_i^*-\widehat\mu_i),
\]

\[
\boxed{
\widehat\Delta_d
=
\frac{\sum_iw_iH_i}{W},
\qquad W=\sum_iw_i.
}
\]

Equivalently,

\[
H_i=\widehat\mu_i^d+\widehat r_i(Y_i^*-\widehat\mu_i)-Y_i^*.
\]

This explicitly subtracts the observed mean on the same records.

An unchanged record may still have a nonzero residual correction because shifted mass arrives at its exposure. **Do not multiply the entire score by the “moved” indicator.**

For fixed fitted nuisances, the remainder has the mixed-error form:

\[
E_T[\widehat H]-\Delta_d
=
-E_T[
(\widehat\alpha-\alpha)
(\widehat\mu-\mu)
].
\]

This motivates orthogonal correction and double robustness; it does not prove that the fitted transformers converge sufficiently fast. ([arxiv.org](https://arxiv.org/abs/1608.00060))

### 4.4 CV-TMLE confirmation

Use the same frozen out-of-fold initial models and target. Perform a low-dimensional pooled targeting update; **do not fine-tune transformer weights on held-out outcomes**.

Target the policy mean with clever covariate \(h_i=\widehat r_i\):

| Outcome | Fluctuation |
|---|---|
| Continuous | \(\mu_\epsilon(a,x)=\widehat\mu(a,x)+\epsilon\widehat r(a,x)\) |
| Binary | \(\operatorname{logit}\mu_\epsilon=\operatorname{logit}\widehat\mu+\epsilon\widehat r\) |
| Mortality rate | \(\log\lambda_\epsilon=\log\widehat\lambda+\epsilon\widehat r\), using the normalized Poisson loss |

Solve

\[
\sum_iw_i\widehat r_i
\{Y_i^*-\widehat\mu_i^*\}=0,
\]

then compute

\[
\widehat\Delta_{\mathrm{CVTMLE}}
=
\frac1W\sum_iw_i
\{\widehat\mu^*(d(A_i,X_i),X_i)-Y_i^*\}.
\]

Evaluate the fluctuation at shifted exposures using \(\widehat r(d(A_i,X_i),X_i)\), not the observed-exposure value.

This is a proposed point-treatment CV-targeting implementation grounded in targeted estimation and CV-TMLE. Its weighted spatial implementation requires independent tests. ([pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC4117410/?utm_source=openai))

Report one-step and CV-TMLE together. Disagreement triggers investigation, not selection of the favorable answer.

### 4.5 Spatial, cluster, and survey variance

For one endpoint, define normalized contributions

\[
u_i=\frac{w_i}{W}(H_i-\widehat\Delta).
\]

For several endpoints, retain aligned vectors \(u_i\). Average **scores and influence contributions by original observation** across the three neural seeds; seeds are not independent epidemiological replications.

For independent clusters \(g\), let \(U_g=\sum_{i\in g}u_i\):

\[
\widehat V_{\mathrm{cluster}}
=
\frac{G}{G-1}\sum_g U_gU_g^\top.
\]

For spatial covariance:

\[
\widehat V_b
=
\sum_{g,h}K_b(s_g,s_h)U_gU_h^\top.
\]

Use a documented positive-semidefinite distance kernel; prespecify 50/100/200-km sensitivities for tract analyses. County aggregation captures within-county dependence; spatial kernels address additional cross-county dependence.

For stratified survey PSUs:

\[
\widehat V_{\mathrm{survey}}
=
\sum_h\frac{m_h}{m_h-1}
\sum_{p=1}^{m_h}
(U_{hp}-\bar U_h)(U_{hp}-\bar U_h)^\top,
\]

with documented finite-population/replication corrections where available. Handle singleton strata explicitly.

These expressions require appropriate sampling/dependence assumptions. Cluster-DML theory is not a theorem for arbitrary spatial fields, and irregular spatial bootstraps can fail. ([arxiv.org](https://arxiv.org/abs/1909.03489))

Use full-pipeline refitting in a limited bootstrap audit. Keep upstream USALEEP uncertainty, denominator error, exposure allocation, and migration scenarios separate; do not double-count them by automatically adding SE noise to an already noisy-outcome bootstrap.

Primary intervals concern a stated geographic stochastic-process model, not sampling error in a literal census of places. Fixed-frame interpretations need a separate uncertainty contract.

---

## 5. Training protocol

### 5.1 Nested geographic design

The primary US outer split is **five geographically grouped folds within counties**, not whole-county holdout.

- Require at least eight labeled outer-training tracts for an evaluated county and at least four in each inner fitting partition.
- Recheck support after exclusions/buffering.
- Counties failing this operational screen leave the primary target before effect estimation.
- Evaluate zero-, 10-, and 25-km buffering as prespecified dependence sensitivities, with target changes disclosed.

These minimum counts are proposed engineering gates, not sufficient inferential guarantees.

Whole-state/county deletion means **refitting the analysis without that geography**. Prediction into an unseen county is a separate transfer experiment with no invented county intercept.

For birth/survey/mortality applications, keep all records sharing a household, PSU, municipality, repeated geography, or known outcome lineage together. Use region/state offsets rather than fixed effects for the exact exposure-assignment unit when the latter would absorb all exposure variation.

### 5.2 Training sequence

For each outer fold and seed:

1. Build the permissible training and reference sets.
2. Run three inner geographic folds.
3. Fit preprocessing and masked SSL within each inner training partition.
4. Tune outcome and ratio transformers separately.
5. Obtain inner out-of-fold classifier predictions for calibration.
6. Refit SSL and nuisance models on the outer-training partition.
7. Save frozen predictions at \(A\) and \(d(A,X)\).
8. Join held-out outcomes only in the score/targeting stage.

The bounded fine-tuning grid is:

- Learning rate: \(3\times10^{-4}\), \(10^{-3}\).
- Dropout: 0, 0.10.
- Fixed architecture, weight decay \(10^{-4}\).
- Maximum 150 epochs; patience 15.
- Gradient-norm cap 1.0.
- Three registered initialization seeds: 1103, 2207, 3301.

Select by inner factual loss, calibration, and numerical validity—not effect direction, significance, or anchor agreement. Curvature penalties and disease auxiliary tasks are not part of the primary objective.

### 5.3 Leakage invariants

1. Changing any outer-held-out outcome leaves all pre-targeting models and predictions unchanged.
2. Changing held-out \(X\) leaves strict-mode fitted models unchanged; its own prediction may change.
3. No preprocessing, SSL, checkpoint selection, calibration, or county context sees held-out records.
4. All outcomes of a held-out unit are excluded across tasks.
5. No flag-2/3 modeled USALEEP labels enter primary training.
6. Original/shifted copies share every split and computational sampling decision.
7. No cross-fold graph messages, mutable caches, or full-sample demeaning.
8. Raw approved \(X\) remains available to both heads.
9. Outcome supervision within a training fold is permitted; downstream variables as predictors are not thereby permitted.
10. The scalar CV-TMLE update is an explicitly authorized use of held-out outcomes **after nuisance fitting**, not a violation of invariant 1.

---

## 6. Validation programme

### 6.1 Analytic tests before clinical estimation

| Test | Required result |
|---|---|
| Identity policy | Contrast and influence contributions exactly zero |
| Uniform \(A\sim U[0,10]\), \(\delta=2\) | \(r=0\) below 2, \(r=1\) from 2–8, \(r=2\) above 8, ignoring measure-zero boundaries |
| Linear outcome under that policy | \(\Delta=1.6\beta\), not \(2\beta\) |
| Support interval narrower than shift | Identity policy |
| Disconnected support | No bridging unsupported gaps |
| Target weights | Original/shifted copies retain origin weights; weighted pushforward integrates to one |
| One-step algebra | Contrast equals corrected policy mean minus observed mean |
| Orthogonality | Oracle nuisance plus misspecified counterpart reproduces mixed-error behavior |
| CV-TMLE | Autodifferentiation and numerical derivatives recover the intended residual moment |
| Precision | FP32 reference and mixed-precision estimates agree within registered tolerances |
| Entity replication | Duplicating records inside one exposure cluster does not fabricate independent precision |

For conditional means and ratios, test factual calibration **and**

\[
E_T[\widehat r f(A,X)]
\approx
E_T[f(d(A,X),X)]
\]

for a frozen collection of exposure functions, approved covariates, and interactions.

### 6.2 Two complementary simulation suites

**Suite A: fixed-real-covariate SCMs.** Preserve geography, approved \(X\), missingness, and cluster sizes; generate outcomes and, where appropriate, exposure assignments with known truth.

Include:

- Null, linear, nonlinear, and sign-changing effects.
- Measured and omitted confounding at local/regional scales.
- Exact and near-deterministic location–exposure relationships.
- Support gaps, atoms, extreme ratios, and heterogeneous policy eligibility.
- Exposure error, illness-related migration, selected flag-1 availability.
- Survey inclusion, missing biomarkers, under-registration, and noisy denominators.

Store both the **observed-law statistical target** and the **structural causal target**. They need not agree under unmeasured confounding.

**Suite B: independently fitted generators.** Use non-transformer flow/copula or structured latent-variable families trained on separate development data. RealCause and Credence are useful precedents, but their observational realism does not certify counterfactual truth; continuous/spatial adaptations are new work. ([arxiv.org](https://arxiv.org/abs/2011.15007))

Keep final generator families and truth inaccessible to estimator developers and pretraining.

### 6.3 Required observational-equivalence pair

Generate identical observed data from:

\[
A=h(S),
\]

\[
M_0:\quad Y=b(X)+ch(S)+\epsilon,
\]

\[
M_\tau:\quad
Y=b(X)+\tau A+(c-\tau)h(S)+\epsilon.
\]

The observational distributions coincide, but an oxygen intervention holding location fixed has different effects.

The implementation should produce the same statistical estimate from the same observations. It must **not claim that predictive fit chooses the correct causal world**. This captures the central nonidentification problem for spatially determined exposure. ([arxiv.org](https://arxiv.org/abs/2112.14946))

### 6.4 Coverage and release gates

For selected final scenarios, require at least **1,000 independent repetitions per scenario**, including the same tuning and stopping procedure used in production. At nominal 95% coverage, Monte Carlo SE is approximately 0.0069.

Proposed gates:

- One-sided 95% Monte Carlo lower bound for coverage ≥0.925.
- Upper bound for null rejection ≤0.075.
- Absolute bias/empirical SD target ≤0.10; investigate >0.20.
- Mean estimated SE/empirical SD within 0.90–1.10.
- Upper confidence bound for numerical-failure probability ≤1%.
- Publish all failures and registered retry rules.

These are acceptance criteria, not mathematical guarantees.

### 6.5 Falsification ladder

| Order | Workstream and test | Interpretation |
|---|---|---|
| 1 | WS1: reproduce legacy OxyFormer and public `elevcan` | Computational fidelity, not causal validity |
| 2 | All: analytic and known-truth calibration | Correct estimator and uncertainty implementation |
| 3 | WS3: Peru raw Hb | Population-specific physiological anchor |
| 4 | WS4: birth weight/LBW | Disease-relevant anchor; adaptation and selection remain relevant |
| 5 | WS5: reproduce published suicide specification | Association benchmark, not a compulsory causal positive control |
| 6 | WS2/5/6: locked target estimates | Only after earlier implementation gates |

The `elevcan` repository provides reproducible analysis code. ([github.com](https://github.com/dhimmel/elevcan))

No universal negative-control outcome is certified. Terrain, UV/melanoma, screening-sensitive cancers, and external causes remain separately interpreted specificity/mechanism checks.

---

## 7. Workstream mapping

**Access:** O=open; R=free routine registration/agreement; A-routine=administrative project application, explicitly conditional. No fee or project-specific scientific-panel dependency is admitted.

| Workstream | OxyFormer v2 implementation and data |
|---|---|
| **WS1 — Repair/exposure** | Reproduce legacy pipeline; implement fixed physical exposure from O 2010 Census blocks/3DEP; replace global embeddings and inappropriate adjustment. |
| **WS2 — US tract primary** | Hierarchical treatment-query outcome and origin transformers; O USALEEP 2010–2015 flag-1 outcomes plus ACS 2006–2010 SSL/covariates; within-county two-mmHg MTP. |
| **WS3 — Physiology** | Survey-weighted person/household variant; O ENDES 2023, locked 2024 replication; raw Hb and cluster altitude; optional A-routine DHS replications remain noncritical. |
| **WS4 — Birth outcomes** | Binary/categorical Colombia head and continuous Ecuador head; O DANE 2023–2025 and INEC 2024/2015; residence-based exposure, geographic clustering; US/Brazil supplements remain gated. |
| **WS5 — US disease** | County variant with state comparison strata; O current/original cancer aggregates and WONDER 2015–2019 mortality; historical smoking sensitivity and all-state/Utah/Colorado/LDS analyses retained. |
| **WS6 — Mexico mortality** | Age–sex–municipality–year count/rate variant; O EDR event years 2015–2019 and later-period check, common registration lag, CONAPO denominators; state context, not municipality fixed effects. |
| **WS7 — Mechanistic evidence** | Separate public GWAS/MR and HIF-PHI trial evidence records; no pseudo-labels or participant-row pooling into environmental training, and no artificial MTP imposed on incompatible interventions. |

ENDES documentation distinguishes raw and adjusted Hb; DANE publishes weight bands, whereas Ecuador documents weight and maternal residence. Their actual files and joins remain ingestion gates. ([datosabiertos.gob.pe](https://www.datosabiertos.gob.pe/dataset/encuesta-demogr%C3%A1fica-y-de-salud-familiar-endes-2023-instituto-nacional-de-estad%C3%ADstica-e?utm_source=openai)) US cancer/mortality releases and Mexico EDR/CONAPO remain distinct observation systems. ([cdc.gov](https://www.cdc.gov/united-states-cancer-statistics/dataviz/index.html))

For lung cancer, historical NCI smoking estimates remain primary contextual proxies; contemporary health outcomes are not substituted for smoking history. ([sae.cancer.gov](https://sae.cancer.gov/nhis-brfss/))

**Months 2–3 extension:** share lower encoder layers with country/endpoint bottleneck adapters, keeping private response and ratio heads. Fully separate endpoint transformers remain the negative-transfer comparator. Balance training by country–endpoint, not raw record count. Never force equal effects, signs, or baseline risks.

**Multiplicity remains unchanged:** Holm 0.05 for tract life expectancy and US lung incidence; BY FDR 0.05 for the prespecified adult mortality screen— all-cause, lung cancer, IHD, stroke, COPD, diabetes, kidney disease, Alzheimer disease—across endpoint–country tests. Anchors and genetic/trial conclusions have separate families.

---

## 8. Repository implementation

### 8.1 Existing entry points

The supplied audit confirms **`phase26_foundation_model.py`**. Exact filenames for phase1, phase25, phase3, and phase4 were not supplied; their paths must be resolved from the repository in PR1 rather than invented.

| Existing logical entry point | Required integration |
|---|---|
| phase1 | Delegate ingestion/exposure construction to versioned contracts; retain legacy reproduction mode |
| phase25 | Bind to design/feature-role/fold-manifest preparation after confirming its current responsibilities |
| `phase26_foundation_model.py` | Become the compatibility launcher for fold-nested v2 pretraining and nuisance fitting |
| phase3 | Call MTP one-step/CV-TMLE estimation; retain PLR only under explicit benchmark configuration |
| phase4 | Run calibration, every-state deletion, spatial sensitivity, leakage tests, and reporting |

### 8.2 Proposed exact new tree

```text
src/oxyformer/
├── cli.py
├── contracts.py
├── provenance.py
├── data/
│   ├── adapters/
│   │   ├── usaleep.py
│   │   ├── acs.py
│   │   ├── endes.py
│   │   ├── dane_births.py
│   │   ├── inec_births.py
│   │   ├── inegi_edr.py
│   │   └── conapo.py
│   ├── entity_graph.py
│   ├── feature_roles.py
│   └── loaders.py
├── exposure/
│   ├── physics.py
│   ├── population_allocation.py
│   └── quality.py
├── design/
│   ├── eligibility.py
│   ├── support.py
│   ├── policies.py
│   └── splits.py
├── models/
│   ├── tokens.py
│   ├── encoder.py
│   ├── county_context.py
│   ├── treatment_query.py
│   ├── outcome.py
│   ├── origin.py
│   ├── riesz.py
│   ├── likelihoods.py
│   ├── adapters.py
│   └── tabicl_comparator.py
├── training/
│   ├── pretrain.py
│   ├── fit.py
│   ├── nested_cv.py
│   ├── calibration.py
│   └── checkpoint.py
├── estimation/
│   ├── mtp.py
│   ├── targeting.py
│   ├── influence.py
│   └── covariance.py
├── validation/
│   ├── analytic_truth.py
│   ├── scm.py
│   ├── generators.py
│   ├── leakage.py
│   ├── overlap.py
│   ├── geography_probes.py
│   └── coverage.py
└── reporting/
    ├── diagnostics.py
    └── evidence_matrix.py

configs/
├── legacy_entrypoints.yaml
├── sources.yaml
├── endpoints.yaml
├── feature_roles.yaml
├── policies/usaleep_shift2.yaml
├── models/oxyformer_v2.yaml
├── models/ablations.yaml
├── training/nested.yaml
├── validation/final_scenarios.yaml
└── slurm/site.yaml

scripts/
├── build_tasks.py
├── run_stage.py
└── slurm/
    ├── cpu_exposure.sbatch
    ├── gpu_task.sbatch
    ├── simulation.sbatch
    └── summarize.sbatch

tests/
├── test_contracts.py
├── test_exposure.py
├── test_policy_pushforward.py
├── test_scores.py
├── test_targeting.py
├── test_fold_isolation.py
├── test_context_invariance.py
├── test_weights_offsets.py
├── test_covariance.py
├── test_resume.py
└── test_precision.py
```

`legacy_entrypoints.yaml` initially stores verified paths only. Unresolved entries remain null and fail clearly if invoked.

### 8.3 Core interfaces

The following are **new proposed APIs**:

```python
@dataclass(frozen=True)
class EstimandSpec:
    endpoint: str
    target_id: str
    outcome_scale: str
    policy_id: str
    weight_id: str
    adjustment_schema_hash: str
    inference_unit: str

class ShiftOrStayPolicy:
    def apply(self, a_mmhg, policy_covariates) -> PolicyResult: ...

class OutcomeTransformer(nn.Module):
    def mean(self, a_query, x_tokens, raw_x, context,
             group_offset, design=None) -> Tensor: ...

class OriginTransformer(nn.Module):
    def logits(self, a_query, x_tokens, raw_x, context,
               group_offset) -> Tensor: ...

class RieszTransformer(nn.Module):
    def forward(self, a_query, x_tokens, raw_x, context) -> Tensor: ...

def fit_fold(spec, split, data_manifest, model_config, seed) -> FoldArtifacts: ...
def predict_fold(artifacts, covariate_view, policy) -> OOFNuisances: ...

def one_step(nuisances, outcomes, weights, spec) -> Estimate: ...
def cv_tmle(nuisances, outcomes, weights, likelihood, spec) -> Estimate: ...
def spatial_covariance(influence, groups, locations, bandwidth) -> Covariance: ...
def survey_covariance(influence, strata, psu, design) -> Covariance: ...
```

Outcome-free prediction views cannot expose labels. Estimator adapters reject mismatched scales, policies, weights, or schemas rather than silently reinterpret them.

Every artifact records source hashes, unit IDs, parent lineage, split/config/model hashes, environment, seed, and actual parameter count. Clinical data terms govern redistribution independently of code licences.

### 8.4 Environment and licences

Use a minimal native PyTorch loop, native scaled-dot-product attention, validated YAML/Hydra configuration, Parquet derivatives, and local JSONL metrics.

- **PyTorch 2.14.1** is a verified release candidate; actual CUDA wheel/driver compatibility is still a lab smoke-test gate. ([github.com](https://github.com/pytorch/pytorch/releases))
- Start in FP32; permit BF16 transformer operations only after comparison. Keep ratio/calibration heads FP32 and score/targeting/covariance FP64.
- Native attention avoids an initial external FlashAttention build; evaluation must explicitly disable attention dropout. ([docs.pytorch.org](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html))
- Pin Python, packages, checkpoint hashes, and container/wheelhouse artifacts only after successful installation.
- PyTorch does not promise reproducibility across releases/platforms; record the exact tested environment. ([docs.pytorch.org](https://docs.pytorch.org/docs/2.14/notes/randomness.html))

**TabICLv2 comparator:** pin package `2.2.0` and the explicitly named v2 classifier/regressor checkpoints. Core code and the model card declare BSD-3-Clause; archive exact licence bytes and file hashes before use. Do not assume its ordinary interface supports survey-weighted training. ([github.com](https://github.com/soda-inria/tabicl/releases/tag/v2.2.0))

**TabPFN:** all versions are admitted by the owner decision of 2026-10-04 (Arc is a nonprofit; research use) recorded in `configs/approvals.yaml` under `owner_decisions.foundation_models.tabpfn`. That entry fixes the package pin and scope. The decision supersedes this section's earlier attribution-licence review prerequisite and its Arc-legal-clearance exclusion of later weights. Never accept floating default checkpoints. ([huggingface.co](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/main/LICENSE.txt))

### 8.5 First three pull requests

1. **PR1 — Contracts, exposure isolation, and analytic estimator tests.**  
   Resolve legacy paths; preserve reproduction mode; implement manifests, feature roles, fixed policy, analytic pushforward/score tests, and CPU exposure boundary. No clinical-effect release.

2. **PR2 — Transformer engine and strict nested training.**  
   Add tokenizer, SSL, PMA context, treatment queries, raw-\(X\) bypass, outcome/origin transformers, calibration, caches, checkpointing, and leakage tests.

3. **PR3 — Inference, validation, and first gated tract run.**  
   Add CV-TMLE, Riesz comparator, spatial/survey covariance, simulation arrays, TabICLv2 comparator, falsification reporting, and the locked USALEEP analysis.

---

## 9. Compute and timeline

### 9.1 Budget reconciliation

The reported **12–80 GPU-hours** describes a small US fit, not the full nested implementation, ablations, and inferential validation. The **9,000 GPU-hour campaign** is repeated known-truth validation. These are different workloads.

All budgets below are **planning ceilings, not measured runtimes**.

| Stage | Initial GPU-hour allocation | Deliverable |
|---|---:|---|
| Installation, smoke tests, profiling | 25 | Working single-H100 reference |
| Fully nested primary US pipeline | 300 | Five folds, three seeds, bounded tuning, SSL |
| Fixed architecture/foundation comparators | 300 | Registered ablations, not unrestricted search |
| Simulation smoke/screening | 1,700 | Oracle tests and scenario selection |
| Final coverage, split/seed and refit audits | 5,400 | Selected ≥1,000-repetition cells |
| Generator work and simulation contingency | 1,900 | Total simulation authorization: 9,000 |
| Months 2–6 international/adapters | 1,000–3,000, separately approved | New observation-model validation and replication |

With four tuning combinations, three inner folds, five outer folds, and three seeds:

\[
N_{\mathrm{nuisance\ fits}}
=5\times3\times(3\times4+1)\times2=390.
\]

SSL is reusable across fine-tuning choices **only when the training records, SSL configuration, split, and seed are identical**. Approximately 60 such SSL fits cover inner and outer refits under this schedule.

Simulation runtime is the major unknown. Final coverage experiments must reproduce tuning/stopping when those are part of the estimator. Reusing X-only SSL across simulated outcomes is allowed only for a declared fixed-\(X\), fixed-split experiment. It cannot justify reusing outcome-trained states.

If profiling shows 9,000 GPU-hours is insufficient, reduce screening breadth or request additional capacity; do not quietly remove tuning from validation or weaken the final replication count.

### 9.2 Slurm design

Parallelize folds/seeds and complete simulation repetitions. Do not distribute a small tract network across eight GPUs simply because they are available.

```bash
#!/bin/bash
# scripts/slurm/gpu_task.sbatch — PROPOSED
#SBATCH --job-name=oxy-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

: "${PYTHON:?}"
: "${TASK_MANIFEST:?}"
: "${RUN_ROOT:?}"
: "${JOB_SCRATCH:?}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
export HF_HOME="${JOB_SCRATCH}/task-${TASK_ID}/hf"
mkdir -p "$HF_HOME"

exec "$PYTHON" -m oxyformer.cli run-task \
  --manifest "$TASK_MANIFEST" \
  --index "$TASK_ID" \
  --run-root "$RUN_ROOT" \
  --resume auto
```

Submit with site-approved GPU, CPU, RAM, time, account, and partition flags. Those limits are **UNVERIFIED**. Slurm supports array concurrency caps; start with 8–16 workers and increase to 32–64 only after storage/scheduler tests. ([slurm.schedmd.com](https://slurm.schedmd.com/job_array.html))

Each task owns its writable caches. Preemptible jobs save atomic model/optimizer/scheduler/RNG/sampler checkpoints and validate completed-output hashes before skipping work. Use administrative-approved reserved capacity for long final calibration, not an assumed entitlement to all 256 GPUs.

### 9.3 Delivery calendar

| Period | Work and acceptance |
|---|---|
| **Week 1: October 5–11** | Cluster inventory; source payloads/hashes; actual USALEEP flag counts; exposure/feature-role contracts; legacy reproduction; analytic score tests; support design reservation |
| **Week 2: October 12–18** | Fold-nested SSL and both nuisance transformers; PMA/raw-\(X\) interfaces; calibration; prediction/label perturbation tests; one-GPU timing |
| **Week 3: October 19–25** | One-step/CV-TMLE and Riesz implementation; spatial null/nonlinear simulations; freeze primary recipe and final validation scenarios; conditional-support dashboard |
| **Week 4: October 26–November 1** | First gated tract result if support/calibration pass; otherwise documented stop; launch remaining final coverage cells; publish reproducibility packet and compute accounting |
| **Months 2–3** | Complete calibration; ENDES and birth anchors; US lung/Mexico pipelines; country/task adapters with separate-transformer comparators |
| **Months 4–6** | Locked replications, survey/registration sensitivity, compatible hierarchical synthesis, final evidence matrix and manuscripts |

The month-one commitment is the first **fully audited engine and tract feasibility verdict**, not guaranteed completion of all international analyses.

---

## 10. Fixed ablations and stop rules

### Registered ablation table

| ID | Change from primary | Purpose |
|---|---|---|
| A0 | Full OxyFormer v2 | Primary |
| A1 | No SSL; same transformer trained from scratch | Value of unlabeled covariates |
| A2 | No county PMA; retain county offsets/raw \(X\) | Value and risks of context |
| A3 | Early treatment fusion instead of query attention | Treatment-conditioning mechanism |
| A4 | Transformer encoder plus varying-coefficient head | Smooth continuous-response comparator |
| A5 | Direct transformer Riesz correction instead of origin classifier | Correction-nuisance sensitivity |
| A6 | Frozen TabICLv2 outcome nuisance; same primary ratio | Foundation-model comparison |
| A7 | TabICLv2 outcome and ratio on compatible unweighted targets | Full foundation-nuisance comparison |
| A8 | Independent versus shared/adapted endpoint models | Months 2–3 negative-transfer test |
| B0 | Ridge/GAM/boosting and PLR scores | Calibration/sanity only |
| D0 | Coordinates/terrain/location embeddings | Red-team probes only; no causal result |

The raw-\(X\) bypass is never removed from a production candidate. Bottleneck-only models may appear in synthetic failure demonstrations, not clinical estimation.

### Stop rules

Stop or downgrade causal reporting for:

- Missing required raw fields, invalid joins, unresolved data/model permissions.
- Inadequate within-county training or policy support.
- Support appearing only after lossy representation compression.
- Ratio tails or influence concentration outside the validated regime.
- Leakage, cache contamination, or nonfinite scores.
- Failed coverage/SE calibration.
- Material one-step/CV-TMLE disagreement not explained by finite-sample diagnostics.
- Dominance by fewer than 30 information-bearing regions, one state >25% of information, or one block >10%, unless bespoke small-cluster validation justifies inference.
- Effect claims dependent on choosing the favorable architecture, seed, spatial bandwidth, or support threshold.

Ratio \(p_{99}>10\) or weight ESS below 25% of the target sample are initial **warning thresholds**, not universal mathematical cutoffs. Include the moved/affected portion separately; unchanged units can hide poor overlap.

Do not fix support failure by silently shrinking the shift or trimming weights. Any revised policy receives a new estimand ID and a new locked analysis.

---

## 11. Neural-system risks and unresolved identification

| Risk | Measurement/control | What remains unresolved |
|---|---|---|
| **Representation reconstructs location/exposure** | Held-out exposure probes on raw \(X\), \(Z\), context, and their combination; prohibited geographic channels | Legitimate confounders may predict exposure strongly; deleting that information can create bias |
| **Compression hides confounding** | Mandatory raw-\(X\) bypass; reconstruction and assignment-law diagnostics | Finite optimization may still underuse important covariates |
| **County context memorizes neighborhoods** | Training-only references, no labels/A, permutation/cache tests, A2 comparison | Coarse county adjustment does not eliminate local residential sorting |
| **Neural density-ratio instability** | Conditional calibration, functional balance, boundary tests, FP64 scores, Riesz comparison | Rare support regions may remain unestimable |
| **SSL learns shortcuts** | Semantic-family masking, no outcome metadata, isolated geographic folds | A larger unlabeled corpus does not increase the number of independent outcomes |
| **Multitask leakage/negative transfer** | Endpoint input permissions, whole-lineage exclusion, private adapters, separate models | Shared predictive structure does not establish transportable causal effects |
| **Overconfident uncertainty** | Spatial/survey inference and complete-pipeline simulation | No general theorem covers this exact weighted hierarchical-transformer system |
| **Foundation-model provenance/licensing** | Pinned files, separate code/weight licences, offline execution | Unknown training overlap or restrictive downstream-use terms can exclude a comparator |

**Location red-team protocol:** train separate probes that attempt to reconstruct \(A\) from coordinates, terrain, SatCLIP/GeoCLIP/AlphaEarth, approved \(X\), and learned context. Compare conditional densities, local support, and reconstruction across geographic holdouts. Keep the first four sources outside nuisance training regardless of probe performance. Satellite embedding products include later-period environmental information and are not historical baseline covariates for USALEEP. ([github.com](https://github.com/microsoft/satclip))

There is no primary adversary minimizing exposure predictability. If \(Z=f(X)\) is lossy, making \(A\) unpredictable can erase precisely the confounding information needed for adjustment. Conversely, adding exact location can destroy positivity. Neither result is repaired by a larger transformer.

Finally, the following remain substantive identification limits:

- Altitude is a bundle of environmental and social exposures.
- Residence at observation is not lifetime or pregnancy-long exposure.
- Illness-related migration and survivor/live-birth selection can alter observed populations.
- Historical smoking remains imperfectly measured for lung cancer.
- Flag-1 tract selection can change the target and induce selection bias.
- Hb genetics, HIF-PHI assignment, and residential pressure are different interventions.

**OxyFormer v2 is therefore a transformer-based causal estimation system with explicit identification conditions—not a machine that turns geographic prediction into an oxygen-specific causal claim.** The implementation succeeds when it delivers reproducible, calibrated, supported estimates and correctly identifies where the data cannot support the desired conclusion.