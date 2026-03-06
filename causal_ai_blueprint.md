# Causal AI blueprint for altitude / oxygen / disease demo

## Goal

Build a persuasive small-data demo that upgrades the current county-level altitude analysis from a single regression into a transfer-learned, regularized, causal AI system that can support a funding case for collecting richer data.

The demo should answer:

1. Does lower ambient oxygen exposure associate with lower cancer incidence and/or mortality after richer adjustment?
2. Is the signal consistent across cancer sites and other disease endpoints?
3. Can we separate total effects from mediator-driven effects through obesity / diabetes?
4. Which additional data streams most improve identifiability and stability?

## Current local data support

The current folder already supports a county-level pilot.

- `BYAREA_COUNTY.csv`: county cancer incidence and mortality by site.
- `counties_x_elevation.csv`: county mean elevation.
- `ACSST5Y2021.S1901-Data.csv`: county income.
- `DiabetesPercentage.csv`: county diabetes prevalence.
- `ObesityAll.csv`: county obesity prevalence.

Current overlap across local county-level covariates is about 3,141 counties. With the existing all-race / both-sex filters and complete local covariates:

- all-cancer mortality support: ~1,210 counties
- all-cancer incidence support: ~1,022 counties

Several site-specific incidence endpoints also have usable support (for example prostate, lung and bronchus, breast, colon and rectum, melanoma, urinary bladder, kidney and renal pelvis, non-Hodgkin lymphoma, leukemias, and pancreas).

## Core estimands

The demo should estimate two separate causal targets.

### 1) Total effect of oxygen proxy on disease

Treatment:

- continuous oxygen-exposure proxy derived from county elevation

Outcome:

- primary: age-adjusted all-cancer incidence
- secondary: age-adjusted all-cancer mortality
- tertiary: site-specific incidence / mortality and selected cardiometabolic outcomes

Interpretation:

- effect of shifting a county from sea-level-equivalent oxygen to higher-altitude-equivalent oxygen, allowing downstream metabolic pathways to operate

### 2) Controlled / mediated decomposition

Mediator candidates:

- obesity
- diabetes

Interpretation:

- direct effect not passing through metabolic mediators
- indirect effect operating through metabolic-state changes

## Exposure definition

Use altitude only as a starting point. The actual treatment fed to the model should be a physiologic oxygen proxy.

Recommended construction:

1. Start with county mean elevation.
2. Convert elevation to barometric pressure.
3. Convert barometric pressure to inspired oxygen partial pressure proxy.
4. Standardize exposure for modeling and also retain a human-readable scale.

For the demo, keep both:

- `elevation_m`
- `oxygen_proxy`

This helps show that the model is biologically informed rather than just geographically correlated.

## Causal graph

Use an explicit DAG and keep variable roles fixed.

### Nodes

- **Exposure**: oxygen proxy from elevation
- **Geography**: state, region, county centroid, latitude, longitude, rurality
- **Environment**: UV, sunlight, PM2.5, temperature / climate
- **Demographics**: age structure, race / ethnicity composition, education, poverty, insurance
- **Healthcare access**: clinician supply, oncology access, hospital density, screening proxies
- **Behavior / risk**: smoking, inactivity, obesity, diabetes, alcohol where available
- **Outcome**: cancer incidence / mortality by site

### Role assignments

- likely confounders: geography, rurality, environment, demographics, care access, smoking
- likely mediators for the total-effect cancer model: obesity, diabetes
- possible negative controls: outcomes unlikely to respond to oxygen biology but sensitive to care-access or rurality artifacts

## Public data augmentation plan

Build a county-year table whenever possible.

### Cancer outcomes

- county-level incidence trends by year
- county-level mortality trends by year
- site-specific panels for common cancers

### Health behaviors / chronic disease

- smoking
- inactivity
- obesity
- diabetes
- insurance coverage
- preventive care and screening proxies if available

### Healthcare access

- hospital density
- primary care supply
- specialist supply
- oncology-related workforce proxies
- travel or shortage indicators if available

### Environment / geography

- UV
- sunlight
- PM2.5
- temperature or climate summaries
- rurality / metro influence
- social vulnerability

### Demography / socioeconomic structure

- age distribution
- sex composition
- race / ethnicity composition
- education
- income / poverty
- housing burden

## Recommended model stack

This should look sophisticated, but it must be honest about small-sample constraints.

### Layer 1: county encoder

Train a shared representation model on a much larger county-year public table using self-supervision and multitask prediction.

Recommended objectives:

- masked-feature reconstruction
- denoising / missing-feature imputation
- multitask prediction of auxiliary public-health endpoints

Encoder inputs:

- county-year structured features from ACS, PLACES, AHRF, SVI, rurality, environment, and geography

Output:

- dense county embedding `z_county_year`

### Layer 2: multitask disease heads

Fit shared-parameter heads for:

- all-cancer incidence
- all-cancer mortality
- major site-specific incidence
- major site-specific mortality
- auxiliary non-cancer disease outcomes

This is the main transfer-learning mechanism. Rare endpoints borrow strength from common endpoints and from adjacent diseases.

### Layer 3: causal head

Use the county embedding plus observed covariates in a doubly robust causal estimator for a continuous treatment.

Recommended options:

- generalized random forest for continuous treatment effects
- double / debiased machine learning with flexible nuisance models
- Bayesian additive outcome model with orthogonal score correction

Outputs:

- marginal dose-response curve for `do(oxygen_proxy = x)`
- heterogeneous effects by county profile
- uncertainty intervals

### Layer 4: spatial and hierarchical shrinkage

Add partial pooling across:

- states
- census divisions / regions
- cancer sites
- years

Add explicit spatial regularization:

- county adjacency graph penalty
- conditional autoregressive or local empirical Bayes spatial component
- latitude / longitude smooth functions

### Layer 5: mediation module

Separate total from direct effects using a mediation-aware model.

Simple first demo:

- model obesity and diabetes as mediator heads
- estimate total effect without conditioning on mediators
- estimate controlled direct effect in a second stage

Advanced extension:

- Bayesian structural equation model with shared latent county embedding

## Transfer-learning strategies

### 1) Panel expansion instead of one cross-section

The single biggest gain is to move from one county snapshot to a county-year panel. Even if the oxygen proxy is time-invariant, the richer panel supports better representation learning and stronger adjustment.

### 2) Multitask transfer across endpoints

Jointly train on cancer incidence, cancer mortality, obesity, diabetes, smoking, cardiovascular endpoints, and care-access outcomes.

Why it helps:

- shared county structure is learned from easier and denser tasks
- site-specific cancers borrow information from all-cancer outcomes
- the representation becomes less sensitive to any one noisy endpoint

### 3) Self-supervised pretraining on county covariates

Use unlabeled or lightly labeled county-year rows to pretrain the encoder before fitting the causal head.

Why it helps:

- the encoder learns correlations among environment, demography, and care-access variables before seeing the small cancer target

### 4) Hierarchical shrinkage priors

Use strong regularization on sparse coefficients and endpoint-specific deviations.

Recommended priors / penalties:

- horseshoe or regularized horseshoe for sparse linear effects
- low-rank factorization for disease-by-covariate interactions
- Gaussian process or spline penalty for smooth dose-response

### 5) Monotonicity-informed structure

Impose soft monotonic structure where justified:

- oxygen proxy should change monotonically with elevation
- some mediator paths may be regularized toward smooth monotone effects rather than unrestricted wiggles

This adds biological plausibility and reduces variance.

### 6) Spatial borrowing

Neighboring counties should share information through graph or spatial priors rather than being treated as independent draws.

### 7) Freeze-most / tune-little fine-tuning

For the final cancer model, freeze most of the pretrained encoder and fine-tune only:

- a small disease-specific adapter
- the causal nuisance models
- the dose-response head

This is critical for avoiding overfit with ~1,000 outcome rows.

### 8) Stacked nuisance learners

In the DML stage, use a small stack of regularized base learners rather than one learner:

- elastic net
- gradient boosting
- random forest / causal forest
- shallow neural net or tabular transformer head

Cross-fitted stacking improves robustness without letting any one learner dominate.

## Preferred demo architecture

### Option A: strongest and most defensible

1. Build county-year public pretraining table.
2. Train self-supervised county encoder.
3. Train multitask disease heads.
4. Freeze encoder.
5. Fit DML / GRF continuous-treatment causal head for oxygen proxy.

Why this is the recommended demo:

- clearly AI-driven
- transfer learning is obvious and justified
- strong regularization is natural
- produces causal estimands instead of pure prediction

### Option B: more Bayesian / publication-friendly

1. Use county embedding as input to a hierarchical Bayesian causal model.
2. Model site-level outcomes jointly with partial pooling.
3. Add spatial random effects and mediator equations.

Why to use it:

- more interpretable uncertainty
- strong small-sample behavior
- easy to justify in epidemiology audiences

### Option C: flashy but riskier

1. Fine-tune a tabular foundation model directly on cancer endpoints.
2. Wrap a causal head around the learned embedding.

Why this is not my first recommendation:

- visually impressive but easier to overclaim
- harder to defend causally than encoder + orthogonal estimator

## Validation plan

### Internal validation

- leave-one-state-out validation
- spatial block validation
- calibration plots for predictive heads
- stability of dose-response across random seeds and splits

### Causal robustness checks

- compare base adjustment vs expanded adjustment sets
- total-effect vs direct-effect estimates
- placebo outcomes
- placebo exposures where possible
- e-value or equivalent sensitivity analysis for unmeasured confounding

### Small-area stability checks

- compare raw rates vs smoothed / hierarchical estimates
- check whether estimated effects are driven by sparse counties

## Demo deliverables

1. **County embedding map** showing representation clusters.
2. **Oxygen dose-response plot** for all-cancer incidence.
3. **Forest plot by cancer site** with pooled and site-specific effects.
4. **Mediator decomposition plot** showing total vs direct effects.
5. **Data value plot** showing how uncertainty shrinks when adding each new data layer.
6. **Counterfactual map** of predicted incidence under sea-level vs high-altitude-equivalent oxygen.

## Minimum viable build

If the goal is a rapid but credible demo, do this first:

### Phase 1

- reproduce the existing local model
- switch primary endpoint to incidence
- build the county master table
- engineer oxygen proxy

### Phase 2

- add PLACES, SVI, RUCC, UV, PM2.5, and expanded ACS features
- fit a multitask regularized county encoder
- fit DML / GRF causal dose-response model

### Phase 3

- add panel years from public cancer data
- add AHRF healthcare-access features
- add mediation and site-level partial pooling

## Recommendation

For this project, the best balance of sophistication and credibility is:

**self-supervised county encoder + multitask transfer learning + spatial hierarchical shrinkage + continuous-treatment DML / GRF causal head**

That gives you a demo that is visibly modern AI, statistically disciplined, and well positioned to justify larger data collection.
