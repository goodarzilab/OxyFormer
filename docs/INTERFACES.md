# Proposed internal scientific artifact APIs

This is the shared boundary for the frozen plan (sections 2, 4, 5 and 8), not a
validated scientific run. Import types from their defining modules. Setuptools
uses namespace discovery under `src`; downstream units add modules/subpackages
without editing any initializer. This unit adds no data, training, source
adapter, stage dispatcher or effect estimator.

The owner approvals in `configs/approvals.yaml` are read-only. Endpoint concepts
are pending. `configs/feature_roles.yaml` therefore grants no field permissions;
`configs/endpoints.yaml` contains semantic definitions but remains unusable.
There are no fabricated USALEEP/ACS columns, resolved target IDs or policy IDs.
A source being named in approvals does not mean its payload mapping is reviewed.
The plan adoption note and owner decisions admit all TabPFN versions, superseding
the older restriction within section 8.4; comparator implementation is outside
this unit. No scientific parameters were changed.

## Identity, immutability and persistence

All records inherit `provenance.Immutable` and are frozen, slotted dataclasses.
Construction copies lists into nested tuples and validates types. No caller-owned
mutable containers or dataframes survive in a covariate view. Use keyword
arguments. `to_json()` emits UTF-8 canonical JSON with sorted keys, finite
numbers, a type tag and schema version 1; `from_json()` rejects unknown fields,
versions and types. Sequence order is significant (especially observation IDs,
columns and values); mapping-like source mappings, feature rules and environment
entries are sorted. `content_hash` is SHA-256 of those exact bytes. No implicit
schema migrations are performed.

`provenance.write_artifact(path, value)` creates once; it never overwrites.
`read_artifact(path, Type, expected_hash)` checks the caller's expected digest
and canonical encoding before accepting a record. Publish complete files from
an attempt-private staging directory via the consuming unit's atomic checkpoint
writer. This helper does not implement checkpoint atomicity or concurrent writes.

`ArtifactLineage` explicitly records source payload hashes, original unit IDs,
parent artifact hashes, split/config/model hashes, environment, seed and actual
parameter count. Non-model or aggregate artifacts use explicit null model/seed/
count values as appropriate; trained models require a count. SourceManifest is
a root artifact with source URI/version, payload, licence and source-schema
hashes plus a reviewed field mapping and review reference. Estimand source
lineage hashes bind the entire source manifests, including mapping provenance.
Derived scientific records embed their lineage: changing a source, parent, split,
config, environment or model changes the containing artifact hash. This detects
substitution against a trusted expected hash; it does not certify truthful
metadata, valid study design or source permissions. Parent hashes name immutable
artifacts, not mutable filenames. The producer must make those parents available.

## Scientific records (`oxyformer.contracts`)

| Type | Boundary |
| --- | --- |
| `EstimandSpec` | Endpoint, frozen target ID, outcome scale, frozen policy ID, weight ID, adjustment registry hash, inference unit, source lineage hash. `assert_compatible` requires equality in every dimension. |
| `SourceManifest` | Root source bytes/schema/licence identity and source-to-canonical mapping. Unreviewed manifests can be serialized but `assert_usable` and data loaders refuse them. |
| `DataManifest` | Spec, reviewed-or-pending sources, ordered `ColumnSpec` schema, feature registry, original IDs, explicit ID/outcome/exposure/weight roles, entity graph hash and lineage. Its `schema_hash` binds ordered column names, dtypes and nullability. |
| `SplitManifest` | Spec, outer/inner level, original IDs with held-out fold IDs, sealed design IDs, excluded IDs, registered seed IDs, entity graph hash and lineage. Design/excluded records never enter `training_ids(fold)`. Inner manifests describe their own permitted parent partition. |
| `CovariateView` | Spec, permissions, original IDs, permitted feature names, copied scalar values and lineage. Use is nuisance, SSL or county context. Contains no labels or reference to the original data object. |
| `OOFNuisances` | Spec, original IDs, fold and seed IDs, `mu_a`, `mu_d`, `r_a`, `r_d`, origin weights and lineage. `mu_d` and `r_d` mean predictions at `d(A,X)`, never at a clipped dose. |
| `Estimate` | Spec, method, point estimate, optional SE, per-original-observation scores/influence contributions, contributing seed IDs and lineage. Values represent the endpoint's declared scale. |

Outcome scale is not a unit conversion hint. Different weights, support policies,
source lineage, schemas or targets require different estimands. Weight IDs must
identify an immutable weight definition: target mass, age standardization and
likelihood exposure are distinct. A null DataManifest weight field explicitly
means unit origin weights, never an inferred population weight. Endpoint adapters
are responsible for validating denominator/offset and likelihood-specific rules.

IDs are stable strings (preserve leading zeros). An OOF row is uniquely keyed by
`(original_id, seed_id)`, with a fold ID. Original/shifted predictions live on the
same row and share the origin weight. The fold and weight cannot vary across
seeds. `loaders.validate_oof` requires the complete original-ID × registered-seed
product for that split, the correct held-out fold, original weights, source/data
parent and split hash. Partial fold outputs may be serialized while assembling;
they cannot enter `join_outcomes` as a complete OOF artifact. Do not stack
shifted copies or seeds as independent epidemiological observations. Estimate
scores and influence contributions are already averaged by original observation
across seeds, as in plan section 4.5; retain that alignment for covariance across
endpoints. The contracts validate representation/alignment, not score arithmetic.

## Permissions, entities and local loaders

`data.feature_roles.FeatureRule` grants a named endpoint/use only with an explicit
approval reference. `FeatureRegistry.require` denies unknown, unapproved or
incompatible uses. Predictor permissions are nuisance/SSL/context; exposure,
exposure proxies, precise geography, downstream health, outcome metadata, labels,
IDs and county cannot be ordinary predictors even if such a use is requested.
County has a separate `county_routing` permission for an approved coarse stratum;
there is no permission to turn county into an embedding or ordinary feature.
Diagnostic access grants no training permission. Semantic classification and the
authenticity of approval references remain the reviewed adapter's responsibility;
this boundary cannot infer that arbitrary renamed numbers are labels.

`EntityGraph` preserves namespaced household, PSU, municipality, repeated-geography
and outcome-lineage links. `components()` takes their transitive closure;
`assert_partition()` requires a complete assignment with no component crossing
partitions. Namespace keys must encode source/time scope where IDs are reused.
Outcome lineage spanning tasks must use a common namespace. County is deliberately
not a universal dependence link: the primary split is within counties. Adapters
must supply known links; absence of a link is not proof of independence.

`load_records(records, manifest, expected_spec, expected_schema_hash)` accepts
local, already-adapted records. It checks all estimand fields, exact schema and
dtypes, source mapping usability and original row IDs/order. It does not download
or parse raw source formats. The trusted adapter verifies actual source file
hashes against SourceManifest before calling it. `LoadedData` is privileged:
training-label/score-side code may read it, prediction/SSL/context must receive
only `data.covariates(columns, use=...)`. `county_routing(name)` is separate.
`CovariateView.column` and indexing reject labels, including absent labels;
construction and deserialization also reject them. Opaque lineage hashes may
change with upstream labels, but covariate values contain no label channel.

Call `validate_split(split, manifest, entity_graph)` before fitting. It binds
data/source/graph identity and enforces whole-lineage partitions, including the
sealed design partition. The split producer additionally enforces geography,
buffering, nested-training isolation, approved fold counts and eligibility minima;
a structural contract alone cannot verify those scientific gates.

Call `join_outcomes(nuisances, data, split, expected_spec)` only in score/targeting
code, after nuisances are frozen. It joins by original ID in nuisance row order
and refuses mismatched estimands, folds, weights and incomplete seed coverage.
Held-out labels are never added to a `CovariateView`. Model units own training-only
label access, supervised losses and checks that fitted states do not see held-out
labels. All synthetic names in tests are fixture-only and convey no production
approval.

## Stage extension and continuation

Each future stage module implements `run_stage(request: StageRequest) -> StageResult`
(the structural `StageRunner` protocol). The registry/CLI is owned by its own
unit; importing these contracts dispatches no code. A request carries stage name,
absolute config/task/dependency paths with file hashes, an absolute attempt output
directory and the full code commit identity. Call `request.verify_inputs()` before
execution. Serialized request hashes bind both content identities and paths;
moving an attempt requires a new request and a declared continuation relationship.

A result has the request hash, exactly one of `pass`, `fail`, `blocked`, a message,
and relative `ArtifactRecord` entries. Each record carries file SHA-256, lineage
and kind. `pass` must declare artifacts. `blocked` describes a missing prerequisite;
`fail` describes an executed gate/task failure. Neither permits dependent work.
Consumers must check `status == "pass"` as well as `result.verify(request)`.
Verification checks request/input identity, output confinement (including symlink
resolution) and bytes; it does not decide scientific success. ArtifactRecord's
content hash and the enclosing result hash bind file digest and lineage together;
a bare file checksum alone cannot authenticate separately stored provenance.

Stage-specific continuation artifacts should include model, optimizer, scheduler,
RNG and sampler state; preprocessing; calibration; fold/reference IDs; config,
checkpoint and source hashes; seed; environment; actual parameter count; immutable
prediction shards and progress position. Declare each as a relative artifact with
a kind (for example `checkpoint`, `preprocessing`, `oof_nuisances`). Verify hashes
and all science identities before resuming or skipping work. Partial artifacts
may accompany blocked/failed results but cannot be consumed as passed outputs.
Checkpoint writers own atomic publication and format-specific validation. No
universal tensor/checkpoint schema or runtime stage is fabricated here.
