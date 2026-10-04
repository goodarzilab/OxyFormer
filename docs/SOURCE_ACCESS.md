# Source acquisition audit

Inspected on 2026-10-04 against frozen plan sections 7–9, its exposure contract
in section 3.1, the orchestrator mandate, and read-only `configs/approvals.yaml`.
The six committed acquisition manifests are **blocked**. They are executable
contracts whose fetch operation refuses before network or output creation until
required access, resource and approval gaps are resolved. No study payload is
committed, and no successful source acquisition or national DEM coverage is
claimed. Catalog metadata and documentation were inspected outside the repo.

`configs/sources.yaml` is JSON-compatible YAML so the CLI needs only the Python
standard library. It resolves source ids to committed `configs/sources/*.json`.
There is no CLI option to load an attempt-directory manifest. Acquisition
artifacts and reports in an attempt directory are outputs, never production
configuration inputs.

## Releases and access evidence

| Manifest | Selected release scope | Remaining blockers |
| --- | --- | --- |
| `us.json` | USALEEP national File A, 2010–2015; complete ACS 2006–2010 tract/block-group sequences and geography files; `dhimmel/elevcan` at `7aed9f29d2371eb4918f337a138608e6b6d9e311` | Worker GET could not retrieve CDC terms; Census terms returned 403; payload contents unverified. No ACS endpoint covariate concepts selected. |
| `census.json` | 49 catalog-listed 2010 block population/housing archives and 49 matching jurisdiction SF1 archives | Census terms unavailable; bulk layout and actual block/SF1 joins need verification. |
| `dem.json` | USGS 3DEP source family only | Owner DEM resolution approval pending; no selected product version, tiles, approved footprints or tile ceilings. |
| `endes.json` | Separate 2023 development and locked 2024 replication | Complete module inventories, 2024 dictionary and accessible source-specific terms unresolved. The Peru open-data portal returned access errors. |
| `births.json` | DANE 2023, 2024, 2025; INEC 2024 and 2015 | DANE download pages use a CAPTCHA flow; unattended access unverified. INEC 2015 dictionary unresolved; contents/units/joins need validation. |
| `mexico.json` | Occurrence years 2015–2019; candidate CONAPO municipal reconstruction/projections 1990–2040 | Common registration lag, corresponding registration-release inventory, later-period check, compatible denominator file, dictionary and geographic crosswalk unresolved. |

The [CDC catalog](https://www.cdc.gov/nchs/nvss/usaleep/usaleep.html) and
[record layout](https://www.cdc.gov/nchs/data/nvss/usaleep/Record_Layout_CensusTract_Life_Expectancy.pdf)
distinguish observed, predicted and mixed mortality flags. The plan's primary
training restriction to flag 1 is preserved; acquiring the raw File A is not
permission to train on flags 2/3. The catalog's public-domain statement was
read through the browser, but a worker GET returned 403. HEAD success does not
resolve that discrepancy.

The [ACS summary-file directory](https://www2.census.gov/acs2010_5yr/summaryfile/)
links the tract/block-group and geographic ZIPs, sequence lookup and technical
manual. These are full raw-source selections, not approved analysis features.
The [2010 block catalog](https://www2.census.gov/geo/tiger/TIGER2010BLKPOPHU/)
and each state directory under [2010 SF1](https://www2.census.gov/census_2010/04-Summary_File_1/)
were inspected to enumerate actual filenames. Current-vintage Census geography
is not substituted. Catalog matching establishes intended release/jurisdiction
pairs, not successful row-level population reconciliation. Census terms at the
recorded endpoint returned 403, so both Census-containing bundles stay blocked.

The pinned [elevcan license](https://raw.githubusercontent.com/dhimmel/elevcan/7aed9f29d2371eb4918f337a138608e6b6d9e311/LICENSE.md)
was inspected: code/data are CC0; figures and writing are CC-BY 4.0. The archive
URL contains the inspected full commit SHA. Its included data remain outside
this repository when acquired. Do not replace that SHA with a floating branch.

[USGS product documentation](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services)
provides general 3DEP use terms and product descriptions. Those public terms do
not select an owner-approved scientific resolution. No exposure constants or
resolution were chosen. On continuation after the initial approval stop, this
unit retained blocked DEM acquisition rather than assuming the pending decision.
`configs/approvals.yaml` was never changed.

[INEI methodological documents](https://proyectos.inei.gob.pe/endes/documentos.asp)
provide the exact 2023 dictionary archive link. Keep 2023 and 2024 separate.
The raw/adjusted Hb distinction, altitude units, survey design variables and
person/household joins must come from release dictionaries, not inferred DHS
column conventions. Failed portal access is recorded as an inspection limit,
not evidence that the agency has never published a resource.

DANE catalog identities are [2023/876](https://microdatos.dane.gov.co/index.php/catalog/876),
[2024/878](https://microdatos.dane.gov.co/index.php/catalog/878), and
[2025/915](https://microdatos.dane.gov.co/index.php/catalog/915). The 2025 archive
was actually listed on 2026-09-29; older preliminary bulletins do not establish
its present absence. Archive links were read from the catalog download buttons,
which invoke a CAPTCHA modal. No CAPTCHA was solved or bypassed. DANE terms
require attribution and restrict making data available to multiple users
without prior written permission. A private attempt-local payload is not
permission for shared redistribution. Weight bands must not be interpreted as
continuous grams; dictionary validation belongs to each release.

The [INEC 2024](https://www.ecuadorencifras.gob.ec/nacidos-vivos-y-defunciones-fetales-2024/)
and [2015](https://www.ecuadorencifras.gob.ec/estadisticas-de-nacimientos-y-defunciones-2015/)
pages list actual archives. Their footers link CC-BY 4.0; preserve the source
pages and recheck archive-specific conditions. The 2024 bundle includes fetal
deaths as well as live births and must be separated at ingestion. The 2015
questionnaire is not accepted as a substitute for the missing variable dictionary.
Occurrence-year completeness and late-registration adjustments are not asserted.

[INEGI EDR](https://www.inegi.org.mx/programas/edr/) publishes registration-based
releases; its [2019 dictionary index](https://www.inegi.org.mx/rnm/index.php/catalog/617/data-dictionary)
and [use terms](https://www.inegi.org.mx/inegi/terminos.html) were inspected.
For a common integer lag L, occurrence years 2015–2019 require registration
releases through 2019+L, with the same y+L cutoff for every occurrence year y.
L remains unset; assuming only 2015–2019 registration files would silently omit
late registrations. The [CONAPO candidate catalog](https://www.gob.mx/conapo/documentos/reconstruccion-y-proyecciones-de-la-poblacion-de-los-municipios-de-mexico-1990-2040)
describes municipal reconstruction/projections; its worker response was a
challenge page. No compatible age–sex–municipality–year denominator or historical
municipality crosswalk was verified. Do not join on geographic codes alone.

## Contract and shard rules

Every resource records its release, role, exact HTTPS URL, relative destination,
format, positive transfer ceiling, observed availability and inspection evidence.
A SHA-256 expectation is optional. An optional positive `expected_bytes` records a separately inspected exact length, distinct from the maximum transfer ceiling. An observed hash without a predeclared
expectation audits retrieved bytes; it does not authenticate a publisher's
release. Live agency URLs can change. Preserve receipts and use their hashes
as downstream input identities; never claim byte reproducibility from a URL alone.

`requirements` enumerates unresolved resources without inventing filenames.
Unavailable required dictionaries, terms, modules or denominators block the
whole acquisition. `validate` checks structure even when blocked; exit 0 from
`validate` is not acquisition readiness. `fetch` refuses blocked manifests.
Readiness changes require resolving all blockers, checking resource availability,
and reviewing the committed contract. The fetcher does not grant new permissions.

`atlas_shards.json` partitions the 48 contiguous states plus DC into the nine
specified Census divisions. Each intended jurisdiction occurs once. Alaska,
Hawaii and territories are outside that declared atlas. Every group remains
blocked with null DEM footprint and byte ceiling. Null means unset, never
unlimited. The current validator deliberately refuses runnable DEM groups until
the owner-approved selection contract is supplied in a reviewed change. Such a
change must require approved outcome-blind footprints, versioned tile resources,
positive per-tile and per-group byte ceilings, and coverage reconciliation.
Jurisdiction membership alone is not national elevation coverage.

## CLI and output behavior

From the repository root, using the supplied environment:

```sh
PYTHONPATH=src /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python \
  -m oxyformer.data.source_manifest validate --source us

# Once this committed source manifest is ready, choose a NEW child directory:
PYTHONPATH=src /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python \
  -m oxyformer.data.source_manifest fetch --source us \
  --output-dir "$SWARM_UNIT_DIR/us-acquisition" --attempts 3 --timeout 30
```

The environment supplies an exclusively owned `SWARM_UNIT_DIR`. Output must be
a fresh strict descendant, with an existing parent. The trusted assigned root may be a symlink alias and is
resolved once; symlinks below that root, existing output directories and
repository destinations are refused. No shared caches or
promotion are supported. The root assignment is trusted; this is not a defense
against a hostile process replacing files concurrently or a caller lying about
its exclusive root. The Python API accepts a mapping for synthetic tests; the
production CLI accepts committed source ids only.

Successful acquisition writes `payload.tar`, `receipts.json` and `download.log`.
Every archive member is a regular file at the manifest's validated destination.
Third-party ZIPs/tarballs are stored opaquely and never extracted. SHA-256 is
streamed for every resource and the final tar; receipts include requested/final
URLs, lengths, attempts, timestamp and the canonical manifest hash. Schema and
content semantics still require downstream ingestion checks.

HTTP is rejected, including redirect downgrades. Empty, oversized, detectably truncated,
HTML-disguised binary responses, unexpected encodings and checksum mismatches
cannot produce a successful receipt. HTTP Content-Length or validated chunked
framing is required unless the manifest declares an expected length or hash.
Close-delimited EOF alone cannot establish completeness, so such responses
without an independent expectation fail with an explicit integrity error.
Correct framing establishes receipt of the declared HTTP body, not the scientific
completeness of a publisher release; that remains an ingestion check. Retries are limited to 1–5 attempts with
capped backoff; 403/404 and content/hash failures are not retried. `--timeout` is
a per-socket-operation timeout, not a total transfer deadline. The caller may
wrap a fetch in its own execution deadline. Failure retains the log and a
non-complete receipt, removes temporary resources and does not publish a tar.
A payload without a complete receipt must never be used downstream.

Offline tests use only synthetic response bytes and forbid socket connections:

```sh
timeout 600s env CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /mnt/weka/home/hgoodarzi/envs/oxyformer/bin/python -m pytest -q tests/test_source_manifest.py
```

The mutation check temporarily disables blocked-manifest rejection and requires
`test_blocked_manifest_cannot_download` to fail, then restores the source and
reruns the full suite. Logs, mutation evidence, review evidence and completion
reports belong in `$SWARM_UNIT_DIR`, never in Git.
