# Source acquisition

Manifests distinguish resource inspection from downstream ingestion and scientific
coverage. US, DEM, Census, ENDES, INEC and DANE births are acquisition-ready.
Every ready resource requirement names inspected resources. DEM also permits
explicit no-product requirements backed by listing and shoreline evidence; those
requirements name no resource. Any unresolved requirement blocks its source
before network access or output creation. INEC remains independently usable.

The 2026-10-04 inspection implements the approvals in `configs/approvals.yaml`:

- **DANE births:** all nine owner-staged ZIPs for 2023–2025 have exact sizes,
  SHA-256 hashes, member inventories and verified ZIP CRCs. Catalogs 876, 878 and
  915 returned their dictionary and terms pages to a non-browser HTTPS client.
  Each year includes births, nonfetal deaths and fetal deaths. Attribution and
  restrictions on redistribution remain recorded; Colombia is not dropped.
- **US:** the staged USALEEP record layout is 70,785 bytes, SHA-256
  `7612e16609d73f958ff6618ae70aab51d6502745c093c307a7f217c967e24431`.
  File A flag definitions were inspected. The approved endpoint covariate registry
  is referenced; raw acquisition does not authorize additional predictors or
  flag-2/3 training. The orchestrator's 2026-10-04 decision under the owner's
  delegated authority is recorded: `acs_tracts` is `tar_gz`, destined for
  `us/acs_tracts.tar.gz`, with expected size **3,369,803,296 bytes**. An independent
  HTTP 206 `bytes=0-262143` request returned exactly that range and object length.
  Inflating the prefix produced a valid `ustar` header and the first directory
  `tab4/sumfile/prod/2006thru2010/group2/`. The upstream `.zip` URL stays unchanged.
  Only the representation is verified here; the 3.4 GB stream was not downloaded.
  The mandatory ACS `archive_validation=gzip_tar_regular_relative` policy reads
  the entire acquired file in bounded chunks before any payload is accepted. It
  verifies gzip CRC32 and ISIZE through EOF, each tar header checksum and member
  length/padding, two zero end-of-archive blocks and all remaining zero padding.
  Only regular files and directories with relative names are permitted; absolute
  or parent-relative paths, links, devices and extension headers fail. No members
  are extracted. Any failure removes staging and the entire partial payload.
  The pre-existing pinned `elevcan` archive retains its size/SHA-256 checks;
  the new strict member policy applies to ACS.
- **DEM:** retain all 954 inspected 1/3 arc-second tiles and the three inspected
  1 arc-second fallback tiles (`n46w083`, `n48w086`, `n49w088`). Their combined
  24,549,593 bytes remain in the east-north-central shard budget. No additional
  elevation product is substituted.
  The orchestrator's second delegated decision on 2026-10-04 records `n27w080`,
  `n29w091`, `n40w074`, `n41w072` and `n43w070` as **no-product cells**. All twenty
  cell/resolution/era listings (1/3 and 1 arc-second, current and historical)
  returned HTTP 200, `IsTruncated=false`, and zero objects. Each listing's URL and
  SHA-256 are recorded in `coverage.no_product_cells`.
  Before recording, each closed nominal one-degree cell (NW tile convention)
  was intersected with every state geometry in the public Census 2010
  [shoreline-clipped 1:500,000 boundary](https://www2.census.gov/geo/tiger/GENZ2010/gz_2010_us_040_00_500k.zip).
  All five intersections were empty, including boundary touches. Both inputs
  use NAD83/EPSG:4269; no reprojection or tolerance was used. The full boundary
  ZIP is 1,072,855 bytes, SHA-256
  `d3f58eabf618304f9a86a8b4435edff9eca86aa2ad88542fdc7b6ce2f8bf0bf9`;
  ZIP CRCs were checked. Bounds, result and boundary identity accompany each cell.
  [Census describes the boundary as generalized and clipped](https://www.census.gov/programs-surveys/geography/technical-documentation/naming-convention/cartographic-boundary-file.html);
  this check does not establish absence of every small offshore feature.
  A land intersection would remain a blocked requirement with its exact finding.
  Coverage is **catalog reconciled with no-product cells**, not certified raster
  or inhabited-block coverage: 962 cells = 957 inspected tile/metadata pairs + 5
  no-product cells. Validation requires the evidence, complete reconciliation in
  the nine atlas divisions, and no tile resource for any no-product cell. Every
  required resource must remain inspected for acquisition readiness.
  **Exposure contract:** a sample in a no-product cell has missing elevation and
  is reported as missing exposure; never fill, interpolate, or renormalize it
  away. This rule is serialized and validated in DEM and atlas manifests for
  downstream exposure consumers; this acquisition module does not sample rasters.
  Nine divisions cover 48 contiguous states plus DC once; border tiles can
  overlap. AK, HI and territories are excluded; current USGS URLs are mutable.
- **Mexico (deferred to months 2–6; no month-1 consumer):** the manifest remains
  exactly as merged. Full ZIP integrity, sizes, hashes and embedded EDR dictionaries were
  inspected for registration years 2015–2024. The approved primary lag is two
  years (2015–2021 registrations); the five-year check uses 2015–2024 registrations
  for occurrence years 2015–2019. The CONAPO midyear municipality/sex/five-year-age
  CSV independently matches 36,658,654 bytes and SHA-256
  `1a8f07be08de082a0c33404f0fbce9d9292845a8c8290153a6ad2e1889bab31a`.
  Its official datos.gob.mx catalog/API declares CC BY 4.0. Mexico stays blocked
  only for the unverified CONAPO field dictionary (official datastore request
  returned HTTP 403 Access Denied) and an uninspected geographic crosswalk to EDR.
  Shared key names alone do not establish denominator compatibility. Old fetal
  dictionary links for EDR 2015/2016 are replaced by the inspected dictionaries
  embedded in the corresponding general-death archives.

Local resources use `transport=local`, a `staging_key` from
`owner_decisions.manual_acquisition`, and a relative `local_path`. Only the
approval supplies the root; the USALEEP approval also fixes the filename and
DANE retains its approved year directories. Size and SHA-256 are mandatory and
rechecked during copying. Staged files are opened read-only; bytes and permissions
are never changed. Traversal, symlinks and nonregular files fail. No CAPTCHA
bypass or application-based access is used. Live HTML/catalog snapshot hashes
are observational inspection evidence, not immutable transfer expectations.

```sh
PYTHONPATH=src python -m oxyformer.data.source_manifest fetch --source births --output-dir "$SWARM_UNIT_DIR/births-acquisition"
```

Output must be a fresh private attempt descendant. Success writes `payload.tar`,
`receipts.json` and `download.log`; failure removes partial payloads. Hashes,
ceilings and 1–5 retries bound transfer. No extraction or shared writes. Tests
use synthetic fixtures without network calls; inspection evidence lives outside
Git in `SWARM_UNIT_DIR/access_report.json`.

Redirects fail before a target request; redirect/error bodies stay unread.
HTTP HTML needs an HTML media type. The adapter normalizes payload framing:
chunked, consistent length, or close-with-expectation. Trailers require terminal
CRLF and ≤64KiB/100 lines. Tree-free XML parsing reaches EOF without external
retrieval; HTML roots fail; media-type parameters do not change the type.
Expectations never excuse framing failure.
