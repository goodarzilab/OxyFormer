# Atlas task execution

`configs/execution/tasks/atlas.yaml` is the reviewed executable task manifest.
Its YAML anchors share identical inspected metadata values; every tile still has
an explicit resource ID. `configs/sources/atlas_shards.json` is an inventory,
not an executable task manifest.

The new `atlas-inputs` stage verifies the real DEM headers and paired XML against
`dem.json` and the committed tasks, then publishes `atlas_shards.json` and
`raster_metadata.json` in its own runner attempt. It inspects 957 TIFFs by their
member offsets in the uncompressed tar. It also streams every member to verify
its receipt digest, without decoding pixels or extracting the archive. The 954 primary and three approved fallback tiles use their actual CRS
and nodata. Vertical units/datum are corroborated by paired XML when omitted from
TIFF tags. The five no-product cells remain explicit inventory entries, never
fabricated rasters. Their missing-exposure policy is unchanged.

`oxyformer.exposure.tasks` adapts the execution config and occupied runner attempt
to the existing exposure module's `exposure.yaml` and empty product directory.
It preserves the original task/dependency bindings, persists the derived request,
and returns the unchanged scientific artifacts with the outer request identity.
The exposure implementation, scientific configuration and approvals are unchanged.

The runner accepts complete acquisitions only where the committed stage registry
declares an acquisition receipt. The declared `payload.tar` hash and size must
match that receipt. A create-once tree baseline lives in the independent
publication store, never in the read-only acquisition. Direct and transitive
consumers check that baseline and retain the normal post-execution change check
and permanent taint marker. All other dependencies still require a published
passing StageResult. Observed changes remain tainted even after restoration,
including when another consumer detects the write during execution. YAML dates
retain their scalar spelling in the runner's JSON config, recorded as
`yaml_timestamp_policy: preserve_scalar_text`; original approval bytes and hashes
remain bound unchanged.

## Orchestrator wiring

All eleven commands use `TASK_MANIFEST=configs/execution/tasks/atlas.yaml`, relative
to the clone. Clone `dev`, write `code_commit.txt`, then **cd into that clone**
before invoking `bash scripts/slurm/run_stage.sh --prepared`: the existing wrapper
resolves a relative task path from the caller's working directory. It already
starts Python with `-B`.

| Plan unit | STAGE | TASK_ID |
| --- | --- | --- |
| atlas-inputs (new) | atlas-inputs | atlas-inputs |
| atlas-new-england-r3 | exposure-atlas | atlas-new-england |
| atlas-mid-atlantic-r3 | exposure-atlas | atlas-mid-atlantic |
| atlas-east-north-r3 | exposure-atlas | atlas-east-north |
| atlas-west-north-r3 | exposure-atlas | atlas-west-north |
| atlas-south-atlantic-r3 | exposure-atlas | atlas-south-atlantic |
| atlas-east-south-r3 | exposure-atlas | atlas-east-south |
| atlas-west-south-r3 | exposure-atlas | atlas-west-south |
| atlas-mountain-r3 | exposure-atlas | atlas-mountain |
| atlas-pacific-r3 | exposure-atlas | atlas-pacific |
| atlas-collect-r2 | atlas-collect | atlas-collect |

Preserve existing code/runtime merge barriers. Add `atlas-inputs` to all nine
shards and collection. The new input unit needs `runtime-probe`, `stage-runner-r7`,
`atlas-tasks-r2`, and `fetch-dem-r3`. Its declared outputs are `code_commit.txt`,
`atlas_shards.json`, `raster_metadata.json`, and `run.log`.

The nine shard plan needs are exactly `runtime-probe`, `exposure-service-r3`,
`stage-runner-r7`, `fetch-census`, `fetch-dem-r3`, `atlas-tasks-r2`, `atlas-inputs`.
Collection keeps `runtime-probe`, `exposure-service-r3`, `stage-runner-r7`, all nine
`atlas-*-r3` units listed above, `atlas-tasks-r2`, and adds `atlas-inputs`.

The coordinator's acquisition variables point at containing attempts. For input
publication and shards export
`SWARM_DEP_FETCH_DEM="$SWARM_DEP_FETCH_DEM_R3/acquisition"`; for shards also export
`SWARM_DEP_FETCH_CENSUS="$SWARM_DEP_FETCH_CENSUS/acquisition"`. Preserve
`SWARM_DEP_ATLAS_INPUTS` as the new stage's attempt root. Collection exports each
registry alias from its plan alias, e.g.
`SWARM_DEP_ATLAS_NEW_ENGLAND="$SWARM_DEP_ATLAS_NEW_ENGLAND_R3"` (no `/acquisition`).
TASK_ID uses the registry identity because downstream publication checks enforce
that producer identity, independently of the coordinator plan unit's retry suffix.

Task needs remain explicit mappings to relative files. Each published stage root
adds three runner control dependencies after its listed files. Acquisitions add
none. Thus shard bindings are inventory 0, inspected metadata 1, Census payload 5
and receipt 6, DEM payload 7 and receipt 8. Collection's ordered shard bindings
start at 5 and advance by six. Do not reorder `needs` without regenerating indices.

## Reproduce generation and admission

With `PYTHONPATH=src` and the approved Python, run:

```sh
python -B scripts/build_atlas_tasks.py --dem-acquisition "$DEM_ACQUISITION" \
  --out /tmp/atlas.yaml
python -B scripts/build_atlas_tasks.py --admit --snapshot-cache \
  --census-acquisition "$CENSUS_ACQUISITION" --dem-acquisition "$DEM_ACQUISITION" \
  --out "$ADMISSION_OUTPUT"
```

Generation reads metadata from bounded headers/XML and streams member checksums. The admission probe runs every task
through the runner using real dependency paths. Only exposure calculation is
replaced with explicitly marked synthetic products, which must never enter the
scientific plan. The optional snapshot cache computes complete acquisition tree
hashes before and after the whole probe, checks stat signatures on every reuse,
and marks the overall record passing only after the final full comparison.
It changes no production verifier. Without it the probe repeats all full hashes.

Acceptance is admission, not an atlas coverage verdict. Actual raster work and
all existing scientific coverage gates remain required on the scheduled run.

Observed receipt differences are tainted before parsing or refusal, including
changes detected by the stable reader. Invalid first-time inputs and unrelated
I/O failures refuse admission without inventing evidence of a mutation. The
published request still hash-binds its acquisition snapshots for transitive use.

On Weka, a control-file rename can leave the next directory stat reporting an
older mtime/ctime; a later stat in the same scan then reports the current value.
Publication retries this specific instability in its own `_execution` directory
at most three times, requiring every recorded entry, byte, mode and size to match
the original observation. It never replaces the content baseline to obtain a
pass, and upstream fingerprinting does not use these retries.
