"""Registered legacy stage: immutable reproduction or explicit repaired B0 work.

Task parameters:
  mode: initial-release | repaired-benchmark | elevcan-reproduction
  entrypoint: phase1 | phase25 | phase26 | phase3 | phase4 (not needed for elevcan)
Historical mode accepts source_archive (hashed dependency), allow_download,
inputs (historical relative name -> hashed dependency), python/rscript, options.
Repaired mode accepts records (JSON records), manifest (DataManifest), confounders
(explicit names). Phase26 additionally needs split (SplitManifest), entities
(EntityGraph), fold, epochs and seed. All paths must be StageRequest dependencies.
Input paths may also be {dependency: unit-id, path: relative-output} references
resolved through the runner's hash-bound configuration. Runtime executables are
selected explicitly; they are inventoried by preflight, not scientific inputs.
Every report states its mode, identities and interpretation. v2 execution belongs
to the owning common stage, selected by the compatibility launcher.
"""
import json
import math
from pathlib import Path
import platform
import subprocess
import tarfile

from oxyformer.contracts import DataManifest, SplitManifest, StageResult
from oxyformer.execution.paths import atomic_json, output_path
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, file_hash, require
from .historical import PrerequisiteMissing, dependency, historical_run
from .launcher import configuration


def _finite(value):
    """JSON null marks undefined historical diagnostics; never invent a number."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def repaired_run(request, parameters, root):
    import numpy as np
    import pandas as pd
    from oxyformer.data.loaders import load_records, validate_split
    from oxyformer.data.entity_graph import EntityGraph
    from .benchmark import approved_confounders, cross_fitted_partial_linear_dml, run_leave_one_state_out

    entry = parameters["entrypoint"]
    require(entry in ("phase3", "phase4", "phase26"),
            "phase1/phase25 repaired preparation must use their versioned v2 owning stages")
    if not all(parameters.get(k) for k in ("manifest", "records", "confounders")):
        raise PrerequisiteMissing("approved adapted data and explicit confounders required", [{
            "action": "provision_approved_data", "fields": ["manifest", "records", "confounders"],
            "instruction": "Provide a reviewed DataManifest and JSON records as hashed dependencies; specify approved raw covariate names. Historical county covariates are not implicitly approved."}])
    manifest = DataManifest.from_json(dependency(request, parameters["manifest"]).read_text())
    records = json.loads(dependency(request, parameters["records"]).read_text())
    data = load_records(records, manifest, manifest.spec, manifest.schema_hash)
    columns = approved_confounders(parameters["confounders"], manifest.registry, manifest.spec.endpoint,
                                  use="ssl" if entry == "phase26" else "nuisance")
    # The benchmark's legacy PLR arithmetic is unweighted. Never discard weights.
    require(manifest.weight_field is None, "legacy B0 benchmark requires explicit unit origin weights")
    require(not parameters.get("auxiliary_targets") and not parameters.get("auxiliary_weight", 0),
            "disease auxiliaries are disabled in repaired modes")
    report = {"status": "pass", "confounders": columns, "registry_hash": manifest.registry.content_hash,
              "data_manifest_hash": manifest.content_hash, "disease_auxiliaries": False,
              "interpretation": "B0 benchmark only; no causal identification or v2 MTP estimate asserted."}
    artifacts = []
    if entry == "phase26":
        if not all(parameters.get(k) for k in ("split", "entities")) or "fold" not in parameters:
            raise PrerequisiteMissing("training fold and entity manifest required", [{
                "action": "provision_split", "fields": ["split", "entities", "fold"],
                "instruction": "Supply validated split/entity artifacts. Global historical preprocessing cannot be reused."}])
        from .foundation import fit_repaired, encode_embeddings
        import torch
        split = SplitManifest.from_json(dependency(request, parameters["split"]).read_text())
        graph = EntityGraph.from_json(dependency(request, parameters["entities"]).read_text())
        validate_split(split, manifest, graph)
        train_ids = split.training_ids(parameters["fold"])
        view = data.covariates(columns, use="ssl")
        indices = [view.original_ids.index(oid) for oid in train_ids]
        values = np.array(view.values, dtype=float)[indices]
        require(not np.isinf(values).any(), "nonfinite training predictors")
        medians = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0 for c in values.T])
        values = np.where(np.isnan(values), medians, values)
        means, scales = values.mean(0), values.std(0)
        scales[scales == 0] = 1
        values = (values - means) / scales
        model, history = fit_repaired(values, epochs=parameters.get("epochs", 120), seed=parameters.get("seed", 20260306))
        weights = root / "foundation.pt"
        torch.save(model.state_dict(), weights)
        embedded = root / "training_embeddings.json"
        atomic_json(root, embedded.name, {"ids": train_ids, "values": encode_embeddings(model, values).tolist(),
                    "use": "training diagnostics only; prohibited as global nuisance covariates"})
        artifacts += [weights, embedded]
        report.update(history=history, training_ids=train_ids, split_hash=split.content_hash,
                      preprocessing={"medians": medians.tolist(), "means": means.tolist(), "scales": scales.tolist()},
                      parameter_count=sum(p.numel() for p in model.parameters()))
    else:
        frame = pd.DataFrame(records).rename(columns={manifest.id_field: "fips"})
        require(frame.fips.map(lambda s: isinstance(s, str) and len(s) == 5 and s.isdigit()).all(),
                "county benchmark requires five-digit county FIPS identifiers")
        kwargs = dict(outcome_column=manifest.outcome_field, treatment_column=manifest.exposure_field,
            covariates=columns, fold_count=parameters.get("fold_count", 5),
            alpha_outcome=25.0, alpha_treatment=10.0, seed=parameters.get("seed", 20260306))
        if entry == "phase3":
            effect, folds, curve = cross_fitted_partial_linear_dml(frame, **kwargs)
            require(math.isfinite(effect["slope"]), "nonfinite benchmark estimate")
            report.update(effect=effect, folds=folds.to_dict("records"), curve=curve.to_dict("records"))
        else:
            rows = run_leave_one_state_out(frame, registry=manifest.registry, endpoint=manifest.spec.endpoint, **kwargs)
            report.update(deletions=rows, status="pass" if all(r["status"] == "pass" for r in rows) else "fail")
    return report, artifacts


def run_stage(request):
    request.verify_inputs()
    repository = Path(__file__).resolve().parents[3]
    output = Path(request.output_dir).resolve(strict=True)
    require(not output.is_relative_to(repository), "legacy output may not be inside the repository")
    config = configuration()
    task = json.loads(Path(request.task_path).read_text())
    parameters = task.get("parameters", {})
    mode = parameters.get("mode")
    require(mode in ("initial-release", "repaired-benchmark", "elevcan-reproduction"), "explicit legacy mode required")
    require(request.stage == ("elevcan-reproduction" if mode == "elevcan-reproduction" else "legacy-reproduction"),
            "stage/mode mismatch; v2 must use its common owning stage")
    if mode != "elevcan-reproduction":
        require(parameters.get("entrypoint") in config["entrypoints"], "unknown legacy entry point")
    root = output_path(output, "legacy/" + mode)
    # Claim a fresh namespace; no input/archive/report is overwritten on retry.
    root.mkdir(parents=True, exist_ok=False)
    artifacts = []
    try:
        if mode == "repaired-benchmark":
            report, artifacts = repaired_run(request, parameters, root)
        else:
            report, artifacts = historical_run(request, parameters, config, root)
    except PrerequisiteMissing as exc:
        report = {"status": "blocked", "reason": str(exc), "provisioning_requests": exc.requests,
                  "numerical_agreement": None}
    except (ImportError, FileNotFoundError) as exc:
        report = {"status": "blocked", "reason": str(exc), "numerical_agreement": None,
                  "provisioning_requests": [{"action": "provision_missing_dependency", "diagnostic": str(exc)}]}
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        report = {"status": "fail", "reason": str(exc), "numerical_agreement": None}
    report.update(mode=mode, entrypoint=parameters.get("entrypoint"), request_hash=request.content_hash,
                  code_identity=request.code_identity)
    if mode in config["sources"]:
        report["source_identity"] = config["sources"][mode]
    report_path = atomic_json(root, "report.json", _finite(report))
    artifacts.append(report_path)
    # Keep useful partial logs on executed failures/timeouts as well.
    if (root / "execution.log").is_file() and root / "execution.log" not in artifacts:
        artifacts.append(root / "execution.log")
    lineage = ArtifactLineage(source_hashes=request.dependency_hashes or (request.config_hash,),
        unit_ids=(str(task.get("id", request.stage)),), parent_hashes=(request.content_hash,),
        split_hash=None, config_hash=request.config_hash, model_hash=None,
        environment=(("python", platform.python_version()),), seed=None, parameter_count=None)
    records = tuple(ArtifactRecord(path=str(p.relative_to(output)), sha256=file_hash(p), lineage=lineage,
        kind="legacy_report" if p == report_path else "legacy_output") for p in artifacts)
    # Export the plan interface without changing the isolated historical report.
    # The shell runner owns run.log; hashing that live stream here is invalid.
    results_path = atomic_json(output, "results.json", _finite(report))
    records += (ArtifactRecord(path=results_path.name, sha256=file_hash(results_path),
        lineage=lineage, kind="legacy_report"),)
    bundle_path = output_path(output, "reproduction_bundle.tar")
    with tarfile.open(bundle_path, "x:") as bundle:
        for artifact in records:
            bundle.add(output / artifact.path, arcname=artifact.path, recursive=False)
    records += (ArtifactRecord(path=bundle_path.name, sha256=file_hash(bundle_path),
        lineage=lineage, kind="legacy_bundle"),)
    manifest_path = atomic_json(output, "artifact_manifest.json", {
        "schema_version": 1, "request_hash": request.content_hash, "status": report["status"],
        "artifacts": [record.to_dict() for record in records],
    })
    records += (ArtifactRecord(path=manifest_path.name, sha256=file_hash(manifest_path),
        lineage=lineage, kind="artifact_manifest"),)
    return StageResult(request_hash=request.content_hash, status=report["status"], artifacts=records,
        message=report.get("reason", mode + ": " + report["status"] + "; see isolated legacy report"))
