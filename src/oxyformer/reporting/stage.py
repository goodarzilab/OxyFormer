"""StageRequest adapter for anchor_review, audit_collection and tract_release.

Task JSON contains exactly four absolute dependency paths: bundle, manifest,
receipts and approvals. The latter must resolve to the repository's read-only
configs/approvals.yaml. Each must be named and hash-bound by StageRequest.
Report artifacts are created only in output_dir; no model or source is edited.
"""
import json
import os
from pathlib import Path
import platform
import tempfile

import yaml

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, ContractError, canonical_json, file_hash, read_artifact, require
from oxyformer.reporting.evidence_matrix import LIMITATIONS, evaluate
from oxyformer.reporting.records import ExpectedTasks, ReportBundle, STAGE_GATES, TaskReceipts
from oxyformer.reporting.render import render_forest, render_html

OWNER_APPROVALS = Path(__file__).resolve().parents[3] / "configs" / "approvals.yaml"


def _publish(root, name, text):
    destination = root / name
    # Link a complete temporary file into place without replacing earlier output.
    with tempfile.NamedTemporaryFile(dir=root, mode="w", encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink()
    return destination


def run_stage(request: StageRequest) -> StageResult:
    report = {"schema_version": 1, "stage": request.stage, "state": "blocked", "releasable": False,
              "evidence_label": "diagnostic-only", "estimators": [], "gates": [], "limitations": list(LIMITATIONS)}
    bundle = None
    try:
        request.verify_inputs()
        config = yaml.safe_load(Path(request.config_path).read_text())
        require(config.get("schema_version") == 1, "unsupported reporting config")
        require(request.stage in STAGE_GATES, "unknown reporting stage")
        task = json.loads(Path(request.task_path).read_text())
        require(set(task) == {"bundle", "manifest", "receipts", "approvals"}, "invalid reporting task fields")
        dependencies = dict(zip(request.dependency_paths, request.dependency_hashes))
        require(set(task.values()) == set(dependencies), "report task/dependency paths mismatch")
        require(Path(task["approvals"]).resolve() == OWNER_APPROVALS.resolve(), "approval path is not owner registry")
        bundle = read_artifact(task["bundle"], ReportBundle, dependencies[task["bundle"]])
        report["estimators"] = [e.to_dict()["payload"] for e in bundle.estimates]
        manifest = read_artifact(task["manifest"], ExpectedTasks, dependencies[task["manifest"]])
        receipts = read_artifact(task["receipts"], TaskReceipts, dependencies[task["receipts"]])
        require(manifest.stage == request.stage, "reporting stage mismatch")
        approvals = yaml.safe_load(Path(task["approvals"]).read_text())
        report = evaluate(bundle, manifest, receipts, approvals, request.config_hash)
        # Recheck immutable dependencies after collection, before publication.
        request.verify_inputs()
    except FileNotFoundError as exc:
        report.update(state="missing", releasable=False, evidence_label="diagnostic-only")
        report["gates"].append({"gate": "inputs", "status": "missing", "reason": str(exc)})
    except (ContractError, ValueError, TypeError, KeyError, OSError, yaml.YAMLError) as exc:
        report.update(state="failed", releasable=False, evidence_label="diagnostic-only")
        report["gates"].append({"gate": "inputs", "status": "failed", "reason": str(exc)})
    report["request_hash"] = request.content_hash
    report["code_identity"] = request.code_identity
    root = Path(request.output_dir).resolve()
    repository = OWNER_APPROVALS.resolve().parents[1]
    for protected in (repository / "outputs", repository / "report", repository / "src", repository / "configs"):
        require(not root.is_relative_to(protected), "report output overlaps protected repository path")
    require(not any(Path(p).resolve().is_relative_to(root) for p in
                    (request.config_path, request.task_path) + request.dependency_paths),
            "report output must be isolated from inputs")
    root.mkdir(parents=True, exist_ok=True)
    lineage = ArtifactLineage(
        source_hashes=tuple(s.payload_hash for s in bundle.sources) if bundle and bundle.sources else (request.task_hash,),
        unit_ids=bundle.original_ids if bundle else ("unavailable-report",),
        parent_hashes=(request.content_hash,) + request.dependency_hashes, split_hash=None,
        config_hash=request.config_hash, model_hash=None,
        environment=(("python", platform.python_version()),), seed=None, parameter_count=None)
    artifacts = []
    for name, content, kind in (("report.json", canonical_json(report), "evidence_report"),
                                ("report.html", render_html(report), "evidence_html"),
                                ("estimators.svg", render_forest(report), "diagnostic_forest")):
        path = _publish(root, name, content)
        artifacts.append(ArtifactRecord(path=name, sha256=file_hash(path), lineage=lineage, kind=kind))
    status = "fail" if report["state"] == "failed" else "blocked" if report["state"] in ("missing", "blocked") else "pass"
    return StageResult(request_hash=request.content_hash, status=status, artifacts=tuple(artifacts),
                       message=f'{report["state"]}: {report["evidence_label"]}; see every gate and limitation in report.json')


def request_main(argv=None):
    """Shared explicit v2 entry point for legacy report/asset scripts."""
    import argparse
    parser = argparse.ArgumentParser(description="Build isolated, gated OxyFormer v2 reporting artifacts")
    parser.add_argument("--v2-request", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_stage(StageRequest.from_json(args.v2_request.read_text()))
    print(result.to_json())
    return 0 if result.status == "pass" else 1
