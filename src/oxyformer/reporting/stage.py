"""StageRequest adapter for anchor_review, audit_collection and tract_release.

Task JSON contains exactly four absolute dependency paths: bundle, manifest,
receipts and approvals. The latter must resolve to the repository's read-only
configs/approvals.yaml. Each must be named and hash-bound by StageRequest.
Report artifacts are created only in output_dir; no model or source is edited.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import tempfile

import yaml

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.provenance import ArtifactLineage, ArtifactRecord, ContractError, canonical_json, file_hash, read_artifact, require
from oxyformer.reporting.diagnostics import sensitivity_records, summarize
from oxyformer.reporting.evidence_matrix import LIMITATIONS, evaluate, require_container
from oxyformer.reporting.records import ExpectedTasks, ReportBundle, STAGE_GATES, TaskReceipts
from oxyformer.reporting.render import render_forest, render_html


def _realpath(path):
    """Resolve existing components strictly, permitting only a missing suffix.

    New report directories and absent protected leaves are valid. Broken links,
    loops, non-directory ancestors and other resolution errors are not. Reject
    a missing component before '..': such a request cannot later verify strictly.
    """
    path = Path(path)
    missing = []
    while True:
        try:
            resolved = Path(os.path.realpath(path, strict=True))
            return resolved.joinpath(*reversed(missing))
        except FileNotFoundError:
            require(not path.is_symlink(), f"unresolvable report path symlink: {path}")
            require(path.name != "..", f"missing report path ancestor: {path}")
            missing.append(path.name)
            path = path.parent


def _make_output_directory(root):
    """Create a resolved, preflighted destination without Python recursion."""
    pending = [root]
    while pending:
        directory = pending[-1]
        try:
            directory.mkdir(exist_ok=True)
        except FileNotFoundError:
            pending.append(directory.parent)
        else:
            pending.pop()


# Classify only the path actually imported, before resolving symlinks. Probing
# another src/stage.py lets an unrelated alias change repository authority.
_MODULE = Path(__file__)
_SOURCE_ROOT = (_MODULE.parents[3] if _MODULE.parts[-4:] ==
                ("src", "oxyformer", "reporting", "stage.py") else None)


@dataclass(frozen=True)
class _RepositoryAuthority:
    repository: Path

    @property
    def config(self):
        return self.repository / "configs/reporting.yaml"

    @property
    def approvals(self):
        return self.repository / "configs/approvals.yaml"


def _repository_authority(request):
    if _SOURCE_ROOT is not None:
        return _RepositoryAuthority(_realpath(_SOURCE_ROOT))
    # Installed code uses the frozen config spelling's repository prefix. A
    # configs/ storage symlink must not move that prefix to its target's parent.
    config = Path(request.config_path)
    require(config.name == "reporting.yaml" and config.parent.name == "configs",
            "installed reporting requires repository configs/reporting.yaml")
    return _RepositoryAuthority(_realpath(config.parents[1]))


def _publish(root, name, text):
    destination = root / name
    content = text.encode("utf-8")
    # Link a complete temporary file into place without replacing earlier output.
    with tempfile.NamedTemporaryFile(dir=root, mode="wb", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            require(not destination.is_symlink() and destination.is_file() and
                    destination.read_bytes() == content,
                    f"report output conflict: {name}; use a new isolated output_dir")
    finally:
        temporary.unlink()
    return destination


def run_stage(request: StageRequest) -> StageResult:
    report = {"schema_version": 1, "stage": request.stage, "state": "blocked", "releasable": False,
              "evidence_label": "diagnostic-only", "estimators": [], "gates": [], "limitations": list(LIMITATIONS)}
    bundle = None
    authority = None
    try:
        authority = _repository_authority(request)
        # Authenticate bundle and manifest independently so an unrelated missing
        # prerequisite cannot hide their supported outputs. Full request verification
        # remains mandatory before evaluation and again before publication.
        task_bytes = Path(request.task_path).read_bytes()
        require(sha256(task_bytes).hexdigest() == request.task_hash, "reporting task hash mismatch")
        task = require_container(json.loads(task_bytes), dict, "reporting task")
        require(set(task) == {"bundle", "manifest", "receipts", "approvals"}, "invalid reporting task fields")
        dependencies = dict(zip(request.dependency_paths, request.dependency_hashes))
        bundle = read_artifact(task["bundle"], ReportBundle, dependencies[task["bundle"]])
        report["bundle_hash"] = bundle.content_hash
        report["estimators"] = [e.to_dict()["payload"] for e in bundle.estimates]
        report["sensitivities"] = sensitivity_records(bundle)
        manifest = read_artifact(task["manifest"], ExpectedTasks, dependencies[task["manifest"]])
        report["manifest_hash"] = manifest.content_hash
        report["diagnostics"] = summarize(bundle, manifest)
        require(set(task.values()) == set(dependencies), "report task/dependency paths mismatch")
        request.verify_inputs()
        config = require_container(yaml.safe_load(Path(request.config_path).read_text(encoding="utf-8")), dict, "reporting config")
        require(config.get("schema_version") == 1, "unsupported reporting config")
        require(_realpath(request.config_path) == _realpath(authority.config),
                "reporting config path is not repository configs/reporting.yaml")
        require(request.stage in STAGE_GATES, "unknown reporting stage")
        require(_realpath(task["approvals"]) == _realpath(authority.approvals), "approval path is not owner registry")
        receipts = read_artifact(task["receipts"], TaskReceipts, dependencies[task["receipts"]])
        require(manifest.stage == request.stage, "reporting stage mismatch")
        approvals = require_container(yaml.safe_load(Path(task["approvals"]).read_text(encoding="utf-8")), dict, "owner approvals")
        report = evaluate(bundle, manifest, receipts, approvals, request.config_hash)
        # Recheck immutable dependencies after collection, before publication.
        request.verify_inputs()
    except FileNotFoundError as exc:
        report.update(state="missing", releasable=False, evidence_label="diagnostic-only")
        report["gates"].append({"gate": "inputs", "status": "missing", "reason": str(exc)})
    except (ContractError, ValueError, TypeError, KeyError, OverflowError, OSError, yaml.YAMLError) as exc:
        report.update(state="failed", releasable=False, evidence_label="diagnostic-only")
        report["gates"].append({"gate": "inputs", "status": "failed", "reason": str(exc)})
    report["request_hash"] = request.content_hash
    report["code_identity"] = request.code_identity
    try:
        return _write_report(request, report, bundle, authority)
    except (ContractError, OSError) as exc:
        # Never reference a stale released report as evidence for this failure.
        return StageResult(request_hash=request.content_hash, status="fail", artifacts=(),
                           message=f"report output unavailable: {exc}")


def _write_report(request, report, bundle, authority):
    require(authority is not None, "repository authority unavailable")
    root = _realpath(request.output_dir)
    repository = authority.repository
    protected_paths = tuple(_realpath(repository / name) for name in ("outputs", "report", "src", "configs"))
    for protected in protected_paths:
        require(not root.is_relative_to(protected) and not protected.is_relative_to(root),
                "report output overlaps protected repository path")
    require(not any(_realpath(p).is_relative_to(root) for p in
                    (request.config_path, request.task_path) + request.dependency_paths),
            "report output must be isolated from inputs")
    _make_output_directory(root)
    lineage = ArtifactLineage(
        source_hashes=tuple(s.payload_hash for s in bundle.sources) if bundle and bundle.sources else (request.task_hash,),
        unit_ids=bundle.original_ids if bundle else ("unavailable-report",),
        parent_hashes=(request.content_hash,) + request.dependency_hashes, split_hash=None,
        config_hash=request.config_hash, model_hash=None,
        environment=(("python", platform.python_version()),), seed=None, parameter_count=None)
    outputs = (("report.json", canonical_json(report), "evidence_report"),
               ("report.html", render_html(report), "evidence_html"),
               ("estimators.svg", render_forest(report), "diagnostic_forest"))
    # Recompute from verified inputs on every invocation, then verify all existing
    # bytes before filling missing files. Matching partial publication can resume;
    # differing bytes never get overwritten or returned as a current report.
    for name, content, _ in outputs:
        path = root / name
        if path.exists() or path.is_symlink():
            require(not path.is_symlink() and path.is_file() and path.read_bytes() == content.encode("utf-8"),
                    f"report output conflict: {name}; use a new isolated output_dir")
    artifacts = []
    for name, content, kind in outputs:
        path = _publish(root, name, content)
        artifacts.append(ArtifactRecord(path=name, sha256=file_hash(path), lineage=lineage, kind=kind))
    status = "fail" if report["state"] == "failed" else "blocked" if report["state"] in ("missing", "blocked") else "pass"
    return StageResult(request_hash=request.content_hash, status=status, artifacts=tuple(artifacts),
                       message=f'{report["state"]}: {report["evidence_label"]}; see every gate and limitation in report.json')


def request_main(argv=None):
    """Shared explicit v2 entry point for legacy report/asset scripts."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Build isolated, gated OxyFormer v2 reporting artifacts")
    parser.add_argument("--v2-request", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_stage(StageRequest.from_json(args.v2_request.read_text(encoding="utf-8")))
    sys.stdout.buffer.write((result.to_json() + "\n").encode("utf-8"))
    return 0 if result.status == "pass" else 1
