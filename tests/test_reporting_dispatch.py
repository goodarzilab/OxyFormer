"""Synthetic reporting requests through the real dispatcher and worker process."""
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from oxyformer.contracts import StageRequest, StageResult
from oxyformer.execution.identity import scientific_fingerprint
from oxyformer.execution.runner import dependency_variable, run
from oxyformer.provenance import ArtifactRecord, file_hash, write_artifact
from test_reporting import CONFIG, ROOT, approve, case  # noqa: F401: shared synthetic fixture

STAGES = {"anchor-review": "anchor_review", "audit-collect": "audit_collection",
          "tract-release": "tract_release"}


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def dispatch(case, tmp_path, monkeypatch):
    """Commit a synthetic repository, publish evidence, then use the real worker.

    Only the evidence producer is a callback. Reporting uses the imported copy
    of the current implementation, unchanged reporting registrations and a real recipe
    fingerprint. No production approval or dependency check is monkeypatched.
    Ignoring bytecode while copying permits both ordinary pytest and -B runs.
    """
    monkeypatch.delenv("SWARM_UNIT_DIR", raising=False)
    monkeypatch.setenv("OXYFORMER_PUBLICATION_STORE", str(tmp_path / "publications"))

    def invoke(name, *, defect=None, task_authority=None, config_authority=False,
               inline=None, nested_repository=False):
        bundle, manifest, receipts = case
        manifest = replace(manifest, stage=STAGES[name])
        if defect == "missing-receipt":
            receipts = replace(receipts, items=receipts.items[1:])
        elif defect == "failed-receipt":
            first = receipts.items[0]
            failed = replace(first.result, status="fail", message="synthetic upstream failure")
            receipts = replace(receipts, items=(replace(first, result=failed),) + receipts.items[1:])
        elif defect == "coverage":
            # Bind the deficient metric to a fresh, verified coverage receipt.
            from test_reporting import publish_coverage
            scenario = replace(bundle.coverage[0], coverage_one_sided_95_lower_bound=.90)
            receipts = publish_coverage(receipts, scenario)
            bundle = replace(bundle, coverage=(scenario,))

        repo = tmp_path / "reporting/src" if nested_repository else tmp_path / "repository"
        shutil.copytree(ROOT / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (repo / "configs/execution").mkdir(parents=True)
        names = ["bundle.json", "manifest.json", "receipts.json", "recipe_lock.json"]
        registry = yaml.safe_load((ROOT / "configs/execution/stages.yaml").read_text())
        # The callback publisher owns its contract; unrelated production stages
        # may require different inputs or outputs as their adapters evolve.
        registry["stages"]["synthetic-reporting-inputs"] = {
            "module": "oxyformer.reporting", "needs": {}, "outputs": names}
        (repo / "configs/execution/stages.yaml").write_text(yaml.safe_dump(registry))
        shutil.copy2(CONFIG, repo / "configs/reporting.yaml")
        approvals = repo / "configs/approvals.yaml"
        if defect == "owner-unapproved":
            approvals.write_bytes((ROOT / "configs/approvals.yaml").read_bytes())
        else:
            owner = approve(bundle, manifest, receipts)
            owner.update(schema_version=1, approved_by="synthetic owner", approved_on=date(2026, 10, 4))
            if defect == "missing-approval":
                owner["owner_decisions"]["reporting_approvals"] = []
            if defect == "wrong-approval-scope":
                for record in owner["owner_decisions"]["reporting_approvals"]:
                    record["config_hash"] = "0" * 64
            approvals.write_text(yaml.safe_dump(owner))
        git(repo, "init", "-q")
        git(repo, "add", "src", "configs")
        git(repo, "-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid",
            "commit", "-qm", "Synthetic reporting dispatch fixture")
        head = git(repo, "rev-parse", "HEAD")

        def attempt(label):
            out = tmp_path / label
            out.mkdir(exist_ok=True)
            (out / "code_commit.txt").write_text(head + "\n")
            return out

        upstream = attempt("evidence")
        source_task = tmp_path / "source-task.json"
        source_task.write_text(json.dumps({"id": "evidence", "stage": "synthetic-reporting-inputs", "outputs": names}))

        def publish(request, module_name, repository):
            for filename, record in zip(names, (bundle, manifest, receipts)):
                write_artifact(upstream / filename, record)
            (upstream / "recipe_lock.json").write_text(json.dumps({
                "scientific_fingerprint": scientific_fingerprint(repository)}))
            return StageResult(request_hash=request.content_hash, status="pass", message="synthetic evidence published", artifacts=tuple(
                ArtifactRecord(path=p, sha256=file_hash(upstream / p), lineage=bundle.estimates[0].lineage,
                               kind="synthetic_reporting_input") for p in names))

        source = run("synthetic-reporting-inputs", upstream, repo, task_file=source_task, execute=publish)
        assert source.status == "pass", source.message
        monkeypatch.setenv(dependency_variable("evidence"), str(upstream))
        out = attempt("reporting")
        task = {"id": "reporting", "stage": name, "needs": {"evidence": names},
                "outputs": ["report.json", "report.html", "estimators.svg"],
                "recipe_lock": {"dependency": "evidence", "path": "recipe_lock.json",
                                "sha256": file_hash(upstream / "recipe_lock.json")},
                "bundle": str(upstream / "bundle.json"), "manifest": str(upstream / "manifest.json"),
                "receipts": str(upstream / "receipts.json"), "approvals": str(approvals)}
        alternate = tmp_path / "other-repository/configs/approvals.yaml"
        alternate.parent.mkdir(parents=True)
        alternate.write_bytes(approvals.read_bytes())
        if task_authority == "copy":
            task["approvals"] = str(alternate)
        elif task_authority == "symlink":
            alias = tmp_path / "owner-alias.yaml"
            alias.symlink_to(approvals)
            task["approvals"] = str(alias)
        elif task_authority == "field":
            task["reporting_config"] = str(tmp_path / "other-repository/configs/reporting.yaml")
        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task))
        result = run(name, out, repo, task_file=task_file, deps_env=True,
                     approvals=alternate if config_authority else None, execute=inline)
        request = StageRequest.from_json((out / "_execution/request.json").read_text())
        result.verify(request)
        assert (out / "report.json").is_file(), result.message
        report = json.loads((out / "report.json").read_text())
        envelope = json.loads((out / "_execution/config.json").read_text())
        assert envelope["yaml_timestamp_policy"] == "preserve_scalar_text"
        assert file_hash(repo / "configs/reporting.yaml") == file_hash(CONFIG)
        return result, report, envelope, repo, out

    return invoke


@pytest.mark.parametrize("name", STAGES)
@pytest.mark.parametrize("nested_repository", [False, True])
def test_dispatch_complete_approved_request(dispatch, name, nested_repository):
    result, report, envelope, repo, out = dispatch(name, nested_repository=nested_repository)
    assert result.status == "pass", result.message
    assert report["state"] == ("released" if name == "tract-release" else "exploratory")
    assert report["releasable"] == (name == "tract-release")
    assert report["stage"] == STAGES[name]
    assert all(g["status"] == "pass" for g in report["gates"])
    assert envelope["approvals"]["approved_on"] == "2026-10-04"
    assert report["config_hash"] == file_hash(repo / "configs/reporting.yaml")
    assert {a.path for a in result.artifacts} >= {"report.json", "report.html", "estimators.svg"}


@pytest.mark.parametrize("name", STAGES)
@pytest.mark.parametrize("defect, expected", [("missing-receipt", "blocked"), ("failed-receipt", "fail"),
    ("missing-approval", "blocked"), ("coverage", "fail"), ("wrong-approval-scope", "blocked")])
def test_dispatch_preserves_reporting_refusals(dispatch, name, defect, expected):
    result, report, *_ = dispatch(name, defect=defect)
    assert result.status == expected, result.message
    assert not report["releasable"]
    assert report["evidence_label"] == "diagnostic-only"
    assert report["estimators"]
    assert report["state"] in ("failed", "missing", "blocked")


@pytest.mark.parametrize("name", STAGES)
def test_dispatch_real_owner_registry_remains_unapproved(dispatch, name):
    result, report, envelope, *_ = dispatch(name, defect="owner-unapproved")
    assert result.status == "blocked", result.message
    assert not report["releasable"]
    assert envelope["approvals"]["approved_by"]


@pytest.mark.parametrize("name", STAGES)
@pytest.mark.parametrize("authority", ["task", "config", "task-config"])
def test_dispatch_refuses_different_authority(dispatch, name, authority):
    result, report, *_ = dispatch(name, task_authority={"task": "copy", "task-config": "field"}.get(authority),
                                config_authority=authority == "config")
    assert result.status == "fail", result.message
    assert not report["releasable"]
    reason = " ".join(g["reason"] for g in report["gates"])
    assert {"task": "approval path is not owner registry", "config": "dispatch authority is not repository",
            "task-config": "invalid reporting task fields"}[authority] in reason


def test_dispatch_accepts_realpath_alias_of_owner_registry(dispatch):
    result, report, *_ = dispatch("tract-release", task_authority="symlink")
    assert result.status == "pass", result.message
    assert report["releasable"]
