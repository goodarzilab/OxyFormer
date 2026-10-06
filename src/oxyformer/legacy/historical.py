"""Retrieve byte-verified historical code; never depend on local git history."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

from oxyformer.execution.paths import output_path, safe_extract
from oxyformer.provenance import file_hash, relative_artifact_path, require


class PrerequisiteMissing(RuntimeError):
    def __init__(self, reason, requests):
        super().__init__(reason)
        self.requests = requests


def dependency(request, name):
    # Stable references let a committed task bind the actual upstream attempt.
    # Absolute paths remain supported for existing callers.
    if isinstance(name, dict):
        require(set(name) == {'dependency', 'path'}, 'invalid legacy dependency reference')
        relative_artifact_path(name['path'])
        config = json.loads(Path(request.config_path).read_text())
        roots = config.get('dependencies', {})
        require(name['dependency'] in roots, 'legacy dependency unit is not declared')
        root = Path(roots[name['dependency']]).resolve(strict=True)
        path = (root / name['path']).resolve(strict=True)
        require(path.is_relative_to(root), 'legacy dependency escapes upstream attempt')
    else:
        path = Path(name).resolve(strict=True)
    approved = {Path(p).resolve(strict=True) for p in request.dependency_paths}
    require(path in approved, f"input must be a hashed StageRequest dependency: {name}")
    return path


def retrieve_source(pin, root, *, archive=None, allow_download=False):
    """Pin both commit URL and archive digest before extracting any executable."""
    if archive is None:
        if not allow_download:
            raise PrerequisiteMissing("historical source archive is not provisioned", [{
                "action": "download_public_archive", "url": pin["archive_url"],
                "sha256": pin["archive_sha256"], "bytes": pin["archive_bytes"],
                "instruction": "Provide source_archive as a hashed dependency, or explicitly set allow_download: true."}])
        archive = output_path(root, "source.tar.gz")
        with urllib.request.urlopen(pin["archive_url"], timeout=60) as response, archive.open("xb") as sink:
            remaining = pin["archive_bytes"]
            while True:
                block = response.read(min(1024 * 1024, remaining + 1))
                if not block:
                    break
                remaining -= len(block)
                require(remaining >= 0, "historical archive exceeds pinned byte length")
                sink.write(block)
    archive = Path(archive)
    require(archive.stat().st_size == pin["archive_bytes"], "historical archive size mismatch")
    require(file_hash(archive) == pin["archive_sha256"], "historical archive SHA-256 mismatch")
    extracted = safe_extract(archive, root, "workspace")
    source = extracted / pin["root"]
    require(source.is_dir(), "historical source identity/root mismatch")
    return source


def runtime_environment(root):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="", R_ENVIRON_USER="/dev/null",
               R_PROFILE_USER="/dev/null")
    for name in ("XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME", "HF_HOME", "TMPDIR"):
        directory = output_path(root, "cache/" + name.lower())
        directory.mkdir(parents=True, exist_ok=True)
        env[name] = str(directory)
    return env


def python_preflight(executable, modules, cwd, env):
    code = """import importlib, json, platform
result = {'python': platform.python_version(), 'modules': {}, 'missing': []}
for name in json.loads(__import__('sys').argv[1]):
    try:
        m = importlib.import_module(name)
        result['modules'][name] = str(getattr(m, '__version__', 'stdlib'))
    except Exception as e:
        result['missing'].append({'module': name, 'error': str(e)})
print(json.dumps(result))
"""
    try:
        result = subprocess.run([executable, "-B", "-c", code, json.dumps(modules)], cwd=cwd,
                                env=env, capture_output=True, text=True, timeout=60, check=True)
        inventory = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise PrerequisiteMissing("historical Python environment unavailable", [{
            "action": "provision_python", "executable": executable, "modules": modules,
            "instruction": "Provision a compatible isolated Python environment and set parameters.python.",
            "error": str(exc)}]) from exc
    if inventory["missing"]:
        raise PrerequisiteMissing("historical Python dependencies unavailable", [{
            "action": "install_python_dependencies", "executable": executable,
            "missing": inventory["missing"], "instruction": "Install compatible numpy/pandas/torch in an isolated environment; record versions and rerun. No historical lockfile exists."}])
    return inventory


def r_preflight(executable, packages, cwd, env):
    code = ('p <- c(' + ','.join(json.dumps(p) for p in packages) + '); '
            'missing <- p[!vapply(p, requireNamespace, logical(1), quietly=TRUE)]; '
            'if(length(missing)) {cat(paste(missing,collapse=",")); quit(status=42)}; '
            'if(!capabilities("cairo")) {cat("cairo graphics"); quit(status=42)}; '
            'sessionInfo()')
    try:
        result = subprocess.run([executable, "--vanilla", "-e", code], cwd=cwd, env=env,
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PrerequisiteMissing("Rscript is unavailable", [{"action": "provision_R",
            "executable": executable, "packages": packages, "error": str(exc),
            "instruction": "Provision an isolated historical-compatible R environment and set parameters.rscript."}]) from exc
    missing_system = [name for name in ("gs",) if shutil.which(name, path=env.get("PATH")) is None]
    if result.returncode or missing_system:
        raise PrerequisiteMissing("historical R/graphics dependencies unavailable", [{
            "action": "provision_R_dependencies", "packages": packages, "system": missing_system,
            "diagnostic": result.stdout + result.stderr,
            "instruction": "Install the listed R packages, Cairo graphics and Ghostscript in an isolated compatible environment; retain sessionInfo(). No version lockfile exists."}])
    return {"R_session": result.stdout}


# Only verified scientific flags are forwarded. Paths are always workspace-owned.
OPTION_TYPES = {
    "phase1": {}, "phase25": {},
    "phase26": {"embedding-dim": int, "token-dim": int, "layers": int, "heads": int,
                "epochs": int, "batch-size": int, "learning-rate": float, "mask-rate": float,
                "weight-decay": float, "embedding-penalty": float, "auxiliary-weight": float,
                "consistency-weight": float, "seed": int},
    "phase3": {"include-site-models": bool, "use-foundation-embeddings": bool,
               "site-support-incidence": int, "site-support-mortality": int, "max-site-models-per-event": int},
    "phase4": {"include-site-models": bool, "use-foundation-embeddings": bool,
               "site-support-incidence": int, "site-support-mortality": int, "max-site-models-per-event": int,
               "seeds": str, "min-state-rows": int},
}


def historical_run(request, parameters, config, root):
    mode = parameters["mode"]
    pin = config["sources"][mode]
    source = retrieve_source(pin, root,
        archive=dependency(request, parameters["source_archive"]) if parameters.get("source_archive") else None,
        allow_download=parameters.get("allow_download") is True)
    env = runtime_environment(root)
    if mode == "elevcan-reproduction":
        require(not parameters.get("inputs") and not parameters.get("options"), "elevcan uses the pinned county dataset and unmodified specification")
        executable = parameters.get("rscript", "Rscript")
        inventory = r_preflight(executable, pin["r_dependencies"], source, env)
        command = [executable, "--vanilla", pin["entrypoint"]]
        for name in ("output", "figures", "tables"):
            shutil.rmtree(source / name)
            (source / name).mkdir()
        (source / "output/figdata").mkdir()
        expected = [source / p for p in pin["outputs"]]
    else:
        entry = parameters["entrypoint"]
        settings = config["entrypoints"][entry]
        # Verify each imported sibling too, notably phase4 -> phase3.
        for item in config["entrypoints"].values():
            require(file_hash(source / item["path"]) == item["sha256_initial"], "historical script identity mismatch")
        options = parameters.get("options", {})
        require(set(options) <= set(OPTION_TYPES[entry]), "unknown historical option or path override")
        required = list(settings["inputs"])
        if options.get("include-site-models"):
            required += ["outputs/phase1/site_support.csv", "outputs/phase1/cancer_endpoints_long.csv"]
        if options.get("use-foundation-embeddings"):
            required += ["outputs/phase26/county_foundation_embeddings.csv"]
        for relative, path in parameters.get("inputs", {}).items():
            require(relative in required, "unrecognized historical input destination")
            target = output_path(source, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dependency(request, path), target)
        missing = [p for p in required if not (source / p).is_file()]
        if missing:
            raise PrerequisiteMissing("historical data inputs missing", [{"action": "provision_historical_inputs",
                "paths": missing, "instruction": "Supply inputs mapping each historical relative path to a hashed dependency. Phase25 needs a verified phase2 table; no replacement is inferred."}])
        executable = parameters.get("python", sys.executable)
        inventory = python_preflight(executable, settings["python_dependencies"], source, env)
        command = [executable, "-B", settings["path"]]
        target = source / settings["output_dir"]
        if target.exists():
            shutil.rmtree(target)
        if entry == "phase1":
            command += ["--folder", str(source), "--output-dir", str(target)]
        else:
            command += ["--root-dir", str(source)]
        for name, value in options.items():
            kind = OPTION_TYPES[entry][name]
            require(type(value) is kind or (kind is float and type(value) is int), "wrong historical option type")
            if kind is bool:
                if value:
                    command.append("--" + name)
            else:
                command += ["--" + name, str(value)]
        expected = [target / p for p in settings["outputs"]]
    log = output_path(root, "execution.log")
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=parameters.get("timeout_seconds", 3600), check=False)
    missing = [str(p.relative_to(root)) for p in expected if not p.is_file()]
    passed = result.returncode == 0 and not missing
    return {"status": "pass" if passed else "fail", "source": pin,
            "command": command, "environment": inventory, "returncode": result.returncode,
            "missing_outputs": missing, "numerical_agreement": None,
            "interpretation": "Computational reproduction only; no numerical agreement or causal validity certified.",
            "provisioning_requests": [] if passed else [{"action": "inspect_historical_failure",
                "instruction": "Inspect execution.log; provision compatible dependency versions if historical APIs fail. Do not substitute repaired algorithms."}]}, [log, *[p for p in expected if p.is_file()]]
