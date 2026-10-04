"""Fold-owned, local-only foundation nuisances; optional packages load on fit.

The public mean/logits/probability methods use the merged nuisance argument
order. Here x_tokens is a permitted CovariateView, and raw_x is its complete
numeric matrix (None maps to NaN). Foundation models do not consume learned
PMA tokens or county offsets; nonzero values are refused. No preprocessing is
fitted outside the fold's training context. Predictions use one query row at a
time to make query composition irrelevant even for transductive backends.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module, metadata
from pathlib import Path
import platform
import operator
import re

import numpy as np
import torch

from oxyformer.contracts import CovariateView, SplitManifest
from oxyformer.design.policies import PolicyPairs
from oxyformer.provenance import ContractError, canonical_json, check_hash, require


PACKAGE_PINS = {"tabicl": "2.2.0", "tabpfn": "9.1.0"}
RUNTIME_PACKAGES = ("torch", "numpy", "scipy", "scikit-learn")


def runtime_environment(package: str) -> tuple[tuple[str, str], ...]:
    """Provisioner records this after installation; consumers compare exactly."""
    try:
        return tuple(sorted([("python", platform.python_version())] +
                            [(p, metadata.version(p)) for p in (*RUNTIME_PACKAGES, package)]))
    except metadata.PackageNotFoundError as exc:
        raise ContractError("missing optional dependency; request comparator provisioning") from exc


@dataclass(frozen=True, kw_only=True)
class Checkpoint:
    package: str
    package_version: str
    repository: str
    revision: str
    filename: str
    sha256: str
    path: str
    environment: tuple[tuple[str, str], ...]

    def __post_init__(self):
        require(self.package in PACKAGE_PINS and self.package_version == PACKAGE_PINS[self.package],
                "comparator package version must equal the approved pin")
        require(bool(re.fullmatch(r"[0-9a-f]{40}", self.revision)),
                "floating checkpoint revisions are forbidden")
        check_hash(self.sha256, "checkpoint SHA-256")
        require(bool(self.repository) and self.filename.endswith(".ckpt")
                and Path(self.filename).name == self.filename, "explicit checkpoint filename required")
        require(Path(self.path).is_absolute() and Path(self.path).name == self.filename,
                "explicit absolute local checkpoint path required; floating defaults forbidden")
        env = tuple(sorted(tuple(x) for x in self.environment))
        require(len(env) == len({x[0] for x in env}) and
                set(dict(env)) == {"python", *RUNTIME_PACKAGES, self.package},
                "complete runtime fingerprint required")
        require(dict(env)[self.package] == self.package_version, "runtime package pin mismatch")
        object.__setattr__(self, "environment", env)

    def verify(self):
        require(runtime_environment(self.package) == self.environment, "runtime fingerprint mismatch")
        path = Path(self.path)
        require(path.is_file(), "local checkpoint missing; automatic downloads are forbidden")
        with path.open("rb") as stream:
            digest = sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        require(digest.hexdigest() == self.sha256, "checkpoint hash mismatch")

    @property
    def fingerprint(self):
        return sha256(canonical_json({k: v for k, v in vars(self).items() if k != "path"}).encode()).hexdigest()


def _unit_weights(weights, size: int, semantics: str):
    require(semantics == "unit", "unsupported target/survey weight semantics; comparator blocked")
    weights = np.asarray(weights, dtype=float)
    require(weights.shape == (size,) and np.isfinite(weights).all() and np.all(weights == 1),
            "unsupported sample weights; comparator blocked, never silently unweighted")


def _matrix(view: CovariateView) -> np.ndarray:
    require(type(view) is CovariateView and view.use == "nuisance", "nuisance CovariateView required")
    require(all(v is None or type(v) in (bool, int, float) for row in view.values for v in row),
            "foundation adapter supports numeric/binary approved X only")
    result = np.array([[np.nan if v is None else v for v in row] for row in view.values], dtype=float)
    require(result.ndim == 2 and result.shape[1] > 0 and not np.isinf(result).any(), "invalid raw X")
    return result


class TabICLComparator:
    """Frozen TabICLv2 outcome or origin nuisance, never an effect estimator.

    Supports identity and binary Bernoulli means, numeric/binary X with NaN,
    and strictly unit weights. Counts, binomial denominators, survey targets,
    learned context, external offsets and implicit context subsampling fail.
    Operational caps: 30,000 context rows, 500 columns including treatment.
    Paired origin contexts count both copies. Caps are conservative project
    limits, not assertions of a hard model architecture limit.
    """
    package = "tabicl"
    max_context_rows = 30_000
    max_features = 500

    def __init__(self, checkpoint: Checkpoint, *, task: str, family: str, seed: int):
        require(checkpoint.package == self.package, "wrong checkpoint package")
        require(task in ("outcome", "origin"), "unknown nuisance task")
        require(family in ("identity", "bernoulli") and (task != "origin" or family == "bernoulli"),
                "unsupported outcome family; comparator blocked")
        require(not isinstance(seed, (bool, np.bool_)), "explicit nonnegative integer seed required")
        try:
            seed = operator.index(seed)
        except TypeError as exc:
            raise ContractError("explicit nonnegative integer seed required") from exc
        require(seed >= 0, "explicit nonnegative integer seed required")
        role = "regressor" if family == "identity" else "classifier"
        require(role in checkpoint.filename, "checkpoint task/family mismatch")
        self.checkpoint, self.task, self.family, self.seed = checkpoint, task, family, seed
        self._estimator = None

    def _make_estimator(self):
        try:
            module = import_module("tabicl")
        except ImportError as exc:
            raise ContractError("missing optional dependency tabicl; request provisioning") from exc
        cls = module.TabICLRegressor if self.family == "identity" else module.TabICLClassifier
        return cls(model_path=self.checkpoint.path, checkpoint_version=self.checkpoint.filename,
                   allow_auto_download=False, device="cpu", use_amp=False, use_fa3=False,
                   n_estimators=8, norm_methods=["none", "power"], feat_shuffle_method="latin",
                   outlier_threshold=4., batch_size=1, kv_cache=False, offload_mode=False,
                   random_state=self.seed, n_jobs=1, verbose=False)

    def _training_context(self, view, split, fold, weights, weight_semantics):
        require(self._estimator is None, "comparator already fitted; use a new instance for each fold")
        require(type(split) is SplitManifest, "SplitManifest required")
        split.spec.assert_compatible(view.spec)
        require(view.original_ids == split.training_ids(fold), "context must equal fold training IDs in order")
        require(view.lineage.split_hash == split.content_hash, "context split fingerprint mismatch")
        require(self.seed in split.seed_ids, "seed not registered in split")
        _unit_weights(weights, len(view.original_ids), weight_semantics)
        raw = _matrix(view)
        require(raw.shape[1] + 1 <= self.max_features, "foundation feature limit exceeded")
        return raw

    def _fit(self, view, split, fold, matrix, labels):
        require(2 <= len(matrix) <= self.max_context_rows, "foundation context limit exceeded; no subsampling")
        labels = np.asarray(labels, dtype=float)
        require(labels.shape == (len(matrix),) and np.isfinite(labels).all(), "invalid training labels")
        if self.family == "bernoulli":
            require(set(labels) == {0., 1.}, "binary context requires both classes; no continuous/binomial labels")
        self.checkpoint.verify()
        estimator = self._make_estimator()
        estimator.fit(matrix.copy(), labels.copy())
        self._estimator = estimator
        self.training_ids = tuple(view.original_ids)
        self.columns = view.columns
        self.spec = view.spec
        self.split = split
        self.fold = fold
        self.context_hash = sha256(canonical_json({
            "view": view.content_hash, "split": split.content_hash, "fold": fold,
            "matrix_sha256": sha256(matrix.tobytes()).hexdigest(),
            "labels_sha256": sha256(labels.tobytes()).hexdigest(),
            "runtime": self.checkpoint.fingerprint, "seed": self.seed,
            "task": self.task, "family": self.family,
        }).encode()).hexdigest()
        return self

    def fit_outcome(self, view, split, fold, a, y, *, sample_weight, weight_semantics):
        require(self.task == "outcome", "outcome fit requires outcome comparator")
        raw = self._training_context(view, split, fold, sample_weight, weight_semantics)
        a = np.asarray(a, dtype=float)
        require(a.shape == (len(raw),) and np.isfinite(a).all(), "invalid training treatment")
        return self._fit(view, split, fold, np.column_stack((a, raw)), y)

    def fit_origin(self, view, split, fold, pairs: PolicyPairs, *, weight_semantics):
        require(self.task == "origin" and type(pairs) is PolicyPairs, "origin fit requires PolicyPairs")
        n = len(view.original_ids)
        require(pairs.original_ids == view.original_ids * 2 and
                pairs.transformed == (False,) * n + (True,) * n, "origin pair order mismatch")
        require(pairs.policy_id == view.spec.policy_id and pairs.weight_id == view.spec.weight_id,
                "origin pair estimand mismatch")
        _unit_weights(pairs.origin_weights, 2 * n, weight_semantics)
        raw = self._training_context(view, split, fold, pairs.origin_weights[:n], weight_semantics)
        a = np.asarray(pairs.a_mmhg, dtype=float)
        require(a.shape == (2 * n,) and np.isfinite(a).all(), "invalid paired treatment")
        return self._fit(view, split, fold, np.column_stack((a, np.tile(raw, (2, 1)))), pairs.transformed)

    def _predict(self, a_query, x_tokens, raw_x, context, group_offset):
        require(self._estimator is not None, "comparator is not fitted")
        require(runtime_environment(self.package) == self.checkpoint.environment, "runtime fingerprint mismatch")
        view = x_tokens
        require(type(view) is CovariateView, "foundation x_tokens must be CovariateView")
        self.spec.assert_compatible(view.spec)
        require(view.columns == self.columns and view.lineage.split_hash == self.split.content_hash,
                "query schema/split fingerprint mismatch")
        held_out = {oid for oid, fold in zip(self.split.original_ids, self.split.fold_ids) if fold == self.fold}
        require(set(view.original_ids) <= held_out, "queries must belong to this fitted fold's held-out IDs")
        raw = _matrix(view)
        supplied_tensor = torch.as_tensor(raw_x).detach().cpu()
        require(not supplied_tensor.is_complex(), "raw-X must be real numeric/binary values")
        supplied = supplied_tensor.to(torch.float64).numpy()
        # Floating representations may round the immutable view values. Integer
        # and boolean representations must match without truncation/coercion.
        expected = (torch.as_tensor(raw, dtype=supplied_tensor.dtype).to(torch.float64).numpy()
                    if supplied_tensor.is_floating_point() else raw)
        require(supplied.shape == raw.shape and np.array_equal(supplied, expected, equal_nan=True),
                "complete approved raw-X bypass required")
        for value in (context, group_offset):
            require(value is None or (not getattr(value, "requires_grad", False) and
                                     bool((torch.as_tensor(value) == 0).all())),
                    "foundation comparator does not support PMA tokens or group offsets")
        require(isinstance(a_query, torch.Tensor) and a_query.ndim == 3 and
                a_query.shape[0] == len(raw) and a_query.shape[2] == 1 and
                a_query.is_floating_point() and bool(torch.isfinite(a_query).all()), "invalid query shape/values")
        doses = a_query.detach().cpu().to(torch.float64).numpy()[..., 0]
        output = np.empty(doses.shape, dtype=float)
        # A single query prevents other queried rows from entering preprocessing
        # or transductive inference. The fitted training context is unchanged.
        for i in range(len(raw)):
            for j in range(doses.shape[1]):
                row = np.concatenate(([doses[i, j]], raw[i]))[None, :]
                if self.family == "identity":
                    value = np.asarray(self._estimator.predict(row, output_type="mean")).reshape(-1)
                    require(value.shape == (1,), "unexpected regression prediction shape")
                    output[i, j] = value[0]
                else:
                    classes = np.asarray(self._estimator.classes_)
                    require(classes.shape == (2,) and set(classes) == {0, 1}, "binary classes mismatch")
                    prob = np.asarray(self._estimator.predict_proba(row))
                    require(prob.shape == (1, 2), "unexpected binary probability shape")
                    output[i, j] = prob[0, int(np.flatnonzero(classes == 1)[0])]
        require(np.isfinite(output).all(), "nonfinite foundation prediction")
        if self.family == "bernoulli":
            require(((output >= 0) & (output <= 1)).all(), "invalid foundation probability")
        result = torch.as_tensor(output, dtype=a_query.dtype, device=a_query.device)
        require(bool(torch.isfinite(result).all()), "prediction overflows requested dtype")
        return result

    def mean(self, a_query, x_tokens, raw_x, context, group_offset, design=None):
        require(self.task == "outcome", "mean requires outcome comparator")
        require(design is None, "foundation comparator uses raw mmHg; no treatment design override")
        return self._predict(a_query, x_tokens, raw_x, context, group_offset)

    def probability(self, a_query, x_tokens, raw_x, context, group_offset):
        require(self.task == "origin", "origin probability requires origin comparator")
        return self._predict(a_query, x_tokens, raw_x, context, group_offset)

    def logits(self, a_query, x_tokens, raw_x, context, group_offset):
        p = self.probability(a_query, x_tokens, raw_x, context, group_offset)
        require(bool(((p > 0) & (p < 1)).all()), "boundary probability; no implicit clipping")
        return torch.logit(p)
