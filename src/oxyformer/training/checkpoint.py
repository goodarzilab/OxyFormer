"""Immutable, atomic continuation artifacts. No pickle or shared writable cache.

A caller supplies a trusted CheckpointArtifact (or its expected hash when reading
its JSON descriptor). Hashes establish identity, not authenticity on their own.
The archive codec accepts only JSON primitives, containers and numeric tensors;
it never imports classes or executes checkpoint-provided code. The writer and
reader require uncompressed ZIP members, so loading cannot expand compressed data.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import random
import signal
import tempfile
from typing import Literal
import zipfile

import numpy as np
import torch

from oxyformer.provenance import (
    ArtifactLineage, Immutable, canonical_json, check_hash, require,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointIdentity(Immutable):
    training_ids: tuple[str, ...]
    data_hash: str
    split_hash: str
    config_hash: str
    preprocessing_hash: str
    seed: int
    scientific_code_hash: str
    environment: tuple[tuple[str, str], ...]

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(bool(self.training_ids) and len(set(self.training_ids)) == len(self.training_ids),
                "invalid checkpoint training IDs")
        require(self.seed >= 0, "negative seed")
        for digest in (self.data_hash, self.split_hash, self.config_hash,
                       self.preprocessing_hash, self.scientific_code_hash):
            check_hash(digest)
        require(bool(self.environment), "checkpoint environment required")
        require(len(dict(self.environment)) == len(self.environment), "duplicate environment keys")
        object.__setattr__(self, "environment", tuple(sorted(self.environment)))


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointArtifact(Immutable):
    path: str
    sha256: str
    identity: CheckpointIdentity
    lineage: ArtifactLineage
    complete: bool
    reason: Literal["max_epochs", "patience", "slice_limit", "requested", "batch_limit"]
    epoch: int
    step: int
    predecessor_hash: str | None

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(Path(self.path).is_absolute(), "checkpoint path must be absolute")
        check_hash(self.sha256)
        if self.predecessor_hash is not None:
            check_hash(self.predecessor_hash)
        require(self.complete == (self.reason in ("max_epochs", "patience")),
                "completion and stopping reason disagree")
        require(self.epoch >= 0 and self.step >= 0, "negative progress")


class CheckpointRequest:
    """A latched, one-attempt request; handlers perform no checkpoint I/O.

    A requested object stays requested. Resume with a fresh CheckpointRequest
    (or stop_request=None); automatic clearing could discard a genuine signal.
    """

    def __init__(self):
        self.requested = False

    def request(self, signum=None, frame=None):
        self.requested = True

    @contextmanager
    def signals(self):
        # Signal installation deliberately requires the main thread. Callers
        # running a worker thread can request a checkpoint through this object.
        import threading
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGUSR1):
                previous[sig] = signal.signal(sig, self.request)
        try:
            yield self
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def capture_rng(device: torch.device | str = "cpu") -> dict:
    """Capture global CPU RNGs and only the selected training accelerator.

    Unused visible GPUs are not part of this single-device training state.
    A CPU run never acquires a dependency on previously initialized CUDA state.
    """
    device = torch.device(device)
    require(device.type in ("cpu", "cuda"), "unsupported RNG device")
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state: dict, device: torch.device | str = "cpu") -> None:
    device = torch.device(device)
    require(device.type in ("cpu", "cuda"), "unsupported RNG device")
    require((state["cuda"] is not None) == (device.type == "cuda"), "RNG device mismatch")
    if device.type == "cuda":
        require(torch.cuda.is_available(), "checkpoint requires CUDA RNG")
    random.setstate(state["python"])
    kind, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def _encode(value, arrays: list) -> dict:
    if isinstance(value, torch.Tensor):
        require(value.layout == torch.strided, "only dense checkpoint tensors are supported")
        array = value.detach().cpu().numpy()
        require(array.dtype.kind in "biuf", "unsupported checkpoint tensor dtype")
        index = len(arrays)
        arrays.append(array)
        return {"tensor": index}
    if value is None or type(value) in (str, bool, int, float):
        canonical_json(value)  # Refuse nonfinite scalars.
        return {"scalar": value}
    if type(value) in (tuple, list):
        return {"tuple" if type(value) is tuple else "list": [_encode(x, arrays) for x in value]}
    if isinstance(value, dict):
        require(all(type(k) in (str, int) for k in value), "unsupported checkpoint dictionary key")
        return {"dict": [[k, _encode(v, arrays)] for k, v in value.items()]}
    raise ValueError(f"unsupported checkpoint value: {type(value).__name__}")


def _decode(node, archive, used: set):
    require(type(node) is dict and len(node) == 1, "invalid checkpoint node")
    tag, value = next(iter(node.items()))
    if tag == "scalar":
        require(value is None or type(value) in (str, bool, int, float), "invalid scalar")
        canonical_json(value)
        return value
    if tag in ("tuple", "list"):
        require(type(value) is list, "invalid sequence")
        result = [_decode(x, archive, used) for x in value]
        return tuple(result) if tag == "tuple" else result
    if tag == "dict":
        require(type(value) is list, "invalid dictionary")
        result = {}
        for pair in value:
            require(type(pair) is list and len(pair) == 2, "invalid dictionary item")
            key, child = pair
            require(type(key) in (str, int) and key not in result, "invalid/duplicate dictionary key")
            result[key] = _decode(child, archive, used)
        return result
    require(tag == "tensor" and type(value) is int and value >= 0, "invalid tensor node")
    name = f"tensors/{value}.npy"
    require(name not in used, "duplicate tensor reference")
    used.add(name)
    array = np.load(BytesIO(archive.read(name)), allow_pickle=False)
    require(type(array) is np.ndarray and array.dtype.kind in "biuf", "unsafe tensor dtype")
    return torch.from_numpy(array.copy())


def model_state_hash(state: dict) -> str:
    """Bind names, dtypes, shapes and exact tensor bytes to model lineage."""
    require(isinstance(state, dict) and bool(state), "missing checkpoint model state")
    require(all(type(name) is str and isinstance(value, torch.Tensor)
                for name, value in state.items()), "invalid checkpoint model state")
    digest = sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(canonical_json([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _publish(path: Path, data: bytes) -> None:
    """Durable create-once publication on the attempt's own filesystem."""
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        os.link(temporary, path)  # Atomic, refuses an existing destination.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def save_checkpoint(output_dir: Path, *, identity: CheckpointIdentity,
                    lineage: ArtifactLineage, state: dict, complete: bool, reason: str,
                    predecessor: CheckpointArtifact | None = None) -> CheckpointArtifact:
    """Publish one immutable archive and its JSON descriptor in a private root.

    All mutable training components, including nested controller progress, are
    committed in the same archive. A crash before descriptor publication leaves
    an unreferenced archive, never a partially accepted checkpoint.
    """
    require(model_state_hash(state["model"]) == lineage.model_hash, "checkpoint model hash mismatch")
    arrays = []
    progress = state["progress"]
    metadata = {"format": "oxyformer-checkpoint", "version": 1,
                "identity": identity.to_dict(), "lineage": lineage.to_dict(),
                "complete": complete, "reason": reason,
                "predecessor_hash": predecessor.content_hash if predecessor else None,
                "state": _encode(state, arrays)}
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("metadata.json", canonical_json(metadata))
        for index, array in enumerate(arrays):
            item = BytesIO()
            np.save(item, array, allow_pickle=False)
            archive.writestr(f"tensors/{index}.npy", item.getvalue())
    data = stream.getvalue()
    digest = sha256(data).hexdigest()
    output_dir = Path(output_dir).resolve(strict=True)
    path = output_dir / f"checkpoint-{digest}.ofc"
    artifact = CheckpointArtifact(path=str(path), sha256=digest, identity=identity,
                                  lineage=lineage, complete=complete, reason=reason,
                                  epoch=progress["epoch"], step=progress["step"],
                                  predecessor_hash=metadata["predecessor_hash"])
    _publish(path, data)
    _publish(path.with_suffix(".json"), artifact.to_json().encode())
    return artifact


def load_checkpoint(artifact: CheckpointArtifact, expected_identity: CheckpointIdentity) -> dict:
    """Verify the trusted descriptor and exact science identity before decoding."""
    require(type(artifact) is CheckpointArtifact, "trusted CheckpointArtifact required")
    require(artifact.identity == expected_identity, "incompatible checkpoint identity")
    data = Path(artifact.path).read_bytes()
    require(sha256(data).hexdigest() == artifact.sha256, "checkpoint file hash mismatch")
    require(zipfile.is_zipfile(BytesIO(data)), "untrusted checkpoint serialization")
    with zipfile.ZipFile(BytesIO(data)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate checkpoint members")
        require(all(info.compress_type == zipfile.ZIP_STORED and
                    info.compress_size == info.file_size for info in archive.infolist()),
                "compressed checkpoint members are unsupported")
        metadata_bytes = archive.read("metadata.json")
        metadata = json.loads(metadata_bytes)
        require(canonical_json(metadata).encode() == metadata_bytes, "noncanonical checkpoint metadata")
        require(set(metadata) == {"format", "version", "identity", "lineage", "complete",
                                  "reason", "predecessor_hash", "state"}, "unknown checkpoint metadata")
        require(metadata["format"] == "oxyformer-checkpoint" and metadata["version"] == 1,
                "untrusted checkpoint serialization")
        for field in ("identity", "lineage"):
            require(metadata[field] == getattr(artifact, field).to_dict(), f"checkpoint {field} mismatch")
        for field in ("complete", "reason", "predecessor_hash"):
            require(metadata[field] == getattr(artifact, field), f"checkpoint {field} mismatch")
        used = {"metadata.json"}
        state = _decode(metadata["state"], archive, used)
        require(set(names) == used, "unexpected checkpoint members")
    require(type(state) is dict and "progress" in state, "missing checkpoint progress")
    require(model_state_hash(state["model"]) == artifact.lineage.model_hash,
            "checkpoint model hash mismatch")
    require(state["progress"]["epoch"] == artifact.epoch and state["progress"]["step"] == artifact.step,
            "checkpoint progress mismatch")
    return state
