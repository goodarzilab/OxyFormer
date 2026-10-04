"""Immutable, versioned JSON values and content-addressed local artifacts."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints


class ContractError(ValueError):
    """An artifact cannot safely cross the scientific boundary."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def nonempty(value: str, name: str) -> None:
    require(bool(value.strip()), f"{name} must be nonempty")


def check_hash(value: str, name: str = "hash") -> None:
    require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"invalid {name}")


def unique(values: tuple, name: str) -> None:
    require(len(set(values)) == len(values), f"duplicate {name}")


def _coerce(value, annotation):
    """Copy mutable sequences; reject unknown fields/types and nonfinite data."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, UnionType):
        for choice in args:
            try:
                return _coerce(value, choice)
            except (ContractError, TypeError):
                pass
        raise ContractError(f"value does not match {annotation}")
    if origin is Literal:
        require(any(type(value) is type(x) and value == x for x in args), "invalid enum value")
        return value
    if origin is tuple:
        require(isinstance(value, (list, tuple)), "expected sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(x, args[0]) for x in value)
        require(len(value) == len(args), "wrong tuple length")
        return tuple(_coerce(x, t) for x, t in zip(value, args))
    if is_dataclass(annotation):
        if type(value) is annotation:
            return value
        require(isinstance(value, dict), "expected contract object")
        try:
            return annotation(**value)
        except TypeError as exc:
            raise ContractError(f"invalid {annotation.__name__} fields") from exc
    if annotation is float:
        require(type(value) in (float, int), "expected number")
        require(math.isfinite(value), "nonfinite number")
        return float(value)
    if annotation in (str, int, bool, type(None)):
        require(type(value) is annotation, f"expected {annotation.__name__}")
        return value
    raise ContractError(f"unsupported contract type {annotation}")


def _plain(value):
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_plain(x) for x in value]
    return value


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class Immutable:
    def __post_init__(self):
        hints = get_type_hints(type(self))
        for field in fields(self):
            object.__setattr__(self, field.name, _coerce(getattr(self, field.name), hints[field.name]))

    def to_dict(self) -> dict:
        return {"schema_version": 1, "type": type(self).__name__, "payload": _plain(self)}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str):
        def no_duplicates(pairs):
            result = {}
            for key, value in pairs:
                require(key not in result, f"duplicate JSON key {key}")
                result[key] = value
            return result
        try:
            value = json.loads(text, object_pairs_hook=no_duplicates)
        except (ValueError, TypeError) as exc:
            raise ContractError("invalid JSON") from exc
        require(isinstance(value, dict), "expected envelope")
        require(set(value) == {"schema_version", "type", "payload"}, "invalid envelope fields")
        require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                "unsupported schema version")
        require(value["type"] == cls.__name__, "artifact type mismatch")
        return _coerce(value["payload"], cls)

    @property
    def content_hash(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactLineage(Immutable):
    source_hashes: tuple[str, ...]
    unit_ids: tuple[str, ...]
    parent_hashes: tuple[str, ...]
    split_hash: str | None
    config_hash: str
    model_hash: str | None
    environment: tuple[tuple[str, str], ...]
    seed: int | None
    parameter_count: int | None

    def __post_init__(self):
        Immutable.__post_init__(self)
        require(bool(self.source_hashes), "source hashes required")
        require(bool(self.unit_ids), "unit IDs required")
        for value in self.source_hashes + self.parent_hashes + (self.config_hash,):
            check_hash(value)
        for value in (self.split_hash, self.model_hash):
            if value is not None:
                check_hash(value)
        unique(self.unit_ids, "unit IDs")
        for value in self.unit_ids:
            nonempty(value, "unit ID")
        require(bool(self.environment), "environment required")
        unique(tuple(k for k, _ in self.environment), "environment keys")
        for key, value in self.environment:
            nonempty(key, "environment key")
            nonempty(value, "environment value")
        # Environment is a mapping; its input order has no scientific meaning.
        object.__setattr__(self, "environment", tuple(sorted(self.environment)))
        if self.parameter_count is not None:
            require(self.parameter_count >= 0, "negative parameter count")
        if self.model_hash is not None:
            require(self.parameter_count is not None, "model requires actual parameter count")


def file_hash(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact(path: str | Path, value: Immutable) -> str:
    """Create once. Existing bytes are never silently overwritten."""
    with Path(path).open("xb") as stream:
        stream.write(value.to_json().encode("utf-8"))
    return value.content_hash


def read_artifact(path: str | Path, cls, expected_hash: str):
    check_hash(expected_hash)
    data = Path(path).read_bytes()
    require(sha256(data).hexdigest() == expected_hash, "artifact hash mismatch")
    value = cls.from_json(data.decode("utf-8"))
    require(value.content_hash == expected_hash, "noncanonical artifact encoding")
    return value


def relative_artifact_path(value: str) -> None:
    path = PurePosixPath(value)
    require(bool(value) and not path.is_absolute() and ".." not in path.parts
            and path.as_posix() == value and value != "." and "\\" not in value,
            "artifact path must be a normalized relative path")


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRecord(Immutable):
    path: str
    sha256: str
    lineage: ArtifactLineage
    kind: str

    def __post_init__(self):
        Immutable.__post_init__(self)
        relative_artifact_path(self.path)
        check_hash(self.sha256)
        nonempty(self.kind, "artifact kind")
