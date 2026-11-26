"""Content identity of an ONNX graph and the local external data it references."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypedDict


class OnnxBundleError(ValueError):
    """The graph or one of its referenced files cannot be identified safely."""


@dataclass(frozen=True)
class ExternalFile:
    location: str
    sha256: str
    size_bytes: int


class OnnxIdentity(TypedDict):
    schema_version: int
    graph_sha256: str
    graph_size_bytes: int
    external_file_count: int
    bundle_sha256: str


@dataclass(frozen=True)
class OnnxBundle:
    graph_sha256: str
    graph_size_bytes: int
    external_files: tuple[ExternalFile, ...]

    def record(self) -> dict[str, Any]:
        """Full local journal entry, independent of the installation directory."""
        return {
            "schema_version": 1,
            "graph_sha256": self.graph_sha256,
            "graph_size_bytes": self.graph_size_bytes,
            "external_files": [asdict(item) for item in self.external_files],
        }

    def identity(self) -> OnnxIdentity:
        """Compact, path-free identity suitable for a public compatibility report."""
        canonical = json.dumps(self.record(), sort_keys=True, separators=(",", ":"))
        return {
            "schema_version": 1,
            "graph_sha256": self.graph_sha256,
            "graph_size_bytes": self.graph_size_bytes,
            "external_file_count": len(self.external_files),
            "bundle_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        }


def parse_onnx_identity(value: object) -> OnnxIdentity:
    """Read only the public fields; never fall back to graph-only evidence on malformed data."""
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        raise OnnxBundleError("Invalid ONNX bundle identity")
    graph, digest = value.get("graph_sha256"), value.get("bundle_sha256")
    size, count = value.get("graph_size_bytes"), value.get("external_file_count")
    if (
        value["schema_version"] != 1
        or not isinstance(graph, str)
        or not re.fullmatch(r"[0-9a-f]{64}", graph)
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or type(size) is not int
        or size <= 0
        or type(count) is not int
        or count < 0
    ):
        raise OnnxBundleError("Invalid ONNX bundle identity")
    if count == 0 and OnnxBundle(graph, size, ()).identity()["bundle_sha256"] != digest:
        raise OnnxBundleError("Single-file ONNX bundle digest does not match its graph")
    return OnnxIdentity(
        schema_version=1,
        graph_sha256=graph,
        graph_size_bytes=size,
        external_file_count=count,
        bundle_sha256=digest,
    )


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _external_locations(message: Any, tensor_type: Any) -> Iterator[str]:
    # Visit every TensorProto, including tensor attributes, sparse tensors and subgraphs.
    if isinstance(message, tensor_type):
        if message.data_location == tensor_type.EXTERNAL:
            locations = [item.value for item in message.external_data if item.key == "location"]
            if len(locations) != 1 or not locations[0]:
                raise OnnxBundleError("ONNX external tensor must declare exactly one location")
            if any(item.key == "basepath" and item.value for item in message.external_data):
                raise OnnxBundleError("ONNX external data cannot override the model directory")
            yield locations[0]
        elif message.external_data:
            raise OnnxBundleError("ONNX external data has no external tensor location type")
        return
    for field, value in message.ListFields():
        if field.message_type is not None:
            if field.is_repeated:
                for child in value:
                    yield from _external_locations(child, tensor_type)
            else:
                yield from _external_locations(value, tensor_type)


def _relative_location(location: str) -> str:
    path = PurePosixPath(location)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in location
        or "\x00" in location
        or "://" in location
        or PureWindowsPath(location).drive
    ):
        raise OnnxBundleError("ONNX external data must be relative to the model directory")
    return path.as_posix()


def inspect_onnx_bundle(path: Path) -> OnnxBundle:
    """Hash whole referenced files once, regardless of tensor offsets or lengths."""
    import onnx

    if not path.is_file():
        raise OnnxBundleError("ONNX graph must be an existing regular file")
    try:
        graph = path.read_bytes()
        size = len(graph)
        digest = hashlib.sha256(graph).hexdigest()
        # Parse exactly the bytes hashed above, without resolving external tensor data.
        model = onnx.load_model_from_string(graph)
        del graph
    except Exception as exc:
        raise OnnxBundleError("Cannot read ONNX graph to identify its external data") from exc
    locations = {_relative_location(loc) for loc in _external_locations(model, onnx.TensorProto)}
    del model
    files = []
    try:
        root = path.parent.resolve()
        for location in sorted(locations):
            external = (root / location).resolve()
            if not external.is_relative_to(root) or not external.is_file():
                raise OnnxBundleError(
                    "ONNX external data must be an existing regular file inside the model directory"
                )
            files.append(ExternalFile(location, sha256_file(external), external.stat().st_size))
        return OnnxBundle(digest, size, tuple(files))
    except (OSError, RuntimeError) as exc:
        raise OnnxBundleError("Cannot read ONNX graph or its referenced external data") from exc
