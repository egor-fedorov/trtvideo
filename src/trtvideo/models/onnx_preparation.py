"""Evidence linking an original ONNX bundle to a prepare-onnx output."""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

from trtvideo.models.onnx_bundle import (
    OnnxBundleError,
    OnnxIdentity,
    inspect_onnx_bundle,
    parse_onnx_identity,
)


class Preparation(TypedDict):
    schema_version: int
    source: OnnxIdentity
    output: OnnxIdentity
    width: int
    height: int
    scale: int | None
    precision: str


def preparation_path(output: Path) -> Path:
    return Path(f"{output}.prepare.json")


def validate_preparation(value: object, output: OnnxIdentity) -> Preparation:
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise OnnxBundleError("Unsupported ONNX preparation manifest; rerun prepare-onnx")
    if parse_onnx_identity(value.get("output")) != output:
        raise OnnxBundleError(
            "Prepared ONNX does not match its preparation manifest; rerun prepare-onnx"
        )
    w, h, scale, precision = (value.get(k) for k in ("width", "height", "scale", "precision"))
    if (
        type(w) is not int
        or w <= 0
        or type(h) is not int
        or h <= 0
        or (scale is not None and (type(scale) is not int or scale <= 0))
        or precision not in ("fp32", "fp16")
    ):
        raise OnnxBundleError("Invalid ONNX preparation parameters; rerun prepare-onnx")
    return Preparation(
        schema_version=1,
        source=parse_onnx_identity(value.get("source")),
        output=output,
        width=w,
        height=h,
        scale=scale,
        precision=precision,
    )


def read_preparation(model: Path, output: OnnxIdentity) -> Preparation | None:
    path = preparation_path(model)
    if not path.exists() and not path.is_symlink():
        return None  # Direct build: only the model actually consumed can be claimed.
    if not path.is_file():
        raise OnnxBundleError("Preparation manifest must be a regular file; rerun prepare-onnx")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise OnnxBundleError("Unreadable ONNX preparation manifest; rerun prepare-onnx") from exc
    return validate_preparation(value, output)


@contextmanager
def record_preparation(
    source: Path,
    before: OnnxIdentity,
    output: Path,
    *,
    width: int,
    height: int,
    scale: int | None,
    precision: str,
) -> Generator[None, None, None]:
    path = preparation_path(output)
    path.unlink(missing_ok=True)
    yield
    if inspect_onnx_bundle(source).identity() != before:
        raise OnnxBundleError("Source ONNX graph or external weights changed during preparation")
    record = Preparation(
        schema_version=1,
        source=before,
        output=inspect_onnx_bundle(output).identity(),
        width=width,
        height=height,
        scale=scale,
        precision=precision,
    )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
