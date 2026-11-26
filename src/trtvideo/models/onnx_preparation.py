"""Evidence linking an original ONNX bundle to a prepare-onnx output."""

from __future__ import annotations

import json
import re
from collections.abc import Generator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import NotRequired, TypedDict

from trtvideo.models.onnx_bundle import (
    OnnxBundleError,
    OnnxIdentity,
    inspect_onnx_bundle,
    parse_onnx_identity,
)

KEEP_IO_TYPES_FOR_FP16 = True


class Converter(TypedDict):
    policy_version: int
    onnx: str
    onnxconverter_common: str
    keep_io_types: bool


def current_fp16_converter() -> Converter:
    try:
        return Converter(
            policy_version=1,
            onnx=version("onnx"),
            onnxconverter_common=version("onnxconverter-common"),
            keep_io_types=KEEP_IO_TYPES_FOR_FP16,
        )
    except PackageNotFoundError as exc:
        raise OnnxBundleError("FP16 preparation requires the model-tools image") from exc


def _read_converter(value: object) -> Converter:
    if not isinstance(value, dict):
        raise OnnxBundleError("Invalid FP16 converter identity; rerun prepare-onnx")
    policy, keep = value.get("policy_version"), value.get("keep_io_types")
    onnx, common = value.get("onnx"), value.get("onnxconverter_common")
    if (
        type(policy) is not int
        or policy < 1
        or type(keep) is not bool
        or not isinstance(onnx, str)
        or not isinstance(common, str)
        or not all(re.fullmatch(r"[0-9][0-9A-Za-z.+_!-]{0,79}", v) for v in (onnx, common))
    ):
        raise OnnxBundleError("Invalid FP16 converter identity; rerun prepare-onnx")
    return Converter(
        policy_version=policy, keep_io_types=keep, onnx=onnx, onnxconverter_common=common
    )


class Preparation(TypedDict):
    schema_version: int
    source: OnnxIdentity
    output: OnnxIdentity
    width: int
    height: int
    scale: int | None
    precision: str
    converter: NotRequired[Converter]


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
    record = Preparation(
        schema_version=1,
        source=parse_onnx_identity(value.get("source")),
        output=output,
        width=w,
        height=h,
        scale=scale,
        precision=precision,
    )
    if "converter" in value:
        record["converter"] = _read_converter(value["converter"])
    return record


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
    if precision == "fp16":
        record["converter"] = current_fp16_converter()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
