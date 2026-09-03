"""Resumable orchestration for end-to-end model compatibility evidence."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from trtvideo.compatibility.evidence import (
    CompatibilityEvidenceError,
    file_identity,
    public_value,
    sha256_file,
)
from trtvideo.compatibility.input import (
    DEFAULT_INPUT_FRAMES,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    FIXTURE_CONTRACT_VERSION,
    VIDEO_SHA256,
    VIDEO_SIZE_BYTES,
    CompatibilityInputError,
    probe_video_size,
)
from trtvideo.diagnostics.doctor import DoctorReport, run_doctor

WORKFLOW_SCHEMA_VERSION = 1
HEARTBEAT_INTERVAL_SEC = 30.0
MODEL_TOOLS_IMAGE = "ghcr.io/egor-fedorov/trtvideo-model-tools"
SourceFormat = Literal["checkpoint", "onnx"]
_SUPPORTED_ONNX_IO_TYPES = {1, 10}  # TensorProto.FLOAT and TensorProto.FLOAT16.


class CompatibilityWorkflowError(RuntimeError):
    """Raised when compatibility orchestration cannot proceed safely."""


@dataclass(frozen=True)
class CompatibilityOptions:
    """User-visible compatibility-check inputs."""

    source_format: SourceFormat
    source_artifact: Path
    model_name: str
    model_source: str
    model_license: str
    output_dir: Path
    input_video: Path | None = None
    scale: int | None = None
    gpu_id: int = 0
    resume: bool = False
    dry_run: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class OnnxContract:
    """Planning-relevant ONNX tensor shape facts."""

    dynamic: bool
    input_name: str
    input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]
    scale: int | None


@dataclass(frozen=True)
class WorkflowPaths:
    """Generated files contained by one compatibility workspace."""

    root: Path
    state: Path
    commands: Path
    source_video: Path
    input_video: Path
    input_manifest: Path
    onnx_dir: Path
    fp32_onnx: Path
    prepared_onnx: Path
    conformance: Path
    engine: Path
    engine_manifest: Path
    timing_cache: Path
    output_video: Path
    report_json: Path
    report_markdown: Path

    @classmethod
    def under(
        cls,
        root: Path,
        *,
        width: int,
        height: int,
        source_onnx: Path | None,
    ) -> WorkflowPaths:
        variant = _variant_name(width, height)
        onnx_dir = root / "onnx"
        fp32_onnx = onnx_dir / f"model_{variant}.onnx"
        if source_onnx is None:
            prepared_onnx = onnx_dir / f"model_{variant}_fp16.onnx"
        else:
            prepared_onnx = onnx_dir / f"{source_onnx.stem}_{variant}_fp16.onnx"
        engine = root / "engine" / "model.engine"
        return cls(
            root=root,
            state=root / "compatibility.state.json",
            commands=root / "commands.txt",
            source_video=root / "input" / "Jacqueville-beach-2026.webm",
            input_video=root / "input" / "compatibility-input.mp4",
            input_manifest=root / "input" / "compatibility-input.json",
            onnx_dir=onnx_dir,
            fp32_onnx=fp32_onnx,
            prepared_onnx=prepared_onnx,
            conformance=onnx_dir / "model.export-conformance.json",
            engine=engine,
            engine_manifest=Path(f"{engine}.json"),
            timing_cache=root / "engine" / "trt.cache",
            output_video=root / "media" / "compatibility-output.mp4",
            report_json=root / "model-compatibility-report.json",
            report_markdown=root / "model-compatibility-issue.md",
        )


@dataclass(frozen=True)
class WorkflowStep:
    """One independently executable and verifiable low-level operation."""

    key: str
    command_name: str
    command: tuple[str, ...]
    duration_hint: str | None = None
    outputs: tuple[Path, ...] = ()
    preserve_outputs_on_failure: bool = False


@dataclass(frozen=True)
class CompatibilityPlan:
    """Resolved commands and contracts for one compatibility run."""

    options: CompatibilityOptions
    paths: WorkflowPaths
    width: int
    height: int
    scale: int | None
    onnx_contract: OnnxContract | None
    steps: tuple[WorkflowStep, ...]


def _variant_name(width: int, height: int) -> str:
    if (width, height) == (1280, 720):
        return "720p"
    if (width, height) == (1920, 1080):
        return "1080p"
    return f"{width}x{height}"


def _onnx_dimension(dimension: Any) -> int | None:
    if getattr(dimension, "dim_param", ""):
        return None
    value = int(getattr(dimension, "dim_value", 0))
    return value if value > 0 else None


def _static_spatial(shape: tuple[int | None, ...]) -> tuple[int, int]:
    height, width = shape[2], shape[3]
    if height is None or width is None:
        raise CompatibilityWorkflowError("Expected a static ONNX spatial shape")
    return height, width


def inspect_onnx_contract(path: Path) -> OnnxContract:
    """Inspect one ONNX graph without initializing TensorRT or a GPU."""
    if not path.is_file():
        raise CompatibilityWorkflowError(f"ONNX source does not exist: {path}")
    try:
        import onnx
    except ImportError as exc:
        raise CompatibilityWorkflowError(
            "ONNX inspection requires a published trtvideo image"
        ) from exc
    try:
        model = onnx.load(path, load_external_data=False)
    except Exception as exc:
        raise CompatibilityWorkflowError(f"Cannot load ONNX source: {path.name}") from exc
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise CompatibilityWorkflowError("ONNX must contain exactly one input and one output")
    input_tensor = model.graph.input[0]
    output_tensor = model.graph.output[0]
    for label, tensor in (("input", input_tensor), ("output", output_tensor)):
        tensor_type = getattr(getattr(tensor, "type", None), "tensor_type", None)
        element_type = int(getattr(tensor_type, "elem_type", 0))
        if element_type not in _SUPPORTED_ONNX_IO_TYPES:
            raise CompatibilityWorkflowError(
                f"ONNX {label} must use FP32 or FP16 bindings, got element type {element_type}"
            )
    input_shape = tuple(
        _onnx_dimension(dimension) for dimension in input_tensor.type.tensor_type.shape.dim
    )
    output_shape = tuple(
        _onnx_dimension(dimension) for dimension in output_tensor.type.tensor_type.shape.dim
    )
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise CompatibilityWorkflowError("ONNX input and output must both be rank-4 NCHW tensors")
    for label, shape in (("input", input_shape), ("output", output_shape)):
        if shape[0] not in {None, 1} or shape[1] not in {None, 3}:
            raise CompatibilityWorkflowError(
                f"ONNX {label} must use batch 1 and three RGB channels, got {shape}"
            )
    fully_static = all(dimension is not None for dimension in (*input_shape, *output_shape))
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata_scale: int | None = None
    raw_scale = metadata.get("trtvideo.upscale_scale")
    if raw_scale is not None:
        try:
            metadata_scale = int(raw_scale)
        except ValueError as exc:
            raise CompatibilityWorkflowError(
                "ONNX trtvideo.upscale_scale metadata is invalid"
            ) from exc
        if metadata_scale <= 0:
            raise CompatibilityWorkflowError("ONNX scale metadata must be positive")
    inferred_scale = metadata_scale
    if fully_static:
        input_h, input_w = _static_spatial(input_shape)
        output_h, output_w = _static_spatial(output_shape)
        if output_h % input_h or output_w % input_w:
            raise CompatibilityWorkflowError("ONNX output is not an integer spatial upscale")
        height_scale = output_h // input_h
        width_scale = output_w // input_w
        if height_scale != width_scale or height_scale <= 0:
            raise CompatibilityWorkflowError("ONNX must use one uniform positive spatial scale")
        if metadata_scale is not None and metadata_scale != height_scale:
            raise CompatibilityWorkflowError("ONNX shape and scale metadata disagree")
        inferred_scale = height_scale
    elif all(dimension is not None for dimension in input_shape):
        raise CompatibilityWorkflowError(
            "Partially dynamic ONNX output is unsupported; use a consistently dynamic graph"
        )
    return OnnxContract(
        dynamic=not fully_static,
        input_name=input_tensor.name,
        input_shape=input_shape,
        output_shape=output_shape,
        scale=inferred_scale,
    )


def _append_verbose(command: list[str], enabled: bool) -> tuple[str, ...]:
    if enabled:
        command.append("--verbose")
    return tuple(command)


def _resolve_dimensions_and_scale(
    options: CompatibilityOptions,
    contract: OnnxContract | None,
    *,
    video_probe: Callable[[Path], tuple[int, int]],
) -> tuple[int, int, int | None]:
    try:
        custom_size = video_probe(options.input_video) if options.input_video is not None else None
    except CompatibilityInputError as exc:
        raise CompatibilityWorkflowError(str(exc)) from exc
    if options.source_format == "checkpoint":
        width, height = custom_size or (DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT)
        return width, height, None
    assert contract is not None
    scale = contract.scale
    if options.scale is not None:
        if options.scale <= 0:
            raise CompatibilityWorkflowError("--scale must be a positive integer")
        if scale is not None and options.scale != scale:
            raise CompatibilityWorkflowError(
                f"--scale {options.scale} conflicts with the ONNX contract ({scale}x)"
            )
        scale = options.scale
    if contract.dynamic:
        if scale is None:
            raise CompatibilityWorkflowError(
                "Dynamic ONNX scale cannot be inferred; pass --scale explicitly"
            )
        width, height = custom_size or (DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT)
        return width, height, scale
    height, width = _static_spatial(contract.input_shape)
    if custom_size is not None and custom_size != (width, height):
        raise CompatibilityWorkflowError(
            "Custom input resolution does not match static ONNX input: "
            f"video={custom_size[0]}x{custom_size[1]}, ONNX={width}x{height}"
        )
    return width, height, scale


def build_plan(
    options: CompatibilityOptions,
    *,
    onnx_inspector: Callable[[Path], OnnxContract] = inspect_onnx_contract,
    video_probe: Callable[[Path], tuple[int, int]] = probe_video_size,
) -> CompatibilityPlan:
    """Resolve a low-level command plan without creating the output workspace."""
    if options.gpu_id < 0:
        raise CompatibilityWorkflowError("GPU id must be non-negative")
    if options.source_format not in {"checkpoint", "onnx"}:
        raise CompatibilityWorkflowError("Source format must be checkpoint or onnx")
    if options.scale is not None and options.scale <= 0:
        raise CompatibilityWorkflowError("--scale must be a positive integer")
    if options.source_format == "checkpoint" and options.scale is not None:
        raise CompatibilityWorkflowError(
            "--scale is only needed for ONNX; checkpoint scale is inferred from the model"
        )
    try:
        public_value(options.model_name, "Model name")
        public_value(options.model_license, "Model license")
        model_source = public_value(options.model_source, "Model source")
    except CompatibilityEvidenceError as exc:
        raise CompatibilityWorkflowError(str(exc)) from exc
    try:
        parsed_source = urlsplit(model_source)
    except ValueError as exc:
        raise CompatibilityWorkflowError("Model source must be a valid public URL") from exc
    if (
        parsed_source.scheme not in {"http", "https"}
        or not parsed_source.netloc
        or parsed_source.username is not None
        or parsed_source.password is not None
    ):
        raise CompatibilityWorkflowError("Model source must be a public HTTP(S) URL")
    if not options.source_artifact.is_file():
        raise CompatibilityWorkflowError(f"Model source does not exist: {options.source_artifact}")
    contract = onnx_inspector(options.source_artifact) if options.source_format == "onnx" else None
    width, height, scale = _resolve_dimensions_and_scale(
        options,
        contract,
        video_probe=video_probe,
    )
    if width % 2 or height % 2:
        raise CompatibilityWorkflowError(
            f"Input dimensions must be even for the NV12 media contract, got {width}x{height}"
        )
    paths = WorkflowPaths.under(
        options.output_dir,
        width=width,
        height=height,
        source_onnx=options.source_artifact if contract and contract.dynamic else None,
    )
    input_command = [
        "prepare-compatibility-input",
        "--output",
        str(paths.input_video),
        "--manifest",
        str(paths.input_manifest),
        "--size",
        f"{width}x{height}",
        "--frames",
        str(DEFAULT_INPUT_FRAMES),
    ]
    input_outputs = [paths.input_video, paths.input_manifest]
    if options.input_video is None:
        input_command.extend(("--source-cache", str(paths.source_video)))
    else:
        input_command.extend(("--input", str(options.input_video)))

    steps = [
        WorkflowStep(
            key="doctor",
            command_name="trtvideo doctor",
            command=(
                "trtvideo",
                "doctor",
                "--gpu-id",
                str(options.gpu_id),
                "--disk-path",
                str(options.output_dir.parent),
            ),
        ),
        WorkflowStep(
            key="input",
            command_name="prepare-compatibility-input",
            command=tuple(input_command),
            outputs=tuple(input_outputs),
        ),
    ]
    source_format = options.source_format
    conformance: Path | None = None
    if source_format == "checkpoint":
        export_command = [
            "export-onnx",
            "--model_path",
            str(options.source_artifact),
            "--output_dir",
            str(paths.onnx_dir),
            "--name",
            "model",
            "--size",
            f"{width}x{height}",
            "--conformance-report",
            str(paths.conformance),
        ]
        steps.append(
            WorkflowStep(
                key="export",
                command_name="export-onnx",
                command=_append_verbose(export_command, options.verbose),
                outputs=(paths.fp32_onnx, paths.conformance),
            )
        )
        prepare_source = paths.fp32_onnx
        build_source = paths.prepared_onnx
        conformance = paths.conformance
        needs_prepare = True
    else:
        assert contract is not None
        prepare_source = options.source_artifact
        build_source = paths.prepared_onnx if contract.dynamic else options.source_artifact
        needs_prepare = contract.dynamic

    if needs_prepare:
        assert scale is not None or source_format == "checkpoint"
        prepare_command = [
            "prepare-onnx",
            str(prepare_source),
            "--output_dir",
            str(paths.onnx_dir),
            "--size",
            f"{width}x{height}",
            "--precision",
            "fp16",
        ]
        if source_format == "onnx":
            prepare_command.extend(("--scale", str(scale)))
        steps.append(
            WorkflowStep(
                key="prepare",
                command_name="prepare-onnx",
                command=_append_verbose(prepare_command, options.verbose),
                outputs=(paths.prepared_onnx,),
            )
        )

    build_command = [
        "env",
        f"CUDA_VISIBLE_DEVICES={options.gpu_id}",
        "build-engine",
        str(build_source),
        "--output",
        str(paths.engine),
        "--manifest",
        str(paths.engine_manifest),
        "--timing-cache",
        str(paths.timing_cache),
    ]
    steps.append(
        WorkflowStep(
            key="engine",
            command_name="build-engine",
            duration_hint="model/GPU-dependent, typically 5-15 minutes",
            command=_append_verbose(build_command, options.verbose),
            outputs=(paths.engine, paths.engine_manifest),
        )
    )
    process_command = [
        "trtvideo",
        "--gpu-id",
        str(options.gpu_id),
        "--engine",
        str(paths.engine),
        "--input",
        str(paths.input_video),
        "--output",
        str(paths.output_video),
        "--max-frames",
        str(DEFAULT_INPUT_FRAMES),
        "--log-interval",
        "24",
    ]
    steps.append(
        WorkflowStep(
            key="smoke",
            command_name="trtvideo",
            command=_append_verbose(process_command, options.verbose),
            outputs=(paths.output_video,),
        )
    )
    report_command = [
        "trtvideo",
        "compatibility-report",
        "--model-name",
        options.model_name,
        "--model-source",
        options.model_source,
        "--model-license",
        options.model_license,
        "--source-format",
        source_format,
        "--source-artifact",
        str(options.source_artifact),
        "--engine",
        str(paths.engine),
        "--input",
        str(paths.input_video),
        "--input-manifest",
        str(paths.input_manifest),
        "--processed-output",
        str(paths.output_video),
        "--expected-frames",
        str(DEFAULT_INPUT_FRAMES),
        "--commands-file",
        str(paths.commands),
        "--image-reference",
        os.environ.get("TRTVIDEO_IMAGE_REF", "unknown"),
        "--gpu-id",
        str(options.gpu_id),
        "--output-dir",
        str(paths.root),
    ]
    if conformance is not None:
        report_command.extend(("--export-conformance", str(conformance)))
    steps.append(
        WorkflowStep(
            key="report",
            command_name="trtvideo compatibility-report",
            command=tuple(report_command),
            outputs=(paths.report_json, paths.report_markdown),
            preserve_outputs_on_failure=True,
        )
    )
    return CompatibilityPlan(
        options=options,
        paths=paths,
        width=width,
        height=height,
        scale=scale,
        onnx_contract=contract,
        steps=tuple(steps),
    )


def _package_version() -> str:
    try:
        return version("trtvideo")
    except PackageNotFoundError:
        return "unknown"


def _doctor_fingerprint(report: DoctorReport) -> dict[str, str]:
    wanted = {
        "GPU",
        "Driver",
        "CUDA",
        "TensorRT",
        "NVDEC",
        "NVENC",
        "CV-CUDA",
        "PyNvVideoCodec",
    }
    fingerprint = {
        check.component: check.detail
        for check in report.checks
        if check.component in wanted and check.passed
    }
    vram = next(
        (check.detail for check in report.checks if check.component == "VRAM" and check.passed),
        None,
    )
    if vram is not None:
        _free, separator, total = vram.partition(" free of ")
        fingerprint["VRAM"] = total if separator else vram
    return fingerprint


def workflow_context(plan: CompatibilityPlan, doctor: DoctorReport) -> dict[str, Any]:
    """Bind resume evidence to source, options, image, and stable GPU facts."""
    options = plan.options
    input_context: dict[str, Any]
    if options.input_video is None:
        input_context = {
            "kind": "pinned-live-action",
            "fixture_contract": FIXTURE_CONTRACT_VERSION,
            "source_sha256": VIDEO_SHA256,
            "source_size_bytes": VIDEO_SIZE_BYTES,
        }
    else:
        input_context = {"kind": "user-supplied", "identity": file_identity(options.input_video)}
    return {
        "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
        "package_version": _package_version(),
        "image": {
            "reference": os.environ.get("TRTVIDEO_IMAGE_REF", "unknown"),
            "revision": os.environ.get("TRTVIDEO_BUILD_REVISION", "unknown"),
            "base": os.environ.get("TRTVIDEO_BASE_IMAGE", "unknown"),
            "dirty": os.environ.get("TRTVIDEO_BUILD_DIRTY", "unknown"),
        },
        "source_format": options.source_format,
        "source": file_identity(options.source_artifact),
        "model": {
            "name": options.model_name,
            "source": options.model_source,
            "license": options.model_license,
        },
        "input": input_context,
        "parameters": {
            "width": plan.width,
            "height": plan.height,
            "frames": DEFAULT_INPUT_FRAMES,
            "scale": plan.scale,
            "gpu_id": options.gpu_id,
            "verbose": options.verbose,
        },
        "gpu": _doctor_fingerprint(doctor),
        "commands": [list(step.command) for step in plan.steps],
    }


class WorkflowState:
    """Atomic successful-step journal with artifact identities."""

    def __init__(self, path: Path, context: dict[str, Any], completed: list[dict[str, Any]]):
        self.path = path
        self.context = context
        self.completed = completed

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        context: dict[str, Any],
        resume: bool,
    ) -> WorkflowState:
        root = path.parent
        if not resume:
            if root.exists() and any(root.iterdir()):
                raise CompatibilityWorkflowError(
                    f"Output directory is not empty; use --resume or choose another path: {root}"
                )
            root.mkdir(parents=True, exist_ok=True)
            state = cls(path, context, [])
            state._write()
            return state
        if not path.is_file():
            raise CompatibilityWorkflowError(f"Resume state does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CompatibilityWorkflowError(f"Resume state is invalid: {path}") from exc
        if payload.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
            raise CompatibilityWorkflowError("Resume state schema is unsupported")
        if payload.get("context") != context:
            raise CompatibilityWorkflowError(
                "Resume state does not match the current source, image, GPU, or options"
            )
        completed = payload.get("completed_steps")
        if not isinstance(completed, list) or any(not isinstance(item, dict) for item in completed):
            raise CompatibilityWorkflowError("Resume state has an invalid completed-step journal")
        return cls(path, context, completed)

    @property
    def completed_keys(self) -> tuple[str, ...]:
        return tuple(str(item.get("key")) for item in self.completed)

    def _write(self) -> None:
        payload = {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "context": self.context,
            "completed_steps": self.completed,
        }
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def complete(self, step: WorkflowStep, root: Path) -> None:
        artifacts = []
        for output in step.outputs:
            if not output.is_file():
                raise CompatibilityWorkflowError(
                    f"{step.command_name} did not produce expected artifact: {output}"
                )
            artifacts.append(
                {
                    "path": output.relative_to(root).as_posix(),
                    "sha256": sha256_file(output),
                    "size_bytes": output.stat().st_size,
                }
            )
        self.completed.append(
            {
                "key": step.key,
                "completed_at": datetime.now(UTC).isoformat(),
                "artifacts": artifacts,
            }
        )
        self._write()

    def reconcile(self, steps: Sequence[WorkflowStep], root: Path) -> None:
        """Invalidate changed generated evidence and every dependent step."""
        expected_keys = [step.key for step in steps]
        completed_keys = list(self.completed_keys)
        if completed_keys != expected_keys[: len(completed_keys)]:
            raise CompatibilityWorkflowError(
                "Resume state steps are not a prefix of the current plan"
            )
        invalid_index: int | None = None
        for index, record in enumerate(self.completed):
            artifacts = record.get("artifacts")
            if not isinstance(artifacts, list):
                invalid_index = index
                break
            for artifact in artifacts:
                if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                    invalid_index = index
                    break
                path = root / artifact["path"]
                if (
                    not path.is_file()
                    or path.stat().st_size != artifact.get("size_bytes")
                    or sha256_file(path) != artifact.get("sha256")
                ):
                    invalid_index = index
                    break
            if invalid_index is not None:
                break
        if invalid_index is None:
            return
        print(
            f"Resume artifact changed; rerunning from {steps[invalid_index].command_name}.",
            file=sys.stderr,
            flush=True,
        )
        for step in steps[invalid_index:]:
            _clean_outputs(step, root)
        self.completed = self.completed[:invalid_index]
        self._write()


def _clean_outputs(step: WorkflowStep, root: Path) -> None:
    resolved_root = root.resolve()
    for output in step.outputs:
        try:
            output.relative_to(root)
        except ValueError as exc:
            raise CompatibilityWorkflowError(
                f"Refusing to clean an artifact outside the output directory: {output}"
            ) from exc
        try:
            output.parent.resolve().relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise CompatibilityWorkflowError(
                f"Refusing to clean an artifact through an unsafe path: {output}"
            ) from exc
        if output.is_dir():
            raise CompatibilityWorkflowError(f"Refusing to recursively clean directory: {output}")
        output.unlink(missing_ok=True)


def run_streaming_command(
    command: Sequence[str],
    *,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SEC,
    clock: Callable[[], float] = time.monotonic,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    emit: Callable[[str], None] | None = None,
) -> None:
    """Run a child with inherited streams and append-only elapsed heartbeats."""
    output = emit or (lambda line: print(line, file=sys.stderr, flush=True))
    started = clock()
    try:
        process = popen_factory(tuple(command))
    except OSError as exc:
        raise CompatibilityWorkflowError(f"Cannot start command: {command[0]}") from exc
    while True:
        try:
            returncode = process.wait(timeout=heartbeat_interval)
        except subprocess.TimeoutExpired:
            output(f"  still running; elapsed {clock() - started:.0f}s")
            continue
        if returncode != 0:
            raise CompatibilityWorkflowError(
                f"Command failed with code {returncode}: {shlex.join(command)}"
            )
        return


def _progress_bar(index: int, total: int, width: int = 20) -> str:
    complete = round(width * index / total)
    return "[" + "=" * complete + "." * (width - complete) + "]"


def _print_step(step: WorkflowStep, index: int, total: int) -> None:
    duration = f" ({step.duration_hint})" if step.duration_hint else ""
    print(
        f"{_progress_bar(index, total)} [{index}/{total}] {step.command_name}{duration}",
        file=sys.stderr,
        flush=True,
    )
    print(f"  $ {shlex.join(step.command)}", file=sys.stderr, flush=True)


def _write_commands(plan: CompatibilityPlan) -> None:
    commands = [step.command for step in plan.steps if step.key != "report"]
    content = "\n".join(shlex.join(command) for command in commands) + "\n"
    plan.paths.commands.write_text(content, encoding="utf-8")


def _run_doctor_step(
    plan: CompatibilityPlan,
    *,
    doctor_runner: Callable[..., DoctorReport],
) -> DoctorReport:
    from trtvideo.cli.doctor import render_report

    step = plan.steps[0]
    _print_step(step, 1, len(plan.steps))
    started = time.monotonic()
    report = doctor_runner(gpu_id=plan.options.gpu_id, disk_path=plan.paths.root.parent)
    print(render_report(report), file=sys.stderr, flush=True)
    if not report.ready:
        raise CompatibilityWorkflowError("trtvideo doctor did not pass")
    print(
        f"  completed in {time.monotonic() - started:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    return report


def print_dry_run(plan: CompatibilityPlan) -> None:
    """Print the complete immutable plan without touching output or GPU state."""
    print(
        "Compatibility check plan; full execution typically takes 10-30 minutes "
        "depending on the model and GPU.",
        file=sys.stderr,
    )
    for index, step in enumerate(plan.steps, start=1):
        _print_step(step, index, len(plan.steps))
    print("Dry run complete; no files were created and no GPU checks ran.", file=sys.stderr)


def run_workflow(
    plan: CompatibilityPlan,
    *,
    doctor_runner: Callable[..., DoctorReport] = run_doctor,
    executor: Callable[[Sequence[str]], None] = run_streaming_command,
) -> WorkflowPaths:
    """Execute a compatibility plan with strict resumability."""
    if plan.options.dry_run:
        print_dry_run(plan)
        return plan.paths
    print(
        "Compatibility check started; full execution typically takes 10-30 minutes "
        "depending on the model and GPU.",
        file=sys.stderr,
        flush=True,
    )
    workflow_started = time.monotonic()
    doctor = _run_doctor_step(plan, doctor_runner=doctor_runner)
    context = workflow_context(plan, doctor)
    state = WorkflowState.open(
        plan.paths.state,
        context=context,
        resume=plan.options.resume,
    )
    remaining_steps = plan.steps[1:]
    state.reconcile(remaining_steps, plan.paths.root)
    _write_commands(plan)
    completed = set(state.completed_keys)
    for index, step in enumerate(remaining_steps, start=2):
        if step.key in completed:
            print(
                f"{_progress_bar(index, len(plan.steps))} [{index}/{len(plan.steps)}] "
                f"SKIP {step.command_name}",
                file=sys.stderr,
                flush=True,
            )
            continue
        _print_step(step, index, len(plan.steps))
        started = time.monotonic()
        _clean_outputs(step, plan.paths.root)
        for output in step.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
        try:
            executor(step.command)
            state.complete(step, plan.paths.root)
        except Exception:
            if not step.preserve_outputs_on_failure:
                _clean_outputs(step, plan.paths.root)
            raise
        print(
            f"  completed in {time.monotonic() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"Compatibility check complete in {time.monotonic() - workflow_started:.1f}s: "
        f"{plan.paths.report_json}",
        file=sys.stderr,
        flush=True,
    )
    return plan.paths
