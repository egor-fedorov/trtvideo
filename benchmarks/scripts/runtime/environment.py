"""Privacy-safe benchmark environment and asset metadata collection."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Hash a file without loading large benchmark assets into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def relative_artifact_path(path: Path, root: Path) -> str:
    """Return a repository-relative path without leaking an absolute host path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"external/{path.name}"


def sanitize_command(command: list[str], root: Path) -> list[str]:
    """Normalize path arguments while preserving an executable command contract."""
    sanitized: list[str] = []
    for value in command:
        path = Path(value)
        if path.is_absolute():
            sanitized.append(relative_artifact_path(path, root))
        else:
            sanitized.append(value)
    return sanitized


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0] if output else None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _cuda_runtime_version(*, importer: Any = importlib.import_module) -> str | None:
    """Read the loaded CUDA runtime version without depending on PyTorch."""
    try:
        cudart = importer("cuda.bindings.runtime")
        result = cudart.cudaRuntimeGetVersion()
        values = result if isinstance(result, tuple) else (result,)
        if len(values) != 2 or values[0] != cudart.cudaError_t.cudaSuccess:
            return None
        version = int(values[1])
    except Exception:
        return None
    major = version // 1000
    minor = (version % 1000) // 10
    patch = version % 10
    return f"{major}.{minor}.{patch}" if patch else f"{major}.{minor}"


def collect_image_identity(*, default_reference: str = "unknown") -> dict[str, str]:
    """Collect the immutable build identity shared by benchmark evidence."""
    return {
        "reference": os.environ.get("TRTVIDEO_IMAGE_REF", default_reference),
        "id": os.environ.get("TRTVIDEO_IMAGE_ID", "unknown"),
        "base_reference": os.environ.get(
            "TRTVIDEO_BASE_IMAGE",
            "nvcr.io/nvidia/tensorrt:26.06-py3",
        ),
        "repository_revision": os.environ.get("TRTVIDEO_BUILD_REVISION", "unknown"),
        "source_dirty": os.environ.get("TRTVIDEO_BUILD_DIRTY", "unknown"),
    }


def collect_environment(gpu: dict[str, Any]) -> dict[str, Any]:
    """Collect only fields allowed by the public benchmark methodology."""
    return {
        "gpu": gpu,
        "cpu": {
            "model": _cpu_model(),
            "logical_cores": os.cpu_count(),
        },
        "software": {
            "python": platform.python_version(),
            "trtvideo": _package_version("trtvideo"),
            "torch": _package_version("torch"),
            "cuda": _cuda_runtime_version(),
            "tensorrt": _package_version("tensorrt"),
            "cvcuda": _package_version("cvcuda-cu12") or _package_version("cvcuda"),
            "pynvvideocodec": _package_version("pynvvideocodec"),
            "nvidia_ml_py": _package_version("nvidia-ml-py"),
            "ffmpeg": _command_version(["ffmpeg", "-version"]),
        },
        "image": collect_image_identity(),
    }


def environment_errors(environment: dict[str, Any]) -> list[str]:
    """Return reproducibility errors that make a public benchmark invalid."""
    image = environment.get("image", {})
    errors = []
    if image.get("id") in {None, "", "unknown"}:
        errors.append("Docker image ID is unknown")
    if image.get("repository_revision") in {None, "", "unknown"}:
        errors.append("Image repository revision is unknown")
    if str(image.get("source_dirty", "unknown")).lower() not in {"0", "false"}:
        errors.append("Image was built from unknown or dirty source state")
    return errors
