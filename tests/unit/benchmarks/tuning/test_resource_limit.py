from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.scripts.tuning.resource_limit import detect_cuda_oom


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _invalid_suite(tmp_path: Path, stderr: str) -> tuple[Path, Path]:
    run_dir = tmp_path / "artefacts" / "candidate" / "run-01"
    stderr_path = run_dir / "warmup.stderr.log"
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.write_text(stderr, encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "status": "invalid",
            "artifacts": {
                "warmup_stderr": stderr_path.relative_to(tmp_path).as_posix(),
            },
        },
    )
    suite_path = run_dir.parent / "suite.json"
    _write_json(
        suite_path,
        {
            "status": "invalid",
            "runs": [
                {
                    "manifest": manifest_path.relative_to(tmp_path).as_posix(),
                }
            ],
        },
    )
    return suite_path, stderr_path


def test_detect_cuda_oom_returns_hashed_evidence(tmp_path: Path) -> None:
    suite_path, stderr_path = _invalid_suite(
        tmp_path,
        "Error Code 2: OutOfMemory (Requested size was 3450470400 bytes.)\n",
    )

    evidence = detect_cuda_oom(root=tmp_path, suite_path=suite_path)

    assert evidence is not None
    assert evidence.kind == "cuda-out-of-memory"
    assert evidence.suite_path == suite_path.relative_to(tmp_path).as_posix()
    assert evidence.stderr_path == stderr_path.relative_to(tmp_path).as_posix()
    assert evidence.stderr_sha256 == hashlib.sha256(stderr_path.read_bytes()).hexdigest()


def test_detect_cuda_oom_accepts_vstrt_cuda_malloc_failure(tmp_path: Path) -> None:
    suite_path, _ = _invalid_suite(
        tmp_path,
        "operator(): 'cudaMalloc(&d_data.data, size)' failed: out of memory\n",
    )

    evidence = detect_cuda_oom(root=tmp_path, suite_path=suite_path)

    assert evidence is not None
    assert evidence.kind == "cuda-out-of-memory"


def test_detect_cuda_oom_does_not_accept_unrelated_failure(tmp_path: Path) -> None:
    suite_path, _ = _invalid_suite(tmp_path, "Failed to initialize VSScript\n")

    assert detect_cuda_oom(root=tmp_path, suite_path=suite_path) is None


def test_detect_cuda_oom_rejects_generic_out_of_memory(tmp_path: Path) -> None:
    suite_path, _ = _invalid_suite(tmp_path, "Host allocator failed: out of memory\n")

    assert detect_cuda_oom(root=tmp_path, suite_path=suite_path) is None


@pytest.mark.parametrize("exception", ["torch.OutOfMemoryError", "torch.cuda.OutOfMemoryError"])
def test_detect_tas_cuda_out_of_memory(tmp_path, exception) -> None:
    suite_path, _ = _invalid_suite(
        tmp_path, f"{exception}: CUDA out of memory. Tried to allocate 2 GiB"
    )
    assert detect_cuda_oom(root=tmp_path, suite_path=suite_path) is not None


def test_thermal_slowdown_is_not_a_resource_ceiling(tmp_path) -> None:
    suite_path, _ = _invalid_suite(
        tmp_path, "Invalid throttle reasons observed: sw_thermal_slowdown"
    )
    assert detect_cuda_oom(root=tmp_path, suite_path=suite_path) is None


def test_cuda_oom_does_not_hide_concurrent_thermal_failure(tmp_path) -> None:
    suite_path, _ = _invalid_suite(tmp_path, "torch.OutOfMemoryError: CUDA out of memory")
    manifest_path = suite_path.parent / "run-01/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["errors"] = ["Invalid throttle reasons observed: sw_thermal_slowdown"]
    _write_json(manifest_path, manifest)
    assert detect_cuda_oom(root=tmp_path, suite_path=suite_path) is None
