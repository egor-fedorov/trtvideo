import sqlite3
from pathlib import Path

import pytest

from benchmarks.scripts.report.publish_diagnostics import (
    DiagnosticPublicationError,
    _merge_intervals,
    _overlap_duration,
    analyze_nsight_sqlite,
)


def _write_trace_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start INTEGER, end INTEGER, bytes INTEGER, copyKind INTEGER
        );
        CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER PRIMARY KEY, name TEXT, label TEXT);
        CREATE TABLE ENUM_VIDEO_ENGINE_TYPE (id INTEGER PRIMARY KEY, name TEXT, label TEXT);
        CREATE TABLE GPU_VIDEO_ENGINE_WORKLOAD (
            start INTEGER, end INTEGER, engineType INTEGER
        );

        INSERT INTO StringIds VALUES (1, 'trtvideo.frame_loop');
        INSERT INTO NVTX_EVENTS VALUES (100, 1100, NULL, 1);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100, 400), (300, 700), (900, 1100);
        INSERT INTO ENUM_VIDEO_ENGINE_TYPE VALUES (0, 'NVDEC', 'NV Decode');
        INSERT INTO ENUM_VIDEO_ENGINE_TYPE VALUES (1, 'NVENC', 'NV Encode');
        INSERT INTO GPU_VIDEO_ENGINE_WORKLOAD VALUES (150, 250, 0), (750, 1000, 1);
        INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1, 'CUDA_MEMCPY_KIND_HTOD', 'H2D');
        INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (2, 'CUDA_MEMCPY_KIND_DTOH', 'D2H');
        INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (8, 'CUDA_MEMCPY_KIND_DTOD', 'D2D');
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (10, 20, 1024, 1);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (200, 250, 2048, 8);
        """
    )
    connection.commit()
    connection.close()


def test_interval_helpers_account_for_concurrent_work() -> None:
    assert _merge_intervals([(0, 5), (3, 8), (10, 12)]) == [(0, 8), (10, 12)]
    assert _overlap_duration([(0, 8), (10, 12)], [(4, 11)]) == 5


def test_analyze_nsight_sqlite_uses_frame_loop_intervals(tmp_path: Path) -> None:
    path = tmp_path / "trace.sqlite"
    _write_trace_database(path)

    result = analyze_nsight_sqlite(path, frames=2)

    assert result["frame_loop_sec"] == pytest.approx(0.000001)
    assert result["cuda_kernel_time_coverage_of_frame_loop_percent"] == pytest.approx(80.0)
    assert result["nvdec_overlap_with_cuda_kernels_percent"] == pytest.approx(100.0)
    assert result["nvenc_overlap_with_cuda_kernels_percent"] == pytest.approx(40.0)
    assert result["startup_h2d_copy_count"] == 1
    assert result["frame_loop_h2d_copy_count"] == 0
    assert result["frame_loop_d2h_copy_count"] == 0
    assert result["frame_loop_d2d_copy_count"] == 1
    assert result["per_frame_d2d_mib"] == pytest.approx(1024 / 1024**2)
    assert result["per_frame_d2d_time_ms"] == pytest.approx(0.000025)
    assert result["material_per_frame_h2d_or_d2h"] is False


def test_analyze_nsight_sqlite_rejects_missing_tables(tmp_path: Path) -> None:
    path = tmp_path / "empty.sqlite"
    sqlite3.connect(path).close()

    with pytest.raises(DiagnosticPublicationError, match="lacks tables"):
        analyze_nsight_sqlite(path, frames=1)
