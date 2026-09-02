from pathlib import Path

import pytest

from trtvideo.pipelines.config import PipelineError, ProcessConfig, default_output_path


def test_default_output_path_uses_processed_suffix() -> None:
    assert default_output_path(Path("videos/input.mp4")) == Path("videos/input_processed.mp4")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gpu_id": -1}, "--gpu-id"),
        ({"max_frames": -1}, "--max-frames"),
        ({"warmup_frames": -1}, "--warmup-frames"),
        ({"log_interval": 0}, "--log-interval"),
        ({"bitrate_mbps": 0.0}, "--bitrate-mbps"),
    ],
)
def test_process_config_rejects_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    values = {
        "engine_path": Path("model.engine"),
        "input_path": Path("input.mp4"),
        "output_path": Path("output.mp4"),
        **overrides,
    }

    with pytest.raises(PipelineError, match=message):
        ProcessConfig(**values)


@pytest.mark.parametrize(
    ("result", "progress"),
    [
        (Path("-"), Path("-")),
        (Path("reports/process.json"), Path("reports/process.json")),
    ],
)
def test_process_config_requires_distinct_machine_output_destinations(
    result: Path,
    progress: Path,
) -> None:
    with pytest.raises(PipelineError, match="must use different destinations"):
        ProcessConfig(
            engine_path=Path("model.engine"),
            input_path=Path("input.mp4"),
            output_path=Path("output.mp4"),
            result_json_path=result,
            progress_jsonl_path=progress,
        )


@pytest.mark.parametrize(
    ("field", "destination"),
    [
        ("result_json_path", Path("model.engine")),
        ("result_json_path", Path("input.mp4")),
        ("progress_jsonl_path", Path("output.mp4")),
        ("profile_json_path", Path("output.mp4")),
        ("benchmark_lifecycle_path", Path("input.mp4")),
    ],
)
def test_process_config_protects_runtime_artifacts_from_machine_reports(
    field: str,
    destination: Path,
) -> None:
    with pytest.raises(PipelineError, match="must not overwrite"):
        ProcessConfig(
            engine_path=Path("model.engine"),
            input_path=Path("input.mp4"),
            output_path=Path("output.mp4"),
            **{field: destination},
        )
