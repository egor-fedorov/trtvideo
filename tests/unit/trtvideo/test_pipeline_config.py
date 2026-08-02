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
