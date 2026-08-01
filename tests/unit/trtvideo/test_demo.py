from pathlib import Path

import pytest

from trtvideo.cli.demo import build_parser
from trtvideo.demo import DemoError
from trtvideo.demo.config import (
    DEMO_FRAMES,
    MODEL_SHA256,
    MODEL_URL,
    DemoPaths,
    DemoVideoContract,
)
from trtvideo.demo.media import build_demo_input_command, validate_demo_probe
from trtvideo.demo.workflow import process_command


def _valid_probe() -> tuple[dict, list[dict]]:
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 2560,
                "height": 1440,
                "pix_fmt": "yuv420p",
                "color_range": "tv",
                "color_space": "bt709",
                "color_transfer": "bt709",
                "color_primaries": "bt709",
                "has_b_frames": 0,
                "avg_frame_rate": "24/1",
                "nb_read_frames": str(DEMO_FRAMES),
            },
            {
                "codec_type": "audio",
                "tags": {"language": "eng"},
                "disposition": {"default": 1},
            },
            {
                "codec_type": "audio",
                "tags": {"language": "jpn"},
                "disposition": {"default": 0},
            },
            {
                "codec_type": "subtitle",
                "disposition": {"forced": 1},
            },
            {
                "codec_type": "attachment",
                "tags": {"filename": "attachment.txt", "mimetype": "text/plain"},
            },
        ],
        "chapters": [
            {"tags": {"title": "First half"}},
            {"tags": {"title": "Second half"}},
        ],
        "format": {
            "tags": {
                "title": "trtvideo Demo",
                "comment": "Generated synthetic input",
            }
        },
    }
    packets = [
        {"pts": index, "dts": index, "flags": "K__" if index == 0 else "___"}
        for index in range(DEMO_FRAMES)
    ]
    return probe, packets


def test_demo_uses_pinned_immutable_model_source() -> None:
    assert "/releases/download/v0.2.1/" in MODEL_URL
    assert len(MODEL_SHA256) == 64


def test_demo_paths_stay_under_cache_root(tmp_path: Path) -> None:
    paths = DemoPaths.under(tmp_path)

    assert paths.weights == tmp_path / "models" / "RealESRGAN_x2plus.pth"
    assert paths.output_video == tmp_path / "output" / "demo_1440p.mkv"
    assert paths.report == tmp_path / "demo-result.json"


def test_demo_input_is_rich_deterministic_mkv(tmp_path: Path) -> None:
    command = build_demo_input_command(DemoPaths.under(tmp_path))

    assert "testsrc2=size=1280x720:rate=24:duration=1" in command
    assert command[command.index("-frames:v") + 1] == "24"
    assert command.count("-map") == 4
    assert "-attach" in command
    assert command[-1].endswith("demo_720p.mkv")


def test_demo_process_uses_explicit_engine(tmp_path: Path) -> None:
    command = process_command(DemoPaths.under(tmp_path), gpu_id=2)

    assert command[0] == "trtvideo"
    assert "--backend" not in command
    assert command[command.index("--gpu-id") + 1] == "2"
    assert command[command.index("--bitrate-mbps") + 1] == "12.0"
    assert command[command.index("--output") + 1].endswith("demo_1440p.mkv")


def test_demo_probe_accepts_complete_media_contract() -> None:
    probe, packets = _valid_probe()

    observed = validate_demo_probe(
        probe,
        packets,
        DemoVideoContract(width=2560, height=1440),
    )

    assert observed["stream_counts"] == {
        "video": 1,
        "audio": 2,
        "subtitle": 1,
        "attachment": 1,
    }
    assert observed["timestamps"] == "strictly_monotonic"


def test_demo_probe_rejects_non_monotonic_timestamps() -> None:
    probe, packets = _valid_probe()
    packets[-1]["dts"] = packets[-2]["dts"]

    with pytest.raises(DemoError, match="DTS"):
        validate_demo_probe(
            probe,
            packets,
            DemoVideoContract(width=2560, height=1440),
        )


def test_demo_parser_accepts_gpu_and_force() -> None:
    args = build_parser().parse_args(["--root", "/tmp/demo", "--gpu-id", "3", "--force"])

    assert args.root == Path("/tmp/demo")
    assert args.gpu_id == 3
    assert args.force is True
