from pathlib import Path

import pytest

from trtvideo.cli.demo import build_parser
from trtvideo.demo import DemoError
from trtvideo.demo import workflow as demo_workflow
from trtvideo.demo.config import (
    DEMO_AUDIO_BITRATE_KBPS,
    DEMO_AUDIO_CHANNELS,
    DEMO_AUDIO_SAMPLE_RATE_HZ,
    DEMO_FRAMES,
    MODEL_SHA256,
    MODEL_URL,
    VIDEO_SHA256,
    VIDEO_START_SECONDS,
    VIDEO_URL,
    DemoPaths,
    DemoVideoContract,
)
from trtvideo.demo.media import (
    build_demo_input_command,
    summarize_demo_chroma,
    validate_demo_color_preservation,
    validate_demo_probe,
)
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
            },
        ],
    }
    packets = [
        {"pts": index, "dts": index, "flags": "K__" if index == 0 else "___"}
        for index in range(DEMO_FRAMES)
    ]
    return probe, packets


def test_demo_uses_pinned_immutable_model_source() -> None:
    assert "/releases/download/v0.2.1/" in MODEL_URL
    assert len(MODEL_SHA256) == 64


def test_demo_uses_pinned_cc_by_sa_live_action_source() -> None:
    assert "upload.wikimedia.org" in VIDEO_URL
    assert "Jacqueville_beach" in VIDEO_URL
    assert len(VIDEO_SHA256) == 64


def test_demo_paths_stay_under_cache_root(tmp_path: Path) -> None:
    paths = DemoPaths.under(tmp_path)

    assert paths.weights == tmp_path / "models" / "RealESRGAN_x2plus.pth"
    assert paths.export_conformance == (
        tmp_path / "models" / "onnx" / "realesrgan_x2plus.export-conformance.json"
    )
    assert paths.output_video == tmp_path / "output" / "demo_1440p.mp4"
    assert paths.source_video == tmp_path / "sources" / "Jacqueville-beach-2026.webm"
    assert paths.input_manifest == tmp_path / "videos" / "demo_720p.input.json"
    assert paths.report == tmp_path / "demo-result.json"


def test_demo_input_cache_is_bound_to_source_and_prepared_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = DemoPaths.under(tmp_path)
    paths.input_video.parent.mkdir(parents=True)
    paths.input_video.write_bytes(b"prepared video")
    demo_workflow._write_input_manifest(paths)

    assert demo_workflow._valid_input_cache(paths)

    paths.input_video.write_bytes(b"changed video")
    assert not demo_workflow._valid_input_cache(paths)

    paths.input_video.write_bytes(b"prepared video")
    monkeypatch.setattr(demo_workflow, "VIDEO_SHA256", "0" * 64)
    assert not demo_workflow._valid_input_cache(paths)


def test_demo_input_is_deterministic_live_action_excerpt(tmp_path: Path) -> None:
    paths = DemoPaths.under(tmp_path)
    command = build_demo_input_command(paths)

    assert "testsrc2=size=1280x720:rate=24:duration=1" not in command
    assert str(paths.source_video) in command
    assert command[command.index("-ss") + 1] == str(VIDEO_START_SECONDS)
    assert command[command.index("-frames:v") + 1] == "120"
    video_filter = command[command.index("-vf") + 1]
    assert video_filter.startswith("fps=fps=24/1:round=near,scale=1280:720:flags=lanczos")
    assert "setparams=range=limited" in video_filter
    assert command.count("-map") == 2
    assert "0:a:0" in command
    assert command[command.index("-b:a") + 1] == f"{DEMO_AUDIO_BITRATE_KBPS}k"
    assert command[command.index("-ac") + 1] == str(DEMO_AUDIO_CHANNELS)
    assert command[command.index("-ar") + 1] == str(DEMO_AUDIO_SAMPLE_RATE_HZ)
    assert "-attach" not in command
    assert "sine=" not in " ".join(command)
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert any(value.startswith("copyright=CC-BY-SA-4.0") for value in command)
    assert command[-1].endswith("demo_720p.mp4")


def test_demo_process_uses_explicit_engine(tmp_path: Path) -> None:
    command = process_command(DemoPaths.under(tmp_path), gpu_id=2)

    assert command[0] == "trtvideo"
    assert "--backend" not in command
    assert command[command.index("--gpu-id") + 1] == "2"
    assert command[command.index("--bitrate-mbps") + 1] == "12.0"
    assert command[command.index("--output") + 1].endswith("demo_1440p.mp4")


def test_demo_probe_accepts_video_with_source_audio() -> None:
    probe, packets = _valid_probe()

    observed = validate_demo_probe(
        probe,
        packets,
        DemoVideoContract(width=2560, height=1440),
    )

    assert observed["stream_counts"] == {
        "video": 1,
        "audio": 1,
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


def test_demo_chroma_summarizes_pinned_frame() -> None:
    observed = summarize_demo_chroma({"ULOW": 106.0, "UHIGH": 150.0, "VLOW": 108.0, "VHIGH": 145.0})

    assert observed == {
        "frame_index": 18,
        "u_percentile_span": 44.0,
        "v_percentile_span": 37.0,
    }


def test_demo_color_preservation_accepts_retained_chroma() -> None:
    observed = validate_demo_color_preservation(
        {"u_percentile_span": 44.0, "v_percentile_span": 37.0},
        {"u_percentile_span": 40.0, "v_percentile_span": 35.0},
    )

    assert observed["u_retention_ratio"] == pytest.approx(40.0 / 44.0)
    assert observed["v_retention_ratio"] == pytest.approx(35.0 / 37.0)


def test_demo_color_preservation_rejects_chroma_collapse() -> None:
    with pytest.raises(DemoError, match="chroma preservation failed"):
        validate_demo_color_preservation(
            {"u_percentile_span": 44.0, "v_percentile_span": 37.0},
            {"u_percentile_span": 20.0, "v_percentile_span": 35.0},
        )


def test_demo_parser_accepts_gpu_and_force() -> None:
    args = build_parser().parse_args(["--root", "/tmp/demo", "--gpu-id", "3", "--force"])

    assert args.root == Path("/tmp/demo")
    assert args.gpu_id == 3
    assert args.force is True
