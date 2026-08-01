import pytest

from trtvideo.cli.process import build_parser


def test_parser_uses_single_gpu_resident_contract() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--engine",
            "models/engines/model.engine",
            "--input",
            "videos/input.mp4",
        ]
    )

    assert parser.prog == "trtvideo"
    assert args.codec == "h264"
    assert args.bitrate_mbps is None


@pytest.mark.parametrize("removed_option", ["--backend", "--crf", "--cuda-graph"])
def test_removed_backend_options_are_rejected(removed_option: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--engine",
                "models/engines/model.engine",
                "--input",
                "videos/input.mp4",
                removed_option,
                "nvcodec" if removed_option == "--backend" else "18",
            ]
        )


def test_engine_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--input", "videos/input.mp4"])


def test_profile_help_warns_about_serialized_non_throughput_measurement() -> None:
    help_text = " ".join(build_parser().format_help().split())

    assert "serializes the pipeline per frame" in help_text
    assert "FPS is not throughput" in help_text
