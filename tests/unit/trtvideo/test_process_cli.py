import pytest

from trtvideo.cli.process import build_parser, process_config_from_args


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


def test_cli_arguments_map_to_typed_process_config() -> None:
    args = build_parser().parse_args(
        [
            "--engine",
            "models/model.engine",
            "--input",
            "videos/input.mp4",
            "--max-frames",
            "12",
            "--profile-json",
            "artefacts/profile.json",
        ]
    )

    config = process_config_from_args(args)

    assert str(config.engine_path) == "models/model.engine"
    assert str(config.output_path) == "videos/input_processed.mp4"
    assert config.max_frames == 12
    assert str(config.profile_json_path) == "artefacts/profile.json"


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
