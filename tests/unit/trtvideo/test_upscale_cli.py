import argparse

import pytest

from trtvideo.cli.upscale import build_parser, validate_args


def test_ffmpeg_accepts_crf() -> None:
    args = argparse.Namespace(backend="ffmpeg", crf=18)

    validate_args(args)


def test_nvcodec_rejects_crf() -> None:
    args = argparse.Namespace(backend="nvcodec", crf=18)

    with pytest.raises(SystemExit):
        validate_args(args)


def test_crf_default_is_unset_until_backend_defaults_apply() -> None:
    args = build_parser().parse_args(
        [
            "--engine",
            "models/engines/model.engine",
            "--input",
            "videos/input.mp4",
        ]
    )

    assert args.backend == "nvcodec"
    assert args.crf is None


def test_engine_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--input", "videos/input.mp4"])
