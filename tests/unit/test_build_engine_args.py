from dataclasses import dataclass

import pytest

from ai_media.cli.build_engine import parse_shape_arg, validate_profile_shapes


@dataclass
class FakeInputTensor:
    name: str
    shape: tuple[int, ...]


def test_parse_shape_arg() -> None:
    assert parse_shape_arg("input:1x3x720x1280") == ("input", (1, 3, 720, 1280))


@pytest.mark.parametrize("value", ["input", ":1x3x720x1280", "input:1x3xbad", "input:1x0x720"])
def test_parse_shape_arg_exits_for_invalid_values(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_shape_arg(value)


def test_validate_profile_shapes_returns_profile() -> None:
    profile = validate_profile_shapes(
        FakeInputTensor(name="input", shape=(-1, 3, -1, -1)),
        ("input", (1, 3, 360, 640)),
        ("input", (1, 3, 720, 1280)),
        ("input", (1, 3, 1080, 1920)),
    )

    assert profile == (
        (1, 3, 360, 640),
        (1, 3, 720, 1280),
        (1, 3, 1080, 1920),
    )


def test_validate_profile_shapes_returns_none_without_profile() -> None:
    assert (
        validate_profile_shapes(
            FakeInputTensor(name="input", shape=(1, 3, 720, 1280)),
            None,
            None,
            None,
        )
        is None
    )


@pytest.mark.parametrize(
    ("min_shape", "opt_shape", "max_shape"),
    [
        (("input", (1, 3, 360, 640)), None, ("input", (1, 3, 1080, 1920))),
        (("other", (1, 3, 360, 640)), ("other", (1, 3, 720, 1280)), ("other", (1, 3, 1080, 1920))),
        (("input", (1, 3, 360)), ("input", (1, 3, 720)), ("input", (1, 3, 1080))),
        (("input", (1, 3, 1080, 1920)), ("input", (1, 3, 720, 1280)), ("input", (1, 3, 360, 640))),
    ],
)
def test_validate_profile_shapes_exits_for_invalid_profiles(
    min_shape: tuple[str, tuple[int, ...]] | None,
    opt_shape: tuple[str, tuple[int, ...]] | None,
    max_shape: tuple[str, tuple[int, ...]] | None,
) -> None:
    with pytest.raises(SystemExit):
        validate_profile_shapes(
            FakeInputTensor(name="input", shape=(-1, 3, -1, -1)),
            min_shape,
            opt_shape,
            max_shape,
        )
