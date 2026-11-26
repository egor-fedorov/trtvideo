from fractions import Fraction

import pytest

from trtvideo.video.geometry import DisplayGeometry
from trtvideo.video.output import preservation


def stream(matrix, **extra):
    text = "\n".join(f"{i:08d}: " + " ".join(map(str, row)) for i, row in enumerate(matrix))
    return {
        "width": 128,
        "height": 72,
        "side_data_list": [{"side_data_type": "Display Matrix", "displaymatrix": text}],
        **extra,
    }


def test_quarter_turn_and_anamorphic_ratio():
    matrix = [[0, -65536, 0], [65536, 0, 0], [0, 128 * 65536, 1 << 30]]
    s = stream(
        matrix, sample_aspect_ratio="4:3", display_aspect_ratio="64:27", tags={"rotate": "-270"}
    )
    assert DisplayGeometry.from_stream(s) == DisplayGeometry(90, Fraction(4, 3))
    s["tags"]["rotate"] = "180"
    with pytest.raises(ValueError, match="Conflicting"):
        DisplayGeometry.from_stream(s)


@pytest.mark.parametrize(
    "matrix",
    [
        [[65536, 0, 0], [0, -65536, 0], [0, 0, 1 << 30]],
        [[65536, 40, 0], [0, 65536, 0], [0, 0, 1 << 30]],
        [[0, -65536, 0], [65536, 0, 0], [65536, 0, 1 << 30]],
    ],
)
def test_non_rotation_transform_is_not_flattened(matrix):
    with pytest.raises(ValueError, match="display matrix"):
        DisplayGeometry.from_stream(stream(matrix))


def test_legacy_mkv_rejected_before_processing(tmp_path, monkeypatch):
    monkeypatch.setattr(preservation, "has_display_rotation_option", lambda: False)
    with pytest.raises(preservation.MediaPreservationError, match="use MP4 or MOV"):
        preservation.rotation_args(DisplayGeometry(270), tmp_path / "out.mkv")
