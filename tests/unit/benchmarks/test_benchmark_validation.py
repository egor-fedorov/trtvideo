from __future__ import annotations

import copy

from trtvideo.benchmarking.validation import OutputContract, validate_output_probe


def canonical_probe(frames: int = 48) -> tuple[dict, list[dict]]:
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
                "nb_read_frames": str(frames),
                "duration": str(frames / 24),
                "bit_rate": "35000000",
            }
        ],
        "format": {"duration": str(frames / 24), "bit_rate": "35100000"},
    }
    packets = [
        {
            "pts": str(index * 1000),
            "dts": str(index * 1000),
            "flags": "K__" if index % 24 == 0 else "___",
        }
        for index in range(frames)
    ]
    return probe, packets


def contract(frames: int = 48) -> OutputContract:
    return OutputContract(
        width=2560,
        height=1440,
        fps="24/1",
        frames=frames,
        gop_frames=24,
        target_bitrate_mbps=35,
    )


def test_validate_output_probe_accepts_canonical_output() -> None:
    probe, packets = canonical_probe()

    result = validate_output_probe(probe, packets, contract=contract())

    assert result["valid"] is True
    assert result["checks"]["pts_monotonic"] is True
    assert result["checks"]["keyframe_interval"] is True
    assert result["observed"]["keyframe_gaps_frames"] == [24]


def test_validate_output_probe_rejects_non_monotonic_timestamps() -> None:
    probe, packets = canonical_probe()
    packets[20]["dts"] = packets[19]["dts"]

    result = validate_output_probe(probe, packets, contract=contract())

    assert result["valid"] is False
    assert result["checks"]["dts_monotonic"] is False
    assert any("index 20" in error for error in result["errors"])


def test_validate_output_probe_rejects_missing_keyframes() -> None:
    probe, packets = canonical_probe()
    for packet in packets[1:]:
        packet["flags"] = "___"

    result = validate_output_probe(probe, packets, contract=contract())

    assert result["valid"] is False
    assert result["checks"]["keyframe_interval"] is False


def test_validate_output_probe_rejects_bitrate_outside_tolerance() -> None:
    probe, packets = canonical_probe()
    probe["streams"][0]["bit_rate"] = "28000000"

    result = validate_output_probe(probe, packets, contract=contract())

    assert result["valid"] is False
    assert result["checks"]["bitrate"] is False
    assert result["observed"]["bitrate_relative_delta"] == 0.2


def test_validate_output_probe_rejects_decode_error_and_color_shift() -> None:
    probe, packets = canonical_probe()
    invalid_probe = copy.deepcopy(probe)
    invalid_probe["streams"][0]["color_space"] = "bt470bg"

    result = validate_output_probe(
        invalid_probe,
        packets,
        contract=contract(),
        decode_error="invalid DTS",
    )

    assert result["valid"] is False
    assert result["checks"]["full_decode"] is False
    assert result["checks"]["color_space"] is False
