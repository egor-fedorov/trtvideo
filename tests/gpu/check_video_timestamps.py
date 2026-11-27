"""Opt-in end-to-end timeline regression using generated media and a small TRT engine."""

import json
import subprocess
import unittest
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from check_binding_strides import _build_identity, trt


def probe(path, stream):
    data = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                stream,
                "-show_packets",
                "-show_streams",
                "-show_data_hash",
                "sha256",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    info = data["streams"][0]
    base = Fraction(info["time_base"])
    packets = sorted(data["packets"], key=lambda p: int(p["pts"]))
    return info, [(int(p["pts"]) * base, p["data_hash"]) for p in packets]


class VideoTimestampsTest(unittest.TestCase):
    def test_lossy_mux_rejected_before_engine_initialization(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, engine = (
                root / "original.mkv",
                root / "existing.mp4",
                root / "unused.engine",
            )
            engine.touch()
            output.write_bytes(b"previous output")
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=s=64x64:r=24:d=0.25",
                    "-c:v",
                    "libx264",
                    "-disposition:v:0",
                    "original",
                    str(source),
                ],
                check=True,
            )
            failed = subprocess.run(
                [
                    "trtvideo",
                    "--engine",
                    str(engine),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("original=1", failed.stderr)
            self.assertNotIn("Initializing TensorRT", failed.stderr)
            self.assertEqual(output.read_bytes(), b"previous output")
            self.assertFalse(list(root.glob(".*.partial.*")))

    def test_gpu_timelines(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "identity.engine"
            _build_identity(engine, (1, 3, 192, 320), trt.float32)
            for name, rate, offset, bframes in [
                ("vfr", "24", "0", "0"),
                ("offset-b", "24", "0.041666667", "2"),
                ("cfr", "30000/1001", "0", "2"),
            ]:
                source = root / f"{name}.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-itsoffset",
                        offset,
                        "-f",
                        "lavfi",
                        "-i",
                        f"testsrc2=size=320x192:rate={rate}:duration=4",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=sample_rate=44100:duration=4",
                        *(
                            ["-vf", "select=lt(n\\,48)+not(mod(n\\,2)),setsar=4/3"]
                            if name != "cfr"
                            else []
                        ),
                        "-fps_mode",
                        "vfr",
                        "-c:v",
                        "libx264",
                        "-bf",
                        bframes,
                        "-colorspace",
                        "bt709",
                        "-color_primaries",
                        "bt709",
                        "-color_trc",
                        "bt709",
                        "-color_range",
                        "tv",
                        "-c:a",
                        "aac",
                        "-movie_timescale",
                        "120000",
                        "-metadata:s:v:0",
                        "language=ita",
                        "-metadata:s:v:0",
                        "handler_name=Camera A",
                        str(source),
                    ],
                    check=True,
                )
                if name == "vfr":
                    rotated = root / "portrait.mp4"
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-display_rotation",
                            "90",
                            "-i",
                            str(source),
                            "-map",
                            "0",
                            "-c",
                            "copy",
                            str(rotated),
                        ],
                        check=True,
                    )
                    source = rotated
                source_v, packets = probe(source, "v:0")
                _, audio = probe(source, "a:0")
                pts = [p[0] for p in packets]
                end = pts[0] + int(source_v["duration_ts"]) * Fraction(source_v["time_base"])
                if end <= pts[-1]:
                    rows = json.loads(
                        subprocess.check_output(
                            [
                                "ffprobe",
                                "-v",
                                "error",
                                "-select_streams",
                                "v:0",
                                "-show_packets",
                                "-show_entries",
                                "packet=pts,duration",
                                "-of",
                                "json",
                                str(source),
                            ]
                        )
                    )["packets"]
                    last = max(rows, key=lambda row: int(row["pts"]))
                    end = pts[-1] + int(last["duration"]) * Fraction(source_v["time_base"])
                if name == "vfr":
                    self.assertEqual((len(pts), end), (72, Fraction(95, 24)))
                cases = [
                    (1, "h264", "mp4"),
                    (0, "h264", "mp4"),
                    (49, "hevc", "mp4"),
                    (500, "hevc", "mp4"),
                    (0, "hevc", "mkv"),
                ]
                for limit, codec, ext in cases:
                    with self.subTest(source=name, limit=limit, codec=codec, container=ext):
                        output = root / f"out.{ext}"
                        subprocess.run(
                            [
                                "trtvideo",
                                "--engine",
                                str(engine),
                                "--input",
                                str(source),
                                "--output",
                                str(output),
                                "--bitrate-mbps",
                                "2",
                                "--codec",
                                codec,
                                *(["--max-frames", str(limit)] if limit else []),
                                "--quiet",
                            ],
                            check=True,
                        )
                        info, actual = probe(output, "v:0")
                        tags = {k.lower(): v for k, v in info["tags"].items()}
                        self.assertEqual(
                            (tags["language"], tags["handler_name"]), ("ita", "Camera A")
                        )
                        self.assertEqual(
                            info.get("sample_aspect_ratio", "1:1"),
                            source_v.get("sample_aspect_ratio", "1:1"),
                        )
                        if name == "vfr":
                            self.assertTrue(
                                any(s.get("rotation") == 90 for s in info.get("side_data_list", []))
                            )
                        count = min(limit or len(pts), len(pts))
                        tolerance = Fraction(info["time_base"]) if ext == "mkv" else 0
                        self.assertEqual(len(actual), count)
                        for (time, _), expected in zip(actual, pts[:count], strict=True):
                            self.assertLessEqual(abs(time - expected), tolerance)
                        cutoff = pts[count] if count < len(pts) else end
                        if ext == "mp4":
                            self.assertEqual(
                                int(info["duration_ts"]) * Fraction(info["time_base"]),
                                cutoff - pts[0],
                            )
                            self.assertGreater(int(info["bit_rate"]), 0)
                        else:
                            hours, minutes, seconds = info["tags"]["DURATION"].split(":")
                            track_end = int(hours) * 3600 + int(minutes) * 60 + Fraction(seconds)
                            self.assertLessEqual(abs(track_end - cutoff), tolerance)
                        self.assertEqual(
                            (
                                info["codec_name"],
                                info["pix_fmt"],
                                info["color_range"],
                                info["color_space"],
                                info["color_transfer"],
                                info["color_primaries"],
                            ),
                            (codec, "yuv420p", "tv", "bt709", "bt709", "bt709"),
                        )
                        _, copied = probe(output, "a:0")
                        expected_audio = [a for a in audio if not limit or a[0] < cutoff]
                        self.assertEqual([a[1] for a in copied], [a[1] for a in expected_audio])
                        for (time, _), (expected, _) in zip(copied, expected_audio, strict=True):
                            self.assertLessEqual(abs(time - expected), tolerance)
                        decoded = subprocess.run(
                            [
                                "ffmpeg",
                                "-v",
                                "error",
                                "-xerror",
                                "-i",
                                str(output),
                                "-fps_mode",
                                "passthrough",
                                "-enc_time_base:v",
                                info["time_base"],
                                "-f",
                                "null",
                                "-",
                            ],
                            check=True,
                            capture_output=True,
                        )
                        self.assertEqual(decoded.stderr, b"")

            raw = root / "without-pts.h264"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(root / "vfr.mp4"),
                    "-map",
                    "0:v",
                    "-c",
                    "copy",
                    "-f",
                    "h264",
                    str(raw),
                ],
                check=True,
            )
            existing = root / "existing.mp4"
            existing.write_bytes(b"previous output")
            failed = subprocess.run(
                [
                    "trtvideo",
                    "--engine",
                    str(engine),
                    "--input",
                    str(raw),
                    "--output",
                    str(existing),
                    "--bitrate-mbps",
                    "2",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("no presentation timestamps", failed.stderr)
            self.assertEqual(existing.read_bytes(), b"previous output")
            self.assertFalse(list(root.glob(".*.partial.*")))


if __name__ == "__main__":
    unittest.main()
