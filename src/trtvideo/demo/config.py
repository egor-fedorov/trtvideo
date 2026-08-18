"""Pinned assets and filesystem contract for the self-contained demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_NAME = "RealESRGAN_x2plus"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
MODEL_SHA256 = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
MODEL_SIZE_BYTES = 67_061_725
MODEL_SCALE = 2
MODEL_ATTRIBUTION = "Real-ESRGAN, Xintao Wang et al."
MODEL_LICENSE_URL = "https://github.com/xinntao/Real-ESRGAN/blob/v0.2.1/LICENSE"

VIDEO_NAME = "Madrid-2021-05-06"
VIDEO_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/transcoded/4/4a/"
    "Madrid-2021-05-06.webm/Madrid-2021-05-06.webm.720p.vp9.webm"
)
VIDEO_SHA256 = "9e3f7980beda92e17f7954d293bff7ac96503177fb9be68ba4b92214a813dc7e"
VIDEO_SIZE_BYTES = 23_170_502
VIDEO_START_SECONDS = 40
VIDEO_DURATION_SECONDS = 5
VIDEO_ATTRIBUTION = "Madrid-2021-05-06 by Nicolas Vigier"
VIDEO_LICENSE = "CC0-1.0"
VIDEO_LICENSE_URL = "https://commons.wikimedia.org/wiki/File:Madrid-2021-05-06.webm"

DEMO_FPS = "24/1"
DEMO_FRAMES = 120
DEMO_INPUT_WIDTH = 1280
DEMO_INPUT_HEIGHT = 720
DEMO_OUTPUT_WIDTH = 2560
DEMO_OUTPUT_HEIGHT = 1440
DEMO_BITRATE_MBPS = 12.0
DEMO_COLOR_FRAME_INDEX = 18
DEMO_MIN_CHROMA_RETENTION_RATIO = 0.6


@dataclass(frozen=True)
class DemoPaths:
    """All persistent paths under the ignored demo cache."""

    root: Path
    weights: Path
    onnx_dir: Path
    fp32_onnx: Path
    fp16_onnx: Path
    export_conformance: Path
    engine: Path
    engine_manifest: Path
    timing_cache: Path
    source_video: Path
    input_video: Path
    output_video: Path
    report: Path

    @classmethod
    def under(cls, root: Path) -> DemoPaths:
        model_dir = root / "models"
        onnx_dir = model_dir / "onnx"
        engine = model_dir / "engines" / "realesrgan_x2plus_720p_fp16.engine"
        return cls(
            root=root,
            weights=model_dir / "RealESRGAN_x2plus.pth",
            onnx_dir=onnx_dir,
            fp32_onnx=onnx_dir / "realesrgan_x2plus_720p.onnx",
            fp16_onnx=onnx_dir / "realesrgan_x2plus_720p_fp16.onnx",
            export_conformance=onnx_dir / "realesrgan_x2plus.export-conformance.json",
            engine=engine,
            engine_manifest=Path(f"{engine}.json"),
            timing_cache=model_dir / "cache" / "trt.cache",
            source_video=root / "sources" / "Madrid-2021-05-06.720p.vp9.webm",
            input_video=root / "videos" / "demo_720p.mkv",
            output_video=root / "output" / "demo_1440p.mkv",
            report=root / "demo-result.json",
        )


@dataclass(frozen=True)
class DemoVideoContract:
    """Expected properties shared by the generated input and enhanced output."""

    width: int
    height: int
    frames: int = DEMO_FRAMES
    fps: str = DEMO_FPS
