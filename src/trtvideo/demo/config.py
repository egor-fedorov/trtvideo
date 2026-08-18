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

VIDEO_NAME = "Jacqueville beach in may 2026 (0)"
VIDEO_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/b/b9/Jacqueville_beach_in_may_2026_%280%29.webm"
)
VIDEO_SHA256 = "58ef8814dd23597f592a272cc65f1bb9064d2ff3173a895a96204655215c447a"
VIDEO_SIZE_BYTES = 21_116_480
VIDEO_START_SECONDS = 14
VIDEO_DURATION_SECONDS = 5
VIDEO_AUTHOR = "Poro26"
VIDEO_ATTRIBUTION = f'"{VIDEO_NAME}" by {VIDEO_AUTHOR}'
VIDEO_SOURCE_PAGE_URL = (
    "https://commons.wikimedia.org/wiki/File:Jacqueville_beach_in_may_2026_(0).webm"
)
VIDEO_LICENSE = "CC-BY-SA-4.0"
VIDEO_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
VIDEO_MODIFICATIONS = (
    "Demo adaptation: excerpted at 14 seconds, resized to 1280x720, converted "
    "to 24 FPS, and transcoded to H.264/AAC; enhanced output by trtvideo."
)

DEMO_FPS = "24/1"
DEMO_FRAMES = 120
DEMO_INPUT_WIDTH = 1280
DEMO_INPUT_HEIGHT = 720
DEMO_OUTPUT_WIDTH = 2560
DEMO_OUTPUT_HEIGHT = 1440
DEMO_BITRATE_MBPS = 12.0
DEMO_COLOR_FRAME_INDEX = 18
DEMO_MIN_CHROMA_RETENTION_RATIO = 0.6
DEMO_MIN_AUDIO_MEAN_DBFS = -35.0


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
    input_manifest: Path
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
            source_video=root / "sources" / "Jacqueville-beach-2026.webm",
            input_video=root / "videos" / "demo_720p.mp4",
            input_manifest=root / "videos" / "demo_720p.input.json",
            output_video=root / "output" / "demo_1440p.mp4",
            report=root / "demo-result.json",
        )


@dataclass(frozen=True)
class DemoVideoContract:
    """Expected properties shared by the generated input and enhanced output."""

    width: int
    height: int
    frames: int = DEMO_FRAMES
    fps: str = DEMO_FPS
