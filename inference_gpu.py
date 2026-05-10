#!/usr/bin/env python3
"""GPU-only video upscaler via PyNvVideoCodec + TensorRT.

Full GPU pipeline: NVDEC -> NV12 -> RGB (cvcuda) -> TRT -> RGB -> NV12 (cvcuda) -> NVENC.

Usage:
    upscale-video-nvcodec --engine models/engines/model_720p.engine --input video.mp4
"""

from upscaler.gpu_pipeline import GpuPipeline


def main():
    GpuPipeline().run()


if __name__ == "__main__":
    main()
